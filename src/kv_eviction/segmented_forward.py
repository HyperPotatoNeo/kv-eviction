"""Segmented forward pass with KV prefix drop between segments.

This is the training-side mirror of vLLM's scheduler-integrated KV cache
compaction. For each compaction boundary the inference engine reported,
we run one forward-pass segment, then drop the oldest
stride_blocks*block_size generation KV entries (and trim the boundary
token), exactly as the inference engine did.

Two operating modes:

1. **Per-segment backward mode (`loss_fn` provided).** The production
   path. Between segments we `.detach()` the retained KV, turning each
   segment into an independent backward-pass unit. The caller supplies
   a `loss_fn` closure that computes the policy loss for just that
   segment's owned token range; after each segment's forward we call
   that closure, run `loss.backward()`, then let autograd tear down
   the graph and free the segment's activations before moving on to
   the next segment. Memory is bounded to O(1 segment) regardless of
   how many compaction events fire. This is the M3 semantic (no cross-
   chunk BPTT, no G_distal term) and it's compatible with FSDP2 and
   prime-rl's per-block activation checkpointing.

2. **Legacy / single-GPU mode (`loss_fn=None`).** Returns the full
   concatenated logits tensor for one final backward at the caller.
   Retained KV is NOT detached between segments, so in principle
   gradients flow backward through the cat chain (BPTT). In practice
   this mode only works under two configurations:
     - Single GPU with `activation_checkpointing=True` (wraps each
       segment's model call in torch.utils.checkpoint externally).
       Useful for offline KL debugging; see smoke #1.
     - Single GPU with no AC at all, on sequences small enough that
       all segment activations fit in memory.
   Under FSDP2 + prime-rl's per-block checkpoint_wrapper, this mode
   deadlocks/errors because torch.utils.checkpoint's non-reentrant
   mode does NOT trigger FSDP2's pre-forward hooks on backward
   re-entry (see plans/phase3_training_integration.md §"Key findings").
   The trainer in production MUST use the per-segment backward mode.

Why the segmentation gives zero train-inference KL mismatch:
- Each segment's forward uses flash_attention_2 (the same kernel vLLM
  uses for decode), so per-segment logits are numerically identical up
  to float precision.
- The KV drop between segments replicates vLLM's eviction exactly
  (same offsets, same trim, same retained identities) so attention
  over retained KV produces the same output as inference.
- Boundary token overlap: the last token of segment k is re-run in
  segment k+1 so its logit (which predicts the first token of the new
  segment) is computed under the post-eviction context, matching what
  inference did when it sampled that token.

Drop boundary is `prompt_aligned_len`, not `prompt_len`, matching the
block-level eviction semantics in vLLM's CompactingKVCacheManager.
Callers pass `prompt_aligned_len` directly; we don't derive it here
because that would require `block_size` as another parameter.

Future work: M4 (BPTT through retained KV, with G_distal gradient term)
requires a cache-snapshot mechanism so torch.utils.checkpoint's
backward re-runs can recover the original pre-mutation cache state.
Not attempted here; we ship M3 first.
"""

import logging
from typing import Callable

import torch
import torch.utils.checkpoint
from torch import Tensor
from transformers import DynamicCache

logger = logging.getLogger(__name__)


# Type alias for the per-segment loss callback. Given the segment's
# temperature-scaled and boundary-trimmed logits, along with the global
# sequence positions they occupy in the concatenated full-sequence logit
# tensor, returns a scalar loss that will be backward'd immediately
# (gradients accumulate into .grad before we move to the next segment).
#
# full_logit_start and full_logit_end are the inclusive / exclusive range
# in the FULL-SEQUENCE coordinate system. The segment's logit at local
# index i corresponds to full-sequence position (full_logit_start + i),
# which predicts input_ids[full_logit_start + i + 1]. For RL losses the
# callback typically slices advantages / old_logprobs / loss_mask at
# positions [full_logit_start + 1, full_logit_end + 1).
SegmentLossFn = Callable[[Tensor, int, int], Tensor]


def compute_num_segments(
    seq_len: int,
    prompt_len: int,
    segment_boundaries: list[int],
) -> int:
    """Compute how many forward-pass segments `segmented_forward` will run.

    This mirrors the range-computation logic inside segmented_forward. The
    trainer uses this to all-reduce-MAX the segment count across DP ranks
    before calling segmented_forward, so every rank runs the same number of
    forwards and NCCL all-gathers stay synchronized.

    Returns:
        Number of segments (= number of forward passes). Always >= 1 for
        non-empty boundaries.
    """
    if not segment_boundaries:
        return 1  # caller dispatches to standard forward
    n = 0
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = min(prompt_len + boundary, seq_len)
        if seg_start < seg_end:
            n += 1
        prev_boundary = boundary
    # Tail segment covers tokens past the last boundary.
    last_covered = prompt_len + segment_boundaries[-1]
    if last_covered < seq_len:
        n += 1
    return max(n, 1)


def _get_kv_from_cache(cache: DynamicCache) -> tuple[list[Tensor], list[Tensor]]:
    """Extract per-layer [seq, heads, dim] K/V tensors from a DynamicCache.

    Handles both the new transformers API (>=4.49: cache.layers[l].keys) and
    the old API (cache.key_cache[l]). Internal shape in the cache is
    [1, heads, seq, dim]; we permute to [seq, heads, dim] for easier
    splicing along the sequence dimension.
    """
    num_layers = len(cache)
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        keys = [
            cache.layers[l].keys[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
        values = [
            cache.layers[l].values[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
    elif hasattr(cache, "key_cache"):
        keys = [
            cache.key_cache[l][0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
        values = [
            cache.value_cache[l][0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
    else:
        raise RuntimeError(f"Unknown DynamicCache API: {type(cache)}")
    return keys, values


def _run_segment_checkpointed(
    model: torch.nn.Module,
    seg_ids: Tensor,
    seg_positions: Tensor,
    prev_keys: list[Tensor] | None,
    prev_values: list[Tensor] | None,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Run a single segment's forward under torch.utils.checkpoint.

    Returns (logits, new_keys, new_values) where each key/value tensor is
    shape [seq, heads, dim] (the _get_kv_from_cache format).

    The checkpoint re-materializes segment activations during backward, so
    the multi-segment retained-KV chain can be trained at long sequences
    without holding every segment's activations in memory simultaneously.

    Why this bypasses HF's gradient-checkpointing / use_cache conflict:
    HF's built-in gradient_checkpointing_enable() forcibly disables
    use_cache at the top-level model forward, which strips past_key_values
    from the output and breaks segmented_forward entirely. We instead
    leave HF GC disabled and apply checkpointing OUTSIDE the model call,
    preserving use_cache=True and past_key_values semantics. The wrapped
    function below is re-run during backward; each re-run constructs a
    fresh DynamicCache from the same input tensors, so there's no stale
    cache mutation hazard.
    """
    # Flatten prev KV to positional tensor args so checkpoint tracks them
    # as inputs. checkpoint requires tensor args (or None/lists of tensors
    # via the non_tensor_args machinery) — flat positional list is cleanest.
    if prev_keys is not None and prev_values is not None:
        kv_in_flat: list[Tensor] = []
        for k, v in zip(prev_keys, prev_values):
            # _get_kv_from_cache returned [seq, heads, dim]; the model wants
            # [1, heads, seq, dim] inside its DynamicCache.
            kv_in_flat.append(k.permute(1, 0, 2).unsqueeze(0).contiguous())
            kv_in_flat.append(v.permute(1, 0, 2).unsqueeze(0).contiguous())
    else:
        kv_in_flat = []

    def _seg_fn(_ids: Tensor, _pos: Tensor, *_kv_flat: Tensor):
        if len(_kv_flat) > 0:
            nl = len(_kv_flat) // 2
            _cache: DynamicCache | None = DynamicCache()
            for l in range(nl):
                _cache.update(_kv_flat[2 * l], _kv_flat[2 * l + 1], l)
        else:
            _cache = None

        _out = model(
            input_ids=_ids,
            position_ids=_pos,
            past_key_values=_cache,
            use_cache=True,
        )
        _raw_logits = _out["logits"] if isinstance(_out, dict) else _out.logits
        if isinstance(_raw_logits, dict):
            _raw_logits = _raw_logits["logits"]

        # Pull the updated cache out of the output; for HF modeling this is
        # returned on the output object; some custom paths may leave it on
        # the input cache we passed in.
        _out_cache = None
        if hasattr(_out, "past_key_values") and _out.past_key_values is not None:
            _out_cache = _out.past_key_values
        elif isinstance(_out, dict) and _out.get("past_key_values") is not None:
            _out_cache = _out["past_key_values"]
        else:
            _out_cache = _cache

        if _out_cache is None:
            raise RuntimeError(
                "Checkpointed segment forward did not produce past_key_values. "
                'Ensure model uses impl="hf".'
            )

        _new_keys, _new_values = _get_kv_from_cache(_out_cache)
        # Flatten for checkpoint tensor-output requirement
        _flat_out: list[Tensor] = [_raw_logits]
        for _k, _v in zip(_new_keys, _new_values):
            _flat_out.append(_k)
            _flat_out.append(_v)
        return tuple(_flat_out)

    ckpt_out = torch.utils.checkpoint.checkpoint(
        _seg_fn,
        seg_ids,
        seg_positions,
        *kv_in_flat,
        use_reentrant=False,
    )
    seg_logits = ckpt_out[0]
    kv_out = ckpt_out[1:]
    num_layers = len(kv_out) // 2
    new_keys = [kv_out[2 * l] for l in range(num_layers)]
    new_values = [kv_out[2 * l + 1] for l in range(num_layers)]
    return seg_logits, new_keys, new_values


def segmented_forward(
    model: torch.nn.Module,
    input_ids: Tensor,  # [1, seq_len]
    position_ids: Tensor,  # [1, seq_len]
    segment_boundaries: list[int],  # cumulative completion token counts
    prompt_len: int,  # raw prompt length
    prompt_aligned_len: int,  # block-aligned eviction boundary
    stride: int,  # tokens to drop per eviction (= stride_blocks * block_size)
    temperature: Tensor,  # [1, seq_len] per-token temperatures
    max_forward_passes: int | None = None,  # FSDP synchronization padding
    activation_checkpointing: bool = False,  # per-segment re-materialization (legacy)
    loss_fn: SegmentLossFn | None = None,  # per-segment backward callback
    bptt_segments: int | None = 1,  # truncation depth for TBPTT
) -> dict[str, Tensor]:
    """Run segmented forward with KV prefix drop between segments, no detach.

    Args:
        model: HF CausalLM model with attn_implementation="flash_attention_2"
            and impl="hf" (custom llama asserts past_key_values is None and is
            not usable here).
        input_ids: Full sequence [1, seq_len] = prompt + completion tokens.
        position_ids: Per-token absolute positions [1, seq_len]. Plain arange
            is correct: RoPE at inference time baked position = prompt_len +
            gen_index, and in this concatenated sequence gen[N] sits at index
            prompt_len + N, so position_ids[i] = i matches inference exactly.
        segment_boundaries: Cumulative completion-token counts at each
            compaction event, in ascending order. Must be non-empty; caller
            is responsible for dispatching to a standard forward when the
            list is empty.
        prompt_len: Number of prompt tokens (the offset into input_ids where
            completion tokens begin).
        prompt_aligned_len: Block-aligned eviction boundary. Without
            protected prefix: ceil(prompt_len / block_size) * block_size.
            With protected prefix: ceil(min(protected_prefix, prompt_len)
            / block_size) * block_size — can be LESS than prompt_len
            because old conversation turns between the protected prefix
            and the prompt end are evictable. The trainer-side drop must
            match the inference-side boundary exactly or train/inference
            KL explodes.
        stride: Number of tokens evicted per compaction event. In the vLLM
            config this is compaction_stride, a multiple of block_size.
        temperature: Per-token temperatures [1, seq_len] used at generation
            time. Logits are scaled by 1/temperature per-token before being
            concatenated into the final output.
        max_forward_passes: Target forward-pass count for FSDP rank
            synchronization. If greater than the actual number of segments,
            the difference is padded with dummy forward passes that
            contribute to the autograd graph but carry zero gradient weight.
            Must be >= len(segment_boundaries) + 1 when set.
        activation_checkpointing: (Legacy, single-GPU only.) When True,
            each segment's model() call is wrapped in
            torch.utils.checkpoint.checkpoint(use_reentrant=False). This
            re-materializes segment activations during backward instead
            of holding them in memory across the full retained-KV chain.
            Must NOT be combined with HF's
            model.gradient_checkpointing_enable() (disables use_cache).
            Must NOT be used under FSDP2 — non-reentrant checkpoint's
            backward re-entry does not trigger FSDP2's pre-forward
            hooks, producing Tensor/DTensor mismatches in RMSNorm.
            Incompatible with loss_fn (only one checkpointing strategy
            at a time).
        loss_fn: (Production path for compaction training.) Per-segment
            loss callback. When provided, the function runs in
            per-segment backward mode: after every `bptt_segments`
            segments' forwards we call backward on the accumulated
            segment losses, detach the retained KV, and move to the
            next BPTT window. Activations from completed segments are
            freed by autograd as each window's backward finishes,
            bounding memory to O(bptt_segments segments). Compatible
            with FSDP2 + prime-rl's per-block AC. Cannot be combined
            with activation_checkpointing=True.
        bptt_segments: Truncation depth (in segments) for truncated
            backpropagation through time. Only meaningful when loss_fn
            is provided. Each BPTT "window" is this many consecutive
            segments; gradients flow through retained KV within a
            window, and the KV is .detach()'d between windows.
                1 (default) = M3 semantics — no cross-chunk BPTT, one
                    backward per segment, O(1 segment) memory.
                K > 1 = TBPTT with depth K segments, O(K segments)
                    memory, gradients span K segments via retained KV.
                None = "full trajectory" — forward every segment in a
                    single window, one backward at the end. Equivalent
                    to M4 gradient semantics (G_distal term intact)
                    but requires O(all segments) memory.
            The final window may be shorter than bptt_segments if the
            total segment count isn't a multiple of bptt_segments.

    Returns:
        - Legacy mode (loss_fn=None):
            {"logits": [1, seq_len, vocab]} — temperature-scaled full
            sequence logits. Caller is responsible for the final
            backward. Shape matches input_ids exactly.
        - Per-segment backward mode (loss_fn provided):
            {"loss": <scalar tensor, detached>, "n_segments": int}
            The scalar is the sum of all per-segment losses (already
            backward'd; gradients have already accumulated into
            model.parameters().grad). Loss is detached so the caller
            can log/reduce it without re-entering the graph.
    """
    assert input_ids.shape[0] == 1, "segmented_forward requires batch_size=1"
    # Empty segment_boundaries is a legal no-event sample: the trainer's
    # unified compaction dispatch (D5 fix) routes every sample through
    # segmented_forward in a compaction run, including short rollouts
    # whose inference never triggered a compaction event. Those samples
    # produce seg_ranges = [(0, seq_len)] below — a single segment
    # covering the whole [prompt + completion] sequence, numerically
    # equivalent to calling model(input_ids, position_ids) directly
    # under flash_attention_2. See plans/phase3_training_integration.md.
    assert prompt_aligned_len > 0, (
        f"prompt_aligned_len must be positive, got {prompt_aligned_len}"
    )
    assert prompt_aligned_len <= input_ids.shape[1], (
        f"prompt_aligned_len ({prompt_aligned_len}) exceeds seq_len "
        f"({input_ids.shape[1]})"
    )
    assert stride > 0, f"stride must be positive, got {stride}"
    assert not (activation_checkpointing and loss_fn is not None), (
        "activation_checkpointing=True and loss_fn are mutually exclusive: "
        "the per-segment backward mode (loss_fn) already frees segment "
        "activations by running .backward() between segments, so the "
        "outer torch.utils.checkpoint wrapping would be redundant and "
        "would also conflict with the per-segment backward cadence."
    )
    per_segment_backward = loss_fn is not None
    if per_segment_backward:
        if bptt_segments is not None:
            assert bptt_segments >= 1, (
                f"bptt_segments must be None or >= 1, got {bptt_segments}"
            )

    device = input_ids.device
    seq_len = input_ids.shape[1]

    # Build the list of segment ranges in input_ids space.
    #
    # There is one segment per compaction event plus a final tail segment
    # covering everything after the last boundary through seq_len.
    #
    # Segment 0: input_ids[0 : prompt_len + boundaries[0])
    # Segment k (k > 0): input_ids[prompt_len + boundaries[k-1] - 1 :
    #                              prompt_len + boundaries[k])
    #     The -1 is the boundary-token overlap: the last token of segment
    #     k-1 is re-fed so its logit (which predicts the first token of
    #     segment k) is recomputed under the post-eviction context.
    # Final tail: input_ids[prompt_len + boundaries[-1] - 1 : seq_len)
    #
    # Tail is only a distinct segment if boundaries[-1] < completion_len.
    # Otherwise the last "segment" in the segment_boundaries list already
    # covers through seq_len.
    seg_ranges: list[tuple[int, int]] = []
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = min(prompt_len + boundary, seq_len)
        if seg_start < seg_end:
            seg_ranges.append((seg_start, seg_end))
        prev_boundary = boundary

    # Add a tail segment to cover generation past the last compaction event.
    #
    # Convention: segment_boundaries has one entry per compaction event (from
    # CompactionEventWire.num_output_tokens_at_compaction). After the last
    # compaction, the model typically continues generating until EOS or
    # max_tokens. Those tail tokens need their own segment that starts from
    # the boundary-overlap token and runs to seq_len.
    #
    # (Note: this convention differs from mkv-rl's window_forward which
    # treats boundaries as segment END POINTS including the final one. In
    # mkv-rl the last boundary equals completion_len and no tail is needed.
    # We mirror CompactionEvent semantics directly instead.)
    if segment_boundaries:
        last_covered = prompt_len + segment_boundaries[-1]
        if last_covered < seq_len:
            seg_ranges.append((last_covered - 1, seq_len))
    elif seq_len > 0:
        # No events: single segment covering the whole sample. We still
        # go through segmented_forward's machinery (per-segment backward
        # mode, FSDP dummy-pass padding, etc.) so every rank in a
        # compaction run issues a uniform collective sequence regardless
        # of whether each rank's sample had any events. Numerically this
        # is a plain text forward on the full sequence under
        # flash_attention_2 — same kernel as the non-compaction path.
        seg_ranges.append((0, seq_len))

    assert seg_ranges, (
        f"No segment ranges produced: seq_len={seq_len}, prompt_len={prompt_len}, "
        f"segment_boundaries={segment_boundaries}"
    )

    # Capture past_key_values from the backbone via a hook (non-checkpointed
    # path only). FSDP2 + custom output-linear wrappers may drop
    # past_key_values from the top-level model output, so a hook on the
    # backbone (model.model) is more reliable.
    #
    # The checkpointed path does NOT use this hook — its inner function
    # reconstructs and returns the cache explicitly to keep everything as
    # tensor-typed outputs that checkpoint can track.
    captured_kv: dict[str, DynamicCache | None] = {}

    def _capture_kv_hook(_module, _input, output):
        if hasattr(output, "past_key_values"):
            captured_kv["past_key_values"] = output.past_key_values
        elif isinstance(output, dict):
            captured_kv["past_key_values"] = output.get("past_key_values")

    backbone = model.model if hasattr(model, "model") else model
    hook_handle = (
        backbone.register_forward_hook(_capture_kv_hook)
        if not activation_checkpointing
        else None
    )

    # Legacy (loss_fn=None) path collects per-segment logit slices for
    # one final backward at the caller. Per-segment-backward path uses
    # accumulated_loss (a Python float) and window_loss (a live graph
    # tensor that accumulates per-segment losses within the current
    # BPTT window), leaving all_logits_pieces empty.
    all_logits_pieces: list[Tensor] = []
    accumulated_loss: float = 0.0
    window_loss: Tensor | None = None
    # Count of real segments processed within the current BPTT window.
    # Resets to 0 at every window boundary.
    segments_in_window: int = 0
    # Non-checkpointed path: DynamicCache passed through segments directly.
    # In per-segment backward mode we build a FRESH DynamicCache from
    # prev_keys/prev_values at the top of each loop iteration (so the
    # cache object itself never crosses segments). Whether those
    # prev_keys still carry autograd edges depends on whether we're
    # inside a BPTT window (edges intact) or just crossed a boundary
    # (detached).
    past_key_values: DynamicCache | None = None
    # Checkpointed path: plain tensor lists (shape [seq, heads, dim]) that
    # serve as positional tensor inputs to the next segment's checkpoint.
    # Per-segment backward path: same shape; detached only at BPTT
    # window boundaries (see the eviction block below).
    prev_keys: list[Tensor] | None = None
    prev_values: list[Tensor] | None = None

    saved_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    try:
        for seg_idx, (seg_start, seg_end) in enumerate(seg_ranges):
            seg_ids = input_ids[:, seg_start:seg_end]
            seg_positions = position_ids[:, seg_start:seg_end]
            seg_temps = temperature[:, seg_start:seg_end]

            # Per-segment backward mode: rebuild a fresh DynamicCache
            # from the detached prev_keys / prev_values before each
            # forward. This guarantees the cache object we pass into
            # model() has never been touched by any prior segment's
            # graph, so the torn-down activations from earlier segments
            # can't be pulled back in via shared state.
            if per_segment_backward and prev_keys is not None:
                fresh_cache = DynamicCache()
                for l, (k_seq, v_seq) in enumerate(zip(prev_keys, prev_values)):
                    # prev_keys[l] is [seq, heads, dim] and already detached
                    k_4d = k_seq.permute(1, 0, 2).unsqueeze(0).contiguous()
                    v_4d = v_seq.permute(1, 0, 2).unsqueeze(0).contiguous()
                    fresh_cache.update(k_4d, v_4d, l)
                past_key_values = fresh_cache

            if activation_checkpointing:
                seg_logits, new_keys, new_values = _run_segment_checkpointed(
                    model, seg_ids, seg_positions, prev_keys, prev_values
                )
            else:
                out = model(
                    input_ids=seg_ids,
                    position_ids=seg_positions,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                raw_logits = out["logits"] if isinstance(out, dict) else out.logits
                # Some custom output heads (e.g. VanillaOutputLinear) wrap
                # logits in a dict. Unwrap once if we see that pattern.
                seg_logits = (
                    raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
                )

            scaled_seg_logits = seg_logits / seg_temps.unsqueeze(-1).to(
                seg_logits.dtype
            )

            is_last_segment = seg_idx == len(seg_ranges) - 1

            # Determine the segment's "owned" range of global logit
            # positions: the positions in the concatenated full-logits
            # tensor that this segment is responsible for computing.
            #
            # Non-final segments drop the last logit (it's re-run by the
            # next segment under the post-eviction context). Final
            # segment keeps all logits.
            if is_last_segment:
                seg_logits_used = scaled_seg_logits
                owned_start = seg_start
                owned_end = seg_end
            else:
                seg_logits_used = scaled_seg_logits[:, :-1, :]
                owned_start = seg_start
                owned_end = seg_end - 1

            if per_segment_backward:
                # Compute this segment's loss via the caller's callback
                # and accumulate into the current BPTT window's loss
                # tensor. The window's combined loss will be backward'd
                # at the end of the window (see eviction block below).
                seg_loss = loss_fn(seg_logits_used, owned_start, owned_end)
                window_loss = seg_loss if window_loss is None else window_loss + seg_loss
                segments_in_window += 1
                # Release our references to the segment's forward-pass
                # tensors we no longer need. The retained KV that feeds
                # the next segment is extracted separately below.
                del seg_loss, seg_logits_used, scaled_seg_logits, seg_logits
                if not activation_checkpointing:
                    del out, raw_logits
            else:
                all_logits_pieces.append(seg_logits_used)

            # Determine whether this segment ends a BPTT window. If it
            # does, we'll need to backward the accumulated window_loss
            # and detach the retained KV that flows into the next
            # window. Windows close at three events:
            #   1. segments_in_window reaches bptt_segments, OR
            #   2. bptt_segments is None and this is the last segment
            #      (full trajectory as one big window), OR
            #   3. it's the last segment regardless (final flush).
            at_window_end = False
            if per_segment_backward:
                if bptt_segments is None:
                    at_window_end = is_last_segment
                else:
                    at_window_end = (
                        segments_in_window >= bptt_segments or is_last_segment
                    )

            # Between-segment KV eviction + cache rebuild.
            #
            # Legacy mode: retained KV keeps its autograd edges so the
            # final backward can flow through the cat chain (BPTT
            # attempt — only works in specific single-GPU modes).
            #
            # Per-segment backward mode: KV keeps autograd edges WITHIN
            # a BPTT window (so gradients can flow back through the
            # cat chain for up to `bptt_segments` steps). At window
            # boundaries we detach after computing the eviction, so
            # the next window starts with fresh leaves.
            if not is_last_segment:
                if activation_checkpointing:
                    keys = new_keys
                    values = new_values
                else:
                    kv_cache = captured_kv.get("past_key_values")
                    assert kv_cache is not None, (
                        "Hook did not capture past_key_values. Ensure model uses "
                        'impl="hf" (the custom llama path asserts past_key_values '
                        "is None and is incompatible with segmented forward)."
                    )
                    captured_kv.clear()
                    keys, values = _get_kv_from_cache(kv_cache)

                num_layers = len(keys)
                kv_seq_len = keys[0].shape[0]

                # Drop range [prompt_aligned_len, prompt_aligned_len + stride),
                # clamped against the actual assistant content we have.
                asst_len = kv_seq_len - prompt_aligned_len
                actual_stride = min(stride, asst_len)
                # Always trim the boundary token (the last KV entry) because
                # the next segment will re-feed that token and its new KV
                # will be appended. Without this trim, the boundary token
                # would appear twice in the KV cache.
                trim = 1

                # Output containers:
                #   - evicted_cache: DynamicCache consumed by the next
                #     segment's model() call in LEGACY mode
                #     (activation_checkpointing=False, loss_fn=None).
                #   - evicted_keys / evicted_values: list-of-tensor form
                #     consumed by:
                #       * the checkpointed-segment helper
                #         (activation_checkpointing=True), AND
                #       * the per-segment backward path
                #         (loss_fn provided), which rebuilds a fresh
                #         DynamicCache from these lists at the top of
                #         the next iteration.
                use_lists = activation_checkpointing or per_segment_backward
                evicted_cache = DynamicCache() if not use_lists else None
                evicted_keys: list[Tensor] = []
                evicted_values: list[Tensor] = []
                for l in range(num_layers):
                    if actual_stride > 0:
                        new_K = torch.cat(
                            [
                                keys[l][:prompt_aligned_len],
                                keys[l][prompt_aligned_len + actual_stride : -trim],
                            ],
                            dim=0,
                        )
                        new_V = torch.cat(
                            [
                                values[l][:prompt_aligned_len],
                                values[l][prompt_aligned_len + actual_stride : -trim],
                            ],
                            dim=0,
                        )
                    else:
                        # actual_stride==0 shouldn't happen in practice
                        # (needs_compaction guards it) but handle defensively.
                        new_K = keys[l][:-trim]
                        new_V = values[l][:-trim]

                    if use_lists:
                        # Keep [seq, heads, dim] for the list-based
                        # inputs. In per-segment-backward mode the source
                        # `keys[l]`/`values[l]` are already detached
                        # above, so new_K/new_V are also detached leaves.
                        evicted_keys.append(new_K)
                        evicted_values.append(new_V)
                    else:
                        # Legacy BPTT path: permute back to
                        # [1, heads, seq, dim] for DynamicCache and
                        # keep autograd edges intact so the final
                        # backward can flow through the cat chain.
                        new_K = new_K.permute(1, 0, 2).unsqueeze(0)
                        new_V = new_V.permute(1, 0, 2).unsqueeze(0)
                        evicted_cache.update(new_K, new_V, l)

                # Correct new_kv_len accounting: when the stride range
                # reaches the end of the KV, the trim range overlaps with it,
                # so a naive (kv_seq_len - actual_stride - trim) would
                # over-subtract. Compute what the cat actually produced:
                kept_asst = max(
                    0, (kv_seq_len - trim) - (prompt_aligned_len + actual_stride)
                )
                new_kv_len = prompt_aligned_len + kept_asst
                logger.debug(
                    "KV eviction seg %d: kv_len %d -> %d (dropped %d stride + "
                    "%d boundary), prompt_aligned=%d, retained_asst=%d",
                    seg_idx,
                    kv_seq_len,
                    new_kv_len,
                    actual_stride,
                    trim,
                    prompt_aligned_len,
                    kept_asst,
                )

                if use_lists:
                    # Drop local refs to the captured K/V. The cache
                    # object (captured via hook in the
                    # non-activation_checkpointing branch) is released
                    # here too since nothing else holds it.
                    del keys, values
                    if not activation_checkpointing:
                        del kv_cache
                    if per_segment_backward and at_window_end:
                        # End of BPTT window: backward the accumulated
                        # loss, then detach the evicted KV so the next
                        # window starts from fresh leaves. The
                        # backward call tears down the current
                        # window's forward graph and frees its
                        # activations. We grab detached copies BEFORE
                        # backward so the next-segment inputs don't
                        # depend on freed graph state.
                        prev_keys = [k.detach() for k in evicted_keys]
                        prev_values = [v.detach() for v in evicted_values]
                        assert window_loss is not None, (
                            "at_window_end reached with no accumulated "
                            "window_loss; per-segment backward state is "
                            "inconsistent"
                        )
                        window_loss.backward()
                        accumulated_loss += float(window_loss.detach().item())
                        window_loss = None
                        segments_in_window = 0
                        del evicted_keys, evicted_values
                    else:
                        # Mid-window (or activation_checkpointing path):
                        # retain autograd edges by passing the
                        # un-detached cat outputs straight through.
                        prev_keys = evicted_keys
                        prev_values = evicted_values
                    past_key_values = None
                else:
                    del keys, values, kv_cache
                    past_key_values = evicted_cache

            # Final-segment flush for per-segment backward mode: the
            # eviction block above only runs on non-final segments,
            # but the last segment also needs its window backward'd.
            # (For multi-window runs with the final segment landing
            # inside a partial window, this is the only place the
            # window_loss gets consumed; for a run where the final
            # window ended on a non-last segment, window_loss will
            # already have been reset to None by the eviction block.)
            if per_segment_backward and is_last_segment and window_loss is not None:
                window_loss.backward()
                accumulated_loss += float(window_loss.detach().item())
                window_loss = None
                segments_in_window = 0

    finally:
        if hook_handle is not None:
            hook_handle.remove()
        model.config.use_cache = saved_use_cache

    # Deliberately NOT calling torch.cuda.empty_cache() here: it's a hard
    # synchronization point that stalls the stream and returns cached blocks
    # to the CUDA driver right before FSDP2 is about to reallocate them for
    # the next micro-batch's all-gather. On a compaction hot path this
    # measurably regresses throughput and can interact badly with
    # activation-offloading pipelines. The inter-segment `del keys, values,
    # kv_cache` inside the loop releases the old cache promptly; relying on
    # the allocator is the same pattern the rest of the trainer uses.
    actual_passes = len(seg_ranges)
    target_passes = max_forward_passes or actual_passes

    if per_segment_backward:
        # In per-segment backward mode we've already called backward on
        # each real segment. For FSDP2 synchronization across ranks, any
        # remaining dummy passes still need to run a forward + dummy
        # backward so every rank issues the same collective sequence.
        # Each dummy backward contributes exactly zero to .grad because
        # the dummy output is multiplied by 0.
        if target_passes > actual_passes:
            _run_dummy_passes_with_backward(
                model,
                input_ids,
                position_ids,
                target_passes - actual_passes,
            )
        return {
            "loss": torch.tensor(accumulated_loss, device=device),
            "n_segments": actual_passes,
        }

    # Legacy path: concatenate per-segment logits and return them.
    full_logits = torch.cat(all_logits_pieces, dim=1)
    del all_logits_pieces

    if target_passes > actual_passes:
        full_logits = _pad_with_dummy_passes(
            model,
            input_ids,
            position_ids,
            full_logits,
            target_passes - actual_passes,
            device,
        )

    assert full_logits.shape[1] == seq_len, (
        f"segmented_forward produced {full_logits.shape[1]} logits, "
        f"expected {seq_len}. segments={len(seg_ranges)}, ranges={seg_ranges}"
    )

    return {"logits": full_logits}


def _run_dummy_passes_with_backward(
    model: torch.nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    num_dummy: int,
) -> None:
    """Run `num_dummy` dummy forward+backward pairs for FSDP2 rank sync.

    In per-segment backward mode, every rank needs to execute the same
    number of (forward, backward) pairs per training step so FSDP2's
    all-gather / reduce-scatter counts stay matched. If this rank
    processed K_local real segments and max across ranks is K_global,
    this fills in the gap with (K_global - K_local) dummy passes.

    Each dummy pass:
      1. Runs a real forward on a 1-token slice (FSDP2 all-gather fires).
      2. Constructs a zero-weighted loss (mean * 0) so backward flows
         into FSDP2 hooks without actually modifying any gradient.
      3. Calls .backward() to trigger FSDP2's reduce-scatter.
    """
    for _ in range(num_dummy):
        d_out = model(
            input_ids=input_ids[:, :1],
            position_ids=position_ids[:, :1],
        )
        d_logits = d_out["logits"] if isinstance(d_out, dict) else d_out.logits
        if isinstance(d_logits, dict):
            d_logits = d_logits["logits"]
        # float().mean() first to avoid bf16 sum overflow producing Inf
        # (Inf * 0 = NaN, which would corrupt gradients).
        dummy_loss = d_logits.float().mean() * 0.0
        dummy_loss.backward()


def _pad_with_dummy_passes(
    model: torch.nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    logits: Tensor,
    num_dummy: int,
    device: torch.device,
) -> Tensor:
    """Run `num_dummy` extra forward passes on a 1-token slice and add
    `dummy_sum * 0` to the logits.

    This keeps FSDP2 all-gather/reduce-scatter counts synchronized across
    ranks with different segment counts. Each dummy forward:
    1. Runs a real forward pass so FSDP backward hooks fire.
    2. Extracts a scalar (float().mean() to avoid bf16 sum overflow that
       could produce Inf; Inf*0=NaN would corrupt gradients).
    3. Accumulates into dummy_sum, which is multiplied by 0 at the end.

    The result is a no-op on the forward value (logits unchanged) and
    contributes exactly 0 to gradients, but keeps the autograd graph alive
    so FSDP all-reduces happen.
    """
    dummy_sum = torch.tensor(0.0, device=device)
    for _ in range(num_dummy):
        d_out = model(
            input_ids=input_ids[:, :1],
            position_ids=position_ids[:, :1],
        )
        d_logits = d_out["logits"] if isinstance(d_out, dict) else d_out.logits
        if isinstance(d_logits, dict):
            d_logits = d_logits["logits"]
        dummy_sum = dummy_sum + d_logits.float().mean()
    return logits + (dummy_sum * 0).to(logits.dtype)
