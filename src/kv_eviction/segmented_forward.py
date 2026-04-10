"""Segmented forward pass with KV prefix drop between segments, NO detach.

This is the training-side mirror of vLLM's scheduler-integrated KV cache
compaction. For each compaction boundary the inference engine reported, we
run one forward-pass segment, then drop the oldest stride_blocks*block_size
generation KV entries (and trim the boundary token), exactly as the
inference engine did. Gradients flow backward through the retained KV
entries — this is the "no detach" variant that gives the M4 / MKV-Full
distal gradient term (BPTT through the chain of retained-KV updates).

Differences from mkv-rl's segmented_forward_detached (the reference):
1. NO .detach() on retained KV between segments — the single load-bearing
   change this module exists for.
2. Drop boundary is prompt_aligned_len, not prompt_len, matching the
   block-level eviction semantics in vLLM's CompactingKVCacheManager.
   Callers must pass prompt_aligned_len directly; we don't compute it
   here because that would require block_size as another parameter.
3. External dispatch: the empty-events case (no compactions) is not handled
   here. Callers should branch to a standard forward() instead of calling
   segmented_forward with an empty list.

Why this gives zero train-inference KL mismatch:
- Each segment's forward uses flash_attention_2 (the same kernel vLLM uses
  for decode), so per-segment logits are numerically identical up to
  float precision.
- The KV drop between segments replicates vLLM's eviction exactly
  (same offsets, same trim, same retained identities) so attention over
  retained KV produces the same output as inference.
- Boundary token overlap: the last token of segment k is re-run in segment
  k+1 so its logit (which predicts the first token of the new segment) is
  computed under the post-eviction context, matching what inference did
  when it sampled that token.

Gradient flow (when you call .backward() on downstream loss):
  loss -> logits of segment N
       -> attention(Q_N, K_retained_{N-1} ++ K_N_new)
       -> K_retained_{N-1} (NOT detached)
       -> ...cat/trim ops...
       -> full cache from segment N-1's forward pass
       -> attention(Q_{N-1}, K_retained_{N-2} ++ K_{N-1}_new)
       -> all the way back to segment 0's parameters.

This is BPTT through the retained-KV chain. O(window) memory per segment
(not O(S^2) for a dense mask) and the same flash_attn kernel as inference.
"""

import logging

import torch
import torch.utils.checkpoint
from torch import Tensor
from transformers import DynamicCache

logger = logging.getLogger(__name__)


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
    prompt_aligned_len: int,  # ceil(prompt_len / block_size) * block_size
    stride: int,  # tokens to drop per eviction (= stride_blocks * block_size)
    temperature: Tensor,  # [1, seq_len] per-token temperatures
    max_forward_passes: int | None = None,  # FSDP synchronization padding
    activation_checkpointing: bool = False,  # per-segment re-materialization
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
        prompt_aligned_len: ceil(prompt_len / block_size) * block_size. The
            physical eviction boundary used by CompactingKVCacheManager —
            block-aligned, not token-aligned. When prompt_len is not a
            multiple of block_size, the first (prompt_aligned_len -
            prompt_len) completion tokens live in the tail of the last
            prompt block and are never evicted. The trainer-side drop must
            match this exactly or train/inference KL explodes.
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
        activation_checkpointing: When True, each segment's model() call is
            wrapped in torch.utils.checkpoint.checkpoint(use_reentrant=False).
            This re-materializes segment activations during backward instead
            of holding them in memory across the full retained-KV chain,
            making full-length (16k+ token, 20+ segment) training feasible
            on a single 80GB GPU. Must NOT be combined with HF's
            model.gradient_checkpointing_enable() — that call disables
            use_cache at the top-level forward and strips past_key_values.

    Returns:
        {"logits": [1, seq_len, vocab]} with per-token temperature scaling
        applied. Shape matches input_ids exactly.
    """
    assert input_ids.shape[0] == 1, "segmented_forward requires batch_size=1"
    assert len(segment_boundaries) > 0, (
        "segment_boundaries must be non-empty; dispatch to standard forward "
        "when the sample has no compaction events"
    )
    assert prompt_aligned_len >= prompt_len, (
        f"prompt_aligned_len ({prompt_aligned_len}) must be >= prompt_len "
        f"({prompt_len})"
    )
    assert stride > 0, f"stride must be positive, got {stride}"

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
    last_covered = prompt_len + segment_boundaries[-1]
    if last_covered < seq_len:
        seg_ranges.append((last_covered - 1, seq_len))

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

    all_logits_pieces: list[Tensor] = []
    # Non-checkpointed path: DynamicCache passed through segments directly.
    past_key_values: DynamicCache | None = None
    # Checkpointed path: plain tensor lists (shape [seq, heads, dim]) that
    # serve as positional tensor inputs to the next segment's checkpoint.
    prev_keys: list[Tensor] | None = None
    prev_values: list[Tensor] | None = None

    saved_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    try:
        for seg_idx, (seg_start, seg_end) in enumerate(seg_ranges):
            seg_ids = input_ids[:, seg_start:seg_end]
            seg_positions = position_ids[:, seg_start:seg_end]
            seg_temps = temperature[:, seg_start:seg_end]

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

            if is_last_segment:
                all_logits_pieces.append(scaled_seg_logits)
            else:
                # Drop the last logit — it predicts the boundary token,
                # which will be recomputed by the next segment (fed via
                # boundary overlap) under the post-eviction context.
                all_logits_pieces.append(scaled_seg_logits[:, :-1, :])

            # Between-segment KV eviction + cache rebuild (no detach).
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

                evicted_cache = DynamicCache() if not activation_checkpointing else None
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

                    # NO .detach() — gradients flow through the retained KV.
                    if activation_checkpointing:
                        # Keep [seq, heads, dim] for the helper's input
                        # format. The helper permutes back to
                        # [1, heads, seq, dim] internally.
                        evicted_keys.append(new_K)
                        evicted_values.append(new_V)
                    else:
                        # Permute back to [1, heads, seq, dim] for DynamicCache.
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

                if activation_checkpointing:
                    del keys, values
                    prev_keys = evicted_keys
                    prev_values = evicted_values
                else:
                    del keys, values, kv_cache
                    past_key_values = evicted_cache

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
    full_logits = torch.cat(all_logits_pieces, dim=1)
    del all_logits_pieces

    actual_passes = len(seg_ranges)
    target_passes = max_forward_passes or actual_passes
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
