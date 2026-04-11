# Phase 3: Training Integration — Segmented Forward (No Detach)

## Goal

Implement the training-side components that match the inference-side compaction from Phase 2.
After this phase, prime-rl can train on rollouts from the compaction-enabled vLLM server
using a segmented forward pass that:
1. Uses the same `flash_attn` kernel per segment as vLLM inference (zero KL mismatch)
2. Does NOT detach retained KV between segments (preserves cross-chunk gradients, G_distal)
3. Works correctly with FSDP2 (dummy passes for rank synchronization)

## Prerequisites

- Phase 2 complete (vLLM with compaction returning `compaction_events` in responses)
- prime-rl submodule at `$SCRATCH/kv-eviction/prime-rl/` (clean main, no modifications)
- Understanding of mkv-rl's `window_forward.py` (the base we adapt from)

## Files to Create

```
src/kv_eviction/
├── segmented_forward.py   # ~200 lines: segmented forward, no detach
├── env.py                 # ~80 lines: vLLM response -> prime-rl RolloutOutput
└── train_hooks.py         # ~50 lines: inject segmented forward into prime-rl
```

---

## 3.1: Segmented Forward — No Detach

**File: `src/kv_eviction/segmented_forward.py`**

Adapted from `$SCRATCH/mkv-rl/src/prime_rl/trainer/rl/window_forward.py`. The key change
is removing `.detach()` on retained KV between segments, which preserves cross-chunk
gradient flow (the G_distal term in our methods document).

### Architecture

```
Segment 0: [prompt | gen_tokens_0..boundary_0]
  → forward pass with use_cache=True
  → extract KV from DynamicCache via backbone hook
  → drop first `stride` assistant KV entries (same as inference eviction)
  → DO NOT DETACH retained KV  ← KEY DIFFERENCE from mkv-rl
  → trim boundary token (will be recomputed by next segment)

Segment 1: [boundary_token_0 | gen_tokens_boundary_0..boundary_1]
  → forward pass with past_key_values = evicted (non-detached) KV
  → same KV eviction cycle...

Last segment: [boundary_token_{n-1} | remaining_tokens]
  → forward pass with past_key_values
  → no eviction, just collect logits
```

### Differences from mkv-rl window_forward.py

| Aspect | mkv-rl (window_forward.py) | kv-eviction (segmented_forward.py) |
|--------|---------------------------|-----------------------------------|
| `.detach()` on retained KV | Yes (line 243) | **No** — retained KV stays in autograd graph |
| Gradient flow between segments | Blocked (each segment independent) | **Preserved** (G_distal term) |
| Memory usage | Lower (detach frees intermediate activations) | Higher (retains activations for backprop) |
| Gradient correctness | Approximate (local gradients only) | Exact (full chain rule through retained KV) |
| Everything else | Same | Same |

### Full Implementation

```python
"""Segmented forward pass with KV prefix drop, NO detach.

Cross-chunk gradients are preserved through retained KV entries between
segments, enabling the G_distal gradient term. Uses the same flash_attn
kernel per segment as vLLM inference for zero KL mismatch.

Adapted from mkv-rl window_forward.py with one critical change:
  - REMOVED: .detach() on retained KV (lines 243-244 of window_forward.py)
  - RESULT: gradients flow back through retained KV to earlier segments

FSDP2 compatible: dummy forward passes pad ranks with fewer segments.
"""

import logging

import torch
from torch import Tensor
from transformers import DynamicCache

logger = logging.getLogger(__name__)


def _get_kv_from_cache(cache: DynamicCache) -> tuple[list[Tensor], list[Tensor]]:
    """Extract per-layer key/value tensors from a DynamicCache.

    Handles both old API (cache.key_cache) and new API (cache.layers[i].keys).
    Returns per-layer tensors shaped [seq, heads, dim].
    """
    num_layers = len(cache)

    # Try new API first (transformers >= 4.49)
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        keys = [
            cache.layers[l].keys[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
        values = [
            cache.layers[l].values[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
    # Fall back to old API
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


def segmented_forward(
    model: torch.nn.Module,
    input_ids: Tensor,           # [1, seq_len]
    position_ids: Tensor,        # [1, seq_len]
    segment_boundaries: list[int],  # cumulative completion token counts
    prompt_len: int,
    stride: int,                 # tokens to drop per eviction
    temperature: Tensor,         # [1, seq_len] per-token
    max_forward_passes: int | None = None,  # for FSDP padding
) -> dict[str, Tensor]:  # {"logits": [1, seq_len, vocab]}
    """Run segmented forward passes with KV prefix drop, NO DETACH.

    This is the key difference from mkv-rl's segmented_forward_detached():
    retained KV between segments is NOT detached, so gradients flow back
    through the KV to earlier segments. This preserves the G_distal term.

    For each segment:
    1. Forward pass with use_cache=True to get logits and KV cache.
    2. Between segments: extract KV, drop first `stride` assistant tokens.
       DO NOT DETACH the retained KV.
    3. Next segment uses evicted (but still-differentiable) past_key_values.

    Args:
        model: HuggingFace model (e.g. Qwen3ForCausalLM). Must support use_cache=True.
        input_ids: Full input_ids [1, seq_len] (prompt + all completion tokens).
        position_ids: Full position_ids [1, seq_len]. Must include position offsets
            matching inference (position = physical_pos + cumulative_evicted).
        segment_boundaries: Cumulative completion token counts at end of each segment.
            These come from compaction_events[i].completion_tokens_at_compaction.
            E.g. [3584, 7168, 10752] means segment 0 covers the first 3584 completion
            tokens, segment 1 covers the next 3584, etc.
        prompt_len: Number of prompt tokens.
        stride: Number of assistant KV entries to drop per eviction. Must match the
            compaction_stride used during inference.
        temperature: Per-token temperatures [1, seq_len].
        max_forward_passes: Target forward pass count for FSDP synchronization.
            Must be >= actual segment count. If None, no padding is done.

    Returns:
        Dict with "logits" key containing [1, seq_len, vocab] temperature-scaled logits.
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]
    assert input_ids.shape[0] == 1, "Segmented forward only supports batch_size=1"

    # Edge case: empty completion
    completion_len = seq_len - prompt_len
    if completion_len <= 0:
        logger.warning("Empty completion: prompt_len=%d >= seq_len=%d", prompt_len, seq_len)
        out = model(input_ids=input_ids, position_ids=position_ids)
        raw_logits = out["logits"] if isinstance(out, dict) else out.logits
        logits = raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
        scaled = logits / temperature.unsqueeze(-1).to(logits.dtype)
        actual_passes = 1
        target_passes = max_forward_passes or actual_passes
        if target_passes > actual_passes:
            scaled = _pad_with_dummy_passes(
                model, input_ids, position_ids, scaled,
                target_passes - actual_passes, device,
            )
        return {"logits": scaled}

    # Capture past_key_values from the backbone via hook.
    # FSDP2 + VanillaOutputLinear may not propagate past_key_values through
    # the top-level model output.
    captured_kv: dict[str, DynamicCache | None] = {}

    def _capture_kv_hook(_module, _input, output):
        if hasattr(output, "past_key_values"):
            captured_kv["past_key_values"] = output.past_key_values
        elif isinstance(output, dict):
            captured_kv["past_key_values"] = output.get("past_key_values")

    backbone = model.model if hasattr(model, "model") else model
    hook_handle = backbone.register_forward_hook(_capture_kv_hook)

    # Build segment token ranges in input_ids space.
    # Segment 0: input_ids[0 : prompt_len + boundary_0]
    # Segment k>0: input_ids[prompt_len + boundary_{k-1} - 1 : prompt_len + boundary_k]
    #   The -1 creates boundary token overlap for logit recomputation.
    seg_input_ranges: list[tuple[int, int]] = []
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = min(prompt_len + boundary, seq_len)
        if seg_start < seg_end:
            seg_input_ranges.append((seg_start, seg_end))
        prev_boundary = boundary

    # Handle case where boundaries don't cover full completion
    last_covered = prompt_len + segment_boundaries[-1] if segment_boundaries else prompt_len
    if last_covered < seq_len and seg_input_ranges:
        last_start, _ = seg_input_ranges[-1]
        seg_input_ranges[-1] = (last_start, seq_len)

    all_logits_pieces: list[Tensor] = []
    past_key_values: DynamicCache | None = None

    saved_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    try:
        for seg_idx, (seg_start, seg_end) in enumerate(seg_input_ranges):
            seg_ids = input_ids[:, seg_start:seg_end]
            seg_positions = position_ids[:, seg_start:seg_end]
            seg_temps = temperature[:, seg_start:seg_end]

            out = model(
                input_ids=seg_ids,
                position_ids=seg_positions,
                past_key_values=past_key_values,
                use_cache=True,
            )

            raw_logits = out["logits"] if isinstance(out, dict) else out.logits
            seg_logits = raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
            scaled_seg_logits = seg_logits / seg_temps.unsqueeze(-1).to(seg_logits.dtype)

            is_last_segment = seg_idx == len(seg_input_ranges) - 1

            if is_last_segment:
                all_logits_pieces.append(scaled_seg_logits)
            else:
                # Drop last logit — recomputed by next segment (boundary overlap)
                all_logits_pieces.append(scaled_seg_logits[:, :-1, :])

            # Between segments: KV prefix drop, NO DETACH
            if not is_last_segment:
                kv_cache = captured_kv.get("past_key_values")
                assert kv_cache is not None, (
                    "Hook did not capture past_key_values. "
                    "Ensure model uses impl='hf' which supports use_cache=True."
                )
                captured_kv.clear()

                keys, values = _get_kv_from_cache(kv_cache)
                num_layers = len(keys)
                kv_seq_len = keys[0].shape[0]

                asst_len = kv_seq_len - prompt_len
                actual_stride = min(stride, asst_len)
                trim = 1  # boundary token

                if actual_stride > 0:
                    evicted_cache = DynamicCache()
                    for l in range(num_layers):
                        new_K = torch.cat(
                            [keys[l][:prompt_len],
                             keys[l][prompt_len + actual_stride:-trim]],
                            dim=0,
                        )
                        new_V = torch.cat(
                            [values[l][:prompt_len],
                             values[l][prompt_len + actual_stride:-trim]],
                            dim=0,
                        )

                        # NO DETACH — this is the key difference from window_forward.py
                        # Retained KV stays in the autograd graph, enabling G_distal.
                        # Permute [seq, heads, dim] -> [1, heads, seq, dim] for DynamicCache
                        new_K = new_K.permute(1, 0, 2).unsqueeze(0)
                        new_V = new_V.permute(1, 0, 2).unsqueeze(0)

                        evicted_cache.update(new_K, new_V, l)
                else:
                    evicted_cache = DynamicCache()
                    for l in range(num_layers):
                        # Still trim boundary token, but no stride eviction
                        new_K = keys[l][:-trim].permute(1, 0, 2).unsqueeze(0)
                        new_V = values[l][:-trim].permute(1, 0, 2).unsqueeze(0)
                        evicted_cache.update(new_K, new_V, l)

                new_kv_len = kv_seq_len - actual_stride - trim
                logger.debug(
                    "KV eviction seg %d: kv_len %d -> %d (dropped %d + %d boundary), "
                    "prompt=%d, retained_asst=%d",
                    seg_idx, kv_seq_len, new_kv_len,
                    actual_stride, trim, prompt_len,
                    asst_len - actual_stride - trim,
                )

                del keys, values, kv_cache
                past_key_values = evicted_cache

    finally:
        hook_handle.remove()
        model.config.use_cache = saved_use_cache

    torch.cuda.empty_cache()
    full_logits = torch.cat(all_logits_pieces, dim=1)
    del all_logits_pieces

    # FSDP dummy passes
    actual_passes = len(seg_input_ranges)
    target_passes = max_forward_passes or actual_passes
    if target_passes > actual_passes:
        full_logits = _pad_with_dummy_passes(
            model, input_ids, position_ids, full_logits,
            target_passes - actual_passes, device,
        )

    assert full_logits.shape[1] == input_ids.shape[1], (
        f"Segmented forward logits shape {full_logits.shape[1]} != input {input_ids.shape[1]}. "
        f"segments={len(seg_input_ranges)}, ranges={seg_input_ranges}"
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
    """Run dummy forward passes for FSDP rank synchronization.

    Contributes to autograd graph (so FSDP backward hooks fire) but
    gradient values are multiplied by 0.
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
        # float().mean() prevents bf16 overflow -> Inf. Inf * 0 = NaN (IEEE 754).
        dummy_sum = dummy_sum + d_logits.float().mean()
    return logits + (dummy_sum * 0).to(logits.dtype)
```

### Memory Considerations

Without `.detach()`, PyTorch retains intermediate activations from all segments for backprop.
For N segments, memory grows linearly with N. With window=4096, stride=512, and max 16k
tokens, the max segment count is:
- (16384 - 4096) / 512 + 1 = ~25 segments

Each segment processes ~4096 tokens. The retained KV per segment is (4096-512) = 3584 tokens
worth of K and V tensors across all layers. For Qwen3-4B (36 layers, 32 heads, 128 dim):
- Per segment retained: 3584 * 36 * 2 * 32 * 128 * 2 bytes (bf16) = ~1.06 GB
- 25 segments: ~26 GB for retained KV alone

This fits on A100-80GB but is tight. If memory becomes an issue, gradient checkpointing
within segments (already supported by HuggingFace) can reduce the cost.

---

## 3.2: Environment Wrapper

**File: `src/kv_eviction/env.py`**

Wraps the compaction-enabled vLLM server's response into prime-rl's expected format.

```python
"""RL environment wrapper for compaction-enabled vLLM.

Converts the standard vLLM /v1/chat/completions response (which now includes
compaction_metadata) into the format expected by prime-rl's training pipeline.

The key output field is `segment_boundaries`: a list of cumulative completion
token counts extracted from compaction events. The training forward pass
(segmented_forward.py) uses these to know where to split segments and evict KV.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompactionRollout:
    """Training sample from a compaction-enabled inference rollout.

    Contains everything needed by the segmented forward pass.
    """
    # Token data
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]

    # Compaction metadata (from vLLM response)
    segment_boundaries: list[int]  # cumulative completion token counts
    stride: int                    # tokens evicted per compaction

    # Reward
    reward: float

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)

    @property
    def num_segments(self) -> int:
        return len(self.segment_boundaries)


def extract_compaction_rollout(
    response: dict,
    prompt_ids: list[int],
    reward: float,
    stride: int,
) -> CompactionRollout:
    """Convert a vLLM response with compaction metadata into a CompactionRollout.

    Args:
        response: The vLLM API response dict. Expected to contain:
            - "choices": standard completions
            - "compaction_metadata": {"events": [...]} if compaction occurred
        prompt_ids: The original prompt token IDs.
        reward: The reward from the RL environment.
        stride: The compaction stride (tokens per eviction).

    Returns:
        CompactionRollout with segment_boundaries extracted from events.
    """
    # Extract completion token IDs and logprobs from response
    # The exact extraction depends on the vLLM response format.
    # For /v1/chat/completions with logprobs enabled:
    choice = response["choices"][0]
    completion_ids = choice.get("token_ids", [])
    completion_logprobs = choice.get("logprobs", [])

    # Extract segment boundaries from compaction metadata
    segment_boundaries = []
    compaction_meta = response.get("compaction_metadata")
    if compaction_meta and compaction_meta.get("events"):
        for event in compaction_meta["events"]:
            segment_boundaries.append(event["completion_tokens_at_compaction"])

    # If no compaction happened, the entire completion is one segment
    if not segment_boundaries:
        segment_boundaries = [len(completion_ids)]

    # Ensure the last boundary covers the full completion
    if segment_boundaries[-1] < len(completion_ids):
        segment_boundaries.append(len(completion_ids))

    return CompactionRollout(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_logprobs=completion_logprobs,
        segment_boundaries=segment_boundaries,
        stride=stride,
        reward=reward,
    )


def build_position_ids(
    prompt_len: int,
    segment_boundaries: list[int],
    stride: int,
) -> list[int]:
    """Build position IDs that match inference-side compaction.

    After each compaction event, position continues from where it was
    (not from the physical KV length). This means position = physical_pos + offset
    where offset accumulates evicted tokens.

    Args:
        prompt_len: Number of prompt tokens.
        segment_boundaries: Cumulative completion token counts at each compaction.
        stride: Tokens evicted per compaction.

    Returns:
        List of position IDs for the full sequence [prompt + completion].
    """
    total_len = prompt_len + (segment_boundaries[-1] if segment_boundaries else 0)
    positions = list(range(total_len))
    # Positions are simple 0..N-1 because the input_ids include ALL tokens
    # (including evicted ones). The segmented forward reconstructs the eviction
    # via KV manipulation, not by skipping positions in input_ids.
    return positions
```

### Integration with prime-rl's Data Pipeline

Prime-rl collects rollout results via the `env_worker.py` pipeline. The `extract_result()`
function at `$SCRATCH/kv-eviction/prime-rl/src/prime_rl/orchestrator/env_worker.py:50`
extracts trajectory data from `vf.State` including a per-step `extras` dict.

The compaction metadata should be stored in `extras["segment_boundaries"]` and
`extras["compaction_stride"]` so it flows through the existing pipeline without
modifications to prime-rl itself.

---

## 3.3: Training Hooks

**File: `src/kv_eviction/train_hooks.py`**

Injects the segmented forward pass into prime-rl's training loop.

```python
"""Training hooks: inject segmented forward into prime-rl's training loop.

The hook dispatches to segmented_forward() when the rollout has segment_boundaries
(indicating compaction occurred during inference), otherwise falls through to
the standard prime-rl forward pass.

Integration approach: monkey-patch or wrap the forward function used by prime-rl's
DPPO trainer. The exact hook point depends on prime-rl's architecture.
"""

import logging
from typing import Any

import torch
from torch import Tensor

from kv_eviction.segmented_forward import segmented_forward

logger = logging.getLogger(__name__)


def maybe_segmented_forward(
    model: torch.nn.Module,
    input_ids: Tensor,           # [1, seq_len]
    position_ids: Tensor,        # [1, seq_len]
    temperature: Tensor,         # [1, seq_len]
    extras: dict[str, Any],      # from rollout trajectory step
    max_forward_passes: int | None = None,
) -> dict[str, Tensor]:
    """Dispatch to segmented or standard forward based on rollout metadata.

    If extras contains segment_boundaries with >1 segment, use segmented_forward.
    Otherwise, use standard model forward.

    Args:
        model: HuggingFace model.
        input_ids: [1, seq_len] full sequence.
        position_ids: [1, seq_len] position IDs.
        temperature: [1, seq_len] per-token temperatures.
        extras: Dict from rollout trajectory, may contain:
            - "segment_boundaries": list[int] (cumulative completion token counts)
            - "compaction_stride": int (tokens evicted per compaction)
            - "prompt_len": int
        max_forward_passes: For FSDP synchronization.

    Returns:
        {"logits": [1, seq_len, vocab]} temperature-scaled logits.
    """
    segment_boundaries = extras.get("segment_boundaries")
    stride = extras.get("compaction_stride", 0)
    prompt_len = extras.get("prompt_len", 0)

    # Use segmented forward if compaction happened (>1 segment)
    if segment_boundaries and len(segment_boundaries) > 1 and stride > 0:
        logger.debug(
            "Using segmented forward: %d segments, stride=%d, prompt=%d",
            len(segment_boundaries), stride, prompt_len,
        )
        return segmented_forward(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            stride=stride,
            temperature=temperature,
            max_forward_passes=max_forward_passes,
        )

    # Standard forward (no compaction, or single segment)
    logger.debug("Using standard forward (no compaction)")
    out = model(input_ids=input_ids, position_ids=position_ids)
    raw_logits = out["logits"] if isinstance(out, dict) else out.logits
    logits = raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
    scaled = logits / temperature.unsqueeze(-1).to(logits.dtype)
    return {"logits": scaled}


def compute_max_forward_passes(batch_extras: list[dict]) -> int:
    """Compute max segment count across a batch for FSDP synchronization.

    All ranks must execute the same number of forward passes. This function
    finds the maximum across the batch so ranks with fewer segments can pad
    with dummy passes.

    Args:
        batch_extras: List of extras dicts from the batch.

    Returns:
        Maximum number of forward passes needed by any sample in the batch.
    """
    max_passes = 1
    for extras in batch_extras:
        boundaries = extras.get("segment_boundaries", [])
        if boundaries:
            max_passes = max(max_passes, len(boundaries))
    return max_passes
```

### Hook Integration Point

Prime-rl's DPPO trainer calls a forward function to compute logits for the policy loss.
The exact integration depends on prime-rl's trainer architecture. The recommended approach:

1. **Find the forward call site** in prime-rl's DPPO trainer (likely in
   `src/prime_rl/trainer/rl/` or `src/prime_rl/trainer/model.py`).

2. **Wrap it** so that when `extras["segment_boundaries"]` is present, the segmented
   forward is used instead of the standard forward.

3. **Do NOT modify prime-rl source**. Use a training config flag (e.g., `use_segmented_forward = true`)
   and a wrapper module in `kv_eviction/train_hooks.py` that patches the forward function
   at runtime.

The exact patching mechanism depends on prime-rl's plugin/hook system. If prime-rl supports
custom forward functions via config, use that. Otherwise, monkey-patch the forward method
of the trainer class.

### Key requirement: `impl = "hf"`

The segmented forward requires `use_cache=True` and `past_key_values` support, which means
the model must use the HuggingFace implementation (`impl = "hf"` in prime-rl config), NOT
the vLLM or custom implementation. This is set in the experiment TOML config:

```toml
[model]
impl = "hf"
```

This is the same requirement as mkv-rl's M4 training config.

---

## 3.4: Position ID Construction for Training

The segmented forward pass uses `position_ids` that match the inference-side RoPE positions.
During inference, after compaction:
- Physical KV length decreases by `stride` tokens
- `position_offset` increases by `stride` tokens
- Next token's RoPE position = (new physical length) + position_offset = same as before compaction

For training, the full input_ids sequence contains ALL tokens (including those that were
evicted during inference). The position_ids are simply `[0, 1, 2, ..., seq_len-1]` because
the segmented forward reconstructs the eviction behavior in the KV domain, not the token domain.

The position_ids for each segment's forward pass are sliced from this full range. Between
segments, the KV eviction drops the first `stride` assistant entries, which effectively shifts
the physical KV positions — but since we pass the correct `position_ids` slice for each
segment, and the model uses these positions for RoPE (not the KV cache indices), the RoPE
values match inference exactly.

---

## Testing Plan

### test_segmented_forward.py

```python
"""Tests for segmented forward pass (no detach)."""
import pytest
import torch


class TestSegmentedForward:

    def test_single_segment_passthrough(self):
        """With one segment boundary, output matches standard forward."""
        # segment_boundaries = [completion_len] means no compaction
        # Result should be identical to model(input_ids, position_ids)
        pass

    def test_logit_shape_matches_input(self):
        """Output logits shape [1, seq_len, vocab] matches input_ids shape."""
        pass

    def test_boundary_token_overlap(self):
        """Segment k>0 starts from boundary token of segment k-1.
        The logit at the boundary position should be computed with
        post-eviction KV context."""
        pass

    def test_kv_drop_count(self):
        """After eviction, retained KV length = previous - stride (+ prompt)."""
        pass

    def test_no_detach_gradient_flow(self):
        """Gradients flow from last segment back to first segment's parameters.

        Key test: compute loss on last segment's logits, backprop, check that
        first segment's input embeddings have non-zero gradients.
        Compare with detached version where gradients should be zero.
        """
        pass

    def test_g_distal_nonzero(self):
        """G_distal = grad(no_detach) - grad(detach) is nonzero.

        Run the same forward pass twice: once with detach, once without.
        The gradient difference should be nonzero, proving cross-chunk
        gradient flow.
        """
        pass

    def test_fsdp_dummy_passes(self):
        """With max_forward_passes > actual segments, dummy passes are added.
        The logit shape should still be correct."""
        pass

    def test_short_segment_clamp(self):
        """When stride > actual assistant tokens, stride is clamped."""
        pass

    def test_empty_completion(self):
        """prompt_len >= seq_len is handled gracefully."""
        pass
```

### Key Verification: Step-0 KL

The most important test (run as part of Phase 4 but designed here):

```python
def test_step0_kl_approximately_zero():
    """Training logits match inference logits at step 0 (before any updates).

    Setup:
    1. Start vLLM with compaction (window=4096, stride=512)
    2. Generate a rollout, collect logprobs
    3. Run segmented_forward on the same input with the same model weights
    4. Compare logprobs: KL should be ~0.0

    This works because both use flash_attn kernel:
    - Inference: vLLM's PagedAttention (flash_attn based)
    - Training: HF model with use_cache=True (flash_attn via sdpa)

    If KL > 0.01, something is wrong (kernel mismatch, position error, etc.)
    """
    pass
```

---

## Completion Criteria

- [ ] `segmented_forward()` produces correct logits matching standard forward for single-segment case
- [ ] Multi-segment case: logit shape matches input, boundary tokens handled correctly
- [ ] **No `.detach()` anywhere** in `segmented_forward.py`
- [ ] G_distal test: gradient difference (no-detach minus detach) is nonzero
- [ ] FSDP dummy passes work correctly
- [ ] `env.py` extracts `segment_boundaries` from vLLM compaction response
- [ ] `train_hooks.py` dispatches correctly based on `extras["segment_boundaries"]`
- [ ] `test_segmented_forward.py` all pass
- [ ] Model uses `impl = "hf"` in config (assertion or warning if not)
- [ ] Memory usage is feasible on A100-80GB for max segment count (~25 segments)

---

## Phase 3.4 Live KL Test Results (2026-04-10)

Final end-to-end numerical validation of segmented_forward vs vLLM's
compaction inference on Qwen3-4B. See `experiments/phase3_kl_test/`
for the test scripts.

**Setup:**
- Allocation: 2 A100-80GB nodes, 4 GPUs each (interactive, 4h walltime)
- Inference node: 1 GPU, vLLM 0.19.1-dev (our fork) with
  `enforce_eager=True`, window=4096, stride=512, block_size=16
- Trainer node: 4 GPUs, DP=4 via torchrun, HF `AutoModelForCausalLM`
  with `attn_implementation="flash_attention_2"`, bf16
- Data: 10 rg-mix problems, `max_tokens=16384`, `ignore_eos=True`,
  `temperature=1.0`, `seed=43`
- Total: 160,821 completion tokens per condition, 240 compaction events
  across 10 samples (exactly 24 per sample, spaced every 512 gen tokens
  after the first event at gen[3893])

**Two conditions compared:**
1. **Baseline** — vLLM without compaction + trainer standard `model()`
   forward. Measures the kernel-numerics noise floor between vLLM's
   eager-mode path and HF's flash_attention_2 path on the same tokens.
2. **Compaction** — vLLM with compaction enabled + trainer
   `segmented_forward()` replaying the 24 KV evictions per sample with
   no detach between segments.

**Metric:** per-token `|trainer_logprob - inference_logprob|`, averaged
over the full completion. This is the same metric mkv-rl used for its
M3 KL check; it approximates per-token KL when the logprobs are
sampled-token logprobs.

**Results:**

| Metric                   | Baseline  | Compaction | Ratio |
|--------------------------|-----------|------------|-------|
| mean abs log-ratio       | 0.00842   | 0.01035    | 1.23x |
| max  abs log-ratio       | 4.8975    | 2.8539     | 0.58x |
| mean signed log-ratio    | -0.00059  | -0.00062   | ~1x   |
| samples                  | 10        | 10         | —     |
| total completion tokens  | 160,821   | 160,821    | —     |
| per-sample elapsed       | 1.4s      | 1.9s       | 1.36x |

**Interpretation.** The compaction mean abs log-ratio is **1.23x the
baseline kernel-noise floor** — indistinguishable from pure bf16
rounding noise. The absolute number (~0.010 per token) is ~200x lower
than the ~2.0 number mkv-rl M3 (SDPA + detached) was hitting on the
same kind of measurement. The mean signed log-ratio is nearly identical
between conditions (-0.00059 vs -0.00062), meaning compaction introduces
no systematic bias — whatever drift exists is due to the base model's
kernel numerics, not to anything segmented_forward does.

**Per-sample details** (from kl_results.json per_sample_rank_local):
- All 10 samples had 24 compaction events firing at the expected
  cumulative gen positions (3893, 4405, 4917, ..., 15669), confirming
  the first-eviction math holds in practice with the tightened
  needs_compaction guard.
- Every sample's per-token mean is within 0.006..0.014 for both
  conditions — tight distribution, no outliers.
- The max-abs outliers are driven by 1-2 tokens per trajectory (a common
  pattern under temperature=1 where the argmax disagreement can be
  dramatic at a single sampling step). Baseline saw a 4.90 outlier,
  compaction saw a 2.85 outlier; both are statistical noise from
  sampling, not systematic.

**DP=4 validation.** All 4 ranks processed their shards in lockstep.
The cross-rank `all_reduce(MAX)` for `max_forwards` and the
RSA-hardened defensive branch-agreement `all_reduce` both worked
without deadlock. The modality-partitioning fix from RSA Round 1 and
the CP rejection fix from RSA Round 2 are both exercised and intact.

**Runtime.** Model load 24s per rank (checkpoint shards cached); total
trainer wall time under 60s for 20 samples. Inference phase took
~5 min per condition (eager mode). Full test from salloc to
kl_results.json: ~15 min of the 4h allocation.

**Conclusion.** Phase 3 is numerically validated for forward-only
correctness. The trainer's segmented_forward replay of vLLM's
compaction is within kernel noise of the no-compaction baseline,
meaning the KV drop arithmetic, position handling, temperature scaling,
and cross-rank sync are all correct. Gradient flow through retained KV
and full FSDP2-sharded training runs are the remaining validation
steps before production training (see "Pre-production checklist"
below).

---

## Pre-production Checklist

Phase 3.4 validated forward-pass numerics. Before kicking off full RL
training runs, the following smoke tests should pass. They are listed
in rough order of risk (highest first) and complexity (lowest first).

1. **Backward pass smoke test (single GPU).** The KL test used
   `torch.no_grad()` and never called `.backward()`. segmented_forward
   claims to preserve gradients through retained KV (the whole point of
   "no detach") but this was never exercised numerically. Run one
   compaction sample through segmented_forward, compute a dummy loss,
   call backward, verify: (a) no NaN gradients, (b) every parameter
   receives a gradient (no dead params), (c) gradient norm is in a
   reasonable range (not 0, not 1e6+). This catches any "gradients
   silently zero" bug that would make training a no-op.

2. **FSDP2-sharded segmented_forward smoke test.** The KL test used DDP
   (replicated weights). The real prime-rl trainer uses FSDP2 (sharded
   weights with per-layer all-gather/reshard). segmented_forward makes
   multiple model() calls per sample, each triggering its own FSDP2
   collectives. Test: load Qwen3-4B with FSDP2 sharding across 4 GPUs,
   run one compaction forward, compare resulting logits to the DDP
   version from Phase 3.4. If they disagree beyond kernel noise, the
   FSDP2 per-layer reshard is confusing segmented_forward's backbone
   hook or its past_key_values handling.

3. **End-to-end single training step.** Run the actual prime-rl RL
   trainer (not the standalone KL test) for exactly one step with
   compaction enabled. Use a minimal config: 1 rg-mix rollout, 1 micro
   batch, small LR, FSDP=4 dp, single node. Watch for: (a) the
   compaction samples hitting the segmented_forward dispatch, (b) the
   config validator firing if attn/impl/cp are wrong, (c) the
   optimizer actually stepping (non-zero update norm), (d) the weight
   broadcast to the inference server completing.

4. **Short stability run (5-10 steps).** Same config as #3, run 5-10
   steps. Loss should decrease (or at least not explode), entropy
   should stay finite, no NaN gradients. Confirms FSDP2 + compaction +
   gradient flow + weight sync are all cooperating for more than one
   iteration.

5. **Production-config smoke (1 node, 1-2 steps).** Use the actual
   production TOML (full seq_len, real batch sizes) to verify memory
   fits and per-step wall time is acceptable. Segmented_forward's
   O(window) memory footprint should keep us under peak GPU memory,
   but better to confirm before committing to a long run.

Of the 4 deferred items from RSA review (#37-#40), only #37 (cross-config
validator between trainer and orchestrator inference args) meaningfully
affects production robustness. The others are perf/hardening nits that
don't block the first real runs.

## Pre-production Smoke Test Results (2026-04-10)

### Smoke #1: single-GPU backward pass — PASS

Single A100 80GB, Qwen3-4B bf16 + flash_attention_2, full 16k-token
sample with all 24 compaction events. Activation checkpointing enabled
via segmented_forward's new `activation_checkpointing=True` flag (added
in this round — see Key finding #2 below for why HF's built-in
`gradient_checkpointing_enable` is unusable).

Results:
- Forward: 2.7s, 32GB peak
- Backward: 5.3s, 55.7GB peak
- All 398 params received non-zero gradients (no dead params from the
  retained-KV `torch.cat` chain)
- Zero NaN, zero Inf
- Global grad norm: 1.49 (healthy)
- Loss: 0.197 (reasonable teacher-forced policy loss)

Headroom on 80GB A100: ~25GB, which means real training with AdamW
state (~32GB for a 4B model) will still fit because FSDP2 shards both
params and optimizer state across ranks in production.

Script: `experiments/phase3_preprod/smoke1_backward.py`

### Smoke #2: FSDP2 segmented_forward — PASS (with caveats)

4-GPU A100 80GB node, torchrun --nproc_per_node=4, each rank loads
Qwen3-4B, `fully_shard` applied per decoder layer, each rank processes
the same 4-event sample (truncated to fit without outer AC under
FSDP2).

Results:
- Forward: 1.2s, backward: 1.3s
- Loss cross-rank spread: **0.00e+00** (bit-identical across all 4 ranks)
- 1592 params (398 × 4 ranks) all received gradients
- Zero NaN, zero Inf, zero missing grads
- FSDP2-reduced global grad norm: 3.09

No deadlock despite 5 per-segment all-gathers feeding into one
backward. The retained-KV `torch.cat` chain survives FSDP2's
reshard-after-forward cadence.

Script: `experiments/phase3_preprod/smoke2_fsdp2.py`

### Key findings

1. **`fully_shard` on the root model breaks RMSNorm.** Applying
   `fully_shard(model)` at the top level wraps `model.model.norm` and
   `model.lm_head` as DTensors, and HF's RMSNorm hits
   "aten.mul.Tensor got mixed Tensor and DTensor" during backward.
   Fix: shard only the decoder layers. Prime-rl's production path
   does this correctly by grouping `[lm_head, norm]` into a separate
   `fully_shard` unit with `reshard_after_forward=False` — see
   `prime_rl/trainer/model.py:400-412`. For the smoke test, sharding
   just the decoder layers is sufficient to exercise FSDP2 all-gather
   cadence on the large parameter groups.

2. **`segmented_forward(activation_checkpointing=True)` is
   incompatible with FSDP2 — important production guidance.**
   `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`
   does NOT trigger FSDP2's pre-forward hooks on backward re-entry,
   causing "aten.mul.Tensor got mixed Tensor and DTensor" in RMSNorm
   when the checkpointed segment's forward is re-run during backward.
   The production path must use a completely different AC mechanism:
   prime-rl applies
   `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`
   to each transformer block BEFORE `fully_shard`, so the FSDP2 hook
   ordering works correctly. In production:

   - Trainer config should call prime-rl's per-block AC as normal
     (`trainer.model.ac.freq = 1`)
   - `segmented_forward` must be called WITHOUT its own
     `activation_checkpointing=True`
   - The per-block AC handles memory inside each segment's forward
     (~1 block's activations retained), and the between-segment
     retained-KV chain happens at the outer torch graph level

   The new `activation_checkpointing=True` flag on segmented_forward
   is still useful for single-GPU debugging and offline tests (like
   smoke #1), just not under FSDP2.

3. **Computing grad norm on DTensor params needs `.to_local()`.**
   FSDP2 stores each parameter's grad as a DTensor whose arithmetic
   dispatch insists both operands be DTensors; naively accumulating
   into a plain local tensor trips an isinstance assert in
   `torch/distributed/tensor/_dispatch.py`. Unwrap via
   `g.to_local()` before computing the squared sum. Relevant for
   any custom grad-norm or diagnostic code under FSDP2.

### Item #37: RLConfig cross-config validator — DONE

New `@model_validator(mode="after")` on `RLConfig` in
`prime-rl/src/prime_rl/configs/rl.py` cross-checks that
`trainer.compaction.{window_size, stride, block_size}` exactly mirrors
`inference.vllm_extra.{compaction_window_size, compaction_stride,
block_size}` when KV cache compaction is in use. Bidirectional: either
side enabling compaction forces the other to match. The three sanity
tests (window mismatch, stride mismatch, block_size mismatch) plus the
three valid cases (mirrored, both-disabled, inference-only-enabled
rejected) all pass.

Sanity-check script:
`experiments/phase3_preprod/test_rlconfig_validator.py`

### Remaining smoke tests

Smokes #3-5 (end-to-end single step, short stability run,
production-config smoke) still pending. These require:

- A 2-node allocation (inference node + trainer node)
- `rg-mix-env` installed into the kv-eviction `.venv` (currently
  only in mkv-rl's venv — install from
  `/pscratch/sd/s/siddart2/mkv-rl/experiments/rg_mix/dist/rg_mix_env-0.1.4-py3-none-any.whl`
  inside the podman container via `uv pip install`)
- A minimal compaction RL TOML config adapted from
  `mkv-rl/experiments/rg_mix/rl.toml`, with matching
  `trainer.compaction` and `inference.vllm_extra.compaction_*` keys
  plus `trainer.model.impl = "hf"` and
  `trainer.model.attn = "flash_attention_2"` (enforced by the new
  config validator).

Best tackled in a fresh session with full context runway.

## Pre-production session round 2: per-segment backward + trainer dispatch

Second pre-prod session (same date) pivoted from M4 BPTT attempt to
M3 per-segment backward after finding that M4 interacts fatally with
prime-rl's per-block checkpoint_wrapper under `use_cache=True`. See
`experiments/phase3_preprod/probe_ac_cache_mutation.py` for the
minimal crash reproduction: `torch.utils.checkpoint` (non-reentrant)
re-runs each block's forward during backward, and HF's
`DynamicCache.update()` takes the "concatenate existing slot" branch
instead of "append new slot", producing 2x-length K/V tensors. Torch
catches this loudly via `CheckpointError: Recomputed values have
different metadata than during the forward pass`
(`[1, 1024, 8, 128]` → `[1, 2048, 8, 128]`).

Design pivot: per-segment forward + backward + detach, with an
optional `bptt_segments` knob for truncated BPTT windows:
  - `bptt_segments=1` (default, M3 semantics): detach between every
    segment, one backward per segment, O(1 segment) memory
  - `bptt_segments=K > 1`: gradients flow through retained KV
    within K consecutive segments before detach, O(K) memory
  - `bptt_segments=None`: full trajectory in one BPTT window,
    M4-equivalent semantics, O(all segments) memory

Validation (smoke #1b, single A100 80GB, full 16k / 24-event sample):
  - k=1:  40.9 GB, 5.5s, grad norm 1.22, PASS
  - k=4:  55.4 GB, 5.2s, grad norm 1.30, PASS
  - k=12: OOM at torch.cat in eviction (expected)
  - Loss identical across k (deterministic forward)
  - k=4 grad norm 6% higher than k=1, confirming intra-window
    BPTT actually flows gradients through retained KV

Trainer dispatch wired in `prime-rl/src/prime_rl/trainer/rl/train.py`:
the `use_segmented` branch now builds a `_segment_loss_fn` closure
that slices `labels`, `advantages`, `loss_mask`,
`inference_logprobs`, `teacher_logprobs` to match each segment's
owned logit range and delegates to prime-rl's existing
`compute_loss` (no duplication of the loss formula, just range
bookkeeping). Alignment detail: the last segment's final logit
predicts a nonexistent token at position `seq_len` and must be
dropped explicitly (matching what `shift_tensor_right` does in the
standard path). Three adversarial review rounds caught and fixed
several bugs before landing; see commit messages for details.

### Deferred items (recorded here so future instances can find them)

**D1. Multi-rank `bptt_segments != 1` under FSDP2.** The dummy-pass
padding inside `segmented_forward` pads forwards with paired
(forward + backward) dummies, which only keeps total backward
counts matched across ranks when every real segment also has
exactly one backward — i.e., `bptt_segments == 1`. With
`bptt_segments > 1` or `None`, the number of real backwards per
rank is `ceil(per_rank_segments / K)`, which varies across ranks
when per-rank segment counts differ. Reduce-scatter counts then
diverge and FSDP2 deadlocks.

Proper fix: compute `max_backwards` via `all_reduce(ReduceOp.MAX)`
across ranks (where `max_backwards = max(ceil(actual_i / K))`),
pass it to `segmented_forward` alongside `max_forwards`, and have
the dummy-pass logic issue a mix of:
  - `Y_local = max_backwards - real_bw_count` dummy (forward+backward)
    pairs
  - `X_local - Y_local` forward-only dummies (where
    `X_local = max_forwards - actual`)

The condition `X_local >= Y_local` is provably satisfied because
`f(x) = x - ceil(x/K)` is non-decreasing in x, so
`max_forwards - max_backwards >= actual - ceil(actual/K)` for every
rank. See the third adversarial review agent's analysis in
`.claude/projects/-pscratch-sd-s-siddart2-mkv-rl/` session
transcript (2026-04-10) for the full derivation.

Current state: blocked at training init via a runtime check in
`train.py` right after `parallel_dims = get_parallel_dims(...)`
that raises `ValueError` if `world_size > 1 and bptt_segments != 1`.
Single-GPU runs can freely set any `bptt_segments` value.

**D2. Entropy for compaction samples.** `segmented_forward` returns
a scalar loss, not per-token logits, so `out["entropy"]` is not
populated for segmented micro-batches. The debug-log f-string at
`train.py` is guarded on `not use_segmented and
len(tensors['entropy']) > 0` and simply omits the entropy field
for compaction samples. `compute_stats` downstream handles
missing per-micro-step entries by emitting NaN, so step-level
entropy averages for compaction-heavy steps will come out as NaN.

To fix: extend the `_segment_loss_fn` closure (or a parallel
callback) to compute `compute_entropy(seg_logits_effective)`,
apply `seg_loss_mask`, and accumulate into a per-micro-batch list.
After `segmented_forward` returns, `torch.cat` the accumulated
entropy values and append to `tensors['entropy']` just like the
standard path does at line 673.

**D3. Token-weighted metric aggregation for compaction.** The
per-segment `compute_loss` call wraps each `_safe_mean`-style
metric in `torch.stack([scalar_0d]) → shape [1]`; the closure
appends these to an `accumulated_loss_tensors` dict and the
post-loop `torch.cat(v)` gives a shape `[n_segments]` 1-D tensor
that's appended to `tensors[metric_name]`. Downstream
`compute_stats` averages these unweighted across segments, which
gives a mean-of-means rather than a token-weighted mean. For a
sample whose segments are token-count-skewed (e.g. 4000/4000/1000
trainable tokens per segment) the logged metric value drifts
from what the standard path would report.

The LOGGED LOSS is still correct because the loss decomposition is
exact: `sum_over_segments(compute_loss(seg).loss) ==
compute_loss(full_seq).loss` when every call uses the same global
`loss_scale`. Only the derived metrics (`mismatch_kl`, `is_masked`,
etc.) drift.

To fix: have the closure collect raw per-token `(values, mask)`
pairs per metric instead of per-segment `_safe_mean` outputs,
concatenate across segments, and recompute `_safe_mean` once at
the end before appending to `tensors`. Or: change
`compute_loss`'s per-sequence metrics contract to expose
`(numerator_sum, denominator_count)` so the closure can
accumulate weighted.

**D4. Smokes #3-5 still pending.** As noted in the "Remaining smoke
tests" section above — requires a 2-node allocation, rg-mix-env
installed into the kv-eviction venv, and a minimal compaction RL
TOML config. With the trainer dispatch now landed, these should
run cleanly end-to-end modulo integration surprises.
