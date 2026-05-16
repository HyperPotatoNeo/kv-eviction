# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per_call_segmented_forward (persistent-cache path).

After the single-forward-pre-eviction refactor (plans/single_forward_pre_eviction.md
Phase 5), per_call_segmented_forward runs ONE HF forward per call against a
DynamicCache carried across calls. Admission events are handled inline via
eviction-aware position_ids — no two-phase split, no cache splice.

These tests use a mock HF-style model that returns deterministic logits
(logits[i] = position_ids[i] broadcast to vocab) so we can verify:
  1. Owned ranges partition [0, full_seq_len).
  2. compute_num_per_call_forwards returns len(calls).
  3. Eviction-aware position_ids match vLLM's RoPE frame after admission.
  4. The persistent cache carries K/V across calls.
  5. Dummy passes pad to max_forward_passes for FSDP2 sync.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kv_eviction.segmented_forward import (
    compute_num_per_call_forwards,
    per_call_segmented_forward,
)


@dataclass
class _CallSpec:
    """Minimal CallWire shape (we only access the fields per_call uses)."""

    submitted_prompt_ids: list[int]
    completion_ids: list[int]
    compaction_events: list = None  # default empty in CallWire schema

    def __post_init__(self):
        if self.compaction_events is None:
            self.compaction_events = []


class _RecordingModel(torch.nn.Module):
    """Mock HF model: returns logits[i] = position_ids[i] (broadcast to vocab).

    Records every forward call's seq_len, position_ids, and incoming
    cache length so we can assert on persistent-cache invariants.
    """

    def __init__(self, vocab_size: int = 32, num_layers: int = 2,
                 num_heads: int = 1, head_dim: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.calls: list[dict] = []
        # A tiny trainable parameter so .backward() has somewhere to flow.
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=False):
        seq_len = input_ids.shape[1]
        base = position_ids.float().unsqueeze(-1).expand(-1, -1, self.vocab_size)
        logits = base + self._dummy

        # Capture pre-extension cache length per layer (for assertions).
        pre_cache_len = 0
        if past_key_values is not None and hasattr(past_key_values, "layers") \
                and past_key_values.layers:
            pre_cache_len = past_key_values.layers[0].keys.shape[2]
        elif past_key_values is not None and hasattr(past_key_values, "key_cache") \
                and past_key_values.key_cache:
            pre_cache_len = past_key_values.key_cache[0].shape[2]

        self.calls.append({
            "seq_len": seq_len,
            "position_ids": position_ids[0].tolist(),
            "input_ids": input_ids[0].tolist(),
            "pre_cache_len": pre_cache_len,
            "had_past_kv": past_key_values is not None,
        })

        # Append K/V to the cache so subsequent calls inherit them.
        if use_cache and past_key_values is not None:
            for layer in range(self.num_layers):
                k = position_ids.float().view(1, 1, seq_len, 1).expand(
                    1, self.num_heads, seq_len, self.head_dim
                ).contiguous()
                v = (position_ids.float() + 100).view(1, 1, seq_len, 1).expand(
                    1, self.num_heads, seq_len, self.head_dim
                ).contiguous()
                past_key_values.update(k, v, layer)

        return {"logits": logits, "past_key_values": past_key_values}


def _merged_from_calls(calls: list[_CallSpec]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the merged sequence for a list of EXTENSION calls.

    For no-admission rollouts, each call's submitted_prompt is a strict
    prefix of the merged sample. The merged sequence equals the LAST
    call's (submitted_prompt + completion).
    """
    last = calls[-1]
    merged = list(last.submitted_prompt_ids) + list(last.completion_ids)
    input_ids = torch.tensor(merged, dtype=torch.long).unsqueeze(0)
    position_ids = torch.arange(len(merged), dtype=torch.long).unsqueeze(0)
    return input_ids, position_ids, len(merged)


# ─── compute_num_per_call_forwards ────────────────────────────────────


def test_compute_num_per_call_forwards_no_admission_one_per_call():
    """One forward per call for no-admission rollouts."""
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4, 5, 6], completion_ids=[7]),
    ]
    assert compute_num_per_call_forwards(calls) == 3


def test_compute_num_per_call_forwards_with_admission_still_one_per_call():
    """Admission events no longer double the count — single forward per call."""
    admission_event = type("E", (), {"num_output_tokens_at_compaction": 0})()
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(
            submitted_prompt_ids=[1, 2, 3, 4],
            completion_ids=[5],
            compaction_events=[admission_event],
        ),
    ]
    assert compute_num_per_call_forwards(calls) == 2


def test_compute_num_per_call_forwards_mixed_counts_total():
    """Mixed admission/non-admission calls: one forward each."""
    admission_event = type("E", (), {"num_output_tokens_at_compaction": 0})()
    calls = [
        _CallSpec(submitted_prompt_ids=[1], completion_ids=[2]),
        _CallSpec(
            submitted_prompt_ids=[1, 2], completion_ids=[3],
            compaction_events=[admission_event],
        ),
        _CallSpec(submitted_prompt_ids=[1, 2, 3], completion_ids=[4]),
        _CallSpec(
            submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5],
            compaction_events=[admission_event],
        ),
    ]
    assert compute_num_per_call_forwards(calls) == 4


def test_compute_num_per_call_forwards_with_midgen_only():
    """Mid-gen events (num_output_tokens > 0) don't change the count."""
    midgen_event = type(
        "E", (),
        {"num_output_tokens_at_compaction": 16},
    )()
    calls = [
        _CallSpec(
            submitted_prompt_ids=[1, 2], completion_ids=[3],
            compaction_events=[midgen_event],
        ),
    ]
    assert compute_num_per_call_forwards(calls) == 1


# ─── No-admission per-call dispatch ───────────────────────────────────


def test_per_call_owned_ranges_partition_merged_frame():
    """Owned ranges across all calls must exactly partition [0, full_seq_len)
    when summed up. No overlaps, no gaps."""
    calls = [
        _CallSpec(submitted_prompt_ids=[10, 11], completion_ids=[20, 21]),  # len 4
        _CallSpec(submitted_prompt_ids=[10, 11, 20, 21, 30, 31], completion_ids=[40]),  # len 7
        _CallSpec(
            submitted_prompt_ids=[10, 11, 20, 21, 30, 31, 40, 50, 51],
            completion_ids=[60, 61],
        ),  # len 11
    ]
    input_ids, pos_ids, full_len = _merged_from_calls(calls)
    model = _RecordingModel()

    owned_ranges: list[tuple[int, int]] = []

    def fake_loss(seg_logits, full_start, full_end):
        owned_ranges.append((full_start, full_end))
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model,
        calls,
        merged_input_ids=input_ids,
        merged_position_ids=pos_ids,
        loss_fn=fake_loss,
        max_forward_passes=3,
        device=torch.device("cpu"),
    )

    assert owned_ranges[0][0] == 0
    for i in range(len(owned_ranges) - 1):
        assert owned_ranges[i][1] == owned_ranges[i + 1][0], (
            f"gap or overlap at boundary {i}: "
            f"prev_end={owned_ranges[i][1]} next_start={owned_ranges[i + 1][0]}"
        )
    assert owned_ranges[-1][1] == full_len


def test_per_call_runs_one_forward_per_call_with_persistent_cache():
    """Each call's forward processes ONLY the new tokens it contributes;
    the cache carries K/V across calls."""
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3, 4]),  # contributes [0, 4)
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7]),  # contributes [4, 7)
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=2,
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 2
    # Call 0: forwards new tokens [0, 4), cache starts empty.
    assert model.calls[0]["seq_len"] == 4
    assert model.calls[0]["position_ids"] == [0, 1, 2, 3]
    assert model.calls[0]["pre_cache_len"] == 0
    # Call 1: forwards new tokens [4, 7), cache contains call 0's K/V.
    assert model.calls[1]["seq_len"] == 3
    assert model.calls[1]["position_ids"] == [4, 5, 6]
    assert model.calls[1]["pre_cache_len"] == 4


def test_per_call_pads_with_dummy_passes_for_fsdp_sync():
    """When max_forward_passes > local forward count, dummy passes fill the gap."""
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=4,
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 4
    # Real forward: 3 tokens. Dummies: 1-token slice each.
    assert model.calls[0]["seq_len"] == 3
    for dummy in model.calls[1:]:
        assert dummy["seq_len"] == 1


# ─── Admission per-call dispatch ──────────────────────────────────────


def test_per_call_admission_single_forward_with_eviction_aware_positions():
    """An admission call runs ONE forward over the post-trim merged
    sequence; position_ids skip the evicted range and bump subsequent
    tokens by total_evicted (matching vLLM's K rotations)."""
    admission_event = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 0,
            "evict_start": 2,
            "tokens_evicted": 2,
            "new_user_fragment_len": 2,
        },
    )()
    # Pre-trim submitted = [10, 11, 99, 99, 30, 31] (6 tokens).
    # phase1_token_count = 6 - 2 = 4. evict_end = 4.
    # L1_kept = 4 - 2 = 2; P2 = nuf_len + comp_len = 2 + 2 = 4.
    # Merged contribution = L1_kept + P2 = 6.
    calls = [
        _CallSpec(
            submitted_prompt_ids=[10, 11, 99, 99, 30, 31],
            completion_ids=[40, 41],
            compaction_events=[admission_event],
        ),
    ]
    # POST-trim merged: [10, 11, 30, 31, 40, 41].
    merged = [10, 11, 30, 31, 40, 41]
    input_ids = torch.tensor([merged], dtype=torch.long)
    pos_ids = torch.arange(len(merged), dtype=torch.long).unsqueeze(0)
    model = _RecordingModel()

    captured_ranges: list[tuple[int, int]] = []

    def fake_loss(seg_logits, start, end):
        captured_ranges.append((start, end))
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=1,
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 1
    # Single forward covers all 6 merged tokens.
    assert model.calls[0]["seq_len"] == 6
    # Eviction-aware positions: [0, 1] for the protected prefix, then
    # [4, 5, 6, 7] for the post-eviction portion (offset by total_evicted=2).
    assert model.calls[0]["position_ids"] == [0, 1, 4, 5, 6, 7]
    # Single loss invocation covering the merged contribution.
    assert captured_ranges == [(0, 6)]


def test_per_call_admission_with_kept_middle_inserts_eviction_gap():
    """Regression: when phase1 has tokens AFTER the evicted range
    (kept_middle), the trainer's position_ids must skip the evicted
    range in the middle of the kept prompt. Previously the code
    treated all L1_kept tokens as contiguous [0, L1_kept), which
    matched vLLM only when evict_end == phase1_token_count.

    Concrete: dummy-chat-env step 2 has sub_len=116, nuf_len=9,
    evict_start=16, tokens_evicted=48 → phase1_token_count=107,
    evict_end=64, L1_kept=59 (= 16 protected + 43 kept middle).
    Correct logical positions: [0..15] ++ [64..106] ++ [107..123]."""
    admission_event = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 0,
            "evict_start": 16,
            "tokens_evicted": 48,
            "new_user_fragment_len": 9,
        },
    )()
    sub_len = 116
    comp_len = 8
    calls = [
        _CallSpec(
            submitted_prompt_ids=list(range(sub_len)),
            completion_ids=list(range(1000, 1000 + comp_len)),
            compaction_events=[admission_event],
        ),
    ]
    # Merged length = (sub_len - tokens_evicted) + comp_len = 68 + 8 = 76
    merged_len = (sub_len - 48) + comp_len
    input_ids = torch.zeros((1, merged_len), dtype=torch.long)
    pos_ids = torch.arange(merged_len, dtype=torch.long).unsqueeze(0)
    model = _RecordingModel()

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=1,
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 1
    expected_positions = (
        list(range(0, 16))      # protected prefix
        + list(range(64, 107))  # kept middle (post-eviction-gap)
        + list(range(107, 124)) # NUF (9) + completion (8) at logical [107, 124)
    )
    assert model.calls[0]["position_ids"] == expected_positions


def test_per_call_aggregate_descriptor_combines_multiple_admission_events():
    """When vLLM emits multiple admission events per call (one per iteration
    of the eviction loop), _aggregate_admission_descriptor sums tokens_evicted
    and takes new_user_fragment_len from the last event."""
    from kv_eviction.segmented_forward import _aggregate_admission_descriptor

    event_a = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 0,
            "evict_start": 4,
            "tokens_evicted": 16,
            "new_user_fragment_len": 8,
        },
    )()
    event_b = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 0,
            "evict_start": 4,
            "tokens_evicted": 16,
            "new_user_fragment_len": 8,
        },
    )()
    call = _CallSpec(
        submitted_prompt_ids=[0] * 100,
        completion_ids=[1] * 10,
        compaction_events=[event_a, event_b],
    )
    desc = _aggregate_admission_descriptor(call)
    assert desc["evict_start"] == 4
    assert desc["total_evicted"] == 32
    assert desc["evict_end"] == 36
    assert desc["new_user_fragment_len"] == 8


def test_per_call_evict_start_zero_rejected():
    """evict_start == 0 (no protected prefix) is rejected."""
    import pytest

    from kv_eviction.segmented_forward import _aggregate_admission_descriptor

    event = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 0,
            "evict_start": 0,
            "tokens_evicted": 4,
            "new_user_fragment_len": 2,
        },
    )()
    call = _CallSpec(
        submitted_prompt_ids=[0] * 10,
        completion_ids=[1] * 2,
        compaction_events=[event],
    )
    with pytest.raises(NotImplementedError, match="evict_start >= 1"):
        _aggregate_admission_descriptor(call)


def test_per_call_cache_splice_helper_drops_evicted_range():
    """_splice_dynamic_cache removes [evict_start, evict_end) from each
    layer's K/V. Kept for use by debug/diagnostic tooling and as the
    inverse-operation building block; the production per-call path
    handles admission via position_ids rather than splicing."""
    from transformers import DynamicCache

    from kv_eviction.segmented_forward import _splice_dynamic_cache

    cache = DynamicCache()
    for layer in range(2):
        k = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
        v = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1) + 100
        cache.update(k, v, layer)

    spliced = _splice_dynamic_cache(cache, evict_start=3, evict_end=7)
    layer0 = spliced.layers[0] if hasattr(spliced, "layers") else None
    if layer0 is not None:
        k = layer0.keys.squeeze().tolist()
        v = layer0.values.squeeze().tolist()
    else:
        k = spliced.key_cache[0].squeeze().tolist()
        v = spliced.value_cache[0].squeeze().tolist()
    assert k == [0, 1, 2, 7, 8, 9]
    assert v == [100, 101, 102, 107, 108, 109]


def test_per_call_admission_then_extension_chains_position_offset():
    """A sample with admission at C0 followed by an extension call C1
    must continue C1's positions in the pre-eviction frame (offset by
    total_evicted from C0)."""
    admission_event = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 0,
            "evict_start": 2,
            "tokens_evicted": 2,
            "new_user_fragment_len": 2,
        },
    )()
    # C0 (admission): pre-trim sub=[10,11,99,99,30,31], comp=[40,41].
    #   Merged contribution: 6 tokens [10,11,30,31,40,41].
    # C1 (extension after C0's trim, no new admission):
    #   sub (post-C0-trim) = [10,11,30,31,40,41,50] (one new user token).
    #   comp = [60].
    #   call_input_len = 7 + 1 = 8 → covers merged [0, 8).
    #   New contribution: merged [6, 8) = [50, 60].
    calls = [
        _CallSpec(
            submitted_prompt_ids=[10, 11, 99, 99, 30, 31],
            completion_ids=[40, 41],
            compaction_events=[admission_event],
        ),
        _CallSpec(
            submitted_prompt_ids=[10, 11, 30, 31, 40, 41, 50],
            completion_ids=[60],
        ),
    ]
    # Merged frame = [10, 11, 30, 31, 40, 41, 50, 60].
    merged = [10, 11, 30, 31, 40, 41, 50, 60]
    input_ids = torch.tensor([merged], dtype=torch.long)
    pos_ids = torch.arange(len(merged), dtype=torch.long).unsqueeze(0)
    model = _RecordingModel()

    captured_ranges: list[tuple[int, int]] = []

    def fake_loss(seg_logits, start, end):
        captured_ranges.append((start, end))
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=2,
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 2
    # C0: positions [0, 1, 4, 5, 6, 7] (eviction gap at 2-3).
    assert model.calls[0]["position_ids"] == [0, 1, 4, 5, 6, 7]
    assert model.calls[0]["pre_cache_len"] == 0
    # C1: continues at position 8 with offset preserved.
    assert model.calls[1]["seq_len"] == 2
    assert model.calls[1]["position_ids"] == [8, 9]
    # Cache from C0 has length 6.
    assert model.calls[1]["pre_cache_len"] == 6
    assert captured_ranges == [(0, 6), (6, 8)]


def test_per_call_admission_with_midgen_aggregator_ignores_midgen():
    """When a call has BOTH an admission event AND a mid-gen event,
    _aggregate_admission_descriptor only considers the admission ones."""
    from kv_eviction.segmented_forward import _aggregate_admission_descriptor

    admission_event = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 0,
            "evict_start": 4,
            "tokens_evicted": 16,
            "new_user_fragment_len": 8,
        },
    )()
    midgen_event = type(
        "E", (),
        {
            "num_output_tokens_at_compaction": 32,
            "evict_start": 4,
            "tokens_evicted": 8,
            "new_user_fragment_len": 0,
        },
    )()
    call = _CallSpec(
        submitted_prompt_ids=[0] * 100,
        completion_ids=[1] * 50,
        compaction_events=[admission_event, midgen_event],
    )
    desc = _aggregate_admission_descriptor(call)
    assert desc["evict_start"] == 4
    assert desc["total_evicted"] == 16
    assert desc["new_user_fragment_len"] == 8


def test_per_call_owned_logits_match_position_ids():
    """For a no-admission rollout, owned logits for each call should
    equal their merged-frame position ids (mock model returns
    logits = position_ids broadcast)."""
    calls = [
        _CallSpec(submitted_prompt_ids=[5, 6], completion_ids=[7]),
        _CallSpec(submitted_prompt_ids=[5, 6, 7, 8], completion_ids=[9]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()

    captured: list[torch.Tensor] = []

    def fake_loss(seg_logits, start, end):
        captured.append(seg_logits.detach().clone())
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=2,
        device=torch.device("cpu"),
    )

    # Call 0: new tokens [0, 3) → 3 positions [0, 1, 2].
    # Call 1: new tokens [3, 5) → 2 positions [3, 4].
    assert captured[0].shape == (1, 3, 32)
    for i in range(3):
        assert torch.all(captured[0][0, i] == float(i)).item()
    assert captured[1].shape == (1, 2, 32)
    for i in range(2):
        assert torch.all(captured[1][0, i] == float(3 + i)).item()
