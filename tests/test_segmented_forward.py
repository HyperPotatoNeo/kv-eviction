# SPDX-License-Identifier: Apache-2.0
"""Unit tests for segmented_forward without a real model.

Uses a mock HF-like model that records every (input_ids, position_ids,
past_key_values) triple it's called with, so we can verify segment slicing
and the KV drop logic without needing GPU inference.

The REAL correctness check (logit match vs vllm inference) lives in Phase
3.4's live KL test on Qwen3-4B. These tests only validate the bookkeeping:
segment ranges, drop offsets, retained KV identities.
"""

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from kv_eviction.segmented_forward import segmented_forward


@dataclass
class MockConfig:
    use_cache: bool = False


@dataclass
class _MockOutput:
    logits: torch.Tensor
    past_key_values: DynamicCache


class _Backbone(torch.nn.Module):
    """Tiny backbone that fabricates KV entries from positional embeddings.

    Each forward pass:
    1. Extends the passed-in DynamicCache by `seq_len` new entries whose
       per-layer K/V tensors encode the positional index (so we can
       trace which original positions survived).
    2. Returns a _MockOutput with the updated cache and dummy hidden states.
    """

    def __init__(self, num_layers: int = 2, num_heads: int = 1, head_dim: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=True):
        assert input_ids.shape[0] == 1, "batch_size=1 only"
        seq_len = input_ids.shape[1]
        if past_key_values is None:
            past_key_values = DynamicCache()
        # Fabricate new K/V: each position encoded as a constant tensor
        # equal to the position_id. Shape [1, heads, seq, dim].
        pos = position_ids[0]  # [seq]
        for l in range(self.num_layers):
            new_K = pos.view(1, 1, seq_len, 1).expand(
                1, self.num_heads, seq_len, self.head_dim
            ).float()
            new_V = new_K.clone()
            past_key_values.update(new_K, new_V, l)
        # Dummy hidden states, ignored by segmented_forward
        hidden = torch.zeros(1, seq_len, 8)
        return _MockOutput(logits=hidden, past_key_values=past_key_values)


class MockModel(torch.nn.Module):
    """Mock HF CausalLM that returns per-position logits and records calls."""

    def __init__(self, vocab_size: int = 100, num_layers: int = 2):
        super().__init__()
        self.config = MockConfig()
        self.model = _Backbone(num_layers=num_layers)
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=True):
        # Record past_kv_len BEFORE the backbone extends the cache, otherwise
        # we'd measure the post-extension size and see `prev_len + seq_len`.
        pre_past_kv_len = (
            past_key_values.layers[0].keys.shape[2]
            if (past_key_values is not None
                and hasattr(past_key_values, "layers")
                and past_key_values.layers)
            else 0
        )
        # Delegate to backbone so the hook captures past_key_values from there.
        backbone_out = self.model(input_ids, position_ids, past_key_values, use_cache)
        # Per-position logits = position_id (broadcast to vocab). This lets
        # us verify the correct tokens ended up in the correct segments.
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, self.vocab_size)
        for i in range(seq_len):
            logits[0, i, :] = float(position_ids[0, i].item())
        self.calls.append({
            "seq_len": seq_len,
            "position_ids": position_ids[0].tolist(),
            "had_past_kv": past_key_values is not None,
            "past_kv_len": pre_past_kv_len,
        })
        return {"logits": logits, "past_key_values": backbone_out.past_key_values}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_boundary_basic():
    """prompt_len=10, boundaries=[20], completion has 30 tokens, seq_len=40.

    Expected segments (my convention):
    - Seg 0: [0, 10+20) = [0, 30): prompt + gen[0..19]
    - Tail:  [10+20-1, 40) = [29, 40): gen[19..29] (with overlap)
    """
    model = MockModel()
    input_ids = torch.arange(40).unsqueeze(0)  # [1, 40]
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 40)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    assert out["logits"].shape == (1, 40, model.vocab_size)

    # Two forward calls: segment 0 and tail.
    assert len(model.calls) == 2, f"Expected 2 calls, got {len(model.calls)}"
    assert model.calls[0]["seq_len"] == 30  # prompt_len + boundary
    assert model.calls[0]["had_past_kv"] is False
    assert model.calls[1]["seq_len"] == 11  # 40 - (10 + 20 - 1) = 11
    assert model.calls[1]["had_past_kv"] is True
    # After segment 0 + eviction, past_kv should be shorter:
    # 30 original - 8 stride - 1 boundary = 21 retained
    assert model.calls[1]["past_kv_len"] == 21, (
        f"Expected 21 retained KV entries, got {model.calls[1]['past_kv_len']}"
    )


def test_multiple_boundaries():
    """prompt_len=10, boundaries=[20, 40, 60], completion_len=70, seq_len=80.

    Expected segments:
    - Seg 0: [0, 30): prompt + gen[0..19]
    - Seg 1: [29, 50): gen[19..39] (with overlap)
    - Seg 2: [49, 70): gen[39..59] (with overlap)
    - Tail:  [69, 80): gen[59..69] (with overlap)
    """
    model = MockModel()
    input_ids = torch.arange(80).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 80)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20, 40, 60],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    assert len(model.calls) == 4, f"Expected 4 calls, got {len(model.calls)}"
    assert [c["seq_len"] for c in model.calls] == [30, 21, 21, 11], (
        f"Segment lengths: {[c['seq_len'] for c in model.calls]}"
    )
    # past_kv lengths after each eviction:
    # After seg 0: 30 - 8 - 1 = 21. Seg 1 feeds 21 entries.
    # After seg 1: (21 + 21) - 8 - 1 = 33. Seg 2 feeds 33 entries.
    # After seg 2: (33 + 21) - 8 - 1 = 45. Tail feeds 45 entries.
    assert model.calls[1]["past_kv_len"] == 21
    assert model.calls[2]["past_kv_len"] == 33
    assert model.calls[3]["past_kv_len"] == 45


def test_boundary_exactly_at_completion_end():
    """Last compaction fires at the very last token. No tail segment."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],  # boundary at completion_len
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    # Only 1 segment: the first. No tail because last_covered == seq_len.
    assert len(model.calls) == 1, f"Expected 1 call, got {len(model.calls)}"
    assert model.calls[0]["seq_len"] == 30


def test_prompt_aligned_len_differs_from_prompt_len():
    """prompt_len=50, prompt_aligned_len=64, boundaries=[30], stride=16.

    The drop should start at position 64 (NOT 50): the 14 gen tokens that
    sit in the tail of the last prompt block (positions 50..63) must be
    retained through every eviction.
    """
    model = MockModel()
    input_ids = torch.arange(200).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 200)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[30],
        prompt_len=50,
        prompt_aligned_len=64,
        stride=16,
        temperature=temperature,
    )
    # Segment 0: [0, 80), past_kv_len=0
    # Tail: [79, 200), past_kv fed
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 80  # prompt_len + boundary[0]
    assert model.calls[1]["seq_len"] == 121  # 200 - 79

    # After eviction: kv had 80 entries. Drop [64, 64+16) = [64, 80) = 16
    # entries. Also drop boundary token at position 79, BUT position 79
    # is already in the stride drop range, so trim doesn't remove anything
    # extra. Retained: [0, 64), which is 64 entries.
    #
    # Wait: in my code, keys[prompt_aligned + stride : -trim] is [80 : -1]
    # = [80 : 79] which is empty. So retained = keys[:64] = 64 entries.
    assert model.calls[1]["past_kv_len"] == 64, (
        f"Expected 64 retained KV entries (prompt_aligned only), got "
        f"{model.calls[1]['past_kv_len']}"
    )


def test_short_assistant_content_clamped_stride():
    """Stride larger than available asst content. actual_stride = asst_len."""
    model = MockModel()
    input_ids = torch.arange(40).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 40)

    # Segment 0 processes [0, 15). kv_seq_len=15. prompt_aligned=10.
    # asst_len = 5. stride=100 -> actual_stride = min(100, 5) = 5.
    # Retained: keys[:10] + keys[10+5:-1] = keys[:10] + keys[15:14] (empty)
    # = 10 entries.
    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[5],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=100,
        temperature=temperature,
    )
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 15
    assert model.calls[1]["past_kv_len"] == 10


def test_fsdp_padding_runs_dummy_passes():
    """max_forward_passes > actual causes dummy forwards to keep FSDP sync."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
        max_forward_passes=5,  # 1 actual + 4 dummy
    )
    # 1 real segment (completion_len==boundary, no tail) + 4 dummies = 5 calls.
    assert len(model.calls) == 5, f"Expected 5 calls, got {len(model.calls)}"
    # The 4 dummy passes are 1-token forward passes on input_ids[:, :1].
    dummy_calls = model.calls[1:]
    for c in dummy_calls:
        assert c["seq_len"] == 1
    # Output shape unchanged.
    assert out["logits"].shape == (1, 30, model.vocab_size)


def test_empty_boundaries_raises():
    """Caller must dispatch to standard forward on empty boundaries."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)

    import pytest
    with pytest.raises(AssertionError, match="must be non-empty"):
        segmented_forward(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            segment_boundaries=[],
            prompt_len=10,
            prompt_aligned_len=10,
            stride=8,
            temperature=temperature,
        )


def test_asst_len_equals_stride_fully_aligned():
    """Regression test for the canonical 'needs_compaction just fired on a
    fully-filled evict block' case: after segment 0 runs, the post-prompt
    KV length equals stride exactly. Verify:
    1. The retained KV is prompt_aligned_len entries (prompt block only).
    2. The eviction drops the boundary token (index kv_seq_len - 1).
    3. The tail segment re-feeds the boundary token under post-eviction
       context.
    (This case was flagged as a potential off-by-one by RSA review R2; the
    tensor is correct but the log-message formula was off by 1 in this
    exact edge case, now fixed.)
    """
    model = MockModel()
    # prompt_len == prompt_aligned_len == 64 (block-aligned prompt), stride=16.
    # boundary=16 means seg 0 processes input_ids[0:80], kv_seq_len=80,
    # asst_len=16=stride.
    input_ids = torch.arange(120).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 120)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[16],
        prompt_len=64,
        prompt_aligned_len=64,
        stride=16,
        temperature=temperature,
    )
    # 2 calls: seg 0 and tail.
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 80
    # After eviction, retained KV = prompt_aligned_len (64) entries. The
    # boundary token (index 79) is dropped via the stride range [64, 80).
    # Trim -1 overlaps with the stride range but doesn't remove anything
    # extra. Accounting previously over-subtracted; now fixed.
    assert model.calls[1]["past_kv_len"] == 64, (
        f"Expected 64 retained KV entries (prompt_aligned only), got "
        f"{model.calls[1]['past_kv_len']}"
    )
    # Tail starts at index prompt_len + boundary - 1 = 79 (the boundary
    # token, re-fed under post-eviction context).
    # seq_len - 79 = 41 tokens in the tail segment.
    assert model.calls[1]["seq_len"] == 41


def test_asst_len_equals_stride_large_stride():
    """Same as above but with stride == 64 blocks to exercise R2's exact
    scenario (pa_len=64, stride=64). Confirms the fix works for any
    stride size."""
    model = MockModel()
    input_ids = torch.arange(200).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 200)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[64],
        prompt_len=64,
        prompt_aligned_len=64,
        stride=64,
        temperature=temperature,
    )
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 128  # prompt_len + boundary
    # asst_len=64=stride. Retained = prompt_aligned_len = 64 entries.
    assert model.calls[1]["past_kv_len"] == 64


def test_temperature_scaling_applied():
    """Per-token temperature scales logits."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    # temperature=2.0 everywhere -> logits halved
    temperature = torch.full((1, 30), 2.0)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    # Our MockModel returns logits[i, :] = position_id[i]. After scaling by
    # 2.0, logits[i, :] = position_id[i] / 2.
    # Position 5 should yield 5.0 / 2.0 = 2.5.
    assert out["logits"][0, 5, 0].item() == 2.5
    assert out["logits"][0, 29, 0].item() == 14.5
