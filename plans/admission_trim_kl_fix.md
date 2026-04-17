# Fix: Admission-Time Compaction KL Mismatch

## The Problem in Plain English

After we fixed the admission-loop bug (compaction now fires fully before
prefill), the KL divergence between trainer and inference is still high.
It should be ~0.0009 (kernel floor — the tiny numerical difference
between two identical flash_attention_2 forward passes). Instead it's
much larger.

**Why:** The trainer is doing a forward pass on completely different
tokens with different RoPE positions than what inference actually ran on.
The segmented_forward KV eviction that's supposed to replay the
inference-time compaction is a no-op due to cascading bugs.

## How Tokens Flow Today (Broken)

### Inference Side

```
1. Orchestrator renders padded prompt for Turn 2:
   [system(100) PAD(12) user1(50) PAD(6) asst1(180) PAD(12) user2(40) PAD(8) <|im_start|>assistant\n]
   = 408 tokens (all block-aligned at <|im_end|> boundaries)

2. Padding interceptor sends these 408 tokens to vLLM via:
   extra_body={"prompt_token_ids": [408 padded tokens]}

3. vLLM scheduler receives the request.
   _maybe_compact_prompt fires (max_turns=4, 6 live turns > 4).

4. Loop iteration 1: evict oldest 2 turns
   evict_start=112 (block-aligned end of system msg)
   evict_end=256  (block-aligned end of 2nd turn)
   total_evicted=144
   Trims request.prompt_token_ids: del [112:256]
   request.position_offset += 144   → 144
   Prompt is now 264 tokens.

5. Loop iteration 2: re-scan, still > max_turns
   evict_start=112 (same — always after system prefix)
   evict_end=240
   total_evicted=128
   Trims again: del [112:240]
   request.position_offset += 128   → 272
   Prompt is now 136 tokens.

6. Loop exits: live_turns < max_turns.

7. Prefill runs on the 136-token trimmed prompt.
   CRITICAL: position_offset is NOT propagated to the model runner
   for new requests (NewRequestData lacks the field).
   So RoPE positions = [0, 1, 2, ..., 135].
   The system prompt gets [0..111], remaining turns get [112..135].

8. Model generates Turn 2's assistant tokens starting at position 136.
```

### Response Side

```
9. OutputProcessor was created BEFORE the engine core trimmed the prompt
   (it's in the API process, the trim happens in the engine core process).
   So response.prompt_token_ids = ORIGINAL 408 tokens (pre-trim).

10. Compaction events are serialized into the response:
    Event 1: {output=0, tokens_evicted=144, position_offset_after=144}
    Event 2: {output=0, tokens_evicted=128, position_offset_after=272}
```

### Orchestrator / Trainer Side (Current — Broken)

```
11. env.py extracts prompt_token_ids from response → 408 original tokens.
    Stores in step.extras["prompt_token_ids"].

12. trajectories.py reads extras["prompt_token_ids"] → prompt_ids = 408 tokens.
    Multi-turn merge: Turn 1 prompt + completion is a prefix of Turn 2's
    408-token prompt → extension property holds → turns merge.

13. Merged sample:
    input_ids = [Turn1_prompt(160) | Turn1_gen(200) | Turn2_ext(248) | Turn2_gen(150)]
    = 758 tokens total.
    prompt_len = 160 (Turn 1's padded prompt).
    position_ids = [0, 1, 2, ..., 757]  ← plain arange, NO offset.

14. Trainer computes prompt_aligned_len:
    pp = first_evt.position_offset_after - first_evt.tokens_evicted
       = 144 - 144 = 0
    Since pp=0: effective_prompt_len = mb_prompt_len = 160
    prompt_aligned_len = ceil(160/16)*16 = 160

15. segment_boundaries = [0+offset, 0+offset] (admission events at output=0,
    offset by generation_offset in the merge).

16. segmented_forward processes segment 0 (all tokens up to boundary).
    Between segments, tries to evict:
    asst_len = kv_seq_len - prompt_aligned_len = kv_seq_len - 160
    actual_stride = min(256, asst_len)

    But for admission events with boundary near start of Turn 2's gen,
    asst_len might be small or the eviction targets the WRONG tokens
    (Turn 1's generation, not old conversation turns).

17. Result: trainer forward pass sees 408 prompt tokens + all gen tokens.
    Inference saw 136 prompt tokens + gen tokens.
    RoPE positions differ everywhere after the system prompt.
    KL is high.
```

### Why Each Bug Matters

| Bug | What Goes Wrong | Impact on KL |
|-----|----------------|--------------|
| Response returns original 408 tokens | Trainer has 272 extra tokens inference never saw | Attention context completely different |
| `pp=0` → `prompt_aligned_len=full_prompt` | KV eviction starts after the full prompt, not after system prefix | No useful tokens get evicted |
| `stride=256` fixed | Even if eviction fired, wrong amount removed (inference evicted 144 then 128) | Wrong retained KV set |
| `position_ids = arange(758)` | Remaining turns at positions [528..] in trainer vs [112..] in inference | RoPE embeddings differ at every generation position |

## How Tokens Should Flow (Fixed)

### Key Insight

Since ALL compaction events are admission-time (`output=0`), the
eviction is already baked into the prompt. Inference runs on the trimmed
prompt. The trainer should too. No KV eviction replay needed — just use
the same trimmed tokens.

### Second Key Insight

`position_offset` is NOT propagated for new requests in vLLM (the
`NewRequestData` struct lacks the field, `CachedRequestState` defaults
to 0). So inference uses `position_ids = [0, 1, ..., len(trimmed)-1]`.
The trainer can use plain `arange` and it matches.

### Fixed Flow

```
Steps 1-10: Same as before (inference + response unchanged).

11. Orchestrator gets response with:
    - prompt_token_ids = 408 original tokens
    - compaction_events = [
        {output=0, evicted=144, offset_after=144, evict_start=112},
        {output=0, evicted=128, offset_after=272, evict_start=112},
      ]

12. NEW: apply_admission_trim(prompt_ids, events):
    - Event 1 (output=0): del prompt_ids[112 : 112+144]
      → 408-144 = 264 tokens
    - Event 2 (output=0): del prompt_ids[112 : 112+128]
      → 264-128 = 136 tokens
    - Strip both events (admission events consumed).
    - remaining_events = []  (no mid-gen events)

13. Step extras now has:
    prompt_token_ids = 136 trimmed tokens (matches inference)
    compaction_events = []  (empty — admission already applied)

14. Multi-turn merge: Turn 2's trimmed prompt does NOT start with
    Turn 1's (prompt + completion) prefix → extension check fails.
    Turn 2 becomes its own TrainingSample. This is fine — the
    existing fallback path handles separate samples correctly
    (same reward/advantage from the rollout).

15. Turn 2's sample:
    input_ids = [136 trimmed prompt tokens | 150 gen tokens] = 286 tokens
    position_ids = [0, 1, ..., 285]  ← plain arange
    compaction_events = None  (empty)
    prompt_len = 136

16. Trainer sees no compaction events → standard forward path
    (D5 unified dispatch: single-segment forward, no eviction).

17. Forward pass: same 136 tokens, same positions [0..285],
    same flash_attention_2 kernel as inference.
    KL = kernel floor (~0.0009).
```

## Changes Required

### 1. Add `evict_start` to CompactionEvent wire format

The scheduler already knows `evict_start` (it computes it as
`align_up(turn_first_start_pos, block_size)`). We just need to carry
it through the wire.

**Why:** The orchestrator needs to know WHERE in the original prompt
to delete tokens. Without `evict_start`, it can't reconstruct the trim
(the formula `position_offset_after - tokens_evicted` gives 0, not the
actual start position).

#### Diffs

**`vllm/vllm/v1/core/compaction/types.py`** — add field to dataclass:
```python
 @dataclass
 class CompactionEvent:
     num_output_tokens_at_compaction: int
     tokens_evicted: int
     position_offset_after: int
     num_prompt_tokens: int = 0
+    evict_start: int = 0
     evicted_token_ids: list[int] = field(default_factory=list)
     last_turn_evicted: int = -1
     num_turns_evicted_after: int = 0
```

**`vllm/vllm/v1/core/sched/scheduler.py`** — set evict_start on events:

In `_maybe_compact_prompt` (line ~1595):
```python
 event = CompactionEvent(
     num_output_tokens_at_compaction=0,
     tokens_evicted=total_evicted,
     position_offset_after=request.position_offset + total_evicted,
     num_prompt_tokens=request.num_prompt_tokens,
+    evict_start=evict_start,
     evicted_token_ids=evicted_token_ids,
     ...
 )
```

In `_compact_request` (mid-gen path, line ~1270):
```python
 event = CompactionEvent(
     ...
+    evict_start=evict_start,
     ...
 )
```

**`vllm/vllm/entrypoints/openai/chat_completion/protocol.py`** — add to payload:
```python
 class CompactionEventPayload(OpenAIBaseModel):
     num_output_tokens_at_compaction: int
     tokens_evicted: int
     position_offset_after: int
     num_prompt_tokens: int = 0
+    evict_start: int = 0
```

**`vllm/vllm/entrypoints/openai/chat_completion/serving.py`** — serialize it:
```python
 CompactionEventPayload(
     num_output_tokens_at_compaction=e.num_output_tokens_at_compaction,
     tokens_evicted=e.tokens_evicted,
     position_offset_after=e.position_offset_after,
     num_prompt_tokens=e.num_prompt_tokens,
+    evict_start=e.evict_start,
 )
```

**`prime-rl/src/prime_rl/transport/types.py`** — add to wire struct:
```python
 class CompactionEventWire(msgspec.Struct, ...):
     num_output_tokens_at_compaction: int
     tokens_evicted: int
     position_offset_after: int
     num_prompt_tokens: int = 0
+    evict_start: int = 0
```

### 2. Apply admission trim in orchestrator

New function + call site in trajectories.py. This is the core fix.

**Why:** Instead of trying to replay KV eviction during training
(which fails due to bugs 1-4), we apply the same token deletion the
scheduler did, so the trainer sees exactly the trimmed prompt that
inference ran on.

#### Diff

**`prime-rl/src/prime_rl/orchestrator/trajectories.py`** — add function
and call it in `prepare_step_tokens`:

```python
def apply_admission_trim(
    prompt_ids: list[int],
    prompt_mask: list[bool],
    events: list[CompactionEventWire],
) -> tuple[list[int], list[bool], list[CompactionEventWire]]:
    """Apply admission-time compaction events to the prompt.

    Admission events (num_output_tokens_at_compaction == 0) represent
    token deletions that the vLLM scheduler applied to the prompt
    BEFORE prefill. The response carries the original (pre-trim)
    prompt, so we replay the deletions here so the trainer sees the
    same tokens inference ran on.

    Events are applied in order (oldest-first), matching the
    scheduler's _apply_trim sequence. Each event's evict_start is
    relative to the CURRENT (already-partially-trimmed) prompt.

    Returns (trimmed_prompt, trimmed_mask, remaining_events) where
    remaining_events contains only mid-generation events (output > 0).
    """
    trimmed_ids = list(prompt_ids)
    trimmed_mask = list(prompt_mask)
    remaining = []
    for evt in events:
        if evt.num_output_tokens_at_compaction == 0:
            start = evt.evict_start
            end = start + evt.tokens_evicted
            del trimmed_ids[start:end]
            del trimmed_mask[start:end]
        else:
            remaining.append(evt)
    return trimmed_ids, trimmed_mask, remaining
```

Call in `prepare_step_tokens` (after reading padded_ids from extras):

```python
 padded_ids = extras.get("prompt_token_ids") if extras else None
 if padded_ids:
     prompt_ids = [int(x) for x in padded_ids]
     prompt_mask = [False] * len(prompt_ids)
+
+# Apply admission-time trims so trainer sees the same tokens as inference.
+step_events = _compaction_events_from_step(step)
+if step_events and padded_ids:
+    prompt_ids, prompt_mask, step_events = apply_admission_trim(
+        prompt_ids, prompt_mask, step_events,
+    )
+    # Stash trimmed events back (only mid-gen events survive).
+    # The caller reads step_compaction_events separately, so we
+    # need to update the step's extras too.
+    _set_step_compaction_events(step, step_events)
```

(Exact integration depends on how `step_compaction_events` is populated
relative to `prepare_step_tokens` — may need a small refactor of the
loop in `build_training_samples`.)

### 3. No other changes needed

- **No `batch.py` changes**: `position_ids = arange(len(input_ids))` is
  correct because inference also uses offset=0 for new requests.

- **No `segmented_forward.py` changes**: With admission events stripped,
  `compaction_events` is empty → D5 unified dispatch does a
  single-segment forward (no eviction). Numerically identical to a plain
  forward.

- **No `train.py` changes**: The existing code already handles empty
  compaction_events correctly (single-segment path).

- **No `env.py` changes**: The interceptor still sends padded prompts to
  vLLM. The response still carries original prompt_token_ids. The trim
  happens downstream in trajectories.py.

### 4. Multi-turn merge behavior change

After trimming, Turn N+1's prompt no longer starts with Turn N's
`(prompt + completion)` prefix (the old turns were removed). The prefix
extension check at `trajectories.py:579` will fail, and each compacted
turn becomes its own `TrainingSample`.

**This is acceptable** because:
- The existing fallback for non-matching prefixes already works
  (samples get the same reward/advantage from the rollout)
- Each turn's training signal is still correct (trimmed prompt matches
  inference)
- The alternative (merging across compaction boundaries) would require
  complex position-offset handling with no KL benefit

## Verification Plan

### 1. Unit test: apply_admission_trim

```python
def test_admission_trim():
    prompt = list(range(100))  # [0, 1, ..., 99]
    mask = [False] * 100
    events = [
        CompactionEventWire(output=0, evicted=20, offset=20, prompt=100, evict_start=30),
        CompactionEventWire(output=0, evicted=15, offset=35, prompt=80, evict_start=30),
    ]
    trimmed, tmask, remaining = apply_admission_trim(prompt, mask, events)
    assert len(trimmed) == 65  # 100 - 20 - 15
    assert trimmed[:30] == list(range(30))  # prefix preserved
    assert trimmed[30] == 50  # first token after both trims
    assert remaining == []  # all admission events consumed
```

### 2. Smoke test: KL should drop

```bash
uv run rl @ experiments/debug_balrog/rl.toml
```

Check wandb metrics:
- `compaction/mismatch_kl` should be ~0.0009 (kernel floor)
- All compaction events should have `output_at_compact=0`

### 3. Log verification

```bash
# Should see trimmed prompts in orchestrator logs
grep "admission_trim" orchestrator.log

# Should see NO segmented_forward eviction
grep "KV eviction seg" trainer.log  # should be empty
```

## Future: Mid-Generation Compaction

This fix handles **admission-only** events. If future configs re-enable
mid-generation compaction (events with `output > 0`), those events will
pass through `apply_admission_trim` unchanged and flow into
`segmented_forward` as before. The existing stride/pp bugs in
segmented_forward would need separate fixes for that path:

1. Use `evt.tokens_evicted` instead of fixed `stride`
2. Use `evt.evict_start` to determine the correct protected prefix
3. Handle position_offset for the mid-gen case (where offset IS
   propagated via CachedRequestState updates)

But that's a separate task — the current config only produces
admission-time events.
