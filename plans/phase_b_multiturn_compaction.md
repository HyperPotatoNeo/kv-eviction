# Phase B: Multi-Turn Compaction Support for BabyAI

## Status: Implemented (protected-prefix simplification)

### Implementation summary

The protected-prefix approach (`compaction_protected_prefix_tokens`) makes the
eviction boundary constant across all compaction events:
`ceil(protected_prefix / block_size) * block_size`. This eliminates the need for
per-event eviction boundaries in `segmented_forward` (Change 4 from original plan
is NOT needed).

Changes made:
1. `CompactionEventWire` gains `num_prompt_tokens: int = 0` (wire-compatible default)
2. `_compaction_events_from_step` propagates `num_prompt_tokens` through all forms
3. `interleave_rollout` replaces `NotImplementedError` with offset-and-merge logic
4. `CompactionConfig` gains `protected_prefix_tokens: int = 0`
5. Trainer uses `min(protected_prefix, prompt_len)` for `prompt_aligned_len`
6. `env.py` propagates `num_prompt_tokens` in all extraction and construction paths

---

## How Compaction Works Today (Single-Turn)

### End-to-End Flow Diagram

```
                         INFERENCE (vLLM)
                         ================

  LLM generates tokens one at a time. After each token, the scheduler
  checks if total KV exceeds the window:

  KV cache state during generation of a SINGLE request:

  Token 0    [PPPPPPPP....................]   P = prompt tokens
  Token 100  [PPPPPPPP|GGGGGGGGGGGG........]  G = generated tokens
  Token 500  [PPPPPPPP|GGGGGGGGGGGGGGGGGGGG]  Exceeds window!
                       ^
                       prompt_aligned_len (block-aligned prompt boundary)

  Compaction fires! Drop `stride` tokens starting at prompt_aligned_len:

  Before:    [PPPPPPPP|EEEE|GGGGGGGGGGGGGGG]  E = tokens to evict
  After:     [PPPPPPPP|GGGGGGGGGGGGGGG......]  Spliced out, continue generating

  A CompactionEvent is recorded:
    { num_output_tokens_at_compaction: 500,   <-- how many tokens generated so far
      tokens_evicted: 512,                    <-- stride (block-aligned)
      position_offset_after: 512 }            <-- cumulative offset for RoPE


                    EVENT PLUMBING (verifiers + kv_eviction)
                    ========================================

  vLLM scheduler                        env.py monkey-patches
  ~~~~~~~~~~~~~~~                        ~~~~~~~~~~~~~~~~~~~~
  CompactionEvent                        Patch 1: from_native_response
  created at                             copies events from openai
  scheduler.py:1078                      ChatCompletion to verifiers
       |                                 Response object (env.py:218)
       v                                       |
  Attached to Request                          v
  (scheduler.py:1083)                    Patch 2: add_model_response
       |                                 copies events from Response
       v                                 to trajectory step extras
  EngineCoreOutput                       as JSON dicts (env.py:251)
  (scheduler.py:1687)                          |
       |                                       v
       v                                 step["extras"]["compaction_events"]
  output_processor.py:638                = [{num_output_tokens_at_compaction: 500,
  -> RequestOutput                          tokens_evicted: 512,
  -> ChatCompletionResponse                 position_offset_after: 512}]
  (with compaction_events field)


                    ORCHESTRATOR (prime-rl)
                    ======================

  trajectories.py: interleave_rollout()

  Step 0 (only step for single-turn):
    _compaction_events_from_step(step)     # trajectories.py:297
    -> [CompactionEventWire(500, 512, 512)]
    -> make_sample(tokens, events)         # trajectories.py:412
    -> TrainingSample.compaction_events = [CompactionEventWire(...)]


                    TRAINER (prime-rl + kv_eviction)
                    ================================

  train.py: rl_step()

  1. Read events from MicroBatch           # train.py:494
     compaction_events = micro_batch.compaction_events

  2. Compute segment boundaries            # train.py:582-584
     segment_boundaries = [e.num_output_tokens_at_compaction
                           for e in compaction_events]
     # e.g. [500] if one compaction fired

  3. Compute prompt_aligned_len            # train.py:580-581
     prompt_aligned_len = ceil(prompt_len / block_size) * block_size

  4. Call segmented_forward()              # train.py:780-791


                    SEGMENTED FORWARD (kv_eviction)
                    ================================

  segmented_forward.py: segmented_forward()

  input_ids = [PPPPPPPP|GGGGGGGGGGGG|GGGGGGGGG]
               prompt   seg 0 gen    seg 1 gen (tail)
                        ^            ^
                        |            boundary[0] = 500
                        prompt_len

  Segment ranges (segmented_forward.py:399-426):
    seg_ranges = [(0, prompt_len + 500),           # seg 0: full prompt + gen until compaction
                  (prompt_len + 500 - 1, seq_len)] # seg 1: overlap token + tail
                  # The -1 is the boundary-token overlap

  Processing loop:
    Seg 0: forward(input_ids[0 : prompt_len+500])
           -> KV cache built for full seg 0
           -> KV eviction: drop [prompt_aligned_len, prompt_aligned_len + stride)
              (segmented_forward.py:617-658)

              Before: KV = [PPPPPPPP|EEEE|GGGGGGGG]
              After:  KV = [PPPPPPPP|GGGGGGGG]
                            ^        ^
                            kept     kept (after splice)

    Seg 1: forward(overlap_token + tail_tokens, past_key_values=evicted_KV)
           -> Logits computed under post-eviction context
           -> Per-segment backward fires (loss_fn callback)

  Result: training replay EXACTLY matches inference KV state at every point.
          Mismatch KL should be at the kernel floor (~0.0009).
```

### Key Files Reference

| Step | File | Lines | What happens |
|------|------|-------|-------------|
| Trigger check | `vllm/v1/core/compaction/manager.py` | 66-105 | `needs_compaction()`: fires when `num_computed_tokens > window_size` |
| Event creation | `vllm/v1/core/sched/scheduler.py` | 1078-1083 | `CompactionEvent(num_output_tokens_at_compaction=request.num_total_generated, ...)` |
| Block splice | `vllm/v1/core/compaction/manager.py` | evict() | Physical block removal from KV cache |
| Output plumbing | `vllm/v1/engine/output_processor.py` | 637-663 | Events flow: EngineCoreOutput -> RequestOutput -> ChatCompletionResponse |
| Patch 1 (client) | `src/kv_eviction/env.py` | 218-241 | Copy events from openai ChatCompletion to verifiers Response |
| Patch 2 (env) | `src/kv_eviction/env.py` | 251-255 | Copy events from Response to trajectory step extras |
| Step extraction | `prime-rl/.../trajectories.py` | 297-333 | `_compaction_events_from_step()`: dict -> CompactionEventWire |
| Sample creation | `prime-rl/.../trajectories.py` | 412 | `make_sample(tokens, step_compaction_events[0])` |
| **Single-turn guard** | `prime-rl/.../trajectories.py` | **435-441** | **`NotImplementedError` if step > 0 has events** |
| Event clamping | `prime-rl/.../batch.py` | 6-27 | Drop events beyond truncation |
| Boundary extract | `prime-rl/.../train.py` | 582-584 | `segment_boundaries = [e.num_output_tokens_at_compaction for e in events]` |
| prompt_aligned | `prime-rl/.../train.py` | 580-581 | `ceil(prompt_len / block_size) * block_size` |
| Forward call | `prime-rl/.../train.py` | 780-791 | `segmented_forward(model, input_ids, segment_boundaries, prompt_len, ...)` |
| Seg ranges | `src/kv_eviction/segmented_forward.py` | 399-435 | Build `[(start, end), ...]` from boundaries |
| KV eviction | `src/kv_eviction/segmented_forward.py` | 617-658 | Drop `[prompt_aligned_len, prompt_aligned_len + stride)` |
| Backward | `src/kv_eviction/segmented_forward.py` | 555-570 | Per-segment loss_fn callback, BPTT window |

---

## What Breaks With Multi-Turn

In a multi-turn env like BabyAI, each turn is a **separate vLLM request**. The LLM
generates a short action (~20 tokens), the env responds with an observation, and the
next turn sends the full conversation history as its prompt.

### The multi-turn timeline

```
  TURN 0:
  vLLM request prompt = [system + obs_0]  (say 200 tokens)
  Generate action_0 (~20 tokens)
  Total KV = 220. Window = 4096. No compaction.
  CompactionEvents: none

  TURN 1:
  vLLM request prompt = [system + obs_0 + action_0 + obs_1]  (say 400 tokens)
  Generate action_1 (~20 tokens)
  Total KV = 420. Window = 4096. No compaction.
  CompactionEvents: none

  ... many turns pass, context grows ...

  TURN 15:
  vLLM request prompt = [system + obs_0 + ... + obs_15]  (say 4200 tokens)
  Generate action_15 (~20 tokens)
  Total KV = 4220 > 4096. COMPACTION FIRES!

  CompactionEvent:
    num_output_tokens_at_compaction = 20   <-- relative to THIS turn's generation
    tokens_evicted = 512
    position_offset_after = 512

  Key fact: num_output_tokens_at_compaction = 20, NOT 4220.
  Because request.num_total_generated starts at 0 for each new request.
```

### Problem 1: The `NotImplementedError` + wrong offset

To understand this, you need to see what the data looks like at each step.

#### What verifiers gives us per turn

Each turn of a multi-turn rollout produces a trajectory step with `prompt_ids` and
`completion_ids`. The prompt grows each turn because it includes the full conversation.

**File:** verifiers produces these in `MultiTurnEnv.rollout()` (the TITO client
tokenizes each turn). prime-rl reads them at `trajectories.py:282-292`.

```
Concrete 3-turn BabyAI example (small numbers for clarity):

TURN 0 (trajectory step 0):
  vLLM request: "system prompt + first observation"
  prompt_ids  = [S S S S O O O O]           (8 tokens: system + obs_0)
  completion_ids = [A A]                     (2 tokens: action_0)
  compaction_events = None                   (total KV=10, window=30, no compaction)

TURN 1 (trajectory step 1):
  vLLM request: "system + obs_0 + action_0 + tool_response_1"
  prompt_ids  = [S S S S O O O O A A T T T]  (13 tokens: all prior context + obs_1)
  completion_ids = [A A]                      (2 tokens: action_1)
  compaction_events = None                    (total KV=15, window=30, no compaction)

TURN 2 (trajectory step 2):
  vLLM request: "system + obs_0 + action_0 + obs_1 + action_1 + obs_2"
  prompt_ids  = [S S S S O O O O A A T T T A A T T T T T T T T T T T T T T T]  (30 tokens)
  completion_ids = [A A]                      (2 tokens: action_2)
  compaction_events = [{num_output_tokens_at_compaction: 2, ...}]
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       This means: "compaction fired 2 tokens into THIS turn's generation"
                       NOT "2 tokens into the full conversation"
                       Because vLLM starts request.num_total_generated = 0 for each new request.
                       See: vllm/vllm/v1/core/sched/scheduler.py:1079
```

#### How `interleave_rollout` merges steps

**File:** `prime-rl/src/prime_rl/orchestrator/trajectories.py:406-454`

The function walks through the steps and checks if each step's `prompt_ids` starts
with the previous step's `prompt_ids + completion_ids` (the "extension property").
If so, it merges the new step into the existing sample.

```
Step 0 (line 409-412):
  Create first sample:
    sample.prompt_ids     = [S S S S O O O O]
    sample.completion_ids = [A A]
  prefix = [S S S S O O O O A A]  (prompt + completion = 10 tokens)

Step 1 (lines 415-444):
  Check extension: step1.prompt_ids[:10] == prefix?
    [S S S S O O O O A A T T T][:10] == [S S S S O O O O A A] ?  YES!
  matched_idx = 0 (extends the existing sample)

  >>> Line 435: if step_compaction_events[1] is not None:  -> None, OK, no crash <<<

  extend_sample (line 442, calls function at lines 373-404):
    new_prompt_ext = step1.prompt_ids[10:]  = [T T T]  (3 tokens: env response)
    sample.completion_ids = [A A] + [T T T] + [A A]
                                    ^^^^^^^   ^^^^^
                                    new prompt  new completion
                                    (mask=False) (mask=True)

  After step 1, sample looks like:
    sample.prompt_ids     = [S S S S O O O O]           (unchanged, still 8 tokens)
    sample.completion_ids = [A A T T T A A]             (7 tokens: turn0 + env + turn1)
    prefix = [S S S S O O O O A A T T T A A]           (15 tokens)

Step 2 (lines 415-444):
  Check extension: step2.prompt_ids[:15] == prefix?  YES!
  matched_idx = 0

  >>> Line 435: if step_compaction_events[2] is not None:  -> NOT None! CRASH! <<<

  raise NotImplementedError(                              # line 436
      "Compaction events on extended trajectory step 2    # line 437
       are not supported yet..."                          # line 438-440
  )

  THE PIPELINE DIES HERE.
```

#### Why the event's value is "wrong" even if we remove the guard

If we just deleted the `NotImplementedError` and let the code continue, the event
would be attached to the sample with `num_output_tokens_at_compaction = 2`. But look
at what the merged sample looks like at this point:

```
  sample.prompt_ids     = [S S S S O O O O]          (8 tokens)
  sample.completion_ids = [A A T T T A A]             (7 tokens from steps 0+1)
                           ^^^^^^^^^^^^^
                           Turn 2's generation hasn't been appended yet.
                           It will be added by extend_sample AFTER this check.

  After extend_sample runs for step 2:
  sample.completion_ids = [A A T T T A A T T T T T T T T T T T T T T T T A A]
                           s0  env1  s1  env2 (15 tokens of env response)   s2

  The boundary value 2 means "2 tokens into turn 2's generation".
  In the merged completion_ids, turn 2's generation starts at position 22.
  So the correct absolute boundary should be 22 + 2 = 24, not 2.

  Later, the trainer (train.py:582) would compute:
    segment_boundaries = [2]   <-- WRONG, should be [24]

  And segmented_forward would build:
    seg_ranges = [(0, 8+2), (8+2-1, seq_len)]  = [(0, 10), (9, 32)]
                      ^^^ only covers the system prompt + action_0!

  Instead of the correct:
    seg_ranges = [(0, 8+24), (8+24-1, seq_len)] = [(0, 32), (31, 32)]
                      ^^^^ covers full conversation up to compaction point
```

#### The fix

**File:** `prime-rl/src/prime_rl/orchestrator/trajectories.py:427-444`

Replace the `NotImplementedError` block with code that computes the offset:

```python
# Lines 427-444, inside the `if matched_idx is not None:` block:

prefix_tokens, sample, _ = active_samples[matched_idx]
prefix_len = len(prefix_tokens)

# --- NEW: offset-and-merge compaction events from this step ---
step_events = step_compaction_events[step_idx]
if step_events is not None:
    # Where does this turn's generation start in sample.completion_ids?
    current_completion_len = len(sample.completion_ids)     # 7 in our example
    new_prompt_ext = tokens["prompt_ids"][prefix_len:]       # env response tokens
    generation_start = current_completion_len + len(new_prompt_ext)  # 7 + 15 = 22

    existing = sample.compaction_events or []
    for e in step_events:
        existing.append(CompactionEventWire(
            num_output_tokens_at_compaction=(
                e.num_output_tokens_at_compaction + generation_start  # 2 + 22 = 24
            ),
            tokens_evicted=e.tokens_evicted,
            position_offset_after=e.position_offset_after,
            num_prompt_tokens=e.num_prompt_tokens,  # NEW field, see Change 1
        ))
    sample.compaction_events = existing

# extend_sample as before (line 442)
extend_sample(sample, prefix_len, step_idx=step_idx)
```

### Problem 2: Eviction boundary mismatch in `segmented_forward`

This is the fundamental issue. Consider what happens during training:

```
  TRAINING: merged input_ids for the full conversation

  input_ids = [P0 P0 P0 | G0 | E1 E1 E1 | G1 | ... | E15 E15 | G15 G15 ...]
               step 0     s0    step 1      s1         step 15    s15
               prompt     comp  env resp    comp       env resp   comp

  prompt_len = len(P0) = 200 (step 0's prompt only)
  prompt_aligned_len = ceil(200 / 16) * 16 = 208

  Compaction event (after offset fix):
    segment_boundary = [generation_start_of_turn15 + 20]
    = some large number, say 4100

  segmented_forward builds:
    seg_ranges = [(0, 200 + 4100), (200 + 4100 - 1, seq_len)]

  Seg 0 forward: processes tokens 0..4300
    KV cache now has 4300 entries
    Eviction: drop [prompt_aligned_len, prompt_aligned_len + stride)
            = drop [208, 208 + 512)    <--- WRONG!

  But during INFERENCE, turn 15's compaction dropped from:
    turn15_prompt_aligned = ceil(4200 / 16) * 16 = 4208
    drop [4208, 4208 + 512)            <--- CORRECT for inference

  Training drops tokens 208-720 (early conversation tokens).
  Inference dropped tokens 4208-4720 (recent turn content).
  COMPLETELY DIFFERENT KV state -> large KL mismatch.
```

The root cause: `segmented_forward` uses a single fixed `prompt_aligned_len` for ALL
evictions. In single-turn this is correct (there's only one prompt boundary). In
multi-turn, each turn has a different prompt boundary because each is a fresh vLLM
request with a longer prompt.

---

## Implementation Plan (revised with protected-prefix simplification)

All changes are DONE. The `protected_prefix_tokens` approach eliminates the need for
per-event eviction boundaries in `segmented_forward` (original Change 4 is NOT needed).

### Change 1: CompactionEventWire gains num_prompt_tokens (DONE)

```
File: prime-rl/src/prime_rl/transport/types.py
  num_prompt_tokens: int = 0   # wire-compatible default via omit_defaults=True
```

Carried for diagnostics/validation. Not used by the trainer for eviction boundaries
when protected_prefix_tokens is set (the constant prefix subsumes per-event info).

### Change 2: Propagate num_prompt_tokens through all extraction paths (DONE)

```
Files:
  prime-rl/.../trajectories.py: _compaction_events_from_step  (dict, list/tuple branches)
  src/kv_eviction/env.py: _extract_compaction_event_dicts     (all 3 branches)
  src/kv_eviction/env.py: compaction_events_from_step_extras  (dict, list/tuple branches)
```

### Change 3: interleave_rollout offset-and-merge (DONE)

```
File: prime-rl/src/prime_rl/orchestrator/trajectories.py (lines 433-456)

Replaced NotImplementedError with:
  current_completion_len = len(sample.completion_ids)   # before extend
  new_prompt_ext_len = len(tokens["prompt_ids"]) - prefix_len
  generation_offset = current_completion_len + new_prompt_ext_len
  # offset each event and append to sample.compaction_events
```

Edge cases handled:
- Cascading compaction (multiple events per step): all offset, order preserved
- Step 0 events + later step events: step 0 via make_sample, later via merge
- Empty events on a step (None): no-op, extend_sample proceeds normally
- Zero-length completion: generation_offset = completion_len + prompt_ext_len (correct)

### Change 4: segmented_forward -- NOT NEEDED

With protected_prefix_tokens, the eviction boundary is constant for all events.
A single scalar prompt_aligned_len (unchanged API) is correct.

### Change 5: CompactionConfig + trainer (DONE)

```
File: prime-rl/src/prime_rl/configs/trainer.py
  protected_prefix_tokens: int = 0  (ge=0)

File: prime-rl/src/prime_rl/trainer/rl/train.py (lines 580-583)
  pp = config.compaction.protected_prefix_tokens
  effective_prompt_len = min(pp, mb_prompt_len) if pp > 0 else mb_prompt_len
  prompt_aligned_len = ceil(effective_prompt_len / bs) * bs
```

Edge cases:
- pp = 0: backward compat, uses prompt_len as before
- pp > prompt_len: clamped to prompt_len
- pp < prompt_len, pp > 0: uses pp (the intended multi-turn case)

### Change 6: env.py num_prompt_tokens propagation (DONE)

See Change 2 above. Both `_extract_compaction_event_dicts` and
`compaction_events_from_step_extras` propagate the field.

---

## Revised Analysis: Protected Prefix Eliminates Per-Event Boundary Tracking

The original plan (Changes 1-5 / Alt C) required per-event eviction boundaries in
`segmented_forward` because each multi-turn compaction event could have a different
prompt boundary (each turn is a longer vLLM request). This would require:
- Per-event `prompt_aligned_len` values passed to `segmented_forward`
- Complex coordinate math to map turn-relative positions to merged-sequence positions
- Changes to the eviction loop in `segmented_forward` to index per-event boundaries

**Key insight:** vLLM's `compaction_protected_prefix_tokens` parameter makes the
eviction boundary constant for ALL events. When set, eviction always starts at
`ceil(protected_prefix / block_size) * block_size`, regardless of per-turn prompt
length. This is because the protected prefix defines a fixed region that eviction
never touches, and the sliding window evicts from immediately after that region.

**Consequence:** `segmented_forward` needs NO changes. The trainer simply computes:
```
effective = min(protected_prefix_tokens, prompt_len)
prompt_aligned_len = ceil(effective / block_size) * block_size
```
This single constant is correct for every compaction event in the sample.

**What we still need:**
1. Offset-and-merge in `interleave_rollout` -- each turn's
   `num_output_tokens_at_compaction` must be shifted by the generation offset so
   segment boundaries are relative to the merged `completion_ids`, not the per-turn
   generation.
2. `num_prompt_tokens` on `CompactionEventWire` -- carried for diagnostics and
   future use (e.g., if we ever need non-protected multi-turn compaction).
3. `protected_prefix_tokens` in `CompactionConfig` -- the trainer must use the same
   constant that vLLM uses.

**What we do NOT need (vs. original plan):**
- Change 4 (per-event eviction boundaries in `segmented_forward`) -- eliminated.
- Change 5's per-event `prompt_aligned_lens` list -- a single scalar suffices.
- Any change to `segmented_forward.py` at all.

This reduces the implementation from 5 changes across 3 repos to 6 targeted edits
in 2 repos, with zero changes to the most complex component (`segmented_forward`).
