# Compaction Entrypoints & Key Functions

End-to-end map of every function involved in KV cache compaction,
from vLLM inference through trajectory plumbing to training replay.
For each function: what it does, where it lives, and whether it needs
modification for Phase B (multi-turn compaction).

---

## 1. vLLM Inference (compaction fires here)

### `CompactingKVCacheManager.needs_compaction()`
**File:** `vllm/vllm/v1/core/compaction/manager.py:66-106`

Decides whether to compact. Two guards:
1. `num_computed_tokens > compaction_window_size`
2. Full-block safety: last evicted block must be fully filled

**Phase B change:** None. This is per-request logic and works the same
regardless of whether the request is turn 0 or turn 5.

---

### `CompactingKVCacheManager.compact_request()`
**File:** `vllm/vllm/v1/core/compaction/manager.py:108+`

Physically frees `stride_blocks` blocks starting at prompt-boundary block
index. Splices the block table. Returns `tokens_evicted`.

**Phase B change:** None. Pure block-level operation, unaware of turns.

---

### `Scheduler._should_compact()`
**File:** `vllm/vllm/v1/core/sched/scheduler.py:1038-1047`

Iterates KV cache managers, calls `needs_compaction()`. Entry point from
the scheduler's per-token loop.

**Phase B change:** None.

---

### `Scheduler._compact_request()`
**File:** `vllm/vllm/v1/core/sched/scheduler.py:1049-1120+`

Orchestrates one compaction event:
1. Calls `compact_request()` on the KV cache manager (physical eviction)
2. **Creates `CompactionEvent`** at line 1078:
   ```python
   event = CompactionEvent(
       num_output_tokens_at_compaction=request.num_total_generated,
       tokens_evicted=total_evicted,
       position_offset_after=request.position_offset + total_evicted,
   )
   ```
3. Mutates request: trims `all_token_ids`, `output_token_ids` at the
   block-aligned boundary (`prompt_aligned_len`), decrements
   `num_computed_tokens`

**Phase B change: YES.** The event currently records
`num_total_generated` (output tokens since start of THIS vLLM request).
For multi-turn, each turn is a separate vLLM request, so
`num_total_generated` resets to 0 each turn. The orchestrator needs to
know the per-turn prompt length to compute absolute positions in the
merged training sample. **Options:**
- Add `num_prompt_tokens` to `CompactionEvent` (preferred)
- Compute it downstream from trajectory step metadata

---

### Scheduler main loop (compaction trigger)
**File:** `vllm/vllm/v1/core/sched/scheduler.py:1629-1648`

After each token is appended and stop-check passes:
```python
while self._should_compact(request):
    tokens_evicted = self._compact_request(request)
    if tokens_evicted == 0:
        break
    request.needs_rebuild = True
```

**Phase B change:** None.

---

### `CompactionEvent` / `CompactionEventPayload`
**File:** `vllm/vllm/v1/core/compaction/types.py:13-42`

Wire type for events. Fields:
- `num_output_tokens_at_compaction`: output token count when compaction fired
- `tokens_evicted`: number of tokens removed
- `position_offset_after`: cumulative position offset for RoPE correction

**Phase B change: YES** — add `num_prompt_tokens: int` field so
downstream can compute the per-turn eviction boundary.

---

## 2. Event Plumbing (vLLM → verifiers → trajectory step)

### `_extract_compaction_event_dicts()`
**File:** `src/kv_eviction/env.py:32-108`

Reads `compaction_events` off an openai `ChatCompletion` response.
Returns `list[dict]` (not CompactionEventWire — must be JSON-serializable
for verifiers state transport). Handles dict, CompactionEventWire, and
pydantic object forms.

**Phase B change: YES** if `CompactionEvent` gets a new field
(`num_prompt_tokens`), this function must extract and propagate it.

---

### `attach_compaction_events_from_response()`
**File:** `src/kv_eviction/env.py:116-132`

Mutates a trajectory step's `extras` dict:
```python
step["extras"]["compaction_events"] = event_dicts
```

**Phase B change:** None (passes through whatever
`_extract_compaction_event_dicts` returns).

---

### Monkey-patch 1: `patched_from_native()`
**File:** `src/kv_eviction/env.py:218-241`

Patches `OpenAIChatCompletionsClient.from_native_response` to copy
`compaction_events` from the raw openai ChatCompletion (where pydantic
`extra="allow"` preserved it) to the verifiers `Response` object. Without
this, verifiers drops the field during its hardcoded field-list
conversion.

**Phase B change:** None (generic field copy).

---

### Monkey-patch 2: `patched_add_model_response()`
**File:** `src/kv_eviction/env.py:251-255`

Patches `MultiTurnEnv.add_model_response` to call
`attach_compaction_events_from_response` on every step. This is what
makes compaction events appear in trajectory steps for ALL env types
(not just those explicitly subclassing `CompactionEnvMixin`).

**Phase B change:** None.

---

### `_install_compaction_event_hooks()`
**File:** `src/kv_eviction/env.py:206-261`

Called at module import time. Installs both monkey-patches above.
Sentinel-guarded for idempotency.

**Phase B change:** None.

---

### `compaction_events_from_step_extras()`
**File:** `src/kv_eviction/env.py:264-305`

Read-side helper: `step["extras"]["compaction_events"]` → `list[CompactionEventWire]`.
Handles dict, CompactionEventWire, and array_like msgspec forms.

**Phase B change: YES** if CompactionEventWire gains a new field.

---

## 3. Trajectory Processing (multi-turn merge)

### `interleave_rollout()`
**File:** `prime-rl/src/prime_rl/orchestrator/trajectories.py:239-465`

Converts `RolloutOutput` → `list[TrainingSample]`. Core logic:
- Step 0: creates first sample with its compaction events
- Steps 1+: checks extension property, merges into existing sample
- **Lines 435-441: `NotImplementedError` guard** — crashes if a
  non-zero step carries compaction events during an extension merge

**Phase B change: YES — the primary blocker.** Must replace the
`NotImplementedError` with offset-and-merge logic:
1. Each step's `num_output_tokens_at_compaction` is relative to that
   turn's generation start (0). In the merged sample, the absolute
   offset = `len(sample.completion_ids)` at the point before
   `extend_sample` appends the new turn's tokens.
2. Offset each event: `event.num_output_tokens_at_compaction += offset`
3. Append to `sample.compaction_events`

---

### `_compaction_events_from_step()`
**File:** `prime-rl/src/prime_rl/orchestrator/trajectories.py:297-333`

Helper inside `interleave_rollout`. Reads step extras → `list[CompactionEventWire]`.
Normalizes dict/list/tuple wire forms.

**Phase B change: YES** if CompactionEventWire gains a new field.

---

### `make_sample()`
**File:** `prime-rl/src/prime_rl/orchestrator/trajectories.py:344-371`

Creates a `TrainingSample` from one step's tokens. Attaches
`compaction_events` if provided.

**Phase B change:** None (already accepts events).

---

### `extend_sample()`
**File:** `prime-rl/src/prime_rl/orchestrator/trajectories.py:373-404`

Appends a new turn to an existing sample: new prompt tokens (mask=False)
then completion tokens (mask=True).

**Phase B change: Possibly.** May need to also accept and append
compaction events with adjusted offsets (or this could live in the
caller in `interleave_rollout`).

---

## 4. Wire Types

### `CompactionEventWire`
**File:** `prime-rl/src/prime_rl/transport/types.py` (msgspec.Struct, array_like=True)

Fields: `num_output_tokens_at_compaction`, `tokens_evicted`, `position_offset_after`

**Phase B change: YES** — add `prompt_len: int` (the prompt length of
the vLLM request that produced this event, needed for per-event eviction
boundaries in segmented_forward).

---

### `TrainingSample`
**File:** `prime-rl/src/prime_rl/transport/types.py:36-60`

Has `compaction_events: list[CompactionEventWire] | None`.

**Phase B change:** None (already supports event lists).

---

### `MicroBatch`
**File:** `prime-rl/src/prime_rl/transport/types.py:72-102`

Has `compaction_events` and `prompt_len`.

**Phase B change: Possibly.** May need a `per_event_prompt_lens: list[int]`
if segmented_forward needs per-event eviction boundaries.

---

## 5. Trainer Dispatch

### `train()` — compaction branch
**File:** `prime-rl/src/prime_rl/trainer/rl/train.py:485-600+`

Reads `micro_batch["compaction_events"]` and `micro_batch["prompt_len"]`.
D5 unified dispatch: `use_segmented = config.compaction.window_size > 0`
(all samples go through segmented_forward when compaction is enabled).

Key computations:
```python
prompt_aligned_len = ceil(prompt_len / block_size) * block_size  # line 581
segment_boundaries = [e.num_output_tokens_at_compaction for e in events]  # line 582-584
```

**Phase B change: YES.** Currently uses a single `prompt_aligned_len`
for all events. Multi-turn needs **per-event eviction boundaries**:
```python
# instead of one prompt_aligned_len, pass a list:
per_event_prompt_aligned = [
    ceil(e.prompt_len / block_size) * block_size
    for e in compaction_events
]
```

---

### `segmented_forward()` call site
**File:** `prime-rl/src/prime_rl/trainer/rl/train.py:780-791`

Passes `prompt_aligned_len` as a scalar. Must change to pass per-event
boundaries for Phase B.

**Phase B change: YES.**

---

## 6. Segmented Forward (training replay of compaction)

### `segmented_forward()`
**File:** `src/kv_eviction/segmented_forward.py:254-267` (signature)

Replays compaction during training. Splits `input_ids` into segments at
event boundaries. Between segments, drops KV entries from
`[prompt_aligned_len, prompt_aligned_len + stride)` to match what vLLM's
scheduler did at inference time.

**Phase B change: YES — the critical change.** Signature must accept
per-event eviction boundaries instead of a single `prompt_aligned_len`:
```python
# current:
prompt_aligned_len: int
# proposed:
prompt_aligned_lens: list[int]  # one per segment boundary
```

---

### KV eviction in segmented_forward
**File:** `src/kv_eviction/segmented_forward.py:617-658`

The actual KV splice between segments:
```python
keys[l][:prompt_aligned_len]                              # keep prompt KV
keys[l][prompt_aligned_len + actual_stride : -trim]       # keep post-eviction KV
```

Uses `prompt_aligned_len` (fixed, from step 0). In multi-turn, each
compaction event has a DIFFERENT eviction boundary (the prompt of that
turn is longer because it includes all prior conversation).

**Phase B change: YES.** Must index into `prompt_aligned_lens[event_idx]`
per eviction event:
```python
pal = prompt_aligned_lens[seg_idx]  # per-event boundary
keys[l][:pal] + keys[l][pal + actual_stride : -trim]
```

---

## Summary: Files Needing Phase B Changes

| File | Function | Change |
|------|----------|--------|
| `vllm/.../compaction/types.py` | `CompactionEvent` | Add `num_prompt_tokens` field |
| `vllm/.../sched/scheduler.py` | `_compact_request()` | Populate `num_prompt_tokens` on event |
| `src/kv_eviction/env.py` | `_extract_compaction_event_dicts()` | Extract new field |
| `src/kv_eviction/env.py` | `compaction_events_from_step_extras()` | Pass through new field |
| `prime-rl/.../transport/types.py` | `CompactionEventWire` | Add `prompt_len` field |
| `prime-rl/.../orchestrator/trajectories.py` | `interleave_rollout()` | Replace NotImplementedError with offset+merge |
| `prime-rl/.../orchestrator/trajectories.py` | `_compaction_events_from_step()` | Handle new field |
| `prime-rl/.../trainer/rl/train.py` | compaction dispatch | Compute per-event `prompt_aligned_len` list |
| `src/kv_eviction/segmented_forward.py` | `segmented_forward()` | Accept + use per-event eviction boundaries |

---

## Flow Diagram

```
  vLLM INFERENCE (per token)
  ─────────────────────────
  scheduler main loop (:1632)
    │
    ├─► _should_compact()              (:1038)
    │     └─► needs_compaction()       (manager.py:66)
    │           checks: computed > window AND full-block
    │
    └─► _compact_request()             (:1049)
          ├─► compact_request()        (manager.py:108)  — physical block eviction
          ├─► CompactionEvent(         (:1078)
          │     num_output_tokens = request.num_total_generated,
          │     tokens_evicted,
          │     position_offset_after)
          ├─► request.compaction_events.append(event)
          └─► trim all_token_ids, output_token_ids, num_computed_tokens

  PLUMBING (inference → training)
  ───────────────────────────────
  vLLM server returns ChatCompletion with compaction_events field
    │
    ▼
  patched_from_native()                (env.py:218)
    copies compaction_events from ChatCompletion → verifiers Response
    │
    ▼
  patched_add_model_response()         (env.py:251)
    │
    └─► attach_compaction_events_from_response()  (env.py:116)
          └─► _extract_compaction_event_dicts()   (env.py:32)
                step["extras"]["compaction_events"] = [{...}, ...]
    │
    ▼
  RolloutOutput returned with trajectory steps carrying events in extras

  TRAJECTORY MERGE
  ────────────────
  interleave_rollout()                 (trajectories.py:239)
    │
    ├─► _compaction_events_from_step() (:297)  — normalize wire formats
    │
    ├─► make_sample()                  (:344)  — step 0 → TrainingSample
    │     with compaction_events attached
    │
    └─► for step 1..N:
          ├─► extension check          (:422)
          ├─► *** NotImplementedError   (:435) ← PHASE B BLOCKER ***
          └─► extend_sample()          (:373)  — merge turn into sample

  TRAINER DISPATCH
  ────────────────
  train()                              (train.py:485)
    │
    ├─► compaction_events = micro_batch["compaction_events"]
    ├─► prompt_aligned_len = ceil(prompt_len / block_size) * block_size
    ├─► segment_boundaries = [e.num_output_tokens_at_compaction for e]
    │
    └─► segmented_forward(             (segmented_forward.py:254)
          input_ids, position_ids,
          segment_boundaries,
          prompt_len, prompt_aligned_len,  ← PHASE B: per-event list
          stride, temperature, loss_fn)

  SEGMENTED FORWARD (training replay)
  ────────────────────────────────────
  segmented_forward()                  (segmented_forward.py:254)
    │
    └─► for each segment:
          ├─► model(input_ids[seg_start:seg_end], past_key_values=kv)
          ├─► loss_fn(logits, ...)     — per-segment backward
          └─► KV eviction              (:617)
                drop keys[prompt_aligned_len : prompt_aligned_len + stride]
                                       ← PHASE B: per-event boundary
```
