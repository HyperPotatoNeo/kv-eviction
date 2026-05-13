# Plan: True vLLM-Style Continuous Batching for Attention Matching

## Status: DRAFT (2026-04-20)

This document describes the recommended implementation path for making the
`attention_matching` strategy behave like a real vLLM continuous-batching
strategy rather than an eval-oriented, non-preemptable baseline.

The key requirement is not "can multiple AM requests exist in a decode batch?"
That already mostly works. The real requirement is:

1. Multiple AM requests can coexist in the running batch.
2. AM requests can be preempted under KV pressure.
3. Preempted AM requests can later resume without being aborted.
4. Resume does not silently degrade AM back into FIFO or full-context behavior.
5. Existing FIFO compaction behavior remains unchanged.

This plan is intentionally conservative. The goal is to add the missing AM
resume/rebuild semantics behind the `compaction_strategy="attention_matching"`
gate without destabilizing the working FIFO path.

---

## Current State

### What already works

- The vLLM fork has an opt-in AM strategy path.
- Worker-side AM compaction can run on live KV and rewrite the current request
  in place.
- The worker can hold per-request AM state in the batch.
- The worker can construct batch-wide `score_mod` functions that apply the
  per-request AM beta biases during decode.

### What does not work

- Once an AM-compacted request is preempted, the scheduler aborts it instead of
  resuming it.
- The AM path is not reconstructable after preemption because the synthetic
  AM prefix lives only in the worker's live KV cache.
- The current AM logical prompt rewrite uses synthetic token IDs (`0`s) as a
  scheduler-facing placeholder. That is good enough for same-step bookkeeping,
  but it is not a valid replay source for a true resume path.
- Resumable streaming session updates are explicitly rejected today.

### Root cause

The current AM integration only persists lightweight logical metadata:

- `synthetic_prefix_len`
- `position_offset`
- logical prompt length / token list rewrite
- per-layer beta tensors in worker memory

That is enough for "worker compacts now, scheduler updates logical state now,
decode continues now." It is not enough for "free the request's KV blocks, put
the request back into the waiting queue, and later restore it on fresh blocks."

The missing piece is a resumable AM state representation.

---

## Definition Of Success

The implementation is complete when all of the following are true:

1. `compaction_strategy="fifo"` remains behaviorally unchanged.
2. Under `compaction_strategy="attention_matching"`, a compacted request can be
   preempted and later resumed instead of being aborted.
3. Forced-preemption tests with multiple long requests pass without crashes,
   block leaks, or silent AM disablement.
4. Greedy outputs from:
   - no-preemption AM
   - forced-preemption AM
   are identical for the same seed/config.
5. The AM path still rejects unsupported model/backend combinations cleanly.

Non-goal for the first delivery:

- Streaming input-session continuation (`request.resumable`) does not need to
  be supported. That is separate from ordinary continuous batching and can
  remain rejected initially.

---

## Recommended Design

### Chosen approach: worker-local resumable AM snapshots

The best implementation path is to make AM-compacted requests preemptable by
storing a resumable snapshot of the compacted AM state on the worker side.

That snapshot should contain enough information to rebuild the request onto a
fresh set of KV blocks without rerunning AM compaction and without trying to
 replay fake synthetic prompt tokens through the model.

At minimum, the snapshot must include:

- compacted KV tensors for the request's current AM state
  - synthetic prefix K
  - synthetic prefix V
  - exact-kept-region K
  - exact-kept-region V
- per-layer AM beta tensors
- `synthetic_prefix_len`
- `target_len`
- `position_offset`
- a generation/version counter so stale snapshots cannot be reused

### Why this is the best path

It avoids the two bad alternatives:

1. Recompute-on-resume from logical token IDs
   - Not valid, because the AM synthetic prefix is not token-addressable.
   - The current synthetic prompt rewrite uses placeholder token IDs and cannot
     be replayed through the model to regenerate the same compacted prefix.

2. Ship full AM tensors through the scheduler on every compaction
   - This would make scheduler/worker traffic much heavier.
   - It would also broaden the scheduler's responsibility into tensor payload
     transport when the scheduler only needs logical metadata.

By keeping the full AM snapshot local to the worker that already owns the
request state, we preserve the current division of responsibilities:

- scheduler owns logical request lifecycle
- worker owns actual KV state

The scheduler only needs to know whether an AM request is resumable and whether
the next scheduling step should perform an AM restore instead of a normal
resume/rebuild.

Assumption for the first implementation:

- the request resumes on the same worker that created the AM snapshot

That matches the current execution model well enough for the baseline and keeps
the first implementation much simpler. Cross-worker AM migration can remain out
of scope initially.

---

## High-Level Lifecycle

### 1. Initial AM compaction

1. Request runs normally until AM compaction triggers.
2. Worker computes the AM compacted prefix and rewrites live KV in place.
3. Worker creates a CPU-side AM snapshot for that request.
4. Worker returns lightweight AM metadata to the scheduler.
5. Scheduler updates logical request state exactly as it does now.

### 2. Preemption

1. Scheduler decides to preempt a running AM request.
2. Instead of aborting on `position_offset > 0`, it checks whether the request
   has a valid AM snapshot.
3. If yes:
   - free the request's GPU KV blocks
   - mark the request `PREEMPTED`
   - keep it in the waiting queue
4. Worker removes the request from the active batch, but keeps the AM snapshot
   and worker-side request state necessary for restore.

### 3. Resume

1. Scheduler allocates fresh blocks for the preempted AM request.
2. Scheduler marks the request as requiring AM restore.
3. Worker restores the AM snapshot into the newly allocated blocks.
4. Worker reinstalls the per-request AM beta state used by `score_mod`.
5. Request re-enters the persistent batch at the compacted logical state and
   continues decoding.

This is the missing capability that turns AM into a real continuous-batching
strategy.

---

## Architectural Changes

## A. Request / scheduler state

### Current issue

The scheduler only knows about:

- `position_offset`
- logical token IDs
- `needs_rebuild`

That is enough for FIFO and for same-step AM bookkeeping, but not enough to
tell whether an AM request is safely resumable.

### Required additions

Add AM-resume metadata to `Request`, for example:

- `attention_matching_active: bool`
- `attention_matching_snapshot_version: int | None`
- `attention_matching_restore_pending: bool`
- `attention_matching_target_len: int | None`

The scheduler should not hold full tensor payloads. It only needs the logical
facts needed to make correct lifecycle decisions.

### Files

- `vllm/vllm/v1/request.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/sched/output.py`

---

## B. Worker-side AM snapshot store

### Current issue

The worker rewrites live KV in place, but that state disappears when blocks are
freed and when preempted requests are removed from worker request state.

### Required additions

Introduce a worker-local AM snapshot structure, for example:

```python
@dataclass
class AttentionMatchingSnapshot:
    version: int
    target_len: int
    synthetic_prefix_len: int
    position_offset: int
    layer_keys: dict[str, torch.Tensor]
    layer_values: dict[str, torch.Tensor]
    layer_betas: dict[str, torch.Tensor]
```

Design constraints:

- Snapshot tensors should live on CPU, preferably pinned CPU memory.
- Snapshot shape should match the current compacted state, not the original
  pre-compaction source region.
- Snapshot creation must happen immediately after successful AM compaction.
- Snapshot ownership is per worker request state, not global scheduler state.

### Files

- `vllm/vllm/v1/core/compaction/am_runtime.py`
- `vllm/vllm/v1/worker/gpu_input_batch.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- optionally a new file:
  `vllm/vllm/v1/core/compaction/am_snapshot.py`

---

## C. Scheduler preemption policy

### Current issue

The scheduler currently treats compacted requests as unrecoverable:

- if `position_offset > 0`
- preemption aborts the request

That is correct for FIFO block deletion. It is overly strict for resumable AM.

### Required change

Replace the current AM behavior with:

- FIFO compacted request:
  - keep current behavior
  - still abort if preempted
- AM compacted request with valid snapshot:
  - preempt normally
  - do not abort
- AM compacted request without snapshot:
  - fail closed
  - abort rather than risking silent corruption

### Files

- `vllm/vllm/v1/core/sched/scheduler.py`

---

## D. Worker finish/remove semantics

### Current issue

The worker currently removes preempted requests as if they were finished. That
throws away the AM state that would be needed for resume.

### Required change

Make preempted AM requests a special case:

- remove them from the active `InputBatch`
- keep their cached request state and AM snapshot
- only fully delete them when the scheduler says they are truly finished

This is the minimum worker-side lifecycle change needed for AM resume.

### Files

- `vllm/vllm/v1/worker/gpu/model_runner.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

---

## E. Resume / rebuild path

### Current issue

The current rebuild path only knows how to rebuild:

- block tables
- token ID buffers
- `num_computed_tokens`
- `position_offset`

That works when KV already exists in place. It does not restore lost AM KV.

### Required change

Add an explicit AM restore path during worker rebuild/resume:

1. detect resumed request with AM snapshot
2. allocate fresh block slots
3. map logical compacted positions to physical slots
4. copy snapshot K/V tensors into the new slots
5. restore `attention_matching_state.layer_betas`
6. restore `num_computed_tokens`, prompt length, and `position_offset`
7. re-add request to the active batch

The key design rule is:

**Do not try to rebuild AM state from the synthetic prompt token rewrite.**

The AM snapshot is the source of truth for restore.

### Files

- `vllm/vllm/v1/core/sched/output.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

---

## F. `score_mod` restoration

### Current issue

AM decode correctness depends on the worker reinstalling per-request beta
biases into the per-layer `score_mod`.

### Required change

On restore:

- the request's `attention_matching_state` must be restored before the next
  decode step
- `_update_attention_matching_score_mods()` must see the restored beta tensors
  exactly as if the request had never left the batch

This is easy to miss and must be tested explicitly.

### Files

- `vllm/vllm/v1/worker/gpu_input_batch.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

---

## Streaming / resumable sessions

This plan does **not** require enabling streaming session continuation in the
first delivery.

Ordinary vLLM continuous batching requires:

- add/remove requests during decode
- preempt/resume under memory pressure

It does **not** require support for resumable streaming input sessions.

Recommendation:

- keep `request.resumable` rejected for AM in the first delivery
- revisit that only after preemption/resume works correctly

This keeps the initial scope tractable.

---

## Recommended Implementation Phases

## Phase AM-CB1: make AM state resumable on the worker

Deliverables:

- worker-local AM snapshot structure
- snapshot creation immediately after AM compaction
- snapshot lifetime tied to worker request state

Success criteria:

- after AM compaction, a snapshot exists for the request
- snapshot tensors match the compacted live KV state

## Phase AM-CB2: allow scheduler preemption for resumable AM requests

Deliverables:

- scheduler differentiates FIFO compacted requests from AM resumable requests
- AM preemption no longer aborts by default

Success criteria:

- AM request can enter `PREEMPTED`
- FIFO request still follows the current abort behavior after compaction

## Phase AM-CB3: implement AM restore on resume

Deliverables:

- resumed AM requests restore compacted KV from snapshot
- restored requests continue decoding correctly

Success criteria:

- forced-preempted AM request resumes and finishes
- no block leaks
- no stale beta state

## Phase AM-CB4: harden with multi-request stress tests

Deliverables:

- tests that force repeated preemption under low KV budgets
- tests that interleave long and short requests
- tests that compare no-preemption vs forced-preemption outputs under greedy decoding

Success criteria:

- repeated AM preempt/resume works
- output equivalence holds in deterministic mode

---

## File-Level Change List

Expected core files:

- `vllm/vllm/v1/request.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/sched/output.py`
- `vllm/vllm/v1/core/compaction/am_runtime.py`
- `vllm/vllm/v1/worker/gpu/model_runner.py`
- `vllm/vllm/v1/worker/gpu_input_batch.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/outputs.py`

Recommended new files:

- `vllm/vllm/v1/core/compaction/am_snapshot.py`
- `vllm/tests/v1/core/test_attention_matching_continuous_batching.py`

Existing tests to extend:

- `vllm/tests/v1/core/test_scheduler_compaction.py`

---

## Testing Strategy

## Unit tests

1. Snapshot roundtrip
   - compact AM request
   - snapshot it
   - restore into fresh slots
   - verify restored KV and betas match expected shapes and lengths

2. Scheduler preemption policy
   - FIFO compacted request still aborts
   - AM request with snapshot preempts
   - AM request without snapshot fails closed

3. Worker retention logic
   - preempted AM request is removed from active batch but not deleted from
     cached worker state

## Integration tests

1. Forced-preemption decode test
   - create low-KV-budget scenario
   - run two or more long AM requests
   - verify at least one request is preempted and later resumed
   - verify all requests finish

2. Deterministic equivalence test
   - temperature 0 / greedy decode
   - compare:
     - AM without forced preemption
     - AM with forced preemption
   - outputs must match exactly

3. Regression test for FIFO isolation
   - ensure `compaction_strategy="fifo"` behavior is unchanged

## Manual validation

Run a real server with:

- `Qwen/Qwen3-4B`
- `max_model_len=16384`
- low `num_blocks` or low cache budget to force preemption
- multiple concurrent long requests

Confirm:

- requests do not abort after first AM compaction
- throughput remains stable
- logs show preemption/resume instead of abort

---

## Risks And Mitigations

## Risk: host-memory overhead from AM snapshots

Mitigation:

- snapshot only the compacted state, not the original source region
- use pinned CPU tensors only where restore latency matters
- add metrics for snapshot bytes per request

## Risk: stale snapshot reused after later compaction

Mitigation:

- version snapshots
- invalidate old versions immediately after a newer compaction
- assert restore version matches request version

## Risk: restored beta state does not match restored KV

Mitigation:

- store beta tensors in the same snapshot version as K/V
- test `_update_attention_matching_score_mods()` after restore explicitly

## Risk: FIFO path accidentally regresses

Mitigation:

- keep all new behavior under `compaction_strategy == "attention_matching"`
- preserve current FIFO preemption semantics
- extend scheduler regression tests

## Risk: trying to use synthetic prompt token IDs as replay source

Mitigation:

- make the plan explicit: token rewrite remains scheduler bookkeeping only
- AM snapshot is the sole restore source of truth

---

## Rejected Alternatives

## 1. Re-run AM compaction on resume from logical tokens

Rejected because the logical AM request state contains synthetic placeholder
prompt tokens, not a replayable source sequence.

## 2. Send full AM tensors through scheduler outputs

Rejected because it bloats scheduler/worker communication and gives the
scheduler ownership over tensor payloads it should not manage.

## 3. Keep the current abort-on-preemption behavior

Rejected because it fails the actual requirement: true vLLM-style continuous
batching.

---

## Recommended Order Of Work

1. Add worker-local AM snapshot support.
2. Change scheduler preemption policy for resumable AM requests.
3. Add AM restore on resumed requests.
4. Add deterministic forced-preemption tests.
5. Only after all of the above, consider enabling streaming-session resume.

This ordering keeps the hard part isolated and preserves the current FIFO path
while AM is under construction.

---

## Final Recommendation

Implement true AM continuous batching by making AM-compacted requests
preemptable and resumable through a worker-local compacted-state snapshot.

That is the narrowest change set that:

- satisfies the real continuous-batching requirement
- keeps FIFO isolated
- avoids invalid replay from synthetic prompt tokens
- and does not force the scheduler to transport large tensor payloads

Anything smaller than that will still leave AM outside the real vLLM
continuous-batching contract.
