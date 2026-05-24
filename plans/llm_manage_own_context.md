# LLM Managed Context

## Goal

Let the model decide which evicted turns it wants back in context.

Today the system uses summary/compaction: old turns are summarized, and vLLM
evicts their KV blocks from the active request. The new idea is to make those
evicted turns addressable. Instead of only summarizing and deleting, we keep an
index entry for each evicted turn span and preserve its KV somewhere. Later, the
model can request one or more span IDs, and vLLM can restore those KV blocks for
the next generation.

The important distinction: the summary/index should tell the model what kind of
information is in an evicted span, not necessarily reveal the answer. For the
fruit test, the index entry should say something like "turn contains the user's
favorite-fruit fact", not "favorite fruit is apple". Otherwise the model solves
the task from text summary rather than restored KV.

## Small Example Diagram

Fruit-recall example: the model should know that an old span is relevant, ask
for that span, and only then recover the hidden fact from restored KV.

```text
Initial long conversation:

  Turn 01: user says favorite fruit = apple
  Turn 02..24: unrelated chatter
  Turn 25: user asks "what is my favorite fruit?"

After compaction:

  Visible model context:
    [system]
    [memory index]
      T0001: early user turn containing a durable user preference
    [recent turns 20..25]

  Hidden evicted KV archive:
    T0001 -> KV blocks for original Turn 01 tokens

Model pass 1:

  Sees index entry but not the answer.
  Emits:
    {"retrieve": ["T0001"]}

Orchestrator:

  Retries the pending answer request with:
    kve_restore_span_ids = ["T0001"]

vLLM pass 2:

  Active context for generation:
    [system]
    [memory index]
    [restored KV span T0001]
    [recent turns 20..25]

  Model can now answer:
    "Your favorite fruit is apple."
```

## Core Loop With Turn Window

The desired setup is:

```text
Settings:
  max_turns = 10
  stride = 5              # total post-compaction turn budget
  recall_max_spans = 2    # recall slots reserved inside that budget
  preserved_recent_turns = stride - recall_max_spans = 3

Before compaction at turn 10:
  [system]
  [T01][T02][T03][T04][T05][T06][T07][T08][T09][T10]

Compaction trigger:
  live complete turns == max_turns == 10
  compact now, before continuing past the turn-10 window

Eviction / summary cycle at turn 10:
  old raw context:
    [T01][T02][T03][T04][T05][T06][T07]
      -> summarized into a compact state update
      -> described in a recall index
      -> raw tokens deleted from visible context
      -> KV spans archived under IDs

  recent context:
    [T08][T09][T10]
      -> remains directly visible/live

After eviction:
  [system]
  [summary of T01..T07]
  [recall index]
    T0001: early user turn with durable preference info
    T0002: tool result about the locked room
    T0003: earlier failed plan and correction
    ...
    T0007: earlier user/assistant exchange
  [T08][T09][T10]

Later, if the model needs old context:
  model emits:
    {"retrieve": ["T0001", "T0003"]}

  orchestrator enforces:
    len(retrieve) <= recall_max_spans

  vLLM restores selected archived KV spans for the next generation.

Effective active turn budget after recall:
  3 directly preserved recent turns + up to 2 recalled spans = stride 5
```

In this plan, `stride` means the total post-compaction effective turn budget,
not just the number of recent complete turns that remain visible. The directly
preserved tail is `stride - recall_max_spans`. With `stride=5` and
`recall_max_spans=2`, compaction at turn 10 keeps only turns 8-10 directly
visible and leaves two slots for model-selected recall.

The vLLM-side field `compaction_eviction_turn_stride` has a different meaning:
it means "how many oldest turns are physically evicted per event." The
implementation needs to map these intentionally instead of treating the two
stride names as interchangeable.

### Recall Capacity Parameter

Use a model-facing parameter named `recall_max_spans`.

Reason: the model is choosing logical span IDs from the recall index, not raw
vLLM KV pages. A word like `blocks` is ambiguous in this repo because "block"
already means a fixed-size KV cache page. One evicted turn span may occupy many
KV blocks.

Recommended names:

- `recall_max_spans`: maximum number of span IDs the model may request in one
  retrieval pass. This is a reserved capacity inside `stride`, not extra
  capacity added on top of it.
- `recall_max_kv_blocks`: optional lower-level safety cap on the total physical
  KV blocks that may be restored in one pass.
- `--compaction-recall-max-spans`: likely vLLM CLI/cache config spelling if
  this becomes a server-side enforcement knob.

Validate `0 <= recall_max_spans <= stride`. The directly preserved recent-tail
count is:

```text
preserved_recent_turns = stride - recall_max_spans
```

For MVP, it is probably safest to require `preserved_recent_turns >= 1` so the
model always sees the current local task state.

## Existing Pieces

The codebase already has most of the scaffolding needed around this:

- `src/kv_eviction/summarization.py` partitions chat messages into turn groups
  and builds summary exchanges.
- `src/kv_eviction/env.py` already intercepts OpenAI chat calls, runs a
  side-channel summary request, splices rewritten messages, forwards
  `prompt_token_ids`, and carries Phase4 rollout-local state.
- `vllm/vllm/v1/core/sched/scheduler.py` already has turn-boundary scanning,
  turn-mode eviction planning, compaction event emission, Phase4 prefix-cache
  pinning, and position-offset inheritance.
- `vllm/vllm/v1/core/compaction/manager.py` is the current physical eviction
  point. It chooses block indices, frees blocks, and deletes entries from
  `req_to_blocks`.
- `vllm/vllm/v1/simple_kv_offload/*` already contains CPU/GPU block-copy
  infrastructure, but it is prefix-cache/hash oriented. It is useful later, but
  not the safest first step for arbitrary evicted-turn restore.
- `experiments/fruit_recall_env/fruit_recall_env.py` is the right first
  behavioral test because the hidden fact starts in turn 1 and the final
  question is at turn 25.

## Recommended MVP

Start with a two-pass retrieval protocol and GPU-pinned evicted spans. Do not
start with special tokens or CPU offload.

1. When the turn count reaches the eviction trigger, generate a new compact
   summary plus short descriptions for the turns that are leaving visible
   context.
2. Delete the prior raw context from the submitted prompt/live request, keeping
   only system context, the new summary/index, and the last
   `stride - recall_max_spans` turns.
3. Archive the evicted KV span references instead of immediately making them
   unreachable.
4. Surface a small text memory index to the model:
   `T0001: early user turn containing a durable user preference.`
5. Let the model output a strict JSON retrieval request when it wants old
   context, capped by `recall_max_spans`:

```json
{"retrieve": ["T0001"]}
```

6. The orchestrator detects that JSON, suppresses it from the environment if
   needed, and retries the same pending turn with:

```python
extra_body = {
    "vllm_xargs": {
        "kve_restore_span_ids": ["T0001"]
    }
}
```

7. vLLM attaches the restored spans to the active request for the retry and
   generates the final answer.

This avoids interrupting decode mid-token. Special tokens can come later after
the simpler control loop proves that restored KV changes behavior.

## Evicted Span Record

Add a scheduler-side archive record. The initial version can keep GPU block
refs pinned. Later versions can replace `block_ids_by_group` with CPU block IDs
or a loaded/offloaded state machine.

```python
EvictedSpanRecord:
    span_id: str
    trace_id: str
    request_id: str
    absolute_turn_start: int
    absolute_turn_end: int
    token_ids: list[int]
    block_ids_by_group: tuple[list[int], ...]
    kv_block_count: int
    logical_start_by_block: list[int]
    position_offset_frame: int
    index_summary: str
    status: "gpu_pinned" | "cpu_offloaded" | "loading" | "expired"
```

The record needs enough metadata to preserve RoPE/frame correctness:

- block IDs per KV group
- physical KV block count for `recall_max_kv_blocks` accounting
- logical starts for each block
- current request `position_offset`
- token IDs for observability and test validation
- turn/span IDs for the model-facing catalog

## vLLM Runtime Shape

### Archive On Eviction

The hook point is `_compact_request` in `scheduler.py`, just before
`compaction_mgr.compact_request(...)` frees and deletes blocks. For turn mode,
the planner already gives `evict_start`, `evict_end`, `last_turn_evicted`, and
`stride_used`.

The current compaction manager should either:

- return the evicted `KVCacheBlock` objects/IDs to the scheduler before freeing,
  or
- accept a callback/archive object so the scheduler can pin/copy blocks before
  they leave `req_to_blocks`.

For MVP, pin evicted blocks by incrementing refs and keep them out of the free
queue. After the archive owns them, the active request can still remove them
from its block table and proceed exactly like current compaction.

### Restore On Request

On a new/retried request, read `kve_restore_span_ids` from
`request.sampling_params.extra_args`.

The restore request must be validated against `recall_max_spans` before vLLM
reattaches anything. If we also add `recall_max_kv_blocks`, vLLM should compute
the physical block cost of the requested spans and reject or truncate requests
that exceed the safety cap.

MVP restore semantics should be logical concatenation:

```text
system/current compact prefix + restored span(s) + pending user fragment
```

This is less magical than hidden external memory and fits the existing block
table/paged-attention model. Hidden memory that does not count as context would
require attention backend changes and should not be the first implementation.

The restore path must:

- fetch/pin archived blocks
- append them to the request's computed block list or create a restore prefix
- preserve logical position/frame metadata
- avoid double-freeing restored blocks when the request finishes
- emit restore metadata back to the client/trainer

## Model-Facing Protocol

Use JSON first:

```json
{"retrieve": ["T0001", "T0004"]}
```

Reasons:

- easy to parse in `env.py` or the evaluator
- easy to enforce `recall_max_spans`
- no tokenizer or special-token surgery
- no need to stop decode inside vLLM
- easy to test in fruit recall

Special tokens can be a later optimization:

```text
<|retrieve:T0001,T0004|>
```

That would require decoding-time interception, cancellation/resume, and careful
handling of partially emitted tokens. It is not the right MVP.

## Summary / Index Generation

Extend the summary pipeline to produce two artifacts during each eviction:

- a compact state summary that replaces old visible context
- compact span index entries that describe the archived turns the model may
  request later

This can ride on the existing side-channel summary call in `env.py`.

Candidate prompt shape:

```text
For each evicted turn span, write 1 short sentence describing what kind of
task-relevant information it contains. Do not reveal exact secret values unless
the user explicitly repeated them outside the span.
Return JSON: [{"span_id": "...", "summary": "..."}]
```

For fruit recall, the desired entry is category-level:

```json
{"span_id": "T0001", "summary": "The user gave a durable personal preference."}
```

not:

```json
{"span_id": "T0001", "summary": "The favorite fruit is apple."}
```

The catalog should be appended to the visible prompt, probably as a system or
developer-style memory index section, while the raw evicted turn remains absent
from visible context.

## Training Integration

Inference-only fruit tests do not need trainer changes.

For RL/SFT parity later, add a new wire type rather than overloading
`CompactionEventWire`:

```python
RestoreEventWire:
    span_ids: list[str]
    restored_token_ids: list[int]
    block_ranges: ...
    position_offset_after: int
```

Then thread it through `CallWire`, `TrainingSample`, and the per-call trainer
replay. The trainer must know when restored KV was available to query tokens.
Until this exists, restored-KV runs should be treated as inference diagnostics,
not matched trainer/inference experiments.

## Design Questions To Decide

1. Index or substitute?

   The index should point to old information without giving away the exact
   hidden value. This is required for fruit recall to measure KV restore.

2. Transient or sticky restore?

   MVP should make restore transient for one retry/answer. Sticky restores make
   accounting and later evictions more complex.

3. Do restored spans count against `max_model_len`?

   MVP should count them. Not counting them means implementing a hidden-memory
   attention path.

4. Are restored spans inserted before or after the compact current prefix?

   Recommended MVP: after protected/system/current compact prefix and before
   the pending user fragment. This gives the final question direct access to the
   restored fact.

5. Can the system require Phase4/prefix caching?

   Yes for MVP. The existing Phase4 path already gives us rollout-local IDs and
   avoids cross-episode prefix-cache contamination.

6. How are span IDs scoped?

   Scope by rollout/session trace ID, not globally. A span ID like `T0001` is
   only meaningful within one active conversation.

7. What happens when a span expired or offload load is slow?

   MVP can fail closed: return no restore and let the model answer without it.
   Later we can expose a retry/error message.

8. What should the recall budget count?

   The model-facing budget should count logical spans, so use
   `recall_max_spans`. This is the number of entries the model can choose from
   the recall index in one pass. It is reserved inside the post-compaction
   `stride` budget, so directly preserved recent turns are
   `stride - recall_max_spans`. If memory pressure requires a physical cap, add
   `recall_max_kv_blocks` separately and enforce both caps.

9. Does eviction delete all previous context?

   It should delete all previous raw context that is outside the kept
   `stride - recall_max_spans` tail from the visible prompt and active request.
   The replacement visible context is summary plus recall index. The old KV
   survives only in the archive, scoped to the session, until it is restored or
   expires.

## Implementation Phases

### Phase 1: Inference-Only GPU-Pinned Prototype

- Add `EvictedSpanRecord` and span archive to the scheduler.
- Capture evicted turn block IDs and token IDs before physical compaction.
- Pin archived blocks so they are not reused.
- Add `recall_max_spans` enforcement in the retrieval parser/orchestrator.
- Emit span catalog metadata on compaction responses.
- Add JSON retrieval parsing in the fruit evaluator or `env.py`.
- Retry with `kve_restore_span_ids`.
- Restore selected span blocks for the retry generation.

Success criteria:

- final visible prompt does not contain the fruit
- model selects the fruit span ID
- restore enabled improves recall
- restore disabled or wrong span selected fails

### Phase 2: Cleaner Orchestrator Protocol

- Move JSON retrieval handling into `kv_eviction.env`.
- Add configurable retrieve instruction text.
- Add metrics:
  - spans archived
  - spans requested
  - restore hits/misses
  - restored tokens
  - restore latency

### Phase 3: CPU Offload

- Reuse `simple_kv_offload` copy primitives for archived spans.
- Add archive states: `gpu_pinned`, `cpu_offloaded`, `loading`, `expired`.
- Add async load scheduling before retry generation.
- Add eviction policy for CPU archive capacity.

Do this only after Phase 1 proves correctness. CPU offload changes lifetimes,
copy synchronization, and failure modes.

### Phase 4: Trainer Replay

- Add `RestoreEventWire`.
- Thread restore metadata through `CallWire` and `TrainingSample`.
- Extend per-call replay so restored spans are present for the same query
  tokens as in vLLM.
- Add exact parity diagnostics analogous to the current fruit recall
  vLLM/Flex replay.

### Phase 5: Special Token / Single-Pass UX

- Add retrieve special tokens only if the JSON two-pass version is correct.
- Implement decode interruption/resume around retrieval tokens.
- Decide whether retrieved spans are visible in generated output or purely
  control messages.

## Fruit Test Plan

Use the existing fruit recall task:

- turn 1: "my favorite fruit is X"
- turns 2-24: distractor OK turns
- turn 25: ask for the fruit

Test matrix:

1. Full context baseline: should answer correctly.
2. Current compaction without restore: final visible prompt should not contain
   fruit; recall may fail depending checkpoint/config.
3. Managed context with correct restore: model emits retrieval JSON for the
   fruit-bearing span, retry restores span, final answer is correct.
4. Managed context with restore disabled: retrieval JSON is ignored; final
   answer should drop.
5. Managed context with wrong span forced: answer should drop.

The key metric is not just final recall. The test must also assert that:

- the fruit-bearing turn is absent from submitted visible tokens
- the selected span ID is the fruit-bearing span
- the retrieval request respects `recall_max_spans`
- vLLM reports a restore hit for that span
- the final answer changes only when the right KV span is restored

## Risks

- RoPE/frame drift: restored blocks must keep their original logical starts and
  position-offset frame.
- Lifetime bugs: archived blocks must not be freed or reused while a restore can
  reference them.
- Cross-episode contamination: span IDs and prefix-cache state must be scoped by
  rollout/session.
- Summary leakage: if index summaries reveal exact facts, the experiment no
  longer tests restored KV.
- Trainer mismatch: until restore events are wired into trainer replay, this is
  an inference-only feature.

## Current Recommendation

Build Phase 1 as a narrow fruit-recall diagnostic:

1. GPU-pinned evicted span archive.
2. Category-level span index.
3. `recall_max_spans` budget on JSON retrieval requests.
4. Retry with `kve_restore_span_ids`.
5. Restore selected KV spans for one answer.

This gives the shortest path to answering the real question: can an LLM learn to
use a compact memory index to request old KV context at the moment it needs it?
