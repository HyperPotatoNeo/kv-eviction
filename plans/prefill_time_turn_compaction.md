# Plan: Prefill-Time Turn Compaction (fix first-token distribution drift)

## Status: Draft / Design Phase

Supersedes the default ordering described in `plans/turn_based_eviction.md`
where compaction fires in `update_from_output` AFTER the new token is
sampled. This plan moves the trigger EARLIER for turn-mode prefill so
every token — including the first generated token — sees the same
post-compaction KV state.

## Problem statement

Current ordering inside `scheduler.py:update_from_output` (lines
~1940-1970):

```
1. Forward pass runs against pre-compaction KV
2. Sample new_token_ids from logits
3. Append new_token_ids to request._all_token_ids / _output_token_ids
4. Stop check
5. _should_compact + _compact_request     <-- compaction here
6. Return; next scheduler step follows
```

For a fresh request whose prompt already contains more than
`compaction_max_turns` completed turns (the common case when each
multi-turn chat() call is a SEPARATE vLLM request with the full
conversation as its prompt):

```
Prefill forward  : KV built from FULL prompt (e.g. 6 turns)
First token      : sampled from logits attending to 6-turn KV
Compaction fires : drop 2 oldest turns
Decode step 2    : forward against 4-turn KV + first token
Second token     : sampled from logits attending to 4-turn KV
...
```

**Distribution drift at the prefill-decode boundary.** The first
generated token is conditioned on a KV that no subsequent token will
ever see. At best this is a harmless one-token inconsistency; at worst
it's the wedge that causes the subtle degeneration symptoms observed
in multi-turn runs (tool-call malformation, mid-word fragments) — the
model commits to a sampling decision under the long context, then
continues under a pruned context that no longer supports that choice.

### When this matters vs. when it doesn't

| Case | Compaction trigger point | First post-compaction token | Drift? |
|---|---|---|---|
| **A: fresh prompt already exceeds max_turns** | end of prefill (after prompt KV is built, BEFORE / AFTER first sample depending on ordering) | first decoded token | **YES** — the first decoded token sees a different KV than subsequent tokens |
| **B: long-running request, model emits `<\|im_end\|>` completing turn N, pushing live_turns over max** | end of the decode step that sampled the `<\|im_end\|>` | next token (which will be `\n<\|im_start\|>user\n...` from streamed input, or the env response) | minimal — the `<\|im_end\|>` itself is a well-defined token regardless of context depth, and subsequent tokens see the compacted KV consistently |
| **C: chunked prefill that crosses the boundary mid-prefill** | at the chunk boundary that first makes live_turns >= max_turns | first decoded token (same as A) | **YES** |

**Case A is the target of this plan.** Case B is already effectively
correct under current ordering (the `<|im_end|>` commits to ending the
turn, which compaction doesn't change). Case C is a subset of A.

## Root cause

The scan for `<|im_end|>` boundaries in the prompt happens LAZILY on
first call to `_effective_prompt_tokens`, which in turn is called by
`_should_compact` — which is only reached in `update_from_output`
AFTER the forward and AFTER sampling. So at prefill time the scheduler
has all the information it needs (the prompt token ids) to decide "this
prompt has too many turns", but it defers the decision until after one
forward pass has already run.

The decision should be made at request admission, not at first
update_from_output.

## Desired ordering

```
1. add_request(req)
     |-- scan prompt for turn_end_positions (already O(n))
     |-- if num_live_turns(prompt) >= max_turns:
     |     trim prompt_token_ids in place, remove the oldest
     |     compaction_eviction_turn_stride turns (block-aligned)
     |     record a CompactionEvent for trainer-side replay
     |-- enqueue
2. Prefill runs on the already-trimmed prompt
3. Last prefill step samples first token from the post-compaction KV
4. Decode continues normally
5. Mid-decode compaction (Case B) remains in update_from_output
```

Key invariant: **by the time the first forward runs on a request, the
request's KV state matches what it should be post-compaction for the
current turn count.** No forward pass ever runs on a KV state that will
be evicted before the next token is sampled.

## Scope

- **In scope:** turn-mode prefill-time compaction on the SAME physical
  trim path that `_compact_request` already uses, triggered at request
  admission.
- **Out of scope:** block-FIFO prefill-time compaction (the window-based
  trigger is inherently KV-size-dependent and requires a forward pass
  to know `num_computed_tokens`; fresh prompts just-under-window are
  allowed to prefill first and compact later without drift because no
  sampling has happened yet at that point — actually needs re-audit,
  see open question 2).
- **Out of scope:** splitting chunked prefill mid-stream. If chunked
  prefill is enabled AND the chunk-scheduling straddles the max_turns
  threshold, we compact at admission based on the FULL prompt (not
  per-chunk), then chunks operate on the trimmed prompt. No mid-chunk
  compaction.

## Design

### New code path

Add a private method on `Scheduler`:

```python
def _maybe_compact_prompt(self, request: Request) -> None:
    """Prefill-time turn compaction. Runs at request admission, before
    any forward. Trims the oldest turns from request.prompt_token_ids
    in place and records a CompactionEvent if triggered.
    No-op if compaction_max_turns == 0 or prompt has fewer than
    max_turns completed turns.
    """
```

Called from `add_request` at the point where a fresh (non-resumed,
non-streaming) request is about to be enqueued — `scheduler.py:2284`,
immediately before `self._enqueue_waiting_request(request)`.

### Mechanics

Reuse as much of `_plan_turn_evict_range` and `_compact_request` as
possible. The only difference: at admission, `request._all_token_ids`
is the prompt only (no output yet), and `num_computed_tokens` is 0.
All evicted content lies within the prompt region; no output tokens
are touched.

1. **Scan boundaries.** Call `_scan_new_turn_boundaries(request)` to
   populate `turn_end_positions` from `prompt_token_ids`. This already
   exists and is O(prompt_len).

2. **Trigger check.** `num_live_turns = (len(positions) - 1) // 2`.
   If `num_live_turns < max_turns`, return.

3. **Plan range.** Reuse `_plan_turn_evict_range(request, block_size)`.
   Same inward-snap / align_up logic (respecting
   `compaction_assume_aligned_turn_boundaries`).
   - Note: block_size is available via any CompactingKVCacheManager. At
     admission time the manager exists (scheduler init complete), so
     `self.kv_cache_manager.coordinator.single_type_managers` is valid.

4. **Trim in place.** Same trim formulas as `_compact_request` lines
   1312-1332, but SIMPLER because everything evicted is in the prompt:
   - `del request._all_token_ids[evict_start:evict_end]`
   - `del request.prompt_token_ids[evict_start:evict_end]` (raw list;
     may need a private setter — prompt_token_ids is currently a
     property/immutable in some versions, check).
   - `request.num_prompt_tokens -= total_evicted`
   - `request.position_offset += total_evicted`
   - **Do NOT** decrement `num_computed_tokens` — it's 0 and stays 0.
   - Rebuild `turn_end_positions` via the same shift-left logic.

5. **Record event.** Same CompactionEvent construction as line 1269,
   with `num_output_tokens_at_compaction=0` (no output yet) and
   `num_turns_evicted_after=stride`. The trainer replays this identically
   to a mid-generation event (segmented_forward just treats
   `num_output_tokens_at_compaction=0` as an eviction right at the
   prompt/output boundary, which is already handled).

6. **Skip block manipulation.** Unlike `_compact_request`, we do NOT
   call `compaction_mgr.compact_request(...)` here — there are no
   blocks allocated yet (prefill hasn't run). The block-level eviction
   happens implicitly: when prefill runs on the trimmed prompt, it
   allocates blocks for the SHORTER sequence. No blocks to free.

### Diagram

```
Before (current, Case A):
  add_request(prompt = sys + u1..u6 + a1..a6)    [6 completed turns]
  enqueue
  schedule() -> prefill forward on 6-turn prompt
  update_from_output -> sample t1 from 6-turn logits
  update_from_output -> check compaction -> evict 2 turns
  schedule() -> decode forward on 4-turn KV + t1
  sample t2 from 4-turn logits   <-- distribution shift here

After (this plan, Case A):
  add_request(prompt = sys + u1..u6 + a1..a6)
  _maybe_compact_prompt        ---+
    scan boundaries              | trim prompt to 4 turns
    detect 6 >= max_turns=4      | record CompactionEvent
    plan range, trim in place    |
    record event              ---+
  enqueue (request.prompt is now sys + u3..u6 + a3..a6)
  schedule() -> prefill forward on 4-turn prompt
  update_from_output -> sample t1 from 4-turn logits
  update_from_output -> check compaction -> no more eviction
  schedule() -> decode forward on 4-turn KV + t1
  sample t2 from 4-turn logits   <-- same distribution as t1
```

## Files to change

| File | Change |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | Add `_maybe_compact_prompt`; call from `add_request` just before `_enqueue_waiting_request`; refactor `_plan_turn_evict_range` if needed to accept a "no KV yet" mode (may just work as-is since it only reads positions + num_computed_tokens). |
| `vllm/v1/core/compaction/types.py` | No change. `CompactionEvent` already carries what trainer needs. |
| `vllm/v1/request.py` | Possibly: expose a setter for `prompt_token_ids` if it's currently immutable. Audit first. |
| `vllm/tests/v1/core/test_scheduler_compaction.py` | New tests: `test_prefill_time_turn_compaction`, `test_prefill_compaction_no_drift_to_decode`, `test_prefill_compaction_emits_event`. |
| `src/kv_eviction/segmented_forward.py` | **No change expected.** CompactionEvent wire format is unchanged; `num_output_tokens_at_compaction=0` events should already replay correctly (seg 0 = empty generation, seg 1 = full completion under post-eviction KV). Verify with a smoke test; if the 0-length seg 0 is a degenerate case, add a guard there. |
| `prime-rl/.../trajectories.py` | **No change expected.** Step-0 events are already the normal path. |

## Implementation order

1. **Audit `prompt_token_ids` mutability.** Check
   `vllm/v1/request.py`'s `Request.__init__` — is `prompt_token_ids`
   stored as a list we can mutate, or a tuple/property? If immutable,
   small patch needed to allow in-place trim. This is the only
   non-obvious blocker.

2. **Extract a shared trim helper.** Pull lines 1295-1366 of
   `_compact_request` into `_apply_trim(request, evict_start, evict_end,
   stride_used, last_turn_evicted, evicted_token_ids)`. Both
   `_compact_request` and `_maybe_compact_prompt` then call this helper.
   Reduces duplication risk.

3. **Implement `_maybe_compact_prompt`.** ~30 lines of new code in
   scheduler.py. Reuses `_scan_new_turn_boundaries`,
   `_num_live_completed_turns`, `_plan_turn_evict_range`, `_apply_trim`.

4. **Call from `add_request`.** One-line insertion at line 2284.

5. **Unit tests.** Extend `test_scheduler_compaction.py` with the
   scenarios listed below.

6. **Smoke test.** Run the BALROG loop from `compaction_test.ipynb`
   with `max_turns=4, stride=2`, measure first-token logit
   distribution via a hook. Before this plan: first token's top-5
   logits should differ from second token's under identical prefix
   content. After: should match.

## Testing plan

### Unit tests (scheduler)

1. `test_prefill_time_turn_compaction_trims_prompt` — request with 6
   turns + max_turns=4, stride=2 → after `add_request`,
   `request.prompt_token_ids` is trimmed to 4 turns; `num_prompt_tokens`
   decremented; `position_offset` set; one CompactionEvent on the
   request with `num_output_tokens_at_compaction == 0`.

2. `test_prefill_time_turn_compaction_no_trigger_below_threshold` —
   request with 3 turns + max_turns=4 → no-op; prompt unchanged;
   `compaction_events` empty.

3. `test_prefill_compaction_preserves_system_prompt` — same invariant
   as block-level tests: first `len(system_prompt_token_ids)` tokens
   of the trimmed prompt equal the original system prompt byte-for-byte.

4. `test_prefill_compaction_respects_aligned_boundaries_flag` — same
   request, run twice with the flag on/off; verify align_up vs
   align_down behavior and tokens_evicted delta.

5. `test_mid_decode_compaction_still_fires` — long-running request
   that generates enough to cross max_turns mid-decode; verify the
   Case B path (update_from_output trigger) still works AFTER the
   Case A prefill-time change landed.

### Integration smoke

6. Reuse `experiments/debug_balrog/compaction_debug.py` (or the
   notebook). With prefill-time compaction enabled, set a breakpoint
   at the FIRST hit of `scheduler.py:1959` on a request whose prompt
   exceeds max_turns. Expected: by the time that breakpoint hits, the
   `request.compaction_events` list already has one event with
   `num_output_tokens_at_compaction == 0` (from admission-time
   compaction). Any subsequent events from decode-time compaction
   come behind it.

### Distribution-drift regression check

7. Compare first-decoded-token logits between the current code and
   the new code on an identical prompt that exceeds max_turns. The
   new code should produce logits matching what the second-decoded-
   token logits would have been under the OLD code (i.e. the model's
   "natural" continuation under the compacted KV). Top-1 tokens
   should agree with high probability; KL divergence from "the
   intended distribution" should drop to kernel floor.

## Open questions

1. **Block-FIFO mode (Case C analog): do we care?** Block-FIFO
   compaction triggers on `num_computed_tokens > window_size`. At
   admission `num_computed_tokens == 0`, so it cannot trigger at
   admission. But a prompt longer than `window_size` will still see
   the same first-token drift once prefill finishes and the post-
   prefill compaction fires. Not covered by this plan. Opening a
   separate plan for block-FIFO prefill-time compaction may be worth
   it, but the turn-mode version is the higher-priority fix because
   turn mode is designed to be semantic-aware.

2. **Chunked prefill interaction.** vLLM's chunked prefill
   (`enable_chunked_prefill=True`) may schedule the prompt in multiple
   chunks. Prefill-time compaction at admission runs BEFORE any chunk
   is scheduled, so the compacted prompt is what gets chunked. No
   special handling needed. Verify with a test that forces chunked
   prefill on.

3. **Streaming / resumable requests.** `add_request` has a
   streaming-update branch that replaces `_all_token_ids` mid-flight
   (`_update_request_as_session`). Should prefill-time compaction
   also run on each streaming update that appends new prompt tokens?
   Deferred: the user's current workflow doesn't use streaming
   requests. If it does, the entry point is
   `_update_request_as_session` at scheduler.py:1517 — analogous hook
   point, but needs care because `num_computed_tokens` may be nonzero.

4. **Trainer replay for num_output_tokens_at_compaction=0 events.**
   Verify segmented_forward.py:399-435 handles the degenerate case
   `segment_boundaries = [0]` correctly. Expected: seg 0 = `(0,
   prompt_len + 0)` = full-prompt-only seg (no output), seg 1 =
   `(prompt_len - 1, seq_len)` = overlap + full output. Should work
   but is a new edge case.

5. **Client-side alternative.** If the user's env layer always
   produces multi-turn prompts as a SEPARATE new request, an
   equivalent fix is client-side: trim the oldest turns from the
   conversation BEFORE calling `llm.generate`. Advantages: no vLLM
   changes; symmetric with segmented_forward's current replay.
   Disadvantages: duplicates tokenizer logic in the client;
   doesn't help long-running requests that stream multiple turns
   within a single request. Recommendation: land the scheduler-side
   fix (this plan) because it's the correct architectural place, and
   it also covers the less-common long-running-request case.

## Risk / known limitations

- If `_plan_turn_evict_range` or `_apply_trim` have any dependency on
  `num_computed_tokens > 0` that we haven't spotted, admission-time
  invocation will break. Mitigation: the refactor in step 2 should be
  accompanied by unit tests that exercise both call sites against the
  same request fixture.
- `add_request` is on the hot path. Prefill-time compaction adds an
  O(prompt_len) scan + O(stride * avg_turn_len) trim. In multi-turn
  RL workloads this is negligible (prompt is read once anyway).
- If the prompt contains zero `<|im_end|>` tokens at admission (e.g.
  a model with non-ChatML templating), `turn_end_positions` is empty,
  `num_live_turns == 0`, and `_maybe_compact_prompt` is a no-op.
  Correct.
