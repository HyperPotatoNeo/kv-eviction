# Plan: Turn-Based KV Cache Compaction

## Status: Draft / Design Phase

Supersedes the earlier draft of this file, which was a looser design
discussion about "lower the protected boundary from full prompt to
system prompt only". This plan commits to a concrete policy + wiring.

## Motivation

Current block-FIFO compaction (`compaction_window_size`,
`compaction_stride`, `compaction_protected_prefix_tokens=-1`) cuts at
arbitrary 16-token block boundaries. In multi-turn environments
(BabyAI, Crafter, etc.) this slices through ChatML message headers
(`<|im_start|>user\n`), mid-word subwords, and partial tool-call JSON.
After 20+ compactions the accumulated fragments degrade the model's
output: observed artifacts include `"I new observation"` (lost `"A "`),
`"TheThe"`, doubled articles, mid-word fragments. Confirmed visually
in `experiments/debug_balrog/compaction_test.ipynb` against
Qwen3-4B-Instruct on `BabyAI-MixedTrainLocal-v0/goto`.

Turn-based compaction replaces "evict the oldest N blocks" with
"evict the oldest K turns", where a **turn = one user + one assistant
message pair**. Semantically this is what a multi-turn RL env actually
wants: drop entire stale turns rather than arbitrary KV slices.

## Desired semantics

Two new knobs on `CacheConfig`:

- `compaction_max_turns: int = 0` — max number of live (not-yet-evicted)
  turns to keep in context. `0` = disable turn mode (fall back to block
  FIFO, backward compat).
- `compaction_eviction_turn_stride: int = 1` — when we exceed
  `max_turns`, evict this many oldest turns at once. Larger stride =
  fewer but larger compaction events.

Policy: once `num_live_turns >= max_turns`, evict turns
`[num_turns_evicted, num_turns_evicted + stride)` — the oldest live
turns. This happens at the same `_should_compact` check point as block
FIFO, so no new scheduler hook.

**Turn definition.** A turn = the 2 consecutive messages
`(user, assistant)`. The system message is turn 0 and is implicitly
protected (never evicted). In a ChatML stream, the K-th turn ends at
the (2K+1)-th `<|im_end|>` token:

```
<|im_start|>system...<|im_end|>    ← 1st im_end, end of turn 0 (sys)
<|im_start|>user...<|im_end|>      ← 2nd im_end, end of user msg of turn 1
<|im_start|>assistant...<|im_end|> ← 3rd im_end, end of turn 1
<|im_start|>user...<|im_end|>      ← 4th im_end, end of user msg of turn 2
<|im_start|>assistant...<|im_end|> ← 5th im_end, end of turn 2
```

So if we let `P = list of absolute positions of the token AFTER each
<|im_end|>`, then `P[0]` is the end of the system message, and
`P[2k]` (for k >= 1) is the end of turn k. `num_completed_turns =
(len(P) - 1) // 2`.

## Key constraints

1. **Native vLLM.** All logic lives in
   `vllm/vllm/v1/core/sched/scheduler.py` and related modules. No
   env-layer hacks, no extra wire fields beyond what we already ship.

2. **Low per-step overhead.** Turn tracking must be O(1) amortized per
   generated token. No per-step full scan of `_all_token_ids`.

3. **Reuses the existing physical eviction path.** The compaction
   manager (`CompactingKVCacheManager.compact_request`) already
   evicts block-aligned ranges given a `prompt_tokens` param. We want
   to keep using it — turn mode just computes a different
   `effective_prompt` (logical boundary where eviction starts) and
   feeds it through unchanged.

4. **Backward compat.** When `compaction_max_turns == 0`, behavior is
   identical to current block-FIFO mode. The existing auto-detect
   protected-prefix mode (`compaction_protected_prefix_tokens=-1`)
   continues to work unchanged.

## Design

### Turn boundary tracking on `Request`

Add three fields to `Request.__init__` (in the `--- Compaction state ---`
block around `vllm/v1/request.py:138`):

```python
# Turn tracking (only populated when compaction_max_turns > 0).
# Absolute positions (in the CURRENT post-eviction _all_token_ids) of
# the first token AFTER each <|im_end|> seen so far. Monotonic.
self.turn_end_positions: list[int] = []
# Cursor: tokens in _all_token_ids[:last_turn_scan_pos] have already
# been scanned for <|im_end|>. Lazy, extended on demand in
# _effective_prompt_tokens(). Reset/adjusted on eviction.
self.last_turn_scan_pos: int = 0
# Count of whole turns (user+assistant pairs) already physically
# evicted by prior compactions on this request. Monotonic.
self.num_turns_evicted: int = 0
```

No behavior change when `compaction_max_turns == 0` — fields stay empty.

### Incremental scan helper on the scheduler

Add a private method in `scheduler.py` next to `_effective_prompt_tokens`:

```python
def _scan_new_turn_boundaries(self, request: Request) -> None:
    """Extend request.turn_end_positions with any <|im_end|> tokens
    in _all_token_ids[last_turn_scan_pos : num_tokens].
    O(n_new_tokens). Called only from _effective_prompt_tokens when
    turn mode is enabled.
    """
    end_id = self._compaction_turn_end_token_id
    if end_id is None:
        return
    toks = request._all_token_ids
    start = request.last_turn_scan_pos
    end = len(toks)
    # Local vars for speed.
    positions = request.turn_end_positions
    for i in range(start, end):
        if toks[i] == end_id:
            positions.append(i + 1)  # position AFTER the im_end token
    request.last_turn_scan_pos = end
```

Bandwidth analysis: the `for i in range` loop runs once per generated
token, but only for tokens we have not yet seen. In steady state,
`_should_compact` is called once per scheduler step per running
request. Between calls, ~1 new token is appended per running request
(decode step). So we do O(1) comparisons per step per request.
Python-loop overhead is the concern; we can accept it because the
scheduler is already doing per-request Python work per step. If this
shows up in profiling, we can vectorize by doing a single
`bytes(toks[start:end]).find(...)`-style pass.

### Getting the `<|im_end|>` token id

Option 1 (picked): add a config field
`compaction_turn_end_token_id: int | None = None`. When `None` and
`compaction_max_turns > 0`, look it up at `Scheduler.__init__` from
`self.model_config.hf_config` / tokenizer and cache on
`self._compaction_turn_end_token_id`. For Qwen3 this is `151645`.

Option 2 (reject): reuse `sampling_params.eos_token_id`. This already
works for the auto-detect protected-prefix code path because Qwen3's
chat template lands on `<|im_end|>` as EOS. But it's fragile — some
models have a separate EOS distinct from the message-end token.

Resolution: support both. Config field wins if set; otherwise probe
the tokenizer via `model_config.get_hf_config()`-style access (same
path the current auto-detect code uses), and if that fails, fall back
to `sampling_params.eos_token_id` at first call. Single-line warning
log on the fallback.

### `_effective_prompt_tokens` in turn mode

Rewrite to route through turn mode when enabled:

```python
def _effective_prompt_tokens(self, request: Request) -> int:
    if self._compaction_max_turns > 0:
        return self._turn_mode_effective_prompt(request)
    # ... existing protected-prefix / full-prompt logic unchanged ...
```

Where `_turn_mode_effective_prompt` is:

```python
def _turn_mode_effective_prompt(self, request: Request) -> int:
    """Return the logical position where the oldest live turn starts.

    Semantics: effective_prompt = position of the first token of the
    oldest live turn = turn_end_positions[0] (system-prompt end).
    CompactingKVCacheManager.needs_compaction treats tokens
    [effective_prompt, num_computed) as "evictable output region",
    which for turn mode means "turns 1..N plus current generation".
    That is exactly what we want: eviction starts at the end of the
    system prompt, and the window / stride guards control when.
    """
    self._scan_new_turn_boundaries(request)
    positions = request.turn_end_positions
    if not positions:
        # No <|im_end|> seen yet. Protect full prompt (conservative).
        return request.num_prompt_tokens
    # positions[0] = end of system message = start of turn 1.
    return positions[0]
```

This makes the existing `needs_compaction` window/stride guard fire
only when `num_computed_tokens` exceeds `window + sys_end`, which is
the right protection for "don't compact until the live turns are big".

But note: this by itself does NOT enforce "evict whole turns" — it
just loosens the protected boundary. We need a second hook to compute
a turn-aligned eviction range.

### Turn-aligned eviction in `_compact_request`

The existing `_compact_request` calls
`mgr.compact_request(request_id, effective_prompt)` which evicts
`stride_blocks * block_size` tokens starting at
`align_up(effective_prompt, block_size)`. For turn mode we need to
evict a range `[A, B)` where both `A` and `B` are constrained to be
as close to turn boundaries as possible, block-aligned.

**Option A (recommended): override `needs_compaction` and `compact_request`
with a turn-aware path in the scheduler only**, without touching
`CompactingKVCacheManager`. The manager already supports a custom
`eviction_fn: Callable[[int, int, int], list[int]]` that returns
explicit block indices. We will:

1. Implement the turn-mode trigger check in the scheduler (replaces
   the `mgr.needs_compaction` call in `_should_compact`). The check
   is: `num_live_turns >= compaction_max_turns`, where
   `num_live_turns = (len(turn_end_positions) - 1) // 2`.
   Also require `num_computed_tokens > window` as a sanity guard (so
   that short-but-turn-heavy conversations still go through the
   normal window).

2. For the physical eviction, compute the turn range in logical token
   space:

   ```python
   k_first = 0  # index into "live turns" (0-indexed post-eviction)
   k_last = min(stride, num_live_turns) - 1
   # Absolute end-positions of each live turn (within _all_token_ids):
   #   live_turn_end[k] = turn_end_positions[2*(k+1)]      # k=0 → pos after 2nd im_end
   live_turn_ends = request.turn_end_positions  # positions[2k] is end of turn k
   turn_first_start = live_turn_ends[0]         # start of turn 1 = end of sys
   turn_last_end    = live_turn_ends[2 * (k_last + 1)]  # end of turn stride
   ```

   Snap both to block boundaries **inward** (strictly inside the turn
   range):

   ```python
   evict_start = ((turn_first_start + bs - 1) // bs) * bs   # align_up
   evict_end   = (turn_last_end // bs) * bs                 # align_down
   if evict_end <= evict_start:
       return 0  # turns too short for this block size; bail out
   total_evicted = evict_end - evict_start
   evict_block_start = evict_start // bs
   evict_block_end   = evict_end // bs
   ```

3. Call the manager with an `eviction_fn` closure that returns
   `list(range(evict_block_start, evict_block_end))`. Reuses all
   existing block-free machinery. No changes needed in
   `manager.py`.

4. Trim `_all_token_ids` / `_output_token_ids` exactly as the current
   generalized trim in `_compact_request` (the formulas already
   handle both prompt and output regions via
   `prompt_tokens_evicted` / `output_tokens_evicted`). Decrement
   `num_prompt_tokens` by `prompt_tokens_evicted`.

5. Update turn-tracking state:

   ```python
   # Drop the boundary markers that were inside the evicted range.
   # Everything after evict_end shifts left by total_evicted.
   new_positions = []
   for p in request.turn_end_positions:
       if p <= evict_start:
           new_positions.append(p)
       elif p >= evict_end:
           new_positions.append(p - total_evicted)
       # else: p is inside the evicted range -> drop
   request.turn_end_positions = new_positions
   request.last_turn_scan_pos = len(request._all_token_ids)
   request.num_turns_evicted += stride
   ```

6. Emit a `CompactionEvent` with the new fields populated (see
   below).

**Block alignment tradeoff (important — document & test).**
Snapping inward at both ends means:
- Up to `block_size - 1` tokens at the START of the first evicted
  turn will NOT be physically evicted. They remain in
  `_all_token_ids` immediately after the system prompt. They are the
  first few tokens of a user message like `<|im_start|>user\nObs`
  — semantically stale but not corrupting the kept content.
- Up to `block_size - 1` tokens at the END of the last evicted turn
  will NOT be evicted. They remain immediately before the first kept
  turn. Typically the tail of an assistant tool-call message.

Both orphan fragments are located in content we wanted to remove
anyway. They are visible to the model as "noise between system and
first kept turn" but they do not corrupt the start of the kept
content or the live generation tail — which was the smoking-gun
failure mode in the current block-FIFO behavior.

With `block_size = 16` and turns of length ~120 tokens (typical
BabyAI), the orphan fragments are ~12% of one turn × 2 per eviction,
and they accumulate if multiple compactions fire. If smoke tests
show degradation even under turn mode, the follow-up is message-level
padding at the chat-template layer to force turn boundaries to be
block-aligned. That's out of scope for this plan.

**Alternative considered and rejected (for now):**
Snapping outward at the start (so the tail of the system prompt gets
evicted) corrupts protected content. Snapping outward at the end
(so the start of the next kept turn gets evicted) corrupts live
content. Both are worse than inward snap. Hybrid strategies don't
help because any outward snap loses content we wanted to keep.

### Config surface

New fields on `vllm/config/cache.py:CacheConfig` (add to
`ignored_factors` in `compute_hash()` — runtime-only, no graph impact):

```python
compaction_max_turns: int = 0
"""KV compaction: max number of live (user+assistant) turns to keep
in context. 0 = disable turn mode (use block-FIFO compaction)."""

compaction_eviction_turn_stride: int = 1
"""KV compaction: how many oldest turns to evict at once when
compaction_max_turns is exceeded. Must be >= 1. Larger = fewer but
bigger compaction events."""

compaction_turn_end_token_id: int | None = None
"""KV compaction: token id marking the end of a chat message (e.g.
<|im_end|> = 151645 for Qwen3). None = auto-detect from model config
at Scheduler init. Only consulted when compaction_max_turns > 0."""
```

Corresponding fields in `vllm/engine/arg_utils.py:EngineArgs`, plus
validation in the compaction-enabled branch:

```python
if self.compaction_max_turns > 0:
    assert self.compaction_window_size > 0, (
        "compaction_max_turns requires compaction_window_size > 0 "
        "as a safety fallback"
    )
    assert self.compaction_eviction_turn_stride >= 1
    assert self.compaction_protected_prefix_tokens in (0, -1), (
        "compaction_max_turns implies auto protected prefix; set "
        "compaction_protected_prefix_tokens to 0 or -1"
    )
```

Pass through to `CacheConfig(...)` alongside the existing compaction
fields.

### Scheduler init

In `Scheduler.__init__`, near `self._compaction_protected_prefix = ...`:

```python
self._compaction_max_turns = self.cache_config.compaction_max_turns
self._compaction_eviction_turn_stride = (
    self.cache_config.compaction_eviction_turn_stride
)
self._compaction_turn_end_token_id: int | None = None
if self._compaction_max_turns > 0:
    tid = self.cache_config.compaction_turn_end_token_id
    if tid is None:
        tid = self._auto_detect_im_end_token_id()  # reads model_config
    self._compaction_turn_end_token_id = tid
    logger.info(
        "[COMPACT] turn mode enabled: max_turns=%d stride=%d im_end_id=%s",
        self._compaction_max_turns,
        self._compaction_eviction_turn_stride,
        tid,
    )
```

### `CompactionEvent` additions

`vllm/v1/core/compaction/types.py` — add two optional fields to the
struct. `omit_defaults=True` already lets us extend safely:

```python
# Turn mode: the "turn index" of the last turn that was evicted by
# this event (0-indexed, inclusive). For block-FIFO mode this stays
# at the default (-1). Lets the trainer correlate events with the
# conversation structure.
last_turn_evicted: int = -1

# Turn mode: cumulative number of turns physically evicted so far.
# Same default-sentinel story as above.
num_turns_evicted_after: int = 0
```

Populated in `_compact_request` only when turn mode is active. The
existing `num_prompt_tokens`, `position_offset_after`, `tokens_evicted`
fields continue to carry the information the trainer's
`segmented_forward` needs.

### Interaction with existing protected-prefix mode

Turn mode **replaces** the protected-prefix computation
(`_effective_prompt_tokens` returns the system-message end). Existing
`compaction_protected_prefix_tokens` is required to be 0 or -1 when
turn mode is on (see validation above). The auto-detect path is still
used as a fallback for `_effective_prompt_tokens` when we haven't yet
seen enough `<|im_end|>` tokens to know where turn 1 starts.

### Interaction with async_scheduling, LMCache, prefix caching

- async_scheduling + compaction is already refused at Scheduler init.
  No change — turn mode also refuses.
- LMCache + compaction is already refused. No change.
- Prefix caching: the first turn's scan is stable (prompt is fixed),
  so `turn_end_positions` can be populated at request creation time
  by a one-shot scan of `_all_token_ids` during `Request.__init__`
  when turn mode is globally enabled. Avoids the first-step scan.
  Optional optimization; not required for correctness.

## Files touched

| File | Change |
|---|---|
| `vllm/config/cache.py` | Add `compaction_max_turns`, `compaction_eviction_turn_stride`, `compaction_turn_end_token_id`; add to `ignored_factors`. |
| `vllm/engine/arg_utils.py` | Mirror fields, validation, pass-through to `CacheConfig`. |
| `vllm/v1/request.py` | Add `turn_end_positions`, `last_turn_scan_pos`, `num_turns_evicted` in compaction-state block. |
| `vllm/v1/core/sched/scheduler.py` | `_compaction_max_turns` etc. in `__init__`, `_auto_detect_im_end_token_id` helper, `_scan_new_turn_boundaries` helper, new turn-mode branch in `_effective_prompt_tokens`, new turn-mode branch in `_should_compact` (check live-turn count) and `_compact_request` (compute turn-aligned evict range, pass `eviction_fn` to manager, update turn-tracking state after trim), populate new `CompactionEvent` fields. |
| `vllm/v1/core/compaction/types.py` | Add `last_turn_evicted`, `num_turns_evicted_after` to `CompactionEvent`. |
| `vllm/v1/core/compaction/manager.py` | No change — already supports `eviction_fn`. |
| `vllm/tests/v1/core/test_scheduler_compaction.py` | New tests (see below). |
| `vllm/tests/v1/core/utils.py` | Accept new params in `create_scheduler()`. |

Explicitly **not** touched:
- `src/kv_eviction/env.py` — no wire changes needed.
- `src/kv_eviction/segmented_forward.py` — CompactionEvent already
  carries the information needed for replay; new fields are
  metadata-only.
- `prime-rl/` — no changes.

## Testing plan

### Unit (in `test_scheduler_compaction.py`)

1. `test_turn_mode_boundary_scanning` — feed a fabricated token
   stream with known `<|im_end|>` positions through the scheduler
   step loop; verify `turn_end_positions` matches after each step.
2. `test_turn_mode_trigger_at_max_turns` — `max_turns=3`,
   `stride=1`; verify no compaction fires at turns 1, 2; fires at
   turn 3 when the 3rd turn's token count pushes past the window.
3. `test_turn_mode_evicts_one_turn` — verify after a single
   `stride=1` compaction: `num_turns_evicted == 1`,
   `turn_end_positions` is shortened correctly (positions shifted
   left), `num_prompt_tokens` decremented, `_all_token_ids` length
   matches, `position_offset` incremented by `total_evicted`.
4. `test_turn_mode_stride_two_evicts_two_turns` — same but
   `stride=2`.
5. `test_turn_mode_block_alignment_edge_cases` — turn ends at
   exact block boundary (no inward snap needed), turn very short
   (inward snap makes `evict_end <= evict_start` and `_compact_request`
   bails out with 0), verify no state corruption on bail-out.
6. `test_turn_mode_backward_compat` — `compaction_max_turns=0`,
   everything behaves like existing block-FIFO tests.
7. `test_turn_mode_compaction_event_fields` — verify
   `CompactionEvent.last_turn_evicted`, `num_turns_evicted_after`,
   `num_prompt_tokens` all populated, msgspec round-trip works.

### Integration smoke (no unit infra needed)

8. Reuse `experiments/debug_balrog/compaction_test.ipynb`. Add a
   second LLM instance configured with:
   ```
   compaction_window_size=1024
   compaction_stride=256
   compaction_max_turns=4
   compaction_eviction_turn_stride=2
   compaction_turn_end_token_id=None  # auto-detect
   ```
   Run the 30-turn BabyAI loop and compare output quality to the
   block-FIFO run already in the notebook. Look specifically for:
   - No `"I new observation"` / `"TheThe"` / mid-word fragments in
     assistant output after turn 5.
   - `compaction_events` fires at turn 5 and every 2 turns after,
     with `last_turn_evicted` incrementing by 2 each time.
   - `num_prompt_tokens` on the final event reflects whole-turn
     evictions (multiples of roughly `turn_len`, modulo block
     alignment slack).
9. If smoke #8 looks good, wire the same config into one of the
   `experiments/compaction_rgmix/` configs and run a 5-step RL smoke
   to confirm Mismatch KL stays at kernel floor (~1e-3) — turn mode
   should not introduce any new KL gap because segmented_forward
   already consumes the same `CompactionEvent` shape.

## Open decisions (flag to user before implementing)

1. **Inward-snap orphan tolerance.** Is it OK to leave ~12% fragments
   of evicted turns in the stream as described above, or do we need
   exact eviction? If exact: the follow-up is message-boundary
   padding in the chat template, which is a larger scope.

2. **Single-turn eviction vs batched.** Default `stride=1` gives more
   predictable output-quality tests but more compaction events
   (more overhead). Default `stride=2` halves the event count. The
   user's example said "stride can be 2". Recommend default=1 and
   expose the knob.

3. **`compaction_window_size` as safety fallback.** In turn mode the
   primary trigger is the live-turn count, but we still require
   `compaction_window_size > 0` so that if turn-counting somehow
   gets desynced (e.g. a model that stops emitting `<|im_end|>`),
   the block-level guard still fires before we OOM. Is that
   acceptable, or should turn mode disable `window_size` entirely?

4. **`CompactionEvent` turn fields.** Are `last_turn_evicted` and
   `num_turns_evicted_after` enough for the trainer to correlate
   events with the conversation, or does `segmented_forward` need
   the full `turn_end_positions` snapshot? Current plan assumes the
   trainer never needs to reconstruct conversation boundaries
   because it replays against `_all_token_ids` which already
   reflects the post-eviction state. Confirm before implementing.

5. **Auto-detect im_end token id.** Where in `model_config` is the
   cleanest place to read this? Qwen3 stores it in
   `tokenizer.special_tokens_map`, but `model_config` may not expose
   the tokenizer directly from inside the scheduler. May need to
   plumb it through `CacheConfig` at engine-init time instead of
   at `Scheduler.__init__`. Small wiring detail, confirm before
   coding.

## Risk / known limitations

- Orphan fragments at block boundaries (documented above).
- Assumes ChatML-like templates with a single message-end token. Not
  portable to models with different delimiters without setting
  `compaction_turn_end_token_id` explicitly.
- No handling of "incomplete turn" at the tail — if the assistant is
  mid-generation and we want to evict, we only evict completed
  turns. An assistant message that has not yet emitted `<|im_end|>`
  is not counted in `num_live_turns` and cannot be evicted. This is
  the desired behavior for live generation but may create a slight
  window-size undershoot right before the first `<|im_end|>` of a
  long assistant message.
- Does not interact with prefix caching reuse across requests. Each
  turn is still a separate vLLM request (handled by the env layer).
  Turn-based compaction operates WITHIN a single long-running request
  — e.g., one request that streams multiple turns of a conversation.
  If the env creates a new request per turn, turn-based compaction
  is a no-op and we fall through to normal block-FIFO guards.
  **Confirm with the user whether the balrog loop uses one request
  per episode or one request per turn** — this affects whether turn
  mode is the right fix at all.
