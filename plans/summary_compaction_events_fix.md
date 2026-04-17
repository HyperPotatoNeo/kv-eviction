# Plan: Block-Pad Eviction-Mode Summary + Capture `compaction_events`

## Context

Eviction-summary training crashed on all three seeds within 15–22
minutes:

```
AssertionError: prompt_aligned_len (6368) exceeds seq_len (6367)
```

Fires in `src/kv_eviction/segmented_forward.py:365` from
`prime-rl/src/prime_rl/trainer/rl/train.py:601`:

```python
prompt_aligned_len = ((effective_prompt_len + bs - 1) // bs) * bs
```

When the prompt isn't a multiple of `block_size`, the round-up can
overshoot `seq_len` for short completions. Rank N crashes → other
ranks hang 1h on `_ALLGATHER_BASE` → NCCL watchdog kills the job.

### Root cause: branch ordering bypasses padding

`env.py:_install_message_padding_interceptor.patched_create` has
two branches:

- **Branch A** (markovian-thinker, line 733): re-tokenize via
  `mcfg.tokenizer.apply_chat_template` + `encode` — **no
  block-aligned padding**. Returns on line 846.
- **Branch B** (block-aligned padding, line 848): calls
  `render_padded_prompt` — the padding machinery.

In eviction-summary configs we set both
`orchestrator.markovian_thinker.enabled = true` (required by the
summary validator) and `orchestrator.compaction_padding.enabled =
true` (for clean vLLM block eviction). Branch A fires first and
branch B never runs. So:

- The main rollout call to vLLM goes through **unpadded**.
- The inner summary call in `_generate_summary` uses `orig_create`
  directly (to skip the interceptor for recursion safety) — also
  **unpadded**.

Turns end mid-block, and the trainer's `prompt_aligned_len` overshoots
on short completions.

### User's framing — why padding is the right fix

"In this case we should just pad — this would fix that issue."
Right. Every other turn in a compaction run gets block-padded so the
trainer's `prompt_aligned_len = prompt_len` math is exact by
construction. Summaries and the post-summary main calls should not
be exceptions. Padding also makes vLLM's block eviction land on clean
turn boundaries, which is the whole reason `compaction_padding` is
on in the first place.

## Fix scope (two parts)

### Part 1 — Apply block-aligned padding in branch A when eviction mode is on

**File: `src/kv_eviction/env.py` — `patched_create` branch A
(lines 810-833).**

After building `new_messages` from the post-summary splice (or the
plain Markovian fallback), when `_padding_config` is not None,
enabled, and `scfg.mode == "eviction"`, render via
`render_padded_prompt` instead of raw `apply_chat_template` +
`encode`. Use the existing `_padding_config.tokenizer`,
`block_size`, `filler_token_id`, `im_end_token_id`.

Markovian-mode branch A: unchanged (raw encode) — no vLLM
compaction, padding is unnecessary.

The resulting `prompt_token_ids` are forwarded via
`extra_body["prompt_token_ids"]` exactly as today. `_stash_prompt_token_ids`
still runs, so the response carries the padded ids — the trainer
sees identical tokens + positions.

### Part 2 — Pad the summary call's own prompt, capture `compaction_events`

**File: `src/kv_eviction/env.py` — `_generate_summary` (lines 475-581).**

When `scfg.mode == "eviction"` **and** `_padding_config` is enabled:

1. Build `summary_messages = full_messages + [I_msg]` as today.
2. Render with `render_padded_prompt` → padded token ids.
3. Pass via `summary_kwargs["extra_body"] =
   {"prompt_token_ids": padded_ids}` (same mechanism branch A and
   branch B use).
4. After the call, also call `_extract_compaction_event_dicts(resp)`
   and include the result in `sample_dict["compaction_events"]`.
   vLLM can emit events during summary-call prefill/decode when
   the padded prompt exceeds `window_size`.

Markovian mode `_generate_summary`: unchanged — no padding, no
event capture. `sample_dict` continues not to carry
`compaction_events` (field omitted or empty list).

**Files: `src/kv_eviction/summarization.py`.**
Add a `compaction_events: list[dict] = field(default_factory=list)`
field to `SummaryTrainSample` and include it in `to_dict` /
`from_dict`.

**File: `prime-rl/src/prime_rl/orchestrator/trajectories.py` —
`_build_summary_sample` (lines 249-305).**

Read `compaction_events` from the payload. Coerce dicts to
`CompactionEventWire` using the same dict/list/typed handling as
`_compaction_events_from_step` at line 474. Pass as
`compaction_events=<list or None>` into `TrainingSample`. Update
the docstring — the "always None" claim is now mode-dependent.

## Why this fixes the bug

With padding on the summary call's prompt, `prompt_token_ids`
lands on a block boundary. In the trainer:

```python
mb_prompt_len  = len(padded_prompt_ids)         # multiple of bs
effective_prompt_len = min(pp, mb_prompt_len)   # ≤ mb_prompt_len
prompt_aligned_len   = ((effective + bs - 1) // bs) * bs
```

When `effective == mb_prompt_len` (the event-less branch): round-up
is a no-op (already aligned). `prompt_aligned_len == prompt_len`.
Completion of any length ≥ 1 → `prompt_aligned_len < seq_len`.
Assertion passes.

When events are present (the normal eviction-mode summary case):
`pp = first_evt.position_offset_after - first_evt.tokens_evicted`
which is block-aligned by vLLM's eviction invariants. Round-up
still a no-op. Assertion passes.

## Files to change

| File | Change |
|---|---|
| `src/kv_eviction/env.py` | Branch A: pad `new_messages` when `scfg.mode=="eviction"` and padding config enabled. `_generate_summary`: same in eviction mode; also extract `compaction_events` from `resp` into `sample_dict`. |
| `src/kv_eviction/summarization.py` | `SummaryTrainSample.compaction_events: list[dict]` + dict roundtrip. |
| `prime-rl/src/prime_rl/orchestrator/trajectories.py` | `_build_summary_sample` pass through `compaction_events` as `CompactionEventWire` list. |
| `tests/test_summarization.py` | Extend roundtrip test with non-empty events. |
| `tests/test_summary_interceptor.py` | Add eviction-mode-with-padding path; assert padded `prompt_token_ids`, captured events. Markovian-mode: unchanged behavior. |
| `prime-rl/tests/unit/orchestrator/test_interleave_summary.py` | Assert `TrainingSample.compaction_events` populated when payload has events. |

Not touched: trainer code, `segmented_forward.py`, validators
(no new required flags — `compaction_padding.enabled=true` is
already mandatory in eviction-summary configs by operator
convention; adding a validator check is optional future work).

## Remaining clarifying questions

1. **Validator hardening (optional).** Should I add a new check in
   `rl.py:validate_markovian_thinker`:
   `summary.mode=="eviction"` implies
   `orchestrator.compaction_padding.enabled=true`? It's belt-and-
   braces — the configs already set this, but a validator would
   catch future regressions. OK to add?

2. **Main-rollout padding under branch A.** Part 1 adds padding to
   branch A when `scfg.mode=="eviction"`. This changes the
   `prompt_token_ids` vLLM sees on the main rollout calls compared
   to the currently-running eviction-summary runs. It's the
   *intended* state (padding is supposed to be on), so this is a
   fix, not a behavior change. Confirming you want this rather than
   padding ONLY the summary call.

3. **Telescoping summary compaction events.** In eviction mode,
   successive summary calls over a long rollout can each carry their
   own compaction events (vLLM evicting during each summary's
   prefill). Do you want per-summary per-sample event attribution
   (my default: yes — each summary sample carries the events emitted
   during its own call), or batch them somehow? Default seems right.

4. **Markovian-mode parity.** Part 1's guard is
   `scfg.mode=="eviction"`. Markovian-mode branch A stays
   byte-identical → parity regression (step-0 rollout token-identical
   to existing runs) holds by construction. Confirming that's what
   you want (no changes to markovian plumbing).

## Test plan

### Unit
- `SummaryTrainSample` roundtrip with 2-event list.
- `_build_summary_sample` with `compaction_events=[{...}]` →
  `TrainingSample.compaction_events` is `list[CompactionEventWire]`.

### Interceptor (mocked `orig_create`, parametrized over mode)
- **Eviction mode, padding enabled**: branch-A main call sends
  padded `prompt_token_ids` (length multiple of `block_size`);
  summary-call `summary_kwargs` carries padded
  `extra_body.prompt_token_ids`; captured sample dict has
  `compaction_events` forwarded from `resp`.
- **Markovian mode**: branch A unpadded as today;
  `_generate_summary` unpadded; no events captured.

### Integration smokes
- **3 eviction-summary seeds, fresh `-v3` run names, 5 steps**:
  no NCCL timeout, no assertion, `markovian_summary/n_per_step > 0`
  by step 2-3, loss finite, mismatch KL ~kernel floor. Sanity:
  dump a summary sample's `prompt_token_ids` len — multiple of 16.
- **3 markovian-summary seeds (existing runs, unchanged)**: no
  relaunch needed; parity regression by construction (branch-A
  markovian path untouched).

## Risks

1. **Padding math drift.** `render_padded_prompt` is the same
   helper used by branch B today. Low risk.
2. **Summary call's prefix cache.** vLLM prefix-caches the padded
   stream; identical mechanism to the main rollout. No new risk.
3. **Event-extraction on `AsyncChatCompletion` response.**
   `_extract_compaction_event_dicts` already handles both attribute
   and `model_extra` access; no special casing needed for the
   summary response.
4. **If `_padding_config` is disabled** in an eviction-summary
   config (operator mistake): branch A still unpadded → bug
   persists. Mitigated by question 1's validator check.
