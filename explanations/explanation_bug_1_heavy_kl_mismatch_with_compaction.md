# Bug 1: Heavy KL Mismatch Between Trainer and Inference with Compaction

## Symptom

```
loss/mismatch_kl_mean >> 0.001
```

Expected: ~0.0009 (kernel floor — the irreducible difference between two
flash_attention_2 evaluations of the same sequence). Observed: orders of
magnitude higher, indicating a **real correctness bug** where the trainer
computes logprobs under a different context than inference did.

## What is mismatch KL?

The trainer recomputes logprobs for every token the inference engine
generated. If the trainer sees the same context as inference, the two
logprobs should be nearly identical. Mismatch KL measures the divergence:

```python
# loss.py:137-139
log_importance_ratio = trainer_logprobs - inference_logprobs
importance_ratio = torch.exp(log_importance_ratio)
mismatch_kl = importance_ratio - log_importance_ratio - 1
```

This is `exp(x) - x - 1` where `x = trainer - inference`. At x=0 (perfect
match), mismatch_kl = 0. It grows quadratically for small x and
exponentially for large x. A "heavy" mismatch means the trainer and
inference see fundamentally different contexts for the same tokens.

## How compaction creates two different logprob regimes

When compaction is active, inference logprobs have a **temporal split**:

```
Token timeline during inference:
  ┌──────────────────────────────────────────────────────────────┐
  │ gen token 0 ... gen token 3896                               │
  │ logprobs computed under FULL context (positions 0..4096)     │
  ├──────────────────────────────────────────────────────────────┤
  │ ← COMPACTION FIRES HERE: evict 512 tokens from KV cache →   │
  ├──────────────────────────────────────────────────────────────┤
  │ gen token 3897 ... gen token N                               │
  │ logprobs computed under REDUCED context (3585 KV entries)    │
  └──────────────────────────────────────────────────────────────┘
```

The stored `inference_logprobs` in the TrainingSample faithfully record
what inference computed at each step. Pre-compaction tokens have full-context
logprobs; post-compaction tokens have reduced-context logprobs.

## How segmented_forward is supposed to match this

The trainer's `segmented_forward` replicates the split:

```
Segment 0: input_ids[0 : prompt_len + boundary]
  → Full forward under flash_attention_2
  → Logprobs match pre-compaction inference logprobs ✓

Between segments: evict KV[prompt_aligned_len : prompt_aligned_len + stride]
  → Same drop as vLLM's scheduler did

Segment 1: input_ids[prompt_len + boundary - 1 : seq_len]
  → Forward under post-eviction KV + boundary token re-feed
  → Logprobs match post-compaction inference logprobs ✓
```

The boundary-token overlap is critical: the last token of segment 0 is
re-fed in segment 1 so its logit (predicting the first post-compaction
token) is recomputed under the reduced context, matching what inference
saw when it sampled that token.

**When this works**, mismatch KL sits at ~0.0009 (kernel floor). The code
has been validated at this level in smoke runs.

## The three failure modes that produce heavy KL

### Failure mode 1: Compaction events silently lost (MOST LIKELY)

If compaction events don't reach the trainer, `segmented_forward` runs
with empty `segment_boundaries`. The D5 fix routes every sample through
`segmented_forward` when `config.compaction.window_size > 0`, but with
no boundaries it produces a single segment covering the full sequence.
This is a plain full-context forward.

```
What the trainer computes (events lost):
  ┌──────────────────────────────────────────────────────────────┐
  │ SINGLE full-context forward over entire sequence             │
  │ Every token's logprob computed under FULL context            │
  └──────────────────────────────────────────────────────────────┘

What inference actually did:
  ┌──────────────────────────┬───────────────────────────────────┐
  │ Full context logprobs    │ REDUCED context logprobs          │
  └──────────────────────────┴───────────────────────────────────┘

  Mismatch: every post-compaction token has trainer_logprob ≠ inference_logprob
```

For a 16K sequence with window=4096, compaction fires ~24 times
(`(16384 - 4096) / 512`). The majority of tokens were generated under
reduced context. A full-context trainer reforward produces systematically
different logprobs for ALL of them → heavy KL.

#### The four-stage event pipeline and where events can be lost

Events flow through four stages. A break at ANY stage silently drops
all events and triggers this failure mode.

```
Stage 1: vLLM scheduler → ChatCompletionResponse
─────────────────────────────────────────────────
scheduler._compact_request() records CompactionEvent on Request
  → request.compaction_events: list[CompactionEvent]

serving.py builds ChatCompletionResponse:
  compaction_events = [CompactionEventPayload(...) for e in events]
  → Wire JSON includes "compaction_events": [...]

✅ Verified: serving.py:1613-1637 correctly converts and attaches events.


Stage 2: ChatCompletion (openai SDK) → verifiers Response
─────────────────────────────────────────────────────────
Problem: openai-python's ChatCompletion has extra="allow" so the
compaction_events field is preserved as an extra attribute. But
verifiers' from_native_response() constructs a NEW Response object
from a hardcoded field list — it does NOT copy extra fields.

Fix: env.py Patch 1 wraps from_native_response to copy events:
  raw_events = getattr(response, "compaction_events", None)
  setattr(verifiers_response, "compaction_events", raw_events)

⚠️  FRAGILE: depends on:
  (a) openai SDK preserving the extra field (pydantic extra="allow")
  (b) The monkey_patch_chat_completion_logprobs() in orchestrator.py
      NOT stripping model_extra when it replaces ChatCompletion.Choice
  (c) The patched method being on the RIGHT class (needs to be on the
      class the runtime actually instantiates, not a parent that got
      overridden)


Stage 3: verifiers Response → TrajectoryStep.extras
────────────────────────────────────────────────────
Problem: verifiers stores trajectory step data as a TrajectoryStep
TypedDict. There is no "compaction_events" key in this TypedDict — events
must go into the generic "extras" dict.

Fix: env.py Patch 2 wraps MultiTurnEnv.add_model_response:
  After the base class appends the step, reads events from the Response
  and writes them into step["extras"]["compaction_events"].

✅ Verified: SingleTurnEnv inherits from MultiTurnEnv, so the patch
   applies to both. The patched method is correctly sentinel-guarded.

⚠️  But: if the env subprocess (spawned via mp.get_context("spawn"))
   doesn't import kv_eviction, the patches are NOT installed in the
   process that runs rollouts.

Fix: envs.py:_env_server_subprocess_entrypoint imports kv_eviction
  before starting the server. ✅ Verified in code.


Stage 4: TrajectoryStep.extras → TrainingSample.compaction_events
─────────────────────────────────────────────────────────────────
trajectories.py:_compaction_events_from_step reads
  step["extras"]["compaction_events"] and converts dicts to
  CompactionEventWire instances.

✅ Verified: handles dict, CompactionEventWire, and array_like forms.

Events must survive JSON serialization through the ZMQ env server.
This is why env.py stores events as plain dicts (not msgspec structs)
— CompactionEventWire instances are not JSON-serializable and would
cause "state_columns value for 'trajectory' is not JSON-serializable".

⚠️  If someone changes the event storage to use CompactionEventWire
   objects directly, the ZMQ round-trip silently drops them.
```

#### How to diagnose

Add logging in `train.py` right after the event extraction:

```python
# After line 494:
compaction_events = micro_batch.get("compaction_events") or []
if config.compaction.window_size > 0:
    logger.warning(
        f"[DIAG] events={len(compaction_events)} "
        f"seq_len={input_ids.shape[1]} "
        f"prompt_len={micro_batch.get('prompt_len')}"
    )
```

If `events=0` on every micro-batch despite the inference server running
with compaction, events are being lost. Work backward through the four
stages to find the break.

Also verify at the vLLM response level:

```python
# In env.py patched_from_native, after extracting raw_events:
logger.warning(f"[DIAG] from_native_response: events={raw_events is not None}")
```

### Failure mode 2: Per-turn truncation breaking coordinate alignment

`parse_response_tokens` in verifiers truncates `completion_ids` when
`prompt_len + completion_len > max_seq_len`, but does NOT truncate
compaction events (which live in the response, not the token lists).

```
vLLM returns for one turn:
  completion_ids:     1200 tokens (full generation)
  compaction_events:  boundaries up to raw=1200

parse_response_tokens with max_seq_len=16384, prompt_len=15300:
  completion_ids:     1084 tokens (TRUNCATED to 16384-15300)
  compaction_events:  boundaries up to raw=1200 (NOT truncated)

Result: events reference positions beyond the truncated completion.
```

This doesn't crash in single-turn (no merge assertion). Instead, the
trainer sees events pointing past the end of `input_ids`. In
`segmented_forward`, the segment construction clamps:

```python
seg_end = min(prompt_len + boundary, seq_len)
```

But the KV eviction still happens at `prompt_aligned_len`, which may now
be wrong relative to what inference actually evicted — because the token
sequence itself is shorter.

**More critically**: the `completion_logprobs` are ALSO truncated by
`parse_response_tokens` (line 53), so the `inference_logprobs` in the
training sample don't cover the full generation. Any segment boundary
pointing into the truncated region produces a logprob comparison against
the wrong tokens.

#### When this triggers

- `max_seq_len` is set on the env by prime-rl (`orchestrator.py:1077`):
  `env.extra_env_kwargs.update(max_seq_len=self.seq_len)`
- With `seq_len=16384`, `max_completion_tokens=15000`, and prompt >1384
  tokens, per-turn truncation fires.
- Single-turn with short prompts (~200 tokens): **does NOT trigger**.
- Multi-turn with growing prompts: **triggers frequently**.

#### The fix

The uncommitted change to `env.py` adds `_disable_per_turn_truncation`,
which sets `env.max_seq_len = None` BEFORE `parse_response_tokens` runs:

```python
async def patched_add_model_response(self, state, prompt_messages, response):
    _disable_per_turn_truncation(self)          # ← sets self.max_seq_len = None
    await orig_add_model_response(self, ...)    # ← calls parse_response_tokens(response, None)
    # ... attach events ...
```

With `max_seq_len=None`, `parse_response_tokens` skips truncation
entirely. Final seq_len clamping is handled by `prepare_sample` in
`batch.py`, which correctly clamps BOTH tokens AND events via
`_clamp_compaction_events`.

**Ensure this fix is deployed.** If running from the committed code
(without the `_disable_per_turn_truncation` change), multi-turn
compaction runs will hit this.

### Failure mode 3: Multi-turn event offset/merge producing wrong boundaries

The uncommitted change to `trajectories.py` replaces a
`NotImplementedError` with actual multi-turn event merging:

```python
generation_offset = current_completion_len + new_prompt_ext_len
offsetted = e.num_output_tokens_at_compaction + generation_offset
```

If `generation_offset` is wrong, ALL segment boundaries for subsequent
turns shift. The trainer's KV eviction happens at different positions
than inference's → different retained context → different logprobs.

This is new code with diagnostic logging (`[DIAG-MERGE]`). If your
mismatch KL correlates with multi-turn rollouts, check these logs.

## Summary: which failure mode are you hitting?

| Symptom | Likely cause |
|---------|-------------|
| `events=0` on every micro-batch in trainer | **FM1**: Events lost in pipeline |
| `[DIAG] MISMATCH` in trajectories.py logs | **FM2**: Per-turn truncation coordinate mismatch |
| KL spikes on multi-turn rollouts specifically | **FM2** or **FM3**: Truncation or merge bug |
| KL elevated uniformly across all samples | **FM1**: No events reaching trainer at all |
| KL correct on short rollouts, wrong on long ones | **FM2**: Truncation only fires on long sequences |

## Diagnostic checklist

1. **Are events reaching the trainer?**
   Add the `[DIAG]` log in `train.py:494`. If `events=0` always, go to step 2.

2. **Are events surviving from_native_response?**
   Add logging in Patch 1. If events are `None` at the native response level,
   the openai SDK / `monkey_patch_chat_completion_logprobs` interaction is
   stripping them.

3. **Are events in the trajectory step extras?**
   Log `step["extras"].get("compaction_events")` in `interleave_rollout`.
   If `None`, Patch 2 isn't firing or events aren't surviving ZMQ
   serialization.

4. **Is per-turn truncation creating coordinate mismatch?**
   Check for `[DIAG] MISMATCH` in trajectories.py logs. If present, deploy
   the `_disable_per_turn_truncation` fix from the uncommitted env.py diff.

5. **Are multi-turn offsets correct?**
   Check `[DIAG-MERGE]` logs. Verify `generation_offset` makes sense:
   it should equal `sum(all previous completion + extension tokens)`.

## The data flow when everything works

```
vLLM inference (compaction active)
  │
  ├─ Token K sampled under FULL context    → logprob_K stored
  ├─ ...
  ├─ Token 3896 sampled under FULL context → logprob_3896 stored
  │
  │  ← COMPACTION: evict 512 tokens from KV →
  │  CompactionEvent(num_output_tokens_at_compaction=3897, ...)
  │
  ├─ Token 3897 sampled under REDUCED context → logprob_3897 stored
  ├─ ...
  └─ Token N sampled under REDUCED context    → logprob_N stored
  │
  v
ChatCompletionResponse
  │  token_ids: [tok_0 ... tok_N]        ← ALL tokens (never trimmed by compaction)
  │  logprobs:  [lp_0 ... lp_N]          ← ALL logprobs (pre + post compaction)
  │  compaction_events: [{boundary=3897, evicted=512, ...}]
  │
  v
Monkey-patch pipeline (env.py)
  │  Patch 1: preserve events on verifiers Response
  │  Patch 2: copy events to trajectory step extras
  │
  v
TrainingSample
  │  prompt_ids + completion_ids = full sequence
  │  inference_logprobs = [0.0]*prompt + [lp_0 ... lp_N]
  │  compaction_events = [{boundary=3897, ...}]
  │
  v
Trainer (train.py → segmented_forward)
  │
  │  Segment 0: forward on input_ids[0 : prompt_len + 3897]
  │    → trainer logprobs match lp_0..lp_3896 (full context) ✓
  │
  │  KV eviction: drop [prompt_aligned_len, prompt_aligned_len + 512)
  │
  │  Segment 1: re-feed boundary token + input_ids[prompt_len + 3896 :]
  │    → trainer logprobs match lp_3897..lp_N (reduced context) ✓
  │
  │  mismatch_kl ≈ 0.0009 (kernel floor) ✓
```
