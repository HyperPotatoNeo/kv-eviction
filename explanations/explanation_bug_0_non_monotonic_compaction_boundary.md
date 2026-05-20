# Bug 0: Non-Monotonic Compaction Boundary

## Error

```
AssertionError: Non-monotonic compaction boundary at step 8:
  offsetted=55 < prev=2870 (raw=1, generation_offset=54)
```

Location: `prime-rl/src/prime_rl/orchestrator/trajectories.py:486`

## What does `interleave_rollout` do?

A multi-turn BALROG game produces a trajectory of N steps. Each step is
an independent vLLM request (fresh prompt, fresh generation). The
orchestrator needs to stitch these into ONE flat `TrainingSample` for the
trainer. It does this in `interleave_rollout`.

Each step produces:
- `prompt_ids`: the tokenized chat history sent to vLLM for this turn
- `completion_ids`: the tokens vLLM generated for this turn
- `compaction_events`: metadata about KV cache evictions during this turn

The merge works by exploiting the **extension property**: turn N+1's
prompt is a prefix-extension of turn N's prompt+completion. So instead
of duplicating shared tokens, we just append the NEW part.

### The flat training sample layout

```
TrainingSample.prompt_ids:
  [  turn 0 prompt  ]

TrainingSample.completion_ids:
  [turn 0 completion] [ext₁] [turn 1 completion] [ext₂] [turn 2 completion] ...

Where extₖ = the new tokens in turn k's prompt beyond the previous prefix
             (the environment observation, assistant reply template, etc.)
```

## What are compaction events and why do they need offsetting?

When vLLM's KV cache exceeds `window_size` during generation, it evicts
`stride` tokens from the middle of the sequence. Each eviction produces
a `CompactionEvent`:

```python
# scheduler.py:1091-1096 — recorded BEFORE trimming
CompactionEvent(
    num_output_tokens_at_compaction=request.num_total_generated,
    tokens_evicted=total_evicted,
    ...
)
```

`num_output_tokens_at_compaction` tells the trainer: "at the point in
generation when this eviction happened, the model had produced THIS MANY
output tokens." The trainer uses this to know where to split the sequence
into segments for `segmented_forward`.

**But each turn is a separate vLLM request.** Turn 5's event says
"eviction fired at output token 200" — meaning token 200 *within turn 5*.
The trainer sees the flat merged sequence. So the orchestrator must
offset: "token 200 of turn 5 = token 1847 of the merged sequence."

### The offset formula (trajectories.py:460-466)

```python
current_completion_len = len(sample.completion_ids)   # merged so far
new_prompt_ext_len = len(tokens["prompt_ids"]) - prefix_len  # new prompt tokens
generation_offset = current_completion_len + new_prompt_ext_len

# Then for each event:
offsetted = e.num_output_tokens_at_compaction + generation_offset
```

`generation_offset` = "where does this turn's generation START in the
merged sample's completion_ids?" It's the length of everything already
merged, plus the prompt extension tokens about to be appended.

### The monotonicity assertion (trajectories.py:484-490)

```python
if existing:
    prev = existing[-1].num_output_tokens_at_compaction
    assert offsetted >= prev
```

This checks: each offsetted event boundary must be >= the previous one.
If turn N+1's first event lands BEFORE turn N's last event in merged
coordinates, the segmented_forward segments would overlap = garbage.

## The three data pipelines and the disconnect

Here's the key: **compaction events and completion tokens flow through
different code paths** and can get out of sync.

### Pipeline 1: completion_ids (tokens)

```
vLLM detokenizer           Each turn is a fresh request. The detokenizer
  .token_ids               accumulates ALL output tokens for this turn.
  (never trimmed            It is never aware of compaction.
   by compaction)           len(detokenizer.token_ids) == num_total_generated
       |
       v
Output Processor           In non-delta mode (line 400-401):
  token_ids = detokenizer     token_ids = self.detokenizer.output_token_ids
  .output_token_ids           Returns the FULL untrimmed list.
       |
       v
ChatCompletion Response    choices[0].token_ids = full list
  (API wire format)          e.g. 1285 tokens for this turn
       |
       v
Verifiers parse_tokens()   completion_ids = response.choices[0].token_ids
  (openai_chat_completions   Still 1285 tokens. No truncation here.
   _client.py:446)
       |
       v
Verifiers ResponseTokens   completion_ids: list[int]  — stored as-is
  (types.py:163)
       |
       v
*** parse_response_tokens ***     <<<<< THE TRUNCATION SITE
  (response_utils.py:49-53)
                                  if prompt_len + completion_len > max_seq_len:
                                      completion_ids = completion_ids[:max_seq_len - prompt_len]
                                      completion_logprobs = completion_logprobs[:max_seq_len - prompt_len]
                                      # e.g. prompt=3069, max_seq_len=4098
                                      # completion_ids truncated from 1285 → 1029
       |
       v
TrajectoryStep.tokens      completion_ids: now 1029 tokens (TRUNCATED)
  (multiturn_env.py:127)     This is what interleave_rollout reads.
       |
       v
interleave_rollout         Uses len(completion_ids) to compute offsets.
  (trajectories.py)          All offset math is in "truncated" coordinates.
```

### Pipeline 2: compaction_events (metadata)

```
vLLM scheduler             CompactionEvent recorded at compaction time:
  (scheduler.py:1091)         num_output_tokens_at_compaction = request.num_total_generated
                              e.g. 1285 (the FULL untrimmed count)
       |
       v
ChatCompletion Response    response.compaction_events = [...]
  (serving.py:1616-1624)     Values are the raw num_total_generated values.
       |
       v
env.py monkey-patch        Copies events from response → step.extras
  (env.py:256-263)           Verbatim. No truncation.
       |
       v
TrajectoryStep.extras      extras["compaction_events"] = raw dicts
  ["compaction_events"]       Values still reference num_total_generated = 1285
       |
       v
interleave_rollout         Reads events and offsets them.
  (trajectories.py:297)      Raw values are in "untrimmed" coordinates.
```

**The disconnect**: `completion_ids` is in "truncated" space (1029
tokens), but event raw values are in "untrimmed" space (up to 1285).
The offset formula uses `len(completion_ids)` (truncated), but compares
against event values (untrimmed).

## Concrete example from the reproduced crash

### Setup

```
Config: seq_len=4098 (= max_seq_len), window=512, stride=256, protected_prefix=256
Environment: BALROG babyai, max_turns=30, max_text_history=16
```

### A specific rollout across turns

#### Turn 0 (step 0): game starts

```
prompt = [system prompt + initial observation] = 422 tokens
vLLM generates 55 completion tokens (a short action like "go forward")
No compaction (422 + 55 = 477 < window 512)

TrajectoryStep.tokens:
  prompt_ids:     [422 tokens]
  completion_ids: [55 tokens]

TrainingSample after step 0:
  prompt_ids:     [422 tokens]
  completion_ids: [55 tokens]   ← current_completion_len = 55
```

#### Turns 1-6 (steps 1-6): prompts grow, compaction starts

Each turn adds an observation + action to the chat history, making the
prompt longer. By turn 2, `prompt + completion > 512`, so compaction
starts evicting tokens during generation.

Each turn's `completion_ids` is short (2-80 tokens for BabyAI actions).
But each step gets compaction events because the inherited prompt is
already above the window — compaction fires immediately on the first
generated token.

```
Step 2: prompt=462, completion=1764 (long exploration step)
  completion_ids: 1764 tokens
  compaction_events: 7 events, max raw = 1588

Step 3: prompt=2227, completion=5 tokens
  compaction_events: 7 events, ALL with raw=1
  (compaction fires 7 times at the very first output token because
   the prompt alone is 2227 >> window 512)
```

**Key**: `raw=1` means "compaction fired when num_total_generated=1",
i.e., after generating just the first token. This is the
protected-prefix prompt eviction — the prompt is so long that every
new token immediately exceeds the window, triggering eviction of old
prompt tokens (between position 256 and prompt end).

#### Step 7: the merged sample grows

```
TrainingSample.completion_ids has accumulated:
  [turn 0 compl] [ext₁] [turn 1 compl] ... [ext₆] [turn 6 compl]
  current_completion_len = 54  (short turns, small extensions)
```

#### Step 7's compaction events are added to the merged list

```
prev = existing[-1].num_output_tokens_at_compaction
     = some value from step 6's events (small, since offsets are small)

step 7's event: raw=1, generation_offset = 54 + 4 = 58
  offsetted = 1 + 58 = 59
  59 >= prev  → OK, monotonicity holds
```

This works because both the offsets (from len(completion_ids)) and the
event raw values are small.

#### Step 7 with a LONG generation: the problem starts

Now consider a different rollout where one turn generates many tokens:

```
Turn 7: prompt=3000 tokens, vLLM generates 1200 completion tokens
  Compaction fires repeatedly during generation.
  Last event: num_output_tokens_at_compaction = 1200

  But: prompt + completion = 3000 + 1200 = 4200 > max_seq_len (4098)

  parse_response_tokens truncates:
    completion_ids = completion_ids[:4098 - 3000] = completion_ids[:1098]

  TrajectoryStep.tokens.completion_ids: 1098 tokens  (TRUNCATED from 1200)
  TrajectoryStep.extras.compaction_events: max raw = 1200  (NOT truncated)
```

When this step gets merged:

```
generation_offset = current_completion_len + new_prompt_ext
                  = 54 + 4 = 58

For the last event of step 7:
  offsetted = 1200 + 58 = 1258
  This gets stored as existing[-1].num_output_tokens_at_compaction = 1258
```

But extend_sample appends only 1098 completion tokens (the truncated
amount):

```
sample.completion_ids grows by 1098  (not 1200)
After extend: current_completion_len = 54 + 4 + 1098 = 1156
```

#### Step 8: the crash

```
Turn 8: prompt=4098 tokens (overlong!), vLLM generates 63 tokens
  parse_response_tokens: prompt >= max_seq_len → completion_ids = []
  But events still have raw values up to 63.

generation_offset = current_completion_len + new_prompt_ext
                  = 1156 + 0 = 1156
                  (new_prompt_ext=0 because prefix_len=4098=len(prompt_ids))

But wait — the ACTUAL diagnostic showed:
  current_completion_len = 54
  new_prompt_ext_len = 0
  generation_offset = 54
  prefix_len = 4098

For step 8's first event:
  offsetted = 1 + 54 = 55
  prev = 2870  (from a previous step's event)

  55 < 2870  → ASSERTION FAILS
```

The gap of 2815 tokens = all the completion tokens that were TRUNCATED
by `parse_response_tokens` across earlier steps but whose compaction
events were NOT truncated.

## Why this only happens with multi-turn + compaction + long prompts

Three conditions must ALL be true:

1. **Multi-turn**: multiple steps merged into one sample (so the offset
   logic runs)

2. **Compaction active**: events exist that reference `num_total_generated`

3. **prompt + completion > max_seq_len**: `parse_response_tokens`
   truncates `completion_ids` but NOT the compaction events

Without (3), `len(completion_ids) == num_total_generated` for every turn,
and the offset formula works perfectly. The bug appears when BabyAI's
growing conversation history pushes prompts toward `max_seq_len`,
triggering the truncation that breaks the coordinate alignment.

## Summary diagram

```
         vLLM response for one turn
        ┌──────────────────────────┐
        │ token_ids: 1200 tokens   │──── from detokenizer (never trimmed)
        │ events: raw up to 1200   │──── from num_total_generated
        └──────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │ parse_response_tokens │
        │  prompt=3000          │
        │  max_seq_len=4098     │
        │  3000+1200 > 4098     │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    TRUNCATED               NOT TOUCHED
        │                       │
  completion_ids:          compaction_events:
  1098 tokens              raw up to 1200
        │                       │
        └───────────┬───────────┘
                    │
                    v
          interleave_rollout
          ==================
          offset = len(completion_ids) = based on 1098
          event raw = 1200
          → offset is 102 tokens SHORT
          → next turn's offset starts 102 tokens behind
          → eventually: offsetted < prev → ASSERT
```

## Fix direction

The events and the token lists must be in the same coordinate space.
Three options:

**Option A**: Filter/clamp compaction events to `len(completion_ids)` in
`interleave_rollout` or in the step preparation. Events beyond the
truncated boundary refer to tokens that aren't in the training sample
anyway. This is safe because the trainer only needs to replay compaction
for the tokens it actually trains on.

**Option B**: Truncate compaction events alongside tokens in the env
monkey-patch (`env.py:attach_compaction_events_from_response`). After
`parse_response_tokens` runs, filter events where
`num_output_tokens_at_compaction > len(completion_ids)`.

**Option C**: Don't truncate tokens when compaction is active. Skip the
`max_seq_len` truncation in `parse_response_tokens` for compaction runs,
or increase `seq_len` to be large enough that truncation never triggers.
Risk: OOM in the trainer if sequences are very long.
