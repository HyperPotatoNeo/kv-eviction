# Plan: Block-Aligned Message Padding via prime-rl Patch

## Status: Draft / Follow-up to `block_aligned_message_padding.md`

**Starting commit:** `014179c` (branch `eviction-turn-based`) —
"WIP: protected-prefix compaction + turn-based eviction plan". The
vLLM submodule at this point carries the in-progress turn-mode
implementation (cache/arg_utils/scheduler/request changes) on top of
its `compaction` branch at `c23308919`. Implementation work for this
plan should branch from here.

This plan is an alternative realization of the padding idea in
`block_aligned_message_padding.md` — instead of applying padding
*server-side* in vLLM's chat-completion serving code, we apply it
*client-side* in a prime-rl / kv-eviction monkey-patch, mirroring the
existing compaction-events monkey-patch at
`src/kv_eviction/env.py:249`. vLLM is left untouched.

## Motivation

The prototype in `experiments/debug_balrog/compaction_test.ipynb`
(cells ~144–296, `_render_padded`) already demonstrates that:

- Rendering chat messages via `tokenizer.apply_chat_template(..., tokenize=True)`
  to get raw token ids,
- Inserting `pad_n` filler tokens **before** each `<|im_end|>` so the
  `<|im_end|>` occupies the last slot of a block (i.e. the position
  right after `<|im_end|>` is a multiple of `block_size`),
- Feeding the padded token list to vLLM via `prompt_token_ids=…`,

works cleanly with turn-mode compaction and produces block-aligned turn
boundaries. Observed in the notebook log: `raw=368 -> padded=372 (+4)`
per turn-0, `pads=[3,1]` — i.e. 3 filler tokens to land the system
`<|im_end|>` at the end of block 22 (368-token boundary → pad to 368,
`<|im_end|>` at slot 367, next token at 368 — a clean block edge).

The scheduler then evicts `[336, 688)` and `[336, 608)` — exact
multiples of 16 at both ends, matching the turn boundaries. No orphan
fragments.

The remaining open question is how to port this prototype from a local
`LLM(...).generate(prompts=[{"prompt_token_ids": …}])` call into the
prime-rl rollout path, which today talks to vLLM's OpenAI-compatible
server via `client.chat.completions.create(messages=…)` (no
pre-tokenized path). This plan lays out that port.

## Invariant & constraints (inherited)

These are identical to `block_aligned_message_padding.md` — the plan
fails if they are violated:

1. **Inference and training MUST use identical padded token streams.**
   If the trainer reconstructs the sample from un-padded messages, the
   per-token alignment is off by `sum(pad_n)` and gradients land on
   wrong positions.
2. **Padding tokens must be ignored in the loss.** Training-side mask
   must zero them out — they are scaffolding, not learned content.
3. **Padding must not bias generation.** Filler tokens are inserted
   only in the prompt, *before* `<|im_end|>` tokens that have already
   been committed (either by the chat template for past turns, or by
   the sampler for completed assistant messages). The sampler is never
   forced to predict a filler token.
4. **`block_size` must be known at chat-template apply time.** The
   orchestrator must read it from the trainer-side compaction config
   (or a new orchestrator-side knob).

## Where to hook

The current flow (see the `Explore` map from pre-plan research):

```
orchestrator/trajectories.py:74-87
        │  _render_messages() for initial prompt (unused here — verifiers
        │  re-renders anyway for /v1/chat/completions)
        ▼
verifiers.envs.multiturn_env:88-97  get_prompt_messages()
        │  builds per-turn messages list (role/content dicts)
        ▼
verifiers.clients.openai_chat_completions_client:280,288
        │  client.chat.completions.create(messages=prompt, ...)
        ▼
vLLM /v1/chat/completions  (server applies chat template → token ids)
```

The existing monkey-patch in `src/kv_eviction/env.py:249-309` already
demonstrates the pattern: intercept `OpenAIChatCompletionsClient`
methods at import time, wrap them, keep the rest of verifiers
untouched. We add a **third patch**: wrap `get_response` (or the
narrower `_make_chat_request` used on line 280/288) so that when
padding is enabled:

1. Apply the chat template locally to get raw token ids.
2. Run `_render_padded` → padded token ids + per-`<|im_end|>` pad counts.
3. Send the padded ids to vLLM using a pre-tokenized endpoint (see
   "Transport" below).
4. On the response, carry the padded `prompt_token_ids` + pad metadata
   back to the trajectory step so the trainer can see them.

## Phasing

Phase 1 = inference-only. Phase 2 = training-side alignment. They land
independently; Phase 1 is testable without Phase 2.

### Phase 1 — Orchestrator/env-side padding (no trainer changes)

Goal: every request to vLLM carries padded `prompt_token_ids` so turn
boundaries land on block edges. Gate on a new orchestrator config flag,
default off → bit-for-bit no-op.

| File | Change |
|---|---|
| `src/kv_eviction/padding.py` *(new)* | Port `_render_padded` + `_filler_token_id` from the notebook. Pure function: `(messages, tools, tokenizer, block_size, im_end_id, filler_id) -> (raw_ids, padded_ids, pads, render_kwargs)`. Single source of truth. |
| `src/kv_eviction/env.py` | Add a third monkey-patch: wrap `OpenAIChatCompletionsClient.get_response` (or the inner call on line 280/288) to do the padding + pre-tokenized dispatch when `KV_EVICTION_PAD_MESSAGES=1` (env) or the orchestrator has set it via a shared module-level flag. Also stash `prompt_token_ids` + `pads` on the response for the downstream `from_native_response` patch. |
| `src/kv_eviction/env.py` (existing patch 1) | Extend `patched_from_native` to also forward the padded `prompt_token_ids` and `pads` onto the verifiers response (same extras mechanism already used for `compaction_events`). |
| `prime-rl/src/prime_rl/orchestrator/config.py` | Add `pad_messages_to_block: bool = False`, `block_size: int = 16`, `filler_token_id: int | None = None` (None → auto via `tokenizer.pad_token_id`, Qwen3 → 151643). Validation: if true, require the trainer-side flag `[trainer.compaction].message_padding_aware = True` (added in Phase 2). |
| `prime-rl/src/prime_rl/orchestrator/trajectories.py` | When padding is enabled, carry the padded `prompt_token_ids` from the first step's extras into `TrainingSample.prompt_token_ids` (overriding the orchestrator-side re-tokenization path). This keeps the trainer's per-token ground truth byte-identical to what vLLM saw. |

### Phase 2 — Trainer-side padding awareness

Goal: a compaction-enabled training run can consume padded samples
end-to-end without computing gradients on filler tokens.

| File | Change |
|---|---|
| `prime-rl/src/prime_rl/trainer/rl/config.py` | Add `message_padding_aware: bool = False` and `filler_token_id: int | None = None` to `trainer.compaction`. Validator: must match orchestrator. |
| `prime-rl/src/prime_rl/trainer/rl/data.py` (or wherever `_micro_batch_to_tensor` builds the loss mask) | Build `is_padding = (input_ids == filler_id)`, AND it into the existing loss mask. Padded positions contribute zero gradient. |
| `src/kv_eviction/segmented_forward.py` | No eviction-path changes — the existing `prompt_aligned_len` computation still works because the padded prompt is what vLLM actually evicted from, and `num_prompt_tokens` on the event is the post-padding prompt length. The loss mask is applied by the caller; segmented_forward itself is loss-agnostic. Add a sanity assertion that any `compaction_events[0].eviction_start` (if/when that field lands) matches the block-aligned boundary we'd compute from the padded prompt. |
| `plans/block_aligned_message_padding.md` — trainer section | Merge / supersede with this one. The vLLM-side patch plan there becomes deprecated in favor of this client-side approach. |

Phase 2 is **required** before enabling padding in a compaction RL
training run. Phase 1 alone is safe for inference-only smokes (the
notebook prototype is effectively Phase 1).

## Transport: how to send pre-tokenized input over the OpenAI API

The crux of the port. Verifiers uses
`client.chat.completions.create(messages=…)` — OpenAI's schema, no
`prompt_token_ids` field. Three options:

### Option A: `extra_body={"prompt_token_ids": [...]}` on chat completions *(recommended, pending confirmation)*

vLLM's OpenAI-compatible server accepts extra fields on
`ChatCompletionRequest` — check `vllm/entrypoints/openai/protocol.py`
for whether `prompt_token_ids` is already accepted or can be added as a
small server-side patch. If supported, when set, vLLM skips chat
template application and uses the provided ids verbatim. Client code:

```python
response = await self.client.chat.completions.create(
    model=model,
    messages=prompt,              # ignored when prompt_token_ids is set
    extra_body={"prompt_token_ids": padded_ids},
    **normalize_sampling_args(sampling_args),
)
```

**Pros:** cleanest port — response shape (roles, tool_calls,
finish_reason) is preserved, compaction_events machinery unchanged.
**Cons:** requires verifying / patching vLLM's chat-completion schema
to accept this extra. If not supported natively, this becomes a small
vLLM-side change (still much smaller than the full server-side padding
plan).

### Option B: Route to `/v1/completions` instead of `/v1/chat/completions`

The completions endpoint natively accepts `prompt` as a list of ints.
We tokenize+pad client-side, send via `client.completions.create(...)`,
then reconstruct the chat-shaped response envelope for verifiers.

**Pros:** no vLLM changes at all.
**Cons:** loses `tool_calls` parsing (BALROG BabyAI uses the hermes
tool-call parser — we'd have to replicate that parsing client-side).
Response-shape adapter is non-trivial. Likely rules this out for BALROG
in the short term.

### Option C: Send padded text (decode back to string) via `messages`

Decode padded tokens → text → send as a single `user` message with the
full rendered prompt. Relies on vLLM re-tokenizing the text deterministically.

**Pros:** zero server changes.
**Cons:** filler tokens may not round-trip cleanly through `decode →
encode` (esp. `<|endoftext|>` as filler). And we lose the role/tool
structure in the request (vLLM won't apply chat template anyway since
we'd be sending pre-rendered text, but also can't guarantee
tokenization matches). Fragile. Not recommended.

**Decision:** Default to Option A, with a fallback implementation of
Option B for environments where the vLLM fork does not (yet) accept
`prompt_token_ids` on chat completions. Confirm Option A viability as
the first implementation step (a 5-minute scan of
`vllm/entrypoints/openai/protocol.py` and `serving_chat.py`).

## Carrying padded ids through to the trainer

vLLM already echoes the resolved `prompt_token_ids` back on the
`ChatCompletion` object (under `choices[0].prompt_token_ids` or in an
extra field, depending on version). We piggyback on the existing
monkey-patch:

1. `patched_get_response` (new) attaches the padded ids it generated
   locally to the response it returns.
2. `patched_from_native` (existing) reads that field and stashes it on
   the verifiers response in `extras`.
3. `patched_add_model_response` (existing) copies from the response
   into the trajectory step's extras.
4. `orchestrator/trajectories.py` reads step extras and writes
   `TrainingSample.prompt_token_ids` (first-step only).

The compaction_events pipeline is unchanged — pads and events travel
together on the same response.

## Validation

### Phase 1 (inference only)
- Re-run `experiments/debug_balrog/compaction_test.ipynb` pointed at
  the OpenAI-compatible server rather than the local `LLM` instance,
  with `pad_messages_to_block=True`. The server logs should show
  `evict=[336,688)` / `evict=[336,608)` etc. — the same block-exact
  boundaries the notebook's local path already produces.
- Confirm assistant output quality is indistinguishable from the
  notebook's local-LLM padded run (no `"I new observation"` fragments,
  no `"TheThe"`, etc.).

### Phase 2 (end-to-end RL)
- 5-step RL smoke on `experiments/debug_balrog/rl.toml` with
  `pad_messages_to_block=True` + `compaction_max_turns=4`,
  `compaction_eviction_turn_stride=2`. Mismatch KL must stay at kernel
  floor (~1e-3). This is the direct analogue of the current failing
  KL=1.27 case once Bug 2's auto-detect is fixed OR once padding
  eliminates the need for auto-detect entirely (see below).
- Logging hook: per-batch, count positions where `is_padding` is true
  and assert loss contribution on those positions is exactly 0.

## Interaction with Bug 2 (`explanation_bug_2_*.md`)

Bug 2 is the `prompt_aligned_len` mismatch between trainer (400) and
vLLM (368) in protected-prefix auto-detect mode. The root cause is that
vLLM's eviction starts at the block-aligned auto-detected system-prompt
boundary, but the trainer's fallback computes a different block
boundary from the turn-0 prompt length.

Padding eliminates this bug by construction: with
`pad_messages_to_block=True`, every `<|im_end|>` (including the end of
the system prompt) lands on a block boundary. The trainer's
`ceil(mb_prompt_len / 16) * 16` formula and vLLM's
`ceil(effective_prompt / 16) * 16` formula both return the same
physical slot, because after padding `mb_prompt_len ≡ effective_prompt
(mod block_size)` — they differ only by whole blocks of filler that
both sides see identically.

So Phase 2 of this plan is a **harder but structurally cleaner** fix
for Bug 2 than adding `eviction_start` to `CompactionEvent`. If we ship
padding, the Bug 2 wire-format change becomes optional (nice-to-have
for defense-in-depth; not required for correctness).

## Open decisions (flag before implementing)

1. **Option A viability.** Does vLLM's
   `ChatCompletionRequest` accept `prompt_token_ids` as an extra? If
   not, is the small server-side patch to allow it acceptable, or do
   we commit to Option B (completions endpoint + hermes tool-call
   reparsing) instead? This determines whether the patch is truly
   client-side-only.
2. **Where to read `block_size` on the orchestrator side.** Currently
   `inference.vllm_extra.block_size` is server-owned. Options:
   duplicate it as `orchestrator.compaction.block_size` (explicit,
   risks drift), or plumb it from the `inference` config section.
   Recommend: add a cross-validator in `trainer/rl/config.py` that
   asserts the three copies (orchestrator, trainer, inference) agree.
3. **Filler token choice.** Notebook uses `tokenizer.pad_token_id`
   (Qwen3 → 151643 = `<|endoftext|>`). That works for inference (model
   conditions on it harmlessly), but the trainer-side loss mask has to
   distinguish "real `<|endoftext|>` in the data" from "filler" — they
   share the id. Alternatives:
   - Use a dedicated unused token id (requires probing the tokenizer
     for a safe one).
   - Rely on the assumption that `<|endoftext|>` never appears inside
     a real turn (true for BALROG chat traffic, probably true in
     general for chat models).
   - Plumb a per-sample `is_padding` mask through the wire instead of
     recomputing it from token id (cleaner but a wire-format change).
4. **Streaming vs non-streaming.** Verifiers uses non-streaming
   completions by default. Confirm no streaming path exists for the
   BALROG pipeline before committing to a non-streaming-only patch.
5. **Prefix caching.** Padding is deterministic given
   `(messages, tools, block_size, filler_id)`, so APC hashes are
   stable. Unit-test this — a 2-request smoke with identical prompts
   must hit the cache.

## Order of operations

1. Confirm **Option A** (scan vLLM's chat completion protocol; 5 min).
2. Implement `src/kv_eviction/padding.py` as a straight port of
   `_render_padded` + `_filler_token_id` from the notebook.
3. Add the third monkey-patch to `env.py` behind a default-off flag.
4. Validate Phase 1 against the notebook (server-side vs. local
   generate should produce byte-identical padded prompts and
   identical compaction event streams on a fixed seed).
5. Land Phase 2 (trainer loss mask + config). Gate the RL smoke on
   Mismatch KL < 0.005.
6. Once Phase 2 is green, deprecate the server-side plan in
   `block_aligned_message_padding.md` (or mark superseded).

## Files touched (summary)

New:
- `src/kv_eviction/padding.py`
- `plans/prime_rl_message_padding_patch.md` (this file)

Modified:
- `src/kv_eviction/env.py` — new patch #3 wrapping `get_response`,
  extend patch #1 to forward padded ids.
- `prime-rl/src/prime_rl/orchestrator/config.py` — new knobs.
- `prime-rl/src/prime_rl/orchestrator/trajectories.py` — carry padded
  ids into `TrainingSample`.
- `prime-rl/src/prime_rl/trainer/rl/config.py` — `compaction.message_padding_aware`,
  `compaction.filler_token_id`; cross-validation.
- `prime-rl/src/prime_rl/trainer/rl/data.py` — loss-mask AND with
  `is_padding`.

Not modified (vs. the server-side plan's scope):
- vLLM chat template / serving code **unless** Option A needs a
  one-liner to whitelist `prompt_token_ids` on `ChatCompletionRequest`.
- `src/kv_eviction/segmented_forward.py` — no eviction-path changes
  (padding only shifts where the boundary lands, not how eviction is
  computed).
- `vllm/v1/core/compaction/*` — untouched.
