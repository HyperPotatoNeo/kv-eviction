# Plan: Block-Aligned Message Padding for Exact Turn Eviction

## Status: Draft / Follow-up to `turn_based_eviction.md`

## Motivation

Turn-based compaction (see `turn_based_eviction.md`) evicts whole
user+assistant pairs from the KV cache. But vLLM's KV lives in fixed
16-token blocks and the block pool only frees whole blocks. If a turn
ends mid-block, we have to either:

- **Snap inward** — keep the straddling block alive. ~12% of an
  evicted turn lingers as orphan tokens at each block boundary. This
  is the choice in `turn_based_eviction.md`.
- **Snap outward + recompute** — free the straddling block and
  re-prefill the surviving tokens. Requires a new selective-prefill
  code path; pays cost on every compaction.
- **Avoid the problem upstream** — pad each chat message so that
  `<|im_end|>` always lands on a block boundary. This plan.

If turn mode in production shows the orphan fragments degrading model
output (e.g. `"I new observation"`-style artifacts surviving turn-mode
eviction at higher compaction rates), this plan is the cleaner fix.

## Idea

For each rendered chat message, append `pad_n` filler tokens after the
`<|im_end|>` so that the message length is a multiple of `block_size`:

```
pad_n = (-msg_len) % block_size      # 0..block_size-1
```

Pick a filler token id that:
1. The model has seen during training (or generalizes harmlessly to).
2. Is structurally invisible — e.g. another `<|im_end|>`, whitespace
   token, or a dedicated `<pad>` if the model has one.
3. Round-trips through the chat template without breaking the parser.

Most likely candidate for Qwen3: repeat `<|im_end|>` (id 151645).
The model already conditions on these, and a run of multiple
consecutive `<|im_end|>` tokens does not break the chat template
parser (the next `<|im_start|>` resets state).

Result: every turn boundary is block-aligned. Inward snap == exact
eviction. No orphan fragments. No recompute. No selective-prefill
kernel work.

## Cost

- **Wasted KV slots:** ~`block_size / 2` tokens per message on
  average. With `block_size=16` and 5 messages per turn (sys, U_1,
  A_1, U_2, A_2), that's ~40 wasted tokens per 5-message exchange,
  or ~8 tokens per message. Over a 30-turn BabyAI episode this is
  ~240 wasted KV slots — usually tolerable inside a 4096+ token
  window.
- **Distribution drift risk:** the model never saw padded ChatML
  during pretraining. We must keep the padded form on the training
  side too — otherwise inference and training diverge. This means
  every place that reconstructs a sample for training has to apply
  the same padding rule.

## Constraints

1. **Inference and training MUST use identical padded forms.** If the
   trainer reconstructs a sample without padding, the per-token loss
   alignment is off by `pad_n` per message and gradients are computed
   on the wrong positions. This is the load-bearing invariant.
2. **Padding token must be ignored in the loss.** Training-side mask
   must zero out the loss on the padding tokens. They are scaffolding,
   not learned content.
3. **Padding must not affect generation.** When the assistant is
   sampling its own message, padding only kicks in *after* the
   `<|im_end|>` is sampled. The sampler must not be biased toward
   producing extra `<|im_end|>` tokens because of padding.
4. **Block size must be known at chat-template apply time.** vLLM's
   block size is a CacheConfig field. We need to read it from the
   engine config and pass it into the chat-template renderer.

## Phasing

The work splits cleanly along the vLLM ↔ training boundary, and the
two halves can land independently (Phase 1 alone is shippable and
testable — it just can't be turned on for a compaction-enabled
training run yet).

### Phase 1 — vLLM-side padding (no trainer changes)

Goal: make the vLLM server able to emit padded prompts so that every
`<|im_end|>` lands on a block boundary. Behavior gated on a new
config flag, default off → bit-for-bit no-op compared to today.

| File | Change |
|---|---|
| `vllm/vllm/config/cache.py` | Add `pad_messages_to_block: bool = False`. Add to `ignored_factors`. |
| `vllm/vllm/engine/arg_utils.py` | Surface the new arg. Validate that it is only true when `compaction_max_turns > 0` (no point otherwise). |
| `vllm/vllm/entrypoints/openai/chat_completion/serving.py` | After applying chat template, post-process the token list: insert `pad_n` filler tokens after each `<|im_end|>`. Echo the padded `prompt_token_ids` back on the response so consumers can see the exact form. |
| `vllm/tests/v1/core/test_scheduler_compaction.py` | With padding enabled + `compaction_max_turns=2, stride=1`, fabricate a multi-turn stream and assert the surviving content (excluding padding) is byte-for-byte identical to the original messages with the evicted turns removed — no orphan fragments anywhere. |

Validation gate for Phase 1:
- Inference smoke (`experiments/debug_balrog/compaction_test.ipynb`)
  with `pad_messages_to_block=True, compaction_max_turns=4, stride=2`.
  Verify generation quality matches an uncompacted run on the same
  seed/episode. No trainer involvement.
- Existing block-FIFO and turn-based compaction tests still pass.

Phase 1 is **not safe** to use in a compaction-enabled training run
on its own. The trainer would receive padded `prompt_token_ids` but
not know to mask the padding tokens out of the loss → gradients on
filler positions. Either disable padding for training runs until
Phase 2 lands, or run Phase 1 only behind inference-only smokes.

### Phase 2 — Trainer/prime-rl padding awareness

Goal: make `segmented_forward` and the rest of the training pipeline
mask filler tokens out of loss/gradient computation, so a
compaction-enabled training run can use Phase 1 padding end-to-end.
Only kicks in when Phase 1 is enabled.

| File | Change |
|---|---|
| `src/kv_eviction/env.py` | Read padded `prompt_token_ids` directly from the vLLM response. Do NOT re-render — re-rendering would strip the padding. |
| `src/kv_eviction/segmented_forward.py` | Build a per-token `is_padding` mask from input_ids by matching the filler token id; multiply into the loss mask so padding positions contribute zero gradient. |
| `prime-rl/.../transport/types.py` | Optionally add `is_padding_mask` to the wire format (alternative: derive it on the trainer side from the filler token id, which avoids the wire change). |
| `prime-rl/.../trainer/batch.py` | Carry `is_padding` (or filler token id) into the batch object. |
| `prime-rl/.../orchestrator/trajectories.py` | Preserve padded form across trajectory merge — do NOT strip padding tokens on the way to the trainer. |
| `prime-rl/.../configs/trainer.py` | Validation: if vLLM is configured with `pad_messages_to_block=True` and compaction is enabled, the trainer must be aware of the filler token id. |

Validation gate for Phase 2:
- A 5-step RL smoke with `pad_messages_to_block=True` and turn-based
  compaction enabled. Mismatch KL must stay at kernel floor (~1e-3).
- Loss-curve sanity check: per-token loss on padding positions is
  exactly zero (verifiable via a logging hook).

Explicitly **not** changed in either phase:
- The compaction manager / scheduler eviction logic. With aligned
  message ends, the inward-snap formula in `turn_based_eviction.md`
  becomes exact automatically — no eviction-path edits needed.
- The `CompactionEvent` wire format.

## Open questions

1. **Filler token choice.** Repeated `<|im_end|>` is structurally safe
   but may push the model toward early-stopping if training sees them
   too often. Alternative: dedicated unused token id. Alternative:
   add a single `<|im_end|>` plus newline tokens. Need a small
   experiment to confirm Qwen3 doesn't degrade.
2. **Where to apply padding.** Server-side (chat template
   post-processing) is simplest but couples vLLM to the padding
   rule. Client-side (env builds padded messages before sending) is
   cleaner separation but requires the env to know `block_size`.
3. **Streaming output.** When the assistant is emitting tokens and
   the response is streamed back to the env, do we strip the
   training-side padding from the user-visible text, or surface it?
   Recommend: strip from the visible text channel; preserve in the
   `prompt_token_ids` echoed back so the trainer can see it.
4. **Interaction with prefix caching.** Padded prefixes still hash
   identically across requests (deterministic padding rule), so APC
   should work unchanged. Worth a unit test.

## Order of operations

1. Land `turn_based_eviction.md` first (inward-snap baseline, no
   padding). Confirm it improves output quality vs block FIFO.
2. If orphan fragments are still visible / harmful in production
   logs, implement this plan as the exact-eviction follow-up.
3. Otherwise, leave this plan unimplemented — the cost of touching
   training-side code may not be worth the marginal quality win.
