# SPDX-License-Identifier: Apache-2.0
"""
Block-aligned chat message padding.

Ports the `_render_padded` + `_filler_token_id` recipe from
`experiments/debug_balrog/compaction_test.ipynb` into a reusable module
used by the orchestrator-side monkey-patch in `env.py`.

## Why this exists

Turn-based KV eviction (see
`vllm/v1/core/compaction/manager.py::compact_request`) operates on whole
PagedAttention blocks of size `block_size`. When a turn boundary
(`<|im_end|>`) falls inside a block rather than at its last slot, the
scheduler has two options:

1. Inward-snap: evict only the blocks entirely before the last `<|im_end|>`
   of the turn. Wastes capacity; leaves orphan fragments of the just-
   evicted turn in the KV. Also complicates the trainer's
   `prompt_aligned_len` math (Bug 2).
2. Exact-edge: evict up to and including the block that ends the turn.
   Requires the position immediately AFTER the turn's `<|im_end|>` to
   be a multiple of `block_size`.

Padding filler tokens *after* each `<|im_end|>` so that the next turn
starts at a block boundary (i.e. position after `<|im_end|>` is
`k*block_size` for some k) makes option 2 possible by construction.
Both inference and training see the same padded ids, so the
kernel-floor KL gap stays zero.

The AFTER layout also preserves the natural position of `<|im_end|>`
within each message — the model's learned EOS behavior is untouched,
and `<|im_end|>`'s absolute position is stable across re-renders
(filler counts float in the inter-turn gap, not inside the message).

## Contract

- Input: `messages`, `tools`, `block_size`, `filler_id`, `im_end_id`,
  tokenizer. Optionally `add_generation_prompt` (default True).
- Output: `(raw_ids, padded_ids, per_im_end_pads)`.
- `padded_ids` has filler tokens inserted AFTER each `<|im_end|>` so
  that the position immediately following each filler run is a multiple
  of `block_size` (i.e. the next turn's first token starts at a block
  boundary). `<|im_end|>` itself sits at its natural position — the
  model's learned EOS behavior is unchanged. `add_generation_prompt`
  appends `<|im_start|>assistant\\n` AFTER the last `<|im_end|>`'s
  filler run; that trailing region is not padded (the in-flight turn
  is never evictable under turn-mode and will be re-padded on the next
  call once closed).

## Determinism

Given the same `(messages, tools, block_size, filler_id, im_end_id,
tokenizer)`, the output is byte-identical across processes — enables
prefix-cache hits on the vLLM server across rollout calls with the
same prefix.
"""

from typing import Any

# `tokenizer` is duck-typed here. The real object is a transformers
# PreTrainedTokenizerBase (the same instance prime-rl already uses for
# training). We avoid importing transformers at module load time.
Tokenizer = Any


def resolve_filler_token_id(
    tokenizer: Tokenizer,
    override: int | None,
) -> int:
    """Pick the filler token id used to pad message bodies.

    Resolution order:
        override -> tokenizer.pad_token_id -> encode(' ')[-1] -> <|endoftext|>

    For Qwen3 this resolves to `<|endoftext|>` (151643), matching the
    notebook prototype. See Q3 in
    `plans/prime_rl_message_padding_patch.md` for the collision-risk
    discussion: the trainer's `is_padding` mask uses `(input_ids == filler_id)`
    and would also mask any *real* `<|endoftext|>` in data. Accepted as
    vanishingly rare in chat data. If you need a zero-collision filler,
    override with a token id that cannot appear in real content (e.g. a
    never-used reserved vocab slot).
    """
    if override is not None:
        return int(override)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    enc = tokenizer.encode(" ", add_special_tokens=False)
    if enc:
        return int(enc[-1])
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert isinstance(eot, int) and eot >= 0, "no usable filler token"
    return eot


def resolve_im_end_token_id(tokenizer: Tokenizer) -> int:
    """Look up the `<|im_end|>` token id. Required for Qwen-family chat
    templates; callers pass this to `render_padded_prompt`. Asserts the
    tokenizer actually has it (turn-based compaction is meaningless
    otherwise)."""
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(im_end_id, int) and im_end_id >= 0, (
        f"<|im_end|> not in tokenizer vocab (got {im_end_id!r}). "
        "Block-aligned padding requires a Qwen-style chat template."
    )
    return im_end_id


def render_padded_prompt(
    *,
    tokenizer: Tokenizer,
    messages: list[dict],
    tools: list[dict] | None,
    block_size: int,
    filler_token_id: int,
    im_end_token_id: int,
    add_generation_prompt: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Render messages -> token ids -> insert filler tokens AFTER each
    `<|im_end|>` so the next turn's first token starts at a
    block_size-aligned position.

    Position after the filler run is a multiple of `block_size`,
    matching what the turn-mode scheduler tracks in `turn_end_positions`
    (the scan-time snap in vllm/v1/core/sched/scheduler.py yields the
    same position).

    Returns:
        raw_ids: chat-template output BEFORE padding. Needed for
            debugging and for asserting `len(padded_ids) >= len(raw_ids)`.
        padded_ids: chat-template output AFTER inserting filler tokens
            after each `<|im_end|>`. This is what vLLM runs on (via
            `extra_body={"prompt_token_ids": padded_ids}`).
        per_im_end_pads: number of filler tokens inserted after each
            `<|im_end|>`, in order of appearance. Useful for reconstruct-
            ing the padding mask and for sanity checks.

    Implementation notes:
        - We go via string + `tokenizer.encode`, not `tokenize=True`.
          Some chat templates (notably with tools) return the rendered
          string regardless of `tokenize=True`, so the string path is
          the one we trust.
        - `add_special_tokens=False` because the chat template already
          emits any required special tokens.
        - `add_generation_prompt=True` (default) appends the trailing
          `<|im_start|>assistant\\n` AFTER the last `<|im_end|>`'s
          filler run. That region is not padded (see module docstring:
          in-flight turn is never evictable under turn-mode; it gets
          re-padded next call once the assistant turn closes).
    """
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    raw = tokenizer.encode(rendered, add_special_tokens=False)
    # Defensive: ensure all ints. Some tokenizers return tokens as
    # np.int64 / tensor scalars which downstream serialization over
    # HTTP chokes on.
    bad = [(i, type(t).__name__) for i, t in enumerate(raw) if not isinstance(t, int)]
    if bad:
        raise TypeError(
            f"non-int tokens in tokenizer.encode output: first 5={bad[:5]}"
        )

    out: list[int] = []
    pads: list[int] = []
    for tok in raw:
        out.append(tok)
        if tok == im_end_token_id:
            remainder = len(out) % block_size
            n = (block_size - remainder) % block_size
            if n:
                out.extend([filler_token_id] * n)
            pads.append(n)
    # Final alignment: the generation prefix (<|im_start|>assistant\n) sits
    # after the last <|im_end|> filler run and is not itself padded by the
    # loop above, leaving len(out) = 16k + len(gen_prefix) instead of 16k.
    # Pad here so len(padded_ids) is always a multiple of block_size, which
    # guarantees prompt_aligned_len == prompt_len in the trainer (no overflow).
    remainder = len(out) % block_size
    if remainder:
        n = block_size - remainder
        out.extend([filler_token_id] * n)
        pads.append(n)
    return raw, out, pads
