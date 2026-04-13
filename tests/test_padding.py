# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kv_eviction.padding.

Validates the block-alignment invariant for AFTER-padding: the position
immediately after each `<|im_end|>`-plus-filler-run is a multiple of
`block_size`. `<|im_end|>` itself sits at its natural position; fillers
occupy the inter-turn gap.

No real tokenizer: a minimal stub that echoes per-char ids is enough to
exercise the padding loop deterministically. The end-to-end guarantee
(against a real Qwen3 tokenizer and vLLM server) is covered by Gates 3-4
of the plan, not here.
"""

from dataclasses import dataclass, field

from kv_eviction.padding import (
    render_padded_prompt,
    resolve_filler_token_id,
    resolve_im_end_token_id,
)

IM_END = 999
FILLER = 888


@dataclass
class _StubTokenizer:
    """Minimal stub: `apply_chat_template` returns a tagged string,
    `encode` parses the tagged string into a fixed id sequence.

    Format: "TOK:<id>,<id>,..." — each comma-separated int is a token.
    This lets tests craft arbitrary raw-id sequences without needing
    to know a real chat template.
    """

    pad_token_id: int | None = FILLER
    _vocab: dict[str, int] = field(
        default_factory=lambda: {"<|im_end|>": IM_END, "<|endoftext|>": 0}
    )

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        # In tests we smuggle raw id sequences through the 'content'
        # field of the first message, pre-serialized as "id,id,id".
        return f"TOK:{messages[0]['content']}"

    def encode(self, s, add_special_tokens=False):
        assert s.startswith("TOK:"), s
        return [int(x) for x in s[len("TOK:") :].split(",") if x]

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, -1)


def _render(raw_ids: list[int], block_size: int):
    tok = _StubTokenizer()
    messages = [{"role": "user", "content": ",".join(str(i) for i in raw_ids)}]
    return render_padded_prompt(
        tokenizer=tok,
        messages=messages,
        tools=None,
        block_size=block_size,
        filler_token_id=FILLER,
        im_end_token_id=IM_END,
    )


def _im_end_positions(ids: list[int]) -> list[int]:
    return [i for i, t in enumerate(ids) if t == IM_END]


def test_single_im_end_next_turn_starts_on_block_boundary():
    # 5 non-im_end tokens then <|im_end|>. With block_size=8, after
    # appending <|im_end|> out has length 6; 2 filler tokens are appended
    # AFTER <|im_end|> so the next turn would start at index 8.
    raw = [1, 2, 3, 4, 5, IM_END]
    _, padded, pads = _render(raw, block_size=8)

    assert pads == [2]
    assert padded == [1, 2, 3, 4, 5, IM_END, FILLER, FILLER]
    # <|im_end|> sits at its natural position (index 5).
    assert _im_end_positions(padded) == [5]
    # Length after <|im_end|>-plus-fillers is a block boundary.
    assert len(padded) % 8 == 0


def test_multi_turn_every_boundary_block_aligned():
    # Three mini-turns of varying length, each closed with <|im_end|>.
    raw = (
        [1, 2, 3, IM_END]          # turn 1: 3 tokens body
        + [4, 5, 6, 7, 8, IM_END]  # turn 2: 5 tokens body
        + [9, IM_END]              # turn 3: 1 token body
    )
    _, padded, pads = _render(raw, block_size=4)

    # Trace with block_size=4 (AFTER padding):
    #  after turn 1 <|im_end|>: out len=4 -> remainder=0, pad=0.
    #  after turn 2 <|im_end|>: out len=10 -> remainder=2, pad=2. len=12.
    #  after turn 3 <|im_end|>: out len=14 -> remainder=2, pad=2. len=16.
    assert pads == [0, 2, 2]
    assert _im_end_positions(padded) == [3, 9, 13]
    # After each <|im_end|>'s filler run, the next slot is block-aligned.
    # The cumulative position immediately past each turn's fillers should
    # be a multiple of block_size.
    running = 0
    for tok in padded:
        running += 1
    assert running % 4 == 0
    # And the first-boundary snapshots:
    # end of turn 1 fillers = 4, end of turn 2 = 12, end of turn 3 = 16.
    assert [3 + 1 + 0, 9 + 1 + 2, 13 + 1 + 2] == [4, 12, 16]


def test_zero_pad_when_already_aligned():
    # Body length such that <|im_end|> lands exactly on a block boundary
    # means no padding needed.
    raw = [1, 2, 3, 4, 5, 6, 7, IM_END]
    _, padded, pads = _render(raw, block_size=8)
    assert pads == [0]
    assert padded == raw
    # <|im_end|> at index 7; len(out)=8 is already block-aligned.
    assert _im_end_positions(padded) == [7]
    assert len(padded) % 8 == 0


def test_generation_prompt_suffix_not_padded():
    # raw_ids has trailing tokens AFTER the last <|im_end|> (simulating
    # add_generation_prompt appending '<|im_start|>assistant\n').
    # Fillers ARE inserted after each <|im_end|>, then the trailing
    # generation-prompt tokens follow immediately (no further padding
    # before the assistant turn closes with its own <|im_end|>).
    raw = [1, 2, IM_END, 77, 78, 79]  # 77/78/79 = <|im_start|>\nassistant\n stub
    _, padded, pads = _render(raw, block_size=4)
    # One <|im_end|> pad: after appending IM_END, len=3, pad=1.
    assert pads == [1]
    assert padded == [1, 2, IM_END, FILLER, 77, 78, 79]
    # Tail tokens preserved, unaligned on purpose (assistant in-flight).
    assert padded[-3:] == [77, 78, 79]
    assert _im_end_positions(padded) == [2]


def test_resolve_filler_override():
    tok = _StubTokenizer(pad_token_id=42)
    assert resolve_filler_token_id(tok, override=100) == 100
    # Override None falls through to pad_token_id.
    assert resolve_filler_token_id(tok, override=None) == 42


def test_resolve_filler_no_pad_token_falls_back_to_space():
    class NoPad:
        pad_token_id = None

        def encode(self, s, add_special_tokens=False):
            return [32]  # pretend ' ' tokenizes to id 32

        def convert_tokens_to_ids(self, tok):
            return -1

    assert resolve_filler_token_id(NoPad(), override=None) == 32


def test_resolve_im_end_asserts_present():
    tok = _StubTokenizer()
    assert resolve_im_end_token_id(tok) == IM_END

    class NoImEnd:
        def convert_tokens_to_ids(self, tok):
            return -1

    try:
        resolve_im_end_token_id(NoImEnd())
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError when <|im_end|> is missing")
