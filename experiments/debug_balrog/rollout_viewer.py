# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio>=5.0",
#     "msgspec",
#     "transformers",
# ]
# ///
"""Gradio viewer for debug_balrog rollout microbatches.

Decodes the msgspec-encoded microbatches saved to rank_0.bin, renders
each rollout's token stream with prompt/filler/completion segments colored
and compaction-eviction ranges annotated.

Usage:
    uv run experiments/debug_balrog/rollout_viewer.py \\
        --rollouts-path /tmp/kv-eviction/debug_balrog_padding_smoke/rollouts/step_0/rank_0.bin \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --share
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import gradio as gr
import msgspec
from transformers import AutoTokenizer


# Duplicated from prime_rl.transport.types to avoid importing prime_rl
# (which triggers torch + flash_attn loading through its _compat shim).
# Wire format is stable via msgspec array_like + omit_defaults.
class CompactionEventWire(
    msgspec.Struct, array_like=True, gc=False, omit_defaults=True
):
    num_output_tokens_at_compaction: int
    tokens_evicted: int
    position_offset_after: int
    num_prompt_tokens: int = 0


class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperatures: list[float]
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: list[list[list[int]]] | None = None
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None
    image_grid_thw: list[list[int]] | None = None
    compaction_events: list[CompactionEventWire] | None = None
    prompt_len: int | None = None


FILLER_TOKEN_ID = 151643  # Qwen <|endoftext|>
IM_END_TOKEN_ID = 151645
IM_START_TOKEN_ID = 151644
BLOCK_SIZE = 16


def load_microbatches(path: Path) -> list[MicroBatch]:
    decoder = msgspec.msgpack.Decoder(list[MicroBatch])
    with open(path, "rb") as f:
        return decoder.decode(f.read())


def _render_span(tokens: list[int], tokenizer, color: str, label: str) -> str:
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    return (
        f'<div style="border-left:3px solid {color};padding:4px 8px;'
        f'margin:4px 0;background:rgba(0,0,0,0.02)">'
        f'<div style="font-size:11px;color:{color};font-weight:600">{label}</div>'
        f'<pre style="white-space:pre-wrap;font-family:monospace;font-size:12px;'
        f'margin:4px 0 0 0">{html.escape(text)}</pre></div>'
    )


def render_microbatch(mb: MicroBatch, tokenizer) -> str:
    """Render one MicroBatch as HTML with eviction events annotated."""
    prompt_len = mb.prompt_len or 0
    input_ids = list(mb.input_ids)
    events = mb.compaction_events or []

    parts: list[str] = []

    # Summary
    fillers_in_prompt = sum(1 for t in input_ids[:prompt_len] if t == FILLER_TOKEN_ID)
    parts.append(
        f'<div style="padding:8px;background:#f6f8fa;border-radius:4px;margin-bottom:8px">'
        f"<b>Summary</b> &middot; total tokens: {len(input_ids)} &middot; "
        f"prompt_len: {prompt_len} &middot; filler tokens in prompt: "
        f"{fillers_in_prompt} &middot; events: {len(events)}"
        f"</div>"
    )

    # Eviction events table
    if events:
        rows = []
        for i, ev in enumerate(events):
            rows.append(
                f"<tr><td>{i}</td><td>{ev.num_output_tokens_at_compaction}</td>"
                f"<td>{ev.tokens_evicted}</td><td>{ev.position_offset_after}</td>"
                f"<td>{ev.num_prompt_tokens}</td></tr>"
            )
        parts.append(
            '<div style="margin-bottom:8px"><b>Compaction events</b>'
            '<table style="border-collapse:collapse;font-size:12px;margin-top:4px">'
            "<tr><th style='padding:2px 8px'>#</th>"
            "<th style='padding:2px 8px'>n_out@compact</th>"
            "<th style='padding:2px 8px'>tokens_evicted</th>"
            "<th style='padding:2px 8px'>pos_offset_after</th>"
            "<th style='padding:2px 8px'>num_prompt_tokens</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )

    # Prompt: render as a single block with <|im_end|> boundaries highlighted
    parts.append(
        _render_span(input_ids[:prompt_len], tokenizer, "#1f6feb", f"PROMPT [0:{prompt_len})")
    )

    # Completion: split by eviction ranges
    # Events are in completion-token space: num_output_tokens_at_compaction is
    # the boundary where the NEXT segment begins.
    completion = input_ids[prompt_len:]
    cursor = 0
    for i, ev in enumerate(events):
        seg_end = ev.num_output_tokens_at_compaction
        if seg_end <= cursor or seg_end > len(completion):
            continue
        parts.append(
            _render_span(
                completion[cursor:seg_end],
                tokenizer,
                "#8250df",
                f"COMPLETION segment {i} [{cursor}:{seg_end}) "
                f"(then evict {ev.tokens_evicted} tokens from KV)",
            )
        )
        cursor = seg_end
    if cursor < len(completion):
        parts.append(
            _render_span(
                completion[cursor:],
                tokenizer,
                "#116329",
                f"COMPLETION tail [{cursor}:{len(completion)})",
            )
        )

    return "".join(parts)


def build_ui(batches: list[MicroBatch], tokenizer):
    def view(index: int):
        if not (0 <= index < len(batches)):
            return f"<b>Index out of range (0..{len(batches)-1})</b>"
        return render_microbatch(batches[index], tokenizer)

    with gr.Blocks(title="KV-Eviction rollout viewer") as demo:
        gr.Markdown(
            f"# debug_balrog rollout viewer\n"
            f"Loaded **{len(batches)}** microbatches. "
            f"Blue = prompt (padded, block-aligned), "
            f"purple = completion segment (pre-eviction), "
            f"green = completion tail (post last eviction)."
        )
        idx = gr.Slider(
            minimum=0,
            maximum=max(0, len(batches) - 1),
            step=1,
            value=0,
            label="microbatch index",
        )
        out = gr.HTML()
        idx.change(view, inputs=[idx], outputs=[out])
        demo.load(view, inputs=[idx], outputs=[out])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rollouts-path",
        default="/tmp/kv-eviction/debug_balrog_padding_smoke/rollouts/step_0/rank_0.bin",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print(f"Loading rollouts from {args.rollouts_path}")
    batches = load_microbatches(Path(args.rollouts_path))
    print(f"Loaded {len(batches)} microbatches")
    print(f"Loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    demo = build_ui(batches, tokenizer)
    demo.launch(share=args.share, server_port=args.port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
