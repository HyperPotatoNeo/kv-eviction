#!/usr/bin/env python3
"""Compaction -> text transition memory probe.

Mimics the order prime-rl's prepare_batch uses: all compaction
micro-batches first, then all text micro-batches. Measures whether the
text forward's peak differs when preceded by compaction calls vs when
run standalone.

Hypothesis: if segmented_forward leaves allocator state or residual
tensors that bloat the first text forward's peak, we'll see the peak
for the "after compaction" text forward exceed the peak for an
"isolated" text forward. If they match, the transition is clean and
the smoke-#4 OOM is not a segmented-forward residue problem.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/pscratch/sd/s/siddart2/kv-eviction/src")
from kv_eviction.segmented_forward import segmented_forward  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BLOCK_SIZE = 16
COMPACTION_STRIDE = 512
ROLLOUT_PATH = Path(
    "/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/results/rollouts_compaction.json"
)


def _gb(n_bytes: int) -> float:
    return n_bytes / 1e9


def log_mem(tag: str, device: torch.device) -> None:
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    ts = time.strftime("%H:%M:%S")
    print(
        f"[{ts}] {tag:<50s} alloc={_gb(alloc):7.3f} GB  "
        f"reserved={_gb(reserved):7.3f} GB  peak={_gb(peak):7.3f} GB",
        flush=True,
    )


def truncate_sample(sample: dict, num_events: int) -> dict:
    events = sample["compaction_events"][:num_events]
    if events:
        last_boundary = events[-1]["num_output_tokens_at_compaction"]
        tail = COMPACTION_STRIDE // 4
        new_comp_len = min(last_boundary + tail, len(sample["completion_ids"]))
    else:
        new_comp_len = min(2 * COMPACTION_STRIDE, len(sample["completion_ids"]))
    return {
        "prompt_ids": sample["prompt_ids"],
        "completion_ids": sample["completion_ids"][:new_comp_len],
        "compaction_events": events,
    }


def build_compaction_inputs(sample: dict, device: torch.device) -> dict:
    prompt_len = len(sample["prompt_ids"])
    seq_len = prompt_len + len(sample["completion_ids"])
    input_ids = torch.tensor(
        [sample["prompt_ids"] + sample["completion_ids"]],
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    temperature = torch.ones(1, seq_len, device=device, dtype=torch.bfloat16)
    prompt_aligned_len = ((prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    segment_boundaries = [
        int(e["num_output_tokens_at_compaction"]) for e in sample["compaction_events"]
    ]
    return dict(
        input_ids=input_ids,
        position_ids=position_ids,
        temperature=temperature,
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_aligned_len,
        segment_boundaries=segment_boundaries,
        seq_len=seq_len,
    )


def build_text_inputs(sample: dict, device: torch.device, text_seq_len: int) -> dict:
    """Produce a packed text micro-batch of size [1, text_seq_len] using
    the tokens from a compaction sample (prompt + completion, truncated).
    Matches how prime-rl's packer presents a full-context micro-batch.
    """
    all_ids = sample["prompt_ids"] + sample["completion_ids"]
    all_ids = all_ids[:text_seq_len]
    while len(all_ids) < text_seq_len:
        all_ids.append(1)
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(text_seq_len, device=device).unsqueeze(0)
    return dict(input_ids=input_ids, position_ids=position_ids, seq_len=text_seq_len)


def run_compaction_call(
    call_idx: int,
    model,
    inputs: dict,
    device: torch.device,
) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    log_mem(f"  compaction call {call_idx} ENTRY", device)

    def _loss_fn(seg_logits, s, e):
        return seg_logits.float().mean()

    segmented_forward(
        model=model,
        input_ids=inputs["input_ids"],
        position_ids=inputs["position_ids"],
        segment_boundaries=inputs["segment_boundaries"],
        prompt_len=inputs["prompt_len"],
        prompt_aligned_len=inputs["prompt_aligned_len"],
        stride=COMPACTION_STRIDE,
        temperature=inputs["temperature"],
        max_forward_passes=None,
        activation_checkpointing=False,
        loss_fn=_loss_fn,
        bptt_segments=1,
    )
    log_mem(f"  compaction call {call_idx} EXIT", device)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    log_mem(f"  compaction call {call_idx} after grad reset", device)


def run_text_call(call_idx: int, model, inputs: dict, device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    log_mem(f"  text call {call_idx} ENTRY", device)
    out = model(
        input_ids=inputs["input_ids"],
        position_ids=inputs["position_ids"],
    )
    logits = out["logits"] if isinstance(out, dict) else out.logits
    loss = logits.float().mean()
    log_mem(f"  text call {call_idx} post-forward", device)
    loss.backward()
    log_mem(f"  text call {call_idx} post-backward", device)
    del out, logits, loss
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    log_mem(f"  text call {call_idx} after grad reset", device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-compaction", type=int, default=5)
    parser.add_argument("--num-text", type=int, default=3)
    parser.add_argument("--events-per-sample", type=int, default=24)
    parser.add_argument("--text-seq-len", type=int, default=16000)
    parser.add_argument("--order", choices=["c_then_t", "t_then_c", "interleave"],
                        default="c_then_t")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    log_mem("before model load", device)
    print(f"loading {MODEL} bf16 flash_attn_2", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
    log_mem("after model load", device)

    rollouts = json.loads(ROLLOUT_PATH.read_text())
    comp_inputs = [
        build_compaction_inputs(
            truncate_sample(rollouts["samples"][i], args.events_per_sample),
            device,
        )
        for i in range(args.num_compaction)
    ]
    text_inputs = [
        build_text_inputs(rollouts["samples"][i], device, args.text_seq_len)
        for i in range(args.num_text)
    ]
    log_mem("inputs ready", device)

    seq = []
    if args.order == "c_then_t":
        seq = [("c", i) for i in range(args.num_compaction)] + \
              [("t", i) for i in range(args.num_text)]
    elif args.order == "t_then_c":
        seq = [("t", i) for i in range(args.num_text)] + \
              [("c", i) for i in range(args.num_compaction)]
    else:  # interleave
        for i in range(max(args.num_compaction, args.num_text)):
            if i < args.num_compaction:
                seq.append(("c", i))
            if i < args.num_text:
                seq.append(("t", i))

    print(f"\n=== running order={args.order}: {len(seq)} calls ===", flush=True)
    for idx, (kind, j) in enumerate(seq):
        print(f"\n-- step {idx}: {'COMPACTION' if kind == 'c' else 'TEXT'} "
              f"(sample {j}) --", flush=True)
        if kind == "c":
            run_compaction_call(j, model, comp_inputs[j], device)
        else:
            run_text_call(j, model, text_inputs[j], device)

    print("\n=== done ===", flush=True)
    log_mem("final", device)


if __name__ == "__main__":
    main()
