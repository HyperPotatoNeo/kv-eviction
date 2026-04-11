#!/usr/bin/env python3
"""Memory leak probe for segmented_forward.

Hypothesis under test: with bptt_segments=1 the peak memory of
segmented_forward should be bounded by (1 segment's forward graph + 1
bounded retained-KV window), and the memory _at segmented_forward entry_
should be identical for every call across a training step.

If either bound is violated we have a leak.

Design:
- Single GPU (no FSDP) so we isolate segmented_forward from prime-rl's
  distributed state. Any growth observed here IS a segmented_forward-only
  problem; conversely, if this is flat, the real leak is in the
  FSDP2/optimizer/training-loop interaction and a subsequent probe
  instruments smoke #4 directly.
- Uses real rollouts from rollouts_compaction.json (24 events each,
  ~16k tokens).
- Runs multiple segmented_forward calls back-to-back using DIFFERENT
  samples each call, mimicking N consecutive compaction micro-batches
  in a single training step. Between calls we DO NOT reset p.grad — we
  accumulate, matching what prime-rl's training loop actually does.
- Most of the model is frozen (requires_grad=False) so we don't need
  16 GB per rank for fp32 .grad on a 4B model. A single tiny learnable
  parameter ("probe_tail") is added and the per-segment loss is routed
  through it, so backward has a path. This lets us measure the
  segmented_forward structure alone without drowning in optimizer-state
  noise.
- We log torch.cuda.memory_allocated() + max_memory_allocated() at every
  checkpoint. The key comparison is the value BEFORE call K vs BEFORE
  call K+1 — that's the delta that should be zero.

Usage (inside the kv-eviction venv, on an 80GB A100 node):
    python memory_probe_segforward.py --num-calls 5 --events-per-sample 8

The --events-per-sample lets us truncate each rollout so one segment
sees a realistic compaction but the full test fits in memory for a
40GB or 80GB GPU.
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
        f"[{ts}] {tag:<45s} alloc={_gb(alloc):7.3f} GB  "
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


def build_inputs(sample: dict, device: torch.device) -> dict:
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


def run_one_call(
    call_idx: int,
    model,
    probe_param: torch.nn.Parameter,
    inputs: dict,
    device: torch.device,
    log_per_segment: bool,
    real_loss: bool = False,
) -> None:
    """Run a single segmented_forward call with memory snapshots.

    log_per_segment=True wraps the loss_fn to log memory at every segment
    invocation, giving intra-call memory traces. Adds runtime overhead
    (Python-side logging) but does not perturb GPU state.
    """
    torch.cuda.reset_peak_memory_stats(device)
    log_mem(f"  call {call_idx} ENTRY", device)

    segment_idx = [0]

    def _probe_loss_fn(
        seg_logits: torch.Tensor,
        full_logit_start: int,
        full_logit_end: int,
    ) -> torch.Tensor:
        if real_loss:
            # Real loss path: use the actual segment logits, so backward
            # propagates gradients back through every matmul in the
            # forward graph and populates p.grad for every model param.
            # This is the gradient-flow pattern the real trainer uses.
            loss = seg_logits.float().mean()
        else:
            # Frozen-model path: route through probe_param only. Useful
            # to isolate segmented_forward structural memory from the
            # gradient-accumulation memory. Zero contribution to any
            # model parameter.
            dead = seg_logits.sum().detach() * 0.0
            loss = probe_param * 1.0 + dead
        if log_per_segment:
            log_mem(
                f"    call {call_idx} seg {segment_idx[0]:2d} LOSS_FN entry",
                device,
            )
        segment_idx[0] += 1
        return loss

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
        loss_fn=_probe_loss_fn,
        bptt_segments=1,
    )
    log_mem(f"  call {call_idx} EXIT (post segmented_forward)", device)

    # Zero the probe_param grad and (if unfrozen) the full model grads,
    # matching what the real trainer does after optimizer.step() + zero_grad().
    # We do not want cross-call grad accumulation confounding the leak test.
    if probe_param.grad is not None:
        probe_param.grad = None
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    log_mem(f"  call {call_idx} after grad reset", device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-calls", type=int, default=6)
    parser.add_argument("--events-per-sample", type=int, default=8)
    parser.add_argument("--log-per-segment", action="store_true")
    parser.add_argument("--samples", type=str, default="0,1,2,3,4")
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Leave model parameters with requires_grad=True so the "
        "full backward populates real p.grad buffers for every weight. "
        "Exercises the gradient-accumulation path that the frozen-model "
        "probe skips. Uses ~8GB extra for the grad buffer (bf16, 4B params).",
    )
    parser.add_argument(
        "--real-loss",
        action="store_true",
        help="Use a real per-segment loss (logits.float().mean()) that "
        "actually contributes to gradient flow, instead of routing "
        "through a probe scalar. Requires --unfreeze.",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Strip all compaction events from the samples before calling "
        "segmented_forward. Exercises the D5 unified-dispatch path where "
        "an event-less sample runs as a single-segment forward (equivalent "
        "to a plain text forward on the unpacked sample). Used to verify "
        "the empty-boundaries case works end-to-end.",
    )
    parser.add_argument(
        "--mixed-events",
        action="store_true",
        help="Alternate samples: even-indexed calls keep their full events, "
        "odd-indexed calls have events stripped. Reproduces the smoke #4 "
        "dispatch pattern where a single step has both multi-segment and "
        "single-segment samples going through segmented_forward.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    sample_idxs = [int(s) for s in args.samples.split(",")]

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
    if not args.unfreeze:
        # Freeze ALL model params so we don't allocate ~8 GB of bf16 .grad
        # on a non-FSDP single-rank run. The probe_param provides an
        # autograd leaf for window_loss.backward().
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        print("UNFROZEN: all params keep requires_grad=True", flush=True)
        # Note: model.eval() does not affect requires_grad; just dropout/BN.
    probe_param = torch.nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
    log_mem("after model load", device)

    rollouts = json.loads(ROLLOUT_PATH.read_text())
    samples_for_probe = [
        truncate_sample(rollouts["samples"][i], args.events_per_sample)
        for i in sample_idxs
    ]
    if args.no_events:
        # Drop compaction events from every sample. segmented_forward
        # should now treat each sample as a single segment covering the
        # whole [prompt+completion] sequence. Sanity check that the D5
        # unified dispatch path executes without crashing.
        for s in samples_for_probe:
            s["compaction_events"] = []
        print("--no-events: stripped all compaction_events from samples", flush=True)
    elif args.mixed_events:
        # Alternate: even-indexed samples keep events, odd-indexed get
        # stripped to [] so segmented_forward runs a single-segment
        # forward on them. Mirrors the D5-fix smoke #4 scenario where
        # the same step contains both event-bearing and event-less
        # samples, all routed through segmented_forward.
        for i, s in enumerate(samples_for_probe):
            if i % 2 == 1:
                s["compaction_events"] = []
        print(
            "--mixed-events: stripped events from odd-indexed samples "
            f"({sum(1 for s in samples_for_probe if not s['compaction_events'])} "
            f"event-less, {sum(1 for s in samples_for_probe if s['compaction_events'])} "
            "with events)",
            flush=True,
        )
    print(
        f"prepared {len(samples_for_probe)} samples "
        f"(events={args.events_per_sample} each)",
        flush=True,
    )
    # Pre-build input tensors for each sample on GPU so we do not count
    # per-call host-to-device transfer in the per-call memory snapshots.
    inputs_list = [build_inputs(s, device) for s in samples_for_probe]
    log_mem("after inputs prepared (GPU)", device)

    # ----- the actual probe: N sequential segmented_forward calls -----
    print(f"\n=== running {args.num_calls} segmented_forward calls ===", flush=True)
    for k in range(args.num_calls):
        inputs = inputs_list[k % len(inputs_list)]
        n_events = len(inputs["segment_boundaries"])
        print(f"\n-- call {k} (sample {sample_idxs[k % len(inputs_list)]}, "
              f"events={n_events}, seq_len={inputs['seq_len']}) --", flush=True)
        run_one_call(
            k,
            model,
            probe_param,
            inputs,
            device,
            args.log_per_segment,
            real_loss=args.real_loss,
        )

    print("\n=== done ===", flush=True)
    log_mem("final", device)


if __name__ == "__main__":
    main()
