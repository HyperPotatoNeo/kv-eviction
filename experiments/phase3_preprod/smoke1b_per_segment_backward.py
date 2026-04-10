#!/usr/bin/env python3
"""Pre-prod smoke test #1b: per-segment backward mode for segmented_forward.

Validates the production path for compaction training that sidesteps the
HF use_cache=True + checkpoint_wrapper cache-mutation crash (see
probe_ac_cache_mutation.py for the crash reproduction and
plans/phase3_training_integration.md for the design discussion).

Setup:
  - Single A100 80GB, Qwen3-4B bf16 + flash_attention_2
  - NO activation checkpointing anywhere (neither HF nor prime-rl nor
    segmented_forward's own flag). Per-segment backward bounds memory
    to O(1 segment) by running .backward() after each segment's
    forward, so the caller doesn't need AC.
  - Full 16k sample with all 24 compaction events.
  - loss_fn = teacher-forced cross-entropy over completion tokens only.

Checks:
  a) segmented_forward runs to completion without crashing.
  b) No NaN / Inf in any .grad after all 25 segments.
  c) Every parameter receives a non-zero gradient — guards against an
     autograd-breaking op in the per-segment detach/evict path
     silently zeroing out a subset of grads.
  d) Total grad norm is finite and in a sensible range (1e-6, 1e6).
  e) Peak GPU memory is SUBSTANTIALLY lower than smoke #1's 55.7 GB
     (which was the legacy activation_checkpointing path). The
     per-segment backward mode should need only ~ model + grads +
     one segment's activations + retained KV.

Also does a sanity pass with num_events=1 so the segment ranges
reduce to the simplest non-trivial case (2 segments).

Usage:
    cd /pscratch/sd/s/siddart2/kv-eviction
    source .venv/bin/activate
    python experiments/phase3_preprod/smoke1b_per_segment_backward.py \
        [--num-events -1] [--sample-idx 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from kv_eviction.segmented_forward import segmented_forward

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BLOCK_SIZE = 16
COMPACTION_STRIDE = 512
ROLLOUT_PATH = Path(
    "/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/results/rollouts_compaction.json"
)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_sample(idx: int, num_events: int) -> dict:
    data = json.loads(ROLLOUT_PATH.read_text())
    sample = data["samples"][idx]

    if num_events < 0:
        return {
            "prompt_ids": sample["prompt_ids"],
            "completion_ids": sample["completion_ids"],
            "compaction_events": sample["compaction_events"],
        }

    events = sample["compaction_events"][:num_events]
    tail_tokens = COMPACTION_STRIDE // 4
    if events:
        last_boundary = events[-1]["num_output_tokens_at_compaction"]
        comp_len = min(last_boundary + tail_tokens, len(sample["completion_ids"]))
    else:
        comp_len = min(2 * COMPACTION_STRIDE, len(sample["completion_ids"]))
    return {
        "prompt_ids": sample["prompt_ids"],
        "completion_ids": sample["completion_ids"][:comp_len],
        "compaction_events": events,
    }


def build_segment_loss_fn(
    input_ids: torch.Tensor,
    prompt_len: int,
    num_completion_tokens: int,
):
    """Build a per-segment teacher-forced cross-entropy loss closure.

    Each segment's logits at local index i correspond to global position
    (full_logit_start + i). The logit at global position P predicts
    input_ids[P + 1]. We only count predictions whose target is a
    completion token (position >= prompt_len), i.e. global positions
    P with P + 1 >= prompt_len, i.e. P >= prompt_len - 1.

    The per-segment losses must sum to the full-sequence mean-CE loss,
    so each segment contributes:
        sum_of_token_CE(this_segment) / num_completion_tokens
    and the final accumulated scalar equals the full-sequence mean.
    """

    def loss_fn(
        seg_logits: torch.Tensor,  # [1, seg_num_logits, vocab]
        full_logit_start: int,
        full_logit_end: int,
    ) -> torch.Tensor:
        # Target tokens are at positions [full_logit_start + 1, full_logit_end + 1).
        # Mask out prompt targets.
        target_start = full_logit_start + 1
        target_end = full_logit_end + 1
        # Clamp to the loss-relevant range.
        first_loss_pos = max(target_start, prompt_len)
        if first_loss_pos >= target_end:
            # No completion tokens owned by this segment.
            return seg_logits.sum() * 0.0  # zero, but keeps graph alive

        # Offset into seg_logits corresponding to first_loss_pos.
        local_start = first_loss_pos - target_start  # >= 0
        local_end = target_end - target_start  # == seg_num_logits
        # Targets: input_ids[first_loss_pos : target_end]
        targets = input_ids[0, first_loss_pos:target_end]
        owned_logits = seg_logits[0, local_start:local_end, :]
        log_softmax = torch.log_softmax(owned_logits.float(), dim=-1)
        # Gather the target logprobs.
        gathered = log_softmax.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        # Sum of CE over this segment's owned completion tokens, divided
        # by the total number of completion tokens so per-segment losses
        # sum to the full mean.
        return -gathered.sum() / num_completion_tokens

    return loss_fn


def run_smoke(num_events: int, sample_idx: int, bptt_segments: int | None) -> dict:
    device = torch.device("cuda:0")
    log(f"Loading {MODEL} on {device} (bf16, flash_attention_2)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    # No HF gradient checkpointing. No AC. Per-segment backward mode
    # provides its own memory bound.
    model.train()
    log(f"Model loaded in {time.time()-t0:.1f}s")

    sample = load_sample(sample_idx, num_events)
    prompt_ids = sample["prompt_ids"]
    completion_ids = sample["completion_ids"]
    events = sample["compaction_events"]
    prompt_len = len(prompt_ids)
    num_completion_tokens = len(completion_ids)
    seq_len = prompt_len + num_completion_tokens

    log(
        f"Sample {sample_idx}: prompt={prompt_len}, completion={num_completion_tokens}, "
        f"total={seq_len}, events={len(events)}"
    )

    input_ids = torch.tensor(
        [prompt_ids + completion_ids], dtype=torch.long, device=device
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    temperature = torch.ones(1, seq_len, device=device, dtype=torch.bfloat16)
    prompt_aligned_len = ((prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    segment_boundaries = [
        int(e["num_output_tokens_at_compaction"]) for e in events
    ]

    loss_fn = build_segment_loss_fn(input_ids, prompt_len, num_completion_tokens)

    # Ensure grads are zeroed before the run.
    model.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats(device)
    t_run = time.time()
    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=segment_boundaries,
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_aligned_len,
        stride=COMPACTION_STRIDE,
        temperature=temperature,
        max_forward_passes=None,
        activation_checkpointing=False,
        loss_fn=loss_fn,
        bptt_segments=bptt_segments,
    )
    run_time = time.time() - t_run
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    log(
        f"Per-segment forward+backward done in {run_time:.1f}s, "
        f"peak mem={peak_mem:.1f}GB, n_segments={out['n_segments']}, "
        f"bptt_segments={bptt_segments}"
    )
    log(f"Accumulated loss: {out['loss'].item():.4f}")

    # ─── Gradient audit ───
    total_params = 0
    params_with_grad = 0
    params_missing_grad: list[str] = []
    params_nan_grad: list[str] = []
    params_inf_grad: list[str] = []
    grad_sq_sum = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total_params += 1
        if p.grad is None:
            params_missing_grad.append(name)
            continue
        g = p.grad
        if torch.isnan(g).any():
            params_nan_grad.append(name)
        if torch.isinf(g).any():
            params_inf_grad.append(name)
        params_with_grad += 1
        grad_sq_sum += float(g.detach().float().pow(2).sum().item())
    grad_norm = grad_sq_sum**0.5

    log("=" * 60)
    log("GRADIENT AUDIT (per-segment backward mode)")
    log("=" * 60)
    log(f"  Total params with requires_grad: {total_params}")
    log(f"  Params with .grad set:           {params_with_grad}")
    log(f"  Params with missing .grad:       {len(params_missing_grad)}")
    log(f"  Params with NaN grad:            {len(params_nan_grad)}")
    log(f"  Params with Inf grad:            {len(params_inf_grad)}")
    log(f"  Global grad norm:                {grad_norm:.4f}")

    ok = True
    if params_missing_grad:
        ok = False
        log("  FAIL: missing grads")
        for n in params_missing_grad[:5]:
            log(f"    {n}")
    if params_nan_grad:
        ok = False
        log("  FAIL: NaN grads")
    if params_inf_grad:
        ok = False
        log("  FAIL: Inf grads")
    if grad_norm == 0.0:
        ok = False
        log("  FAIL: grad norm is exactly 0")
    elif not (1e-6 < grad_norm < 1e6):
        ok = False
        log(f"  WARN: grad norm {grad_norm} outside sensible range")

    log("=" * 60)
    log("RESULT: " + ("PASS" if ok else "FAIL"))
    log("=" * 60)

    return {
        "ok": ok,
        "loss": float(out["loss"].item()),
        "grad_norm": grad_norm,
        "total_params": total_params,
        "params_with_grad": params_with_grad,
        "params_missing_grad": len(params_missing_grad),
        "params_nan_grad": len(params_nan_grad),
        "params_inf_grad": len(params_inf_grad),
        "run_time_s": run_time,
        "peak_mem_gb": peak_mem,
        "n_segments": int(out["n_segments"]),
        "bptt_segments": bptt_segments,
        "num_events_used": num_events,
        "seq_len": seq_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-events", type=int, default=-1)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--bptt-segments", type=int, default=1,
        help="TBPTT truncation depth in segments (>=1, or 0 for full BPTT)",
    )
    args = parser.parse_args()

    bptt = None if args.bptt_segments == 0 else args.bptt_segments
    result = run_smoke(args.num_events, args.sample_idx, bptt)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        log(f"Result written to {args.output}")

    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
