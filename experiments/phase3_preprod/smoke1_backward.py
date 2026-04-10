#!/usr/bin/env python3
"""Pre-production smoke test #1: backward pass through segmented_forward.

Goal: verify that .backward() through segmented_forward (no detach) produces
sane gradients — no NaNs, no silently-dead parameters from autograd-breaking
slice/cat ops, gradient norm in a reasonable range. Phase 3.4 only validated
the forward pass (logit agreement with inference). This is the first time we
exercise the retained-KV torch.cat chain in the backward direction.

Setup:
  - Single GPU (no DDP/FSDP), Qwen3-4B bf16 + flash_attention_2
  - Per-segment activation checkpointing via the activation_checkpointing
    flag in segmented_forward. This wraps each segment's model() call in
    torch.utils.checkpoint.checkpoint (NOT HF's gradient_checkpointing_enable,
    which forcibly disables use_cache and breaks past_key_values).
  - Load one compaction sample from phase3_kl_test/results/rollouts_compaction.json
  - By default: full sample, all 24 events, no truncation
  - Optional --num-events N to truncate (for quick iteration during debug)
  - Compute a plain sum-of-log-softmax-of-sampled-tokens loss (negative, so
    .backward() gives a nonzero signal in the same direction RL would)
  - Checks:
      a) No NaN / Inf in any .grad
      b) Every non-frozen parameter with requires_grad receives a gradient
         (dead-param check — guards against autograd-breaking ops in the
         retained-KV chain that would silently zero a subset of grads)
      c) Global grad norm is finite and in a sensible range

Usage:
    cd /pscratch/sd/s/siddart2/kv-eviction
    source .venv/bin/activate
    python experiments/phase3_preprod/smoke1_backward.py \
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
    """Load sample idx. If num_events < 0, use the full sample with all
    events. Otherwise truncate to the first num_events events and cut the
    completion to cover that window plus a handful of tail tokens.
    """
    data = json.loads(ROLLOUT_PATH.read_text())
    sample = data["samples"][idx]

    if num_events < 0:
        # Full sample, all events, full completion
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


def build_loss(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Loss = -mean(log p(sampled_token)) over completion tokens.

    Standard teacher-forcing: the logit at position (prompt_len + i - 1)
    predicts completion[i]. We skip the prompt positions because the loss
    should only flow through completion tokens (matches how RL computes
    policy logprobs).
    """
    seq_len = input_ids.shape[1]
    # Positions of logits that predict completion tokens: [prompt_len-1, seq_len-2]
    # The token at position p+1 is predicted by logit at position p.
    log_softmax = torch.log_softmax(logits.float(), dim=-1)
    comp_positions = torch.arange(prompt_len - 1, seq_len - 1, device=logits.device)
    comp_targets = input_ids[0, prompt_len:]
    gathered = log_softmax[0, comp_positions, comp_targets]  # [num_comp]
    return -gathered.mean()


def run_smoke(num_events: int, sample_idx: int) -> dict:
    device = torch.device("cuda:0")
    log(f"Loading {MODEL} on {device} (bf16, flash_attention_2)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    log(f"Model loaded in {time.time()-t0:.1f}s")

    # Do NOT call model.gradient_checkpointing_enable() — HF's built-in GC
    # forcibly flips use_cache=False at the top-level forward, which strips
    # past_key_values from the output and breaks segmented_forward's entire
    # retained-KV flow. Instead we pass activation_checkpointing=True to
    # segmented_forward, which wraps each segment's model() call in
    # torch.utils.checkpoint.checkpoint externally while leaving use_cache
    # untouched.
    model.train()

    sample = load_sample(sample_idx, num_events)
    prompt_ids = sample["prompt_ids"]
    completion_ids = sample["completion_ids"]
    events = sample["compaction_events"]
    prompt_len = len(prompt_ids)
    seq_len = prompt_len + len(completion_ids)

    log(
        f"Sample {sample_idx}: prompt={prompt_len}, completion={len(completion_ids)}, "
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

    torch.cuda.reset_peak_memory_stats(device)
    t_fw = time.time()
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
        activation_checkpointing=True,
    )
    logits = out["logits"]
    fw_time = time.time() - t_fw
    log(
        f"Forward done in {fw_time:.1f}s, logits={tuple(logits.shape)}, "
        f"peak mem={torch.cuda.max_memory_allocated(device)/1e9:.1f}GB"
    )

    loss = build_loss(logits, input_ids, prompt_len)
    log(f"Loss = {loss.item():.4f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    t_bw = time.time()
    loss.backward()
    bw_time = time.time() - t_bw
    log(
        f"Backward done in {bw_time:.1f}s, "
        f"peak mem={torch.cuda.max_memory_allocated(device)/1e9:.1f}GB"
    )

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
    log("GRADIENT AUDIT")
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
        log("  FAIL: missing grads on:")
        for n in params_missing_grad[:10]:
            log(f"    {n}")
        if len(params_missing_grad) > 10:
            log(f"    ... and {len(params_missing_grad) - 10} more")
    if params_nan_grad:
        ok = False
        log("  FAIL: NaN grads on:")
        for n in params_nan_grad[:10]:
            log(f"    {n}")
    if params_inf_grad:
        ok = False
        log("  FAIL: Inf grads on:")
        for n in params_inf_grad[:10]:
            log(f"    {n}")
    if grad_norm == 0.0:
        ok = False
        log("  FAIL: global grad norm is exactly 0")
    elif not (1e-6 < grad_norm < 1e6):
        ok = False
        log(f"  WARN: global grad norm {grad_norm} outside [1e-6, 1e6]")

    log("=" * 60)
    log("RESULT: " + ("PASS" if ok else "FAIL"))
    log("=" * 60)

    return {
        "ok": ok,
        "loss": float(loss.item()),
        "grad_norm": grad_norm,
        "total_params": total_params,
        "params_with_grad": params_with_grad,
        "params_missing_grad": len(params_missing_grad),
        "params_nan_grad": len(params_nan_grad),
        "params_inf_grad": len(params_inf_grad),
        "forward_time_s": fw_time,
        "backward_time_s": bw_time,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "num_events_used": num_events,
        "seq_len": seq_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-events", type=int, default=-1,
                        help="Truncate to first N events (default: -1 = full sample)")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Which rollout sample to use (default 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write result JSON here")
    args = parser.parse_args()

    result = run_smoke(args.num_events, args.sample_idx)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        log(f"Result written to {args.output}")

    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
