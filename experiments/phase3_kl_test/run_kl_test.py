#!/usr/bin/env python3
"""Phase 3.4 live KL test — trainer side.

Loads the two rollout JSONs produced by run_inference.py, loads Qwen3-4B via
HF in a DP=4 DDP setup, runs segmented_forward on compaction samples and
standard forward on baseline samples, computes per-token log-ratio between
trainer logprobs and inference logprobs (the "KL" proxy used in mkv-rl),
and reports per-condition statistics.

Expected outcome:
  - Compaction mean abs log-ratio close to baseline mean abs log-ratio
    (ideally both ~0). The baseline bounds the kernel-numerics noise we
    can't eliminate (bf16 + flash_attn rounding etc).
  - Compaction max abs log-ratio within a few multiples of baseline max.

Usage (inside podman-hpc container, kv-eviction .venv activated):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 run_kl_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import segmented_forward from the kv_eviction package we just wrote.
from kv_eviction.segmented_forward import (
    compute_num_segments,
    segmented_forward,
)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BLOCK_SIZE = 16  # must match the vLLM inference config in run_inference.py
COMPACTION_STRIDE = 512
OUTPUT_DIR = Path("/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/results")

# Set on dist init
RANK = 0
WORLD_SIZE = 1


def log(msg: str) -> None:
    if RANK == 0:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def setup_dist() -> tuple[int, int, torch.device]:
    """Initialize torch.distributed. Returns (rank, world_size, device)."""
    global RANK, WORLD_SIZE
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Single-process fallback (for debugging)
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    RANK = rank
    WORLD_SIZE = world
    return rank, world, device


def load_model(device: torch.device) -> torch.nn.Module:
    """Load Qwen3-4B via HF with flash_attention_2 + bf16.

    Each DP rank loads a full replica (Qwen3-4B bf16 is ~8GB, fits easily
    on an 80GB A100 alongside activations). No FSDP sharding — this test
    is about logit correctness, not memory efficiency.
    """
    log(f"Loading {MODEL} on {device} (bf16, flash_attention_2)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    log(f"Model loaded in {time.time()-t0:.0f}s")
    return model


@torch.no_grad()
def score_sample_standard(
    model: torch.nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    inference_logprobs: list[float],
    device: torch.device,
) -> dict:
    """Score a baseline (non-compaction) sample via standard model forward.

    Builds input_ids = prompt + completion (no eos append), runs a single
    forward, computes per-token logprob of completion tokens, compares to
    inference_logprobs.
    """
    input_ids = torch.tensor(
        [prompt_ids + completion_ids], dtype=torch.long, device=device,
    )
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    out = model(input_ids=input_ids, position_ids=position_ids)
    logits = out.logits  # [1, seq, vocab]
    # logprob of token at position p is softmax(logits[p-1])[token_id[p]]
    # For completion tokens: positions [len(prompt_ids), ..., seq_len-1]
    log_softmax = torch.log_softmax(logits.float(), dim=-1)
    prompt_len = len(prompt_ids)
    trainer_logprobs: list[float] = []
    for i, tok in enumerate(completion_ids):
        # The logit that predicts completion[i] is at position (prompt_len + i - 1)
        pos = prompt_len + i - 1
        if pos < 0:
            # Shouldn't happen with non-empty prompt.
            trainer_logprobs.append(float("nan"))
            continue
        trainer_logprobs.append(float(log_softmax[0, pos, tok].item()))
    return _compare(trainer_logprobs, inference_logprobs)


@torch.no_grad()
def score_sample_compaction(
    model: torch.nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    inference_logprobs: list[float],
    compaction_events: list[dict],
    device: torch.device,
    max_forwards: int,
) -> dict:
    """Score a compaction sample via segmented_forward.

    Builds input_ids = prompt + completion, converts compaction_events to
    segment_boundaries, runs segmented_forward with the correct
    prompt_aligned_len, extracts per-token logprobs, compares to inference.
    """
    input_ids = torch.tensor(
        [prompt_ids + completion_ids], dtype=torch.long, device=device,
    )
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    temperature = torch.ones(1, seq_len, device=device, dtype=torch.bfloat16)

    prompt_len = len(prompt_ids)
    prompt_aligned_len = ((prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    segment_boundaries = [
        int(e["num_output_tokens_at_compaction"]) for e in compaction_events
    ]

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=segment_boundaries,
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_aligned_len,
        stride=COMPACTION_STRIDE,
        temperature=temperature,
        max_forward_passes=max_forwards,
    )
    # segmented_forward returns PRE-scaled logits; since temperature=1 the
    # scaling is a no-op and we can softmax directly.
    logits = out["logits"]  # [1, seq, vocab]
    log_softmax = torch.log_softmax(logits.float(), dim=-1)

    trainer_logprobs: list[float] = []
    for i, tok in enumerate(completion_ids):
        pos = prompt_len + i - 1
        if pos < 0:
            trainer_logprobs.append(float("nan"))
            continue
        trainer_logprobs.append(float(log_softmax[0, pos, tok].item()))
    return _compare(trainer_logprobs, inference_logprobs)


def _compare(trainer_logprobs: list[float], inference_logprobs: list[float]) -> dict:
    """Compute per-token log-ratio statistics and return a summary dict."""
    assert len(trainer_logprobs) == len(inference_logprobs), (
        f"length mismatch: trainer={len(trainer_logprobs)}, "
        f"inference={len(inference_logprobs)}"
    )
    import math
    diffs: list[float] = []
    abs_diffs: list[float] = []
    for t, inf in zip(trainer_logprobs, inference_logprobs):
        if math.isnan(t) or math.isnan(inf):
            continue
        d = t - inf
        diffs.append(d)
        abs_diffs.append(abs(d))
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean_log_ratio": 0.0, "mean_abs_log_ratio": 0.0, "max_abs_log_ratio": 0.0}
    return {
        "n": n,
        "mean_log_ratio": sum(diffs) / n,
        "mean_abs_log_ratio": sum(abs_diffs) / n,
        "max_abs_log_ratio": max(abs_diffs),
    }


def load_rollouts() -> tuple[dict, dict]:
    """Load both rollout JSONs. All ranks read from the shared filesystem."""
    comp_path = OUTPUT_DIR / "rollouts_compaction.json"
    base_path = OUTPUT_DIR / "rollouts_baseline.json"
    assert comp_path.exists(), f"missing {comp_path}"
    assert base_path.exists(), f"missing {base_path}"
    return json.loads(comp_path.read_text()), json.loads(base_path.read_text())


def shard_indices(num_samples: int, rank: int, world: int) -> list[int]:
    """Return sample indices this rank should process.

    Uses a contiguous shard (not strided) so ranks processing
    longer/more-compacted samples are grouped together, reducing the
    max_forwards all_reduce impact on samples that don't need padding.
    """
    per_rank = (num_samples + world - 1) // world
    start = rank * per_rank
    end = min(start + per_rank, num_samples)
    return list(range(start, end))


def run_condition(
    model: torch.nn.Module,
    rollouts: dict,
    mode: str,  # "compaction" or "baseline"
    device: torch.device,
    rank: int,
    world: int,
) -> dict:
    """Run one condition (compaction or baseline) across all DP ranks.

    Contract: every DP rank iterates K = max(per_rank_sample_count) steps.
    At each step, every rank participates in all collectives (max_forwards
    all_reduce). Ranks with fewer real samples run a dummy step using
    samples[0] so the collective counts still match — dummy results are
    discarded from the aggregate stats.
    """
    assert mode in ("compaction", "baseline")
    samples = rollouts["samples"]
    num_samples = len(samples)
    per_rank_indices = shard_indices(num_samples, rank, world)

    per_rank_count = len(per_rank_indices)
    if dist.is_initialized() and WORLD_SIZE > 1:
        counts_t = torch.tensor([per_rank_count], device=device, dtype=torch.int32)
        dist.all_reduce(counts_t, op=dist.ReduceOp.MAX)
        k = int(counts_t.item())
    else:
        k = per_rank_count

    per_sample_results: list[dict] = []
    total_n = 0
    sum_abs = 0.0
    sum_signed = 0.0
    max_abs = 0.0

    for step in range(k):
        if step < per_rank_count:
            idx = per_rank_indices[step]
            is_real = True
        else:
            # Dummy step: samples[0] just to populate the collectives.
            idx = 0
            is_real = False

        sample = samples[idx]
        prompt_ids = sample["prompt_ids"]
        completion_ids = sample["completion_ids"]
        inference_logprobs = sample["inference_logprobs"]
        compaction_events = sample.get("compaction_events", [])

        # Compute max_forwards across ranks. Always run this all_reduce so
        # all ranks participate in the same number of collectives, regardless
        # of mode, matching the RSA-hardened trainer dispatch.
        prompt_len = len(prompt_ids)
        seq_len_local = prompt_len + len(completion_ids)
        if mode == "compaction":
            n_forwards_local = compute_num_segments(
                seq_len_local, prompt_len, [
                    int(e["num_output_tokens_at_compaction"]) for e in compaction_events
                ],
            )
        else:
            n_forwards_local = 1
        if dist.is_initialized() and WORLD_SIZE > 1:
            t_nf = torch.tensor([n_forwards_local], device=device, dtype=torch.int32)
            dist.all_reduce(t_nf, op=dist.ReduceOp.MAX)
            max_forwards = int(t_nf.item())
        else:
            max_forwards = n_forwards_local

        t0 = time.time()
        if mode == "compaction":
            stats = score_sample_compaction(
                model=model,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                inference_logprobs=inference_logprobs,
                compaction_events=compaction_events,
                device=device,
                max_forwards=max_forwards,
            )
        else:
            stats = score_sample_standard(
                model=model,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                inference_logprobs=inference_logprobs,
                device=device,
            )
        elapsed = time.time() - t0

        if is_real:
            per_sample_results.append({
                "idx": idx,
                "completion_len": len(completion_ids),
                "num_events": len(compaction_events),
                "max_forwards_used": max_forwards,
                "elapsed_s": elapsed,
                **stats,
            })
            total_n += stats["n"]
            sum_abs += stats["mean_abs_log_ratio"] * stats["n"]
            sum_signed += stats["mean_log_ratio"] * stats["n"]
            max_abs = max(max_abs, stats["max_abs_log_ratio"])
            print(
                f"  rank{rank} step{step} idx{idx} mode={mode} "
                f"completion_len={len(completion_ids)} "
                f"events={len(compaction_events)} "
                f"mean_abs_logratio={stats['mean_abs_log_ratio']:.4f} "
                f"max_abs={stats['max_abs_log_ratio']:.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    # All-reduce per-condition stats across ranks.
    if dist.is_initialized() and WORLD_SIZE > 1:
        buf = torch.tensor(
            [float(total_n), sum_abs, sum_signed, max_abs], device=device,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        total_n_g = float(buf[0].item())
        sum_abs_g = float(buf[1].item())
        sum_signed_g = float(buf[2].item())
        # max is reduced separately
        max_buf = torch.tensor([max_abs], device=device)
        dist.all_reduce(max_buf, op=dist.ReduceOp.MAX)
        max_abs_g = float(max_buf.item())
    else:
        total_n_g = float(total_n)
        sum_abs_g = sum_abs
        sum_signed_g = sum_signed
        max_abs_g = max_abs

    mean_abs_logratio = sum_abs_g / total_n_g if total_n_g > 0 else 0.0
    mean_signed_logratio = sum_signed_g / total_n_g if total_n_g > 0 else 0.0
    return {
        "num_samples": num_samples,
        "total_completion_tokens": int(total_n_g),
        "mean_abs_log_ratio": mean_abs_logratio,
        "max_abs_log_ratio": max_abs_g,
        "mean_signed_log_ratio": mean_signed_logratio,
        "per_sample_rank_local": per_sample_results,
    }


def main() -> None:
    rank, world, device = setup_dist()
    log("=" * 60)
    log(f"Phase 3.4 KL test — trainer side (rank {rank}/{world})")
    log("=" * 60)

    comp_rollouts, base_rollouts = load_rollouts()
    log(
        f"Loaded {len(comp_rollouts['samples'])} compaction rollouts, "
        f"{len(base_rollouts['samples'])} baseline rollouts"
    )

    model = load_model(device)

    # Baseline first (it's simpler and establishes the kernel-noise floor).
    log("--- Baseline condition (standard forward) ---")
    baseline_stats = run_condition(
        model, base_rollouts, "baseline", device, rank, world,
    )

    log("--- Compaction condition (segmented_forward) ---")
    compaction_stats = run_condition(
        model, comp_rollouts, "compaction", device, rank, world,
    )

    if rank == 0:
        log("=" * 60)
        log("RESULTS")
        log("=" * 60)
        log(f"Baseline (standard forward, no compaction):")
        log(f"  samples: {baseline_stats['num_samples']}")
        log(f"  total completion tokens: {baseline_stats['total_completion_tokens']}")
        log(f"  mean abs log-ratio: {baseline_stats['mean_abs_log_ratio']:.5f}")
        log(f"  max abs log-ratio:  {baseline_stats['max_abs_log_ratio']:.5f}")
        log(f"  mean signed log-ratio: {baseline_stats['mean_signed_log_ratio']:.5f}")
        log("")
        log(f"Compaction (segmented_forward, no detach):")
        log(f"  samples: {compaction_stats['num_samples']}")
        log(f"  total completion tokens: {compaction_stats['total_completion_tokens']}")
        log(f"  mean abs log-ratio: {compaction_stats['mean_abs_log_ratio']:.5f}")
        log(f"  max abs log-ratio:  {compaction_stats['max_abs_log_ratio']:.5f}")
        log(f"  mean signed log-ratio: {compaction_stats['mean_signed_log_ratio']:.5f}")
        log("")
        ratio = compaction_stats['mean_abs_log_ratio'] / max(
            baseline_stats['mean_abs_log_ratio'], 1e-9
        )
        log(f"Compaction mean_abs_log_ratio / baseline mean_abs_log_ratio = {ratio:.2f}x")

        out = {
            "config": {
                "model": MODEL,
                "block_size": BLOCK_SIZE,
                "compaction_stride": COMPACTION_STRIDE,
                "world_size": world,
            },
            "baseline": baseline_stats,
            "compaction": compaction_stats,
            "ratio_compaction_over_baseline": ratio,
        }
        out_path = OUTPUT_DIR / "kl_results.json"
        out_path.write_text(json.dumps(out, indent=2))
        log(f"Wrote {out_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
