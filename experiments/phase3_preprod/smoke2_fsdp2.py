#!/usr/bin/env python3
"""Pre-production smoke test #2: FSDP2-sharded segmented_forward.

Goal: verify segmented_forward works correctly under FSDP2 parameter
sharding. Key questions:
  1. Do per-segment model() calls trigger FSDP2 all-gathers without
     deadlocking when multiple forwards feed into one backward?
  2. Are the final gradients finite, non-zero, and consistent across
     ranks (they should be identical because every rank processes the
     same sample)?
  3. Does the retained-KV torch.cat chain survive FSDP2's
     reshard-after-forward semantics during the multi-segment backward?

Scope limitation: this smoke does NOT test segmented_forward's
`activation_checkpointing=True` path under FSDP2. That specific
combination is known-broken because torch.utils.checkpoint's
non-reentrant mode doesn't trigger FSDP2's pre-forward hooks on
backward re-entry (hits "aten.mul.Tensor got mixed Tensor and DTensor"
in RMSNorm).

Prime-rl's production path handles AC differently: it wraps each
transformer block with torch.distributed.algorithms._checkpoint.checkpoint_wrapper
BEFORE applying fully_shard, so the hook ordering works. In production
the trainer should NOT enable segmented_forward's outer
activation_checkpointing — it should use prime-rl's per-block
checkpoint_wrapper pattern and call segmented_forward without its own
outer checkpointing. Smoke #3 (real trainer dispatch) validates that
combined FSDP2 + per-block AC + segmented_forward path.

To fit memory without outer AC we truncate to a few compaction events
(default 4 → 5 segments, ~3k token seq). That's plenty to exercise the
multi-forward-single-backward FSDP2 interaction.

Each rank processes the SAME sample (rollout 0) so all ranks should
see identical loss and identical gradient norm. Any cross-rank
divergence signals a bug.

Launched via torchrun --nproc_per_node=4. Uses fully_shard on each
transformer block (not the root, which causes DTensor interactions
with RMSNorm at the root level).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers import AutoModelForCausalLM

from kv_eviction.segmented_forward import segmented_forward

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BLOCK_SIZE = 16
COMPACTION_STRIDE = 512
ROLLOUT_PATH = Path(
    "/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/results/rollouts_compaction.json"
)

RANK = 0
WORLD_SIZE = 1


def log(msg: str) -> None:
    if RANK == 0:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def log_all_ranks(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] rank={RANK} {msg}", flush=True)


def setup_dist() -> tuple[int, int, torch.device]:
    global RANK, WORLD_SIZE
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    RANK = rank
    WORLD_SIZE = world
    return rank, world, device


def load_sample(num_events: int = 4) -> dict:
    """Load sample 0 truncated to first `num_events` compaction events.

    With 4 events: 5 segments, ~3k tokens, per-rank memory fits easily
    under FSDP2's 4-way param sharding without needing outer
    activation checkpointing (which breaks under FSDP2 with
    torch.utils.checkpoint non-reentrant mode).
    """
    data = json.loads(ROLLOUT_PATH.read_text())
    sample = data["samples"][0]
    events = sample["compaction_events"][:num_events]
    # Truncate completion to cover exactly up through the last kept
    # event plus a short tail so the final segment isn't empty.
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


def apply_fsdp2(model: torch.nn.Module) -> None:
    """Apply fully_shard to just the transformer decoder layers.

    This is the minimum pattern that exercises FSDP2's all-gather /
    reshard cadence on the large parameter groups (attention + MLP
    weights in each decoder layer). It intentionally does NOT shard the
    root model — sharding the root wraps model.norm and model.lm_head
    as DTensors, and HF's RMSNorm hits a "mixed Tensor/DTensor" error
    when torch.utils.checkpoint re-runs its forward during backward.

    Prime-rl's production path handles this by grouping `[lm_head,
    norm]` into their own fully_shard unit and explicitly setting
    `reshard_after_forward=False` on that group. For this smoke test the
    simpler path is enough: the decoder-layer shards are what actually
    stress-test the segmented_forward + checkpoint + FSDP2 interaction
    (the retained-KV chain flows through decoder-layer attention, which
    is where the interesting parameter all-gather cadence happens).
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    fsdp_kwargs = {
        "mp_policy": mp_policy,
        "reshard_after_forward": True,
    }
    language_model = model.model
    for layer in language_model.layers:
        fully_shard(layer, **fsdp_kwargs)


def build_loss(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    log_softmax = torch.log_softmax(logits.float(), dim=-1)
    seq_len = input_ids.shape[1]
    comp_positions = torch.arange(prompt_len - 1, seq_len - 1, device=logits.device)
    comp_targets = input_ids[0, prompt_len:]
    gathered = log_softmax[0, comp_positions, comp_targets]
    return -gathered.mean()


def compute_grad_norm(model: torch.nn.Module) -> tuple[float, list[int]]:
    """Compute global grad L2 norm across all sharded params.

    FSDP2 stores each parameter's grad as a DTensor whose local shard
    lives on the current rank. The DTensor API's arithmetic routines
    insist both operands be DTensors; naively accumulating
    `local_sq += g.float().pow(2).sum()` into a plain local tensor
    trips an isinstance assert deep in _dispatch.py.

    The robust path is to unwrap each DTensor to its local tensor via
    `.to_local()` (this returns the rank's shard as a plain tensor),
    compute the squared-sum on the local shard, and all-reduce at the
    end. For non-DTensor params (unsharded parts of the model) the same
    `.to_local()` trick isn't available, so we fall through to the
    plain tensor path.
    """
    from torch.distributed.tensor import DTensor

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    local_sq = torch.tensor(0.0, device=device, dtype=torch.float64)
    nan_count = 0
    inf_count = 0
    missing_count = 0
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            missing_count += 1
            continue
        g = p.grad
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if torch.isnan(g_local).any():
            nan_count += 1
        if torch.isinf(g_local).any():
            inf_count += 1
        local_sq = local_sq + g_local.detach().float().pow(2).sum().to(torch.float64)

    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    global_norm = float(local_sq.item()) ** 0.5

    # Reduce counters across ranks (NaN/Inf/missing flags are local, so
    # the sum gives us the global count of offending params).
    counters = torch.tensor(
        [total, missing_count, nan_count, inf_count],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(counters, op=dist.ReduceOp.SUM)
    return global_norm, counters.tolist()


def main() -> None:
    rank, world, device = setup_dist()
    log(f"world_size={world}, device={device}")
    log(f"Loading {MODEL}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    log(f"Model loaded on {device} in {time.time()-t0:.1f}s")

    apply_fsdp2(model)
    dist.barrier()
    log("FSDP2 sharding applied to all ranks")
    model.train()

    sample = load_sample()
    prompt_ids = sample["prompt_ids"]
    completion_ids = sample["completion_ids"]
    events = sample["compaction_events"]
    prompt_len = len(prompt_ids)
    seq_len = prompt_len + len(completion_ids)

    log(f"Sample: prompt={prompt_len}, completion={len(completion_ids)}, events={len(events)}")

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
    dist.barrier()
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
        activation_checkpointing=False,
    )
    logits = out["logits"]
    fw_time = time.time() - t_fw
    log(
        f"Forward done in {fw_time:.1f}s, logits={tuple(logits.shape)}, "
        f"peak mem={torch.cuda.max_memory_allocated(device)/1e9:.1f}GB"
    )

    loss = build_loss(logits, input_ids, prompt_len)
    # Sanity: all ranks should have the same loss since they processed
    # identical inputs. Verify by all_reducing max and min.
    loss_detached = loss.detach().clone()
    loss_max = loss_detached.clone()
    loss_min = loss_detached.clone()
    dist.all_reduce(loss_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(loss_min, op=dist.ReduceOp.MIN)
    loss_spread = (loss_max - loss_min).abs().item()
    log(f"Loss = {loss.item():.4f}, cross-rank spread = {loss_spread:.2e}")

    t_bw = time.time()
    loss.backward()
    bw_time = time.time() - t_bw
    log(
        f"Backward done in {bw_time:.1f}s, "
        f"peak mem={torch.cuda.max_memory_allocated(device)/1e9:.1f}GB"
    )

    grad_norm, counters = compute_grad_norm(model)
    total, missing, nans, infs = counters
    log("=" * 60)
    log("FSDP2 GRADIENT AUDIT")
    log("=" * 60)
    log(f"  World size:                      {world}")
    log(f"  Total params across ranks:       {total}")
    log(f"  Missing grads (across ranks):    {missing}")
    log(f"  NaN grads (across ranks):        {nans}")
    log(f"  Inf grads (across ranks):        {infs}")
    log(f"  Global (FSDP-reduced) grad norm: {grad_norm:.4f}")
    log(f"  Loss cross-rank spread:          {loss_spread:.2e}")

    ok = True
    if missing > 0:
        ok = False
        log("  FAIL: some params missing .grad")
    if nans > 0 or infs > 0:
        ok = False
        log("  FAIL: NaN/Inf grads present")
    if loss_spread > 1e-3:
        ok = False
        log("  FAIL: loss differs across ranks (should be identical)")
    if not (1e-6 < grad_norm < 1e6):
        ok = False
        log(f"  FAIL: grad norm {grad_norm} out of sensible range")

    log("=" * 60)
    log("RESULT: " + ("PASS" if ok else "FAIL"))
    log("=" * 60)

    if rank == 0:
        Path("/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_preprod/results/smoke2_result.json").write_text(
            json.dumps({
                "ok": ok,
                "world_size": world,
                "loss": float(loss.item()),
                "loss_cross_rank_spread": loss_spread,
                "grad_norm": grad_norm,
                "total_params": total,
                "missing_grad": missing,
                "nan_grad": nans,
                "inf_grad": infs,
                "forward_time_s": fw_time,
                "backward_time_s": bw_time,
                "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                "seq_len": seq_len,
                "num_events": len(events),
            }, indent=2)
        )

    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
