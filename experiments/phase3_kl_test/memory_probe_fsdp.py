#!/usr/bin/env python3
"""FSDP2 memory probe for compaction -> text transition.

This probe mirrors prime-rl's FSDP2 setup exactly:
  - per-block `fully_shard` on every decoder layer
  - MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)
  - separate shard of embed_tokens (when word embeddings are NOT tied)
  - [lm_head, norm] sharded with reshard_after_forward=False
  - root model sharded last
  - constant-reshard optimizer (AdamW fp32) so step 1 allocates real
    optimizer state

and runs a sequence of compaction micro-batches followed by a text
micro-batch, logging per-rank memory at every micro-batch boundary.
This matches the smoke #4 failure mode:
  - step 0: 5 compaction micro-batches, then 1 text micro-batch
  - optimizer.step() (allocates Adam state for the first time)
  - step 1: 5 compaction micro-batches, then 1 text micro-batch
    (this is where smoke #4 OOMs)

Qwen3-Instruct-2507 uses tied word embeddings, so the "separate embed
shard" path is skipped and the full root shard wraps everything (matches
prime-rl's warning "skipping the last-layer no-reshard optimization").

Launch on a 4-GPU node:
  torchrun --standalone --nproc-per-node=4 memory_probe_fsdp.py \\
    --num-compaction 5 --events-per-sample 24 --num-text 1 --text-seq-len 16384

Emits memory_allocated() / max_memory_allocated() per rank at each step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/pscratch/sd/s/siddart2/kv-eviction/src")
from kv_eviction.segmented_forward import segmented_forward  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BLOCK_SIZE = 16
COMPACTION_STRIDE = 512
ROLLOUT_PATH = Path(
    "/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/results/rollouts_compaction.json"
)


def _gb(n: int) -> float:
    return n / 1e9


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][R{rank}] {msg}", flush=True)


def log_mem(tag: str, device: torch.device) -> None:
    rank = int(os.environ.get("RANK", "0"))
    alloc = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    ts = time.strftime("%H:%M:%S")
    print(
        f"[{ts}][R{rank}] {tag:<50s} alloc={_gb(alloc):7.3f} GB  "
        f"peak={_gb(peak):7.3f} GB  reserved={_gb(reserved):7.3f} GB",
        flush=True,
    )


def setup_fsdp_qwen3(device: torch.device):
    """Load Qwen3-4B and apply FSDP2 in the same pattern prime-rl uses.

    Qwen3-4B has tied word embeddings so the embed-only shard is skipped;
    the root-level fully_shard at the end wraps everything remaining.
    """
    log(f"loading {MODEL} on {device} bf16 flash_attn_2")
    t0 = time.time()
    # Load to CPU first (prime-rl does this: "Cannot load model to meta
    # device only, loading to CPU instead"). Then we'll move to GPU as
    # we apply FSDP2.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    log(f"model loaded in {time.time()-t0:.0f}s (CPU)")

    # Move to GPU on this rank before sharding. FSDP2 shards DTensors
    # across the mesh; each rank holds a shard.
    model = model.to(device)
    log_mem("after model to device", device)

    mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    fsdp_config = {
        "mp_policy": mp_policy,
        "reshard_after_forward": True,
    }

    # Per-block shard each decoder layer (prime-rl:model.py:382-393).
    language_model = model.model
    transformer_layers = language_model.layers
    for block in transformer_layers:
        fully_shard(block, mesh=mesh, **fsdp_config)

    # Qwen3 has tied word embeddings so skip the embed+(lm_head,norm)
    # special shard (prime-rl:model.py:395 check and :414 warning).

    # Finally shard the root so embed_tokens / lm_head / norm get
    # swept up in one big shard.
    fully_shard(model, mesh=mesh, **fsdp_config)
    log_mem("after fully_shard(root)", device)
    return model, mesh


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


def build_text_inputs(sample: dict, device: torch.device, seq: int) -> dict:
    all_ids = (sample["prompt_ids"] + sample["completion_ids"])[:seq]
    while len(all_ids) < seq:
        all_ids.append(1)
    return dict(
        input_ids=torch.tensor([all_ids], dtype=torch.long, device=device),
        position_ids=torch.arange(seq, device=device).unsqueeze(0),
        seq_len=seq,
    )


def run_compaction(call_idx: int, model, inputs: dict, device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    log_mem(f"  compaction {call_idx} ENTRY", device)

    def _loss_fn(seg_logits, s, e):
        # Use bf16 directly — matching prime-rl's compute_loss call chain,
        # which does NOT upcast full logits to fp32 (uses selective_log_softmax).
        # .float() here would double the memory of a [1, seq, vocab] tensor
        # and bust the budget on seq_len=16384.
        return seg_logits.mean()

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
    log_mem(f"  compaction {call_idx} EXIT", device)


def run_text(call_idx: int, model, inputs: dict, device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    log_mem(f"  text {call_idx} ENTRY", device)
    out = model(
        input_ids=inputs["input_ids"],
        position_ids=inputs["position_ids"],
    )
    logits = out["logits"] if isinstance(out, dict) else out.logits
    loss = logits.float().mean()
    log_mem(f"  text {call_idx} post-forward", device)
    loss.backward()
    log_mem(f"  text {call_idx} post-backward", device)
    del out, logits, loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-compaction", type=int, default=5)
    parser.add_argument("--num-text", type=int, default=1)
    parser.add_argument("--events-per-sample", type=int, default=24)
    parser.add_argument("--text-seq-len", type=int, default=16384)
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()

    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")

    log_mem("start", device)
    model, mesh = setup_fsdp_qwen3(device)
    log_mem("after FSDP setup", device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, weight_decay=0.01, betas=(0.9, 0.9)
    )
    log_mem("after optimizer init (lazy)", device)

    # Data
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
    log_mem("after data prep", device)

    # Run steps: each step runs num_compaction compaction mbs then
    # num_text text mbs (mirrors prime-rl's modality-sorted order),
    # then optimizer.step() + zero_grad().
    for step in range(args.num_steps):
        log(f"=== STEP {step} ===")
        for k in range(args.num_compaction):
            run_compaction(k, model, comp_inputs[k], device)
        for k in range(args.num_text):
            run_text(k, model, text_inputs[k], device)
        log_mem(f"step {step} PRE-OPTIM", device)
        optimizer.step()
        log_mem(f"step {step} POST-OPTIM.step", device)
        optimizer.zero_grad()
        log_mem(f"step {step} POST-zero_grad", device)

    log("=== DONE ===")
    log_mem("final", device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
