# Phase 4: Experiments

## Current status (2026-04-11)

**Smokes #1-#4 passing. Smoke #5 (production config) is the next work
item, unblocked after D5 resolution.**

Smoke progression:

| Smoke | Purpose | Status | Notes |
|---|---|---|---|
| #1 | Single-GPU backward pass | PASS | Single rank, no FSDP. `experiments/smoke1_single_gpu_backward/`. |
| #1b | Per-segment backward probe | PASS | Verifies bptt_segments=1 frees segment activations between segments. |
| #2 | FSDP2 segmented_forward | PASS | 4-rank FSDP2, dummy-pass rank sync, no cross-rank collective drift. |
| #3 | Single end-to-end RL step | PASS (cosmetically) | Dispatch never actually fired on event-bearing samples due to events-plumbing bugs; real validation came from smoke #4 v5 step 0. |
| #4 | 5-step stability | PASS (v6) | Earlier v1-v5 attempts were blocked by events plumbing chain (msgspec->JSON, verifiers Response field drop, subprocess import, MultiTurnEnv hook, trainer dispatch), AC + DynamicCache double-update, torch.compile entropy thrash, and finally the D5 compaction→text OOM. v6 passes all 5 steps with peak 45.9 GiB, Mismatch KL 0.0009 kernel floor. |
| #4b | Full-context baseline | PASS | Runs the same 5 steps with compaction disabled so the Mismatch KL and reward are comparable. Peak 74.9 GiB. Mismatch KL 0.0007. |
| **#5** | **Production-config smoke** | **PENDING** | Next work item. Tune batch_size / seq_len / bptt_segments above smoke #4 settings, run for enough steps to see reward trajectory, verify memory / mismatch KL / reward all stay healthy. Compare against smoke #4b full-context baseline on the same data. |

Published forks that make this reproducible from a fresh clone:

- `https://github.com/HyperPotatoNeo/kv-eviction` (top-level)
- `https://github.com/HyperPotatoNeo/vllm` (branch `compaction`)
- `https://github.com/HyperPotatoNeo/tba-prime` (branch `kv-eviction`)

## Goal

Compare two training conditions on the same reasoning-gym workload:
1. **Full Context** — baseline: stock vLLM + standard prime-rl forward
   (smoke #4b, `experiments/smoke4b_full_context/`)
2. **Markovian KV / compaction** — compaction-enabled vLLM (window=4096,
   stride=512) + segmented forward via the unified D5 dispatch
   (smoke #4 v6, `experiments/smoke4_rl_stability_run/`)

Each condition uses 2 nodes (4× A100-80GB each): 1 node for inference
(vLLM DP=4), 1 node for training (FSDP2 DP=4). Model:
`Qwen/Qwen3-4B-Instruct-2507`.

## Prerequisites

- Phase 2 complete: vLLM with compaction returning `compaction_events`
  in responses. DONE.
- Phase 3 complete: segmented_forward + unified dispatch + events
  plumbing from vLLM → verifiers → trajectory → TrainingSample →
  MicroBatch → trainer. DONE.
- Both smoke #4 and smoke #4b pass with Mismatch KL at kernel floor.
  DONE.

---

## 4.1: Experiment Configurations

### Full Context Config

**File: `$SCRATCH/kv-eviction/experiments/full_context.toml`**

```toml
# Full Context baseline — standard vLLM + standard training
# 2 nodes: node 0 = inference (DP=4), node 1 = training (FSDP over 4 GPUs)

seed = 42

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"
impl = "hf"              # HuggingFace implementation for training forward
max_model_len = 16384    # Full 16k context, no compaction

[inference]
host = "0.0.0.0"
port = 8000
dp = 4                   # 4 independent vLLM workers on node 0
tp = 1
# No compaction args — standard vLLM behavior

[training]
algorithm = "dppo"
batch_size = 32
micro_batch_size = 1     # Per-GPU micro-batch (FSDP across 4 GPUs)
gradient_accumulation_steps = 8
learning_rate = 1e-6
weight_decay = 0.01
warmup_steps = 50
max_steps = 2000
# Standard forward (no segmented_forward)
use_segmented_forward = false

[training.grpo]
# Use prime-rl defaults — do NOT override loss params
num_generations = 8
temperature = 1.0
top_p = 0.95

[environment]
name = "rg-mix-env"
# 5 reasoning_gym tasks with inverse pass@1 weighting
max_tokens = 16384       # Matches max_model_len

[logging]
wandb_project = "kv-eviction"
wandb_run_name = "full-context"
log_interval = 10
eval_interval = 100
```

### Markovian KV Config

**File: `$SCRATCH/kv-eviction/experiments/markovian_kv.toml`**

```toml
# Markovian KV — compaction vLLM + segmented forward (no detach)
# 2 nodes: node 0 = inference (DP=4 with compaction), node 1 = training (FSDP)

seed = 42

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"
impl = "hf"              # Required for use_cache=True in segmented forward
max_model_len = 32768    # Higher than full context — compaction keeps KV bounded

[inference]
host = "0.0.0.0"
port = 8000
dp = 4
tp = 1
# Compaction args — passed to vLLM server
compaction_window_size = 4096
compaction_stride = 512
compaction_strategy = "sliding"

[training]
algorithm = "dppo"
batch_size = 32
micro_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 1e-6
weight_decay = 0.01
warmup_steps = 50
max_steps = 2000
# Enable segmented forward for compacted rollouts
use_segmented_forward = true
compaction_stride = 512  # Must match inference side

[training.grpo]
num_generations = 8
temperature = 1.0
top_p = 0.95

[environment]
name = "rg-mix-env"
max_tokens = 16384       # Total generation tokens (vLLM compacts along the way)

[logging]
wandb_project = "kv-eviction"
wandb_run_name = "markovian-kv"
log_interval = 10
eval_interval = 100
```

### Key Differences Between Configs

| Parameter | Full Context | Markovian KV |
|-----------|-------------|--------------|
| max_model_len | 16384 | 32768 (compaction bounds actual KV) |
| compaction_window_size | (not set) | 4096 |
| compaction_stride | (not set) | 512 |
| use_segmented_forward | false | true |
| Training forward | Standard single-pass | Segmented (flash_attn per segment, no detach) |
| KV at inference time | Grows to 16k | Bounded at 4096 |
| Speed (inference) | Slower (long sequences) | Faster (shorter KV after compaction) |
| Memory (training) | Standard | Higher (retained KV activations across segments) |

---

## 4.2: SLURM Launch Script

**File: `$SCRATCH/kv-eviction/experiments/launch.sh`**

```bash
#!/bin/bash
# Launch kv-eviction experiment on NERSC Perlmutter
# Usage: sbatch --export=CONFIG=full_context.toml experiments/launch.sh
#    or: sbatch --export=CONFIG=markovian_kv.toml experiments/launch.sh
#
# Allocates 2 nodes:
#   Node 0: vLLM inference server (DP=4)
#   Node 1: prime-rl DPPO trainer (FSDP over 4 GPUs)

#SBATCH -A m5017
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -euo pipefail

PROJECT=/pscratch/sd/s/siddart2/kv-eviction
CONFIG="${CONFIG:-full_context.toml}"
CONFIG_PATH="$PROJECT/experiments/$CONFIG"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman

# Get node hostnames
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_INFERENCE="${NODES[0]}"
NODE_TRAINER="${NODES[1]}"

echo "Job $SLURM_JOB_ID: $CONFIG"
echo "  Inference node: $NODE_INFERENCE"
echo "  Trainer node:   $NODE_TRAINER"
echo "  Config: $CONFIG_PATH"
echo ""

CONTAINER_IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8"
CONTAINER_NAME="kv-eviction-${SLURM_JOB_ID}"

# Common podman-hpc args
PODMAN_ARGS=(
    --rm -it
    --user "$(id -u):$(id -g)"
    --replace
    --group-add keep-groups
    --userns keep-id
    --gpu --nccl --shm-size=8g
    -e SCRATCH -e HOME
    -e WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8
    -v "$HOME":"$HOME"
    -v "/pscratch/sd/s/siddart2:/pscratch/sd/s/siddart2"
    -w "$PROJECT"
)

# --- Node 0: Inference Server ---
echo "Starting inference server on $NODE_INFERENCE..."
ssh "$NODE_INFERENCE" "
    export HOME=$HOME
    export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN
    podman-hpc run ${PODMAN_ARGS[*]} \
        --name ${CONTAINER_NAME}-inference \
        $CONTAINER_IMAGE \
        bash -c '
            unset NCCL_SOCKET_IFNAME
            source $PROJECT/.venv/bin/activate
            python -m kv_eviction.serve --config $CONFIG_PATH
        '
" &
INFERENCE_PID=$!

# Wait for server to be ready
echo "Waiting for inference server..."
sleep 30
for i in $(seq 1 60); do
    if ssh "$NODE_INFERENCE" "curl -s http://localhost:8000/health" >/dev/null 2>&1; then
        echo "Inference server ready after ${i}0s"
        break
    fi
    sleep 10
done

# --- Node 1: Trainer ---
echo "Starting trainer on $NODE_TRAINER..."
ssh "$NODE_TRAINER" "
    export HOME=$HOME
    export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN
    podman-hpc run ${PODMAN_ARGS[*]} \
        --name ${CONTAINER_NAME}-trainer \
        $CONTAINER_IMAGE \
        bash -c '
            unset NCCL_SOCKET_IFNAME
            source $PROJECT/.venv/bin/activate
            python -m kv_eviction.train \
                --config $CONFIG_PATH \
                --inference-url http://${NODE_INFERENCE}:8000
        '
" &
TRAINER_PID=$!

# Wait for both to finish
wait $INFERENCE_PID $TRAINER_PID
echo "Job complete."
```

**Note**: This is a template. The exact commands (`kv_eviction.serve`, `kv_eviction.train`)
depend on how prime-rl's entry points work. The mkv-rl project uses `uv run inference`
and `uv run train` commands. The kv-eviction project should follow the same pattern or
wrap them appropriately.

---

## 4.3: Verification Checklist

Before running full experiments, verify each of these independently:

### V1: Block Eviction + Splice Correctness

```python
"""Verify that block eviction produces a valid shorter block table."""
# 1. Start vLLM with compaction enabled
# 2. Send a prompt that generates >window_size tokens
# 3. Inspect request state: block_table should be shorter after compaction
# 4. Verify freed blocks are returned to pool (block count accounting)
```

### V2: Position Offset Monotonicity

```python
"""Verify position_offset increases monotonically and matches tokens evicted."""
# For each compaction event:
#   position_offset_after == previous_offset + tokens_evicted
# position_offset never decreases
```

### V3: No-Compaction = Standard vLLM

```python
"""With compaction disabled (window=0), output is identical to stock vLLM."""
# 1. Run same prompt with compaction-enabled vLLM (window=0) and stock vLLM
# 2. Token IDs should be identical (same seed)
# 3. Logprobs should be identical (bitwise)
```

### V4: Step-0 KL Approximately Zero

```python
"""Training logits match inference logits at step 0."""
# 1. Generate rollout with compaction vLLM (window=4096, stride=512)
# 2. Extract token_ids, segment_boundaries, logprobs
# 3. Run segmented_forward with same model weights
# 4. Compute KL divergence between inference and training logprobs
# 5. Assert KL < 0.01 (should be ~0.0 since both use flash_attn)
#
# This is THE critical correctness test. If this fails, the training
# signal is corrupted from step 0.
```

### V5: G_distal Nonzero

```python
"""Cross-chunk gradient term is nonzero (no-detach works)."""
# 1. Run segmented_forward (no detach) on a sample, compute loss, backprop
# 2. Record gradients for early-segment parameters
# 3. Run segmented_forward_detached (from mkv-rl) on the same sample
# 4. Record gradients for same parameters
# 5. G_distal = grad_no_detach - grad_detach
# 6. Assert norm(G_distal) > 0 (should be significant, not just noise)
```

### V6: FSDP Gradient Norm Match

```python
"""Gradients are correct under FSDP (single-GPU matches FSDP)."""
# 1. Run segmented_forward on single GPU, record gradient norms
# 2. Run same forward under FSDP (4 GPUs), record gradient norms
# 3. Norms should match within floating-point tolerance
# This validates that dummy forward passes don't corrupt gradients
```

### V7: Speed Gain After Eviction

```python
"""Inference is faster with compaction (shorter KV = faster attention)."""
# 1. Generate 16k tokens WITHOUT compaction — record time
# 2. Generate 16k tokens WITH compaction (window=4096) — record time
# 3. Compaction should be faster because attention kernel sees <=4096 KV
#    instead of growing to 16k
```

---

## 4.4: Test Script for Logit Match

**File: `$SCRATCH/kv-eviction/tests/test_logit_match.py`**

```python
"""End-to-end logit match test: inference vs training.

Requires a running vLLM server with compaction enabled.
Run with: pytest tests/test_logit_match.py -v --inference-url http://localhost:8000

This test verifies the most critical property: training and inference
produce the same logits for the same input. Any mismatch means the
RL training signal is corrupted from step 0.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def inference_url(request):
    return request.config.getoption("--inference-url", default="http://localhost:8000")


def pytest_addoption(parser):
    parser.addoption("--inference-url", default="http://localhost:8000")


class TestLogitMatch:

    def test_step0_kl_full_context(self, inference_url):
        """Full context: standard forward should match standard vLLM logprobs."""
        # 1. Send prompt to vLLM (no compaction), get logprobs
        # 2. Load same model weights in HF
        # 3. Run standard forward
        # 4. Compare logprobs
        # Expected: KL ~ 0.0
        pass

    def test_step0_kl_compaction(self, inference_url):
        """Compaction: segmented forward should match compaction vLLM logprobs."""
        # 1. Send prompt to compaction vLLM, get logprobs + compaction_events
        # 2. Load same model weights in HF
        # 3. Run segmented_forward with segment_boundaries from events
        # 4. Compare logprobs
        # Expected: KL ~ 0.0
        pass

    def test_g_distal_nonzero(self):
        """G_distal gradient term is nonzero when using no-detach."""
        # See V5 above
        pass

    def test_speed_gain(self, inference_url):
        """Compaction inference is faster than full context for long sequences."""
        # See V7 above
        pass
```

---

## 4.5: Metrics to Track

### Per-Step Metrics (logged to W&B)

| Metric | Description |
|--------|-------------|
| `train/loss` | DPPO policy loss |
| `train/kl` | KL between current policy and reference |
| `train/grad_norm` | Gradient L2 norm |
| `train/g_distal_norm` | Norm of G_distal (no-detach minus detach gradient) — Markovian KV only |
| `eval/reward_mean` | Mean reward on eval set |
| `eval/reward_std` | Reward standard deviation |
| `eval/pass_at_1` | Per-task pass@1 |
| `inference/tokens_per_second` | Generation throughput |
| `inference/compaction_events` | Number of compactions per rollout (Markovian KV only) |
| `inference/kv_length_mean` | Mean KV cache length at end of generation |
| `training/forward_time` | Time for forward pass (segmented vs standard) |
| `training/segments_per_sample` | Number of segments per training sample |
| `training/memory_peak_gb` | Peak GPU memory during training forward |

### Summary Metrics (final comparison)

| Metric | Full Context | Markovian KV | Delta |
|--------|-------------|--------------|-------|
| Final reward | ? | ? | ? |
| Convergence speed (steps to X reward) | ? | ? | ? |
| Inference throughput (tok/s) | ? | ? | ? |
| Training throughput (samples/s) | ? | ? | ? |
| Peak memory (GB) | ? | ? | ? |
| Step-0 KL | ~0.0 | ~0.0 | ~0.0 |
| G_distal norm | N/A | >0 | N/A |

---

## 4.6: Expected Results

### Full Context (Baseline)
- Standard behavior, no compaction
- Attention cost grows quadratically with sequence length up to 16k
- Training: single forward pass per sample
- Should match standard prime-rl results

### Markovian KV
- Inference: bounded attention cost (~4096 KV length max)
- Training: ~25 forward passes per sample (segments), but each over ~4096 tokens
- G_distal should be nonzero, demonstrating cross-chunk gradient flow
- Reward should be competitive with full context (hypothesis: KV compaction + G_distal
  is sufficient for reasoning tasks)
- Speed: faster inference per token (shorter KV), slower training per sample (multiple segments)

### Key Questions the Experiment Answers
1. Does block-level KV compaction produce correct logprobs? (V4: step-0 KL)
2. Does G_distal improve learning vs detached KV? (compare with mkv-rl M3 results)
3. Is the inference speed gain from shorter KV significant?
4. Is the training memory cost of no-detach feasible on A100-80GB?

---

## 4.7: Debugging Playbook

### If step-0 KL > 0.01
1. Check that model uses `impl = "hf"` in training config
2. Check that position_ids are constructed correctly (should be simple 0..N-1 range)
3. Check that `segment_boundaries` match between inference and training
4. Check that `stride` matches between inference and training
5. Run single-segment case (no compaction) to isolate: if KL is still high, the issue
   is in the basic forward path, not compaction
6. Check for numerical differences: bf16 vs fp32, different flash_attn versions

### If G_distal is zero
1. Verify `.detach()` is NOT called anywhere in `segmented_forward.py`
2. Check that `DynamicCache.update()` preserves the autograd graph (no hidden detach)
3. Check that `torch.cat()` on KV tensors preserves requires_grad
4. Run a simple test: create a tensor with requires_grad=True, permute/cat/unsqueeze it,
   verify grad still flows

### If training OOMs
1. Reduce `micro_batch_size` or `max_tokens`
2. Enable gradient checkpointing in the HF model config
3. If still OOM, consider adding `.detach()` to the oldest N segments (keeping only the
   most recent K segments differentiable). This trades G_distal for memory.

### If FSDP hangs
1. Check that `max_forward_passes` is synchronized across all ranks
2. Check that `compute_max_forward_passes()` is called on all ranks with the same batch
3. Verify dummy passes are executing (add debug logging)

---

## Completion Criteria

- [ ] `full_context.toml` and `markovian_kv.toml` configs exist and are valid
- [ ] `launch.sh` successfully allocates 2 nodes and starts both containers
- [ ] V1-V7 verification checks all pass
- [ ] Full Context baseline runs for 2000 steps without errors
- [ ] Markovian KV runs for 2000 steps without errors
- [ ] W&B logs show expected metrics for both conditions
- [ ] Step-0 KL < 0.01 for both conditions
- [ ] G_distal norm > 0 for Markovian KV condition
- [ ] Final reward comparison documented
- [ ] Speed comparison (inference tok/s, training samples/s) documented

---

## Appendix: current smoke #4 v6 metrics (for smoke #5 to beat or match)

Reference numbers from the D5-validated run — use these as the bar
smoke #5 must meet at production scale before calling it a successful
prod smoke.

| Step | Time (s) | Loss | Entropy | Mismatch KL | Grad Norm | Peak Mem | Reward |
|---|---|---|---|---|---|---|---|
| 0 | 65.7 | 0.0009 | 0.3086 | 0.0010 | 0.1013 | 38.4 GiB | 0.266 |
| 1 | 38.5 | 0.0019 | 0.2891 | 0.0009 | 0.1199 | 45.9 GiB | 0.318 |
| 2 | 55.5 | 0.0021 | 0.2891 | 0.0009 | 0.0925 | 45.9 GiB | 0.297 |
| 3 | 508.5 | 0.0020 | 0.2930 | 0.0009 | 0.1075 | 45.9 GiB | 0.472 |
| 4 | 179.8 | 0.0026 | 0.2676 | 0.0009 | 0.1171 | 45.9 GiB | 0.352 |

Notes:
- Mismatch KL holds at 0.0009 across every step = kernel floor (same
  flash_attention_2 kernel as inference, measured offline at 0.0007 via
  `experiments/phase3_kl_test/compare_segforward_modes.py`). Smoke #5
  should see the same or very close.
- Peak memory flat at 45.9 GiB across steps 1-4 = no per-step leak.
- Step 3's 508 s is an orchestrator-side rollout generation stall
  (memory stayed flat during that step, trainer path was idle waiting
  for the batch). Not a training issue. Expected variance.
- Smoke #4b full-context baseline: Mismatch KL 0.0007, peak 74.9 GiB.
  Compaction is about 1 point higher on KL and 29 GiB lower on peak
  memory, which is the tradeoff we designed for.
