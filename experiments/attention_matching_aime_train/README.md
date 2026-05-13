# Attention-Matching RL Smoke Runs

This directory contains pull-and-run configs for testing RL training while the
rollout model uses vLLM attention-matching KV compaction.

## What Runs

`uv run rl @ <config>` starts the full prime-rl stack:

- a vLLM inference server with `compaction_strategy = "attention_matching"`
- the orchestrator, which samples rollouts from that compacting server
- the FSDP2 RL trainer, which replays `compaction_events` through
  `kv_eviction.segmented_forward`

This trains the policy/model weights from compacted-inference rollouts. It does
not train the attention-matching compactor itself.

## Quick Start

From the repo root after setup:

```bash
source .venv/bin/activate

# Fast wiring check. Requires 4 GPUs: 2 inference, 2 trainer.
uv run rl @ experiments/attention_matching_aime_train/rl_preflight.toml

# Short 8k smoke with eval.
uv run rl @ experiments/attention_matching_aime_train/rl_8k_smoke.toml
```

For a longer 16k run:

```bash
uv run rl @ experiments/attention_matching_aime_train/rl.toml
```

## Required Invariants

The trainer-side compaction config must mirror the vLLM inference config:

```toml
[trainer.compaction]
window_size = 1024
stride = 256
block_size = 16

[inference.vllm_extra]
block_size = 16
compaction_strategy = "attention_matching"
compaction_window_size = 1024
compaction_stride = 256
```

The trainer must use:

```toml
[trainer.model]
impl = "hf"
attn = "flash_attention_2"
```

`async_scheduling` must stay disabled for compacting vLLM runs.

## What To Check

Useful signals in the logs and W&B:

- inference logs show `compaction_strategy = attention_matching`
- rollout outputs carry `compaction_events` and `num_compaction_events`
- trainer batches in compaction runs route through `segmented_forward`
- `loss/mismatch_kl_mean` stays near the kernel floor rather than exploding

If `compaction_events` are missing, the most likely issue is that
`kv_eviction` was not importable in the orchestrator/env-server process, so the
verifiers response hooks were not installed.
