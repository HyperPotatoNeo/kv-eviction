#!/bin/bash
# compaction_rgmix — trainer-side launcher (FSDP2 DP=4, 4 GPUs).
# Runs INSIDE the skyrl container on the trainer node.
set -e

TOML_PATH="${1:?Usage: node2_trainer.sh <resolved_rl_toml_path>}"

# Strip darshan from LD_PRELOAD — host I/O profiler not available in container
export LD_PRELOAD=$(echo "$LD_PRELOAD" | tr ':' '\n' | grep -v darshan | paste -sd ':')

cd /pscratch/sd/s/siddart2/kv-eviction
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
# WANDB_API_KEY is passed in by launch.sh via podman -e; require it here.
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in the container environment}"

# rg-mix-env is installed into the kv-eviction venv directly as
# rg_mix_env-0.1.4 (uv pip install of the wheel at
# /pscratch/sd/s/siddart2/mkv-rl/experiments/rg_mix/dist/), so
# verifiers.load_environment("rg-mix-env") resolves without PYTHONPATH.

echo "=== compaction_rgmix trainer (2-1 split: 4 GPUs, DP=4) ==="
echo "Host:   $(hostname)"
echo "Config: $TOML_PATH"

uv run rl @ "$TOML_PATH"
