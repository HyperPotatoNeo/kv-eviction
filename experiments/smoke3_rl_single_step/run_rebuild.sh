#!/bin/bash
# Launch rebuild_venv.sh inside the skyrl container on this allocated node.
set -euo pipefail

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

echo "=== Rebuilding kv-eviction venv on $(hostname) ==="
nvidia-smi -L 2>/dev/null | head -4 || echo "(no nvidia-smi here)"

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-rebuild \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache \
  -e VLLM_USE_PRECOMPILED=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOME":"$HOME" \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$HOME/kv-eviction" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash /pscratch/sd/s/siddart2/kv-eviction/experiments/smoke3_rl_single_step/rebuild_venv.sh

echo ""
echo "=== Rebuild done on $(hostname) ==="
