#!/bin/bash
# One-shot wrapper to run kv-eviction/setup.sh inside the skyrl container
# on a single interactive compute node.
set -euo pipefail

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

echo "=== Running kv-eviction setup.sh on $(hostname) ==="
nvidia-smi -L | head -4

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-setup \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOME":"$HOME" \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$HOME/kv-eviction" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash /pscratch/sd/s/siddart2/kv-eviction/setup.sh

echo ""
echo "=== Setup done on $(hostname) ==="
