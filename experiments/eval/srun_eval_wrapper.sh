#!/bin/bash
set -euo pipefail
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

echo "=== $(date) on $(hostname) ==="
nvidia-smi -L

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-eval \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e HF_HOME="$HOME/hf_cache" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOME":"$HOME" \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$HOME/kv-eviction/experiments/eval" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash /pscratch/sd/s/siddart2/kv-eviction/experiments/eval/_container_run.sh

echo "=== done $(date) ==="
