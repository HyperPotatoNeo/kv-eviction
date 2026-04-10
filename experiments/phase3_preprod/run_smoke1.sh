#!/bin/bash
set -euo pipefail
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

echo "=== Smoke 1: $(hostname) ==="
nvidia-smi -L | head -4

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-smoke1 \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e HF_HOME="$HOME/hf_cache" \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$HOME":"$HOME" \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$HOME/kv-eviction/experiments/phase3_preprod" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c '
    set -euo pipefail
    unset NCCL_SOCKET_IFNAME
    source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate
    python smoke1_backward.py --num-events -1 --sample-idx 0 \
      --output results/smoke1_result.json
  '
echo "=== Smoke 1 done ==="
