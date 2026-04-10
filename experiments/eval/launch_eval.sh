#!/bin/bash
# Launch compaction eval on Perlmutter.
#
# Step 1: Get a node
#   salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4 -N 1
#
# Step 2: Run this script ON the allocated node
#   bash /pscratch/sd/s/siddart2/kv-eviction/experiments/eval/launch_eval.sh

set -euo pipefail

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

echo "=== Node: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A') ==="

podman-hpc run --rm -it \
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
  bash -c '
    set -euo pipefail
    unset NCCL_SOCKET_IFNAME
    source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate
    echo "Python: $(python --version)"
    echo "vLLM:   $(python -c "import vllm; print(vllm.__version__)")"
    echo "GPUs:   $(python -c "import torch; print(torch.cuda.device_count())")"
    echo ""
    python run_compaction_eval.py
  '

echo ""
echo "=== Done ==="
echo "Results: $HOME/kv-eviction/experiments/eval/results/"
