#!/bin/bash
# Phase 3.4 inference side — runs on node A inside podman-hpc container.
#
# Usage: srun --nodes=1 --ntasks=1 --gpus-per-node=4 -w <nodeA> \
#            bash /pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/run_inference_node.sh
#
# Or invoked from launch.sh inside an salloc.

set -euo pipefail

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

echo "=== Inference node: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'N/A') ==="

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-phase34-inference \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e HF_HOME="$HOME/hf_cache" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOME":"$HOME" \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$HOME/kv-eviction/experiments/phase3_kl_test" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c '
    set -euo pipefail
    unset NCCL_SOCKET_IFNAME
    source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate
    echo "Python: $(python --version)"
    echo "vLLM:   $(python -c "import vllm; print(vllm.__version__)")"
    echo "GPUs:   $(python -c "import torch; print(torch.cuda.device_count())")"
    echo ""
    python run_inference.py
  '

echo ""
echo "=== Inference node done ==="
