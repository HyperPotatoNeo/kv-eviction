#!/bin/bash
# Phase 3.4 trainer side — runs on node B inside podman-hpc container.
#
# Launches torchrun with nproc_per_node=4 to test DP=4. Expects the rollout
# JSONs to already exist in results/ (run the inference node first).
#
# Usage: srun --nodes=1 --ntasks=1 --gpus-per-node=4 -w <nodeB> \
#            bash /pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/run_trainer_node.sh

set -euo pipefail

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

echo "=== Trainer node: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A') ==="

# Sanity check: rollout JSONs must exist
if [ ! -f "$HOME/kv-eviction/experiments/phase3_kl_test/results/rollouts_compaction.json" ]; then
    echo "ERROR: rollouts_compaction.json missing. Run run_inference_node.sh first."
    exit 1
fi
if [ ! -f "$HOME/kv-eviction/experiments/phase3_kl_test/results/rollouts_baseline.json" ]; then
    echo "ERROR: rollouts_baseline.json missing. Run run_inference_node.sh first."
    exit 1
fi

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-phase34-trainer \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e HF_HOME="$HOME/hf_cache" \
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
    echo "torch:  $(python -c "import torch; print(torch.__version__)")"
    echo "GPUs:   $(python -c "import torch; print(torch.cuda.device_count())")"
    echo ""
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node=4 \
      run_kl_test.py
  '

echo ""
echo "=== Trainer node done ==="
