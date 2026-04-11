#!/bin/bash
# Runs on the compute node. Launches podman-hpc with the mode-comparison
# script inside the container. Sourced via srun from run_compare_modes.sh.
set -euo pipefail

: "${SAMPLE_IDX:?not set}"
: "${NUM_EVENTS:?not set}"

echo "--- compute node: $(hostname) ---"
nvidia-smi -L | head -4 || true

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-modecompare \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e HF_HOME="$HOME/hf_cache" \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAMPLE_IDX="$SAMPLE_IDX" \
  -e NUM_EVENTS="$NUM_EVENTS" \
  -v "$HOME":"$HOME" \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$HOME/kv-eviction/experiments/phase3_kl_test" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c '
    set -euo pipefail
    unset NCCL_SOCKET_IFNAME
    source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate
    echo "--- inside container, visible GPUs ---"
    python -c "import torch; props=torch.cuda.get_device_properties(0); print(props.name, round(props.total_memory/1e9, 1), \"GB\")"
    python compare_segforward_modes.py \
        --sample-idx "$SAMPLE_IDX" \
        --num-events "$NUM_EVENTS" \
        --output "results/compare_modes_sample_${SAMPLE_IDX}_events_${NUM_EVENTS}.json"
  '
