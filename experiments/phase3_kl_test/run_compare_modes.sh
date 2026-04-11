#!/bin/bash
# Launch compare_segforward_modes.py inside the skyrl container on the
# allocated compute node. Must be launched inside an salloc that has at
# least 1 GPU node.
#
# IMPORTANT: a bare `salloc ... bash script.sh` runs the script on the
# LOGIN node (which on Perlmutter has a 40 GB A100), not on the allocated
# 80 GB compute node. We srun below to force execution onto the compute
# node.
set -euo pipefail

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: SLURM_JOB_NODELIST not set. Launch inside salloc." >&2
    exit 1
fi
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
COMPUTE_NODE="${NODES[0]}"

export SAMPLE_IDX="${1:-0}"
export NUM_EVENTS="${2:-1}"

echo "=== compare_segforward_modes ==="
echo "  launcher host: $(hostname)"
echo "  compute node:  $COMPUTE_NODE"
echo "  SAMPLE_IDX:    $SAMPLE_IDX"
echo "  NUM_EVENTS:    $NUM_EVENTS"

srun --nodes=1 --ntasks=1 -w "$COMPUTE_NODE" --gpus-per-node=4 \
  --export=ALL,SAMPLE_IDX,NUM_EVENTS \
  bash /pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/_compare_modes_inner.sh

echo ""
echo "=== done ==="
