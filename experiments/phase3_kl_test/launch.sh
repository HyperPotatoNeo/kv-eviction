#!/bin/bash
# Phase 3.4 KL test orchestrator.
#
# Step 1: User allocates 2 interactive GPU nodes:
#   salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 \
#          --gpus-per-node 4 -N 2
#
# Step 2: Inside the allocation, run this script:
#   bash /pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/launch.sh
#
# This script:
#   1. Identifies the two allocated nodes (node A = inference, node B = trainer)
#   2. Runs run_inference_node.sh on node A via srun (compaction + baseline
#      rollouts, saved to results/)
#   3. Runs run_trainer_node.sh on node B via srun (torchrun DP=4, reads
#      rollouts, computes KL)
#
# Inference and trainer phases run sequentially because the test is offline
# (inference must complete before KL comparison can start).

set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# Enumerate allocated nodes
if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: not running inside an salloc. Allocate 2 nodes first:"
    echo "  salloc -A m5017 -C 'gpu&hbm80g' --qos=interactive --time 4:00:00 --gpus-per-node 4 -N 2"
    exit 1
fi

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#NODES[@]}" -lt 2 ]; then
    echo "ERROR: allocation has ${#NODES[@]} nodes, need >= 2"
    exit 1
fi

NODE_A="${NODES[0]}"
NODE_B="${NODES[1]}"
echo "=== Phase 3.4 KL test orchestrator ==="
echo "Inference node (A): $NODE_A"
echo "Trainer node   (B): $NODE_B"
echo ""

# Fresh results dir
rm -rf results
mkdir -p results

# ─── Phase 1: inference on node A ───
echo "--- Phase 1: inference on $NODE_A ---"
srun --nodes=1 --ntasks=1 -w "$NODE_A" \
     --gpus-per-node=4 \
     --output="results/inference_${NODE_A}.out" \
     --error="results/inference_${NODE_A}.err" \
     bash "$SCRIPT_DIR/run_inference_node.sh"
echo "--- Phase 1 done ---"
echo ""

# ─── Phase 2: trainer on node B ───
echo "--- Phase 2: trainer DP=4 on $NODE_B ---"
srun --nodes=1 --ntasks=1 -w "$NODE_B" \
     --gpus-per-node=4 \
     --output="results/trainer_${NODE_B}.out" \
     --error="results/trainer_${NODE_B}.err" \
     bash "$SCRIPT_DIR/run_trainer_node.sh"
echo "--- Phase 2 done ---"

echo ""
echo "=== Phase 3.4 KL test complete ==="
echo "Results: $SCRIPT_DIR/results/"
echo ""
if [ -f "$SCRIPT_DIR/results/kl_results.json" ]; then
    echo "--- KL summary ---"
    python3 -c "
import json
r = json.load(open('$SCRIPT_DIR/results/kl_results.json'))
print(f\"Baseline    mean_abs_logratio: {r['baseline']['mean_abs_log_ratio']:.5f}  max: {r['baseline']['max_abs_log_ratio']:.5f}\")
print(f\"Compaction  mean_abs_logratio: {r['compaction']['mean_abs_log_ratio']:.5f}  max: {r['compaction']['max_abs_log_ratio']:.5f}\")
print(f\"ratio compaction/baseline: {r['ratio_compaction_over_baseline']:.2f}x\")
"
fi
