#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:?Usage: run_in_allocation.sh <compaction_rgmix|full_context_rgmix> <run_dir>}"
RUN_DIR="${2:?Usage: run_in_allocation.sh <compaction_rgmix|full_context_rgmix> <run_dir>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATUS_PATH="$RUN_DIR/status.json"
OUTPUT_DIR="$RUN_DIR/outputs"
INF0_LOG="$RUN_DIR/inference0.log"
INF1_LOG="$RUN_DIR/inference1.log"
TRAIN_LOG="$RUN_DIR/trainer.log"
READY_TIMEOUT="${INFERENCE_READY_TIMEOUT:-900}"
TRAIN_EXIT=1

: "${RG_MIX_DATASET_PATH:?RG_MIX_DATASET_PATH must be exported before launch}"
: "${SLURM_JOB_ID:?run this script inside an active SLURM allocation}"
: "${SLURM_JOB_NODELIST:?run this script inside an active SLURM allocation}"

case "$EXPERIMENT" in
  compaction_rgmix|full_context_rgmix)
    EXP_DIR="$REPO_DIR/experiments/$EXPERIMENT"
    ;;
  *)
    echo "ERROR: unsupported experiment: $EXPERIMENT"
    exit 1
    ;;
esac

BASE_RL="$EXP_DIR/rl.toml"
BASE_INF="$EXP_DIR/inference.toml"
RESOLVED_RL="$RUN_DIR/resolved_rl.toml"
RESOLVED_INF="$RUN_DIR/resolved_inference.toml"

write_status() {
  local status="$1"
  cat > "$STATUS_PATH" <<EOF
{"experiment":"$EXPERIMENT","run_dir":"$RUN_DIR","job_id":"${SLURM_JOB_ID}","repo_dir":"$REPO_DIR","status":"$status","trainer_log":"$TRAIN_LOG","inference0_log":"$INF0_LOG","inference1_log":"$INF1_LOG","resolved_rl":"$RESOLVED_RL","resolved_inference":"$RESOLVED_INF","output_dir":"$OUTPUT_DIR","launch_mode":"${MILA_LAUNCH_MODE:-allocation}"}
EOF
}

cleanup() {
  local exit_code=$?
  local final_status="failed"
  if [[ "${TRAIN_EXIT}" -eq 0 && $exit_code -eq 0 ]]; then
    final_status="completed"
  fi
  write_status "$final_status"
  if [[ -n "${PID_INF0:-}" ]]; then
    kill "$PID_INF0" 2>/dev/null || true
  fi
  if [[ -n "${PID_INF1:-}" ]]; then
    kill "$PID_INF1" 2>/dev/null || true
  fi
  wait "${PID_INF0:-}" "${PID_INF1:-}" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$RUN_DIR"
cp "$BASE_RL" "$RUN_DIR/base_rl.toml"
cp "$BASE_INF" "$RUN_DIR/base_inference.toml"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [[ "${#NODES[@]}" -lt 3 ]]; then
  echo "ERROR: expected at least 3 allocated nodes, got ${#NODES[@]}"
  exit 1
fi

NODE_INF0="${NODES[0]}"
NODE_INF1="${NODES[1]}"
NODE_TRAIN="${NODES[2]}"

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

DATASET_PATH_ESCAPED="$(escape_sed_replacement "$RG_MIX_DATASET_PATH")"
OUTPUT_DIR_ESCAPED="$(escape_sed_replacement "$OUTPUT_DIR")"

sed \
  -e "s|__INFERENCE_NODE_0__|$NODE_INF0|g" \
  -e "s|__INFERENCE_NODE_1__|$NODE_INF1|g" \
  -e "s|^output_dir = .*|output_dir = \"$OUTPUT_DIR_ESCAPED\"|" \
  -e "s|/pscratch/sd/s/siddart2/datasets/rg_mix_7500|$DATASET_PATH_ESCAPED|g" \
  "$BASE_RL" > "$RESOLVED_RL"

cp "$BASE_INF" "$RESOLVED_INF"

echo "=== Mila experiment launch ==="
echo "Experiment:  $EXPERIMENT"
echo "Job:         ${SLURM_JOB_ID}"
echo "Run dir:     $RUN_DIR"
echo "Inference 0: $NODE_INF0"
echo "Inference 1: $NODE_INF1"
echo "Trainer:     $NODE_TRAIN"
echo "Dataset:     $RG_MIX_DATASET_PATH"

write_status "running"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 --exclusive -w "$NODE_INF0" \
  bash "$SCRIPT_DIR/node_inference.sh" "$RESOLVED_INF" > "$INF0_LOG" 2>&1 &
PID_INF0=$!

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 --exclusive -w "$NODE_INF1" \
  bash "$SCRIPT_DIR/node_inference.sh" "$RESOLVED_INF" > "$INF1_LOG" 2>&1 &
PID_INF1=$!

bash "$SCRIPT_DIR/wait_for_inference.sh" "$NODE_INF0" "$NODE_INF1" 8000 "$READY_TIMEOUT"

set +e
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 --exclusive -w "$NODE_TRAIN" \
  bash "$SCRIPT_DIR/node_trainer.sh" "$RESOLVED_RL" > "$TRAIN_LOG" 2>&1
TRAIN_EXIT=$?
set -e

exit "$TRAIN_EXIT"
