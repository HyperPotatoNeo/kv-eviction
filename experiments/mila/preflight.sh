#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:?Usage: preflight.sh <compaction_rgmix|full_context_rgmix> [--require-allocation]}"
shift || true

REQUIRE_ALLOCATION=0
for arg in "$@"; do
  case "$arg" in
    --require-allocation)
      REQUIRE_ALLOCATION=1
      ;;
    *)
      echo "ERROR: unknown argument: $arg"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

if [[ -f "$SCRIPT_DIR/local_env.sh" ]]; then
  # Optional local, gitignored Mila credentials/config.
  source "$SCRIPT_DIR/local_env.sh"
fi

case "$EXPERIMENT" in
  compaction_rgmix|full_context_rgmix)
    ;;
  *)
    echo "ERROR: unsupported experiment: $EXPERIMENT"
    exit 1
    ;;
esac

: "${WANDB_API_KEY:?WANDB_API_KEY must be set before launch}"
: "${RG_MIX_DATASET_PATH:?RG_MIX_DATASET_PATH must be set before launch}"

if [[ ! -d "$RG_MIX_DATASET_PATH" ]]; then
  echo "ERROR: dataset path does not exist: $RG_MIX_DATASET_PATH"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: missing repo virtualenv python: $PYTHON_BIN"
  exit 1
fi

if [[ ! -x "$REPO_DIR/.venv/bin/inference" ]]; then
  echo "ERROR: missing inference entrypoint: $REPO_DIR/.venv/bin/inference"
  exit 1
fi
if [[ ! -x "$REPO_DIR/.venv/bin/rl" ]]; then
  echo "ERROR: missing trainer entrypoint: $REPO_DIR/.venv/bin/rl"
  exit 1
fi

echo "=== Mila preflight ==="
echo "Experiment:  $EXPERIMENT"
echo "Repo:        $REPO_DIR"
echo "Dataset:     $RG_MIX_DATASET_PATH"
echo "Python:      $PYTHON_BIN"

echo "[1/4] Shell syntax checks"
bash -n \
  "$SCRIPT_DIR"/check_run.sh \
  "$SCRIPT_DIR"/node_inference.sh \
  "$SCRIPT_DIR"/node_trainer.sh \
  "$SCRIPT_DIR"/preflight.sh \
  "$SCRIPT_DIR"/run_compaction_rgmix_interactive.sh \
  "$SCRIPT_DIR"/run_full_context_rgmix_interactive.sh \
  "$SCRIPT_DIR"/run_in_allocation.sh \
  "$SCRIPT_DIR"/submit_compaction_rgmix.sh \
  "$SCRIPT_DIR"/submit_full_context_rgmix.sh \
  "$SCRIPT_DIR"/triage_run.sh \
  "$SCRIPT_DIR"/verify_run.sh \
  "$SCRIPT_DIR"/wait_for_inference.sh \
  "$SCRIPT_DIR"/compaction_rgmix.sbatch \
  "$SCRIPT_DIR"/full_context_rgmix.sbatch

echo "[2/4] Python compile checks"
"$PYTHON_BIN" -m py_compile "$SCRIPT_DIR/triage_run.py"
"$PYTHON_BIN" -m compileall -q \
  "$REPO_DIR/src/kv_eviction" \
  "$REPO_DIR/prime-rl/src/prime_rl" \
  "$REPO_DIR/vllm/vllm"

echo "[3/4] Experiment config presence"
for path in \
  "$REPO_DIR/experiments/$EXPERIMENT/rl.toml" \
  "$REPO_DIR/experiments/$EXPERIMENT/inference.toml"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required config: $path"
    exit 1
  fi
done

if (( REQUIRE_ALLOCATION == 1 )); then
  echo "[4/4] Live allocation checks"
  : "${SLURM_JOB_ID:?run this preflight inside an active allocation}"
  : "${SLURM_JOB_NODELIST:?run this preflight inside an active allocation}"
  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  if [[ "${#NODES[@]}" -lt 3 ]]; then
    echo "ERROR: expected at least 3 allocated nodes, got ${#NODES[@]}"
    exit 1
  fi
  for node in "${NODES[@]:0:3}"; do
    echo "  Checking node $node"
    if ! srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 --exclusive -w "$node" \
      bash -lc "test -d '$RG_MIX_DATASET_PATH'"; then
      echo "ERROR: dataset path is not visible on node $node"
      exit 1
    fi
    gpu_count="$(
      srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 --exclusive -w "$node" \
        bash -lc 'nvidia-smi -L | wc -l' 2>/dev/null | tr -d '[:space:]'
    )"
    if [[ -z "$gpu_count" || "$gpu_count" -lt 4 ]]; then
      echo "ERROR: expected at least 4 visible GPUs on $node, got ${gpu_count:-0}"
      exit 1
    fi
  done
fi

echo "Preflight passed."
