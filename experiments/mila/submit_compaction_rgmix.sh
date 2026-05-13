#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$HOME/kv-runs}"
TIME_LIMIT="${MILA_TIME_LIMIT:-48:00:00}"
ACCOUNT="${MILA_ACCOUNT:-}"
PARTITION="${MILA_PARTITION:-}"
DRY_RUN=0

if [[ -f "$SCRIPT_DIR/local_env.sh" ]]; then
  # Optional local, gitignored Mila credentials/config.
  source "$SCRIPT_DIR/local_env.sh"
fi

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "ERROR: unknown argument: $arg"
      exit 1
      ;;
  esac
done

: "${RG_MIX_DATASET_PATH:?RG_MIX_DATASET_PATH must point to the rg-mix dataset on Mila}"

bash "$SCRIPT_DIR/preflight.sh" compaction_rgmix

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="compaction_rgmix_${STAMP}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

export RG_MIX_DATASET_PATH

SBATCH_ARGS=(
  --parsable
  --job-name="$RUN_NAME"
  --nodes=3
  --ntasks-per-node=1
  --gpus-per-node=4
  --time="$TIME_LIMIT"
  --output="$RUN_DIR/slurm-%j.out"
  --error="$RUN_DIR/slurm-%j.err"
  --export=ALL
)

if [[ -n "$ACCOUNT" ]]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

SBATCH_CMD=(sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/compaction_rgmix.sbatch" "$RUN_DIR")

if (( DRY_RUN == 1 )); then
  printf 'DRY RUN: '
  printf '%q ' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

JOB_ID="$("${SBATCH_CMD[@]}")"
GIT_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"

cat > "$RUN_DIR/status.json" <<EOF
{"experiment":"compaction_rgmix","run_name":"$RUN_NAME","run_dir":"$RUN_DIR","job_id":"$JOB_ID","git_sha":"$GIT_SHA","status":"submitted"}
EOF

echo "Submitted compaction baseline."
echo "  Job ID:  $JOB_ID"
echo "  Run dir: $RUN_DIR"
