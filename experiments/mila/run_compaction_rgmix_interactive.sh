#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$HOME/kv-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="compaction_rgmix_${STAMP}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"

if [[ -f "$SCRIPT_DIR/local_env.sh" ]]; then
  # Optional local, gitignored Mila credentials/config.
  source "$SCRIPT_DIR/local_env.sh"
fi

: "${SLURM_JOB_ID:?run this script inside an active salloc allocation}"
: "${SLURM_JOB_NODELIST:?run this script inside an active salloc allocation}"
: "${RG_MIX_DATASET_PATH:?RG_MIX_DATASET_PATH must point to the rg-mix dataset on Mila}"

mkdir -p "$RUN_DIR"

GIT_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"
cat > "$RUN_DIR/status.json" <<EOF
{"experiment":"compaction_rgmix","run_name":"$RUN_NAME","run_dir":"$RUN_DIR","job_id":"$SLURM_JOB_ID","git_sha":"$GIT_SHA","status":"submitted","launch_mode":"interactive_allocation"}
EOF

bash "$SCRIPT_DIR/preflight.sh" compaction_rgmix --require-allocation

echo "Launching compaction baseline in allocation $SLURM_JOB_ID"
echo "  Run dir: $RUN_DIR"

export MILA_LAUNCH_MODE=interactive_allocation

bash "$SCRIPT_DIR/run_in_allocation.sh" compaction_rgmix "$RUN_DIR"
