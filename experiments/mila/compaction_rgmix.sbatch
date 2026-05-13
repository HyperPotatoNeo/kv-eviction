#!/bin/bash
#SBATCH --job-name=compaction_rgmix
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

set -euo pipefail

RUN_DIR="${1:?Usage: compaction_rgmix.sbatch <run_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MILA_LAUNCH_MODE=sbatch

bash "$SCRIPT_DIR/run_in_allocation.sh" compaction_rgmix "$RUN_DIR"
