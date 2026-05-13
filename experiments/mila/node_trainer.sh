#!/usr/bin/env bash
set -euo pipefail

TOML_PATH="${1:?Usage: node_trainer.sh <resolved_rl_toml_path>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: missing virtualenv at $VENV_ACTIVATE"
  exit 1
fi

: "${WANDB_API_KEY:?WANDB_API_KEY must be set before launching the trainer}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"
unset NCCL_SOCKET_IFNAME

if ! command -v rl >/dev/null 2>&1; then
  echo "ERROR: missing 'rl' entrypoint in the activated virtualenv"
  exit 1
fi

echo "=== Mila trainer ==="
echo "Host:   $(hostname)"
echo "Repo:   $REPO_DIR"
echo "Config: $TOML_PATH"
echo "Python: $(python --version)"

rl @ "$TOML_PATH"
