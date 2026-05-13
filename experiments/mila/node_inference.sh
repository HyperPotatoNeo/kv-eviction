#!/usr/bin/env bash
set -euo pipefail

INF_TOML="${1:?Usage: node_inference.sh <inference_toml_path>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: missing virtualenv at $VENV_ACTIVATE"
  exit 1
fi

cd "$REPO_DIR"
source "$VENV_ACTIVATE"
unset NCCL_SOCKET_IFNAME

if ! command -v inference >/dev/null 2>&1; then
  echo "ERROR: missing 'inference' entrypoint in the activated virtualenv"
  exit 1
fi

echo "=== Mila inference server ==="
echo "Host:   $(hostname)"
echo "Repo:   $REPO_DIR"
echo "Config: $INF_TOML"
echo "Python: $(python --version)"
echo "vLLM:   $(python -c 'import vllm; print(vllm.__version__)')"

inference @ "$INF_TOML"
