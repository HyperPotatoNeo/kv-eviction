#!/bin/bash
# Standalone validator sanity check. Runs inside the podman container so
# prime_rl is importable. No GPU needed — pydantic validation only.
set -euo pipefail
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd "$HOME"

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name kv-validator-test \
  --group-add keep-groups --userns keep-id --shm-size=2g \
  -e SCRATCH="$HOME" -e HOME="$HOME" \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOME":"$HOME" \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$HOME/kv-eviction/experiments/phase3_preprod" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c '
    set -euo pipefail
    source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate
    python test_rlconfig_validator.py
  '
