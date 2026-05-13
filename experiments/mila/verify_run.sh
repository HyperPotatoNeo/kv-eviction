#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?Usage: verify_run.sh <run_dir|job_id>}"
RUN_ROOT="${RUN_ROOT:-$HOME/kv-runs}"

resolve_run_dir() {
  local target="$1"
  if [[ -d "$target" ]]; then
    printf '%s\n' "$target"
    return 0
  fi
  python - "$RUN_ROOT" "$target" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1]).expanduser()
target = sys.argv[2]
for status_path in run_root.glob("*/status.json"):
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        continue
    if data.get("job_id") == target:
        print(status_path.parent)
        sys.exit(0)
sys.exit(1)
PY
}

RUN_DIR="$(resolve_run_dir "$TARGET")" || {
  echo "ERROR: could not resolve run directory for $TARGET"
  exit 1
}
STATUS_PATH="$RUN_DIR/status.json"

if [[ ! -f "$STATUS_PATH" ]]; then
  echo "ERROR: missing status file: $STATUS_PATH"
  exit 1
fi

read_status_field() {
  local field="$1"
  python - "$STATUS_PATH" "$field" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1]).read())
value = data.get(sys.argv[2], "")
print(value)
PY
}

JOB_ID="$(read_status_field job_id)"
STATUS="$(read_status_field status)"
LAUNCH_MODE="$(read_status_field launch_mode)"

SLURM_STATE="$(squeue -h -j "$JOB_ID" -o "%T" 2>/dev/null | head -n1 || true)"
if [[ -z "$SLURM_STATE" ]]; then
  SLURM_STATE="$(sacct -n -X -j "$JOB_ID" --format=State 2>/dev/null | head -n1 | awk '{print $1}' || true)"
fi
if [[ -z "$SLURM_STATE" ]]; then
  SLURM_STATE="UNKNOWN"
fi

echo "Verifying run: $RUN_DIR"
echo "  status.json: $STATUS"
echo "  launch_mode: $LAUNCH_MODE"
echo "  SLURM:       $SLURM_STATE"

if [[ "$LAUNCH_MODE" == "interactive_allocation" ]]; then
  if [[ "$STATUS" != "completed" ]]; then
    echo "ERROR: interactive allocation run is not marked completed in status.json"
    exit 1
  fi
else
  if [[ "$SLURM_STATE" != "COMPLETED" ]]; then
    echo "ERROR: SLURM state is not COMPLETED"
    exit 1
  fi
fi

required_paths=(
  "$RUN_DIR/status.json"
  "$RUN_DIR/base_rl.toml"
  "$RUN_DIR/base_inference.toml"
  "$RUN_DIR/resolved_rl.toml"
  "$RUN_DIR/resolved_inference.toml"
  "$RUN_DIR/trainer.log"
  "$RUN_DIR/inference0.log"
  "$RUN_DIR/inference1.log"
  "$RUN_DIR/outputs"
  "$RUN_DIR/outputs/logs"
  "$RUN_DIR/outputs/checkpoints"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path"
    exit 1
  fi
done

if ! find "$RUN_DIR/outputs/checkpoints" -maxdepth 2 -name STABLE | grep -q .; then
  echo "ERROR: no stable checkpoints found under $RUN_DIR/outputs/checkpoints"
  exit 1
fi

if grep -q "Traceback (most recent call last)" "$RUN_DIR/trainer.log"; then
  echo "ERROR: traceback found in trainer log"
  exit 1
fi
if grep -q "Traceback (most recent call last)" "$RUN_DIR/inference0.log"; then
  echo "ERROR: traceback found in inference0 log"
  exit 1
fi
if grep -q "Traceback (most recent call last)" "$RUN_DIR/inference1.log"; then
  echo "ERROR: traceback found in inference1 log"
  exit 1
fi

if ! grep -q "Step " "$RUN_DIR/trainer.log"; then
  echo "ERROR: trainer log does not contain step progress"
  exit 1
fi

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/triage_run.sh" --require-complete "$RUN_DIR"

echo "Verification passed."
echo "  Run dir:  $RUN_DIR"
echo "  Job ID:   $JOB_ID"
echo "  Checkpts: $(find "$RUN_DIR/outputs/checkpoints" -maxdepth 1 -type d -name 'step_*' | wc -l | tr -d ' ')"
