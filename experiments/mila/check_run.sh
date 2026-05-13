#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?Usage: check_run.sh <run_dir|job_id>}"
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
EXPERIMENT="$(read_status_field experiment)"
STATUS="$(read_status_field status)"

SLURM_STATE="$(squeue -h -j "$JOB_ID" -o "%T" 2>/dev/null | head -n1 || true)"
if [[ -z "$SLURM_STATE" ]]; then
  SLURM_STATE="$(sacct -n -X -j "$JOB_ID" --format=State 2>/dev/null | head -n1 | awk '{print $1}' || true)"
fi
if [[ -z "$SLURM_STATE" ]]; then
  SLURM_STATE="UNKNOWN"
fi

echo "Run dir:    $RUN_DIR"
echo "Experiment: $EXPERIMENT"
echo "Job ID:     $JOB_ID"
echo "Status:     $STATUS"
echo "SLURM:      $SLURM_STATE"

for path in \
  "$RUN_DIR/trainer.log" \
  "$RUN_DIR/inference0.log" \
  "$RUN_DIR/inference1.log" \
  "$RUN_DIR/resolved_rl.toml" \
  "$RUN_DIR/resolved_inference.toml"; do
  if [[ -e "$path" ]]; then
    echo "Found:      $path"
  else
    echo "Missing:    $path"
  fi
done

if [[ -f "$RUN_DIR/trainer.log" ]]; then
  echo ""
  echo "=== trainer.log tail ==="
  tail -n 30 "$RUN_DIR/trainer.log" || true
fi
