#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-compaction_rgmix}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$HOME/kv-runs}"
LAST_MESSAGE_PATH="$RUN_ROOT/codex_${EXPERIMENT}_last_message.txt"

if [[ -f "$SCRIPT_DIR/local_env.sh" ]]; then
  # Optional local, gitignored Mila credentials/config.
  source "$SCRIPT_DIR/local_env.sh"
fi

case "$EXPERIMENT" in
  compaction_rgmix)
    LAUNCHER="bash experiments/mila/run_compaction_rgmix_interactive.sh"
    ;;
  full_context_rgmix)
    LAUNCHER="bash experiments/mila/run_full_context_rgmix_interactive.sh"
    ;;
  *)
    echo "ERROR: unsupported experiment: $EXPERIMENT"
    echo "Usage: $0 [compaction_rgmix|full_context_rgmix]"
    exit 1
    ;;
esac

: "${SLURM_JOB_ID:?run this script inside an active salloc allocation}"
: "${SLURM_JOB_NODELIST:?run this script inside an active salloc allocation}"
: "${RG_MIX_DATASET_PATH:?RG_MIX_DATASET_PATH must be set before launch}"

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI is not installed or not on PATH"
  exit 1
fi

LOGIN_STATUS="$(codex login status 2>&1 || true)"
if [[ "$LOGIN_STATUS" != *"Logged in"* ]]; then
  echo "ERROR: Codex is not logged in. Run 'codex' first."
  exit 1
fi

mkdir -p "$RUN_ROOT"

CODEX_ARGS=(
  exec
  -C "$REPO_DIR"
  --add-dir "$RUN_ROOT"
  -o "$LAST_MESSAGE_PATH"
)

if [[ -n "${CODEX_MODEL:-}" ]]; then
  CODEX_ARGS+=(-m "$CODEX_MODEL")
fi

if [[ "${CODEX_DANGEROUS_AUTO:-0}" == "1" ]]; then
  CODEX_ARGS+=(--dangerously-bypass-approvals-and-sandbox)
else
  CODEX_ARGS+=(--full-auto)
fi

echo "Starting Codex Mila loop"
echo "  Experiment: $EXPERIMENT"
echo "  Repo:       $REPO_DIR"
echo "  Run root:   $RUN_ROOT"
echo "  Allocation: $SLURM_JOB_ID"
echo "  Final note: $LAST_MESSAGE_PATH"
if [[ "${CODEX_DANGEROUS_AUTO:-0}" == "1" ]]; then
  echo "  Mode:       dangerous auto"
else
  echo "  Mode:       full-auto sandboxed"
fi

cat <<EOF | codex "${CODEX_ARGS[@]}" -
You are operating inside an already-active 3-node Mila SLURM allocation and must manage the ${EXPERIMENT} experiment in this repository until the success criteria are satisfied.

Use this exact workflow and refuse to call the experiment valid unless the final verification passes:

1. Run:
   \`bash experiments/mila/preflight.sh ${EXPERIMENT} --require-allocation\`
   If preflight fails, inspect the failure, patch code or config in this repo if you can fix it, and rerun preflight. Do not launch the experiment until preflight passes.

2. Launch the experiment in the background from the repo root using:
   \`${LAUNCHER}\`
   Capture its stdout/stderr to a launch log under \`${RUN_ROOT}\` so you can keep monitoring the run while the launcher stays alive.

3. Determine the run directory for the launch. Prefer parsing it from the launcher output. If needed, infer it from the newest \`${EXPERIMENT}_*\` directory under \`${RUN_ROOT}\`.

4. Monitor the live run using all of these:
   - \`bash experiments/mila/check_run.sh "\$RUN_DIR"\`
   - \`tail -n 200 "\$RUN_DIR/trainer.log"\`
   - \`tail -n 120 "\$RUN_DIR/inference0.log"\`
   - \`tail -n 120 "\$RUN_DIR/inference1.log"\`
   Poll about once per minute while the run is active. Do not busy-loop.

5. If the logs show a clear coding/runtime failure or the launcher exits unsuccessfully, stop treating that run as viable. Inspect the logs and any available \`triage_summary.json\`, patch code/config in the repo, rerun preflight, and relaunch a fresh run.

6. When the run ends, execute:
   \`bash experiments/mila/verify_run.sh "\$RUN_DIR"\`

7. If \`verify_run.sh\` fails for any reason, inspect:
   - \`"\$RUN_DIR/triage_summary.json"\`
   - trainer/inference logs
   - resolved TOMLs if relevant
   Patch the repo, rerun preflight, and relaunch from scratch.

8. Only stop when \`verify_run.sh\` exits 0.

Important rules:
- Treat warnings from \`triage_run.sh\` as failures. A suspicious run is not a valid baseline.
- Use the run directory, not only the SLURM job id, as the primary identity for a run.
- Avoid repeating an identical failed run without changing something. If a failure repeats, make a concrete patch or report a concrete external blocker.
- If you are blocked by something external you cannot fix from the repo or shell session, such as an expired allocation, missing dataset, missing credentials, or missing dependency, stop and report that blocker precisely.
- Keep your progress updates concise.
- In your final message, report the successful run directory and the exact verification command that passed.
EOF
