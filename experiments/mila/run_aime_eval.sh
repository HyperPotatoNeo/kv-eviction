#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full_context}"
PROFILE="${2:-real}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -z "${WANDB_API_KEY:-}" && -f "$SCRIPT_DIR/local_env.sh" ]]; then
  # Default local developer path: opt into W&B automatically when a local
  # environment file exists, without requiring every launch shell to source it.
  # shellcheck disable=SC1090
  source "$SCRIPT_DIR/local_env.sh"
fi
RUN_ROOT="${RUN_ROOT:-$HOME/kv-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUN_ROOT/aime_${MODE}_${PROFILE}_${STAMP}"
INFER_LOG="$RUN_DIR/inference.log"
EVAL_LOG="$RUN_DIR/eval.log"
TIMING_LOG="$RUN_DIR/wall_clock_times.txt"
READY_TIMEOUT="${INFERENCE_READY_TIMEOUT:-900}"
RUN_START_EPOCH="$(date +%s)"

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
ENV_ID="primeintellect/aime2024"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

case "$MODE" in
  full_context)
    DEFAULT_INF_TOML="$REPO_DIR/experiments/full_context_rgmix/inference.toml"
    ;;
  compaction)
    DEFAULT_INF_TOML="$REPO_DIR/experiments/compaction_rgmix/inference.toml"
    ;;
  attention_matching)
    DEFAULT_INF_TOML="$REPO_DIR/experiments/attention_matching_aime/inference.toml"
    ;;
  kv_corruption)
    DEFAULT_INF_TOML="$REPO_DIR/experiments/kv_corruption_aime/inference.toml"
    ;;
  kv_noise)
    DEFAULT_INF_TOML="$REPO_DIR/experiments/kv_noise_aime/inference.toml"
    ;;
  shuffle_robustness)
    DEFAULT_INF_TOML="$REPO_DIR/experiments/shuffle_robustness_aime/inference.toml"
    ;;
  *)
    echo "ERROR: mode must be 'full_context', 'compaction', 'attention_matching', 'kv_corruption', 'kv_noise', or 'shuffle_robustness'"
    exit 1
    ;;
esac

INF_TOML="${INF_TOML_OVERRIDE:-$DEFAULT_INF_TOML}"

case "$PROFILE" in
  smoke)
    NUM_EXAMPLES="${NUM_EXAMPLES:-2}"
    ROLLOUTS="${ROLLOUTS:-1}"
    MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
    TEMPERATURE="${TEMPERATURE:-1.0}"
    TOP_P="${TOP_P:-0.96}"
    ;;
  real)
    NUM_EXAMPLES="${NUM_EXAMPLES:-30}"
    ROLLOUTS="${ROLLOUTS:-4}"
    MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
    TEMPERATURE="${TEMPERATURE:-1.0}"
    TOP_P="${TOP_P:-0.96}"
    ;;
  *)
    echo "ERROR: profile must be 'smoke' or 'real'"
    exit 1
    ;;
esac

mkdir -p "$RUN_DIR"

cleanup() {
  local exit_code=$?
  if [[ -n "${MONITOR_PID:-}" && $exit_code -ne 0 ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [[ -n "${INF_PID:-}" ]]; then
    kill "$INF_PID" 2>/dev/null || true
    wait "$INF_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT

if [[ ! -f "$INF_TOML" ]]; then
  echo "ERROR: missing inference config: $INF_TOML"
  exit 1
fi

if [[ ! -f "$REPO_DIR/.venv/bin/activate" ]]; then
  echo "ERROR: missing virtualenv at $REPO_DIR/.venv"
  exit 1
fi

echo "=== AIME eval launch ==="
echo "Run dir:      $RUN_DIR"
echo "Mode:         $MODE"
echo "Profile:      $PROFILE"
echo "Model:        $MODEL_NAME"
echo "Environment:  $ENV_ID"
echo "Inference:    $INF_TOML"
echo "Examples:     $NUM_EXAMPLES"
echo "Rollouts:     $ROLLOUTS"
echo "Temperature:  $TEMPERATURE"
echo "Top-p:        $TOP_P"
echo "Max tokens:   $MAX_TOKENS"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "W&B:          enabled"
else
  echo "W&B:          disabled"
fi

cp "$INF_TOML" "$RUN_DIR/$(basename "$INF_TOML")"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  bash -lc "
    set -euo pipefail
    cd '$REPO_DIR'
    source .venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export WANDB_PROJECT='${WANDB_PROJECT:-kv-eviction}'
    if [[ -n '${WANDB_ENTITY:-}' ]]; then
      export WANDB_ENTITY='${WANDB_ENTITY:-}'
    fi
    python scripts/wandb_aime_eval_monitor.py \
      --run-dir '$RUN_DIR' \
      --mode '$MODE' \
      --profile '$PROFILE' \
      --model '$MODEL_NAME' \
      --expected-count '$NUM_EXAMPLES' \
      --poll-seconds 10 \
      --launcher-pid '$$'
  " >"$RUN_DIR/wandb.log" 2>&1 &
  MONITOR_PID=$!
fi

SERVER_START_EPOCH="$(date +%s)"
bash -lc "
  set -euo pipefail
  cd '$REPO_DIR'
  source .venv/bin/activate
  export VLLM_API_KEY='$VLLM_API_KEY'
  unset NCCL_SOCKET_IFNAME
  inference @ '$INF_TOML'
" >"$INFER_LOG" 2>&1 &
INF_PID=$!

echo "Waiting for inference server..."
waited=0
until curl -fsS \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  if ! kill -0 "$INF_PID" 2>/dev/null; then
    echo "ERROR: inference server exited early. Tail:"
    tail -n 80 "$INFER_LOG" || true
    exit 1
  fi
  if (( waited >= READY_TIMEOUT )); then
    echo "ERROR: inference server not ready after ${READY_TIMEOUT}s"
    tail -n 80 "$INFER_LOG" || true
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done

READY_EPOCH="$(date +%s)"
echo "Inference ready after ${waited}s"

EVAL_START_EPOCH="$(date +%s)"
set +e
bash -lc "
  set -euo pipefail
  cd '$REPO_DIR'
  source .venv/bin/activate
  export VLLM_API_KEY='$VLLM_API_KEY'
  python scripts/vf_eval_with_kv_eviction.py '$ENV_ID' \
    --provider local \
    --model '$MODEL_NAME' \
    --num-examples '$NUM_EXAMPLES' \
    --rollouts-per-example '$ROLLOUTS' \
    --max-concurrent '$MAX_CONCURRENT' \
    --sampling-args '{\"max_tokens\":${MAX_TOKENS},\"temperature\":${TEMPERATURE},\"top_p\":${TOP_P}}' \
    --state-columns compaction_events,num_compaction_events,shuffle_events,num_shuffle_events,noise_events,num_noise_events \
    --disable-env-server \
    --save-results \
    --output-dir '$RUN_DIR/vf_eval'
" | tee "$EVAL_LOG"
EVAL_STATUS="${PIPESTATUS[0]}"
set -e
EVAL_END_EPOCH="$(date +%s)"
if (( EVAL_STATUS != 0 )); then
  exit "$EVAL_STATUS"
fi

python3 "$REPO_DIR/scripts/compaction_success_by_count.py" "$RUN_DIR/vf_eval" \
  | tee "$RUN_DIR/compaction_success_by_count.txt"
python3 "$REPO_DIR/scripts/shuffle_success_by_count.py" "$RUN_DIR/vf_eval" \
  | tee "$RUN_DIR/shuffle_success_by_count.txt"
python3 "$REPO_DIR/scripts/noise_success_by_count.py" "$RUN_DIR/vf_eval" \
  | tee "$RUN_DIR/noise_success_by_count.txt"

END_EPOCH="$(date +%s)"
{
  echo "mode=$MODE"
  echo "profile=$PROFILE"
  echo "run_dir=$RUN_DIR"
  echo "server_startup_seconds=$((READY_EPOCH - SERVER_START_EPOCH))"
  echo "eval_wall_clock_seconds=$((EVAL_END_EPOCH - EVAL_START_EPOCH))"
  echo "total_wall_clock_seconds=$((END_EPOCH - RUN_START_EPOCH))"
} | tee "$TIMING_LOG"

if [[ -n "${MONITOR_PID:-}" ]]; then
  wait "$MONITOR_PID" || true
fi

echo "Eval finished. Results under $RUN_DIR"
