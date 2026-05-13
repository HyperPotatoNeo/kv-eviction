#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/mila/d/dane.malenfant/kv-eviction"
cd "$REPO_DIR"

SESSION_STAMP="$(date +%Y%m%d_%H%M%S)"
export KV_NOISE_LAUNCH_STAMP="$SESSION_STAMP"
NUM_GPUS="${NUM_GPUS:-1}"
if [ "$NUM_GPUS" = "4" ]; then
  DEFAULT_INF_TOML="$REPO_DIR/experiments/kv_noise_aime/inference_dp4.toml"
else
  DEFAULT_INF_TOML="$REPO_DIR/experiments/kv_noise_aime/inference.toml"
fi
SALLOC_NODELIST_ARGS=()
if [ -n "${ALLOC_NODELIST:-}" ]; then
  SALLOC_NODELIST_ARGS=(--nodelist="$ALLOC_NODELIST")
fi

salloc \
  --partition=short-unkillable \
  "${SALLOC_NODELIST_ARGS[@]}" \
  --gres=gpu:a100l:"$NUM_GPUS" \
  --cpus-per-task=24 \
  --mem=128G \
  --time=3:00:00 \
  srun \
  --ntasks=1 \
  --gres=gpu:a100l:"$NUM_GPUS" \
  --cpus-per-task=24 \
  bash -lc '
    set -euo pipefail
    cd /home/mila/d/dane.malenfant/kv-eviction
    echo "Running on $(hostname)"
    nvidia-smi
    source .venv/bin/activate
    if [ -f experiments/mila/local_env.sh ]; then
      source experiments/mila/local_env.sh
    fi
    export WANDB_PROJECT=kv-eviction
    export NUM_EXAMPLES="${NUM_EXAMPLES:-30}"
    export ROLLOUTS="${ROLLOUTS:-1}"
    export MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
    export MAX_TOKENS="${MAX_TOKENS:-16384}"
    export INF_TOML_OVERRIDE="'"$DEFAULT_INF_TOML"'"
    bash experiments/mila/run_aime_eval.sh kv_noise real
  '
