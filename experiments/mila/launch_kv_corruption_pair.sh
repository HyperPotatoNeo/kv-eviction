#!/usr/bin/env bash
set -euo pipefail

# Launch a matched AM vs non-AM AIME corruption pair.
#
# Required/important environment variables:
#   CORRUPTION=noise|shuffle
#   PROB=0.50
#   NUM_EXAMPLES=30
#   MAX_TOKENS=16384
#   MAX_CONCURRENT=16
#
# Noise-only:
#   NOISE_STD=0.01
#   NOISE_TARGET=both
#   NOISE_MODE=gaussian|zero
#   NOISE_REGION=all|old_context_only
#   NOISE_KEEP_RECENT_TOKENS=0
#   NOISE_PROTECT_SYNTHETIC=false|true
#
# Shuffle-only:
#   SHUFFLE_REGION=all|old_context_only
#   SHUFFLE_KEEP_RECENT_TOKENS=0
#   SHUFFLE_PROTECT_SYNTHETIC=false|true
#   SHUFFLE_KV_ONLY=false|true
#
# The two runs differ only by AM compaction controls. Corruption seed, model,
# sampling, prompt protection, and vLLM runtime settings are matched.

REPO_DIR="/home/mila/d/dane.malenfant/kv-eviction"
cd "$REPO_DIR"

CORRUPTION="${CORRUPTION:-noise}"
PROB="${PROB:-0.50}"
NUM_EXAMPLES="${NUM_EXAMPLES:-30}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
NUM_GPUS="${NUM_GPUS:-4}"
NOISE_STD="${NOISE_STD:-0.01}"
NOISE_TARGET="${NOISE_TARGET:-both}"
NOISE_MODE="${NOISE_MODE:-gaussian}"
NOISE_REGION="${NOISE_REGION:-all}"
NOISE_KEEP_RECENT_TOKENS="${NOISE_KEEP_RECENT_TOKENS:-0}"
NOISE_PROTECT_SYNTHETIC="${NOISE_PROTECT_SYNTHETIC:-false}"
SHUFFLE_REGION="${SHUFFLE_REGION:-all}"
SHUFFLE_KEEP_RECENT_TOKENS="${SHUFFLE_KEEP_RECENT_TOKENS:-0}"
SHUFFLE_PROTECT_SYNTHETIC="${SHUFFLE_PROTECT_SYNTHETIC:-false}"
SHUFFLE_KV_ONLY="${SHUFFLE_KV_ONLY:-false}"
SEED="${SEED:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_PROB="${PROB//./p}"
SAFE_STD="${NOISE_STD//./p}"
CONFIG_DIR="$REPO_DIR/experiments/generated/kv_corruption_pairs/${STAMP}_${CORRUPTION}_p${SAFE_PROB}"
if [[ "$CORRUPTION" == "noise" ]]; then
  CONFIG_DIR="${CONFIG_DIR}_std${SAFE_STD}"
fi
mkdir -p "$CONFIG_DIR"

case "$CORRUPTION" in
  noise)
    AM_EXTRA=$(
      cat <<EOF
noise_control_chunk_size = ${CHUNK_SIZE}
noise_control_probability = ${PROB}
noise_control_std = ${NOISE_STD}
noise_control_seed = ${SEED}
noise_control_target = "${NOISE_TARGET}"
noise_control_mode = "${NOISE_MODE}"
noise_control_region = "${NOISE_REGION}"
noise_control_keep_recent_tokens = ${NOISE_KEEP_RECENT_TOKENS}
noise_control_protect_synthetic = ${NOISE_PROTECT_SYNTHETIC}
EOF
    )
    FULL_EXTRA="$AM_EXTRA"
    FULL_MODE="kv_noise"
    ;;
  shuffle)
    AM_EXTRA=$(
      cat <<EOF
shuffle_control_chunk_size = ${CHUNK_SIZE}
shuffle_control_probability = ${PROB}
shuffle_control_seed = ${SEED}
shuffle_control_region = "${SHUFFLE_REGION}"
shuffle_control_keep_recent_tokens = ${SHUFFLE_KEEP_RECENT_TOKENS}
shuffle_control_protect_synthetic = ${SHUFFLE_PROTECT_SYNTHETIC}
shuffle_control_kv_only = ${SHUFFLE_KV_ONLY}
EOF
    )
    FULL_EXTRA="$AM_EXTRA"
    FULL_MODE="full_context"
    ;;
  *)
    echo "ERROR: CORRUPTION must be noise or shuffle"
    exit 1
    ;;
esac

cat >"$CONFIG_DIR/am.toml" <<EOF
seed = ${SEED}

enable_prefix_caching = false

[server]
host = "0.0.0.0"
port = 8000

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"
max_model_len = ${MAX_TOKENS}
enforce_eager = false

[parallel]
dp = ${NUM_GPUS}
tp = 1

[vllm_extra]
block_size = 16
async_scheduling = false
compaction_window_size = 4096
compaction_stride = 1024
compaction_strategy = "attention_matching"
attention_backend = "FLEX_ATTENTION"
attention_matching_max_queries_per_kv_head = 64
attention_matching_query_source = "random_queries"
attention_matching_protect_user_prompts = "first_user"
${AM_EXTRA}
gpu_memory_utilization = ${GPU_MEMORY_UTILIZATION}
EOF

cat >"$CONFIG_DIR/full_context.toml" <<EOF
seed = ${SEED}

enable_prefix_caching = false

[server]
host = "0.0.0.0"
port = 8000

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"
max_model_len = ${MAX_TOKENS}
enforce_eager = false

[parallel]
dp = ${NUM_GPUS}
tp = 1

[vllm_extra]
block_size = 16
async_scheduling = false
attention_backend = "FLEX_ATTENTION"
attention_matching_protect_user_prompts = "first_user"
${FULL_EXTRA}
gpu_memory_utilization = ${GPU_MEMORY_UTILIZATION}
EOF

SESSION_TAG="kv_${CORRUPTION}_p${SAFE_PROB}"
if [[ "$CORRUPTION" == "noise" ]]; then
  SESSION_TAG="${SESSION_TAG}_${NOISE_MODE}_${NOISE_TARGET}_std${SAFE_STD}_r${NOISE_KEEP_RECENT_TOKENS}"
else
  SESSION_TAG="${SESSION_TAG}_r${SHUFFLE_KEEP_RECENT_TOKENS}"
  if [[ "$SHUFFLE_KV_ONLY" == "true" ]]; then
    SESSION_TAG="${SESSION_TAG}_kvonly"
  fi
fi
SESSION_TAG="${SESSION_TAG}_${STAMP}"

tmux new-session -d -s "${SESSION_TAG}_am" \
  "cd '$REPO_DIR' && MODE=attention_matching PROFILE=real NUM_GPUS='$NUM_GPUS' NUM_EXAMPLES='$NUM_EXAMPLES' ROLLOUTS=1 MAX_CONCURRENT='$MAX_CONCURRENT' MAX_TOKENS='$MAX_TOKENS' INF_TOML_OVERRIDE='$CONFIG_DIR/am.toml' bash experiments/mila/run_aime_salloc.sh"

tmux new-session -d -s "${SESSION_TAG}_full" \
  "cd '$REPO_DIR' && MODE='$FULL_MODE' PROFILE=real NUM_GPUS='$NUM_GPUS' NUM_EXAMPLES='$NUM_EXAMPLES' ROLLOUTS=1 MAX_CONCURRENT='$MAX_CONCURRENT' MAX_TOKENS='$MAX_TOKENS' INF_TOML_OVERRIDE='$CONFIG_DIR/full_context.toml' bash experiments/mila/run_aime_salloc.sh"

echo "config_dir=$CONFIG_DIR"
echo "am_session=${SESSION_TAG}_am"
echo "full_context_session=${SESSION_TAG}_full"
