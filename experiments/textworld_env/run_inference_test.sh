#!/bin/bash
# run_inference_test.sh — Single-node smoke test for the textworld port.
#
# Runs from the login node inside a SLURM allocation. Parses the compute
# node hostname from $SLURM_JOB_NODELIST and SSHes into it to set up the
# skyrl container and run both test modes back-to-back:
#   1. compaction (turn-based eviction, client-side block-16 padding)
#   2. full_context (no compaction, no padding)
# Each test: spawn DP=4 vLLM → wait ready → run 100-sample eval → kill.
# Results are written to $EXP_DIR/results_{compaction,full_context}.json.
#
# Submit:
#   salloc -A m4881 -C "gpu&hbm80g" --qos=interactive --time 2:00:00 \
#          --gpus-per-node 4 -N 1 \
#          bash experiments/textworld_env/run_inference_test.sh
set -euo pipefail

module unload darshan 2>/dev/null || true

SCRATCH=/pscratch/sd/s/siddart2
KV_DIR="$SCRATCH/kv-eviction"
EXP_DIR="$KV_DIR/experiments/textworld_env"
COMPACTION_INF_TOML="$KV_DIR/experiments/compaction_textworld/inference.toml"
FULL_CTX_INF_TOML="$KV_DIR/experiments/full_context_textworld/inference.toml"
DATASET="$SCRATCH/datasets/textworld_cooking_mix"

CONTAINER_IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8"
CONTAINER="skyrl-textworld-test"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
NUM_EXAMPLES=${NUM_EXAMPLES:-100}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
EVAL_SET_JSON=${EVAL_SET_JSON:-$EXP_DIR/eval_sets/textworld_eval_100_seed42.json}
PADDING_BLOCK_SIZE=16

export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: No SLURM allocation. Submit via salloc first."
    exit 1
fi

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#NODES[@]}" -lt 1 ]; then
    echo "ERROR: No compute nodes in allocation"
    exit 1
fi
COMPUTE_NODE="${NODES[0]}"

# Compute-node runner script lives on shared $SCRATCH so SSH-ed invocations
# on the compute node can read it. Rewritten on every launcher run.
RUNNER="$EXP_DIR/_eval_runner.sh"

echo "=============================================="
echo "textworld port inference smoke test"
echo "=============================================="
echo "Job:            ${SLURM_JOB_ID:-local}"
echo "Login host:     $(hostname)"
echo "Compute node:   $COMPUTE_NODE"
echo "Dataset:        $DATASET"
echo "Model:          $MODEL"
echo "Samples:        $NUM_EXAMPLES"
echo "Max turns:      $MAX_EPISODE_STEPS"
echo "Concurrent:     $MAX_CONCURRENT"
echo "=============================================="

# ─── Step 1: Write the compute-node runner script ───
cat > "$RUNNER" <<'EOF'
#!/bin/bash
# Runs INSIDE the skyrl container on the compute node.
set -euo pipefail
MODE="$1"                 # "compaction" or "full_context"
INF_TOML="$2"
OUT_JSON="$3"
NUM_EXAMPLES="$4"
MAX_EPISODE_STEPS="$5"
MAX_CONCURRENT="$6"
PADDING_BLOCK_SIZE="$7"   # 16 for compaction, 0 for full_context baseline
EVAL_SET_JSON="$8"

export LD_PRELOAD=$(echo "${LD_PRELOAD:-}" | tr ':' '\n' | grep -v darshan | paste -sd ':')
cd /pscratch/sd/s/siddart2/kv-eviction
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

echo "=== [$MODE] starting vLLM inference server (DP=4) ==="
echo "    config: $INF_TOML"
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run inference @ "$INF_TOML" \
    > "/tmp/textworld_${MODE}_inf.log" 2>&1 &
INF_PID=$!

cleanup() {
    echo "=== [$MODE] stopping inference server (pid $INF_PID) ==="
    kill $INF_PID 2>/dev/null || true
    wait $INF_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "=== [$MODE] waiting for server readiness ==="
MAX_WAIT=900
WAITED=0
READY=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 $INF_PID 2>/dev/null; then
        echo "[$MODE] ERROR: inference server died. Tail:"
        tail -80 "/tmp/textworld_${MODE}_inf.log"
        exit 1
    fi
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen"; then
        READY=1
        echo "[$MODE] server ready at ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    [ $((WAITED % 30)) -eq 0 ] && echo "[$MODE] waiting... (${WAITED}/${MAX_WAIT}s)"
done

if [ $READY -ne 1 ]; then
    echo "[$MODE] ERROR: server not ready after ${MAX_WAIT}s"
    tail -80 "/tmp/textworld_${MODE}_inf.log"
    exit 1
fi

echo "=== [$MODE] running eval_textworld.py ==="
export DUMMY_API_KEY=dummy
python experiments/textworld_env/eval_textworld.py \
    --dataset /pscratch/sd/s/siddart2/datasets/textworld_cooking_mix \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --num-examples "$NUM_EXAMPLES" \
    --eval-source eval \
    --eval-set-json "$EVAL_SET_JSON" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --max-concurrent "$MAX_CONCURRENT" \
    --padding-block-size "$PADDING_BLOCK_SIZE" \
    --output-json "$OUT_JSON"
echo "=== [$MODE] eval completed; wrote $OUT_JSON ==="
EOF
chmod +x "$RUNNER"

# ─── Step 2: Start the detached container on the compute node ───
echo "[setup] starting container $CONTAINER on $COMPUTE_NODE"
ssh -o StrictHostKeyChecking=no "$COMPUTE_NODE" bash -s <<EOSSH
set -e
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN
cd $SCRATCH
podman-hpc rm -f $CONTAINER 2>/dev/null || true
podman-hpc run --rm -d \
    --user "\$(id -u):\$(id -g)" --replace --name $CONTAINER \
    --group-add keep-groups --userns keep-id --net=host \
    --gpu --nccl --shm-size=8g \
    -e SCRATCH -e HOME \
    -v $SCRATCH:$SCRATCH \
    -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
    -w $KV_DIR \
    $CONTAINER_IMAGE sleep infinity
EOSSH
echo "[setup] container $CONTAINER up"

exec_in_container() {
    local mode=$1 inf_toml=$2 out_json=$3 padding=$4
    ssh -o StrictHostKeyChecking=no "$COMPUTE_NODE" \
        "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
         podman-hpc exec $CONTAINER bash $RUNNER $mode $inf_toml $out_json \
         $NUM_EXAMPLES $MAX_EPISODE_STEPS $MAX_CONCURRENT $padding $EVAL_SET_JSON"
}

teardown() {
    echo ""
    echo "=== tearing down container ==="
    ssh -o StrictHostKeyChecking=no "$COMPUTE_NODE" \
        "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
         podman-hpc rm -f $CONTAINER 2>/dev/null || true"
}
trap teardown EXIT

COMPACTION_OUT="$EXP_DIR/results_compaction.json"
FULL_CTX_OUT="$EXP_DIR/results_full_context.json"

# ─── Step 3: Compaction test ───
echo ""
echo "=============================================="
echo "TEST 1 of 2: compaction (turn-based eviction)"
echo "=============================================="
exec_in_container compaction "$COMPACTION_INF_TOML" "$COMPACTION_OUT" "$PADDING_BLOCK_SIZE"

# Let GPU memory settle between runs.
sleep 10

# ─── Step 4: Full-context test ───
echo ""
echo "=============================================="
echo "TEST 2 of 2: full_context (no eviction)"
echo "=============================================="
exec_in_container full_context "$FULL_CTX_INF_TOML" "$FULL_CTX_OUT" 0

echo ""
echo "=============================================="
echo "RESULTS"
echo "=============================================="
echo "--- compaction ---"
cat "$COMPACTION_OUT" 2>/dev/null || echo "(missing)"
echo ""
echo "--- full_context ---"
cat "$FULL_CTX_OUT" 2>/dev/null || echo "(missing)"
