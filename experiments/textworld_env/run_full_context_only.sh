#!/bin/bash
# run_full_context_only.sh — Parallel full-context test for the textworld port.
#
# Slim variant of run_inference_test.sh that runs ONLY the full_context
# mode. Intended to run in a separate salloc so it completes concurrently
# with the main launcher's compaction test. Writes results to
# results_full_context.json (may be re-overwritten by the main launcher).
#
# Submit (parallel to the main test):
#   salloc -A m4881 -C "gpu&hbm80g" --qos=interactive --time 1:00:00 \
#          --gpus-per-node 4 -N 1 \
#          bash experiments/textworld_env/run_full_context_only.sh
set -euo pipefail

module unload darshan 2>/dev/null || true

SCRATCH=/pscratch/sd/s/siddart2
KV_DIR="$SCRATCH/kv-eviction"
EXP_DIR="$KV_DIR/experiments/textworld_env"
FULL_CTX_INF_TOML="$KV_DIR/experiments/full_context_textworld/inference.toml"
DATASET="$SCRATCH/datasets/textworld_cooking_mix"

CONTAINER_IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8"
# Distinct container name so we don't collide with the main launcher's
# skyrl-textworld-test container on the other compute node.
CONTAINER="skyrl-textworld-fconly"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
NUM_EXAMPLES=${NUM_EXAMPLES:-100}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}

export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: No SLURM allocation."
    exit 1
fi

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
COMPUTE_NODE="${NODES[0]}"

# Share the runner script from the main launcher — same contract. We reuse
# $EXP_DIR/_eval_runner.sh if it exists (written by the main launcher), or
# fall back to a minimal inline version.
RUNNER="$EXP_DIR/_eval_runner.sh"
if [ ! -f "$RUNNER" ]; then
    cat > "$RUNNER" <<'EOF'
#!/bin/bash
set -euo pipefail
MODE="$1"; INF_TOML="$2"; OUT_JSON="$3"
NUM_EXAMPLES="$4"; MAX_EPISODE_STEPS="$5"; MAX_CONCURRENT="$6"; PADDING_BLOCK_SIZE="$7"
export LD_PRELOAD=$(echo "${LD_PRELOAD:-}" | tr ':' '\n' | grep -v darshan | paste -sd ':')
cd /pscratch/sd/s/siddart2/kv-eviction
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
echo "=== [$MODE] starting vLLM inference server (DP=4) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run inference @ "$INF_TOML" > "/tmp/textworld_${MODE}_inf.log" 2>&1 &
INF_PID=$!
trap 'kill $INF_PID 2>/dev/null || true; wait $INF_PID 2>/dev/null || true' EXIT
WAITED=0; READY=0
while [ $WAITED -lt 900 ]; do
    kill -0 $INF_PID 2>/dev/null || { tail -80 "/tmp/textworld_${MODE}_inf.log"; exit 1; }
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen; then
        READY=1; echo "[$MODE] server ready at ${WAITED}s"; break
    fi
    sleep 5; WAITED=$((WAITED+5)); [ $((WAITED%30)) -eq 0 ] && echo "[$MODE] waiting... (${WAITED}/900s)"
done
[ $READY -eq 1 ] || { tail -80 "/tmp/textworld_${MODE}_inf.log"; exit 1; }
export DUMMY_API_KEY=dummy
python experiments/textworld_env/eval_textworld.py \
    --dataset /pscratch/sd/s/siddart2/datasets/textworld_cooking_mix \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --num-examples "$NUM_EXAMPLES" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --max-concurrent "$MAX_CONCURRENT" \
    --padding-block-size "$PADDING_BLOCK_SIZE" \
    --output-json "$OUT_JSON"
echo "=== [$MODE] eval completed; wrote $OUT_JSON ==="
EOF
    chmod +x "$RUNNER"
fi

echo "=============================================="
echo "full_context PARALLEL test"
echo "=============================================="
echo "Job:            ${SLURM_JOB_ID:-local}"
echo "Login host:     $(hostname)"
echo "Compute node:   $COMPUTE_NODE"
echo "Container:      $CONTAINER"
echo "=============================================="

# Start the container on the compute node
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

teardown() {
    echo "=== tearing down $CONTAINER ==="
    ssh -o StrictHostKeyChecking=no "$COMPUTE_NODE" \
        "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
         podman-hpc rm -f $CONTAINER 2>/dev/null || true"
}
trap teardown EXIT

FULL_CTX_OUT="$EXP_DIR/results_full_context.json"

echo ""
echo "=============================================="
echo "RUNNING: full_context (no eviction)"
echo "=============================================="
ssh -o StrictHostKeyChecking=no "$COMPUTE_NODE" \
    "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
     podman-hpc exec $CONTAINER bash $RUNNER full_context $FULL_CTX_INF_TOML $FULL_CTX_OUT \
     $NUM_EXAMPLES $MAX_EPISODE_STEPS $MAX_CONCURRENT 0"

echo ""
echo "=============================================="
echo "full_context RESULTS"
echo "=============================================="
cat "$FULL_CTX_OUT" 2>/dev/null || echo "(missing)"
