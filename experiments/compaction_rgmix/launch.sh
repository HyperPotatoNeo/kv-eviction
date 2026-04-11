#!/bin/bash
# compaction_rgmix — production RL run with KV cache compaction.
#
# 3-node 2-1 split:
#   Node 0: vLLM DP=4 inference (compaction window=4096 stride=512)
#   Node 1: vLLM DP=4 inference (compaction window=4096 stride=512)
#   Node 2: FSDP2 DP=4 trainer (bptt_segments=1, M3 semantics)
#
# Submit via:
#   sbatch -A m4881 -C "gpu&hbm80g" --qos=premium --time 48:00:00 \
#          --gpus-per-node 4 --nodes=3 \
#          experiments/compaction_rgmix/launch.sh
#
# Interactive equivalent (debug only, max 4h):
#   salloc -A m4881 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 \
#          --gpus-per-node 4 -N 3
#   bash experiments/compaction_rgmix/launch.sh
set -e
set -o pipefail  # so `ssh ... | tee` propagates the ssh/trainer exit
                 # status into $PIPESTATUS[0] below, not tee's always-0.

#SBATCH --job-name=compaction_rgmix
#SBATCH --output=/pscratch/sd/s/siddart2/kv-eviction/experiments/compaction_rgmix/slurm_%j.out
#SBATCH --error=/pscratch/sd/s/siddart2/kv-eviction/experiments/compaction_rgmix/slurm_%j.err

# Unload darshan I/O profiler — it injects libdarshan.so.0 via LD_PRELOAD
# which doesn't exist inside containers, breaking C extension imports.
module unload darshan 2>/dev/null || true

SCRATCH=/pscratch/sd/s/siddart2
KV_DIR="$SCRATCH/kv-eviction"
EXP_DIR="$KV_DIR/experiments/compaction_rgmix"
CONTAINER_IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8"
CONTAINER_INF0="skyrl-cmrg-inf0"
CONTAINER_INF1="skyrl-cmrg-inf1"
CONTAINER_TRAIN="skyrl-cmrg-train"

TOML_TEMPLATE="$EXP_DIR/rl.toml"
TOML_RESOLVED="$EXP_DIR/resolved_rl_${SLURM_JOB_ID:-local}.toml"
INF_TOML="$EXP_DIR/inference.toml"

# ─── Step 1: Validate SLURM allocation ───
if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: No SLURM allocation. Submit via sbatch or allocate:"
    echo "  sbatch -A <account> -C 'gpu&hbm80g' --qos=premium --time 48:00:00 --gpus-per-node 4 --nodes=3 $0"
    exit 1
fi

# WANDB_API_KEY must be set in the submitter environment (passed
# through to the container via -e WANDB_API_KEY below).
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is not set. Export it before submitting."
    exit 1
fi

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#NODES[@]}" -lt 3 ]; then
    echo "ERROR: Need 3 nodes, got ${#NODES[@]}"
    exit 1
fi

NODE_INF0="${NODES[0]}"
NODE_INF1="${NODES[1]}"
NODE_TRAIN="${NODES[2]}"

echo "==========================================="
echo "compaction_rgmix prod run (3 nodes, 48h)"
echo "==========================================="
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "RL config:   $TOML_TEMPLATE"
echo "INF config:  $INF_TOML"
echo "Node 0 (inf): $NODE_INF0"
echo "Node 1 (inf): $NODE_INF1"
echo "Node 2 (trn): $NODE_TRAIN"
echo "==========================================="

# ─── Step 2: Resolve inference URLs in RL TOML ───
sed -e "s/__INFERENCE_NODE_0__/$NODE_INF0/g" \
    -e "s/__INFERENCE_NODE_1__/$NODE_INF1/g" \
    "$TOML_TEMPLATE" > "$TOML_RESOLVED"
echo "Resolved RL config: $TOML_RESOLVED"

# ─── Step 3: Setup detached containers on all three nodes ───
setup_container() {
    local NODE=$1 CNAME=$2
    echo "Setting up $CNAME on $NODE..."
    ssh "$NODE" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && cd $SCRATCH && podman-hpc run --rm -d --user \$(id -u):\$(id -g) --replace --name $CNAME --group-add keep-groups --userns keep-id --net=host --gpu --nccl --shm-size=8g -e SCRATCH -e HOME -e WANDB_API_KEY=$WANDB_API_KEY -v $SCRATCH:$SCRATCH -v /global/homes/s/siddart2:/global/homes/s/siddart2 -w $KV_DIR $CONTAINER_IMAGE sleep infinity"
}

setup_container "$NODE_INF0"  "$CONTAINER_INF0"  &
setup_container "$NODE_INF1"  "$CONTAINER_INF1"  &
setup_container "$NODE_TRAIN" "$CONTAINER_TRAIN" &
wait
echo "Containers ready."

# ─── Step 4: Launch inference on both inference nodes ───
INF0_LOG="$EXP_DIR/inference0_${SLURM_JOB_ID:-local}.log"
INF1_LOG="$EXP_DIR/inference1_${SLURM_JOB_ID:-local}.log"

echo "=== Launching inference on $NODE_INF0 ==="
ssh "$NODE_INF0" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_INF0 bash $EXP_DIR/node1_inference.sh $INF_TOML" > "$INF0_LOG" 2>&1 &
PID_INF0=$!

echo "=== Launching inference on $NODE_INF1 ==="
ssh "$NODE_INF1" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_INF1 bash $EXP_DIR/node1_inference.sh $INF_TOML" > "$INF1_LOG" 2>&1 &
PID_INF1=$!

# ─── Step 5: Wait for BOTH inference servers to be ready ───
echo "Waiting for both inference servers..."
MAX_WAIT=900
WAITED=0
READY0=0
READY1=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 $PID_INF0 2>/dev/null; then
        echo "ERROR: Inference on $NODE_INF0 died. Last 50 lines:"
        tail -50 "$INF0_LOG"
        kill $PID_INF1 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 $PID_INF1 2>/dev/null; then
        echo "ERROR: Inference on $NODE_INF1 died. Last 50 lines:"
        tail -50 "$INF1_LOG"
        kill $PID_INF0 2>/dev/null || true
        exit 1
    fi
    if [ $READY0 -eq 0 ] && ssh "$NODE_INF0" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_INF0 curl -s http://localhost:8000/v1/models" 2>/dev/null | grep -q "Qwen"; then
        READY0=1
        echo "  Node 0 ready at ${WAITED}s"
    fi
    if [ $READY1 -eq 0 ] && ssh "$NODE_INF1" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_INF1 curl -s http://localhost:8000/v1/models" 2>/dev/null | grep -q "Qwen"; then
        READY1=1
        echo "  Node 1 ready at ${WAITED}s"
    fi
    if [ $READY0 -eq 1 ] && [ $READY1 -eq 1 ]; then
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "  waiting... ($WAITED/$MAX_WAIT s) [inf0=$READY0 inf1=$READY1]"
done

if [ $READY0 -ne 1 ] || [ $READY1 -ne 1 ]; then
    echo "ERROR: Inference not ready after ${MAX_WAIT}s (inf0=$READY0 inf1=$READY1). Tails:"
    echo "--- inf0 ---"; tail -50 "$INF0_LOG"
    echo "--- inf1 ---"; tail -50 "$INF1_LOG"
    kill $PID_INF0 $PID_INF1 2>/dev/null || true
    exit 1
fi
echo "Both inference servers ready."

# ─── Step 6: Launch trainer on Node 2 ───
#
# IMPORTANT: we redirect trainer output straight to file (not through
# `tee`) so $! and `wait $PID` cleanly reflect the trainer's exit code.
# A prior version used `ssh ... | tee LOG &` which made $! the tee PID —
# so a crashed trainer looked like a successful run to SLURM.
TRAIN_LOG="$EXP_DIR/trainer_${SLURM_JOB_ID:-local}.log"
echo "=== Launching trainer on $NODE_TRAIN ==="
ssh "$NODE_TRAIN" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_TRAIN bash $EXP_DIR/node2_trainer.sh $TOML_RESOLVED" > "$TRAIN_LOG" 2>&1 &
PID_TRAIN=$!

# Live-tail the trainer log to the launch.sh stdout so SLURM %j.out
# shows training progress alongside the inference spinup summary.
tail -f "$TRAIN_LOG" &
PID_TAIL=$!

echo "==========================================="
echo "compaction_rgmix launched!"
echo "  Inf0:    $NODE_INF0 (PID $PID_INF0, log: $(basename $INF0_LOG))"
echo "  Inf1:    $NODE_INF1 (PID $PID_INF1, log: $(basename $INF1_LOG))"
echo "  Trainer: $NODE_TRAIN (PID $PID_TRAIN, log: $(basename $TRAIN_LOG))"
echo "==========================================="

set +e  # don't let wait's non-zero kill us before cleanup runs
wait $PID_TRAIN
TRAIN_EXIT=$?
set -e
kill $PID_TAIL 2>/dev/null || true

echo "Trainer exited with status $TRAIN_EXIT"
echo "Stopping inference servers..."
kill $PID_INF0 $PID_INF1 2>/dev/null || true
wait $PID_INF0 $PID_INF1 2>/dev/null || true

echo "Cleaning up containers..."
ssh "$NODE_INF0"  "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc rm -f $CONTAINER_INF0"  2>/dev/null || true
ssh "$NODE_INF1"  "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc rm -f $CONTAINER_INF1"  2>/dev/null || true
ssh "$NODE_TRAIN" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc rm -f $CONTAINER_TRAIN" 2>/dev/null || true

echo "Done (trainer exit $TRAIN_EXIT)."
exit $TRAIN_EXIT
