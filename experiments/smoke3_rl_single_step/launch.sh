#!/bin/bash
# Smoke #3: end-to-end single RL step with KV compaction.
#
# 1-1 node split (2-node salloc):
#   Node 0: vLLM DP=4 inference server (--compaction-window-size 4096)
#   Node 1: FSDP2 DP=4 trainer (bptt_segments=1, M3 semantics)
#
# Step 1: Allocate 2 interactive GPU nodes (m4881 or m5017):
#   salloc -A m4881 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 \
#          --gpus-per-node 4 -N 2
#
# Step 2: Inside the allocation, run this script:
#   bash /pscratch/sd/s/siddart2/kv-eviction/experiments/smoke3_rl_single_step/launch.sh
set -e

# Unload darshan I/O profiler — it injects libdarshan.so.0 via LD_PRELOAD
# which doesn't exist inside containers, breaking C extension imports.
module unload darshan 2>/dev/null || true

SCRATCH=/pscratch/sd/s/siddart2
KV_DIR="$SCRATCH/kv-eviction"
EXP_DIR="$KV_DIR/experiments/smoke3_rl_single_step"
CONTAINER_IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8"
CONTAINER_INF="skyrl-smoke3-inf"
CONTAINER_TRAIN="skyrl-smoke3-train"

TOML_TEMPLATE="$EXP_DIR/rl_smoke3.toml"
TOML_RESOLVED="$EXP_DIR/resolved_rl_smoke3_${SLURM_JOB_ID:-local}.toml"
INF_TOML="$EXP_DIR/inference_smoke3.toml"

# ─── Step 1: Validate SLURM allocation ───
if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: No SLURM allocation. Allocate first:"
    echo "  salloc -A m4881 -C 'gpu&hbm80g' --qos=interactive --time 4:00:00 --gpus-per-node 4 -N 2"
    exit 1
fi

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#NODES[@]}" -lt 2 ]; then
    echo "ERROR: Need 2 nodes, got ${#NODES[@]}"
    exit 1
fi

NODE0="${NODES[0]}"  # Inference
NODE1="${NODES[1]}"  # Trainer

echo "==========================================="
echo "Smoke #3: single end-to-end RL step"
echo "==========================================="
echo "RL config:  $TOML_TEMPLATE"
echo "INF config: $INF_TOML"
echo "Node 0 (inference): $NODE0"
echo "Node 1 (trainer):   $NODE1"
echo "==========================================="

# ─── Step 2: Resolve inference URL in RL TOML ───
sed -e "s/__INFERENCE_NODE__/$NODE0/g" "$TOML_TEMPLATE" > "$TOML_RESOLVED"
echo "Resolved RL config: $TOML_RESOLVED"

# ─── Step 3: Setup detached containers on both nodes ───
setup_container() {
    local NODE=$1 CNAME=$2
    echo "Setting up $CNAME on $NODE..."
    ssh "$NODE" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && cd $SCRATCH && podman-hpc run --rm -d --user \$(id -u):\$(id -g) --replace --name $CNAME --group-add keep-groups --userns keep-id --net=host --gpu --nccl --shm-size=8g -e SCRATCH -e HOME -e WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8 -v $SCRATCH:$SCRATCH -v /global/homes/s/siddart2:/global/homes/s/siddart2 -w $KV_DIR $CONTAINER_IMAGE sleep infinity"
}

setup_container "$NODE0" "$CONTAINER_INF" &
setup_container "$NODE1" "$CONTAINER_TRAIN" &
wait
echo "Containers ready."

# ─── Step 4: Launch inference on Node 0 ───
echo "=== Launching inference on $NODE0 ==="
ssh "$NODE0" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_INF bash $EXP_DIR/node1_inference.sh $INF_TOML" > "$EXP_DIR/inference_${SLURM_JOB_ID:-local}.log" 2>&1 &
PID_INF=$!

# ─── Step 5: Wait for inference to be ready ───
echo "Waiting for inference server..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 $PID_INF 2>/dev/null; then
        echo "ERROR: Inference died. Last 50 lines:"
        tail -50 "$EXP_DIR/inference_${SLURM_JOB_ID:-local}.log"
        exit 1
    fi
    if ssh "$NODE0" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_INF curl -s http://localhost:8000/v1/models" 2>/dev/null | grep -q "Qwen"; then
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "  waiting... ($WAITED/$MAX_WAIT s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Inference not ready after ${MAX_WAIT}s. Last 50 lines:"
    tail -50 "$EXP_DIR/inference_${SLURM_JOB_ID:-local}.log"
    kill $PID_INF 2>/dev/null
    exit 1
fi
echo "Inference ready on $NODE0!"

# ─── Step 6: Launch trainer on Node 1 ───
echo "=== Launching trainer on $NODE1 ==="
ssh "$NODE1" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc exec $CONTAINER_TRAIN bash $EXP_DIR/node2_trainer.sh $TOML_RESOLVED" 2>&1 | tee "$EXP_DIR/trainer_${SLURM_JOB_ID:-local}.log" &
PID_TRAIN=$!

echo "==========================================="
echo "Smoke #3 launched!"
echo "  Inference: $NODE0 (PID $PID_INF, log: inference_${SLURM_JOB_ID:-local}.log)"
echo "  Trainer:   $NODE1 (PID $PID_TRAIN, log: trainer_${SLURM_JOB_ID:-local}.log)"
echo "==========================================="

wait $PID_TRAIN
TRAIN_EXIT=$?

echo "Stopping inference..."
kill $PID_INF 2>/dev/null; wait $PID_INF 2>/dev/null || true

echo "Cleaning up containers..."
ssh "$NODE0" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc rm -f $CONTAINER_INF" 2>/dev/null || true
ssh "$NODE1" "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && podman-hpc rm -f $CONTAINER_TRAIN" 2>/dev/null || true

echo "Done (exit $TRAIN_EXIT)."
exit $TRAIN_EXIT
