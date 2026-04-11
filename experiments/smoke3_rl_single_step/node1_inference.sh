#!/bin/bash
# Smoke #3 — Node 0 inference side (DP=4, 4 GPUs).
# Runs INSIDE the skyrl container on the inference node.
set -e

INF_TOML="${1:-/pscratch/sd/s/siddart2/kv-eviction/experiments/smoke3_rl_single_step/inference_smoke3.toml}"

# Strip darshan from LD_PRELOAD — host I/O profiler not available in container
export LD_PRELOAD=$(echo "$LD_PRELOAD" | tr ':' '\n' | grep -v darshan | paste -sd ':')

cd /pscratch/sd/s/siddart2/kv-eviction
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

echo "=== Smoke #3 inference server (DP=4) ==="
echo "Host:   $(hostname)"
echo "Config: $INF_TOML"
echo "Python: $(python --version)"
echo "vLLM:   $(python -c 'import vllm; print(vllm.__version__)')"

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run inference @ "$INF_TOML"
