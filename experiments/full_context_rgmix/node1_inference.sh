#!/bin/bash
# full_context_rgmix — inference-side launcher (DP=4, 4 GPUs).
# Runs INSIDE the skyrl container on an inference node.
set -e

INF_TOML="${1:?Usage: node1_inference.sh <inference_toml_path>}"

export LD_PRELOAD=$(echo "$LD_PRELOAD" | tr ':' '\n' | grep -v darshan | paste -sd ':')

cd /pscratch/sd/s/siddart2/kv-eviction
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

echo "=== full_context_rgmix inference server (DP=4) ==="
echo "Host:   $(hostname)"
echo "Config: $INF_TOML"
echo "Python: $(python --version)"
echo "vLLM:   $(python -c 'import vllm; print(vllm.__version__)')"

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run inference @ "$INF_TOML"
