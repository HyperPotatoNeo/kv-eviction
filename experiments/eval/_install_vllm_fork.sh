#!/bin/bash
# Install vllm fork editable with VLLM_USE_PRECOMPILED=1.
# Run this ONCE on the login node (needs internet for the precompiled wheel download).
set -euo pipefail
unset NCCL_SOCKET_IFNAME
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
export VLLM_USE_PRECOMPILED=1
source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate

echo "=== Uninstalling old vllm ==="
uv pip uninstall vllm 2>&1 || true

echo ""
echo "=== Installing vllm fork editable (with precompiled extensions) ==="
cd /pscratch/sd/s/siddart2/kv-eviction/vllm
uv pip install -e . --no-deps --no-build-isolation 2>&1 | tail -20

echo ""
echo "=== Verification ==="
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import vllm._C; print('vllm._C OK')"
python -c "from vllm.v1.core.compaction import CompactingKVCacheManager; print('Compaction manager OK')"
python -c "from vllm.engine.arg_utils import EngineArgs; print('compaction_window_size field:', hasattr(EngineArgs, 'compaction_window_size'))"
python -c "from vllm.config.cache import CacheConfig; print('CacheConfig.compaction_window_size:', CacheConfig.compaction_window_size)"
echo ""
echo "=== DONE ==="
