#!/bin/bash
set -euo pipefail
unset NCCL_SOCKET_IFNAME
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate

echo "=== Installing eval deps ==="
uv pip install "reasoning-gym==0.1.25" "verifiers==0.1.9.post3" "datasets>=3.0" 2>&1 | tail -5

echo "Python: $(python --version)"
python -c "import vllm; print(f'vLLM: {getattr(vllm, \"__version__\", \"N/A\")} at {vllm.__file__}')"
python -c "from vllm.engine.arg_utils import EngineArgs; print('Compaction fields:', hasattr(EngineArgs, 'compaction_window_size'))"
python -c "from vllm.v1.core.compaction.manager import CompactingKVCacheManager; print('Compaction manager OK')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
python -c "import reasoning_gym; print('reasoning_gym OK')"
echo ""

cd /pscratch/sd/s/siddart2/kv-eviction/experiments/eval
python run_compaction_eval.py
