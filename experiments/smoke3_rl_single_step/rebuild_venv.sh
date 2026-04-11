#!/bin/bash
# Rebuild kv-eviction venv AFTER the uv venv --clear incident partially
# wiped site-packages. Differs from setup.sh in Step 1 only:
#
#   setup.sh: `uv pip install "vllm==0.19.0"` -> upstream, no compaction code
#   this:     `VLLM_USE_PRECOMPILED=1 uv pip install -e ./vllm` -> fork editable
#
# The fork source at /pscratch/sd/s/siddart2/kv-eviction/vllm already has all
# the precompiled .so extensions committed into vllm/vllm/*.so so the editable
# install reuses them without recompiling.
#
# Run INSIDE the skyrl container on a compute node:
#   salloc -A m4881 -C "gpu&hbm80g" --qos=interactive --time 2:00:00 \
#          --gpus-per-node 4 -N 1 \
#          bash /pscratch/sd/s/siddart2/kv-eviction/experiments/smoke3_rl_single_step/run_rebuild.sh
set -euo pipefail

PROJECT=/pscratch/sd/s/siddart2/kv-eviction
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
export VLLM_USE_PRECOMPILED=1
cd "$PROJECT"

echo "=== Creating venv ==="
uv venv .venv --python python3.12 --clear
source .venv/bin/activate

echo "=== Step 1: vLLM fork editable (precompiled) ==="
VLLM_USE_PRECOMPILED=1 uv pip install -e ./vllm

echo "=== Step 2: transformers v5 from HF git ==="
uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@c1c3424"

echo "=== Step 3: flash-attn pre-built wheel ==="
uv pip install "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"

echo "=== Step 4: prime-rl extra deps ==="
uv pip install \
    "beartype>=0.21.0" "jaxtyping>=0.3.2" "tomli-w>=1.2.0" \
    "prime>=0.5.37" "aiolimiter>=1.2.1" "setproctitle>=1.3.0" \
    "tilelang>=0.1.8" "liger-kernel>=0.5.10" \
    "ring-flash-attn>=0.1.8" "wandb>=0.24.2" \
    "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@0760204" \
    "torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e" \
    "pydantic-config @ git+https://github.com/samsja/pydantic_config.git@main" \
    "dion @ git+https://github.com/samsja/dion.git@d891eeb" \
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"

echo "=== Step 5: prime-rl editable (no-deps) ==="
uv pip install -e ./prime-rl --no-deps

echo "=== Step 6: kv-eviction editable ==="
uv pip install -e .

echo ""
echo "=== Verification ==="
python << 'PYEOF'
import sys, torch
print(f"Python {sys.version.split()[0]}")
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
import vllm; print(f"vllm {vllm.__version__}")
import transformers; print(f"transformers {transformers.__version__}")
import flash_attn; print(f"flash_attn {flash_attn.__version__}")
# Fork-specific imports
from vllm.v1.core.compaction.manager import CompactingKVCacheManager
print("vllm fork compaction module OK")
from transformers.models.nemotron_h.modular_nemotron_h import NemotronHMamba2Mixer  # noqa
print("transformers.models.nemotron_h OK")
from verifiers.utils.save_utils import make_serializable  # noqa
print("verifiers.utils.save_utils OK")
from prime_rl.transport.types import TrainingSample  # noqa
from prime_rl.trainer.model import forward  # noqa
print("prime-rl imports OK")
import kv_eviction; print("kv_eviction OK")
print("\n=== REBUILD COMPLETE ===")
PYEOF
