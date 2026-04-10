# kv-eviction

Native vLLM KV cache compaction for RL training. Fork of vLLM 0.19 with
scheduler-integrated block-level eviction.

When a request's KV length exceeds `--compaction-window-size`, the scheduler
evicts the oldest post-prompt blocks (one `--compaction-stride` at a time).
The request becomes physically shorter -- attention is faster, no KV recompute.

## Setup

### Prerequisites

- NERSC Perlmutter (or any system with A100 GPUs + CUDA 12.8)
- `podman-hpc` container: `docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`
- Python 3.12, `uv` package manager (both included in the container)

### 1. Get a compute node

```bash
# Interactive (4h max)
salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4

# Enter the container
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
podman-hpc run --rm -it \
  --user "$(id -u):$(id -g)" --replace --name skyrl \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME \
  -v "$SCRATCH":"$SCRATCH" -v "$HOME":"$HOME" \
  -w "$SCRATCH/kv-eviction" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 /bin/bash
```

### 2. Install (inside container)

```bash
# Run the setup script (creates venv, installs vLLM + deps)
bash setup.sh

# Or manually:
cd /pscratch/sd/s/siddart2/kv-eviction
uv venv .venv --python python3.12 --clear
source .venv/bin/activate
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache

# vLLM from our fork (editable, includes compaction)
uv pip install -e ./vllm

# kv-eviction package
uv pip install -e .
```

### 3. Verify installation

```bash
source .venv/bin/activate
python -c "
from vllm.v1.core.compaction import CompactingKVCacheManager, CompactionEvent
print('Compaction imports OK')
import vllm; print(f'vLLM {vllm.__version__}')
"
```

## Inference

### Quick start -- vLLM server with compaction

```bash
source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate

# Launch vLLM with compaction enabled
# window=4096: eviction triggers when KV exceeds 4096 tokens
# stride=512:  evict 512 tokens (32 blocks) per compaction event
# block_size defaults to 16, stride must be a multiple of it
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --max-model-len 16384 \
  --enable-prefix-caching false \
  --compaction-window-size 4096 \
  --compaction-stride 512 \
  --tensor-parallel-size 1 \
  --port 8000
```

### Query the server

```bash
# Standard OpenAI-compatible API -- compaction is transparent
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Solve step by step: what is 1234 * 5678?"}],
    "max_tokens": 8192,
    "temperature": 0.7
  }' | python -m json.tool
```

### Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Write a long essay about AI safety."}],
    max_tokens=8192,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### Offline inference (no server)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    max_model_len=16384,
    enable_prefix_caching=False,
    # Compaction args
    compaction_window_size=4096,
    compaction_stride=512,
)

outputs = llm.generate(
    ["Explain quantum computing in detail."],
    SamplingParams(max_tokens=8192, temperature=0.7),
)
print(outputs[0].outputs[0].text)
```

## How it works

```
Request generating tokens...
  [prompt: 200 tokens] [gen: 3800 tokens]  total KV = 4000  (below window)

Next token generated:
  [prompt: 200 tokens] [gen: 3801 tokens]  total KV = 4001  (still below)

...after 96 more tokens:
  [prompt: 200 tokens] [gen: 3897 tokens]  total KV = 4097  (exceeds window!)

Compaction fires:
  1. Evict 512 oldest generation tokens (32 blocks) from KV cache
  2. Trim token IDs to match
  3. Reduce num_computed_tokens by 512
  4. Add position_offset += 512 for correct RoPE
  5. Mark request for model runner rebuild

After compaction:
  [prompt: 200 tokens] [gen: 3385 tokens]  total KV = 3585  (back under window)
  Physical seq_len = 3585  (faster attention!)
  RoPE positions = physical + 512  (correct absolute positions)
```

The request continues generating as if nothing happened. Attention cost stays
bounded by the window size instead of growing linearly with generation length.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--compaction-window-size` | `0` (off) | KV token count that triggers eviction |
| `--compaction-stride` | `0` | Tokens to evict per event (must be multiple of block_size) |

### Constraints

- `--enable-prefix-caching false` required (compaction splices blocks)
- Pipeline parallelism (`--pipeline-parallel-size > 1`) not supported
- `stride` must be a multiple of `block_size` (default 16)
- `window_size` must be greater than `stride`

### Recommended settings

| Use case | Window | Stride | Notes |
|----------|--------|--------|-------|
| Long-form generation | 4096 | 512 | Good balance of speed and context |
| Very long generation | 8192 | 1024 | More context retained |
| Aggressive compaction | 2048 | 512 | Maximum speed, less context |

## Project structure

```
kv-eviction/
  vllm/                  # Submodule: vLLM 0.19 fork (compaction branch)
    vllm/v1/core/compaction/
      manager.py         # CompactingKVCacheManager + CompactionEvent
  prime-rl/              # Submodule: clean prime-rl (for Phase 3 training)
  src/kv_eviction/       # Integration layer (Phase 3: segmented forward)
  plans/                 # Detailed implementation specs
  setup.sh               # One-command install
```
