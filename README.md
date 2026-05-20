# kv-eviction

Native vLLM KV cache compaction for RL training. Fork of vLLM 0.19 with
scheduler-integrated block-level eviction.

When a request's KV length exceeds `--compaction-window-size`, the scheduler
evicts the oldest post-prompt blocks (one `--compaction-stride` at a time).
The request becomes physically shorter -- attention is faster, no KV recompute.

## Setup

### Prerequisites

- A CUDA 12.8 environment with A100 (or newer) GPUs
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager
- A container runtime is recommended but not required. A suitable base
  image is `docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`,
  which bundles CUDA 12.8, Python 3.12, and `uv`.

### 1. Clone the repo

```bash
git clone --recursive https://github.com/HyperPotatoNeo/kv-eviction.git
cd kv-eviction
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Install

```bash
# One-command install (creates venv, installs vLLM editable + all deps)
bash setup.sh
```

Or manually, for development:

```bash
uv venv .venv --python python3.12 --clear
source .venv/bin/activate

# Editable install of the compaction-enabled vLLM fork. VLLM_USE_PRECOMPILED=1
# downloads vLLM's CI-built .so artifacts and symlinks them into the source
# tree so Python edits under vllm/vllm/v1/core/compaction/ take effect
# without a full C++/CUDA rebuild.
VLLM_USE_PRECOMPILED=1 uv pip install -e ./vllm --no-build-isolation

# kv-eviction integration layer
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
source .venv/bin/activate

# Launch vLLM with compaction enabled.
# window=4096: eviction triggers when KV exceeds 4096 tokens
# stride=512:  evict 512 tokens (32 blocks) per compaction event
# block_size defaults to 16, stride must be a multiple of it
# async_scheduling=False is required (see Constraints below)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --max-model-len 16384 \
  --enable-prefix-caching false \
  --async-scheduling false \
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
    async_scheduling=False,  # required with compaction (see Constraints)
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
- `--async-scheduling false` required. In async mode vLLM pre-schedules the next
  step before the current step's output is processed, so
  `num_output_placeholders` is always nonzero when `update_from_output` runs.
  The compaction trigger is guarded on `num_output_placeholders == 0`, so with
  async scheduling compaction never fires and the run silently degenerates to
  full context. The scheduler asserts this at init.
- Pipeline parallelism (`--pipeline-parallel-size > 1`) not supported
- `stride` must be a multiple of `block_size` (default 16)
- `window_size` must be greater than `stride`

### Recommended settings

| Use case | Window | Stride | Notes |
|----------|--------|--------|-------|
| Long-form generation | 4096 | 512 | Good balance of speed and context |
| Very long generation | 8192 | 1024 | More context retained |
| Aggressive compaction | 2048 | 512 | Maximum speed, less context |

## Eval: rg-mix-env (100 problems, 4 samples, ancestral)

Qwen3-4B-Instruct-2507, DP=4/TP=1, max_tokens=16384, temperature=1.0,
top_p=1, top_k=-1. Script at `experiments/eval/`.

| Metric | Full Context | Compaction (w=4096, s=512) |
|---|---|---|
| pass@1 | 0.4675 | 0.3700 |
| pass@4 | 0.7300 | 0.6600 |
| wall time (max DP chunk) | 947 s | 510 s |
| aggregate throughput | 3,757 tok/s | 7,432 tok/s |
| avg output tokens/sample | 8,896 | 9,469 |

Compaction buys ~1.86x throughput at the cost of ~10 points of absolute pass@1.
The quality cost is concentrated on tasks that require long-range reasoning
across the whole generation (zebra puzzles -0.28, sokoban -0.05, cryptarithm
-0.11). Local tasks (countdown, arc_1d) are neutral or slightly better.

## RL training

Two matched end-to-end RL configs live in `experiments/`. Both train
Qwen3-4B-Instruct-2507 on the `rg-mix-env` reasoning task with GRPO
through prime-rl's FSDP2 trainer and a separately-launched vLLM
inference pool. They differ only in whether KV compaction is active —
everything else (optimizer, loss, dataset, seed, batch size, rollouts,
sequence length, checkpointing) is identical, so the pair is an
apples-to-apples comparison.

| Experiment | Inference | Trainer forward |
|---|---|---|
| `experiments/compaction_rgmix/` | vLLM with `compaction_window_size=4096`, `compaction_stride=512` | `segmented_forward` (no detach), per-segment backward (`bptt_segments=1`) |
| `experiments/full_context_rgmix/` | vLLM with compaction disabled | standard packed forward with per-block activation checkpointing |

### Shared configuration

| | |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| `seq_len` | 16384 |
| `max_completion_tokens` | 15000 |
| `batch_size` / `rollouts_per_example` | 128 / 8 |
| Optimizer | AdamW, `lr=2e-6`, `betas=(0.9, 0.9)`, `wd=0.01`, `max_norm=1.0` |
| Loss | `kl_tau=0.0` (reference-policy KL off) |
| `max_steps` | 500, checkpoint every 50 steps |
| Dataset | `rg-mix-env` (5 reasoning tasks, 7500 train examples, seed=42) |
| Weight sync | filesystem broadcast (trainer writes, inference reads) |

Each run uses 3 nodes with 4×A100-80GB GPUs each: two nodes run a
DP=4 vLLM inference server each (8 engines total), one node runs the
FSDP2 trainer. The orchestrator round-robins rollouts across the two
inference URLs.

### Why activation checkpointing only for full context

Compaction training requires a `DynamicCache` with `use_cache=True` so
retained KV flows between segments in `segmented_forward`. Prime-rl's
non-reentrant `checkpoint_wrapper` re-runs each decoder layer's forward
during backward, which double-appends K/V to the same cache object and
fails with `CheckpointError: Recomputed values have different metadata`.
The full-context path has no `DynamicCache`, so AC is safe and is
enabled at `freq=1` to keep the 16k sequence length within the 80 GB
budget. Compaction stays within budget without AC because each segment
is individually short (window=4096).

### Launching a run

Both experiments include a SLURM + podman-hpc launcher script
(`launch.sh`) as a reference implementation. It is written for a
cluster that exposes 3 GPU nodes over SSH, mounts a shared filesystem
on `/pscratch`, and uses `podman-hpc` for containers — adapt as
needed. The pieces you need in any environment are:

1. **Allocate 3 GPU nodes** (2 for inference, 1 for training). The
   filesystem used for `output_dir` (default `outputs/compaction_rgmix`
   or `outputs/full_context_rgmix`) must be visible from all three
   nodes so weight broadcast can hand weights from trainer to
   inference.

2. **Resolve inference URLs in the trainer TOML.** Both `rl.toml`
   files contain placeholder hostnames `__INFERENCE_NODE_0__` and
   `__INFERENCE_NODE_1__`. Before launching the trainer, substitute
   the actual hostnames of your two inference nodes:
   ```bash
   sed -e "s/__INFERENCE_NODE_0__/<host0>/g" \
       -e "s/__INFERENCE_NODE_1__/<host1>/g" \
       experiments/compaction_rgmix/rl.toml > /tmp/rl_resolved.toml
   ```

3. **Start both inference servers** (one on each inference node).
   This is the `node1_inference.sh` step inside a container on each
   inference node — it just runs:
   ```bash
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0,1,2,3 uv run inference @ experiments/compaction_rgmix/inference.toml
   ```
   Wait until both servers return a Qwen model from
   `GET /v1/models` on port 8000 before continuing.

4. **Start the trainer** on the third node against the resolved TOML:
   ```bash
   source .venv/bin/activate
   uv run rl @ /tmp/rl_resolved.toml
   ```

The trainer streams rollouts asynchronously from the inference pool,
runs GRPO updates over the micro-batches, and pushes updated weights
to the filesystem after each step. Inference reloads weights
automatically via the broadcast worker hook.

### What to watch during training

- `progress/reward/mean` — should trend up; baseline after 5 steps at
  smaller batch (smoke run) was 0.26 → 0.47
- `loss/mismatch_kl_mean` — trainer-vs-inference logprob agreement.
  With flash_attention_2 on both sides this sits at the kernel floor
  ~0.0009. Any climb is a real correctness bug.
- `progress/entropy/mean` — flat or mild downward drift is fine;
  sharp spikes or collapse below ~0.1 indicates mode collapse.
- `progress/seq_len/mean` — watch for length explosion toward the
  15000 `max_completion_tokens` ceiling. Growing rollout length is
  fundamental to rollout-only GRPO in open-ended tasks and can stall
  steps on long tails.
- `Peak Mem.` in the trainer log — compaction run peaks around 46 GiB
  at batch_size=64 on 4×A100-80GB, scales roughly linearly in
  seq_len, not batch size (each micro-batch is one sample).

### Sequence-length truncation in compaction runs

When compaction is active, two per-turn and per-sample truncation
stages are disabled to keep completion token counts and compaction
event coordinates in the same space:

| Stage | Variable | Location | Compaction runs | Non-compaction runs |
|-------|----------|----------|-----------------|---------------------|
| Per-turn | `env.max_seq_len` | `kv_eviction/env.py` monkey-patch on `MultiTurnEnv.add_model_response` | Set to `None` — per-turn truncation skipped | Unchanged (verifiers truncates as normal) |
| Final sample | `seq_len` | `prime-rl trainer/batch.py` `prepare_sample` | Skipped when `compaction_enabled=True` | Active — truncates sample + clamps events |

**Why**: verifiers' `parse_response_tokens` truncates `completion_ids`
when `prompt_len + completion_len > max_seq_len`, but does NOT truncate
compaction events (which live in response metadata). This puts token
counts and event coordinates in different spaces, causing a
non-monotonic boundary assertion crash in `interleave_rollout`. The
trainer-side `seq_len` truncation has the same issue unless
`_clamp_compaction_events` is called, but skipping it entirely in
compaction runs is cleaner: `segmented_forward` with `bptt_segments=1`
processes one segment at a time, so peak memory is bounded by the
largest segment (~prompt_len tokens), not total sequence length.

**Scope**: both disables apply ONLY to multi-turn compaction envs.
The `env.py` monkey-patch targets `MultiTurnEnv.add_model_response`
specifically — `SingleTurnEnv` is never patched. The `prepare_sample`
skip gates on `compaction_enabled=True`, which is set from the trainer
config (`compaction.window_size > 0`).

**Potential failure mode**: if `kv_eviction.env` is imported in a
process that also runs non-compaction `MultiTurnEnv` instances, those
envs will have `max_seq_len` set to `None` (the monkey-patch applies
to the base class unconditionally at import time). This is harmless
when Stage 2 (`seq_len`) is active, since the trainer-side truncation
catches oversized samples. But if both stages are disabled (e.g.,
`compaction_enabled=True` in the trainer config while some envs don't
actually use compaction), non-compaction multi-turn samples will flow
through untruncated, which can cause OOM in the standard forward path.
**Fix**: ensure `compaction_enabled` is only set when ALL envs in the
run are backed by a compaction-enabled vLLM server. Do not mix
compaction and non-compaction multi-turn envs in the same process.

### Multi-turn RL with turn-based eviction (BabyAI / BALROG)

The `experiments/debug_balrog/` directory contains a matched pair of configs
for multi-turn BabyAI environments via the BALROG harness:

| Config | Description |
|---|---|
| `rl_full.toml` | Turn-based KV eviction + segmented-forward training |
| `rl_no_eviction.toml` | Full-context baseline (no eviction) |

Both run on a single node with 8 GPUs (4 infer + 4 train) and can be
launched via:

```bash
python experiments/debug_balrog/launch_eai.py experiments/debug_balrog/rl_full.toml
```

#### Turn-based eviction (vLLM side)

Token-wise compaction (sliding window) evicts a fixed number of tokens
regardless of message boundaries. For multi-turn environments this is
disruptive — eviction can slice mid-message, leaving orphaned tokens from
a partial user or assistant turn in the KV cache. Turn-based eviction
instead drops whole completed `user + assistant` turn pairs:

```toml
[inference.vllm_extra]
compaction_window_size = 4096
compaction_stride = 512
block_size = 16
compaction_protected_prefix_tokens = -1  # auto: protect system prompt
# Turn-based mode — evict whole turns instead of raw token windows.
compaction_max_turns = 4                 # keep at most 4 turns in the window
compaction_eviction_turn_stride = 2      # drop 2 turns per eviction event
compaction_assume_aligned_turn_boundaries = true  # see padding note below
async_scheduling = false
```

Setting `compaction_max_turns` enables turn-based mode in the vLLM
scheduler. The stride (`compaction_eviction_turn_stride`) controls how
many turns are dropped when the window overflows. `compaction_protected_prefix_tokens = -1`
auto-detects the system prompt length by scanning for the first EOS token
and protects it from eviction.

#### Block-aligned padding interceptor (orchestrator side)

PagedAttention evicts whole blocks (16 tokens). If a turn boundary falls
mid-block, evicting "up to the end of turn N" actually evicts N plus a
partial tail — the next turn's first tokens are already in the same block.
The solution is to pad each message's closing `<|im_end|>` token so it
lands on a block boundary, guaranteeing that turn boundaries are exactly
block-aligned:

```toml
[orchestrator.compaction_padding]
enabled = true
block_size = 16  # must match inference.vllm_extra.block_size

[orchestrator]
# Must be false: the padding interceptor lives inside AsyncCompletions.create.
# use_token_client=true bypasses chat.completions.create entirely and sends
# prompt_token_ids directly, skipping the interceptor.
use_token_client = false
```

With `compaction_assume_aligned_turn_boundaries = true` the vLLM scheduler
snaps `evict_end` upward (align_up) to absorb the padding of the last
evicted turn, rather than aligning down and leaving an orphan tail.

#### No-eviction baseline requires activation checkpointing

Without eviction the context grows up to `max_model_len = 16384` tokens.
Processing a 16K-token sequence in a standard packed forward OOMs on
80 GB GPUs without activation checkpointing. The no-eviction trainer
config therefore enables full AC:

```toml
[trainer.model]
impl = "auto"
attn = "flash_attention_2"
ac = { mode = "full" }   # required — avoids OOM on 16K sequences
```

Note: AC is intentionally **disabled** for compaction training. The
segmented-forward path uses a live `DynamicCache` across segments;
prime-rl's non-reentrant `checkpoint_wrapper` double-appends K/V on
the recompute pass and raises `CheckpointError`. Compaction stays within
memory budget without AC because each segment is short (≤ window_size
tokens).

#### Enabling `compaction_padding` on a no-eviction baseline

`compaction_padding.enabled = true` with `trainer.compaction.window_size = 0`
is intentional: the interceptor still sends `prompt_token_ids` via
`extra_body` to bypass vLLM's `tool_choice="auto"` validation (which
rejects requests that include pre-tokenised prompts). The prime-rl
`RLConfig` validator accepts this combination — block-size consistency
checks are only enforced when `window_size > 0`.

### Constraints to keep in mind

- `[trainer.compaction]` in `rl.toml` MUST mirror the inference
  side's `[vllm_extra].compaction_window_size` / `compaction_stride` /
  `block_size` exactly. A mismatch silently corrupts the policy
  gradient; prime-rl's `RLConfig` validator catches it only when
  inference is co-configured in the same TOML, so in a split
  deployment the sync is manual.
- Compaction requires `impl = "hf"` and `attn = "flash_attention_2"`
  on the trainer side — the `TrainerConfig` validator enforces this
  when `compaction.window_size > 0`.
- `stride` must be a multiple of `block_size` (default 16), and
  `window_size` must be greater than `stride`.

## Project structure

```
kv-eviction/
  vllm/                  # Submodule: vLLM 0.19 fork (compaction branch)
    vllm/v1/core/compaction/
      manager.py         # CompactingKVCacheManager + CompactionEvent
  prime-rl/              # Submodule: prime-rl with segmented-forward training path
  src/kv_eviction/       # Integration layer (Phase 3: segmented forward)
  experiments/
    compaction_rgmix/    # RL training run WITH compaction (paper main)
    full_context_rgmix/  # RL training run WITHOUT compaction (baseline)
    eval/                # Inference-only eval harness (rg-mix-env)
  plans/                 # Detailed implementation specs
  setup.sh               # One-command install
```
