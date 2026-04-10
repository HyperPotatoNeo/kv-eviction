# Phase 1: Repository Setup

## Status

Partially complete. The repo skeleton exists at `$SCRATCH/kv-eviction/` with git initialized,
submodules added, and `pyproject.toml` + `setup.sh` created. This phase documents the full
intended state and remaining items.

## Goal

A clean monorepo at `$SCRATCH/kv-eviction/` with:
1. vLLM v0.19.0 fork as a git submodule (branch `compaction`)
2. clean prime-rl as a git submodule (main branch)
3. `src/kv_eviction/` package for the integration layer
4. Test scaffolding
5. Environment setup script that works inside the Perlmutter container

## Current State (what already exists)

```
$SCRATCH/kv-eviction/
├── .git/                     # initialized
├── .gitmodules               # both submodules declared
├── vllm/                     # submodule: forked vLLM v0.19.0
├── prime-rl/                 # submodule: PrimeIntellect-ai/prime-rl
├── src/kv_eviction/
│   └── __init__.py           # empty
├── experiments/              # empty
├── tests/                    # empty
├── plans/                    # this directory
├── pyproject.toml            # minimal, editable install for src/
├── setup.sh                  # container setup: venv, pip installs, verification
├── CLAUDE.md                 # project overview
└── .gitmodules
```

## Target State (full Phase 1 completion)

```
$SCRATCH/kv-eviction/
├── vllm/                               # Submodule on branch `compaction`
│   └── (stock vLLM v0.19.0 — no modifications yet)
│
├── prime-rl/                           # Submodule on main (clean, no modifications)
│
├── src/kv_eviction/
│   ├── __init__.py                     # Version string, package marker
│   ├── env.py                          # Stub (Phase 3)
│   ├── segmented_forward.py            # Stub (Phase 3)
│   └── train_hooks.py                  # Stub (Phase 3)
│
├── experiments/
│   ├── full_context.toml               # Stub (Phase 4)
│   └── markovian_kv.toml               # Stub (Phase 4)
│
├── tests/
│   ├── test_compaction_manager.py      # Stub (Phase 2)
│   ├── test_position_tracking.py       # Stub (Phase 2)
│   ├── test_segmented_forward.py       # Stub (Phase 3)
│   └── test_logit_match.py             # Stub (Phase 4)
│
├── plans/
│   ├── phase1_repo_setup.md            # This file
│   ├── phase2_vllm_compaction.md
│   ├── phase3_training_integration.md
│   └── phase4_experiments.md
│
├── pyproject.toml
├── setup.sh
├── CLAUDE.md
└── .gitmodules
```

## Remaining Work

### 1. Verify vLLM submodule is on the `compaction` branch

The vLLM submodule should track a fork of `vllm/vllm` at the `v0.19.0` tag, on a branch
called `compaction`. All Phase 2 changes will go on this branch.

```bash
cd $SCRATCH/kv-eviction/vllm
git branch         # should show compaction
git log --oneline -1  # should be at v0.19.0 tag
```

If the submodule points to the upstream (not a fork), the user needs to:
1. Fork `vllm/vllm` on GitHub
2. Update `.gitmodules` to point to the fork URL
3. Create and push the `compaction` branch from the `v0.19.0` tag

### 2. Verify prime-rl submodule is clean main

```bash
cd $SCRATCH/kv-eviction/prime-rl
git branch         # should show main
git status         # should be clean
```

### 3. Create stub files

Create the following stubs so the directory structure is established and imports work.
Each stub should have a module docstring explaining its Phase 2/3/4 purpose.

**`src/kv_eviction/__init__.py`** (update existing empty file):
```python
"""kv_eviction: Native vLLM KV cache compaction for RL training."""
__version__ = "0.1.0"
```

**`src/kv_eviction/env.py`** (stub):
```python
"""RL environment wrapper for compaction-enabled vLLM.

Implemented in Phase 3. Wraps the standard vLLM /v1/chat/completions response
(which now includes compaction_events metadata) into prime-rl's RolloutOutput format.
"""
```

**`src/kv_eviction/segmented_forward.py`** (stub):
```python
"""Segmented forward pass with KV prefix drop, NO detach.

Implemented in Phase 3. Adapted from mkv-rl window_forward.py.
Key difference: removes .detach() on retained KV between segments,
preserving cross-chunk gradients (G_distal term).
"""
```

**`src/kv_eviction/train_hooks.py`** (stub):
```python
"""Training hooks that inject segmented forward into prime-rl's training loop.

Implemented in Phase 3. Dispatches to segmented_forward when the config
specifies markovian_kv mode, otherwise uses standard prime-rl forward.
"""
```

**`tests/test_compaction_manager.py`** (stub):
```python
"""Tests for CompactingKVCacheManager (Phase 2).

Tests: block eviction, splice correctness, position offset tracking,
block pool accounting, no-compaction passthrough.
"""
```

**`tests/test_position_tracking.py`** (stub):
```python
"""Tests for position offset correctness through scheduler and model runner (Phase 2).

Tests: RoPE position = physical_pos + position_offset after eviction,
monotonicity of position_offset, correct seq_len after compaction.
"""
```

**`tests/test_segmented_forward.py`** (stub):
```python
"""Tests for segmented forward pass (Phase 3).

Tests: logit shape correctness, boundary token overlap, KV drop counts,
FSDP dummy pass padding, no-detach gradient flow.
"""
```

**`tests/test_logit_match.py`** (stub):
```python
"""End-to-end logit match test: inference vs training (Phase 4).

Tests: step-0 KL approximately 0.0 (flash_attn kernel match between
segmented forward and vLLM inference), G_distal nonzero.
"""
```

### 4. Verify setup.sh works

The existing `setup.sh` installs vLLM 0.19.0 from pip (not from the submodule). This is
intentional for Phase 1 — the stock vLLM is used as the runtime, and our compaction
modifications in Phase 2 will be applied as a monkey-patch or by installing the submodule
in editable mode instead.

**Important decision for Phase 2**: When Phase 2 modifications begin, `setup.sh` should
switch from `uv pip install "vllm==0.19.0"` to `uv pip install -e ./vllm` so that our
forked vLLM source is used. The current pip install is fine for Phase 1 verification.

To test:
```bash
# On a compute node inside the container:
cd $SCRATCH/kv-eviction
bash setup.sh
source .venv/bin/activate

# Quick verification
python -c "import kv_eviction; print(kv_eviction.__version__)"
python -c "from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager; print('OK')"
python -c "from vllm.v1.core.block_pool import BlockPool; print('OK')"
python -c "from vllm.v1.request import Request; print('OK')"
```

### 5. pyproject.toml completeness

The current `pyproject.toml` is minimal. It should be expanded to include test dependencies:

```toml
[project]
name = "kv-eviction"
version = "0.1.0"
description = "Native vLLM KV cache compaction for RL training"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
test = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

## Key Files Reference

| File | Absolute Path | Purpose |
|------|--------------|---------|
| vLLM FullAttentionManager | `vllm/v1/core/single_type_kv_cache_manager.py:400` | Base class for CompactingKVCacheManager (Phase 2) |
| vLLM BlockPool | `vllm/v1/core/block_pool.py` | Block allocation/freeing API |
| vLLM Request | `vllm/v1/request.py:58` | Will add position_offset field (Phase 2) |
| vLLM Scheduler | `vllm/v1/core/sched/scheduler.py` | Will add compact_if_needed hook (Phase 2) |
| vLLM SchedulerOutput | `vllm/v1/core/sched/output.py:184` | Will add compaction_updates field (Phase 2) |
| vLLM gpu_model_runner | `vllm/v1/worker/gpu_model_runner.py` | Will add position_offset to position calc (Phase 2) |
| mkv-rl window_forward.py | `$SCRATCH/mkv-rl/src/prime_rl/trainer/rl/window_forward.py` | Reference for segmented forward (Phase 3 base) |
| mkv-rl worker.py | `$SCRATCH/mkv-rl/src/prime_rl/inference/mkv/worker.py` | Current KV eviction (what we are replacing) |

## Completion Criteria

Phase 1 is complete when:
- [ ] `git submodule status` shows both submodules initialized at correct commits
- [ ] vLLM submodule is on `compaction` branch from v0.19.0 tag
- [ ] prime-rl submodule is on clean main
- [ ] All stub files exist with docstrings
- [ ] `setup.sh` runs successfully inside the container
- [ ] `python -c "import kv_eviction"` works after setup
- [ ] All vLLM base class imports succeed (FullAttentionManager, BlockPool, Request)
- [ ] `git status` is clean (everything committed)
