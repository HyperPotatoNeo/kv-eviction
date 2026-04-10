# kv-eviction

## Project Overview

Native vLLM KV cache compaction for RL training. Fork of vLLM 0.19 with scheduler-integrated
block-level eviction + segmented forward training (no detach) on clean prime-rl.

## Architecture

- `vllm/` — Submodule: forked vLLM v0.19.0. Compaction additions in `vllm/v1/core/compaction/`
- `prime-rl/` — Submodule: clean PrimeIntellect-ai/prime-rl main. Used as-is for DPPO training.
- `src/kv_eviction/` — Integration layer: segmented forward, env wrapper, train hooks
- `plans/` — Detailed per-phase implementation plans (authoritative specs)

## Key Design Decisions

- **V1 engine** (vLLM 0.19 default). NOT MRV2 (experimental).
- **Block-aligned eviction** — whole PagedAttention blocks. Stride = multiple of block_size.
- **Splice block_table** after eviction (not null_block — kernel reads garbage from middle holes).
- **position_offset** on Request for correct RoPE after eviction.
- **Segmented forward, no detach** for training — same flash_attn kernel as inference → zero KL mismatch.
- **NOT FlexAttention** — SDPA 4D mask forces math backend (different kernel = systematic KL).

## Full Plan

See `plans/` directory and memory file `plan_kv_eviction.md` in Claude memory.

## Environment

Same as parent CLAUDE.md in $SCRATCH — NERSC Perlmutter, podman-hpc container, uv for packages.
