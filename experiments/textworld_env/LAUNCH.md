# Launching the TextWorld experiments

Two end-to-end experiments use the `textworld-env` package:

- **`experiments/compaction_textworld/`** — multi-turn RL with native vLLM
  turn-based KV cache eviction (`compaction_max_turns=10`,
  `compaction_eviction_turn_stride=3`), block-aligned message padding on
  the client side.
- **`experiments/full_context_textworld/`** — matched baseline with no
  eviction. Same hyperparameters otherwise, so reward / convergence
  differences attribute cleanly to the compaction policy.

Both experiments target the **hard-mixed TextWorld cooking dataset**
(5000 games across 5 difficulty tiers: easy-nav, current, hard,
hard-12room, hard-drop).

## 0. One-time prerequisites

Assumes a fresh clone that has already gone through `bash setup.sh` (see
the top-level `README.md`). After setup:

```bash
source .venv/bin/activate
python -c "import textworld_env; print('env OK')"
```

## 1. Dataset

The 5000-game mix is deterministically regeneratable. On Perlmutter it's
already saved at `/pscratch/sd/s/siddart2/datasets/textworld_cooking_mix`.

On a fresh box:

```bash
# Writes to ${KV_EVICTION_DATA_ROOT:-$PWD/data}/textworld_cooking_mix
# Takes ~20 min on a single CPU node. Deterministic seed=42.
bash experiments/textworld_env/prepare_dataset.sh
```

The resulting directory contains:

```
textworld_cooking_mix/
├── metadata.json        # difficulty map + RELATIVE game_files paths
├── dataset/             # HF save_to_disk format (5000 rows)
├── games/               # .z8 + .json + .ni per game (~3.2 GB)
└── eval_dataset/        # optional held-out eval split
```

`metadata.json` stores **relative** `games/game_XXXXX.z8` paths so the
directory is relocatable across machines.

## 2. Inference-only smoke test (validate the port)

Before committing to a 600-step training run, run the 100-sample
inference test. Two modes, two separate salloc allocations, run in
parallel — wallclock ~12 min each.

```bash
# Parallel compaction test (DP=4, turn-based eviction, client padding on)
salloc -A <account> -C "gpu&hbm80g" --qos=interactive --time 1:00:00 \
       --gpus-per-node 4 -N 1 \
       bash experiments/textworld_env/run_compaction_only.sh &

# Parallel full-context baseline (DP=4, no eviction, no padding)
salloc -A <account> -C "gpu&hbm80g" --qos=interactive --time 1:00:00 \
       --gpus-per-node 4 -N 1 \
       bash experiments/textworld_env/run_full_context_only.sh &
wait
```

Or run both sequentially in a single 2h allocation:

```bash
salloc -A <account> -C "gpu&hbm80g" --qos=interactive --time 2:00:00 \
       --gpus-per-node 4 -N 1 \
       bash experiments/textworld_env/run_inference_test.sh
```

Each run writes a JSON summary:

```
experiments/textworld_env/results_compaction.json
experiments/textworld_env/results_full_context.json
```

Both runs shuffle the full 5000-row dataset with `seed=42` and take the
first 100, so they evaluate on **identical** games. Expected base-model
reward ~0.15–0.18 (no training).

### Eval knobs (env vars read by the runner scripts)

| var | default | meaning |
|---|---|---|
| `NUM_EXAMPLES` | 100 | rollouts per run |
| `MAX_EPISODE_STEPS` | 50 | hard cap on env turns per rollout |
| `MAX_CONCURRENT` | 32 | verifiers `max_concurrent` for async rollouts |

## 3. Training runs (prod)

Both `rl.toml` configs are prod-sized: `max_steps=600`, `batch_size=128`,
`rollouts_per_example=8`, `seq_len=16384`, `kl_tau=0.0`, `lr=1e-6`,
`temperature=1.0`, `max_completion_tokens=12000`.

The configs use **placeholders** that must be resolved at launch time:

- `__TEXTWORLD_DATASET__` — path to the `textworld_cooking_mix`
  directory from §1.
- `__INFERENCE_NODE_0__`, `__INFERENCE_NODE_1__` — hostnames of the
  inference nodes in a 3-node 2-1 split (2 inference, 1 trainer).

A launch script must `sed` these into a resolved copy of the TOML
before passing it to `uv run rl @ ...`. See
`experiments/compaction_rgmix/launch.sh` for the canonical
3-node 2-1 split pattern; the textworld version only differs by:

1. Adding a `__TEXTWORLD_DATASET__` sed step alongside the existing
   `__INFERENCE_NODE_*__` substitutions
2. Repointing the TOML paths at `compaction_textworld/` or
   `full_context_textworld/`
3. Container name changes

### Single-node launch (inline substitution, for dev/debug)

For a quick local run on a single 4x A100 node (reduced batch, no
multi-node orchestration):

```bash
# Resolve the dataset placeholder first
sed "s|__TEXTWORLD_DATASET__|/pscratch/sd/s/siddart2/datasets/textworld_cooking_mix|g; \
     s|__INFERENCE_NODE_0__|localhost|g; \
     s|__INFERENCE_NODE_1__|localhost|g" \
    experiments/compaction_textworld/rl.toml \
    > /tmp/resolved_rl.toml

# In one terminal: inference server
uv run inference @ experiments/compaction_textworld/inference.toml

# In another terminal: trainer (will connect to localhost:8000)
uv run rl @ /tmp/resolved_rl.toml
```

Replace `compaction_textworld` with `full_context_textworld` to run the
baseline.

## 4. Compaction policy reference

### What `compaction_textworld` does

- Client pads each outgoing chat request so every `<|im_end|>` lands on a
  16-token PagedAttention block boundary (`[orchestrator.compaction_padding].enabled=true`,
  `block_size=16`).
- vLLM scans the incoming prompt at admission time, counts live completed
  user+assistant turns via `<|im_end|>` positions.
- When `live_turns >= 10`, evicts the oldest 3 turns and re-checks in a
  loop until `live < 10`.
- System prompt is protected via `compaction_protected_prefix_tokens=-1`
  (auto-detect from the first `<|im_end|>`).
- `compaction_assume_aligned_turn_boundaries=true` lets the scheduler
  `align_up` the eviction end so no tail fragment of the last evicted
  turn is orphaned in the kept KV region.
- Block-FIFO compaction (`window_size=4096, stride=512`) is set as a
  safety fallback — it's a hard requirement when `max_turns > 0`.
- Trainer mirrors the window/stride/block_size/protected_prefix settings
  in `[trainer.compaction]` and uses the segmented_forward path with
  `bptt_segments=1`. Per-block activation checkpointing is **rejected
  at config load** for compaction runs (prime-rl commit `4c851fc92`).

### What `full_context_textworld` does

- Same model / data / sampling / optimizer / batch / rollouts / seed.
- **No** `[trainer.compaction]` block — standard trainer path.
- **No** `compaction_*` fields in `[vllm_extra]`.
- Keeps `[orchestrator.compaction_padding].enabled=true` and
  `use_token_client=false` for bit-identical tokenization against the
  compaction run.
- Per-block activation checkpointing is **allowed** (`[trainer.model.ac]`).

## 5. Known gaps

- `eval_textworld.py`'s per-rollout trajectory metrics
  (`compaction_events_per_rollout`, `trajectory_len`, `completion_len`)
  currently report 0 in the output JSON because `out["trajectory"]` is
  empty on the `RolloutOutput` path we use. Reward aggregation works
  correctly. The server-side compaction event logs in
  `/tmp/textworld_compaction_inf.log` (inside the container) do contain
  per-request `admission req=chatcmpl ... evict=[a,b) ...` entries, so
  event counts can be recovered post-hoc with `grep`.
- The launch scripts for multi-node training (`launch.sh`,
  `node1_inference.sh`, `node2_trainer.sh`) are not committed for
  `compaction_textworld` / `full_context_textworld`. See
  `compaction_rgmix/launch.sh` for the template to copy.
