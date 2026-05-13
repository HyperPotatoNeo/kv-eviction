# Mila Experiment Wrappers

These scripts submit and verify the two rg-mix scientific baselines on Mila
without relying on the Perlmutter-specific `podman-hpc` launchers.

They assume:

- the repo lives on Mila at a shared path such as `~/src/kv-eviction`
- the repo-local virtualenv exists at `./.venv`
- heavy setup was done on a compute node, not the login node
- `WANDB_API_KEY` is exported before submission
- `RG_MIX_DATASET_PATH` points to the rg-mix dataset on Mila

Required environment:

```bash
export WANDB_API_KEY=...
export RG_MIX_DATASET_PATH=/path/on/mila/to/rg_mix_7500
```

Or create a local gitignored file at `experiments/mila/local_env.sh` with:

```bash
export WANDB_API_KEY=...
export RG_MIX_DATASET_PATH=/path/on/mila/to/rg_mix_7500
```

Launches now run a fail-closed preflight first:

- shell syntax checks for the Mila wrappers
- Python compile checks for `src/kv_eviction`, `prime-rl/src/prime_rl`, and `vllm/vllm`
- required config and dataset-path checks
- for interactive allocation launches, live node checks for dataset visibility and GPU count

Optional environment:

```bash
export MILA_ACCOUNT=...
export MILA_PARTITION=...
export MILA_TIME_LIMIT=48:00:00
export RUN_ROOT=$HOME/kv-runs
export INFERENCE_READY_TIMEOUT=900
```

Start a 3-hour interactive allocation on Mila:

```bash
salloc --nodes=3 --ntasks-per-node=1 --gpus-per-node=4 --time=03:00:00 \
  ${MILA_ACCOUNT:+--account="$MILA_ACCOUNT"} \
  ${MILA_PARTITION:+--partition="$MILA_PARTITION"}
```

Then launch the compaction baseline inside that live allocation:

```bash
bash experiments/mila/run_compaction_rgmix_interactive.sh
```

Or the full-context baseline:

```bash
bash experiments/mila/run_full_context_rgmix_interactive.sh
```

If you want Codex itself to own the full patch / relaunch / verify loop inside
that live allocation, run:

```bash
bash experiments/mila/run_codex_loop.sh compaction_rgmix
```

or:

```bash
bash experiments/mila/run_codex_loop.sh full_context_rgmix
```

Set `CODEX_DANGEROUS_AUTO=1` if you want the Codex CLI to bypass its local
sandbox/approval layer entirely inside the cluster allocation.

If you prefer a regular batch job instead, submit the compaction baseline:

```bash
bash experiments/mila/submit_compaction_rgmix.sh
```

Submit the full-context baseline:

```bash
bash experiments/mila/submit_full_context_rgmix.sh
```

Inspect a run by run directory or SLURM job id:

```bash
bash experiments/mila/check_run.sh ~/kv-runs/compaction_rgmix_20260419_120000
bash experiments/mila/check_run.sh 12345678
```

Verify that a finished run produced the expected artifacts:

```bash
bash experiments/mila/verify_run.sh ~/kv-runs/compaction_rgmix_20260419_120000
```

Triage a run for suspicious results even if it technically finished:

```bash
bash experiments/mila/triage_run.sh ~/kv-runs/compaction_rgmix_20260419_120000
```

`triage_run.sh` fails closed on:

- tracebacks, runtime errors, OOMs, NCCL failures, or explicit `ERROR:` lines in trainer/inference logs
- missing parsed trainer step lines
- suspiciously high mismatch KL
- suspiciously low throughput
- peak memory too close to 80 GiB
- flat reward over the recent training window

It writes `triage_summary.json` into the run directory so Codex can use a
single machine-readable verdict when deciding whether to patch/relaunch.

The wrappers create a unique run directory under `RUN_ROOT` containing:

- `status.json`
- `base_rl.toml`
- `base_inference.toml`
- `resolved_rl.toml`
- `resolved_inference.toml`
- `trainer.log`
- `inference0.log`
- `inference1.log`
- `slurm-<jobid>.out`
- `slurm-<jobid>.err`
- `outputs/` (prime-rl run directory)

For interactive-allocation launches, there may be no `slurm-<jobid>.out` or
`slurm-<jobid>.err` files because the work is started directly inside the
existing allocation shell. The run directory still contains the resolved
configs, trainer log, inference logs, outputs, and `status.json`.
