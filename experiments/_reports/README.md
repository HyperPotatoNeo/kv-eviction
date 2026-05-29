# TextWorld Experiment Reports

This folder tracks the paper-facing TextWorld experiment state and generated analysis.

## Files

- `tracker.yaml`: human-owned experiment tracker. Keep run names, scratch dirs, W&B IDs, config paths, and current EAI job IDs here.
- `report_textworld.py`: report builder. It reads `tracker.yaml`, scratch logs, local W&B files, and optional W&B API history.
- `reports/`: generated markdown and SVG plots. Ignored by git.
- `cache/`: generated CSVs and W&B history cache. Ignored by git.

## Standard Flow

Preferred Codex flow:

```text
Use $textworld-analysis to rerun the TextWorld experiment analysis.
```

The skill syncs tracker job IDs from launcher records, queries EAI state into `cache/eai_status.csv`, regenerates the markdown report and plots, and reports stale-log caveats.

1. Launch or relaunch jobs with the launcher:

   ```bash
   .venv/bin/python experiments/debug_balrog/launch_eai.py \
     experiments/_local_jobs/kv_eviction/rl_eai_eviction_turns4_stride2_seed0_flexmask_8gpu4x4.toml
   ```

   The launcher writes the latest EAI job ID to `experiments/debug_balrog/jobs/<wandb.name>.json`.

2. Sync `tracker.yaml` after relaunches.

   Update the matching `paper.runs[].eai_job_id` from the job JSON. This keeps the tracker useful even after old jobs fail and replacements are submitted.

3. Generate the report:

   ```bash
   .venv/bin/python experiments/_reports/report_textworld.py --wandb
   ```

   Use `--refresh-wandb` only when remote W&B history access is fixed or you explicitly want to retry the API cache:

   ```bash
   .venv/bin/python experiments/_reports/report_textworld.py --wandb --refresh-wandb
   ```

4. Open the rendered report:

   ```text
   experiments/_reports/reports/textworld_main.md
   ```

## Relaunch Decisions

Use EAI job state as the source of truth for active replacements:

```bash
eai job get <job_id> --no-header --field state
```

Scratch logs can contain old failures after a relaunch into the same output directory. The report scanner may still see old `phase4_kv_mismatch`, checkpoint, or SIGKILL text unless the output directory was wiped. Before relaunching anything, check the current EAI job ID in `tracker.yaml` or `experiments/debug_balrog/jobs/` and confirm its state.

Practical rule:

- `RUNNING`, `QUEUED`, `QUEUING`: do not relaunch.
- `SUCCEEDED`, `COMPLETED`: do not relaunch.
- `FAILED`, `CANCELLED`, `INTERRUPTED`, `UNKNOWN`: relaunch if the run is still needed.

## Metrics

The report currently extracts:

- Eval success rate: `eval/textworld-env/success_rate`
- Eval hard success rate: `eval/textworld-env/hard_success_rate`
- Eval hard success count: `eval/textworld-env/hard_success_count`
- Eval average score: `eval/textworld-env/avg@1`
- Training reward: `reward/textworld-env/mean`
- Step time: `time/step`

Training reward plots include a trailing running average. GPU-hour plots use:

```text
GPU-hours = cumulative(time/step seconds) * 8 / 3600
```

## Current Caveat

The current W&B credentials authenticate but do not see `laurent-charlin/textworld-eviction`. The report therefore relies on local `.wandb` files under `/scratch/epp/...` for history curves, plus local `wandb-summary.json` files for summaries.
