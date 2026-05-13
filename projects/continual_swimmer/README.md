# Continual Swimmer Temporal-Deviation Project

This project scaffolds a clean place to test a deviation-corrected policy
update on the `Continuing Swimmer` benchmark from
Elelimy, Szepesvari, White, and Bowling's
"Rethinking the Foundations for Continual Reinforcement Learning".

The goal is to compare:

- baseline PPO on a continuing swimmer task
- PPO plus a temporal deviation correction that penalizes policy updates when
  a recent or predicted future policy scores better on the same online window
- the representation drift induced by both update rules

## Layout

- `src/continual_deviation/`:
  pure-Python core for configs, benchmark helpers, deviation correction, and
  representation analysis
- `projects/continual_swimmer/configs/continuing_swimmer.yaml`:
  default experiment config
- `tests/test_continual_deviation.py`:
  small unit tests for the core math and config loader

## Recommended first experiment

1. Keep the PPO hyperparameters matched to the paper.
2. Save checkpoints every `500k` steps.
3. Build the deviation set from:
   recent checkpoints, best-so-far checkpoint, and one-step unrolled future
   policy predictions if available.
4. Evaluate online return, time-to-collapse, end-vs-peak ratio, deviation
   regret, and representation drift.

## Correction term

The core loss shape in the scaffold is:

```text
L_total = L_policy + λ * [S(π_ref) - S(π_candidate) - margin]_+ * KL(π_candidate || π_ref)
```

where `π_ref` is the best deviation policy in the temporal bank.

## Representation analysis

The project includes helpers for:

- linear CKA between checkpoints
- cosine drift between centered activation tensors
- ridge-regression probes on hidden representations
- layer-output capture with Torch forward hooks

Suggested comparison windows:

- early learning
- peak performance
- post-collapse

## GPU runtime

The scaffold now includes an explicit runtime config so the trainer can stay on
GPU without hidden CPU assumptions.

Key settings in `projects/continual_swimmer/configs/continuing_swimmer.yaml`:

- `runtime.device`: `auto`, `cuda`, `cuda:1`, `cpu`, or `mps`
- `runtime.dtype`: `float32`, `float16`, or `bfloat16`
- `runtime.amp_enabled`: enable autocast for mixed precision
- `runtime.torch_compile`: optionally compile the model with `torch.compile`
- `runtime.allow_tf32`: enable TF32 matmuls on NVIDIA GPUs
- `runtime.pin_memory`: useful for faster host-to-device transfers

The default benchmark config now assumes an NVIDIA training box and uses
`bfloat16` plus autocast, which is a good starting point for A100-class GPUs.

The correction term no longer forces `.cpu().item()` synchronizations on every
update step. Scalar summaries are only materialized if you explicitly call
`result.summary()` for logging.

## Commands

Without reinstalling the repo, run from the repository root with
`PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m continual_deviation.cli describe
PYTHONPATH=src python -m continual_deviation.cli validate-config \
  projects/continual_swimmer/configs/continuing_swimmer.yaml
PYTHONPATH=src python -m continual_deviation.cli device-summary \
  --config projects/continual_swimmer/configs/continuing_swimmer.yaml
PYTHONPATH=src python -m continual_deviation.cli smoke-correction
PYTHONPATH=src python -m continual_deviation.cli smoke-representation
```

If you reinstall the repo in editable mode, the console script
`continual-deviation` will provide the same commands.

## Next integration step

The scaffold intentionally stops short of forcing a particular PPO trainer.
The easiest next step is to plug `corrected_policy_loss(...)` into your PPO
update code, and call `compare_representations(...)` on a fixed observation
bank at each saved checkpoint.

When you wire in the trainer, the intended setup is:

```python
from continual_deviation.runtime import (
    autocast_context,
    move_batch_to_device,
    prepare_model_for_runtime,
)

model, device, dtype = prepare_model_for_runtime(model, config.runtime)
for batch in loader:
    batch = move_batch_to_device(batch, device)
    with autocast_context(config.runtime, device):
        loss = ...
```
