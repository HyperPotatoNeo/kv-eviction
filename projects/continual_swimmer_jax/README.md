# Continual RL JAX Benchmark Project

This is the GPU-first path for the continual-deviation idea.

The design goal is simple: keep the whole hot path on accelerator hardware.

- environment stepping: `mujoco-mjx`
- policy/value nets: JAX/Flax
- optimizer: Optax
- deviation correction: JAX
- representation metrics: JAX

The point is not to rewrite a few helper functions in JAX. The point is to
keep rollout generation, update computation, and evaluation on-device.

## Update rule

The JAX scaffold now supports two algorithm tracks:

- `PPO` / `PPO + temporal deviation correction`
- `AC-PQN` / `AC-PQN + temporal deviation correction`

For your current request, the intended comparison is:

- `AC-PQN`
- `AC-PQN + deviation correction`

The AC-PQN path treats the actor as deterministic and applies the deviation
correction in action space on a fixed anchor observation bank, instead of the
KL-based correction used by the PPO path.

Important benchmark fit:

- `AC-PQN` is currently a continuous-action scaffold
- `PPO` is the safer starter for discrete-action benchmarks
- that means `Craftax` and `Jelly Bean World` currently map more naturally to
  `PPO + deviation correction`, while `Continual World` fits
  `AC-PQN + deviation correction`

## Continual-learning baselines

Three additional continual-learning baselines are scaffolded on top of AC-PQN:

- `Online EWC`: Fisher-weighted parameter anchoring for stability
- `CLEAR`: replay plus policy/value distillation
- `Policy Consolidation`: teacher-cascade regularization across timescales

These are good proxy baselines for this project because they are genuinely
continual-learning-oriented and do not depend on explicit task boundaries.

There is also a separate `random-action` sanity baseline. That one is not a
continual-learning method. It is there to show that any reward acquisition you
see from the learned agents is not just a spurious artifact of the environment
or logging path.

## Benchmark Suite

The scaffold now includes metadata and starter configs for four benchmark
families:

- `Continuing Swimmer (MJX)`: current fully scaffolded all-GPU benchmark
- `Craftax-Classic`: JAX-native and the best next all-GPU benchmark for
  representation analysis
- `Continual World`: strong continual RL benchmark, but environment stepping is
  CPU-side in practice
- `Jelly Bean World`: conceptually useful continual benchmark, but also not
  GPU-native in this scaffold

This is intentional. The CLI now surfaces whether a benchmark is truly
all-GPU-capable, so we do not accidentally treat CPU env benchmarks as if they
matched the MJX path.

## Layout

- `src/continual_deviation_jax/`: JAX/MJX scaffold
- `projects/continual_swimmer_jax/configs/continuing_swimmer_jax.yaml`:
  default all-GPU config
- `projects/continual_swimmer_jax/assets/continuing_swimmer.xml`:
  local MJCF asset for a swimmer-style task

## All-GPU intent

The intended runtime is:

1. Configure XLA before importing JAX.
2. Put the MuJoCo model on device with `mjx.put_model`.
3. Create and step batched `mjx.Data` on device.
4. Keep AC-PQN or PPO loss, deviation correction, and representation metrics as
   JAX arrays until logging.

The default config assumes a Linux NVIDIA GPU machine and sets the XLA flag
recommended in the MJX docs:

- `--xla_gpu_triton_gemm_any=true`

## Variation budget

The scaffold now includes a `variation_budget` config block and utilities for
recording a World's Edge-style variation budget.

Important interpretation:

- in `Continuing Swimmer`, the true environment-side kernel/reward variation
  budget should be near zero because the simulator is stationary
- the useful extra signal is a policy-side budget, measured on a fixed anchor
  observation bank across checkpoints

So the intended logging is:

- `policy_variation`: change in the policy distribution on anchor observations
  for PPO, or action drift on anchor observations for AC-PQN
- `reward_variation`: empirical reward drift on anchors, expected near zero here
- `kernel_variation`: empirical transition drift on anchors, expected near zero
  here unless you later move to a boundary-drifting multi-agent benchmark
- `cumulative_variation`: cumulative sum of the chosen variation terms

## Commands

```bash
PYTHONPATH=src python -m continual_deviation_jax.cli describe
PYTHONPATH=src python -m continual_deviation_jax.cli validate-config \
  projects/continual_swimmer_jax/configs/continuing_swimmer_jax.yaml
PYTHONPATH=src python -m continual_deviation_jax.cli install-hint
PYTHONPATH=src python -m continual_deviation_jax.cli device-summary \
  --config projects/continual_swimmer_jax/configs/continuing_swimmer_jax.yaml
```

Recommended configs for your requested runs:

- [continuing_swimmer_ac_pqn.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn.yaml)
- [continuing_swimmer_ac_pqn_deviation.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_deviation.yaml)

Additional continual-learning baselines:

- [continuing_swimmer_ac_pqn_online_ewc.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_online_ewc.yaml)
- [continuing_swimmer_ac_pqn_clear.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_clear.yaml)
- [continuing_swimmer_ac_pqn_policy_consolidation.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_policy_consolidation.yaml)

Additional benchmark starter configs:

- [craftax_classic_ppo_deviation.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/craftax_classic_ppo_deviation.yaml)
- [continual_world_ac_pqn_deviation.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continual_world_ac_pqn_deviation.yaml)
- [jelly_bean_world_ppo_deviation.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/jelly_bean_world_ppo_deviation.yaml)

Random-action sanity baselines:

- [continuing_swimmer_random.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continuing_swimmer_random.yaml)
- [craftax_classic_random.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/craftax_classic_random.yaml)
- [continual_world_random.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/continual_world_random.yaml)
- [jelly_bean_world_random.yaml](/Users/danemalenfant/PycharmProjects/kv-eviction/projects/continual_swimmer_jax/configs/jelly_bean_world_random.yaml)

For these random runs:

- no policy learning happens
- deviation correction is disabled
- representation metrics are disabled
- the goal is just to establish a reward floor and make sure learned reward is
  materially above random behavior

## Important note

This scaffold is designed so the environment can live on GPU, but it has not
been run locally in this workspace because `jax`, `mujoco-mjx`, `flax`, and
`optax` are not installed here.

For the newly added benchmarks, only `Craftax` shares that same all-GPU story.
`Continual World` and `Jelly Bean World` are scaffolded as benchmark targets
with clear metadata, but their env stepping is still expected to be CPU-side
until we add dedicated adapters.
