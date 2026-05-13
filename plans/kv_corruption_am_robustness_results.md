# KV-Corruption Robustness Experiments for Attention Matching

Date: 2026-04-26

## Goal

Test whether attention matching (AM) can act as a robust compaction baseline under KV-cache corruption. The intended scientific claim is not that AM should always improve AIME accuracy, but that under a corruption model that damages stale exact KV cache, AM may be more robust because it replaces old exact KV spans with a compact synthetic prefix.

All experiments use AIME with:

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Max generated tokens: `16384`
- Temperature: `1.0`
- Top-p: `0.96`
- Rollouts per example: `1`
- Examples: `30`
- vLLM runtime: forked vLLM, `enforce_eager = false`
- Prompt protection: first rendered user prompt protected
- AM setting: relaxed 16K AM, `window=4096`, `stride=1024`, `query_source=random_queries`

## Implemented Controls

Gaussian KV noise was added as an online vLLM corruption control. It perturbs KV cache tensors directly:

```text
K_or_V_chunk <- K_or_V_chunk + Normal(0, noise_control_std)
```

The corruption path is deterministic for a fixed seed, request id, and chunk index. It fails on non-finite values rather than hiding NaNs.

Added controls:

- `noise_control_region = "all" | "old_context_only"`
- `noise_control_keep_recent_tokens = N`
- `noise_control_protect_synthetic = true | false`
- `noise_control_target = "keys" | "values" | "both"`

The AM-favorable design uses `old_context_only`, protects the recent suffix, and protects AM synthetic KV. This intentionally tests whether AM benefits when old exact cache is unreliable but its compact synthetic prefix remains stable.

## Completed Results

### Uniform 50% Noise

Config:

- `noise_control_probability = 0.50`
- `noise_control_std = 0.01`
- `noise_control_target = "both"`
- `noise_control_region = "all"`

Results:

| Condition | Run dir | Accuracy | Avg noise events | W&B |
|---|---:|---:|---:|---|
| AM | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_attention_matching_real_20260425_181640` | `0.600` | `3.267` | https://wandb.ai/dane_mal/kv-eviction/runs/y0g2tu7d |
| Full context | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_kv_noise_real_20260425_184544` | `0.633` | `6.267` | https://wandb.ai/dane_mal/kv-eviction/runs/xwgi0kob |

Interpretation: no AM advantage. The model appears robust to this noise level.

### Uniform 50% Shuffle

Config:

- `shuffle_control_probability = 0.50`
- `shuffle_control_chunk_size = 512`

Results:

| Condition | Run dir | Accuracy | Avg shuffle events | W&B |
|---|---:|---:|---:|---|
| AM | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_attention_matching_real_20260425_163338` | `0.600` | `3.033` | https://wandb.ai/dane_mal/kv-eviction/runs/es8x6r6i |
| Full context | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_full_context_real_20260425_170540` | `0.633` | `6.000` | https://wandb.ai/dane_mal/kv-eviction/runs/r8lndxuw |

Interpretation: no AM advantage. Full context remained slightly higher.

### Old-Context Values Noise, Recent 2048 Protected

Config:

- `noise_control_probability = 1.00`
- `noise_control_std = 0.02`
- `noise_control_target = "values"`
- `noise_control_region = "old_context_only"`
- `noise_control_keep_recent_tokens = 2048`
- `noise_control_protect_synthetic = true`

Results:

| Condition | Run dir | Accuracy | Avg noise events | W&B |
|---|---:|---:|---:|---|
| AM | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_attention_matching_real_20260426_015946` | `0.600` | `2.300` | https://wandb.ai/dane_mal/kv-eviction/runs/du7zfwhl |
| Full context | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_kv_noise_real_20260426_025145` | `0.600` | `9.833` | https://wandb.ai/dane_mal/kv-eviction/runs/0q41zyok |

Interpretation: tie. This is the cleanest AM-favorable test so far, because AM did receive some noise events while full context received more stale-cache corruption. Still, no reward advantage was observed.

### Old-Context Values Noise, Recent 4096 Protected, std 0.01

Config:

- `noise_control_probability = 1.00`
- `noise_control_std = 0.01`
- `noise_control_target = "values"`
- `noise_control_region = "old_context_only"`
- `noise_control_keep_recent_tokens = 4096`
- `noise_control_protect_synthetic = true`

Results:

| Condition | Run dir | Accuracy | Avg noise events | W&B |
|---|---:|---:|---:|---|
| AM | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_attention_matching_real_20260426_032649` | `0.500` | `0.000` | https://wandb.ai/dane_mal/kv-eviction/runs/xpwyv3lm |
| Full context | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_kv_noise_real_20260426_035848` | `0.667` | `5.467` | https://wandb.ai/dane_mal/kv-eviction/runs/d2nbattp |

Interpretation: invalid as an AM corruption comparison because AM had zero noise events. The recent-suffix guard plus AM compaction left no eligible exact old chunks in the AM physical cache.

### Old-Context Values Noise, Recent 4096 Protected, std 0.05

Config:

- `noise_control_probability = 1.00`
- `noise_control_std = 0.05`
- `noise_control_target = "values"`
- `noise_control_region = "old_context_only"`
- `noise_control_keep_recent_tokens = 4096`
- `noise_control_protect_synthetic = true`

Results:

| Condition | Run dir | Accuracy | Avg noise events | W&B |
|---|---:|---:|---:|---|
| AM | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_attention_matching_real_20260426_044948` | `0.533` | `0.000` | https://wandb.ai/dane_mal/kv-eviction/runs/vqpdxkpo |
| Full context | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_kv_noise_real_20260426_042536` | `0.600` | `5.167` | https://wandb.ai/dane_mal/kv-eviction/runs/ylkeiklv |

Interpretation: invalid as an AM corruption comparison for the same reason: AM had zero noise events.

### Old-Context Keys Noise, Recent 4096 Protected, std 0.01

Config:

- `noise_control_probability = 1.00`
- `noise_control_std = 0.01`
- `noise_control_target = "keys"`
- `noise_control_region = "old_context_only"`
- `noise_control_keep_recent_tokens = 4096`
- `noise_control_protect_synthetic = true`

Results:

| Condition | Run dir | Accuracy | Avg noise events | W&B |
|---|---:|---:|---:|---|
| AM | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_attention_matching_real_20260426_053252` | `0.600` | `0.000` | https://wandb.ai/dane_mal/kv-eviction/runs/pxlvh5rd |
| Full context | `/home/mila/d/dane.malenfant/scratch/kv-runs/aime_kv_noise_real_20260426_061150` | `0.567` | `5.600` | https://wandb.ai/dane_mal/kv-eviction/runs/aacwzruu |

Interpretation: not a clean AM corruption comparison because AM had zero noise events. It does show full context is still robust to stale key noise at this scale.

## Main Takeaways

1. The model is robust to the tested Gaussian KV noise levels. Uniform `std=0.01` noise and stale-only `std=0.02` values noise did not degrade full context enough for AM to show an advantage.

2. The best scientific comparison so far is old-context values noise with recent 2048 protected. It produced equal accuracy: AM `0.600`, full context `0.600`.

3. Several recent-4096 old-context settings are not valid AM corruption comparisons because AM received `0.000` average noise events. This happens because AM compaction shrinks the physical cache and the recent-suffix protection can cover all remaining exact tokens.

4. A lower number of corruption events for AM is expected under the AM-favorable hypothesis, but zero events means the AM condition is effectively uncorrupted and should not be used as evidence of robustness.

## Recommended Next Experiments

The next tests should force a measurable full-context degradation while still preserving a fair AM-favorable mechanism.

Recommended first:

```toml
noise_control_region = "old_context_only"
noise_control_keep_recent_tokens = 2048
noise_control_protect_synthetic = true
noise_control_target = "values"
noise_control_probability = 1.00
noise_control_std = 0.10
```

If still no effect:

```toml
noise_control_region = "old_context_only"
noise_control_keep_recent_tokens = 1024
noise_control_protect_synthetic = true
noise_control_target = "values"
noise_control_probability = 1.00
noise_control_std = 0.20
```

If Gaussian noise remains ineffective, implement stale-context dropout/zeroing:

```toml
dropout_control_region = "old_context_only"
dropout_control_keep_recent_tokens = 2048
dropout_control_protect_synthetic = true
dropout_control_target = "values"
dropout_control_probability = 1.00
```

Dropout/zeroing is likely a better corruption model than Gaussian noise because it creates a clear stale-memory failure mode instead of relying on activation-scale sensitivity.

## Allocated Follow-Up Matrix

Date allocated: 2026-04-26

These experiments are queued as matched AM/full-context pairs. Each pair uses the same model, AIME examples, sampling settings, seed, corruption region, and corruption severity. The AM side differs only by enabling attention-matching compaction and protecting AM synthetic KV from old-context corruption.

Shared settings:

- `NUM_EXAMPLES=30`
- `MAX_TOKENS=16384`
- `MAX_CONCURRENT=16`
- `noise_control_region = "old_context_only"` or `shuffle_control_region = "old_context_only"`
- first user prompt protected
- AM synthetic prefix protected
- recent suffix protected with either `2048` or `1024` tokens

Allocated matrix:

| Hypothesis | Corruption | Target | Severity | Keep recent | AM session | Full-context session | Config dir |
|---|---|---|---:|---:|---|---|---|
| Stronger stale content noise should finally degrade full context | Gaussian noise | values | `std=0.10` | `2048` | `kv_noise_p1p00_gaussian_values_std0p10_r2048_20260426_123027_am` | `kv_noise_p1p00_gaussian_values_std0p10_r2048_20260426_123027_full` | `experiments/generated/kv_corruption_pairs/20260426_123027_noise_p1p00_std0p10` |
| Very strong stale content noise with more exposed context | Gaussian noise | values | `std=0.20` | `1024` | `kv_noise_p1p00_gaussian_values_std0p20_r1024_20260426_123026_am` | `kv_noise_p1p00_gaussian_values_std0p20_r1024_20260426_123026_full` | `experiments/generated/kv_corruption_pairs/20260426_123026_noise_p1p00_std0p20` |
| Corrupt retrieval and content mildly | Gaussian noise | keys+values | `std=0.03` | `2048` | `kv_noise_p1p00_gaussian_both_std0p03_r2048_20260426_123027_am` | `kv_noise_p1p00_gaussian_both_std0p03_r2048_20260426_123027_full` | `experiments/generated/kv_corruption_pairs/20260426_123027_noise_p1p00_std0p03` |
| Corrupt retrieval and content more strongly with smaller suffix | Gaussian noise | keys+values | `std=0.05` | `1024` | `kv_noise_p1p00_gaussian_both_std0p05_r1024_20260426_123049_am` | `kv_noise_p1p00_gaussian_both_std0p05_r1024_20260426_123049_full` | `experiments/generated/kv_corruption_pairs/20260426_123049_noise_p1p00_std0p05` |
| Create an explicit stale-memory failure mode | zero/dropout | values | zero | `2048` | `kv_noise_p1p00_zero_values_std0p0_r2048_20260426_123050_am` | `kv_noise_p1p00_zero_values_std0p0_r2048_20260426_123050_full` | `experiments/generated/kv_corruption_pairs/20260426_123050_noise_p1p00_std0p0` |
| Destroy stale KV alignment without changing text tokens | KV-only shuffle | KV chunk | `p=1.00` | `2048` | `kv_shuffle_p1p00_r2048_kvonly_20260426_123049_am` | `kv_shuffle_p1p00_r2048_kvonly_20260426_123049_full` | `experiments/generated/kv_corruption_pairs/20260426_123049_shuffle_p1p00` |

Slurm allocation state at queue time:

- Running: one 4xA100L job, job `9367689`, on `cn-g029`
- Pending behind the per-user GPU limit: jobs `9367690`, `9367691`, `9367692`, `9367693`, `9367694`, `9367696`, `9367697`, `9367698`, `9367699`, `9367700`, `9367701`

Known implementation detail:

- Zero/dropout is implemented as `noise_control_mode = "zero"` on the existing noise-control path. It records as a noise event with `std=0.0`.
- KV-only shuffle is implemented as `shuffle_control_kv_only = true`; it permutes KV cache chunks but does not mutate token IDs in the request state.
- For AM comparisons, inspect `num_noise_events` or `num_shuffle_events`. If AM receives zero events while full context receives events, the pair is not a valid corruption comparison even if the design is AM-favorable.

## Code Artifacts

Implemented configs:

- `experiments/kv_noise_aime/inference_am_old_context_values_noise_dp4.toml`
- `experiments/kv_noise_aime/inference_full_context_old_context_values_noise_dp4.toml`

Reusable paired launcher:

- `experiments/mila/launch_kv_corruption_pair.sh`

Relevant vLLM control fields:

- `noise_control_region`
- `noise_control_keep_recent_tokens`
- `noise_control_protect_synthetic`
- `noise_control_target`
- `noise_control_probability`
- `noise_control_std`
