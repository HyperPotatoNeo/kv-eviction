# KV-Corruption Future Experiments

This file lists possible follow-up robustness experiments for deliberately
corrupting the vLLM KV cache during AIME-style long-context inference. The goal
is to test whether attention-matching compaction remains faithful under
controlled perturbations, and to separate failures caused by attention routing,
stored value content, positional alignment, or implementation artifacts.

## Current Baseline

The current implemented control shuffles already-computed token/KV chunks:

- protect the first user prompt by default
- split the remaining computed context into fixed-size chunks, currently 512
  tokens
- for each completed chunk, sample one deterministic Bernoulli trial using
  `shuffle_control_seed`, `request_id`, and `chunk_index`
- when selected, apply the same deterministic permutation to the scheduler token
  history and to every layer's KV cache for that chunk
- log `shuffle_events` and `num_shuffle_events` for accuracy-by-corruption-count
  analysis

This is an order-corruption experiment. It preserves the token/KV multiset
inside each chunk but destroys local ordering.

## Candidate Corruptions

### Token-Order Corruption

Shuffle within chunks, rotate chunks, reverse chunks, swap adjacent chunks, or
randomly permute whole 512-token blocks.

These variants test sensitivity to context ordering while keeping the same
tokens present. Whole-block permutation is less locally destructive than
within-token shuffle and may be a useful intermediate severity level.

### KV-Value Corruption

Add Gaussian noise to K/V tensors, zero a fraction of KV vectors, randomly scale
KV vectors, flip signs, or replace selected KV vectors with KV from another
position in the same request.

These variants test representation robustness directly. They are useful because
they do not necessarily require changing scheduler token order, but they must be
implemented carefully so the corruption is intentional and reproducible rather
than a hidden numerical bug.

Implemented default Gaussian-noise mode:

- config prefix: `noise_control_*`
- default experiment configs: `experiments/kv_noise_aime/inference.toml` and
  `experiments/kv_noise_aime/inference_dp4.toml`
- chunking: non-overlapping 512-token chunks after the protected first user
  prompt
- sampling: one deterministic Bernoulli trial per completed chunk
- default probability: `0.20`
- default standard deviation: `0.01`
- default target: `"both"` keys and values
- metadata: `noise_events` and `num_noise_events`
- summaries: `scripts/noise_success_by_count.py`
- safety: checks finite KV values before and after noise injection and raises
  on invalid std, invalid target, invalid chunk ranges, or non-finite values

### Key-Only And Value-Only Corruption

Corrupt only keys, only values, or both.

Key-only corruption should mostly disrupt attention routing. Value-only
corruption should mostly disrupt the retrieved content after attention weights
are computed. This distinction is especially useful for attention matching,
because AM is designed around preserving attention behavior.

### Layer-Local And Head-Local Corruption

Apply corruption only to selected layers, selected KV heads, early layers, late
layers, or random head subsets.

This can show whether AIME robustness depends more on early representation
formation, late reasoning layers, or particular KV heads. It can also help
identify whether AM's quality is dominated by a small subset of heads/layers.

### Position And Metadata Corruption

Keep K/V tensors fixed but corrupt logical positions, RoPE offsets, block-table
order, slot mappings, or request-local index metadata.

These experiments are high-risk because they can violate implementation
invariants. They are scientifically useful as negative controls for positional
alignment, but they should fail loudly on invalid state and should never be
mixed with normal AM baselines.

### Cross-Request Contamination

Replace selected chunks with KV from another active request.

This is a strong negative control for request isolation and contamination
sensitivity. It must be isolated behind an explicit experiment flag and should
include assertions that it is never enabled for normal compaction baselines.

### Semantic No-Op Controls

Apply an identity permutation, deterministic no-op KV rewrite, or corruption
only to unreachable/padding KV slots.

These controls test whether the instrumentation itself changes behavior. If a
no-op corruption changes AIME accuracy, latency, or compaction counts, the
experiment harness is suspect.

## Recommended Priority

1. Key-only and value-only chunk shuffle.
2. Chunk rotation and chunk reversal.
3. Whole-block permutation.
4. Gaussian noise on keys only, values only, and both.
5. Layer/head-local versions of the safest corruption modes.
6. Identity/no-op controls for every new corruption family.
7. Cross-request contamination only as an explicit negative-control experiment.
8. Position/metadata corruption only after the above are stable.

## Required Metrics

Every KV-corruption experiment should log:

- total generated tokens
- wall-clock full inference time
- wall-clock corruption time
- `num_shuffle_events` or a mode-specific `num_corruption_events`
- success rates bucketed by `N_corruptions`
- success rates bucketed jointly by `N_compactions`, `N_corruptions`, and output
  length bucket
- per-mode corruption parameters, including seed, probability, chunk size, layer
  selection, head selection, and whether keys, values, or both were corrupted

The joint bucketing is important because both compactions and corruption events
scale with generation length. A simple correlation between `num_compactions` and
`num_corruption_events` is expected and should not be interpreted as causal
without controlling for output length.

## Implementation Constraints

- Corruption must be deterministic under a fixed seed.
- Corruption must fail loudly on NaNs, invalid slot mappings, invalid chunk
  ranges, or unsupported request types.
- Corruption must update all relevant state consistently, or explicitly document
  that it is corrupting only KV tensors.
- Normal AM compaction must remain unchanged unless a corruption flag is enabled.
- Corruption flags should be recorded in output JSONL and W&B.
- No errors should be hidden or coerced into successful outputs.
