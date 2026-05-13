# Attention Matching in the Forked vLLM

## What it is

Attention matching (AM) is a KV compaction method implemented in this repo's
forked `vllm` runtime. Instead of evicting old cache blocks with a simple
windowing rule, AM tries to build a smaller synthetic KV cache whose future
attention behavior matches the original cache as closely as possible.

In this codebase, AM is used as a **baseline compaction method**. The goal is
not to change the model weights or teach a new policy directly inside vLLM. The
goal is to answer a narrower question:

> If we shrink the KV cache intelligently at inference time, can the model keep
> working under long-context pressure and continuous batching?

That makes it a useful scientific baseline for compaction experiments.

## Where it lives

The main runtime path is in the forked `vllm` submodule:

- `vllm/vllm/v1/core/compaction/`
- `vllm/vllm/v1/core/sched/`
- `vllm/vllm/v1/request.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu_input_batch.py`

The scheduler decides when compaction must happen. The worker owns the live KV
cache and performs the actual AM reconstruction.

## How the AM path works

At a high level:

1. A request grows until it crosses the configured compaction window.
2. The scheduler marks the request for compaction instead of letting KV usage
   grow without bound.
3. The worker extracts the source KV region that should be compressed.
4. AM builds a smaller target cache by matching how future query keys would
   attend to the original cache.
5. The compacted KV replaces the larger source region, and generation resumes
   from the shorter cache.

In the current setup, AM typically uses:

- a recent query/key region as the query source
- a target compressed length smaller than the original cached region
- per-layer reconstruction of compacted keys and values

The important property is that AM is trying to preserve **attention behavior**,
not exact token-by-token cache contents.

## Why the forked vLLM matters

This is not an external post-processing script. The point of the fork is that
AM runs **inside the live vLLM engine**:

- it sees real request state
- it runs under real GPU KV pressure
- it interacts with the scheduler directly
- it can be tested under mixed-request continuous batching

That is much more realistic than a standalone harness that compacts a single
sequence offline.

## What changed for continuous batching

The new work extends AM so it behaves correctly under preemption and resume in
the V1 scheduler.

Before this change, AM could compact a request once, but if that request was
preempted later under batch pressure, the AM state was not a real first-class
runtime state. In practice that meant AM was not a faithful continuous-batching
baseline.

The new path adds:

- request-level AM metadata
- scheduler support for AM preemption/resume
- worker-local snapshots of AM-compacted state
- restore of compacted KV state onto newly allocated blocks
- preservation of AM-specific state needed by the scoring path

The result is that an AM-compacted request can now:

1. run in a mixed batch
2. get preempted under KV pressure
3. resume later on the same worker
4. continue from its AM-compacted cache instead of silently falling back

That is the key reason this is now a meaningful continuous-batching baseline.

## Optimization history

The optimization work focused on **execution overhead**, not changing the AM
objective.

Main code paths:

- `vllm/vllm/v1/core/compaction/am_runtime.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu_input_batch.py`
- `prime-rl/src/prime_rl/inference/vllm/server.py`

Scientific guardrails used throughout:

- no change to the AM objective
- query policy changes are explicit config changes, not silent fallbacks
- no layer or head skipping
- no hidden NaN handling
- no silent fallback that changes the algorithm

### 2026-04-21

- [x] Implemented same-worker AM continuous batching support.
- [x] Added AM preempt/resume metadata in request and scheduler state.
- [x] Added worker-local AM snapshot and restore.
- [x] Preserved per-layer AM `beta` state for `score_mod` across resume.
- [x] Forced the local vLLM server to use `generation_config = "vllm"` so
  request-level sampling args such as `temperature=1.0` and `top_p=0.96` are
  not shadowed by model-baked defaults.
- [x] Added AIME AM launch configs and evaluation path in this repo.

### 2026-04-22

- [x] Reduced Python overhead in OMP key selection.
  - bulk tensor assignment instead of Python per-index append
  - removed per-iteration `sum().item()`-style synchronization
- [x] Removed avoidable device sync points in the AM hot path.
- [x] Hoisted request-local slot and query bookkeeping out of inner loops.
- [x] Added cached GPU metadata for AM request state.
  - cached `block_ids` tensor with explicit invalidation
  - persistent device-side AM positions tensor
  - cached query-index tensors
- [x] Reduced worker-side allocation churn during compaction.
  - reusable scratch buffers for gathered KV, synthetic KV, and AM temporaries
- [x] Combined key/value gather paths and other worker hot-path cleanup.
- [x] Stabilized score-mod object identity so layers reuse a persistent AM
  score-mod callable instead of rebuilding a new Python closure every step.
- [x] Batched AM setup across heads.
- [x] Batched OMP/NNLS more deeply across heads.
  - this was the largest measured speed win
  - observed AM compaction mean dropped from about `13.5s` to about `5.4s`
    on the relaxed DP4 AIME smoke benchmark
- [x] Batched `C2` reconstruction across heads.
  - additional measured improvement from about `5.41s` to about `5.15s`

### Attempted on 2026-04-22, but not kept as wins

- [ ] Stage full source KV once per layer and slice from that single buffer.
  - correct, but slightly slower than the current best path on benchmark
- [ ] Rework batched OMP control flow to keep selected columns in a persistent
  gathered tensor across iterations.
  - correct, but slightly slower than the current best path on benchmark
- [ ] Add a cached one-query-per-request fast path for `score_mod`
  `query_doc_ids`.
  - correct, but slightly slower than the current best path on benchmark

### 2026-04-24

- [x] Added configurable AM prompt-prefix protection.
  - default: `attention_matching_protect_user_prompts = "first_user"`
  - `none` restores the old behavior
  - `all_user` extends the protected prefix when streaming/multi-turn prompt
    updates add more rendered prompt tokens
- [x] Added per-rollout compaction metadata to verifiers outputs.
  - saved fields: `compaction_events` and `num_compaction_events`
  - AIME runs now emit a `compaction_success_by_count.txt` table showing
    success rate for `N_compactions = 0, 1, 2, ...`
- [x] Added random AM probe queries as the default query source.
  - default: `attention_matching_query_source = "random_queries"`
  - cache-key probes remain available through `recent_cache_keys` and
    `prefix_cache_keys` for ablations
  - AM random queries use an isolated torch generator so probe generation does
    not perturb sampling RNG state
- [x] Added AIME wall-clock timing output.
  - each run writes `wall_clock_times.txt`
  - reported fields include server startup time, eval/generation wall-clock
    time, and total run wall-clock time

Current best measured path:

- deeper batched OMP/NNLS
- batched `C2`
- earlier worker/cache/scratch improvements
- without the later gather-reduction or OMP-control-path experiments

In practice, these implemented optimizations reduced observed live compaction
latency on the 16k AIME path from the earlier roughly `8.7s` to `9.0s` range,
then from the DP4 relaxed-AM `~13.5s` regime down to about `5.1s` per
compaction event on the best measured path. That does not make AM cheap, but it
does make the continuous-batching baseline materially more usable.

## Possible optimization avenues

There is still meaningful systems optimization headroom that should preserve
scientific faithfulness if implemented carefully.

The best remaining targets are:

- extend the current head-batched AM math into more of the remaining
  reconstruction/control path
- reuse per-request scratch tensors for `compact_keys`, `compact_values`,
  `query_keys`, `synthetic_keys`, `synthetic_values`, and `beta`
- cache more request-local metadata across compaction versions
- cache slot mappings across versions when block layout is unchanged
- cache query index selections across versions when the query policy is unchanged
- cache query-source extracts when they are still valid and cheap to reuse
- reduce repeated `index_select` traffic by staging layer KV once in a
  head-friendly layout
- overlap independent layer work with multiple CUDA streams if memory allows
- reduce score-mod rebuild overhead when AM state changes
- move more of the OMP inner loop into a compiled or fused GPU path
- avoid compaction waves across multiple requests when a more staggered
  scheduler policy can preserve the same semantics
- add exact cross-worker request migration for AM-compacted requests
  - move snapshot/restore state, per-layer `beta`, synthetic-prefix metadata,
    and position offsets across workers without approximation
  - this is a major systems feature, but it is likely one of the highest-value
    future improvements for utilization and long-tail throughput

The most promising next step is likely head-batched AM math in
`am_runtime.py`, because the current implementation still spends a lot of time
doing many small serial per-head operations that are not especially GPU-friendly.

## What AIME showed

The most important experimental finding is that AM quality depends strongly on
how aggressively it is forced to re-compact the same request.

With corrected sampling (`temperature=1.0`, `top_p=0.96`):

- full-context 16k AIME scored `3/5 = 0.6`
- aggressive AM 16k (`window=1024`, `stride=256`) scored `0/5 = 0.0`
- relaxed AM 16k (`window=4096`, `stride=1024`) scored `1/5 = 0.2`

The relaxed run was operationally much milder:

- aggressive AM reached about `version=60`
- relaxed AM reached only about `version=12`

That means relaxing the compaction schedule helped, which supports the idea
that repeated online AM re-compaction is part of the failure mechanism.

However, the relaxed run still did not look healthy overall:

- one trace remained coherent and correct
- several traces still degraded badly in the tail
- truncation remained common

So the current conclusion is not "AM is fine once tuned." The conclusion is:

- aggressive repeated compaction is clearly harmful
- reducing compaction pressure helps
- but the current online AM implementation still has a deeper long-context
  quality problem on AIME

## Likely failure mode

The current online AM path repeatedly compacts a cache that already contains an
earlier AM synthetic prefix.

In other words, it is effectively doing:

- original cache -> synthetic summary
- synthetic summary + newer exact tokens -> newer synthetic summary
- and so on

rather than only compacting original history once.

This recursive synthetic-on-synthetic compaction is the leading explanation for
the observed pattern:

- traces begin coherent
- later reasoning drifts
- tails collapse into repetition or garbage
- stricter settings fail much more badly than laxer settings

That makes recursive approximation drift a more plausible explanation than
simple sampling noise.

## Why this is beneficial

### 1. It is a stronger baseline than FIFO eviction

FIFO or sliding-window eviction is cheap, but it throws away history by
position. AM tries to keep the parts of the cache that matter for future
attention, so it can preserve more behavior at the same target cache size.

### 2. It tests whether compaction can preserve capability

For long reasoning tasks like AIME, the important question is not just "did the
server stay alive?" but "did the model still solve problems after compaction?"
AM is designed to preserve useful context better than naive eviction, so it is
a good baseline for capability retention.

### 3. It works in the real serving stack

Because AM is integrated into the forked vLLM runtime, it can be evaluated
under:

- live batching
- real scheduler pressure
- realistic memory limits
- long-context generation

That makes the results more credible than offline approximations.

### 4. It is scientifically useful even if it is slower

AM is not primarily valuable because it is the cheapest method. It is valuable
because it gives a more principled comparison point:

- naive eviction tells us what happens if we discard history mechanically
- AM tells us what happens if we compress history while trying to preserve
  attention behavior

That helps separate "compaction is bad" from "bad compaction is bad."

## Current limitations

The current implementation is intentionally narrower than a full distributed
migration system.

Known limits:

- restore is same-worker, not cross-worker migration
- AM is more expensive than simple eviction
- numerical failures are treated as real failures and should not be hidden
- runtime quality still has to be validated on real long-context benchmarks,
  not assumed from smoke tests
- repeated online AM re-compaction appears to accumulate approximation error
- current AIME evidence suggests that preserving runtime liveness is easier than
  preserving reasoning quality

## Why this matters for AIME

AIME is useful here because it stresses exactly the failure modes we care about:

- long generations
- heavy reasoning traces
- sensitivity to losing earlier context
- real continuous-batching pressure in the vLLM server

If AM works on AIME inside the forked vLLM path, that is much stronger evidence
than a toy benchmark or a single-request smoke test.

## Practical summary

The new forked-vLLM AM path is beneficial because it turns attention matching
from a one-off compaction mechanism into a real continuous-batching baseline.
It now lives in the serving runtime, survives preemption/resume on the same
worker, and can be tested on long-context reasoning workloads such as AIME while
preserving the scientific meaning of the baseline.

The current evidence also shows a clear limitation: the baseline is now fast
enough and integrated enough to test realistically, but the present online AM
design still loses too much quality under repeated long-context compaction. The
next step is therefore not just more profiling, but changing the AM recurrence
behavior so it does not repeatedly summarize its own earlier synthetic state.
