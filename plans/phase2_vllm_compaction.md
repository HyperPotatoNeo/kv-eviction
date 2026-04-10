# Phase 2: vLLM KV Cache Compaction Feature

## Goal

Implement block-level KV cache compaction as a native vLLM V1 scheduler feature. After this
phase, a stock vLLM server launched with `--compaction-window-size 4096 --compaction-stride 512`
will automatically evict the oldest post-prompt blocks when a request's KV length exceeds the
window, producing correct logprobs and returning compaction event metadata in the response.

No prime-rl or training code is involved in this phase. The output is a modified vLLM that
passes all existing vLLM tests plus new compaction-specific tests.

## Prerequisites

- Phase 1 complete (repo structure, vLLM submodule)
- vLLM v0.19.0 source available at `$SCRATCH/kv-eviction/vllm/`

---

## Core Design Principle: Compaction = Partial Preemption

Instead of maintaining separate "logical" and "physical" num_computed_tokens (which conflicts
with 11+ vLLM invariants), we **modify the request to look like a shorter sequence**:

1. Splice `req_to_blocks` (remove evicted blocks, free to pool)
2. Reduce `request.num_computed_tokens` by tokens evicted
3. Trim `request._all_token_ids` (remove evicted token entries)
4. Trim `request._output_token_ids` (same)
5. Set `request.position_offset += tokens_evicted` (for RoPE only)
6. Mark request for rebuild (model runner does remove→re-add with fresh state)

After these mutations, every vLLM consumer sees a consistent shorter sequence. No
dual-tracking, no special cases in allocate_slots, no CachedRequestData changes.

The ONLY special handling is `position_offset` added to positions for correct RoPE
embeddings — a 3-line change in the model runner's position computation.

### Why this works

| vLLM Consumer | What it reads | After compaction |
|---|---|---|
| `schedule()` num_new_tokens | `num_tokens - num_computed` | Both reduced by same amount → delta=1 ✓ |
| `allocate_slots` | `num_computed` vs `len(req_to_blocks)` | Both physical → match ✓ |
| `token_ids_cpu[req, num_computed]` | Next input token | all_token_ids trimmed → correct ✓ |
| `compute_slot_mapping(positions)` | `pos // block_size` → block index | Physical positions → within block_table ✓ |
| `seq_lens` | `num_computed + num_scheduled` | Physical → correct attention length ✓ |
| `positions` (RoPE) | `physical + offset` | Correct absolute position ✓ |

### Why RoPE is correct after compaction

KV entries retain their original RoPE from computation time (baked into K tensors).
After evicting S tokens: `position_offset += S`. New token at physical position P:
- RoPE position = P + offset (correct absolute position)
- Attending to retained KV at physical P' with baked RoPE(P' + old_offset): relative = correct
- Prompt KV at physical P_p with baked RoPE(P_p): relative = correct

---

## Files to Create (new)

All new files go in `vllm/v1/core/compaction/`:

```
vllm/v1/core/compaction/
├── __init__.py         # Re-exports CompactingKVCacheManager, CompactionEvent
└── manager.py          # CompactingKVCacheManager + CompactionEvent (all in one file)
```

NOTE: strategy.py and metadata.py were eliminated during review. The 3-line FIFO eviction
logic is inlined into compact_request(). CompactionEvent is defined in manager.py.
RandomEviction can be added later as a Callable parameter if needed for ablation.

## Files to Modify (existing vLLM)

```
vllm/v1/request.py                        # +position_offset, +compaction_events, +num_total_generated
vllm/v1/core/sched/scheduler.py           # +compaction in update_from_output, +rebuild tracking
vllm/v1/core/sched/output.py              # +rebuild_req_ids, +position_offsets on CachedRequestData
vllm/v1/core/sched/utils.py               # stop condition uses num_total_generated
vllm/v1/core/kv_cache_coordinator.py      # +CompactingKVCacheManager construction
vllm/v1/worker/gpu_model_runner.py        # +position_offset in RoPE, +rebuild handling
vllm/v1/worker/gpu_input_batch.py         # +position_offsets array, +CachedRequestState field
vllm/engine/arg_utils.py                  # +compaction engine args
```

Estimated: ~220 new lines + ~70 modifications across ~11 files.

---

## 2.1: CompactingKVCacheManager (+ CompactionEvent)

**File: `vllm/v1/core/compaction/manager.py`**

Extends `FullAttentionManager`. Adds `compact_request()` that splices blocks and frees them.
CompactionEvent is defined here (no separate metadata.py). Eviction logic is inlined
(no separate strategy.py — 3-line FIFO doesn't need a Protocol).

```python
from dataclasses import dataclass
from typing import Callable
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager


@dataclass(frozen=True)
class CompactionEvent:
    """Record of a single compaction. Stored on Request, included in response."""
    num_output_tokens_at_compaction: int  # total generated when this fired
    tokens_evicted: int
    blocks_evicted: int
    position_offset_after: int  # cumulative


class CompactingKVCacheManager(FullAttentionManager):

    def __init__(self, *args,
                 compaction_window_size: int = 0,
                 compaction_stride: int = 0,
                 eviction_fn: Callable | None = None,  # (total, prompt, stride) -> list[int]
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compaction_window_size = compaction_window_size
        self.compaction_stride = compaction_stride
        self.eviction_fn = eviction_fn  # None = default FIFO
        if compaction_stride > 0:
            assert compaction_stride % self.block_size == 0

    @property
    def stride_blocks(self) -> int:
        return self.compaction_stride // self.block_size

    def needs_compaction(self, request_id: str, num_computed_tokens: int,
                         prompt_tokens: int) -> bool:
        """Check if compaction should fire. Guards against prompt>window case."""
        if self.compaction_window_size <= 0:
            return False
        if num_computed_tokens <= self.compaction_window_size:
            return False
        # Only compact if enough generation blocks exist to evict
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None:
            return False
        prompt_blocks = (prompt_tokens + self.block_size - 1) // self.block_size
        gen_blocks = len(blocks) - prompt_blocks
        return gen_blocks >= self.stride_blocks

    def compact_request(self, request_id: str, prompt_tokens: int) -> int:
        """Evict blocks. Returns tokens_evicted.

        Steps:
        1. Ask strategy which block indices to evict
        2. Free those blocks to pool
        3. Splice req_to_blocks (delete entries using slice for O(n) not O(n*k))
        4. Return tokens evicted
        """
        blocks = self.req_to_blocks[request_id]
        prompt_blocks = (prompt_tokens + self.block_size - 1) // self.block_size

        # Eviction: default FIFO (oldest post-prompt blocks), or custom callable
        if self.eviction_fn is not None:
            evict_indices = self.eviction_fn(len(blocks), prompt_blocks, self.stride_blocks)
        else:
            # Inline FIFO: evict oldest post-prompt blocks
            gen_blocks = len(blocks) - prompt_blocks
            actual = min(self.stride_blocks, gen_blocks)
            evict_indices = list(range(prompt_blocks, prompt_blocks + actual))

        if not evict_indices:
            return 0

        # Free evicted blocks to pool
        evicted_blocks = [blocks[i] for i in evict_indices]
        self.block_pool.free_blocks(evicted_blocks)

        # Splice using contiguous slice deletion for efficiency (O(n) not O(n*k))
        # For SlidingWindowEviction, indices are always contiguous
        start, end = evict_indices[0], evict_indices[-1] + 1
        if end - start == len(evict_indices):
            # Contiguous range — fast path
            del blocks[start:end]
        else:
            # Non-contiguous (e.g., RandomEviction) — reverse delete
            for i in sorted(evict_indices, reverse=True):
                del blocks[i]

        return len(evict_indices) * self.block_size
```

### Key Design Notes

1. **Slice deletion** (`del blocks[start:end]`) is O(n) — a single memmove. Not O(n*k).
2. **block_pool.free_blocks()** handles ref counting and returns blocks to the free queue.
3. **No null_block replacement** — we physically remove entries. This is safe because we
   also trim `num_computed_tokens` and `all_token_ids` to match, so no consumer ever
   sees a gap between the block count and the token count.

---

## 2.4: Request Modifications

**File: `vllm/v1/request.py`**

Add fields to `Request.__init__()` (around line 135):

```python
# --- Compaction state ---
# Cumulative evicted tokens. Used ONLY for RoPE position correction.
# RoPE position = physical_position + position_offset
self.position_offset: int = 0

# Monotonic counter: total output tokens EVER generated (never decremented).
# Used by check_stop for max_tokens because len(_output_token_ids) shrinks
# after compaction trims evicted tokens. Without this, the request never stops.
self.num_total_generated: int = 0

# History of compaction events for this request (included in API response).
self.compaction_events: list = []

# Flag: request was compacted and needs model runner rebuild.
self.needs_rebuild: bool = False
```

Also modify `append_output_token_ids()` to increment the counter:
```python
def append_output_token_ids(self, token_ids):
    if isinstance(token_ids, int):
        self._output_token_ids.append(token_ids)
        self._all_token_ids.append(token_ids)
        self.num_total_generated += 1
    else:
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)
        self.num_total_generated += len(token_ids)
    self.update_block_hashes()
```

**File: `vllm/v1/core/sched/utils.py`** — Fix stop condition (line 113-114):
```python
# BEFORE:
# request.num_tokens >= max_model_len       (line 113)
# request.num_output_tokens >= max_tokens   (line 114)

# AFTER:
request.num_prompt_tokens + request.num_total_generated >= max_model_len
request.num_total_generated >= max_tokens
```

Note: `self.num_prompt_tokens` already exists on Request (line 119). Reuse it.

---

## 2.5: Scheduler Integration

**File: `vllm/v1/core/sched/scheduler.py`**

### 2.5.1: Compaction check in `update_from_output()`

Inside `update_from_output()` (the per-request loop starting at line 1345), AFTER the
request's output tokens are appended and stop condition is checked, add:

```python
# --- After _update_request_with_output and stop checks ---
# IMPORTANT: Must be AFTER stop check, BEFORE EngineCoreOutput construction.
# Must NOT compact stopped/finished requests.

# Compact in a while-loop to handle large prefill chunks or spec decode
# that can jump num_computed past window + stride in a single step.
if (not stopped
    and self._compaction_enabled  # bool set at init, False when window=0
    and request.num_output_placeholders == 0):  # no async placeholders pending
    while self._should_compact(request):
        tokens_evicted = self._compact_request(request)
        if tokens_evicted == 0:
            break  # No more gen blocks to evict (prompt > window case)
        request.needs_rebuild = True
    if request.needs_rebuild:
        # Ensure next _make_cached_request_data sends trimmed all_token_ids
        self.prev_step_scheduled_req_ids.discard(request.request_id)
```

### 2.5.2: _should_compact and _compact_request methods

```python
def _should_compact(self, request: Request) -> bool:
    """Check if any KV cache group needs compaction for this request."""
    for mgr in self.kv_cache_manager.coordinator.single_type_managers:
        if hasattr(mgr, 'needs_compaction') and mgr.needs_compaction(
            request.request_id, request.num_computed_tokens,
            request.num_prompt_tokens,
        ):
            return True
    return False

def _compact_request(self, request: Request) -> int:
    """Compact a request: splice blocks, trim tokens, update state.

    After this, the request looks like a shorter sequence to all consumers.
    """
    total_evicted = 0
    for mgr in self.kv_cache_manager.coordinator.single_type_managers:
        if not hasattr(mgr, 'compact_request'):
            continue
        tokens_evicted = mgr.compact_request(
            request.request_id, request.num_prompt_tokens
        )
        if tokens_evicted > 0:
            total_evicted = tokens_evicted
            break  # Only one KV group for standard models

    if total_evicted == 0:
        return 0

    # Record event BEFORE mutating state
    event = CompactionEvent(
        num_output_tokens_at_compaction=request.num_output_tokens,
        tokens_evicted=total_evicted,
        blocks_evicted=total_evicted // self.block_size,
        position_offset_after=request.position_offset + total_evicted,
    )
    request.compaction_events.append(event)

    # --- Mutate request to look like a shorter sequence ---

    # 1. Trim all_token_ids: remove evicted tokens (oldest post-prompt)
    prompt_len = request.num_prompt_tokens
    evict_start = prompt_len  # start of eviction in all_token_ids
    evict_end = prompt_len + total_evicted
    del request._all_token_ids[evict_start:evict_end]
    # Rebuild the ConstantList wrapper
    request.all_token_ids = ConstantList(request._all_token_ids)

    # 2. Trim output_token_ids: remove first total_evicted entries
    del request._output_token_ids[:total_evicted]
    request.output_token_ids = ConstantList(request._output_token_ids)

    # 3. Reduce num_computed_tokens
    request.num_computed_tokens -= total_evicted

    # 4. Update position offset
    request.position_offset += total_evicted

    return total_evicted
```

### 2.5.3: Rebuild tracking in _make_cached_request_data

When building `CachedRequestData`, check if any request needs rebuild:

```python
# In _make_cached_request_data, for each req:
if req.needs_rebuild:
    # Send full block_ids (not just new) and full all_token_ids
    full_block_ids = tuple(
        [blk.block_id for blk in group]
        for group in self.kv_cache_manager.coordinator.get_req_blocks(req.request_id)
    )
    new_block_ids.append(full_block_ids)
    all_token_ids[req_id] = req.all_token_ids.copy()
    rebuild_req_ids.add(req_id)
    req.needs_rebuild = False
else:
    new_block_ids.append(
        req_to_new_blocks[req_id].get_block_ids(allow_none=True)
    )
```

---

## 2.6: CachedRequestData Modification

**File: `vllm/v1/core/sched/output.py`**

Add two fields to `CachedRequestData`:

```python
@dataclass
class CachedRequestData:
    # ... existing fields ...

    # Requests that were compacted and need full block_table + token_ids rebuild.
    rebuild_req_ids: set[str] = field(default_factory=set)

    # Position offsets for compacted requests (req_id → cumulative offset).
    # Only populated for rebuild requests. Model runner uses for RoPE correction.
    position_offsets: dict[str, int] = field(default_factory=dict)
```

In `_make_cached_request_data`, for rebuild requests:
```python
if req.needs_rebuild:
    rebuild_req_ids.add(req_id)
    position_offsets[req_id] = req.position_offset
    # Send full block_ids and all_token_ids (bypass scheduled_in_prev_step check)
    all_token_ids[req_id] = req.all_token_ids.copy()
    full_block_ids = tuple(...)
    new_block_ids.append(full_block_ids)
    req.needs_rebuild = False
elif not scheduled_in_prev_step:
    all_token_ids[req_id] = req.all_token_ids.copy()
    # ... normal new_block_ids ...
```

The `needs_rebuild` branch MUST come before the `scheduled_in_prev_step` check
to ensure trimmed `all_token_ids` is always sent for compacted requests.

---

## 2.7: Model Runner — Rebuild Handling + RoPE Offset

**File: `vllm/v1/worker/gpu_model_runner.py`**

### 2.7.1: Process rebuild requests in update_states()

In the `update_states()` loop (line 1210+), add handling for compacted requests:

```python
for i, req_id in enumerate(req_data.req_ids):
    req_state = self.requests[req_id]
    num_computed_tokens = req_data.num_computed_tokens[i]
    new_block_ids = req_data.new_block_ids[i]
    ...
    req_index = self.input_batch.req_id_to_index.get(req_id)

    # Handle compacted requests: remove from batch, re-add with fresh state
    if req_id in req_data.rebuild_req_ids:
        # Update req_state with compacted data
        req_state.num_computed_tokens = num_computed_tokens
        req_state.block_ids = new_block_ids
        req_state.position_offset = req_data.position_offsets.get(req_id, 0)
        if req_id in req_data.all_token_ids:
            all_tids = req_data.all_token_ids[req_id]
            prompt_len = len(req_state.prompt_token_ids or [])
            req_state.output_token_ids = list(all_tids[prompt_len:])
            req_state.num_tokens = len(all_tids)
        # Remove from persistent batch and queue for re-add
        if req_index is not None:
            self.input_batch.remove_request(req_id)
        reqs_to_add.append(req_state)
        continue

    # ... existing update logic ...
```

The `remove_request` + `reqs_to_add` path triggers `input_batch.add_request()` which:
- Copies prompt_token_ids + output_token_ids to token_ids_cpu (from trimmed state)
- Sets num_computed_tokens_cpu = physical value
- Rebuilds block_table row via `add_row(block_ids)` (overwrite)
- All state is fresh and consistent

### 2.7.2: Position offset for RoPE

Add `position_offset` to `CachedRequestState` and `InputBatch`.

**InputBatch (`gpu_input_batch.py`):**
```python
# In __init__: use a pre-allocated pinned CPU tensor (same pattern as num_computed_tokens)
self.position_offsets_cpu = np.zeros(max_num_reqs, dtype=np.int64)
# NOTE: no separate GPU tensor needed. The offset is gathered per-step using
# the numpy array and bulk-copied to a pre-allocated GPU buffer (see below).

# In add_request:
self.position_offsets_cpu[req_index] = request.position_offset
```

**CachedRequestState:**
```python
position_offset: int = 0  # Set from Request.position_offset
```

**Position computation (`gpu_model_runner.py`, lines 1996-2009):**

```python
# BEFORE: positions = num_computed + query_pos (used for BOTH slot_mapping and RoPE)
# AFTER: compute physical positions for slot_mapping, then add offset for RoPE

# Physical positions (for slot_mapping):
physical_positions = (
    self.num_computed_tokens[req_indices_gpu].to(torch.int64)
    + self.query_pos.gpu[:total_num_scheduled_tokens]
)

# Slot mapping uses physical positions (block_idx = pos // block_size)
self.input_batch.block_table.compute_slot_mapping(
    num_reqs,
    self.query_start_loc.gpu[: num_reqs + 1],
    physical_positions,
)

# RoPE positions = physical + offset (correct absolute positions)
# Use pre-allocated GPU buffer to avoid per-step tensor allocation.
# position_offsets_gpu is a [max_num_reqs] int64 tensor, initialized once.
# Copy from CPU numpy array in bulk (same pattern as num_computed_tokens):
#   self.position_offsets_gpu[:num_reqs].copy_(
#       torch.from_numpy(self.input_batch.position_offsets_cpu[:num_reqs]),
#       non_blocking=True)
# Then gather:
self.positions[:total_num_scheduled_tokens] = (
    physical_positions
    + self.position_offsets_gpu[req_indices_gpu].to(torch.int64)
)
```

**seq_lens is unchanged** — it already uses `self.num_computed_tokens` (physical) correctly:
```python
self.seq_lens[:num_reqs] = (
    self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
)
```

---

## 2.8: KVCacheCoordinator Integration

**File: `vllm/v1/core/kv_cache_coordinator.py`**

In the coordinator's `__init__`, when compaction is enabled and spec is FullAttentionSpec,
directly construct CompactingKVCacheManager:

```python
from vllm.v1.core.compaction.manager import CompactingKVCacheManager

# In the manager creation loop:
for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
    spec = kv_cache_group.kv_cache_spec
    if compaction_config and isinstance(spec, FullAttentionSpec):
        manager = CompactingKVCacheManager(
            spec, block_pool=self.block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=i,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            compaction_window_size=compaction_config.window_size,
            compaction_stride=compaction_config.stride,
        )
    else:
        manager = get_manager_for_kv_cache_spec(...)
```

---

## 2.9: Engine Configuration

**File: `vllm/engine/arg_utils.py`**

```python
# Add to EngineArgs:
compaction_window_size: int = 0          # 0 = disabled
compaction_stride: int = 0               # Must be multiple of block_size
compaction_strategy: str = "sliding"     # "sliding" or "random"

# CLI arguments:
parser.add_argument("--compaction-window-size", type=int, default=0)
parser.add_argument("--compaction-stride", type=int, default=0)
parser.add_argument("--compaction-strategy", type=str, default="sliding",
                    choices=["sliding", "random"])
```

---

## 2.10: Incompatible Feature Guards + Preemption Protection

Assert at startup that compaction is not combined with incompatible features:

```python
if compaction_window_size > 0:
    assert not enable_prefix_caching, "Prefix caching incompatible with compaction"
    assert not pipeline_parallel_size > 1, "PP incompatible with compaction"
    assert not async_scheduling, "Async scheduling incompatible with compaction"
    assert compaction_stride % block_size == 0, "Stride must be multiple of block_size"
    assert compaction_window_size > compaction_stride, "Window must exceed stride"
```

### Step boundary guarantee (CRITICAL ORDERING)

Compaction correctness relies on this execution order:

```
Step N:
  schedule() → SchedulerOutput_N (pre-compaction block_table)
  execute_model(SchedulerOutput_N) → forward with pre-compaction KV
  update_from_output() → COMPACTION FIRES HERE
    → splice blocks, trim tokens, free evicted blocks to pool
    → set needs_rebuild=True

Step N+1:
  schedule() → allocate_slots (freed blocks available for OTHER requests)
  → _make_cached_request_data (sends full_block_ids for compacted request)
  execute_model(SchedulerOutput_N+1)
    → update_states: remove+re-add compacted request (fresh block_table)
    → forward with post-compaction KV (shorter seq_len = speed gain)
```

Between step N's compaction and step N+1's model execution:
- Freed blocks CAN be allocated to other requests (correct — we don't need them)
- Retained blocks are NEVER freed (they stay in req_to_blocks)
- Retained GPU KV memory is UNTOUCHED (block_pool.free_blocks only manages metadata)

If async scheduling overlaps step N+1's schedule with step N's execution, the
model runner might not have rebuild data when it needs it. Hence the async guard.

### Preemption protection (CRITICAL)

After compaction trims `_all_token_ids`, the evicted tokens are **unrecoverable**. If the
request is later preempted (`num_computed_tokens = 0`, blocks freed, request goes to waiting),
re-prefill would recompute from the TRIMMED all_token_ids with a stale position_offset,
producing wrong RoPE for the prompt (offset applied to prompt tokens that should have
position 0..prompt_len-1).

**Fix: Prevent preemption of compacted requests.**

In `_preempt_request()` or the preemption candidate selection, skip requests with
`position_offset > 0`. If all candidates are compacted, abort the lowest-priority
request instead of preempting.

```python
# In preemption candidate selection:
def _select_preemption_victim(self, ...):
    for req in candidates:
        if req.position_offset > 0:
            continue  # Cannot preempt — trimmed tokens unrecoverable
        return req
    # All candidates are compacted — cannot safely preempt any
    return None  # Fall back to rejecting new request
```

### Async scheduling guard

Do not compact requests with pending async placeholders:
```python
# Already in the compaction check (section 2.5.1):
if request.num_output_placeholders > 0:
    # Skip — placeholder resolution must complete first
    pass
```

---

## Performance Impact (verified by adversarial review)

### Steady-state (per decode step, no compaction firing, 64 concurrent requests)

| Operation | Cost | Notes |
|---|---|---|
| `_compaction_enabled` bool check | ~30ns | Single bool, skips everything when window=0 |
| `_should_compact` per-request (64x) | ~13μs | 200ns/req: attribute chain + 2 int comparisons |
| position_offsets GPU bulk copy | ~0 | Batched in existing CPU→GPU copy, async. No per-step alloc |
| position_offsets GPU gather+add | ~0 | Single `self.position_offsets_gpu[req_indices]` tensor op |
| rebuild_req_ids `in` check (64x) | ~3μs | Set membership, ~50ns/req |

**Net steady-state overhead: ~16μs/step (~0.3% of a 5ms decode step).**
When compaction is disabled (window=0): ~30ns (one bool check).

### Compaction event (once per 512 steps per request)

| Operation | Cost | Notes |
|---|---|---|
| compact_request() splice | ~1-2μs | `del blocks[32:64]` — single memmove |
| block_pool.free_blocks() | ~3-5μs | 32 blocks, linked list operations |
| all_token_ids + output_token_ids trim | ~2μs | Two `del list[start:end]` |
| remove_request (model runner) | ~1μs | Dict pops, set discards (20 ops) |
| add_request (model runner) | ~12μs | token_ids_cpu copy (~15KB), block_table add_row |
| condense (model runner) | ~12μs | Swap one row, dominated by token_ids_cpu copy |

**Total: ~30μs once per 512 steps = ~0.06μs amortized per step.**

### Net throughput gain

At window=4096, requests generating 8k+ tokens:
- Without compaction: seq_len grows to 8k+, attention cost ∝ seq_len
- With compaction: seq_len capped at ~4096, attention cost constant
- **~2x decode throughput at 2x window length, growing linearly thereafter**
- Attention savings at 2x window: ~500μs-1ms per step
- Compaction overhead: ~16μs per step
- **Net: ~97% of the theoretical attention savings realized**

### Why no KV recompute

Verified by adversarial review tracing actual source code:
- `block_table.clear_row()` only zeroes numpy index, not GPU KV memory
- `block_pool.free_blocks()` only manages metadata (ref_cnt, free queue), no GPU ops
- `add_request()` copies CPU metadata only; GPU block_table updated via bulk `commit_block_table`
- Scheduler sees `num_new_tokens = 1` after compaction (both num_tokens and num_computed reduced equally)
- No prefill triggered. Request continues as a normal decode with shorter KV.

---

## Implementation Order

1. **manager.py** — CompactingKVCacheManager + CompactionEvent (all in one file, ~70 lines)
2. **__init__.py** — 1-line re-export
3. **request.py** — add 4 fields (position_offset, num_total_generated, compaction_events, needs_rebuild) + modify append_output_token_ids
4. **utils.py** — fix check_stop to use num_total_generated
5. **output.py** — add rebuild_req_ids + position_offsets to CachedRequestData
6. **scheduler.py** — _compaction_enabled flag, compaction check + _compact_request + rebuild tracking
7. **kv_cache_coordinator.py** — CompactingKVCacheManager construction
8. **gpu_input_batch.py** — position_offsets_cpu array + CachedRequestState.position_offset field
9. **gpu_model_runner.py** — rebuild handling + position_offsets_gpu buffer + RoPE offset
10. **arg_utils.py** — engine arguments + incompatibility guards (prefix caching, PP, async sched)
11. **Tests** — unit + integration
12. **Verify no-compaction passthrough** — window=0 → identical behavior

**Total: 2 new files, 8 modified files = 10 files. ~190 new lines + ~60 modified.**

## Critical Invariants

1. **After compaction**: `num_computed_tokens == physical KV length` (approximately: `sum(len(group)) * block_size`, accounting for partial blocks)
2. **After compaction**: `len(all_token_ids) == num_prompt_tokens + len(output_token_ids) == num_computed_tokens + 1` (the +1 is the just-appended token)
3. **Monotonic counter**: `num_total_generated` never decreases, used for stop condition
4. **Position continuity**: next token position = `num_computed + 0 + position_offset` = correct absolute position
5. **Prompt protection**: eviction indices never < `prompt_blocks`
6. **Block pool accounting**: freed blocks returned before splice
7. **No-op passthrough**: window=0 → all compaction paths are guarded, zero behavioral change
8. **Preemption safety**: requests with `position_offset > 0` are never preempted
9. **Gen blocks guard**: `needs_compaction` returns False when gen_blocks < stride_blocks

## Completion Criteria

- [ ] `CompactingKVCacheManager.compact_request()` evicts correct blocks and splices
- [ ] `needs_compaction` guards against prompt>window (no-op when gen_blocks < stride)
- [ ] Request state (num_computed, all_token_ids, output_token_ids) consistent after compaction
- [ ] `num_total_generated` incremented on every append, used in check_stop
- [ ] Stop condition works correctly through multiple compaction cycles
- [ ] Model runner rebuilds correctly for compacted requests
- [ ] `position_offset` transported via CachedRequestData → CachedRequestState → InputBatch
- [ ] RoPE positions correct (physical + offset = absolute)
- [ ] Slot mapping correct (uses physical positions, computed before offset addition)
- [ ] seq_lens correct (physical + scheduled)
- [ ] Preempted requests with position_offset>0 are skipped (preemption protection)
- [ ] While-loop handles multi-stride compaction in single step
- [ ] `prev_step_scheduled_req_ids` discarded for rebuild requests (ensures all_token_ids sent)
- [ ] Engine args parsed and propagated
- [ ] Incompatible features rejected at startup (prefix caching, PP)
- [ ] No-compaction mode passes all existing vLLM tests
- [ ] Compaction unit tests pass
- [ ] vLLM server with compaction generates text correctly (smoke test)

## Phase 2.1 Hotfix: Prompt-block alignment (2026-04-10)

**Bug.** `Scheduler._compact_request` trimmed `_all_token_ids[prompt_len : prompt_len + total_evicted]`, but the manager physically frees blocks at indices `[prompt_blocks, prompt_blocks + stride_blocks)` where `prompt_blocks = ceil(prompt_len / block_size)`. Physical eviction starts at `prompt_aligned_len = prompt_blocks * block_size`, not at `prompt_len`. When `prompt_len % block_size != 0`, the first `prompt_aligned_len - prompt_len` generated tokens share the partial last prompt block and are NEVER physically evicted; the old trim removed tokens with different identities (same count by coincidence), silently desynchronizing the logical token list from physical KV. Downstream, the rebuild path in `gpu_model_runner.py:1273-1276` writes the mis-trimmed list into the GPU `all_token_ids` state tensor and sample kernels (penalties, bad_words, prompt_logprob) index it positionally, operating on wrong token identities.

**Fix.** `Scheduler._compact_request` now computes `prompt_aligned_len` from the block_size of the `CompactingKVCacheManager` that actually performed the eviction (not `self.block_size`, to survive future hybrid/multi-group KV caches), then trims `_all_token_ids[prompt_aligned_len : prompt_aligned_len + total_evicted]` and `_output_token_ids[gen_tail : gen_tail + total_evicted]` where `gen_tail = prompt_aligned_len - prompt_len`. Identities now match the physical KV.

**Regression test.** `tests/v1/core/test_scheduler_compaction.py` covers both the non-block-aligned case (prompt_len=50, block_size=16) and the block-aligned sanity case (prompt_len=48). Both verify token identities position-by-position. Verified green against a live scheduler instance at 2026-04-10.

## Phase 2.1 Hotfix #2: Partial evict-block guard + LMCache refusal (2026-04-10)

**Bug #2 (partial last evict block).** The prior `needs_compaction` guard was `num_computed > window_size AND gen_blocks >= stride_blocks`, where `gen_blocks = len(blocks) - prompt_blocks`. Under a user config that picks `window_size` close to `prompt_len` (or any test/ablation that does), compaction could fire when the LAST block to be evicted had only a few real tokens — the rest of its slots were allocated but unwritten. The manager returned `tokens_evicted = stride_blocks * block_size` anyway, and the scheduler decremented `num_computed_tokens` by that amount. Under-decrement was impossible; over-decrement was real. The resulting state had `num_computed_tokens < len(_all_token_ids) - 1`, meaning the next forward pass would be told to compute a token at a position that was already computed (a prompt token or an already-sampled token), corrupting attention. The trim also silently deleted the pending just-sampled token from `_all_token_ids`, losing it from the trajectory.

**Fix #2.** `CompactingKVCacheManager.needs_compaction` now also requires `num_computed_tokens >= (prompt_blocks + stride_blocks) * block_size`. This ensures the last block we'd evict is fully filled, so `tokens_evicted = stride_blocks * block_size` always matches the actual number of logical tokens being removed. In realistic configs (`window_size >> stride_blocks * block_size`) this is a no-op because the window guard dominates; it kicks in only when window is configured aggressively small. Regression test: `test_compaction_waits_for_full_evict_block` with `window_size=64, prompt_len=50, block_size=16, stride=16` — verifies compaction defers from `num_computed=65` (naive fire) to `num_computed=80` (safe fire, last evict block full), preserving the pending sample.

**Bug #3 (LMCache compatibility).** LMCache KV transfer's V1 adapter at `distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:1426` reads `request._output_token_ids[0]` as a "first_tok" fingerprint for the KV transfer protocol. After any compaction that index no longer refers to the first generated token, which would mis-identify sequences on the transfer path.

**Fix #3.** `Scheduler.__init__` asserts that if `cache_config.compaction_window_size > 0` and `kv_transfer_config.kv_connector` contains "lmcache" (case-insensitive), startup fails with a clear error. The check runs BEFORE the KV connector is constructed so the `lmcache` package is never imported in this path. Regression test: `test_compaction_rejects_lmcache_connector`.

## Known Limitations (carried into Phase 3+)

- **num_cached_tokens is stats-only drift.** It is not decremented when compaction trims the logical view. Only affects metrics/stats output, not scheduler or worker correctness.
