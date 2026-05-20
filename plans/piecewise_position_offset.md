# Piecewise position_offset: preserve prefix-cache reuse across admission

**Status:** Proposed 2026-05-16.
**Problem:** vLLM's `request.position_offset: int` is a single value applied
uniformly to all K/Q rotations in a request. After chained admissions, the
K cache holds vectors with HETEROGENEOUS logical positions (sys at offset 0,
prior-turn survivors at offset 48, new prefill at offset 80 etc.). A single
offset cannot describe this — so we either drop the survivors (correct but
loses prefix-cache reuse) or get duplicate logical positions in cache (the
bug we just diagnosed).

**Fix:** Track the LOGICAL POSITION of each cache block (`logical_start`).
Q and K rotations use the block's logical_start, not a single request-level
offset. Admission picks a new "offset for future writes" that ensures
monotonicity with existing survivors.

## Why this preserves prefix caching

After chained admission, the cache state for step 3 looks like:

```
Slot range  | logical_start | K rotated at       | Source
[0..15]     | 0             | logical [0..15]    | sys (inherited from step 1)
[16..31]    | 96            | logical [96..111]  | step 2's prefill survivor
[32..67]    | 112           | logical [112..147] | step 3's new prefill
[68..75]    | 148           | logical [148..155] | step 3's decode
```

All monotonic. All preserved (no re-prefill needed for sys or step-2 survivor).
Step 3's new prefill writes 36 K instead of my current fix's 52.

## Data model changes

### KVCacheBlock (vllm/v1/core/kv_cache_utils.py)

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None
    logical_start: int = -1  # NEW: logical position of K vector at slot 0
                              # of this block. -1 = not yet written.
    ...

    def reset_hash(self): ...
    def reset_logical_start(self): self.logical_start = -1  # NEW
```

### Block allocation (vllm/v1/core/kv_cache_manager.py)

When a fresh block is allocated for a request (cache miss path):
```python
block.logical_start = block_idx_in_request * block_size + request.position_offset
```

Where `request.position_offset` is the offset that will be applied to writes
in this block. For cache-hit blocks, `logical_start` is already set from when
the block was first allocated in the donor request.

### Block free (vllm/v1/core/block_pool.py)

In `free_blocks`, when ref_cnt drops to 0, also reset logical_start:
```python
if blk.ref_cnt == 0:
    blk.reset_hash()
    blk.reset_logical_start()
```

This ensures a freed block doesn't carry a stale logical_start when re-allocated.

## Smart offset computation (vllm/v1/core/sched/scheduler.py::_compact_request)

After `_apply_trim`, compute:

```python
if prompt_tokens_evicted > 0:
    blocks = compaction_mgr.req_to_blocks[request.request_id]
    max_logical = -1
    for b in blocks:
        if b.logical_start >= 0:
            max_logical = max(max_logical, b.logical_start + block_size - 1)
    if max_logical >= 0:
        # Ensure new prefill starts above max survivor logical.
        required_offset = max_logical + 1 - request.num_computed_tokens
        request.position_offset = max(request.position_offset, required_offset)
```

Note: `request.position_offset` was already bumped by `total_evicted` in
`_apply_trim`. We take the max with `required_offset` to ensure monotonicity
even when prior survivors push the requirement higher.

## Per-token logical position (vllm/v1/worker/gpu_model_runner.py)

Replace the single-offset formula in `_prepare_inputs`:

```python
# OLD:
# self.positions = physical_positions + position_offsets_gpu[req_idx]

# NEW:
# block_idx = physical_positions // block_size
# offset_within = physical_positions % block_size
# logical = block_logical_starts_gpu[req_idx, block_idx] + offset_within
```

Where `block_logical_starts_gpu` is a per-(request, block_idx) tensor uploaded
each step alongside `block_table`. Shape: `(num_reqs, max_blocks_per_req)`.

For NEW K writes (prefill of empty blocks): the block's logical_start is set
at allocation time using the current `request.position_offset` — so K is
rotated at the correct logical position automatically.

## Wire format (vllm/v1/core/sched/output.py)

`NewRequestData` and `CachedRequestData` need `block_logical_starts` per
request to ship the per-block info to the worker:

```python
@dataclass
class NewRequestData:
    ...
    block_logical_starts: list[int]  # NEW, parallel to block_ids
```

Same for `CachedRequestData.block_logical_starts: dict[req_id, list[int]]`.

## CompactionEvent (vllm/v1/core/compaction/types.py)

`CompactionEvent.position_offset_after` is now the SMART offset computed by
`_compact_request`, not just `prior_offset + total_evicted`. Existing field
semantics still hold; the value just may be higher under chained admission.

The trainer reads `position_offset_after` from each event and uses it to
construct piecewise position_ids in the merged frame. See trainer changes
below.

## Trainer (src/kv_eviction/segmented_forward.py)

`per_call_segmented_forward`'s admission branch currently does:
```python
cum_position_offset += desc["total_evicted"]
```

Change to read the event's reported offset:
```python
cum_position_offset = admission_event.position_offset_after
```

This already captures whatever smart value vLLM picked. The trainer's
piecewise construction (protected_positions + kept_middle_positions +
post_positions) works as-is, but the offset values come from vLLM's smart
computation rather than the trainer's local accumulation.

## Implementation order

1. KVCacheBlock.logical_start + reset_logical_start (~10 LOC).
2. Block allocation sets logical_start (~20 LOC).
3. Block free resets logical_start (~5 LOC).
4. _compact_request smart offset (replace current "free survivors" hack)
   (~30 LOC).
5. gpu_model_runner per-block position computation (~50 LOC, touches GPU
   buffers).
6. SchedulerOutput / NewRequestData / CachedRequestData wire fields
   (~40 LOC).
7. CompactionEvent already has position_offset_after; just ensure it's the
   smart value (no change needed).
8. Trainer per_call_segmented_forward reads position_offset_after from event
   (~10 LOC).
9. Tests + end-to-end smoke.

Total: ~165 LOC of vLLM changes + ~10 LOC of trainer changes.

## Risk register

1. **Cache-miss allocation timing**: logical_start must be set BEFORE the
   block is written by the worker. allocate_slots runs before
   _apply_inline_admission_eviction, and the worker's K write happens after.
   So logical_start should be set at allocate_slots time, using the
   request's offset AT THAT TIME (which is pre-admission, before any bump
   in this step). For blocks allocated specifically for the post-admission
   prefill range, the offset should be the POST-admission value. This needs
   careful ordering. Mitigation: re-compute logical_start in
   _apply_inline_admission_eviction for the prefill blocks after the smart
   bump.

2. **GPU buffer size**: `block_logical_starts_gpu` is (num_reqs, max_blocks).
   For 256 reqs × 512 blocks × 4 bytes = 0.5 MB. Trivial.

3. **Prefix cache hit logical_start mismatch**: a block in the cache pool
   has logical_start from its first writer. When a new request inherits it
   via hash match, the inheriting request's "expected" logical_start (=
   block_idx * block_size + this_request.position_offset) might not equal
   the block's. This is FINE — the inheriting request just uses the block's
   logical_start for Q rotations. The model attends K at whatever logical
   positions they're rotated at, with Q at the same logical frame.

4. **First-write contention**: if two requests want to write a block with
   the same hash but different logical_starts, the first to allocate wins.
   Other request inherits the first's logical_start. Q rotations align.
   But the second request's "view" of physical position p has logical
   different from `p + this_request.position_offset`. That's expected and
   correct (= the piecewise nature).

5. **Trainer / vLLM agreement**: as long as CompactionEvent reports the
   correct (smart) offset, trainer's piecewise position_ids reconstruction
   matches vLLM's per-token logical positions. KL stays at kernel floor.
