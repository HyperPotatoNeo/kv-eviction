# Bug 2: prompt_aligned_len Mismatch in Protected-Prefix Multi-Turn Compaction

## Symptom

```
Step 0 | Mismatch KL: 1.2689
```

Expected: ~0.0009 (kernel floor). Observed: 1.27 — three orders of magnitude
above the floor, indicating a **systematic** positional mismatch between the
trainer's KV eviction and vLLM's KV eviction.

Events ARE reaching the trainer (9.6 avg / 28 max per sample). Per-turn
truncation IS disabled. Multi-turn merge IS running. This is NOT a pipeline
event-loss bug (Bug 1 FM1/FM2). The bug is in how the trainer interprets
the events it receives.

## Root cause

The trainer's `segmented_forward` evicts KV at `prompt_aligned_len` — a
**single fixed position** computed once per sample. In protected-prefix
mode (`protected_prefix_tokens=-1`), the auto-detect formula in `train.py`
is broken: it always yields 0 for the first compaction event, falling back
to the turn-0 prompt length. This does not match vLLM's eviction start,
which is the block-aligned auto-detected system prompt boundary.

```
  vLLM eviction start:   ceil(367 / 16) * 16 = 368
  Trainer eviction start: ceil(389 / 16) * 16 = 400

  Off by 32 tokens → entirely different surviving KV → heavy KL.
```

## The two eviction-start computations, side by side

### vLLM (scheduler.py)

```python
def _effective_prompt_tokens(self, request):
    if self._compaction_protected_prefix == -1:
        # Scan prompt_token_ids for the first eos_token_id.
        # In BALROG BabyAI: eos found at position 366 → boundary = 367.
        return min(boundary, request.num_prompt_tokens)

def _compact_request(self, request):
    effective_prompt = self._effective_prompt_tokens(request)
    # evict_start = ceil(effective_prompt / block_size) * block_size
    #             = ceil(367 / 16) * 16 = 368
    ...
    event = CompactionEvent(
        num_output_tokens_at_compaction=request.num_total_generated,
        tokens_evicted=256,            # stride
        position_offset_after=256,     # cumulative offset = tokens_evicted
        num_prompt_tokens=1013,        # prompt length at compaction time
    )
```

Eviction removes KV[368 : 368+256] = KV[368 : 624], which are old
conversation tokens between the system prompt end and the current turn.

### Trainer (train.py:580-590)

```python
pp = config.compaction.protected_prefix_tokens   # -1
if pp == -1:
    first_evt = compaction_events[0]
    pp = first_evt.position_offset_after - first_evt.tokens_evicted
    # pp = 256 - 256 = 0  ← ALWAYS ZERO for the first event

effective_prompt_len = min(pp, mb_prompt_len) if pp > 0 else mb_prompt_len
# pp=0, so: effective_prompt_len = mb_prompt_len = 389 (turn-0 prompt)

prompt_aligned_len = ((effective_prompt_len + bs - 1) // bs) * bs
# prompt_aligned_len = ceil(389 / 16) * 16 = 400
```

### segmented_forward (segmented_forward.py:649-653)

```python
# Between segments: evict at the WRONG position
new_K = torch.cat([
    keys[l][:prompt_aligned_len],                          # keep [0 : 400]
    keys[l][prompt_aligned_len + actual_stride : -trim],   # keep [656 :]
], dim=0)
```

Eviction removes KV[400 : 400+256] = KV[400 : 656].

### The mismatch

```
vLLM kept:    [0:368) + [624:...)   ← system prompt + recent context
Trainer kept: [0:400) + [656:...)   ← system prompt + 32 extra old tokens

Tokens 368-399:  vLLM evicted, trainer retained
Tokens 624-655:  vLLM retained, trainer evicted
```

Every post-compaction token sees a different context window. The logprob
divergence is systematic across ALL post-eviction segments → KL of 1.27.

## Why the auto-detect formula is always zero

The formula `first_evt.position_offset_after - first_evt.tokens_evicted`
tries to recover the position offset *before* the first compaction. For
the first compaction event on any request:

- `position_offset_after = tokens_evicted` (no prior offset)
- So `pp = tokens_evicted - tokens_evicted = 0`

This gives the cumulative offset before the first compaction (correctly: 0),
NOT the eviction start position. The formula conflates "prior offset" with
"protected prefix length". These are completely different quantities.

## Why this only manifests in protected-prefix multi-turn

| Scenario | vLLM evict_start | Trainer prompt_aligned_len | Match? |
|----------|-----------------|---------------------------|--------|
| Single-turn, pp=0 | ceil(prompt_len/16)*16 | ceil(prompt_len/16)*16 | ✅ |
| Multi-turn, pp=0 | ceil(turn_N_prompt/16)*16 | ceil(turn_0_prompt/16)*16 | ❌ (but events start with turn-0 prompt length) |
| Multi-turn, pp=-1 | ceil(367/16)*16 = **368** | ceil(389/16)*16 = **400** | ❌ |

In single-turn with `pp=0`, vLLM protects the full prompt and evicts only
output tokens. The auto-detect formula yields 0, falling back to
`mb_prompt_len` = the one and only prompt length. The eviction starts
match. This is the case validated in smoke runs #1-#4.

In multi-turn with `pp=-1`, vLLM detects the system prompt boundary
(367 tokens) and evicts old conversation tokens within the prompt. The
trainer doesn't know about the 367-token boundary — it uses 389 (turn-0
prompt). The 32-token gap produces the observed KL.

## Confirming from the live smoke logs

**vLLM inference.log** — auto-detected system prompt boundary:
```
[COMPACT] auto-detected system prompt boundary: 367 tokens (prompt_len=389)
[COMPACT] auto-detected system prompt boundary: 367 tokens (prompt_len=449)
[COMPACT] auto-detected system prompt boundary: 367 tokens (prompt_len=1013)
```

**vLLM inference.log** — every compaction evicts at position 368:
```
[COMPACT] req=chatcmpl effective_prompt=367 num_prompt=1013 evict_start=368 evicted=256
[COMPACT] req=chatcmpl prompt_evicted=256 output_evicted=0 new_prompt_len=757
```

ALL 256 evicted tokens are prompt tokens (old conversation). Zero output
tokens evicted. This happens on every compaction because the multi-turn
prompt grows much faster than generation in BabyAI.

**Trainer** — computes `prompt_aligned_len` from turn-0 prompt (389):
- `pp = position_offset_after(256) - tokens_evicted(256) = 0`
- Falls back to `mb_prompt_len = 389`
- `prompt_aligned_len = 400`
- Evicts KV[400:656] instead of KV[368:624]

## The fix

The information the trainer needs (where vLLM actually evicted) is not
present in the current CompactionEvent wire format. Two approaches:

### Approach 1: Add eviction_start to CompactionEvent (recommended)

Add an `eviction_start` field to both `CompactionEvent` (vLLM) and
`CompactionEventWire` (prime-rl) that records `ceil(effective_prompt /
block_size) * block_size` — the exact physical position where eviction
started.

**vLLM side** (compaction/types.py):
```python
class CompactionEvent(msgspec.Struct, ...):
    ...
    eviction_start: int = 0   # new field, omit_defaults preserves compat
```

**vLLM side** (scheduler.py:_compact_request):
```python
evict_start = ((effective_prompt + block_size - 1) // block_size) * block_size
event = CompactionEvent(
    ...,
    eviction_start=evict_start,
)
```

**Trainer side** (train.py):
```python
# Replace the broken auto-detect with the authoritative value:
prompt_aligned_len = compaction_events[0].eviction_start
```

`segmented_forward` then uses the correct position (368) and the KV
eviction matches vLLM exactly.

### Approach 2: Replicate the EOS scan in the trainer (quick fix)

The trainer has `input_ids` and can scan for the first EOS token, exactly
as vLLM does in `_effective_prompt_tokens`. This avoids changing the wire
format:

```python
if pp == -1:
    # Replicate vLLM's auto-detection: first eos in the prompt.
    eos_id = tokenizer.eos_token_id
    boundary = mb_prompt_len
    for i in range(mb_prompt_len):
        if input_ids[0, i].item() == eos_id:
            boundary = i + 1
            break
    prompt_aligned_len = ((boundary + bs - 1) // bs) * bs
```

**Drawbacks**: requires access to the tokenizer in the training loop
(which may not be convenient), and must exactly match vLLM's scan logic.
If the chat template changes or the EOS convention differs, this silently
breaks again.

### Per-event eviction_start (future enhancement)

Currently, `prompt_aligned_len` is fixed across all segments. This works
when:
- Protected prefix stays constant (typical)
- `num_prompt_tokens >= protected_prefix` throughout

If `num_prompt_tokens` drops below the protected prefix after many
evictions, vLLM's `effective_prompt` changes per-event, and the single
`prompt_aligned_len` is wrong. To handle this, `segmented_forward` would
need per-segment eviction starts from the events. Adding `eviction_start`
to CompactionEvent naturally supports this generalization.

## Diagnostic checklist for this bug

1. **Is `protected_prefix_tokens` set to -1 in the config?**
   Check both `[trainer.compaction]` and `[inference.vllm_extra]` in the
   experiment's TOML.

2. **Do the vLLM logs show `auto-detected system prompt boundary < prompt_len`?**
   If yes, vLLM is in protected-prefix mode and eviction starts inside the
   prompt, not after it.

3. **Is the auto-detect formula giving 0?**
   Add logging in train.py after line 588:
   ```python
   logger.warning(f"[DIAG] pp auto-detect: pos_off={first_evt.position_offset_after} "
                  f"evicted={first_evt.tokens_evicted} pp={pp}")
   ```
   If `pp=0`, this bug is active.

4. **Compare evict_start values.**
   vLLM logs: `evict_start=368`
   Trainer: `prompt_aligned_len = ceil(mb_prompt_len/16)*16` (log it)
   If they differ → this is your mismatch source.

## Impact

This bug affects any compaction run with `protected_prefix_tokens=-1` (or
any non-zero protected prefix smaller than the prompt). The larger the gap
between the protected prefix and the turn-0 prompt, the larger the KL
mismatch.

For the debug_balrog config:
- Protected prefix: 367 → evict_start 368
- Turn-0 prompt: 389 → prompt_aligned_len 400
- Gap: 32 tokens → KL 1.27

For production rg-mix (single-turn, pp=0): unaffected (evict_start =
prompt_aligned_len by construction).
