# Plan: Fix no-eviction baseline to use same request pipeline as eviction run

## Problem

The no-eviction run (`rl_no_eviction.toml`) currently gets reward=0 because it
uses a different request path than the eviction run. With `enable_auto_tool_choice
= true` and `tool_call_parser = "hermes"`, vLLM extracts structured tool calls
from model output, leaving `content` empty. The balrog env expects to parse
`<tool_call>` tags from raw text (the same way the eviction run works), so it
sees empty or malformed responses and assigns 0 reward.

Root cause trace:
1. No-eviction run has no `[orchestrator.compaction_padding]` → interceptor inactive
2. Requests go to vLLM as standard chat completions with `tool_choice="auto"`
3. Without `enable_auto_tool_choice`, vLLM returns 400 on every request
4. Fix attempt: added `enable_auto_tool_choice = true` + `tool_call_parser =
   "hermes"` → 400s resolved, but now vLLM returns structured tool calls instead
   of raw text → env can't parse actions → reward = 0

## Goal

Make the no-eviction baseline use the exact same request pipeline as the eviction
run. The only difference between the two configs should be whether vLLM's
scheduler fires KV cache eviction or not.

## Request pipeline in the eviction run (reference)

```
balrog env
  → compaction padding interceptor (env.py)
    → tokenizes messages
    → pads each message to block boundary
    → sends prompt_token_ids to vLLM (bypasses chat template + tool validation)
  → vLLM generates raw text
  → env parses <tool_call> tags from completion text
```

## Changes to rl_no_eviction.toml

### 1. Restore `use_token_client = false`
Same as the eviction run. The TITO path doesn't change the 400 issue and is
not how the eviction run works.

### 2. Add `[orchestrator.compaction_padding]` section
```toml
[orchestrator.compaction_padding]
enabled = true
block_size = 16
log_evicted_text = false
```
This activates the interceptor in `env.py`. Even though no eviction fires on
the vLLM side, the interceptor still:
- tokenizes each chat message
- pads closing `<|im_end|>` to block boundary (harmless without eviction)
- sends `prompt_token_ids` to vLLM instead of the message list

This bypasses `tool_choice` validation entirely (no `tools` field in a
`prompt_token_ids` request), and the model generates raw text which the env
parses as usual.

### 3. Remove `enable_auto_tool_choice` and `tool_call_parser` from `[inference.vllm_extra]`
No longer needed — requests won't go through the chat completion tool
validation path.

### 4. Restore `use_token_client = false`
Already covered in step 1, but confirm it's back to match the eviction run.

## Resulting diff (rl_no_eviction.toml)

```toml
[orchestrator]
batch_size = 64        # match eviction run
rollouts_per_example = 8
oversampling_factor = 1.0
use_token_client = false   # restored: same as eviction run

[orchestrator.compaction_padding]
enabled = true             # activates the prompt_token_ids interceptor
block_size = 16            # matches vLLM block_size
log_evicted_text = false   # no eviction to log

# ... (no changes to other orchestrator sections) ...

[inference.vllm_extra]
async_scheduling = false
# enable_auto_tool_choice and tool_call_parser REMOVED
```

## Why this is the right baseline

- Same tokenization path → byte-identical prompt token IDs for same input
- Same raw-text generation → same `<tool_call>` parsing in the env
- Same reward signal computation
- Only difference: vLLM's `compaction_window_size` / `compaction_max_turns`
  are absent → no eviction fires → KV cache grows unbounded per episode

## Verification

After relaunching, check:
```bash
# Should see step completions, not empty-trajectory spam
tail -f .../runs2-no-eviction/logs/orchestrator.log

# Should see 200s, not 400s
grep "POST /v1/chat" .../runs2-no-eviction/logs/inference.log | head -5

# Should see nonzero rewards in wandb
# project: kv-eviction / run: debug-balrog-no-eviction
```
