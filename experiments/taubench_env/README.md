# tau-bench-train

Training-capable verifiers wrapper around Sierra's original `tau-bench`
(https://github.com/sierra-research/tau-bench). Unlike `tau2-bench` and
`tau3-bench` (eval-only), `tau-bench` ships a real retail train split:

- `domain="retail"` → 500 train tasks (`dataset`) + 115 test tasks (`eval_dataset`)
- `domain="airline"` → 50 test tasks (aliased as both `dataset` and `eval_dataset`; no upstream train split)

The env id is `tau-bench-train` (verifiers maps this to the installed module
`tau_bench_train`, which this package installs).

## User simulator routing

Upstream `tau_bench.envs.user.LLMUserSimulationEnv` calls `litellm.completion(...)`
with no hook for `api_base` / `api_key`. We monkey-patch the module-level
`completion` symbol at wrapper import time (idempotent, sentinel-guarded) so
calls inherit `api_base` / `api_key` from environment variables, which the
wrapper sets from the `user_base_url` and `user_api_key_var` kwargs. This
routes the user simulator at a local vLLM endpoint without forking tau-bench.

## Example usage

```toml
[[orchestrator.train.env]]
id = "tau-bench-train"
args = { domain = "retail",
         user_model = "openai/Qwen/Qwen3-4B-Instruct-2507",
         user_base_url = "http://localhost:8000/v1",
         user_api_key_var = "OPENAI_API_KEY",
         max_num_steps = 30 }
```

The `openai/` prefix on `user_model` tells litellm to use the OpenAI-compatible
client path (vLLM serves this at `/v1/chat/completions`). `OPENAI_API_KEY` is
read from env — our EAI launcher sets it to `EMPTY` when unset, which vLLM
accepts.
