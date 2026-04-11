# kv_eviction: Native vLLM KV Cache Compaction for RL
#
# Importing kv_eviction (or anything under it like kv_eviction.types,
# which prime-rl's orchestrator does at startup) triggers kv_eviction.env's
# module-level monkey-patches that plumb compaction_events from vLLM's
# ChatCompletion responses through verifiers' client adapter and
# MultiTurnEnv.add_model_response into the trajectory step's extras dict.
# Without these patches, verifiers silently drops vLLM's compaction_events
# field during response conversion and the trainer never sees any events,
# always takes the full-context forward path, and reports inflated
# Mismatch KL because inference used evicted KV but trainer used full KV.
from . import env as env  # noqa: F401
