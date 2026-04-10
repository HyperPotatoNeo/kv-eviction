# SPDX-License-Identifier: Apache-2.0
"""Wire types for the kv-eviction integration layer.

CompactionEventWire is the canonical wire format for transporting vLLM KV
cache compaction events from inference to the trainer.

Architecture note: the authoritative struct definition lives in prime-rl at
prime_rl.transport.types, because it's a field on TrainingSample / MicroBatch
and those structs are owned by prime-rl. We re-export it here so the rest of
the kv-eviction integration (env wrapper, trainer dispatch, segmented forward)
has a single canonical import path that doesn't leak prime-rl internals:

    from kv_eviction.types import CompactionEventWire

Keeping the definition in prime-rl means:
1. prime-rl stays free of vllm imports (prime-rl never sees
   vllm.v1.core.compaction.types.CompactionEvent).
2. No inverted dependency: prime-rl doesn't import from kv_eviction.
3. TrainingSample's msgspec field type is a local type, not a foreign import.

The env wrapper is the single translation boundary: it reads vllm's
pydantic CompactionEventPayload from ChatCompletionResponse and converts
to CompactionEventWire before building TrainingSamples.
"""

from prime_rl.transport.types import CompactionEventWire

__all__ = ["CompactionEventWire"]
