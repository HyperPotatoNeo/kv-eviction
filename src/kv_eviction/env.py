# SPDX-License-Identifier: Apache-2.0
"""Verifiers env wrapper that captures vLLM KV cache compaction events.

Any verifiers env (SingleTurnEnv, MultiTurnEnv, ToolEnv, ...) that uses the
compaction-enabled vllm server will receive a `compaction_events` field on
its ChatCompletion response objects. This module provides a mixin that
reads those events off the response and stashes them in the per-step
`extras` dict. Downstream, prime-rl's `interleave_rollout` reads the first
step's `extras["compaction_events"]` and attaches it to the TrainingSample.

Usage:

    class MyCompactionEnv(CompactionEnvMixin, SingleTurnEnv):
        ...

or for ad-hoc wrapping, use `attach_compaction_events_from_response` as a
utility inside your own env's add_model_response override.

Why a mixin instead of a concrete class: users already subclass
SingleTurnEnv, MultiTurnEnv, etc. The mixin is cooperative and doesn't
care which base class it sits alongside, as long as the MRO places it
before the verifiers env so its `add_model_response` runs first.
"""

from typing import Any

from verifiers.types import Messages, ModelResponse, State, TrajectoryStep

from kv_eviction.types import CompactionEventWire


def _extract_compaction_events(
    response: ModelResponse,
) -> list[CompactionEventWire] | None:
    """Pull compaction events off a vllm ChatCompletion response.

    vllm's server attaches an OpenAI-extension field `compaction_events` at
    the top level of ChatCompletionResponse. The official openai-python SDK
    preserves unknown fields via pydantic's model_extra and also exposes them
    as regular attribute access, so `response.compaction_events` works.

    Returns None when compaction is disabled on the server, the request had
    no compaction events, or the response type is something unexpected.
    """
    if response is None:
        return None
    raw = getattr(response, "compaction_events", None)
    if raw is None:
        return None
    # vllm emits a list of dicts: {num_output_tokens_at_compaction,
    # tokens_evicted, position_offset_after}. Convert each to the msgspec
    # wire struct. Skip malformed entries defensively rather than crashing
    # the rollout: the reward has already been earned at this point, losing
    # compaction metadata downgrades training fidelity but doesn't invalidate
    # the trajectory for other methods.
    events: list[CompactionEventWire] = []
    for e in raw:
        if isinstance(e, dict):
            try:
                events.append(
                    CompactionEventWire(
                        num_output_tokens_at_compaction=int(
                            e["num_output_tokens_at_compaction"]
                        ),
                        tokens_evicted=int(e["tokens_evicted"]),
                        position_offset_after=int(e["position_offset_after"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return events or None


def attach_compaction_events_from_response(
    step: TrajectoryStep,
    response: ModelResponse,
) -> None:
    """Mutate the given TrajectoryStep's extras to include compaction_events.

    Idempotent: overwrites any existing "compaction_events" entry on the
    step's extras dict with the events pulled from the response.
    """
    events = _extract_compaction_events(response)
    if events is None:
        return
    if step.get("extras") is None:
        step["extras"] = {}
    step["extras"]["compaction_events"] = events


class CompactionEnvMixin:
    """Cooperative mixin: pulls compaction_events off each vllm response and
    attaches them to the trajectory step's extras dict.

    Intended usage:
        class MyEnv(CompactionEnvMixin, SingleTurnEnv):
            ...

    The MRO places CompactionEnvMixin before the verifiers base, so this
    override runs before the base's. We call super() to delegate the actual
    step construction + trajectory append, then reach into the last trajectory
    step and attach compaction metadata from the response.
    """

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ) -> None:
        await super().add_model_response(state, prompt_messages, response)  # type: ignore[misc]
        trajectory: list[TrajectoryStep] = state.get("trajectory", [])
        if not trajectory:
            return
        # The base class just appended a step. Attach compaction metadata to it.
        attach_compaction_events_from_response(trajectory[-1], response)


def compaction_events_from_step_extras(
    extras: dict[str, Any] | None,
) -> list[CompactionEventWire] | None:
    """Read-side helper for orchestrator code: pull compaction events from
    a trajectory step's extras dict, returning None if absent or invalid.

    Used by the interleave_rollout path in prime-rl to pass compaction events
    from vf.RolloutOutput into TrainingSample.
    """
    if not extras:
        return None
    events = extras.get("compaction_events")
    if not events:
        return None
    # Already-typed (produced by this process) OR round-tripped through
    # msgspec serialization (may have been encoded/decoded as lists). Handle
    # both: if the items are already CompactionEventWire, pass through;
    # otherwise attempt to construct.
    out: list[CompactionEventWire] = []
    for e in events:
        if isinstance(e, CompactionEventWire):
            out.append(e)
        elif isinstance(e, dict):
            out.append(
                CompactionEventWire(
                    num_output_tokens_at_compaction=int(
                        e["num_output_tokens_at_compaction"]
                    ),
                    tokens_evicted=int(e["tokens_evicted"]),
                    position_offset_after=int(e["position_offset_after"]),
                )
            )
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            # array_like msgspec form: [n, tokens_evicted, position_offset_after]
            out.append(
                CompactionEventWire(
                    num_output_tokens_at_compaction=int(e[0]),
                    tokens_evicted=int(e[1]),
                    position_offset_after=int(e[2]),
                )
            )
    return out or None
