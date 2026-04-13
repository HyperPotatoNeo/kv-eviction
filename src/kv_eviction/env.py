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

import logging
from typing import Any

from verifiers.types import Messages, Response as ModelResponse, State, TrajectoryStep

from kv_eviction.types import CompactionEventWire

logger = logging.getLogger(__name__)


def _extract_compaction_event_dicts(
    response: ModelResponse,
) -> list[dict] | None:
    """Pull compaction events off a vllm ChatCompletion response as a list
    of plain JSON-serializable dicts.

    vllm's server attaches an OpenAI-extension field `compaction_events` at
    the top level of ChatCompletionResponse. The official openai-python SDK
    preserves unknown fields via pydantic's model_extra and also exposes them
    as regular attribute access, so `response.compaction_events` works.

    We return dicts (NOT msgspec CompactionEventWire instances) because
    verifiers routes trajectory-step extras through a JSON-serializability
    check (`state_columns value for 'trajectory' is not JSON-serializable`).
    Msgspec structs are not JSON-serializable as-is, so storing them there
    causes every rollout to fail with `state_columns value ... is not
    JSON-serializable: list`. The conversion to CompactionEventWire happens
    later in prime-rl's `orchestrator.trajectories._compaction_events_from_step`
    which already handles the dict form defensively (see its
    `elif isinstance(e, dict):` branch).

    Returns None when compaction is disabled on the server, the request had
    no compaction events, or the response type is something unexpected.
    """
    if response is None:
        return None
    raw = getattr(response, "compaction_events", None)
    # When the attribute is missing but pydantic v2 stashed it in model_extra,
    # fall through to the extras dict.
    if raw is None and hasattr(response, "model_extra"):
        extras = response.model_extra or {}
        raw = extras.get("compaction_events")
    if raw is None:
        return None
    events: list[dict] = []
    for e in raw:
        if isinstance(e, dict):
            try:
                events.append(
                    {
                        "num_output_tokens_at_compaction": int(
                            e["num_output_tokens_at_compaction"]
                        ),
                        "tokens_evicted": int(e["tokens_evicted"]),
                        "position_offset_after": int(e["position_offset_after"]),
                        "num_prompt_tokens": int(e.get("num_prompt_tokens", 0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(e, CompactionEventWire):
            # Someone already converted (unlikely in our flow, but defensive).
            events.append(
                {
                    "num_output_tokens_at_compaction": e.num_output_tokens_at_compaction,
                    "tokens_evicted": e.tokens_evicted,
                    "position_offset_after": e.position_offset_after,
                    "num_prompt_tokens": e.num_prompt_tokens,
                }
            )
        else:
            # Object form (e.g. pydantic CompactionEventPayload from a
            # locally-constructed vllm response — rare). Best-effort attribute
            # read.
            try:
                events.append(
                    {
                        "num_output_tokens_at_compaction": int(
                            getattr(e, "num_output_tokens_at_compaction")
                        ),
                        "tokens_evicted": int(getattr(e, "tokens_evicted")),
                        "position_offset_after": int(
                            getattr(e, "position_offset_after")
                        ),
                        "num_prompt_tokens": int(
                            getattr(e, "num_prompt_tokens", 0)
                        ),
                    }
                )
            except (AttributeError, TypeError, ValueError):
                continue
    return events or None


# Backwards-compat alias for the pre-JSON-fix name.
def _extract_compaction_events(response):
    return _extract_compaction_event_dicts(response)


def attach_compaction_events_from_response(
    step: TrajectoryStep,
    response: ModelResponse,
) -> None:
    """Mutate the given TrajectoryStep's extras to include compaction_events
    as a list of JSON-serializable dicts (not msgspec CompactionEventWire
    instances — see `_extract_compaction_event_dicts` docstring for why).

    Idempotent: overwrites any existing "compaction_events" entry on the
    step's extras dict with the events pulled from the response.
    """
    event_dicts = _extract_compaction_event_dicts(response)
    if event_dicts is None:
        return
    if step.get("extras") is None:
        step["extras"] = {}
    step["extras"]["compaction_events"] = event_dicts


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


# ─── Module-level monkey-patches ───
#
# The CompactionEnvMixin approach above requires every env author to
# explicitly subclass it. In practice, env packages like rg-mix-env do
# NOT subclass it — their class definitions look like
# `class RGMixEnv(vf.SingleTurnEnv):` with no mention of compaction. And
# the upstream verifiers library strips unknown fields during
# ChatCompletion -> verifiers.Response conversion (see
# verifiers/clients/openai_chat_completions_client.py:from_native_response,
# which constructs Response(id=..., created=..., model=..., usage=...,
# message=...) from a hardcoded field list).
#
# Result: when the compaction-enabled vLLM attaches `compaction_events`
# to its ChatCompletionResponse JSON, the openai-python SDK preserves
# the field (ChatCompletion model has extra="allow"), but verifiers'
# client adapter DROPS it when constructing its own Response, and the
# trajectory step's `extras` never gets populated, and
# `trainer.compaction_events` is always None. The segmented forward
# never fires and the trainer reforwards compaction rollouts in full
# context against post-eviction inference logprobs → large, spurious
# Mismatch KL.
#
# Fix: at module import time, monkey-patch two verifiers hook points to
# plumb compaction_events all the way through:
#
#   1. OpenAIChatCompletionsClient.from_native_response: copy
#      compaction_events from the native openai ChatCompletion (where
#      pydantic extra="allow" preserved it) to the verifiers Response
#      object (whose CustomBaseModel also has extra="allow", so setattr
#      works).
#
#   2. MultiTurnEnv.add_model_response: after the base class appends a
#      TrajectoryStep with extras={}, read the response's
#      compaction_events attribute and copy into the step's extras. This
#      is identical to what CompactionEnvMixin does; we just apply it
#      unconditionally to the base class so every env subclass benefits.
#
# Both patches are idempotent (sentinel-attribute guarded) so repeated
# imports of this module are safe. The patches only fire if the
# verifiers package is importable — if verifiers is missing, they
# silently no-op so unrelated kv_eviction consumers aren't affected.


_MAX_SEQ_LEN_WARNING_EMITTED = False


def _disable_per_turn_truncation(env: Any) -> None:
    """Disable verifiers' per-turn max_seq_len truncation for compaction runs.

    verifiers' parse_response_tokens truncates completion_ids when
    prompt_len + completion_len > max_seq_len, but does NOT truncate the
    compaction events (which are in response metadata, not the token
    lists). This puts completion_ids and compaction event coordinates in
    different spaces, causing a non-monotonic boundary assertion in
    interleave_rollout.

    With compaction active, per-turn truncation is unnecessary: the
    trainer's prepare_sample (batch.py) handles final seq_len clamping
    with proper compaction event filtering via _clamp_compaction_events.

    We null out max_seq_len on the env instance so parse_response_tokens
    skips the truncation and completion_ids stays aligned with the events.
    """
    global _MAX_SEQ_LEN_WARNING_EMITTED
    max_seq_len = getattr(env, "max_seq_len", None)
    if max_seq_len is not None:
        if not _MAX_SEQ_LEN_WARNING_EMITTED:
            logger.warning(
                "kv_eviction: compaction hooks active — ignoring env.max_seq_len=%d "
                "for per-turn token truncation. Compaction events require untruncated "
                "completion_ids to maintain coordinate alignment. Final seq_len "
                "clamping is handled by the trainer's prepare_sample.",
                max_seq_len,
            )
            _MAX_SEQ_LEN_WARNING_EMITTED = True
        env.max_seq_len = None


def _install_compaction_event_hooks() -> None:
    try:
        from verifiers.clients import openai_chat_completions_client as _vf_client
        from verifiers.envs import multiturn_env as _vf_mt
    except ImportError:
        return

    # --- Patch 1: OpenAIChatCompletionsClient.from_native_response ---
    base_client_cls = _vf_client.OpenAIChatCompletionsClient
    orig_from_native = base_client_cls.from_native_response
    if not getattr(orig_from_native, "__kv_eviction_patched__", False):

        async def patched_from_native(self, response):  # type: ignore[no-untyped-def]
            # Call the original conversion to get the verifiers Response.
            verifiers_response = await orig_from_native(self, response)
            # Read compaction_events off the raw openai ChatCompletion.
            # openai-python's ChatCompletion has pydantic extra="allow",
            # so vLLM's compaction_events field is preserved on the
            # native response either as an attribute or in model_extra.
            raw_events = getattr(response, "compaction_events", None)
            if raw_events is None and hasattr(response, "model_extra"):
                extra = response.model_extra or {}
                raw_events = extra.get("compaction_events")
            if raw_events is not None:
                # Verifiers' Response class is CustomBaseModel with
                # extra="allow", so setattr on an extra field is
                # supported in pydantic v2. If for some reason it's
                # not, fall back to stashing in model_extra directly.
                try:
                    setattr(verifiers_response, "compaction_events", raw_events)
                except Exception:
                    if hasattr(verifiers_response, "model_extra"):
                        if verifiers_response.model_extra is None:
                            verifiers_response.__pydantic_extra__ = {}
                        verifiers_response.model_extra["compaction_events"] = raw_events
            return verifiers_response

        patched_from_native.__kv_eviction_patched__ = True  # type: ignore[attr-defined]
        base_client_cls.from_native_response = patched_from_native  # type: ignore[assignment]

    # --- Patch 2: MultiTurnEnv.add_model_response ---
    #
    # Two responsibilities:
    #   a) Attach compaction events from the response to the trajectory step.
    #   b) Disable per-turn max_seq_len truncation (see _disable_per_turn_truncation).
    base_env_cls = _vf_mt.MultiTurnEnv
    orig_add_model_response = base_env_cls.add_model_response
    if not getattr(orig_add_model_response, "__kv_eviction_patched__", False):

        async def patched_add_model_response(self, state, prompt_messages, response):  # type: ignore[no-untyped-def]
            _disable_per_turn_truncation(self)
            await orig_add_model_response(self, state, prompt_messages, response)
            trajectory = state.get("trajectory", [])
            if trajectory:
                attach_compaction_events_from_response(trajectory[-1], response)

        patched_add_model_response.__kv_eviction_patched__ = True  # type: ignore[attr-defined]
        base_env_cls.add_model_response = patched_add_model_response  # type: ignore[assignment]


_install_compaction_event_hooks()


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
                    num_prompt_tokens=int(e.get("num_prompt_tokens", 0)),
                )
            )
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            # array_like msgspec form: [n, tokens_evicted, position_offset_after, num_prompt_tokens]
            out.append(
                CompactionEventWire(
                    num_output_tokens_at_compaction=int(e[0]),
                    tokens_evicted=int(e[1]),
                    position_offset_after=int(e[2]),
                    num_prompt_tokens=int(e[3]) if len(e) >= 4 else 0,
                )
            )
    return out or None
