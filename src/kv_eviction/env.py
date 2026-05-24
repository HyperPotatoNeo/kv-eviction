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
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from verifiers.types import Messages, Response as ModelResponse, State, TrajectoryStep

from kv_eviction.padding import render_padded_prompt
from kv_eviction.summarization import (
    build_exchange,
    build_post_summary_messages,
    count_summary_exchanges,
    extract_completion_logprobs,
    extract_completion_token_ids,
    extract_completion_token_ids as _extract_completion_token_ids_for_phase4,
    extract_prompt_token_ids as _summary_extract_prompt_token_ids,
    partition_messages,
    sanitize_summary,
)
from kv_eviction.truncation import truncate_messages_to_last_k_turns
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
                        "evict_start": int(e.get("evict_start", 0)),
                        "new_user_fragment_len": int(
                            e.get("new_user_fragment_len", 0)
                        ),
                        "kept_indices": [
                            int(x) for x in (e.get("kept_indices") or [])
                        ],
                        "kept_token_ids": [
                            int(x) for x in (e.get("kept_token_ids") or [])
                        ],
                        "last_turn_evicted": int(
                            e.get("last_turn_evicted", -1)
                        ),
                        "num_turns_evicted_after": int(
                            e.get("num_turns_evicted_after", 0)
                        ),
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
                    "evict_start": e.evict_start,
                    "new_user_fragment_len": getattr(e, "new_user_fragment_len", 0),
                    "kept_indices": list(getattr(e, "kept_indices", []) or []),
                    "kept_token_ids": list(getattr(e, "kept_token_ids", []) or []),
                    "last_turn_evicted": getattr(e, "last_turn_evicted", -1),
                    "num_turns_evicted_after": getattr(
                        e, "num_turns_evicted_after", 0
                    ),
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
                        "evict_start": int(getattr(e, "evict_start", 0)),
                        "new_user_fragment_len": int(
                            getattr(e, "new_user_fragment_len", 0)
                        ),
                        "kept_indices": [
                            int(x) for x in (getattr(e, "kept_indices", []) or [])
                        ],
                        "kept_token_ids": [
                            int(x) for x in (getattr(e, "kept_token_ids", []) or [])
                        ],
                        "last_turn_evicted": int(
                            getattr(e, "last_turn_evicted", -1)
                        ),
                        "num_turns_evicted_after": int(
                            getattr(e, "num_turns_evicted_after", 0)
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


def _extract_prompt_token_ids(response: ModelResponse) -> list[int] | None:
    """Pull `prompt_token_ids` off a response object.

    Block-aligned-padding mode: the AsyncCompletions interceptor
    (`_install_message_padding_interceptor`) stashes the padded ids on
    the native ChatCompletion before returning it to verifiers. Patch #1
    then copies it onto the verifiers Response. Either location works;
    we handle both for robustness."""
    if response is None:
        return None
    ids = getattr(response, "prompt_token_ids", None)
    if ids is None and hasattr(response, "model_extra"):
        extras = response.model_extra or {}
        ids = extras.get("prompt_token_ids")
    if ids is None:
        return None
    # Defensive copy + type coercion so downstream JSON-serialization works.
    try:
        return [int(x) for x in ids]
    except (TypeError, ValueError):
        return None


def attach_prompt_token_ids_from_response(
    step: TrajectoryStep,
    response: ModelResponse,
) -> None:
    """Mutate the given TrajectoryStep's extras to include `prompt_token_ids`
    — the padded token stream vLLM actually ran on.

    Idempotent: overwrites any existing entry. No-op when the response
    has no `prompt_token_ids` (padding disabled for this request)."""
    ids = _extract_prompt_token_ids(response)
    if ids is None:
        return
    if step.get("extras") is None:
        step["extras"] = {}
    step["extras"]["prompt_token_ids"] = ids


def _extract_padding_token_ids(response: ModelResponse) -> list[int] | None:
    """Pull `padding_token_ids` off a response object.

    vLLM's auto-pad-on-finish surfaces these on ChatCompletionResponse
    (see vllm/entrypoints/openai/chat_completion/protocol.py). When
    auto-pad fires for a request, these are the filler tokens vLLM
    appended to the KV cache after the final completion token so the
    trailing block lands in the prefix cache. Empty/None when auto-pad
    is disabled or did not fire."""
    if response is None:
        return None
    ids = getattr(response, "padding_token_ids", None)
    if ids is None and hasattr(response, "model_extra"):
        extras = response.model_extra or {}
        ids = extras.get("padding_token_ids")
    if not ids:
        return None
    try:
        return [int(x) for x in ids]
    except (TypeError, ValueError):
        return None


def attach_padding_token_ids_from_response(
    step: TrajectoryStep,
    response: ModelResponse,
) -> None:
    """Mutate the given TrajectoryStep's extras to include vLLM's auto-pad
    `padding_token_ids` for this turn.

    Idempotent. No-op when the response has no padding_token_ids
    (auto-pad disabled or this turn didn't trigger it)."""
    ids = _extract_padding_token_ids(response)
    if ids is None:
        return
    if step.get("extras") is None:
        step["extras"] = {}
    step["extras"]["padding_token_ids"] = ids


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
        attach_padding_token_ids_from_response(trajectory[-1], response)
        attach_summary_trainsample_from_response(trajectory[-1], response)


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

        def _forward_extra(verifiers_response, key, value):
            """Copy a vLLM-extension field onto the verifiers Response.
            Handles pydantic v2 `extra="allow"` via setattr, falls back to
            `model_extra` if setattr is rejected."""
            if value is None:
                return
            try:
                setattr(verifiers_response, key, value)
            except Exception:
                if hasattr(verifiers_response, "model_extra"):
                    if verifiers_response.model_extra is None:
                        verifiers_response.__pydantic_extra__ = {}
                    verifiers_response.model_extra[key] = value

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
            _forward_extra(verifiers_response, "compaction_events", raw_events)

            # Block-aligned padding extension: the AsyncCompletions
            # interceptor (Patch #3) stashes `prompt_token_ids` on the
            # native response so training/downstream code sees the exact
            # token stream vLLM ran on, not a re-tokenization of messages.
            raw_ptids = getattr(response, "prompt_token_ids", None)
            if raw_ptids is None and hasattr(response, "model_extra"):
                extra = response.model_extra or {}
                raw_ptids = extra.get("prompt_token_ids")
            _forward_extra(verifiers_response, "prompt_token_ids", raw_ptids)

            # Markovian summary extension: the AsyncCompletions
            # interceptor stashes a summary_trainsample dict on the
            # native response whenever a summary exchange was spliced
            # into the outgoing messages. Forward it so Patch #2 can
            # attach it to the trajectory step's extras for the
            # orchestrator to emit as a standalone TrainingSample.
            raw_summary = getattr(response, "summary_trainsample", None)
            if raw_summary is None and hasattr(response, "model_extra"):
                extra = response.model_extra or {}
                raw_summary = extra.get("summary_trainsample")
            _forward_extra(verifiers_response, "summary_trainsample", raw_summary)

            # Auto-pad extension: vLLM emits `padding_token_ids` on the
            # native ChatCompletionResponse whenever the auto-pad-on-
            # finish path appended filler tokens to land the trailing
            # block in the prefix cache. Forward to the verifiers
            # Response so Patch #2 can attach to the step's extras.
            raw_pad = getattr(response, "padding_token_ids", None)
            if raw_pad is None and hasattr(response, "model_extra"):
                extra = response.model_extra or {}
                raw_pad = extra.get("padding_token_ids")
            _forward_extra(verifiers_response, "padding_token_ids", raw_pad)

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
                attach_prompt_token_ids_from_response(trajectory[-1], response)
                attach_padding_token_ids_from_response(trajectory[-1], response)
                attach_summary_trainsample_from_response(trajectory[-1], response)

        patched_add_model_response.__kv_eviction_patched__ = True  # type: ignore[attr-defined]
        base_env_cls.add_model_response = patched_add_model_response  # type: ignore[assignment]


_install_compaction_event_hooks()


# ─── Block-aligned message padding (orchestrator-side) ───
#
# When enabled by the orchestrator via `configure_message_padding(...)`,
# the AsyncCompletions.create interceptor below:
#   1. Reads `messages` + `tools` off each chat.completions.create kwargs.
#   2. Renders a block-aligned padded token stream via
#      `kv_eviction.padding.render_padded_prompt`.
#   3. Merges `{"prompt_token_ids": padded}` into `extra_body` so the
#      server-side render_chat bypass (see vLLM fork) skips chat
#      templating and feeds these ids to the engine verbatim.
#   4. Stashes the padded ids on the returned ChatCompletion as an
#      attribute so Patch #1 (from_native_response) forwards them to the
#      verifiers Response and Patch #2 (add_model_response) attaches them
#      to the trajectory step's extras.
#
# The interceptor is a module-level monkey-patch of
# `openai.resources.chat.completions.completions.AsyncCompletions.create`.
# We intercept there (not at verifiers' `get_response`) because verifiers
# does not forward arbitrary kwargs to create() — it builds the kwarg list
# explicitly. Patching one level deeper means we don't need to touch
# verifiers at all.
#
# When `_padding_config` is None or `enabled=False`, the wrapper is a
# pure passthrough — zero runtime cost.


@dataclass
class MessagePaddingConfig:
    """Config installed by the orchestrator at startup. All fields are
    plumbed from prime-rl's `orchestrator.compaction_padding` section;
    `block_size` MUST be identical across inference / orchestrator /
    trainer (cross-validated at config load time)."""

    enabled: bool
    tokenizer: Any
    block_size: int
    filler_token_id: int
    im_end_token_id: int
    # Phase4 incremental prompt assembly. When True, after the first call
    # in a rollout (asyncio task), subsequent calls submit only
    # `prev_kept_state + new_user_fragment + fillers` instead of the
    # re-rendered full chat history. Requires the vLLM server to run with
    # `enable_prefix_caching=True` to actually realize the cache hit on
    # the prev_kept portion. Mirrors compaction_debug.py's PHASE4 path.
    phase4_enabled: bool = False


_padding_config: MessagePaddingConfig | None = None


# Markovian Thinker globals — forward-declared here because
# `_install_message_padding_interceptor()` below installs a closure
# (`patched_create`) that reads `_markovian_config` and mutates
# `_markovian_stats` on every request. If import was interrupted or
# raced between the installer call and the later module-level
# definitions, every subsequent rollout raised
# `NameError: name '_markovian_config' is not defined` (observed on EAI
# when env.py was written mid-import over NFS). The full config
# dataclass, constructor, and autoconfigure helper stay at their
# original location below — only the globals move up. Type is
# string-forward-referenced to keep the dataclass in its current spot.
_markovian_config: "MarkovianThinkerRuntimeConfig | None" = None
# Forward-declared for the same reason as `_markovian_config`: the
# patched_create closure captures these at install time. Full dataclass
# + configure helpers live below, alongside the Markovian equivalents.
_summary_config: "MarkovianSummaryRuntimeConfig | None" = None
_markovian_stats: dict[str, int] = {
    "n_truncations": 0,
    "n_messages_dropped": 0,
    "n_summaries": 0,
    "n_summary_failures": 0,
    "summary_prompt_tokens": 0,
    "summary_output_tokens": 0,
    "summary_latency_ms": 0,
}

# Recursion guard: set True inside `_generate_summary` before calling
# `orig_create` for the side-channel summary request, so the re-entrant
# invocation of `patched_create` short-circuits and does not try to
# intercept / re-summarize the summary call. contextvars (not
# threading.local) so async tasks migrating between threads still see
# the correct value.
_IN_SUMMARY_CALL: ContextVar[bool] = ContextVar(
    "_IN_SUMMARY_CALL", default=False
)


async def _generate_summary(
    orig_create,  # callable — the non-patched AsyncCompletions.create
    self_,  # the AsyncCompletions instance (passed as first positional arg)
    scfg,  # MarkovianSummaryRuntimeConfig
    *,
    outer_kwargs: dict,
    full_messages: list[dict],
) -> tuple[str | None, dict | None]:
    """Fire a side-channel summary request against the rollout model.

    Returns ``(text, train_sample_dict)`` on success, or ``(None, None)``
    on failure (empty response, or raised exception when
    ``on_error="drop"``). The caller decides what to do on ``None`` —
    typically, plain Markovian truncation fallback.

    ``train_sample_dict`` is a :class:`SummaryTrainSample`-serialized
    dict carrying the prompt tokens vLLM processed, the sampled
    completion tokens, and per-token logprobs. The caller attaches this
    to the outer response via :func:`_attach_summary_trainsample` so
    the orchestrator can emit a standalone ``TrainingSample`` from it
    in ``interleave_rollout``.

    ``orig_create`` is passed in explicitly (instead of captured via
    closure inside ``_install_message_padding_interceptor``) so this
    function is unit-testable with a mock.

    Recursion guard: ``_IN_SUMMARY_CALL`` is set True for the duration
    of the inner ``orig_create`` call so any subsequent re-entry into
    ``patched_create`` short-circuits and leaves the summary request
    un-intercepted. ``contextvars`` (not ``threading.local``) because
    the interceptor is async and tasks may migrate between threads.
    """
    import time as _time

    I_msg, _ = build_exchange(scfg.instruction_text, "")
    summary_messages = list(full_messages) + [I_msg]

    summary_kwargs = {
        "model": outer_kwargs.get("model"),
        "messages": summary_messages,
        "max_tokens": scfg.max_len_summary,
        "temperature": scfg.temperature,
        "top_p": scfg.top_p,
        "logprobs": True,
        "top_logprobs": 0,
    }

    # Eviction mode + padding enabled: render the summary call's prompt
    # block-aligned so its ``prompt_token_ids`` land on a block boundary.
    # Otherwise the trainer's ``prompt_aligned_len = ceil(prompt_len /
    # block_size) * block_size`` can overshoot ``seq_len`` on short
    # summaries and trip segmented_forward's invariant assert.
    pad_cfg = _padding_config
    if (
        scfg.mode == "eviction"
        and pad_cfg is not None
        and pad_cfg.enabled
    ):
        try:
            _raw, padded_ids, _pads = render_padded_prompt(
                tokenizer=pad_cfg.tokenizer,
                messages=summary_messages,
                tools=outer_kwargs.get("tools"),
                block_size=pad_cfg.block_size,
                filler_token_id=pad_cfg.filler_token_id,
                im_end_token_id=pad_cfg.im_end_token_id,
            )
        except Exception:
            logger.exception(
                "kv_eviction: summary-call render_padded_prompt failed; "
                "falling back to server-side rendering"
            )
        else:
            extra_body = dict(summary_kwargs.pop("extra_body", None) or {})
            extra_body["prompt_token_ids"] = padded_ids
            summary_kwargs["extra_body"] = extra_body

    token = _IN_SUMMARY_CALL.set(True)
    t0 = _time.perf_counter()
    try:
        resp = await orig_create(self_, **summary_kwargs)
    except Exception:
        if scfg.on_error == "raise":
            raise
        logger.warning(
            "kv_eviction: Markovian summary request failed; falling back "
            "to plain truncation",
            exc_info=True,
        )
        _markovian_stats["n_summary_failures"] += 1
        return None, None
    finally:
        _IN_SUMMARY_CALL.reset(token)
    latency_ms = int((_time.perf_counter() - t0) * 1000)

    try:
        raw_text = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        raw_text = ""
    text, was_sanitized = sanitize_summary(raw_text.strip())
    if not text:
        _markovian_stats["n_summary_failures"] += 1
        return None, None

    # Extract training-sample payload. If any extraction returns empty
    # we still return the text (the summary itself is usable for the
    # message-list splice); the sample_dict may just lack logprobs or
    # token ids, in which case interleave_rollout skips the emission.
    prompt_ids = _summary_extract_prompt_token_ids(resp)
    completion_ids = extract_completion_token_ids(resp)
    completion_logprobs = extract_completion_logprobs(resp)
    # Eviction mode: capture vLLM-side compaction events that fired
    # during the summary call's prefill/decode so the trainer treats
    # the summary sample as a compaction sample (events branch in
    # train.py's prompt_aligned_len math).
    summary_events: list[dict] = []
    if scfg.mode == "eviction":
        summary_events = list(_extract_compaction_event_dicts(resp) or [])
    sample_dict: dict | None = {
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "completion_logprobs": completion_logprobs,
        "model": outer_kwargs.get("model") or "",
        "compaction_events": summary_events,
    }

    if scfg.log_summaries:
        logger.info(
            "[SUMMARY] (%s, %d chars%s, %d prompt / %d completion tokens) %s",
            scfg.mode,
            len(text),
            ", sanitized" if was_sanitized else "",
            len(prompt_ids),
            len(completion_ids),
            text[:200],
        )

    _markovian_stats["n_summaries"] += 1
    _markovian_stats["summary_prompt_tokens"] += len(prompt_ids)
    _markovian_stats["summary_output_tokens"] += len(completion_ids)
    _markovian_stats["summary_latency_ms"] += latency_ms
    return text, sample_dict


def _attach_summary_trainsample(response: Any, sample_dict: dict) -> None:
    """Stash a :class:`SummaryTrainSample` dict on a ChatCompletion so
    Patch #1 (``from_native_response``) can forward it to the verifiers
    Response and Patch #2 (``add_model_response``) can copy it into the
    trajectory step's extras.

    Mirrors :func:`_stash_prompt_token_ids`: writes via ``setattr``,
    falls back to ``model_extra`` on pydantic subclasses that reject
    direct attribute writes.
    """
    try:
        setattr(response, "summary_trainsample", sample_dict)
    except Exception:
        if hasattr(response, "model_extra"):
            if response.model_extra is None:
                response.__pydantic_extra__ = {}
            response.model_extra["summary_trainsample"] = sample_dict


def _extract_summary_trainsample(response: Any) -> dict | None:
    """Pull a summary_trainsample dict off a response. Returns ``None``
    when absent. Mirrors :func:`_extract_compaction_event_dicts` —
    tolerant of both attribute access and ``model_extra``."""
    if response is None:
        return None
    raw = getattr(response, "summary_trainsample", None)
    if raw is None and hasattr(response, "model_extra"):
        extra = response.model_extra or {}
        raw = extra.get("summary_trainsample")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def attach_summary_trainsample_from_response(
    step: TrajectoryStep,
    response: ModelResponse,
) -> None:
    """Mutate the given TrajectoryStep's extras to include
    ``summary_trainsample`` — the training payload for the synthesized
    summary turn. Idempotent; no-op when the response has no summary."""
    sample = _extract_summary_trainsample(response)
    if sample is None:
        return
    if step.get("extras") is None:
        step["extras"] = {}
    step["extras"]["summary_trainsample"] = sample


def configure_message_padding(
    *,
    enabled: bool,
    tokenizer: Any,
    block_size: int,
    filler_token_id: int,
    im_end_token_id: int,
    phase4_enabled: bool = False,
) -> None:
    """Install the orchestrator-wide message-padding config.

    Called once by prime-rl's orchestrator at startup, before any
    rollouts fire. Idempotent — repeated calls overwrite the previous
    config (useful for tests).

    When `enabled=False`, the interceptor is still installed on the
    AsyncCompletions class (no way to un-install a monkey-patch
    cleanly) but becomes a no-op passthrough. This keeps behavior
    bit-identical to the pre-patch state when padding is disabled —
    see Gate 5 in `plans/prime_rl_message_padding_patch.md`.
    """
    global _padding_config
    _padding_config = MessagePaddingConfig(
        enabled=enabled,
        tokenizer=tokenizer,
        block_size=block_size,
        filler_token_id=filler_token_id,
        im_end_token_id=im_end_token_id,
        phase4_enabled=phase4_enabled,
    )
    if enabled:
        logger.info(
            "kv_eviction: block-aligned message padding ENABLED "
            "(block_size=%d, filler_token_id=%d, im_end_token_id=%d, "
            "phase4_enabled=%s)",
            block_size,
            filler_token_id,
            im_end_token_id,
            phase4_enabled,
        )


def _stash_prompt_token_ids(response: Any, ids: list[int]) -> None:
    """Attach ``prompt_token_ids`` to an OpenAI ChatCompletion so Patch #1
    (``from_native_response``) can forward it to the verifiers Response
    and Patch #2 (``add_model_response``) can copy it into the trajectory
    step's extras.

    ``ChatCompletion`` is a pydantic BaseModel with ``extra="allow"``, so
    setattr writes to ``__pydantic_extra__``. We fall back to writing
    ``model_extra`` directly when setattr is rejected by unusual pydantic
    subclasses.
    """
    try:
        setattr(response, "prompt_token_ids", ids)
    except Exception:
        if hasattr(response, "model_extra"):
            if response.model_extra is None:
                response.__pydantic_extra__ = {}
            response.model_extra["prompt_token_ids"] = ids


# ─── Phase4 incremental prompt assembly ───
#
# Mirrors `experiments/textworld_env/compaction_debug.py`'s chat_phase4 +
# derive_next_prev_state helpers (lines 281-368). For multi-turn rollouts
# with vLLM prefix caching enabled, each turn submits
#
#     prev_kept_state + [<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n]
#     + block-aligning fillers
#
# instead of re-rendering the full chat history every turn. `prev_kept_state`
# is the vLLM-authoritative survivors after the previous turn's eviction
# (or the previous turn's full submitted ids when no compaction fired),
# plus the previous turn's asst output + inter-message separator + filler.
#
# Per-rollout state is stashed on the asyncio task object — when the
# rollout coroutine finishes, the state is GC'd automatically. No
# cross-rollout contamination: each rollout runs in its own task.


def _get_phase4_state() -> dict | None:
    """Return the Phase4 state dict for the current async task, or None."""
    import asyncio
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    if task is None:
        return None
    return getattr(task, "_kv_eviction_phase4_state", None)


def _get_or_create_phase4_state() -> dict | None:
    """Return the current task's Phase4 state, creating it if possible."""
    import asyncio
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    if task is None:
        return None
    state = getattr(task, "_kv_eviction_phase4_state", None)
    if state is None:
        state = {}
        try:
            setattr(task, "_kv_eviction_phase4_state", state)
        except (AttributeError, TypeError):
            return None
    if "trace_id" not in state:
        state["trace_id"] = f"task-{id(task):x}"
        state["call_idx"] = 0
    return state


def _set_phase4_prev_state(prev_state_tokens: list[int]) -> None:
    """Stash the post-call KV state token sequence on the current async
    task so the NEXT chat() call in this rollout can build its prompt
    incrementally. No-op if not in an async task."""
    state = _get_or_create_phase4_state()
    if state is None:
        return
    state["prev_state_tokens"] = list(prev_state_tokens)


def _pad_tokens_after_im_end(
    tokens: list[int],
    start_offset: int,
    im_end_id: int,
    block_size: int,
    filler_id: int,
) -> tuple[list[int], int]:
    """Append tokens onto start_offset, inserting block-aligning fillers
    after each <|im_end|>. Mirrors compaction_debug.py:_pad_after_im_end."""
    out: list[int] = []
    running = start_offset
    total_pad = 0
    for tok in tokens:
        out.append(tok)
        running += 1
        if tok == im_end_id:
            remainder = running % block_size
            n = (block_size - remainder) % block_size
            if n:
                out.extend([filler_id] * n)
                running += n
                total_pad += n
    return out, total_pad


def _build_phase4_incremental_prompt(
    messages: list[dict],
    cfg: MessagePaddingConfig,
) -> tuple[list[int], int] | None:
    """Build the Phase4 incremental prompt: prev_state + padded new_user_fragment.

    Returns None when no prior state exists yet (first call), the last
    message isn't a plain-string user message, or any other condition
    that should fall back to full-history rendering.
    """
    state = _get_phase4_state()
    if state is None:
        return None
    prev_state = state.get("prev_state_tokens")
    if not prev_state:
        return None

    if not messages or messages[-1].get("role") != "user":
        return None
    content = messages[-1].get("content")
    if not isinstance(content, str):
        # Multimodal / tool-result content — fall back.
        return None

    # _NEW_USER_FRAGMENT template from compaction_debug.py:255. Renders
    # exactly the suffix Qwen3's chat template would have appended for
    # one new user message + asst generation prompt.
    fragment_text = (
        f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    )
    fragment_ids = cfg.tokenizer.encode(
        fragment_text, add_special_tokens=False
    )
    padded_fragment, _ = _pad_tokens_after_im_end(
        fragment_ids,
        len(prev_state),
        cfg.im_end_token_id,
        cfg.block_size,
        cfg.filler_token_id,
    )
    return list(prev_state) + padded_fragment, len(prev_state)


def _extract_kept_token_ids_last_event(response: Any) -> list[int] | None:
    """Read kept_token_ids off the last real eviction event in the response.

    Synthetic Phase4 inherit events deliberately carry empty kept_token_ids.
    Walk backward so those metadata-only events do not mask a preceding real
    eviction event and make Phase4 fall back to the full submitted prompt.
    """
    raw = getattr(response, "compaction_events", None)
    if raw is None and hasattr(response, "model_extra"):
        raw = (response.model_extra or {}).get("compaction_events")
    if not raw:
        return None
    for event in reversed(raw):
        if isinstance(event, dict):
            kept = event.get("kept_token_ids")
        else:
            kept = getattr(event, "kept_token_ids", None)
        if not kept:
            continue
        try:
            return [int(x) for x in kept]
        except (TypeError, ValueError):
            continue
    return None


def _update_phase4_state_from_response(
    response: Any,
    submitted_ids: list[int],
    cfg: MessagePaddingConfig,
) -> None:
    """Compute prev_state for the NEXT call in this rollout, mirroring
    V's auto-pad layout (just filler, no separator) so the orchestrator's
    incremental prompt matches V's actual cache content row-for-row.

    prev_state = kept_token_ids (if compaction fired, last event) OR
    full submitted prompt (otherwise) + asst output tokens + filler to
    block-align the next <|im_start|>user.

    Note: the prior implementation inserted a "\\n" separator before
    the filler to match Qwen3's chat-template-rendered form. But V's
    auto-pad (vllm/v1/core/sched/scheduler.py: auto_pad branch) emits
    only filler — no "\\n". The mismatch caused a 1-token drift per
    turn boundary between the trainer's persistent_cache layout and
    V's HBM cache layout. By turn 5 this accumulated to a 13-slot
    misalignment, which produced 4.5+ nat L1 K divergence on admission
    calls and ~0.05 production Mismatch KL. Removing the sep insertion
    restores layout parity (verified empirically: testbed admission KL
    0.039 -> 0.002, max 39.4 -> 0.285).
    """
    kept = _extract_kept_token_ids_last_event(response)
    if kept is None:
        kept = list(submitted_ids)
    asst = _extract_completion_token_ids_for_phase4(response)
    state = list(kept) + list(asst)
    remainder = len(state) % cfg.block_size
    n = (cfg.block_size - remainder) % cfg.block_size
    if n:
        state.extend([cfg.filler_token_id] * n)
    _set_phase4_prev_state(state)


def _install_message_padding_interceptor() -> None:
    """Monkey-patch `AsyncCompletions.create` with two independent branches:

    Branch A — Markovian Thinker client-side truncation: drop all but the
    last K turn groups from ``messages`` before the request leaves the
    orchestrator. vLLM runs a normal full-context completion on the
    truncated message list with no compaction.

    Branch B — block-aligned message padding: pre-tokenize ``messages``
    into a filler-padded token stream and pass it to vLLM via
    ``extra_body={"prompt_token_ids": ...}`` so turn-based KV eviction
    lands on block boundaries.

    The validator in ``prime-rl/src/prime_rl/configs/rl.py`` forbids
    enabling both simultaneously, so in practice at most one branch fires
    per request. The branches are independent and compose safely if that
    changed.

    Idempotent — sentinel-attribute guarded so repeated imports / test
    teardowns don't stack wrappers. No-op passthrough when neither config
    is enabled.
    """
    try:
        from openai.resources.chat.completions.completions import (
            AsyncCompletions,
        )
    except ImportError:
        return

    orig_create = AsyncCompletions.create
    if getattr(orig_create, "__kv_eviction_padding_patched__", False):
        return

    async def patched_create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Recursion guard — see `_IN_SUMMARY_CALL` docstring. We do this
        # first so the side-channel summary request bypasses the entire
        # interceptor (both branches).
        if _IN_SUMMARY_CALL.get():
            return await orig_create(self, *args, **kwargs)

        # --- Branch A: Markovian Thinker client-side truncation ---
        mcfg = _markovian_config
        if mcfg is not None and mcfg.enabled:
            messages = kwargs.get("messages")
            if messages is not None:
                orig_len = len(messages)
                scfg = _summary_config

                summary_fired = False
                summary_sample_dict: dict | None = None
                if (
                    scfg is not None
                    and scfg.enabled
                    and scfg.compaction_max_turns > 0
                    and scfg.instruction_text
                ):
                    n_groups, sys_prefix, body_groups, tail = partition_messages(
                        messages
                    )
                    n_real = n_groups - count_summary_exchanges(
                        messages, scfg.instruction_text
                    )
                    if n_real > scfg.compaction_max_turns:
                        summary_text, summary_sample_dict = await _generate_summary(
                            orig_create,
                            self,
                            scfg,
                            outer_kwargs=kwargs,
                            full_messages=messages,
                        )
                        if summary_text:
                            new_messages = build_post_summary_messages(
                                mode=scfg.mode,
                                sys_prefix=sys_prefix,
                                body_groups=body_groups,
                                tail=tail,
                                instruction_text=scfg.instruction_text,
                                summary_text=summary_text,
                                n_preserved_turns=(mcfg.stride or 0),
                                resume_text=scfg.resume_text,
                            )
                            summary_fired = True
                        else:
                            # Summary generation failed: fall through to
                            # plain Markovian truncation.
                            new_messages = truncate_messages_to_last_k_turns(
                                messages,
                                max_turns=mcfg.max_turns,
                                stride=mcfg.stride,
                                log_fn=(
                                    (lambda m: logger.info("[MARKOVIAN] %s", m))
                                    if mcfg.log_truncated_messages
                                    else None
                                ),
                            )
                    else:
                        new_messages = truncate_messages_to_last_k_turns(
                            messages,
                            max_turns=mcfg.max_turns,
                            stride=mcfg.stride,
                            log_fn=(
                                (lambda m: logger.info("[MARKOVIAN] %s", m))
                                if mcfg.log_truncated_messages
                                else None
                            ),
                        )
                else:
                    new_messages = truncate_messages_to_last_k_turns(
                        messages,
                        max_turns=mcfg.max_turns,
                        stride=mcfg.stride,
                        log_fn=(
                            (lambda m: logger.info("[MARKOVIAN] %s", m))
                            if mcfg.log_truncated_messages
                            else None
                        ),
                    )

                kwargs["messages"] = new_messages

                # Eviction-mode splice + padding enabled: render the
                # post-summary message list with block-aligned filler
                # padding so the trainer's ``prompt_aligned_len`` math
                # is exact (prompt_len already block-aligned → no
                # rounding overshoot). Otherwise: use the raw
                # apply_chat_template + encode path, which is what the
                # pre-summary markovian baseline relies on.
                pad_cfg_A = _padding_config
                use_padding_A = (
                    scfg is not None
                    and scfg.mode == "eviction"
                    and pad_cfg_A is not None
                    and pad_cfg_A.enabled
                )
                truncated_ids = None
                if use_padding_A:
                    try:
                        _raw, truncated_ids, _pads = render_padded_prompt(
                            tokenizer=pad_cfg_A.tokenizer,
                            messages=new_messages,
                            tools=kwargs.get("tools"),
                            block_size=pad_cfg_A.block_size,
                            filler_token_id=pad_cfg_A.filler_token_id,
                            im_end_token_id=pad_cfg_A.im_end_token_id,
                        )
                    except Exception:
                        logger.exception(
                            "kv_eviction: branch-A eviction-mode "
                            "render_padded_prompt failed; falling back "
                            "to raw chat-template encode"
                        )
                        truncated_ids = None
                if truncated_ids is None:
                    # Re-tokenize so the trainer uses the exact token stream
                    # vLLM will run on (see `plans/markovian_thinker_baseline.md`
                    # — "The prompt_token_ids divergence").
                    rendered = mcfg.tokenizer.apply_chat_template(
                        new_messages,
                        tools=kwargs.get("tools"),
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    truncated_ids = mcfg.tokenizer.encode(
                        rendered, add_special_tokens=False
                    )

                # Forward the pre-tokenized stream to vLLM via extra_body
                # (see the pre-summary version of this branch for the
                # long explanation of why).
                extra_body = dict(kwargs.pop("extra_body", None) or {})
                extra_body["prompt_token_ids"] = truncated_ids
                kwargs["extra_body"] = extra_body

                response = await orig_create(self, *args, **kwargs)
                _stash_prompt_token_ids(response, truncated_ids)
                if summary_sample_dict is not None:
                    _attach_summary_trainsample(response, summary_sample_dict)

                # Observability: truncation counters fire even when the
                # summary path ran (the interceptor still reduced or
                # rewrote the message list in some way).
                if len(new_messages) != orig_len or summary_fired:
                    _markovian_stats["n_truncations"] += 1
                    _markovian_stats["n_messages_dropped"] += max(
                        0, orig_len - len(new_messages)
                    )
                return response

        # --- Branch B: block-aligned message padding ---
        cfg = _padding_config
        if cfg is None or not cfg.enabled:
            return await orig_create(self, *args, **kwargs)

        phase4_state = _get_or_create_phase4_state() if cfg.phase4_enabled else None
        phase4_trace_id = (
            str(phase4_state.get("trace_id")) if phase4_state is not None else ""
        )
        phase4_call_idx = (
            int(phase4_state.get("call_idx", 0))
            if phase4_state is not None
            else -1
        )
        if phase4_state is not None:
            phase4_state["call_idx"] = phase4_call_idx + 1

        messages = kwargs.get("messages")
        tools = kwargs.get("tools")
        logger.debug(
            "[PAD-TRACE] interceptor fired: messages_is_none=%s "
            "num_messages=%s has_tools=%s",
            messages is None,
            len(messages) if messages is not None else "n/a",
            tools is not None,
        )
        if messages is None:
            # Someone called create() positionally or without messages
            # (streaming edge cases, non-chat paths). Don't touch it.
            return await orig_create(self, *args, **kwargs)

        # Phase4 incremental mode: on calls AFTER the first one in a
        # rollout (asyncio task), build prev_state + padded new user
        # fragment instead of re-rendering the full chat history.
        # Requires the vLLM server to run with enable_prefix_caching=True
        # so the prev_state portion hits the prefix cache. Falls back to
        # full-history render on first call / tools / non-string content.
        padded: list[int] | None = None
        phase4_expected_cached_tokens = 0
        used_phase4 = False
        if cfg.phase4_enabled and tools is None:
            try:
                built_phase4 = _build_phase4_incremental_prompt(messages, cfg)
            except Exception:
                logger.exception(
                    "kv_eviction: phase4 incremental build failed; "
                    "falling back to full-history render"
                )
                built_phase4 = None
            if built_phase4 is not None:
                padded, phase4_expected_cached_tokens = built_phase4
                used_phase4 = True
                logger.debug(
                    "[PHASE4] incremental: prev_state=%d total=%d",
                    phase4_expected_cached_tokens,
                    len(padded),
                )

        if padded is None:
            try:
                _raw, padded, _pads = render_padded_prompt(
                    tokenizer=cfg.tokenizer,
                    messages=messages,
                    tools=tools,
                    block_size=cfg.block_size,
                    filler_token_id=cfg.filler_token_id,
                    im_end_token_id=cfg.im_end_token_id,
                )
            except Exception:
                # If padding fails (unusual chat template, bad messages),
                # log and fall back to the unpadded path rather than
                # breaking the rollout. The trainer's padding-mode assertion
                # will fail-loud if this drift silently propagates.
                logger.exception(
                    "kv_eviction: render_padded_prompt failed; falling back to "
                    "unpadded chat template"
                )
                return await orig_create(self, *args, **kwargs)

            logger.debug(
                "[PAD-TRACE] padded: raw->padded len %d->%d fillers_inserted=%d",
                len(_raw),
                len(padded),
                sum(_pads),
            )

        # Merge into extra_body. `extra_body` is an officially-supported
        # passthrough kwarg on openai-python's create(); its contents go
        # straight into the HTTP request body, where vLLM's
        # ChatCompletionRequest pydantic model picks up the new
        # `prompt_token_ids` field.
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        extra_body["prompt_token_ids"] = padded
        if phase4_state is not None and phase4_trace_id:
            vllm_xargs = dict(extra_body.get("vllm_xargs") or {})
            vllm_xargs["kve_phase4_trace_id"] = phase4_trace_id
            vllm_xargs["kve_phase4_call_idx"] = int(phase4_call_idx)
            if used_phase4 and phase4_expected_cached_tokens > 0:
                vllm_xargs["kve_phase4_expected_cached_tokens"] = int(
                    phase4_expected_cached_tokens
                )
            extra_body["vllm_xargs"] = vllm_xargs
        elif used_phase4 and phase4_expected_cached_tokens > 0:
            vllm_xargs = dict(extra_body.get("vllm_xargs") or {})
            vllm_xargs["kve_phase4_expected_cached_tokens"] = int(
                phase4_expected_cached_tokens
            )
            extra_body["vllm_xargs"] = vllm_xargs
        kwargs["extra_body"] = extra_body

        # Force logprobs to be requested. Some upstream callers
        # (verifiers' env wrappers under certain code paths) skip the
        # logprobs flag, and vLLM defaults to NOT returning logprobs
        # → trainer.inference_logprobs ends up all-zeros → Mismatch KL
        # of ~0.67. Setting setdefault here is a no-op when the caller
        # already passes logprobs=True.
        kwargs.setdefault("logprobs", True)
        kwargs.setdefault("top_logprobs", 0)

        response = await orig_create(self, *args, **kwargs)
        _stash_prompt_token_ids(response, padded)

        # Phase4: derive prev_state for the NEXT call in this rollout.
        # We update state even on the first call (when used_phase4 is
        # False) so subsequent calls have prev_state available.
        if cfg.phase4_enabled:
            try:
                _update_phase4_state_from_response(response, padded, cfg)
            except Exception:
                logger.exception(
                    "kv_eviction: phase4 state update failed; next call "
                    "will fall back to full-history render"
                )
        del used_phase4  # placeholder for future stats
        return response

    patched_create.__kv_eviction_padding_patched__ = True  # type: ignore[attr-defined]
    AsyncCompletions.create = patched_create  # type: ignore[assignment]


_install_message_padding_interceptor()


def _autoconfigure_padding_from_env() -> None:
    """Auto-enable block-aligned message padding from environment variables.

    The orchestrator process sets these before spawning the verifiers env
    server subprocess (which runs in a fresh `mp.spawn` interpreter and
    thus won't inherit the orchestrator's `configure_message_padding(...)`
    call). The subprocess imports `kv_eviction` via its entrypoint shim,
    which triggers this function and re-configures padding from env vars.

    Env var contract (all required when KV_EVICTION_PADDING_MODEL is set):
      KV_EVICTION_PADDING_MODEL          — tokenizer name_or_path
      KV_EVICTION_PADDING_BLOCK_SIZE     — int, must match inference/trainer
      KV_EVICTION_PADDING_FILLER_ID      — int, already-resolved filler id
      KV_EVICTION_PADDING_IM_END_ID      — int, already-resolved im_end id
      KV_EVICTION_PADDING_PHASE4         — optional "1" to enable Phase4
                                            incremental prompt assembly

    No-ops if already configured (idempotent) or if env vars are absent.
    """
    import os as _os

    global _padding_config
    if _padding_config is not None and _padding_config.enabled:
        return
    model_name = _os.environ.get("KV_EVICTION_PADDING_MODEL")
    if not model_name:
        return
    try:
        block_size = int(_os.environ["KV_EVICTION_PADDING_BLOCK_SIZE"])
        filler_id = int(_os.environ["KV_EVICTION_PADDING_FILLER_ID"])
        im_end_id = int(_os.environ["KV_EVICTION_PADDING_IM_END_ID"])
    except (KeyError, ValueError) as e:
        logger.warning(
            "kv_eviction: KV_EVICTION_PADDING_MODEL set but other "
            "KV_EVICTION_PADDING_* vars missing/invalid (%s); padding NOT "
            "enabled in this process",
            e,
        )
        return
    phase4_enabled = _os.environ.get("KV_EVICTION_PADDING_PHASE4", "0") == "1"
    from transformers import AutoTokenizer  # local import to keep env.py lean

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    configure_message_padding(
        enabled=True,
        tokenizer=tokenizer,
        block_size=block_size,
        filler_token_id=filler_id,
        im_end_token_id=im_end_id,
        phase4_enabled=phase4_enabled,
    )


_autoconfigure_padding_from_env()


# ─── Markovian Thinker: client-side message truncation ───
#
# When enabled by the orchestrator via `configure_markovian_thinker(...)`,
# the AsyncCompletions.create interceptor (Branch A above) truncates each
# chat completion request's `messages` to the last K complete turn groups
# BEFORE the request reaches vLLM. vLLM runs a normal, full-context
# completion on the truncated prompt — no compaction, no eviction, no
# `CompactionEvent`s. The orchestrator re-tokenizes the truncated
# messages and stashes the exact token ids on the response so the
# trainer forwards against the same tokens vLLM saw (see
# `plans/markovian_thinker_baseline.md` → "The prompt_token_ids divergence").
#
# A validator in prime-rl (`validate_markovian_thinker` on RLConfig)
# rejects configurations that enable Markovian alongside vLLM or trainer
# compaction, block-aligned padding, or the TITO token client.


@dataclass
class MarkovianThinkerRuntimeConfig:
    """Runtime config installed by the orchestrator at startup. Fields
    come from prime-rl's `orchestrator.markovian_thinker` section."""

    enabled: bool
    tokenizer: Any
    max_turns: int
    log_truncated_messages: bool
    # Turn-preserve count applied on truncation triggers. `None` = keep
    # last max_turns (legacy single-knob behavior). Integer N ≥ 1 = keep
    # last N (decoupled: max_turns is the trigger, stride is the keep
    # count). Also used by the markovian-mode summary splice to carry N
    # real turns after the summary exchange.
    stride: int | None = None


# `_markovian_config` and `_markovian_stats` are forward-declared above
# the `_install_message_padding_interceptor()` call — see comment there.


def configure_markovian_thinker(
    *,
    enabled: bool,
    tokenizer: Any,
    max_turns: int,
    log_truncated_messages: bool = False,
    stride: int | None = None,
) -> None:
    """Install the orchestrator-wide Markovian Thinker config.

    Called once by prime-rl's orchestrator at startup, before any
    rollouts fire. Idempotent — repeated calls overwrite the previous
    config.
    """
    global _markovian_config
    _markovian_config = MarkovianThinkerRuntimeConfig(
        enabled=enabled,
        tokenizer=tokenizer,
        max_turns=max_turns,
        log_truncated_messages=log_truncated_messages,
        stride=stride,
    )
    if enabled:
        logger.info(
            "kv_eviction: Markovian Thinker ENABLED (max_turns=%d, stride=%s, log=%s)",
            max_turns,
            stride,
            log_truncated_messages,
        )


def pop_markovian_stats() -> dict[str, int]:
    """Drain-and-reset the Markovian counters. Called once per
    orchestrator step to emit `markovian/*` and `markovian_summary/*`
    metrics to wandb.
    """
    global _markovian_stats
    out = dict(_markovian_stats)
    _markovian_stats = {
        "n_truncations": 0,
        "n_messages_dropped": 0,
        "n_summaries": 0,
        "n_summary_failures": 0,
        "summary_prompt_tokens": 0,
        "summary_output_tokens": 0,
        "summary_latency_ms": 0,
    }
    return out


def _autoconfigure_markovian_from_env() -> None:
    """Auto-enable Markovian Thinker from environment variables.

    The orchestrator sets these before spawning the verifiers env server
    subprocess (mp.spawn starts a fresh interpreter that won't inherit
    the parent's `configure_markovian_thinker(...)` call). The subprocess
    imports `kv_eviction` via its entrypoint shim, which triggers this
    function and re-configures truncation from env vars.

    Env var contract:
      KV_EVICTION_MARKOVIAN_ENABLED    — "1" enables; absence disables.
      KV_EVICTION_MARKOVIAN_MAX_TURNS  — int (trigger threshold).
      KV_EVICTION_MARKOVIAN_MODEL      — tokenizer name_or_path.
      KV_EVICTION_MARKOVIAN_STRIDE     — optional int (post-trigger
        turn-preserve count). Absent → legacy (keep max_turns).

    No-ops if already configured or if env vars are absent.
    """
    import os as _os

    global _markovian_config
    if _markovian_config is not None and _markovian_config.enabled:
        return
    if _os.environ.get("KV_EVICTION_MARKOVIAN_ENABLED") != "1":
        return
    max_turns_str = _os.environ.get("KV_EVICTION_MARKOVIAN_MAX_TURNS")
    model_name = _os.environ.get("KV_EVICTION_MARKOVIAN_MODEL")
    if not max_turns_str or not model_name:
        logger.warning(
            "kv_eviction: KV_EVICTION_MARKOVIAN_ENABLED=1 but "
            "KV_EVICTION_MARKOVIAN_MAX_TURNS / KV_EVICTION_MARKOVIAN_MODEL "
            "missing; Markovian Thinker NOT enabled in this process"
        )
        return
    max_turns = int(max_turns_str)
    log_truncated = _os.environ.get("KV_EVICTION_MARKOVIAN_LOG") == "1"
    stride_str = _os.environ.get("KV_EVICTION_MARKOVIAN_STRIDE")
    stride = int(stride_str) if stride_str else None
    from transformers import AutoTokenizer  # local import to keep env.py lean

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    configure_markovian_thinker(
        enabled=True,
        tokenizer=tokenizer,
        max_turns=max_turns,
        log_truncated_messages=log_truncated,
        stride=stride,
    )


_autoconfigure_markovian_from_env()


# ─── Markovian Summary: summarization-based eviction ───
#
# When enabled by the orchestrator via `configure_markovian_summary(...)`,
# the AsyncCompletions.create interceptor's Branch A fires a side-channel
# summary request once the number of real turn groups exceeds
# `compaction_max_turns`, then splices a `{user: instruction, assistant:
# summary}` exchange into the outgoing message list. The summary itself
# is a trainable model turn — its tokens + logprobs are captured on the
# outer response via `extras["summary_trainsample"]` for the orchestrator
# to emit as a standalone TrainingSample.
#
# Two modes (both ride on the existing Markovian interceptor):
#   - "markovian": full client-side reset to `sys + [I, S] + tail`.
#     vLLM-side compaction must be OFF.
#   - "eviction": append-only `sys + body + [I, S] + tail`.
#     vLLM-side compaction (block or turn) must be ON.
#
# See `plans/markovian_summary.md` for the full design.


@dataclass
class MarkovianSummaryRuntimeConfig:
    """Runtime config installed by the orchestrator at startup. Fields
    come from prime-rl's `orchestrator.markovian_thinker.summary` section."""

    enabled: bool
    mode: str  # "markovian" | "eviction"
    compaction_max_turns: int
    max_len_summary: int
    instruction_text: str
    resume_text: str
    temperature: float
    top_p: float
    on_error: str  # "drop" | "raise"
    log_summaries: bool


# `_summary_config` is forward-declared above the
# `_install_message_padding_interceptor()` call, same as `_markovian_config`.


def configure_markovian_summary(
    *,
    enabled: bool,
    mode: str,
    compaction_max_turns: int,
    max_len_summary: int,
    instruction_text: str,
    resume_text: str = "",
    temperature: float = 0.3,
    top_p: float = 0.95,
    on_error: str = "drop",
    log_summaries: bool = False,
) -> None:
    """Install the orchestrator-wide Markovian Summary config.

    Called once by prime-rl's orchestrator at startup, before any
    rollouts fire. Idempotent — repeated calls overwrite the previous
    config.

    Validated upstream by `validate_markovian_summary` in
    prime-rl's `rl.py`. Does minimal sanity checking here.
    """
    if mode not in ("markovian", "eviction"):
        raise ValueError(
            f"configure_markovian_summary: invalid mode={mode!r}; "
            "expected 'markovian' or 'eviction'"
        )
    if on_error not in ("drop", "raise"):
        raise ValueError(
            f"configure_markovian_summary: invalid on_error={on_error!r}; "
            "expected 'drop' or 'raise'"
        )

    global _summary_config
    _summary_config = MarkovianSummaryRuntimeConfig(
        enabled=enabled,
        mode=mode,
        compaction_max_turns=compaction_max_turns,
        max_len_summary=max_len_summary,
        instruction_text=instruction_text,
        resume_text=resume_text,
        temperature=temperature,
        top_p=top_p,
        on_error=on_error,
        log_summaries=log_summaries,
    )
    if enabled:
        logger.info(
            "kv_eviction: Markovian Summary ENABLED "
            "(mode=%s, compaction_max_turns=%d, max_len_summary=%d, on_error=%s)",
            mode,
            compaction_max_turns,
            max_len_summary,
            on_error,
        )


def _autoconfigure_markovian_summary_from_env() -> None:
    """Auto-enable Markovian Summary from environment variables.

    The orchestrator sets these before spawning the verifiers env server
    subprocess (mp.spawn starts a fresh interpreter that won't inherit
    the parent's `configure_markovian_summary(...)` call). The subprocess
    imports `kv_eviction` via its entrypoint shim, which triggers this
    function.

    Env var contract (scalars):
      KV_EVICTION_MARKOVIAN_SUMMARY_ENABLED              — "1" enables.
      KV_EVICTION_MARKOVIAN_SUMMARY_MODE                 — "markovian" or "eviction"
      KV_EVICTION_MARKOVIAN_SUMMARY_COMPACTION_MAX_TURNS — int
      KV_EVICTION_MARKOVIAN_SUMMARY_MAX_LEN_SUMMARY      — int
      KV_EVICTION_MARKOVIAN_SUMMARY_TEMPERATURE          — float
      KV_EVICTION_MARKOVIAN_SUMMARY_TOP_P                — float
      KV_EVICTION_MARKOVIAN_SUMMARY_ON_ERROR             — "drop" | "raise"
      KV_EVICTION_MARKOVIAN_SUMMARY_LOG                  — "1" enables debug

    Long strings (instruction_text, resume_text) via JSON env var:
      KV_EVICTION_MARKOVIAN_SUMMARY_STRINGS_JSON
          — {"instruction_text": "...", "resume_text": "..."}

    No-ops if already configured or env vars are absent.
    """
    import json as _json
    import os as _os

    global _summary_config
    if _summary_config is not None and _summary_config.enabled:
        return
    if _os.environ.get("KV_EVICTION_MARKOVIAN_SUMMARY_ENABLED") != "1":
        return

    try:
        mode = _os.environ["KV_EVICTION_MARKOVIAN_SUMMARY_MODE"]
        compaction_max_turns = int(
            _os.environ["KV_EVICTION_MARKOVIAN_SUMMARY_COMPACTION_MAX_TURNS"]
        )
        max_len_summary = int(
            _os.environ["KV_EVICTION_MARKOVIAN_SUMMARY_MAX_LEN_SUMMARY"]
        )
    except (KeyError, ValueError) as e:
        logger.warning(
            "kv_eviction: KV_EVICTION_MARKOVIAN_SUMMARY_ENABLED=1 but "
            "required scalar env vars missing/invalid (%s); Markovian "
            "Summary NOT enabled in this process",
            e,
        )
        return

    temperature = float(
        _os.environ.get("KV_EVICTION_MARKOVIAN_SUMMARY_TEMPERATURE", "0.3")
    )
    top_p = float(_os.environ.get("KV_EVICTION_MARKOVIAN_SUMMARY_TOP_P", "0.95"))
    on_error = _os.environ.get("KV_EVICTION_MARKOVIAN_SUMMARY_ON_ERROR", "drop")
    log_summaries = (
        _os.environ.get("KV_EVICTION_MARKOVIAN_SUMMARY_LOG", "0") == "1"
    )

    strings_json = _os.environ.get("KV_EVICTION_MARKOVIAN_SUMMARY_STRINGS_JSON")
    instruction_text = ""
    resume_text = ""
    if strings_json:
        try:
            parsed = _json.loads(strings_json)
            instruction_text = str(parsed.get("instruction_text", ""))
            resume_text = str(parsed.get("resume_text", ""))
        except (ValueError, TypeError) as e:
            logger.warning(
                "kv_eviction: KV_EVICTION_MARKOVIAN_SUMMARY_STRINGS_JSON "
                "invalid (%s); using empty instruction_text/resume_text",
                e,
            )
    if not instruction_text:
        logger.warning(
            "kv_eviction: Markovian Summary enabled via env vars but "
            "instruction_text is empty; summaries will use an empty "
            "prompt and count_summary_exchanges will disable itself"
        )

    configure_markovian_summary(
        enabled=True,
        mode=mode,
        compaction_max_turns=compaction_max_turns,
        max_len_summary=max_len_summary,
        instruction_text=instruction_text,
        resume_text=resume_text,
        temperature=temperature,
        top_p=top_p,
        on_error=on_error,
        log_summaries=log_summaries,
    )


_autoconfigure_markovian_summary_from_env()


def padded_ids_from_step_extras(
    extras: dict[str, Any] | None,
) -> list[int] | None:
    """Read-side helper for orchestrator code: pull `prompt_token_ids`
    (the block-aligned padded token stream vLLM ran on) from a
    trajectory step's extras dict, returning None if absent.

    Used by prime-rl's `interleave_rollout` to thread padded ids onto
    `TrainingSample.prompt_token_ids`, so the trainer does not
    re-tokenize from `messages` (which would lose the padding).
    """
    if not extras:
        return None
    ids = extras.get("prompt_token_ids")
    if not ids:
        return None
    try:
        return [int(x) for x in ids]
    except (TypeError, ValueError):
        return None


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
                    evict_start=int(e.get("evict_start", 0)),
                )
            )
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            # array_like msgspec form: [n, tokens_evicted, position_offset_after, num_prompt_tokens, evict_start]
            out.append(
                CompactionEventWire(
                    num_output_tokens_at_compaction=int(e[0]),
                    tokens_evicted=int(e[1]),
                    position_offset_after=int(e[2]),
                    num_prompt_tokens=int(e[3]) if len(e) >= 4 else 0,
                    evict_start=int(e[4]) if len(e) >= 5 else 0,
                )
            )
    return out or None
