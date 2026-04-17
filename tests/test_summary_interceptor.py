# SPDX-License-Identifier: Apache-2.0
"""Interceptor tests for the Markovian Summary Branch-A extension in
``kv_eviction.env._install_message_padding_interceptor``.

Uses a stub ``orig_create`` (AsyncMock) plus a stub tokenizer to
exercise both summary modes (``markovian``, ``eviction``), the
recursion guard, re-fire prevention, error paths, and stats counters
without touching vLLM or a real tokenizer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import kv_eviction.env as env


INSTR = "Please summarize everything important for the task."


class _StubTok:
    """Minimal tokenizer stub: apply_chat_template returns a stable
    marker string, encode returns a deterministic token list. The
    actual values don't matter for interceptor logic — we just need
    the re-tokenize step to succeed."""

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        return f"TOK:{len(messages)}"

    def encode(self, s, add_special_tokens=False):
        return [1, 2, 3]


def _make_summary_response(
    text="SUMMARY",
    *,
    prompt_ids=None,
    completion_ids=None,
    logprobs=None,
):
    """Build a ChatCompletion-ish SimpleNamespace that
    ``_generate_summary`` can extract a summary + train-sample payload
    from. ``logprobs`` is a list of floats; ``None`` means no logprobs
    at all."""
    msg = SimpleNamespace(content=text)
    lp = None
    if logprobs is not None:
        lp = SimpleNamespace(
            content=[SimpleNamespace(token=f"t{i}", logprob=x) for i, x in enumerate(logprobs)]
        )
    choice = SimpleNamespace(message=msg, token_ids=completion_ids, logprobs=lp)
    return SimpleNamespace(
        choices=[choice],
        prompt_token_ids=prompt_ids,
    )


def _outer_response():
    return SimpleNamespace(id="outer-0", prompt_token_ids=None)


def _sys(content="sys"):
    return {"role": "system", "content": content}


def _user(content="u"):
    return {"role": "user", "content": content}


def _asst(content="a"):
    return {"role": "assistant", "content": content}


def _turn(i):
    return [_user(f"u{i}"), _asst(f"a{i}")]


@pytest.fixture(autouse=True)
def reset_configs():
    """Clear all interceptor state between tests."""
    env._markovian_config = None
    env._summary_config = None
    env._padding_config = None
    env._markovian_stats = {
        "n_truncations": 0,
        "n_messages_dropped": 0,
        "n_summaries": 0,
        "n_summary_failures": 0,
        "summary_prompt_tokens": 0,
        "summary_output_tokens": 0,
        "summary_latency_ms": 0,
    }
    yield
    env._markovian_config = None
    env._summary_config = None
    env._padding_config = None


def _install_mt(max_turns=8):
    env.configure_markovian_thinker(
        enabled=True, tokenizer=_StubTok(), max_turns=max_turns
    )


def _install_summary(
    *,
    enabled=True,
    mode="markovian",
    compaction_max_turns=2,
    max_len_summary=128,
    on_error="drop",
    instruction_text=INSTR,
    resume_text="",
):
    env.configure_markovian_summary(
        enabled=enabled,
        mode=mode,
        compaction_max_turns=compaction_max_turns,
        max_len_summary=max_len_summary,
        instruction_text=instruction_text,
        resume_text=resume_text,
        temperature=0.3,
        top_p=0.95,
        on_error=on_error,
        log_summaries=False,
    )


def _run_with_fake_orig(fake_orig, run_coro_factory):
    """Reinstall the interceptor on top of ``fake_orig`` then run
    ``run_coro_factory()`` once through ``asyncio.run``. Returns
    whatever the coroutine returned."""
    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create

        async def run():
            return await run_coro_factory(patched)

        return asyncio.run(run())


# ─── Trigger / mode branching ───


def test_markovian_mode_full_reset_shape():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if "logprobs" in kwargs and kwargs["logprobs"] is True:
            return _make_summary_response("SUMMARY-TEXT")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # 2 calls: summary request, then outer rewritten request.
    assert len(calls) == 2
    summary_call, outer_call = calls
    assert summary_call["logprobs"] is True
    assert summary_call["max_tokens"] == 128
    # Summary messages = full history + instruction.
    assert summary_call["messages"][-1] == {"role": "user", "content": INSTR}
    assert summary_call["messages"][:-1] == msgs
    assert "tools" not in summary_call
    assert "tool_choice" not in summary_call

    # Outer call: markovian mode = sys + tail + [I, S] (body dropped, no
    # resume_text configured → no trailing user turn).
    outer_messages = outer_call["messages"]
    assert outer_messages == [
        _sys(),
        _user("pending"),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY-TEXT"},
    ]


def test_markovian_mode_appends_resume_text_when_set():
    """When resume_text is configured, the outer call gets a trailing
    user turn after the summary exchange so vLLM has a pending
    generation prompt."""
    _install_mt(max_turns=8)
    _install_summary(
        mode="markovian",
        compaction_max_turns=2,
        resume_text="Please continue.",
    )

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if "logprobs" in kwargs and kwargs["logprobs"] is True:
            return _make_summary_response("SUM")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    outer_messages = calls[1]["messages"]
    assert outer_messages == [
        _sys(),
        _user("pending"),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUM"},
        {"role": "user", "content": "Please continue."},
    ]


def test_eviction_mode_append_only_shape():
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if "logprobs" in kwargs and kwargs["logprobs"] is True:
            return _make_summary_response("S-EVICT")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    assert len(calls) == 2
    outer_messages = calls[1]["messages"]
    # Eviction mode keeps body groups intact: sys + body + I + S + tail.
    assert outer_messages == [
        _sys(),
        *_turn(1),
        *_turn(2),
        *_turn(3),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "S-EVICT"},
        _user("pending"),
    ]


def test_below_trigger_no_summary_fires():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=4)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return _outer_response()

    # 2 real turns, below the 4-turn trigger.
    msgs = [_sys(), *_turn(1), *_turn(2), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # Only one call (outer). No summary fired.
    assert len(calls) == 1
    # Message list unchanged from input (Markovian's plain-truncation
    # below max_turns=8 is a no-op).
    assert calls[0]["messages"] == msgs


def test_eviction_mode_refire_prevention_after_single_summary():
    """After the summary fires, the next step's message list contains
    a prior summary exchange + 1 new real turn. With the discount, the
    turn count stays ≤ max_turns so the trigger does NOT re-fire.
    Without the discount it would fire every single subsequent step."""
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return _outer_response()

    # 1 real turn + 1 summary exchange + 1 new real turn = 3 groups.
    # n_real = 3 - 1 = 2. 2 > 2 is False → does NOT fire.
    msgs = [
        _sys(),
        *_turn(1),
        _user(INSTR),
        _asst("old-summary"),
        *_turn(2),
        _user("pending"),
    ]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # Exactly one (outer) call — no summary re-fired.
    assert len(calls) == 1


def test_eviction_mode_refires_after_enough_new_real_turns():
    """After enough additional real turns beyond the prior summary, the
    trigger re-fires."""
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("logprobs") is True:
            return _make_summary_response("S2")
        return _outer_response()

    # 1 real turn + 1 summary + 3 new real turns = 5 groups.
    # n_real = 5 - 1 = 4 > 2 → re-fires.
    msgs = [
        _sys(),
        *_turn(1),
        _user(INSTR),
        _asst("s1"),
        *_turn(2),
        *_turn(3),
        *_turn(4),
        _user("pending"),
    ]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)
    assert len(calls) == 2


# ─── Recursion guard ───


def test_recursion_guard_bypasses_interceptor():
    """When _IN_SUMMARY_CALL is True at entry, patched_create must not
    touch messages at all — it just forwards to orig_create."""
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        token = env._IN_SUMMARY_CALL.set(True)
        try:
            return await patched(self=None, model="m", messages=msgs)
        finally:
            env._IN_SUMMARY_CALL.reset(token)

    _run_with_fake_orig(fake_orig, factory)

    # Only one call and the messages are the raw input — no retokenize
    # or rewrite happened.
    assert len(calls) == 1
    assert calls[0]["messages"] == msgs
    # No summary triggered either.
    assert env._markovian_stats["n_summaries"] == 0


# ─── Error paths ───


def test_on_error_drop_falls_back_to_plain_truncation():
    _install_mt(max_turns=2)
    _install_summary(mode="markovian", compaction_max_turns=2, on_error="drop")

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        if kwargs.get("logprobs") is True:
            raise RuntimeError("summary backend down")
        calls.append(dict(kwargs))
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # One outer call (summary failed). Messages are plain-truncated
    # (max_turns=2 → keep last 2 groups + tail).
    assert len(calls) == 1
    outer = calls[0]["messages"]
    # No [I, S] injected.
    assert {"role": "user", "content": INSTR} not in outer
    # Should contain last 2 turns (u2/a2 and u3/a3) + pending.
    assert outer[-1] == _user("pending")
    assert env._markovian_stats["n_summary_failures"] == 1
    assert env._markovian_stats["n_summaries"] == 0


def test_on_error_raise_propagates():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2, on_error="raise")

    async def fake_orig(self, *args, **kwargs):
        if kwargs.get("logprobs") is True:
            raise RuntimeError("boom")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    with pytest.raises(RuntimeError, match="boom"):
        _run_with_fake_orig(fake_orig, factory)


def test_empty_summary_text_treated_as_failure():
    _install_mt(max_turns=2)
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        if kwargs.get("logprobs") is True:
            return _make_summary_response("   ")  # whitespace-only
        calls.append(dict(kwargs))
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # Falls back to plain truncation.
    assert len(calls) == 1
    outer = calls[0]["messages"]
    assert {"role": "user", "content": INSTR} not in outer
    assert env._markovian_stats["n_summary_failures"] == 1
    assert env._markovian_stats["n_summaries"] == 0


# ─── Stats counters ───


def test_stats_counters_increment_on_summary_success():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if kwargs.get("logprobs") is True:
            return _make_summary_response("SUMMARY")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    assert env._markovian_stats["n_summaries"] == 1
    assert env._markovian_stats["n_summary_failures"] == 0
    # Summary path rewrote messages, so the truncation counter fires too.
    assert env._markovian_stats["n_truncations"] == 1


def test_stats_drain_and_reset():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if kwargs.get("logprobs") is True:
            return _make_summary_response("SUMMARY")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)
    drained = env.pop_markovian_stats()
    assert drained["n_summaries"] == 1
    # Counters reset after drain.
    assert env._markovian_stats["n_summaries"] == 0
    assert env._markovian_stats["n_summary_failures"] == 0


# ─── Summary-call kwargs hygiene ───


def test_summary_trainsample_attached_to_outer_response():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if kwargs.get("logprobs") is True:
            return _make_summary_response(
                "S",
                prompt_ids=[11, 12, 13],
                completion_ids=[21, 22],
                logprobs=[-0.1, -0.2],
            )
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="test-model", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    attached = captured["response"].summary_trainsample
    assert attached["prompt_token_ids"] == [11, 12, 13]
    assert attached["completion_token_ids"] == [21, 22]
    assert attached["completion_logprobs"] == pytest.approx([-0.1, -0.2])
    assert attached["model"] == "test-model"


def test_no_summary_no_trainsample_attached():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=8)  # below trigger

    async def fake_orig(self, *args, **kwargs):
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="m", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    assert getattr(captured["response"], "summary_trainsample", None) is None


def test_summary_failure_does_not_attach_trainsample():
    _install_mt(max_turns=2)
    _install_summary(mode="markovian", compaction_max_turns=2, on_error="drop")

    async def fake_orig(self, *args, **kwargs):
        if kwargs.get("logprobs") is True:
            raise RuntimeError("summary down")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="m", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    assert getattr(captured["response"], "summary_trainsample", None) is None


def test_summary_call_strips_tools_and_response_format():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("logprobs") is True:
            return _make_summary_response("S")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(
            self=None,
            model="m",
            messages=msgs,
            tools=[{"type": "function", "function": {"name": "foo"}}],
            tool_choice="auto",
            response_format={"type": "json_object"},
            extra_body={"something": True},
        )

    _run_with_fake_orig(fake_orig, factory)

    # calls[0] = summary; calls[1] = outer
    summary_call = calls[0]
    assert summary_call["logprobs"] is True
    # These must not leak into the summary request.
    assert "tools" not in summary_call
    assert "tool_choice" not in summary_call
    assert "response_format" not in summary_call
    assert "extra_body" not in summary_call


# ─── _extract_summary_trainsample / attach_summary_trainsample_from_response ───


def test_extract_summary_trainsample_attribute():
    sample = {"prompt_token_ids": [1], "completion_token_ids": [2]}
    resp = SimpleNamespace(summary_trainsample=sample)
    assert env._extract_summary_trainsample(resp) == sample


def test_extract_summary_trainsample_none_response():
    assert env._extract_summary_trainsample(None) is None


def test_extract_summary_trainsample_absent_field():
    resp = SimpleNamespace()
    assert env._extract_summary_trainsample(resp) is None


def test_extract_summary_trainsample_non_dict_ignored():
    resp = SimpleNamespace(summary_trainsample="not a dict")
    assert env._extract_summary_trainsample(resp) is None


def test_attach_summary_trainsample_from_response_roundtrip():
    sample = {"prompt_token_ids": [1, 2], "completion_token_ids": [3]}
    resp = SimpleNamespace(summary_trainsample=sample)
    step: dict = {"extras": None}
    env.attach_summary_trainsample_from_response(step, resp)  # type: ignore[arg-type]
    assert step["extras"]["summary_trainsample"] == sample


def test_attach_summary_trainsample_noop_when_absent():
    resp = SimpleNamespace()
    step: dict = {"extras": None}
    env.attach_summary_trainsample_from_response(step, resp)  # type: ignore[arg-type]
    assert step["extras"] is None
