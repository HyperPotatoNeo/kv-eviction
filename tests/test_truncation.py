# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kv_eviction.truncation.truncate_messages_to_last_k_turns.

Covers the 8 invariants listed in `plans/markovian_thinker_baseline.md`:

1. Identity no-op when no truncation is needed.
2. Never mutates the input list or its dicts.
3. System prefix is always preserved byte-for-byte.
4. In-flight tail is always preserved byte-for-byte.
5. Turn groups (user → assistant[?tool_calls → tool → assistant]*) are
   atomic — no group is split.
6. Retained messages appear in original order.
7. Idempotence: truncate(truncate(M, K), K) == truncate(M, K).
8. Output is never longer than input.
"""

from copy import deepcopy

import pytest

from kv_eviction.truncation import truncate_messages_to_last_k_turns


# ─── Fixtures ───


def _sys(content="sys"):
    return {"role": "system", "content": content}


def _user(content="u"):
    return {"role": "user", "content": content}


def _assistant(content="a", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(content="t"):
    return {"role": "tool", "content": content}


def _turn(i):
    """One simple (user, assistant) pair tagged with index i."""
    return [_user(f"u{i}"), _assistant(f"a{i}")]


def _tool_turn(i):
    """A tool-call group: user → assistant(tc) → tool → assistant(final)."""
    return [
        _user(f"u{i}"),
        _assistant(content=None, tool_calls=[{"id": f"tc{i}", "function": {}}]),
        _tool(f"t{i}"),
        _assistant(f"a{i}"),
    ]


# ─── Identity / no-op tests ───


def test_empty_messages():
    assert truncate_messages_to_last_k_turns([], max_turns=4) == []


def test_all_system_returns_unchanged():
    msgs = [_sys("a"), _sys("b")]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=4)
    assert out is msgs


def test_no_user_yet_returns_unchanged():
    msgs = [_sys()]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=4)
    assert out is msgs


def test_only_system_and_user_returns_unchanged():
    msgs = [_sys(), _user("pending")]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=4)
    assert out is msgs


def test_single_complete_turn_noop():
    msgs = [_sys(), *_turn(1)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=1)
    assert out is msgs


def test_fewer_than_k_turns_noop_identity():
    msgs = [_sys(), *_turn(1), *_turn(2)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=4)
    assert out is msgs, "should return same object when no truncation needed"


def test_max_turns_equals_turns_identity():
    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=3)
    assert out is msgs


# ─── Actual truncation tests ───


def test_basic_truncation():
    """5 turns, max_turns=2 keeps sys + turn4 + turn5."""
    msgs = [_sys()] + [m for i in range(1, 6) for m in _turn(i)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=2)
    assert out[0] == _sys()
    assert [m["content"] for m in out[1:]] == ["u4", "a4", "u5", "a5"]


def test_max_turns_one_keeps_only_last():
    msgs = [_sys()] + [m for i in range(1, 5) for m in _turn(i)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=1)
    assert out[0] == _sys()
    assert [m["content"] for m in out[1:]] == ["u4", "a4"]


def test_no_system_message_still_works():
    """Without any leading non-user message, truncation must still drop old turns."""
    msgs = [m for i in range(1, 6) for m in _turn(i)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=2)
    assert [m["content"] for m in out] == ["u4", "a4", "u5", "a5"]


# ─── In-flight tail tests ───


def test_inflight_trailing_user_preserved():
    """Ends with a user message waiting for response — never drop it."""
    msgs = [_sys()] + [m for i in range(1, 5) for m in _turn(i)] + [_user("pending")]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=2)
    # Last three messages (2 complete turns + pending user) are kept.
    assert out[-1] == _user("pending")
    assert [m["content"] for m in out[1:]] == ["u3", "a3", "u4", "a4", "pending"]


def test_inflight_trailing_tool_preserved():
    """Ends mid-tool-call — the assistant(tc) + tool pair is the tail, kept."""
    msgs = (
        [_sys()]
        + [m for i in range(1, 4) for m in _turn(i)]
        + [
            _user("u4"),
            _assistant(content=None, tool_calls=[{"id": "x"}]),
            _tool("pending_tool"),
        ]
    )
    out = truncate_messages_to_last_k_turns(msgs, max_turns=1)
    # Keep sys + last complete turn (turn3) + the in-flight tail.
    assert out[-1]["role"] == "tool"
    assert out[-1]["content"] == "pending_tool"


# ─── Tool-call atomicity ───


def test_tool_call_group_atomicity_kept():
    """Group (user → assistant(tc) → tool → assistant) kept as one unit."""
    msgs = [_sys()] + _tool_turn(1) + _turn(2)
    out = truncate_messages_to_last_k_turns(msgs, max_turns=2)
    # No truncation — 2 groups <= max_turns=2.
    assert out is msgs


def test_tool_call_group_atomicity_dropped():
    """When dropped, drop the WHOLE tool chain, not a prefix of it."""
    msgs = [_sys()] + _tool_turn(1) + _turn(2) + _turn(3)
    out = truncate_messages_to_last_k_turns(msgs, max_turns=2)
    # tool_turn(1) is the oldest group, should be dropped whole.
    roles_and_contents = [(m["role"], m.get("content")) for m in out]
    assert roles_and_contents == [
        ("system", "sys"),
        ("user", "u2"),
        ("assistant", "a2"),
        ("user", "u3"),
        ("assistant", "a3"),
    ]


def test_multi_tool_chain_single_group():
    """Multiple tool calls within one group count as one turn group."""
    msgs = [
        _sys(),
        _user("u1"),
        _assistant(content=None, tool_calls=[{"id": "a"}]),
        _tool("t_a"),
        _assistant(content=None, tool_calls=[{"id": "b"}]),
        _tool("t_b"),
        _assistant("final1"),
        *_turn(2),
    ]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=1)
    # One group = the whole multi-tool chain; second group = turn(2).
    # max_turns=1 keeps only turn(2).
    assert [m.get("content") for m in out] == ["sys", "u2", "a2"]


# ─── Invariants ───


def test_no_mutation_of_input():
    msgs = [_sys()] + [m for i in range(1, 6) for m in _turn(i)]
    snapshot = deepcopy(msgs)
    truncate_messages_to_last_k_turns(msgs, max_turns=2)
    assert msgs == snapshot


def test_monotonic_ordering_preserved():
    msgs = [_sys()] + [m for i in range(1, 11) for m in _turn(i)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=3)
    # Order of retained turns must be [turn8, turn9, turn10].
    contents = [m["content"] for m in out if m["role"] == "user"]
    assert contents == ["u8", "u9", "u10"]


def test_idempotence():
    msgs = [_sys()] + [m for i in range(1, 11) for m in _turn(i)]
    once = truncate_messages_to_last_k_turns(msgs, max_turns=3)
    twice = truncate_messages_to_last_k_turns(once, max_turns=3)
    # Second call must be a no-op — turn count now equals max_turns.
    assert twice is once
    assert once == twice


def test_output_never_longer_than_input():
    msgs = [_sys()] + [m for i in range(1, 11) for m in _turn(i)]
    for k in range(1, 15):
        out = truncate_messages_to_last_k_turns(msgs, max_turns=k)
        assert len(out) <= len(msgs)


# ─── Logging callback ───


def test_log_fn_fires_on_truncation():
    msgs = [_sys()] + [m for i in range(1, 6) for m in _turn(i)]
    calls = []
    truncate_messages_to_last_k_turns(
        msgs, max_turns=2, log_fn=lambda s: calls.append(s)
    )
    assert len(calls) == 1
    assert "dropped 3 groups" in calls[0]


def test_log_fn_silent_on_noop():
    msgs = [_sys()] + _turn(1)
    calls = []
    truncate_messages_to_last_k_turns(
        msgs, max_turns=4, log_fn=lambda s: calls.append(s)
    )
    assert calls == []


# ─── Guard against invalid max_turns ───


@pytest.mark.parametrize("k", [0, -1, -100])
def test_max_turns_less_than_one_is_noop(k):
    msgs = [_sys()] + _turn(1)
    out = truncate_messages_to_last_k_turns(msgs, max_turns=k)
    assert out is msgs


# ─── Stride parameter (decoupled keep count) ───


def test_stride_none_matches_legacy_behavior():
    msgs = [_sys()] + [m for i in range(1, 6) for m in _turn(i)]
    legacy = truncate_messages_to_last_k_turns(msgs, max_turns=2)
    with_none = truncate_messages_to_last_k_turns(msgs, max_turns=2, stride=None)
    assert [m["content"] for m in with_none] == [m["content"] for m in legacy]


def test_stride_keeps_fewer_than_max_turns():
    """max_turns=4 (trigger), stride=2 (keep): 5 complete turns + pending user.
    Trigger fires (5 > 4), keeps last 2 groups instead of last 4."""
    msgs = (
        [_sys()]
        + [m for i in range(1, 6) for m in _turn(i)]
        + [_user("pending")]
    )
    out = truncate_messages_to_last_k_turns(msgs, max_turns=4, stride=2)
    assert [m["content"] for m in out] == [
        "sys", "u4", "a4", "u5", "a5", "pending",
    ]


def test_stride_equal_to_max_turns_matches_legacy():
    msgs = [_sys()] + [m for i in range(1, 6) for m in _turn(i)]
    legacy = truncate_messages_to_last_k_turns(msgs, max_turns=2)
    with_stride = truncate_messages_to_last_k_turns(msgs, max_turns=2, stride=2)
    assert [m["content"] for m in with_stride] == [m["content"] for m in legacy]


def test_stride_respects_trigger_threshold():
    """Even with stride=1, no truncation if n_groups <= max_turns."""
    msgs = [_sys()] + [m for i in range(1, 4) for m in _turn(i)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=4, stride=1)
    assert out is msgs


def test_stride_clamped_below_one():
    """stride=0 (or negative) is clamped up to 1 — at least one turn is kept."""
    msgs = [_sys()] + [m for i in range(1, 5) for m in _turn(i)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=3, stride=0)
    # max_turns=3 trigger fires (4 > 3). stride clamped to 1 → keep last turn.
    assert [m["content"] for m in out] == ["sys", "u4", "a4"]


def test_stride_clamped_above_max_turns():
    """stride > max_turns is clamped down — no nonsense keep-more-than-trigger."""
    msgs = [_sys()] + [m for i in range(1, 5) for m in _turn(i)]
    out = truncate_messages_to_last_k_turns(msgs, max_turns=2, stride=99)
    # Clamped to max_turns=2: keep last 2 turns.
    assert [m["content"] for m in out] == ["sys", "u3", "a3", "u4", "a4"]


def test_stride_preserves_tool_group_atomicity():
    msgs = [_sys()] + _tool_turn(1) + _turn(2) + _turn(3) + _turn(4)
    out = truncate_messages_to_last_k_turns(msgs, max_turns=3, stride=1)
    # Trigger fires (4 > 3), keep only last turn group.
    roles_and_contents = [(m["role"], m.get("content")) for m in out]
    assert roles_and_contents == [
        ("system", "sys"),
        ("user", "u4"),
        ("assistant", "a4"),
    ]
