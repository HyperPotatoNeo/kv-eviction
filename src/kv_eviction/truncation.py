# SPDX-License-Identifier: Apache-2.0
"""
Client-side message truncation for the Markovian Thinker baseline.

Keeps the system prefix and the last K complete "turn groups" of a
conversation, dropping older turns. Used by the AsyncCompletions.create
interceptor in ``env.py`` when the orchestrator enables
``orchestrator.markovian_thinker``.

Contract (see ``plans/markovian_thinker_baseline.md``):

- **System prefix**: all leading messages before the first ``role == "user"``
  message. Always preserved.
- **Turn group**: the atomic unit that must be kept or dropped as a whole.
  A group ends at each ``role == "assistant"`` message without a
  ``tool_calls`` field. Messages between one terminal assistant and the
  next (exclusive on the prior terminal, inclusive on the next) form one
  group. This rule correctly handles simple chat, tool-call chains, and
  multi-tool sequences.
- **In-flight tail**: all trailing messages after the last terminal
  assistant. Protects the pending exchange (e.g., a trailing ``user``
  observation or a mid-tool ``assistant(tc), tool`` pair).

If fewer than or equal to ``max_turns`` complete groups exist, the input
is returned unchanged (same identity).
"""

from collections.abc import Callable


def truncate_messages_to_last_k_turns(
    messages: list[dict],
    *,
    max_turns: int,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict]:
    """Truncate a message list to at most ``max_turns`` recent turn groups.

    Returns the input unchanged (same identity) when no truncation is
    needed. Never mutates the input list or its dicts.

    Does not depend on a tokenizer — operates on message dicts by role.
    """
    if not messages or max_turns < 1:
        return messages

    # 1. System prefix = leading messages before the first user message.
    sys_end = 0
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            break
        sys_end = i + 1

    # 2. Last terminal assistant (no tool_calls) at position >= sys_end.
    last_terminal = -1
    for i in range(len(messages) - 1, sys_end - 1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            last_terminal = i
            break

    if last_terminal < sys_end:
        # No complete turn in the body — nothing to truncate.
        return messages

    body = messages[sys_end : last_terminal + 1]
    tail = messages[last_terminal + 1 :]

    # 3. Segment body into groups by terminal-assistant boundary.
    groups: list[list[dict]] = []
    group_start = 0
    for i, m in enumerate(body):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            groups.append(body[group_start : i + 1])
            group_start = i + 1

    if len(groups) <= max_turns:
        return messages

    dropped = groups[:-max_turns]
    kept = groups[-max_turns:]

    if log_fn is not None:
        n_dropped_msgs = sum(len(g) for g in dropped)
        first = dropped[0][0] if dropped and dropped[0] else None
        last = dropped[-1][-1] if dropped and dropped[-1] else None
        log_fn(
            f"dropped {len(dropped)} groups ({n_dropped_msgs} msgs); "
            f"first.role={first.get('role') if first else '?'}, "
            f"last.role={last.get('role') if last else '?'}"
        )

    result: list[dict] = list(messages[:sys_end])
    for g in kept:
        result.extend(g)
    result.extend(tail)
    return result
