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

from kv_eviction.summarization import (
    count_summary_exchanges,
    partition_messages,
)

__all__ = [
    "truncate_messages_to_last_k_turns",
    "partition_messages",
    "count_summary_exchanges",
]


def truncate_messages_to_last_k_turns(
    messages: list[dict],
    *,
    max_turns: int,
    stride: int | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict]:
    """Truncate a message list at the ``max_turns`` trigger.

    - ``max_turns``: trigger threshold. Truncation fires when the number
      of complete turn groups exceeds this.
    - ``stride``: number of recent turn groups to preserve after
      truncation. When ``None`` (default), keeps ``max_turns`` groups —
      the original single-knob behavior. When specified, must be in
      ``[1, max_turns]``; values outside that range are clamped.

    Returns the input unchanged (same identity) when no truncation is
    needed. Never mutates the input list or its dicts.

    Does not depend on a tokenizer — operates on message dicts by role.
    """
    if not messages or max_turns < 1:
        return messages

    n_groups, sys_prefix, groups, tail = partition_messages(messages)

    if n_groups == 0 or n_groups <= max_turns:
        return messages

    keep = max_turns if stride is None else stride
    # Clamp into a sane range: at least 1 (keeping zero turns is a
    # degenerate full-reset that Markovian Summary owns via its own
    # splice path), at most max_turns (stride > max_turns is nonsense
    # because we'd never drop anything on the trigger).
    if keep < 1:
        keep = 1
    if keep > max_turns:
        keep = max_turns

    dropped = groups[:-keep]
    kept = groups[-keep:]

    if log_fn is not None:
        n_dropped_msgs = sum(len(g) for g in dropped)
        first = dropped[0][0] if dropped and dropped[0] else None
        last = dropped[-1][-1] if dropped and dropped[-1] else None
        log_fn(
            f"dropped {len(dropped)} groups ({n_dropped_msgs} msgs); "
            f"first.role={first.get('role') if first else '?'}, "
            f"last.role={last.get('role') if last else '?'}"
        )

    result: list[dict] = list(sys_prefix)
    for g in kept:
        result.extend(g)
    result.extend(tail)
    return result
