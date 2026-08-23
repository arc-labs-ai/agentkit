"""Pluggable token counters — ApproxTokenCounter (default) + TiktokenCounter (opt-in).

These tests pin:

  1. ``ApproxTokenCounter`` on an empty list is exactly 0 — a zero-cost
     starting point that won't surprise budget pre-checks.
  2. ``ApproxTokenCounter`` matches chars/4 (rounded down) for English
     content PLUS a per-wire-block structural overhead — the rest of
     agentkit's _approx_tokens delegates to the same function, so a
     pre-check stays consistent with the compactor's own view.
  2b. Tool calls are counted. They used to be ignored entirely, which is
     how an 80-message transcript carrying 324,420 chars of tool
     arguments estimated as ~20 tokens and sailed past every compactor.
  3. ``TiktokenCounter`` soft-falls-back to the approximation when
     ``tiktoken`` is absent — no ImportError surfaces to the caller.
     The fallback path is forced via ``monkeypatch`` so the test
     runs whether or not the optional dep happens to be installed.
"""

from __future__ import annotations

import asyncio
import builtins

import pytest

from agentkit.context import ApproxTokenCounter, TiktokenCounter
from agentkit.kernel.types import Message, ToolCall


def _run(coro):
    return asyncio.run(coro)


def test_approx_counter_empty_list_is_zero():
    counter = ApproxTokenCounter()
    assert _run(counter.estimate([])) == 0


def test_approx_counter_matches_chars_over_four_plus_per_message_overhead():
    """chars/4 for the text, PLUS the 4-token structural cost of each wire block.

    The bare chars/4 number used to be the whole answer, which made 500
    empty messages estimate as 0 tokens; every provider charges ~3-4
    tokens per message for role markers and delimiters.
    """
    counter = ApproxTokenCounter(chars_per_token=4.0)
    msgs = [Message("system", "abcd"), Message("user", "efghij")]  # 4 + 6 = 10 chars
    assert _run(counter.estimate(msgs)) == 10 // 4 + 2 * 4


def test_approx_counter_invalid_chars_per_token_raises():
    counter = ApproxTokenCounter(chars_per_token=0.0)
    with pytest.raises(ValueError):
        _run(counter.estimate([Message("user", "anything")]))


def test_tiktoken_counter_soft_falls_back_when_tiktoken_absent(monkeypatch):
    """Force the import-failure branch by blocking ``import tiktoken``.

    The counter must NOT raise; it should silently degrade to the
    chars/4 approximation so a borderline budget check stays sane.
    """
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    counter = TiktokenCounter()
    msgs = [Message("user", "abcd")]
    assert _run(counter.estimate(msgs)) == _run(ApproxTokenCounter().estimate(msgs)) == 1 + 4


# ---- tool-call accounting (the "compaction never fires" bug) --------------


def _tool_turn(i: int, arg_chars: int) -> list[Message]:
    """One assistant tool-call turn + its tool-result turn — the normal
    agentic shape, and the shape both estimators used to score as ~0."""
    return [
        Message(
            "assistant",
            "",
            tool_calls=(ToolCall(id=f"c{i}", name="search", arguments={"q": "x" * arg_chars}),),
        ),
        Message("tool", "", name="search", tool_call_id=f"c{i}"),
    ]


def test_approx_counter_counts_tool_call_arguments():
    """REGRESSION: tool_calls were not counted at all.

    Measured before the fix: 80 messages / 324,420 chars of tool
    arguments estimated as 20 tokens. The arguments are what the
    provider actually bills, so they have to dominate the estimate.
    """
    msgs = [m for i in range(40) for m in _tool_turn(i, 4000)]
    est = _run(ApproxTokenCounter().estimate(msgs))
    assert est > 39_000, f"tool arguments still under-counted: {est}"


def test_approx_counter_and_compaction_estimator_agree():
    """The two estimators MUST return the same number.

    A caller that pre-checks its budget with ``ApproxTokenCounter`` and
    then compacts with a ``Compactor`` was comparing two independent
    chars/4 copies; they drifted into the same tool-blind bug, and only
    one of them was ever fixed at a time. They now share one function.
    """
    from agentkit.capabilities.compaction.base import _approx_tokens

    msgs = [
        Message("system", "sys"),
        *[m for i in range(3) for m in _tool_turn(i, 500)],
        Message("assistant", "done"),
    ]
    assert _approx_tokens(msgs) == _run(ApproxTokenCounter().estimate(msgs))


def test_tool_results_are_counted_once_not_twice():
    """A tool RESULT is plain ``Message.content`` on a role="tool" turn.

    It was already counted before the fix, so adding tool_calls must not
    double-charge it: the estimate for a result-only transcript stays
    chars/4 + the per-block overhead.
    """
    msgs = [Message("tool", "r" * 400, name="n", tool_call_id="id")]
    expected = (400 + len("n") + len("id")) // 4 + 4
    assert _run(ApproxTokenCounter().estimate(msgs)) == expected


def test_empty_transcript_is_exactly_zero():
    """No messages = no tokens. The per-block overhead must not leak a
    non-zero floor into a budget pre-check on a fresh context."""
    from agentkit.capabilities.compaction.base import _approx_tokens

    assert _run(ApproxTokenCounter().estimate([])) == 0
    assert _approx_tokens([]) == 0


def test_empty_messages_still_cost_structural_tokens():
    """500 empty messages estimated as 0 before the fix — they are ~2k
    tokens of role markers and delimiters on the wire."""
    assert _run(ApproxTokenCounter().estimate([Message("user", "")] * 500)) == 500 * 4


def test_short_chat_is_not_inflated_into_needless_compaction():
    """POSITIVE CONTROL — the fix must not over-count.

    A normal short chat has to stay far below a default 12,000-token
    budget, otherwise "compaction never fires" would just become
    "compaction always fires".
    """
    msgs = [
        Message("system", "You are a helpful assistant."),
        Message("user", "What is the capital of France?"),
        Message("assistant", "Paris."),
    ]
    est = _run(ApproxTokenCounter().estimate(msgs))
    assert est < 50
    assert est == sum(len(m.content) for m in msgs) // 4 + 3 * 4


def test_unserialisable_tool_arguments_do_not_raise():
    """A budget pre-check must never explode on an exotic payload.

    ``ToolCall.arguments`` is a ``MappingProxyType`` (``json.dumps``
    rejects it outright) and can hold non-JSON values; the estimator
    falls back rather than raising from inside the pre-check.
    """
    msgs = [
        Message("assistant", "", tool_calls=(ToolCall(id="1", name="t", arguments={"o": object()}),))
    ]
    assert _run(ApproxTokenCounter().estimate(msgs)) > 4


def test_chars_per_token_ratio_is_honoured_for_tool_calls():
    """A custom ratio applies to the tool-call text too — the overhead
    term is a token count and stays outside the division."""
    msgs = [m for i in range(2) for m in _tool_turn(i, 200)]
    coarse = _run(ApproxTokenCounter(chars_per_token=8.0).estimate(msgs))
    fine = _run(ApproxTokenCounter(chars_per_token=4.0).estimate(msgs))
    assert coarse < fine


# ── every estimator in the framework must be the SAME estimator ────────────


def test_all_four_token_estimators_agree() -> None:
    """There were FOUR copies of `sum(len(content)) // 4`, and two of them
    carried docstrings claiming they matched the others. They did, until the
    shared one learned to count tool calls — and then they silently did not.

    That divergence is the whole bug: a caller pre-checks a budget with the
    request-builder's number, the compaction middleware decides whether to
    compact with a second, and the compactor picks messages with a third. One
    transcript, three answers, and the one that decides is the one that was
    wrong (measured: ~81,000 real tokens estimated at 20).

    This test is the reason a fifth copy cannot appear quietly.
    """
    from agentkit.capabilities.compaction.base import _approx_tokens as compactor
    from agentkit.capabilities.request_builder.base import _approx_tokens as builder
    from agentkit.context.tokens import estimate_message_tokens as shared
    from agentkit.middlewares.compaction import _approx_tokens as middleware

    transcripts = {
        "tool-heavy": [
            Message("assistant", "", tool_calls=(ToolCall("i1", "write", {"body": "x" * 4000}),)),
            Message("tool", "ok", tool_call_id="i1", name="write"),
        ],
        "plain chat": [Message("user", "hello there"), Message("assistant", "hi")],
        "empty": [],
        "many empties": [Message("user", "") for _ in range(50)],
    }

    for label, messages in transcripts.items():
        answers = {
            "shared": shared(messages),
            "compactor": compactor(messages),
            "builder": builder(messages),
            "middleware": middleware(messages),
        }
        assert len(set(answers.values())) == 1, f"{label}: estimators disagree — {answers}"


def test_the_shared_estimator_actually_counts_tool_calls() -> None:
    """POSITIVE CONTROL for the test above: four estimators agreeing on ZERO
    would satisfy it. This pins that the agreed number is the right one."""
    from agentkit.capabilities.request_builder.base import _approx_tokens as builder

    heavy = [Message("assistant", "", tool_calls=(ToolCall("i", "w", {"c": "x" * 4000}),))]
    assert builder(heavy) > 900, "a 4000-char tool argument must not estimate as ~0"
    assert builder([Message("user", "hi")]) < 20, "...and a two-word chat must stay small"
