"""Property-based tests for the two CLI event parsers — laws over ALL payloads.

WHY THIS FILE EXISTS
--------------------
These parsers sit behind a subprocess boundary. Their input is whatever the
binary happened to write: a version nobody has tested against, a field that
changed type between releases, a payload truncated by a full disk. Every
example-based test in this suite feeds them a payload somebody wrote by hand,
which means every one of them describes a stream we already know about.

The failures that actually cost this codebase were the other kind. From
``_cli_common``'s own docstring: one undecodable byte on stdout ended a whole
run and reported ``$0.00`` to the budget. From ``codex_cli``: a cached-token
count larger than the total would have produced a negative ``input_tokens``,
which flows into a meter and makes a budget go **up**. Neither is a payload
anyone would think to write down.

So the strategies below are deliberately hostile. They take the payload TYPES
the parsers branch on — the shapes that reach real code rather than the
unknown-type fallthrough — and fill every field with arbitrary JSON. A string
where a dict belongs, a list where a number belongs, a negative token count, a
NaN cost.

Each test is a law, not a scenario:

    parsing terminates            a crash here ends a run that was working
    every delta folds             a parser and a state that disagree lose data
    tokens are never negative     a negative charge makes a budget grow
    cost is finite                NaN into a ledger is worse than wrong
    text only grows               a fold that shortens text is silent loss
    saw_terminal is earned        set it wrongly and truncation stops being visible
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentkit.agents.cognition import claude_cli as cc
from agentkit.agents.cognition import codex_cli as cx
from agentkit.kernel.types import StreamEvent

# Any JSON value a binary could emit, nested. Bounded because the parsers are
# not recursive over depth and a deeper tree only slows the run down.
JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**40), max_value=2**40)
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.text(max_size=40),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=12), children, max_size=4),
    max_leaves=12,
)

# Field names the two parsers actually read. Generating payloads whose keys are
# drawn from this set is what makes the search hit real branches instead of the
# "unknown type — ignore" fallthrough, where nothing interesting can break.
_CLAUDE_FIELDS = st.sampled_from(
    [
        "message", "content", "subtype", "session_id", "duration_ms", "is_error",
        "usage", "total_cost_usd", "structured_output", "event", "delta", "text",
        "thinking", "tool_use_id", "name", "input", "model", "mcp_servers",
    ]
)
_CODEX_FIELDS = st.sampled_from(
    [
        "item", "id", "type", "text", "message", "usage", "error", "info",
        "thread_id", "session_id", "model", "duration_ms", "command",
        "aggregated_output", "exit_code", "changes", "query", "items",
        "total_token_usage", "last_agent_message", "call_id", "server", "tool",
    ]
)


def _payloads(ptypes: list[str], fields: st.SearchStrategy[str]) -> st.SearchStrategy[dict]:
    """A payload of a known type, with arbitrary values in the fields the
    parser reads — plus an occasional wholly random dict, because a binary can
    emit a shape nobody enumerated."""
    known = st.builds(
        lambda t, extra: {"type": t, **extra},
        st.sampled_from(ptypes),
        st.dictionaries(fields, JSON_VALUES, max_size=6),
    )
    arbitrary = st.dictionaries(st.text(max_size=12), JSON_VALUES, max_size=6)
    return st.one_of(known, known, known, arbitrary)  # weight toward real branches


CLAUDE_PAYLOADS = _payloads(
    ["assistant", "result", "stream_event", "system", "user"], _CLAUDE_FIELDS
)
CODEX_PAYLOADS = _payloads(
    [
        "thread.started", "turn.started", "turn.completed", "turn.failed", "error",
        "item.completed", "item.started", "item.updated",
    ],
    _CODEX_FIELDS,
)
# The legacy vocabulary is a second parser with its own branches, reached only
# through the ``{"id": ..., "msg": {...}}`` envelope.
LEGACY_PAYLOADS = st.builds(
    lambda i, msg: {"id": str(i), "msg": msg},
    st.integers(min_value=0, max_value=99),
    _payloads(
        [
            "session_configured", "task_started", "task_complete", "agent_message",
            "agent_message_delta", "agent_reasoning", "agent_reasoning_delta",
            "token_count", "error", "exec_command_begin", "exec_command_end",
            "mcp_tool_call_begin", "mcp_tool_call_end", "patch_apply_begin",
            "patch_apply_end", "turn_aborted",
        ],
        _CODEX_FIELDS,
    ),
)

# Deliberately does NOT set ``max_examples``. An explicit one on the decorator
# OVERRIDES the active profile, so hardcoding 250 here would have made
# ``--hypothesis-profile=deep`` silently do nothing — a deeper sweep that
# quietly runs at the shallow depth is worse than no deep profile at all.
# The count comes from the profile; see ``tests/conftest.py``.
_SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _claude_parse(payload: dict[str, Any], *, partial: bool = False) -> list[tuple]:
    async def go() -> list[tuple]:
        return [pair async for pair in cc._events_from_payload(payload, partial=partial)]

    return asyncio.run(go())


def _codex_parse(payload: dict[str, Any]) -> list[tuple]:
    return list(cx._events_from_payload(payload, cx._ItemLog()))


# ─────────────────────────────────────────────────────────────────────────────
# Law 1 — parsing terminates.
# ─────────────────────────────────────────────────────────────────────────────


@given(CLAUDE_PAYLOADS, st.booleans())
@_SETTINGS
def test_claude_parsing_never_raises(payload: dict[str, Any], partial: bool) -> None:
    """A payload the parser cannot make sense of must be ignored, never fatal.

    ``drive`` catches everything and reports it as ``parse_failed``, so a crash
    here does not take the process down — it ends a run that was working and
    loses the terminal payload, which is how a completed run once reported
    ``$0.00`` to the budget."""
    _claude_parse(payload, partial=partial)


@given(CODEX_PAYLOADS)
@_SETTINGS
def test_codex_parsing_never_raises(payload: dict[str, Any]) -> None:
    _codex_parse(payload)


@given(LEGACY_PAYLOADS)
@_SETTINGS
def test_codex_legacy_parsing_never_raises(payload: dict[str, Any]) -> None:
    """The legacy vocabulary is a whole second parser, and the one most likely
    to meet a shape nobody has run in years."""
    _codex_parse(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Law 2 — every delta folds into the state it was built for.
# ─────────────────────────────────────────────────────────────────────────────


@given(CLAUDE_PAYLOADS, st.booleans())
@_SETTINGS
def test_every_claude_delta_folds(payload: dict[str, Any], partial: bool) -> None:
    """A parser and a state that disagree about a field's type lose the run's
    data at the last moment, after the events have already been streamed."""
    state = cc._TurnState()
    for _event, delta in _claude_parse(payload, partial=partial):
        state.fold(delta)


@given(st.one_of(CODEX_PAYLOADS, LEGACY_PAYLOADS))
@_SETTINGS
def test_every_codex_delta_folds(payload: dict[str, Any]) -> None:
    state = cx._TurnState()
    for _event, delta in _codex_parse(payload):
        state.fold(delta)


# ─────────────────────────────────────────────────────────────────────────────
# Law 3 — usage never goes backwards.
# ─────────────────────────────────────────────────────────────────────────────


@given(CLAUDE_PAYLOADS, st.booleans())
@_SETTINGS
def test_claude_usage_is_never_negative(payload: dict[str, Any], partial: bool) -> None:
    """Every one of these is charged to a meter. A negative token count does
    not merely under-report — it makes a budget grow, so an exhausted run gets
    headroom back by failing in the right way."""
    for _event, delta in _claude_parse(payload, partial=partial):
        if delta.usage is None:
            continue
        assert delta.usage.input_tokens >= 0
        assert delta.usage.output_tokens >= 0
        assert delta.usage.cache_read_tokens >= 0
        assert delta.usage.cache_write_tokens >= 0


@given(st.one_of(CODEX_PAYLOADS, LEGACY_PAYLOADS))
@_SETTINGS
def test_codex_usage_is_never_negative(payload: dict[str, Any]) -> None:
    """Codex reports input tokens INCLUSIVE of the cached prefix and the cached
    count separately, so the two come from different places and a cached count
    larger than the total is exactly the input that used to produce a negative
    number. ``_split_input_tokens`` clamps; this is the law it clamps for."""
    for _event, delta in _codex_parse(payload):
        if delta.usage is None:
            continue
        assert delta.usage.input_tokens >= 0
        assert delta.usage.output_tokens >= 0
        assert delta.usage.cache_read_tokens >= 0
        assert delta.usage.cache_write_tokens >= 0


@given(st.integers(min_value=-(2**30), max_value=2**30), st.integers(min_value=-(2**30), max_value=2**30))
def test_split_input_tokens_is_total_over_every_pair(total: int, cached: int) -> None:
    """The clamp, stated directly and over every ordering — including the one
    that matters, ``cached > total``."""
    fresh, cache_read = cx._split_input_tokens(total, cached)
    assert fresh >= 0 and cache_read >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Law 4 — cost is a number a ledger can add up.
# ─────────────────────────────────────────────────────────────────────────────


@given(CLAUDE_PAYLOADS, st.booleans())
@_SETTINGS
def test_claude_cost_is_finite_and_non_negative(payload: dict[str, Any], partial: bool) -> None:
    """``total_cost_usd`` is read straight off the wire and goes into a
    ``Budget``. A NaN there is not a wrong number, it is a ledger that can no
    longer answer whether it is over."""
    for _event, delta in _claude_parse(payload, partial=partial):
        if delta.usage is None:
            continue
        cost = delta.usage.cost_usd
        assert not math.isnan(cost), "NaN cost would poison every comparison in the ledger"
        assert not math.isinf(cost)
        assert cost >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Law 5 — accumulated text only grows.
# ─────────────────────────────────────────────────────────────────────────────


@given(st.lists(CLAUDE_PAYLOADS, max_size=8), st.booleans())
@_SETTINGS
def test_claude_text_accumulation_is_monotone(payloads: list[dict], partial: bool) -> None:
    """``AgentResult.output`` is this string. A fold that shortened it would
    drop an answer the consumer had already been streamed."""
    state = cc._TurnState()
    seen = 0
    for payload in payloads:
        for _event, delta in _claude_parse(payload, partial=partial):
            state.fold(delta)
            assert len(state.text) >= seen
            seen = len(state.text)


@given(st.lists(st.one_of(CODEX_PAYLOADS, LEGACY_PAYLOADS), max_size=8))
@_SETTINGS
def test_codex_text_accumulation_is_monotone(payloads: list[dict]) -> None:
    state = cx._TurnState()
    log = cx._ItemLog()
    seen = 0
    for payload in payloads:
        for _event, delta in cx._events_from_payload(payload, log):
            state.fold(delta)
            assert len(state.text) >= seen
            seen = len(state.text)


# ─────────────────────────────────────────────────────────────────────────────
# Law 6 — saw_terminal has to be earned.
# ─────────────────────────────────────────────────────────────────────────────

_CLAUDE_TERMINAL = {"result"}
_CODEX_TERMINAL = {"turn.completed", "turn.failed", "error"}
_LEGACY_TERMINAL = {"task_complete", "error"}


@given(CLAUDE_PAYLOADS, st.booleans())
@_SETTINGS
def test_claude_only_the_result_payload_ends_a_turn(payload: dict, partial: bool) -> None:
    """The whole truncation check rests on this flag. A parser that set it on
    an ordinary payload would silently restore the bug it was added to fix —
    a half-written answer reported as complete."""
    for _event, delta in _claude_parse(payload, partial=partial):
        if delta.saw_terminal:
            assert payload.get("type") in _CLAUDE_TERMINAL


@given(CODEX_PAYLOADS)
@_SETTINGS
def test_codex_only_a_terminal_payload_ends_a_turn(payload: dict) -> None:
    for _event, delta in _codex_parse(payload):
        if delta.saw_terminal:
            assert payload.get("type") in _CODEX_TERMINAL


@given(LEGACY_PAYLOADS)
@_SETTINGS
def test_legacy_only_a_terminal_payload_ends_a_turn(payload: dict) -> None:
    for _event, delta in _codex_parse(payload):
        if delta.saw_terminal:
            msg = payload.get("msg") or {}
            assert msg.get("type") in _LEGACY_TERMINAL


# ─────────────────────────────────────────────────────────────────────────────
# Law 7 — an emitted event is well-formed.
# ─────────────────────────────────────────────────────────────────────────────


@given(CLAUDE_PAYLOADS, st.booleans())
@_SETTINGS
def test_claude_events_are_well_typed(payload: dict, partial: bool) -> None:
    """These reach the caller's ``async for``. A ``StreamEvent`` carrying a
    dict where a consumer expects a string is a crash in application code, at
    the point where the framework has already promised the event is valid."""
    for event, _delta in _claude_parse(payload, partial=partial):
        _assert_well_typed(event)


@given(st.one_of(CODEX_PAYLOADS, LEGACY_PAYLOADS))
@_SETTINGS
def test_codex_events_are_well_typed(payload: dict) -> None:
    for event, _delta in _codex_parse(payload):
        _assert_well_typed(event)


def _assert_well_typed(event: StreamEvent | None) -> None:
    if event is None:
        return
    assert isinstance(event.type, str) and event.type
    assert isinstance(event.text, str), f"text must be a str, got {type(event.text)}"
    if event.tool_call is not None:
        assert isinstance(event.tool_call.name, str)
        assert isinstance(event.tool_call.id, str)
        assert isinstance(event.tool_call.arguments, dict)
    if event.tool_result is not None:
        assert isinstance(event.tool_result, str)
