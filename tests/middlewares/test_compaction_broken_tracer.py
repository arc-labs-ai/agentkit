"""A misbehaving tracer must never break compaction.

`compaction.py`'s module docstring promises "Both side-effects are best-effort
— a misbehaving tracer can never break the run", but the `context.compact` span
was opened, entered and exited RAW. The individual attribute writes inside were
each wrapped in `contextlib.suppress`, so the only unguarded parts were exactly
the three the tracer controls.

Measured before the fix, on both halves of the context-manager protocol:
`RUN BROKEN BY TRACER: RuntimeError: boom on span open: 'context.compact'` and
`RUN BROKEN BY TRACER: RuntimeError: boom on span close`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from agentkit.kernel.types import ChatRequest, Delta, Message, Scope, Usage
from agentkit.middlewares import compaction
from agentkit.runtime import Budget, Invoker, RunContext, Services


class _Span:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, **fields: Any) -> None:
        return None


class _OpenBoomTrace:
    """Raises on span OPEN."""

    def __init__(self) -> None:
        self.span_calls = 0
        self.events: list[str] = []

    def span(self, name: str, kind: str, **attrs: Any) -> Any:
        self.span_calls += 1
        raise RuntimeError(f"boom on span open: {name!r}")

    def current_span_id(self) -> None:
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        self.events.append(name)


class _ExitBoomTrace:
    """Opens fine, raises on span EXIT — the half a naive `try/except` around
    `trace.span(...)` would still miss."""

    def __init__(self) -> None:
        self.span_calls = 0
        self.spans: list[_Span] = []
        self.events: list[str] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str, **attrs: Any):
        self.span_calls += 1
        s = _Span(name)
        s.attrs.update(attrs)
        self.spans.append(s)
        yield s
        raise RuntimeError("boom on span close")

    def current_span_id(self) -> None:
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        self.events.append(name)


class _HealthyTrace(_ExitBoomTrace):
    @contextlib.contextmanager
    def span(self, name: str, kind: str, **attrs: Any):
        self.span_calls += 1
        s = _Span(name)
        s.attrs.update(attrs)
        self.spans.append(s)
        yield s


class _EventBoomTrace(_HealthyTrace):
    """The span works; `add_event_to_current_span` is what raises."""

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        raise RuntimeError("boom on event")


class _OkLLM:
    async def stream(self, **_kw: Any):
        yield Delta(text="answer", model="m", provider="p")
        yield Delta(usage=Usage(3, 1, 0.0), finish_reason="stop", model="m", provider="p")


class _DropOldestTurn:
    """A real reduction, so `messages_after` genuinely differs."""

    async def compact(self, messages: list[Any], run: Any) -> list[Any]:
        return [messages[0], *messages[2:]] if len(messages) > 2 else messages


class _EmptyCompactor:
    async def compact(self, messages: list[Any], run: Any) -> list[Any]:
        return []


class _BoomCompactor:
    async def compact(self, messages: list[Any], run: Any) -> list[Any]:
        raise ValueError("summarizer model is down")


REQ = ChatRequest(
    [Message("system", "you are helpful"), Message("user", "old"), Message("user", "new")], "m"
)


def _run(compactor: Any, trace: Any) -> Any:
    inv = Invoker(llm=_OkLLM(), chat_middleware=[compaction(compactor)])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, trace=trace))
    return asyncio.run(inv.chat(REQ, ctx))


def test_a_tracer_that_raises_on_span_OPEN_does_not_break_the_run() -> None:
    trace = _OpenBoomTrace()
    assert _run(_DropOldestTurn(), trace).content == "answer"
    assert trace.span_calls == 1, "the tracer was never even consulted"


def test_a_tracer_that_raises_on_span_EXIT_does_not_break_the_run() -> None:
    trace = _ExitBoomTrace()
    assert _run(_DropOldestTurn(), trace).content == "answer"
    assert trace.span_calls == 1


def test_a_tracer_that_raises_on_the_narrative_EVENT_does_not_break_the_run() -> None:
    """The `context.compacted` event on the surrounding span was already
    suppressed; pinned here so the guard can't be dropped along with the span
    rewrite."""
    assert _run(_DropOldestTurn(), _EventBoomTrace()).content == "answer"


def test_the_rewrite_still_happens_under_a_broken_tracer() -> None:
    """Surviving is not enough — the compaction itself must still apply, or the
    guard has quietly disabled the feature it protects."""
    seen: dict[str, Any] = {}

    class _Capturing:
        async def stream(self, **kw: Any):
            seen["messages"] = list(kw["messages"])
            yield Delta(text="answer", usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="p")

    inv = Invoker(llm=_Capturing(), chat_middleware=[compaction(_DropOldestTurn())])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, trace=_OpenBoomTrace()))
    asyncio.run(inv.chat(REQ, ctx))

    assert [m.content for m in seen["messages"]] == ["you are helpful", "new"]


def test_the_empty_result_fallback_survives_a_broken_tracer() -> None:
    """The defensive fallback lived inside the traced branch; with the span
    guarded it must still refuse to install a zero-message request."""
    seen: dict[str, Any] = {}

    class _Capturing:
        async def stream(self, **kw: Any):
            seen["messages"] = list(kw["messages"])
            yield Delta(text="answer", usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="p")

    inv = Invoker(llm=_Capturing(), chat_middleware=[compaction(_EmptyCompactor())])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, trace=_OpenBoomTrace()))
    asyncio.run(inv.chat(REQ, ctx))

    assert len(seen["messages"]) == 3, "an empty compaction was installed"


def test_a_compactor_that_raises_still_propagates() -> None:
    """The guard is for the TRACER, not for the compactor. A broken summarizer
    is a real failure and must not be swallowed by the span wrapper."""
    with pytest.raises(ValueError, match="summarizer model is down"):
        _run(_BoomCompactor(), _HealthyTrace())


def test_a_healthy_tracer_still_gets_the_compaction_span() -> None:
    """POSITIVE CONTROL. Deleting the span would make every "does not break the
    run" test above pass; it fails here. The `context.compact` span, its
    before/after attributes and the narrative `context.compacted` event all
    survive the rewrite."""
    trace = _HealthyTrace()
    assert _run(_DropOldestTurn(), trace).content == "answer"

    (span,) = trace.spans
    assert span.name == "context.compact"
    assert span.attrs["agentkit.context.strategy"] == "_DropOldestTurn"
    assert span.attrs["agentkit.context.messages_before"] == 3
    assert span.attrs["agentkit.context.messages_after"] == 2
    assert span.attrs["agentkit.context.messages_dropped"] == 1
    assert "context.compacted" in trace.events


def test_a_rejected_compaction_is_still_stamped_on_a_healthy_span() -> None:
    trace = _HealthyTrace()
    _run(_EmptyCompactor(), trace)

    (span,) = trace.spans
    assert span.attrs["agentkit.context.compaction_rejected"] == "empty_result"


def test_compaction_works_with_no_tracer_at_all() -> None:
    """`getattr(run, "trace", None)` can be None on a lean context; that path
    must stay silent, not fall into the broken-tracer warning path."""

    seen: dict[str, Any] = {}

    class _Capturing:
        async def stream(self, **kw: Any):
            seen["messages"] = list(kw["messages"])
            yield Delta(text="answer", usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="p")

    inv = Invoker(llm=_Capturing(), chat_middleware=[compaction(_DropOldestTurn())])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, trace=None))
    asyncio.run(inv.chat(REQ, ctx))

    assert [m.content for m in seen["messages"]] == ["you are helpful", "new"]
