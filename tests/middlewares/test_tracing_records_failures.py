"""A failed call must be exported as a FAILED span.

`_safe_trace_span` closed its `ExitStack` with `close()`, which is
`__exit__(None, None, None)` — it told the span context manager the block
succeeded no matter what actually happened. `adapters/observability/otel.py`
opens `start_as_current_span` and relies on OTel's own `__exit__` for
`record_exception` + the ERROR status, so every provider failure was exported
as a successful span.

Measured before the fix: the provider raised `RuntimeError("provider 500")` and
the trace showed `spans opened: ['chat'], exception recorded: [None]`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from agentkit.kernel.types import ChatRequest, Delta, Message, Scope, ToolRequest, Usage
from agentkit.middlewares import tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services


class _Span:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict[str, Any] = {}
        self.exception: BaseException | None = None
        self.status: str | None = None

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, **fields: Any) -> None:
        return None


class _RecordingTrace:
    """A `TracePort` whose span CM does what a real OTel span does on exit:
    records the exception it is handed and marks the span ERROR."""

    def __init__(self) -> None:
        self.spans: list[_Span] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str, **attrs: Any):
        s = _Span(name)
        self.spans.append(s)
        try:
            yield s
        except BaseException as exc:
            s.exception = exc
            s.status = "ERROR"
            raise

    def current_span_id(self) -> None:
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        return None


class _FailingLLM:
    """`fail_after=0` → raises before the first token; `fail_after=1` → raises
    after one delta has already been streamed."""

    def __init__(self, fail_after: int) -> None:
        self._fail_after = fail_after

    async def stream(self, **_kw: Any):
        if self._fail_after == 0:
            raise RuntimeError("provider 500")
        yield Delta(text="par", model="m", provider="p")
        raise RuntimeError("provider 500 mid-stream")


class _OkLLM:
    async def stream(self, **_kw: Any):
        yield Delta(text="hello", model="m", provider="p")
        yield Delta(usage=Usage(3, 1, 0.0), finish_reason="stop", model="m", provider="p")


def _wire(llm: Any) -> tuple[Invoker, RunContext, _RecordingTrace]:
    trace = _RecordingTrace()
    inv = Invoker(llm=llm, chat_middleware=[tracing()], tool_middleware=[tracing()])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, trace=trace))
    return inv, ctx, trace


REQ = ChatRequest([Message("user", "q")], "m")


def test_a_failure_before_the_first_token_is_recorded_on_the_span() -> None:
    inv, ctx, trace = _wire(_FailingLLM(fail_after=0))

    with pytest.raises(RuntimeError, match="provider 500"):
        asyncio.run(inv.chat(REQ, ctx))

    (span,) = trace.spans
    assert span.name == "chat"
    assert isinstance(span.exception, RuntimeError), "the failure was exported as a successful span"
    assert span.status == "ERROR"


def test_a_failure_MID_stream_is_recorded_on_the_span() -> None:
    """The harder half: deltas already flowed, so the span had been stamped
    with real attributes before the error arrived."""
    inv, ctx, trace = _wire(_FailingLLM(fail_after=1))

    async def go() -> None:
        async for _ in inv.stream(REQ, ctx):
            pass

    with pytest.raises(RuntimeError, match="mid-stream"):
        asyncio.run(go())

    (span,) = trace.spans
    assert isinstance(span.exception, RuntimeError)
    assert span.status == "ERROR"


def test_a_failed_TOOL_call_is_recorded_on_its_span() -> None:
    """`execute_tool` spans go through the same helper — a tool that raises
    must not be exported as a success either."""

    class _BoomTool:
        name = "charge_card"

        async def run(self, arguments: dict[str, Any], ctx: Any) -> Any:
            raise TimeoutError("gateway timeout")

    inv, ctx, trace = _wire(_OkLLM())

    with pytest.raises(TimeoutError):
        asyncio.run(inv.invoke_tool(ToolRequest("charge_card", {"amount": 1}, _BoomTool()), ctx))

    (span,) = trace.spans
    assert span.name == "execute_tool"
    assert isinstance(span.exception, TimeoutError)


def test_the_original_exception_still_reaches_the_caller() -> None:
    """Recording must not become swallowing — including when the tracer's own
    `__exit__` claims to have handled the exception. A tracer is observability
    only; it may never make a provider fault disappear."""

    class _SwallowingTrace(_RecordingTrace):
        @contextlib.contextmanager
        def span(self, name: str, kind: str, **attrs: Any):
            s = _Span(name)
            self.spans.append(s)
            with contextlib.suppress(Exception):  # claims to handle it
                yield s
                return
            s.status = "SUPPRESSED"

    trace = _SwallowingTrace()
    inv = Invoker(llm=_FailingLLM(fail_after=0), chat_middleware=[tracing()])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, trace=trace))

    with pytest.raises(RuntimeError, match="provider 500"):
        asyncio.run(inv.chat(REQ, ctx))


def test_a_successful_call_is_NOT_marked_as_an_error() -> None:
    """POSITIVE CONTROL. A "fix" that unconditionally exits the span with a
    fabricated exception, or that drops spans entirely, fails here: a success
    still opens its span, records no exception, and carries its `gen_ai.*`
    attributes."""
    inv, ctx, trace = _wire(_OkLLM())

    result = asyncio.run(inv.chat(REQ, ctx))

    (span,) = trace.spans
    assert result.content == "hello"
    assert span.exception is None
    assert span.status is None
    assert span.attrs["gen_ai.usage.input_tokens"] == 3
    assert span.attrs["gen_ai.response.finish_reasons"] == "stop"


def test_an_abandoned_stream_closes_its_span_cleanly() -> None:
    """`GeneratorExit` reaches this frame when a consumer walks away. That is
    not a provider fault, so it must not be exported as a span error."""
    inv, ctx, trace = _wire(_OkLLM())

    async def go() -> None:
        gen = inv.stream(REQ, ctx)
        async with contextlib.aclosing(gen):
            async for _ in gen:
                break

    asyncio.run(go())

    (span,) = trace.spans
    assert span.exception is None, "an abandoned stream was exported as a failed call"


def test_a_broken_tracer_still_cannot_break_the_run() -> None:
    """The property `tests/middlewares/test_tracing_broken_span.py` pins for the
    OPEN path, re-checked for the CLOSE path now that close is exception-aware."""

    class _ExitBoomTrace:
        def __init__(self) -> None:
            self.span_calls = 0

        @contextlib.contextmanager
        def span(self, name: str, kind: str, **attrs: Any):
            self.span_calls += 1
            yield _Span(name)
            raise RuntimeError("boom on span close")

        def current_span_id(self) -> None:
            return None

        def add_event_to_current_span(self, name: str, **fields: Any) -> None:
            return None

    trace = _ExitBoomTrace()
    inv = Invoker(llm=_OkLLM(), chat_middleware=[tracing()])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, trace=trace))

    assert asyncio.run(inv.chat(REQ, ctx)).content == "hello"
    assert trace.span_calls == 1
