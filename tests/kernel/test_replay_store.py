"""Tests for the ``ReplayStore`` side-channel (B4c).

Verifies that the kernel's default no-op replay store has the right shape, that the tracing
middleware writes a ``ReplayRecord`` keyed by the chat span's id when a concrete store is
wired, and that the no-op default has no measurable cost on a chat call.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from agentkit.kernel.observation import TraceContext
from agentkit.kernel.replay import ReplayRecord, ReplayStore, NoopReplayStore
from agentkit.kernel.types import ChatRequest, Message
from agentkit.middlewares import tracing
from agentkit.runtime import Invoker, RunContext, Services
from agentkit.runtime.context import NoopTrace
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


# ---- A capturing TracePort that surfaces fixed span ids -------------------------------------


class _CapturingTrace:
    """Returns a deterministic ``TraceContext`` per opened span so tests can assert on
    the replay record's ``span_id`` without depending on a real tracer's id generator."""

    def __init__(self) -> None:
        self._counter = 0
        self.opened: list[TraceContext] = []
        self._stack: list[TraceContext] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str, **attrs: Any):
        self._counter += 1
        tc = TraceContext(trace_id="t" * 32, span_id=f"{self._counter:016x}")
        self._stack.append(tc)
        self.opened.append(tc)

        class _S:
            def set(self, *_a, **_kw):
                return None

            def add_event(self, *_a, **_kw):
                return None

        try:
            yield _S()
        finally:
            self._stack.pop()

    def current_span_id(self) -> TraceContext | None:
        return self._stack[-1] if self._stack else None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        return None


# ---- A capturing ReplayStore ----------------------------------------------------------------


@dataclass
class _CapturingReplayStore:
    records: list[ReplayRecord]

    def __init__(self) -> None:
        self.records = []

    async def put(self, record: ReplayRecord) -> None:
        self.records.append(record)

    async def get(self, span_id: str) -> ReplayRecord | None:
        return next((r for r in self.records if r.span_id == span_id), None)


def _ctx(*, llm, trace, replay) -> RunContext:
    from agentkit.kernel.types import Scope

    services = Services(
        invoker=Invoker(llm=llm, chat_middleware=[tracing()]),
        trace=trace,
        replay=replay,
    )
    return RunContext(correlation_id="r", scope=Scope(), services=services)


# ---- 1. Default NoopReplayStore --------------------------------------------------------


def test_noop_replay_store_put_returns_none():
    """``put`` on the default store accepts any record and returns None — the kernel
    must work without storage wired."""

    async def go():
        store = NoopReplayStore()
        rec = ReplayRecord(span_id="s" * 16, operation="chat", request={}, response=None)
        out = await store.put(rec)
        assert out is None

    _run(go())


def test_noop_replay_store_get_returns_none():
    """``get`` on the default store returns None for any key — no records stored."""

    async def go():
        store = NoopReplayStore()
        assert await store.get("anything") is None
        assert await store.get("s" * 16) is None

    _run(go())


def test_noop_replay_store_is_replaystore_shape():
    """The default impl structurally satisfies ``ReplayStore`` so the protocol check
    holds without a runtime registration step."""
    assert isinstance(NoopReplayStore(), ReplayStore)


def test_services_default_replay_is_noop():
    """A bare ``Services()`` exposes the no-op replay store so ``ctx.replay.put`` is
    always safe to call."""
    services = Services()
    assert isinstance(services.replay, NoopReplayStore)


def test_runcontext_replay_property_proxies_services():
    """``ctx.replay`` mirrors the convenience accessors for trace/observer/store —
    one import, one accessor."""
    from agentkit.kernel.types import Scope

    store = _CapturingReplayStore()
    ctx = RunContext("r", Scope(), services=Services(replay=store))
    assert ctx.replay is store


# ---- 2. Capturing ReplayStore receives one record per chat call --------------------------


def test_capturing_store_receives_one_record_per_chat_call():
    """The tracing middleware writes exactly one ``ReplayRecord`` per chat call when
    a concrete store is wired."""

    async def go():
        llm = FakeLLM("hello")
        trace = _CapturingTrace()
        replay = _CapturingReplayStore()
        ctx = _ctx(llm=llm, trace=trace, replay=replay)
        req = ChatRequest(messages=[Message("user", "hi")], model="m")
        await ctx.invoker.chat(req, ctx)
        assert len(replay.records) == 1
        rec = replay.records[0]
        assert rec.operation == "chat"
        assert rec.request["model"] == "m"
        assert rec.response.content == "hello"

    _run(go())


def test_capturing_store_records_one_per_call_over_multiple_chats():
    """Two chat calls → two records — the middleware doesn't accidentally batch."""

    async def go():
        llm = FakeLLM("hello")
        trace = _CapturingTrace()
        replay = _CapturingReplayStore()
        ctx = _ctx(llm=llm, trace=trace, replay=replay)
        await ctx.invoker.chat(ChatRequest(messages=[Message("user", "a")], model="m"), ctx)
        await ctx.invoker.chat(ChatRequest(messages=[Message("user", "b")], model="m"), ctx)
        assert len(replay.records) == 2

    _run(go())


# ---- 3. Record's span_id matches the chat span's id --------------------------------------


def test_record_span_id_matches_chat_span():
    """The ``ReplayRecord.span_id`` matches the ``current_span_id()`` value captured
    when the chat span was open."""

    async def go():
        llm = FakeLLM("hello")
        trace = _CapturingTrace()
        replay = _CapturingReplayStore()
        ctx = _ctx(llm=llm, trace=trace, replay=replay)
        req = ChatRequest(messages=[Message("user", "hi")], model="m")
        await ctx.invoker.chat(req, ctx)
        assert len(trace.opened) == 1
        chat_span = trace.opened[0]
        assert replay.records[0].span_id == chat_span.span_id

    _run(go())


def test_record_round_trips_through_capturing_store():
    """``get(span_id)`` returns the record that ``put`` accepted — proves the store
    isn't a black hole, even for a trivial in-memory impl."""

    async def go():
        llm = FakeLLM("hello")
        trace = _CapturingTrace()
        replay = _CapturingReplayStore()
        ctx = _ctx(llm=llm, trace=trace, replay=replay)
        req = ChatRequest(messages=[Message("user", "hi")], model="m")
        await ctx.invoker.chat(req, ctx)
        span_id = replay.records[0].span_id
        rec = await replay.get(span_id)
        assert rec is not None
        assert rec.span_id == span_id

    _run(go())


# ---- 4. NoopReplayStore: no overhead / no surfaced writes --------------------------------


def test_noop_replay_store_silently_drops_writes_over_chat():
    """When the default no-op store is wired, chat calls work and the store stays
    empty — no record persisted, no exception raised."""

    async def go():
        llm = FakeLLM("hello")
        trace = _CapturingTrace()
        replay = NoopReplayStore()
        ctx = _ctx(llm=llm, trace=trace, replay=replay)
        req = ChatRequest(messages=[Message("user", "hi")], model="m")
        res = await ctx.invoker.chat(req, ctx)
        # Chat returned cleanly; store stayed empty (get returns None for everything).
        assert res.content == "hello"
        assert await replay.get("anything") is None

    _run(go())


def test_replay_write_failure_does_not_break_run():
    """A misbehaving ``ReplayStore`` (raises on put) must NEVER break the run — the
    chat call still returns its result."""

    class _ExplodingStore:
        async def put(self, rec):
            raise RuntimeError("boom")

        async def get(self, key):
            return None

    async def go():
        llm = FakeLLM("hello")
        trace = _CapturingTrace()
        replay = _ExplodingStore()
        ctx = _ctx(llm=llm, trace=trace, replay=replay)
        req = ChatRequest(messages=[Message("user", "hi")], model="m")
        res = await ctx.invoker.chat(req, ctx)
        assert res.content == "hello"  # exception swallowed; result delivered

    _run(go())


def test_no_span_no_write():
    """When the tracer's ``current_span_id()`` returns None (e.g. the default
    ``NoopTrace``), the middleware skips the replay write — no NoneType key can
    leak into the store."""

    async def go():
        llm = FakeLLM("hello")
        replay = _CapturingReplayStore()
        ctx = _ctx(llm=llm, trace=NoopTrace(), replay=replay)
        req = ChatRequest(messages=[Message("user", "hi")], model="m")
        await ctx.invoker.chat(req, ctx)
        assert replay.records == []

    _run(go())
