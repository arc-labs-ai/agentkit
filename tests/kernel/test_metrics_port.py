"""MetricsPort — the counter + histogram seam.

These tests cover the kernel-level contract: the no-op default
drops everything, a capturing test double records calls, and a
deliberately-exploding impl must NEVER raise into the run path
(``ctx.emit`` + a chat call both keep working when ``metrics`` is
hostile).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

from agentkit.kernel.metrics import MetricsPort, NoopMetrics
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.middlewares import tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


# ── Recording double ─────────────────────────────────────────────────────────


@dataclass
class _CounterCall:
    name: str
    value: int | float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class _HistogramCall:
    name: str
    value: int | float
    tags: dict[str, str] = field(default_factory=dict)


class RecordingMetrics:
    """Test-local ``MetricsPort`` that records every call.

    Counters accumulate per ``(name, frozenset(tags.items()))`` so the
    test can verify accumulation; histograms keep the full sequence of
    values per ``(name, frozenset(tags.items()))`` so distributions can
    be inspected."""

    def __init__(self) -> None:
        self.counter_calls: list[_CounterCall] = []
        self.histogram_calls: list[_HistogramCall] = []

    def add_counter(
        self,
        name: str,
        value: int | float = 1,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.counter_calls.append(_CounterCall(name=name, value=value, tags=dict(tags or {})))

    def record_histogram(
        self,
        name: str,
        value: int | float,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.histogram_calls.append(_HistogramCall(name=name, value=value, tags=dict(tags or {})))

    # Helpers ----------------------------------------------------------------
    def counter_total(self, name: str) -> int | float:
        return sum(c.value for c in self.counter_calls if c.name == name)

    def histogram_values(self, name: str) -> list[int | float]:
        return [h.value for h in self.histogram_calls if h.name == name]


class ExplodingMetrics:
    """Deliberately-hostile ``MetricsPort``. Every method raises. The
    contract is that this MUST NOT propagate into the run."""

    def add_counter(self, name: str, value: int | float = 1, *, tags=None) -> None:
        raise RuntimeError("metrics backend on fire")

    def record_histogram(self, name: str, value: int | float, *, tags=None) -> None:
        raise RuntimeError("metrics backend on fire")


# ── Noop default ─────────────────────────────────────────────────────────────


def test_noop_metrics_accepts_calls_silently():
    """The default ``NoopMetrics`` accepts both surfaces without raising
    and returns None — zero overhead beyond the method dispatch."""
    m = NoopMetrics()
    assert m.add_counter("calls", 1, tags={"model": "m"}) is None
    assert m.add_counter("calls") is None  # default value
    assert m.record_histogram("latency", 12.3, tags={"model": "m"}) is None
    assert m.record_histogram("tokens", 7) is None
    # NoopMetrics is structurally a MetricsPort
    assert isinstance(m, MetricsPort)


def test_recording_metrics_accumulates_counters_and_records_histograms():
    """The capturing double sums counters by name and keeps every
    histogram observation."""
    m = RecordingMetrics()
    m.add_counter("requests", 1, tags={"model": "m1"})
    m.add_counter("requests", 2, tags={"model": "m1"})
    m.add_counter("requests", 5, tags={"model": "m2"})
    assert m.counter_total("requests") == 8

    m.record_histogram("tokens", 100)
    m.record_histogram("tokens", 50)
    m.record_histogram("tokens", 1000)
    assert m.histogram_values("tokens") == [100, 50, 1000]


# ── Exploding impl must not break the run ────────────────────────────────────


def test_exploding_metrics_does_not_break_emit():
    """``ctx.emit`` must complete successfully even if ``ctx.metrics``
    is hostile — emit doesn't read metrics today, but the protocol is
    that a bad metrics impl never propagates."""

    async def go() -> None:
        services = Services(metrics=ExplodingMetrics())
        ctx = RunContext("r", Scope(), Budget(), services)
        await ctx.emit("progress", "step 1")  # must not raise

    _run(go())


def test_exploding_metrics_does_not_break_chat_call():
    """A full chat call through the tracing middleware must succeed
    even when ``ctx.metrics`` raises on every call — the middleware
    wraps its metric writes in ``suppress(Exception)``."""

    async def go() -> None:
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r",
            Scope(),
            Budget(),
            Services(invoker=inv, metrics=ExplodingMetrics()),
        )
        res = await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)
        assert res.content == "ok"

    _run(go())
