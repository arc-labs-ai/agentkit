"""Tracing middleware × MetricsPort integration.

Verifies that the tracing middleware records the OTel-aligned
histograms + error counter expected by production dashboards:

- ``gen_ai.client.token.usage`` — split into input + output series
  via the ``gen_ai.token.type`` tag
- ``gen_ai.client.operation.duration`` — duration in SECONDS per OTel
  convention
- ``gen_ai.client.error.count`` — counter on the exception path

Tags include the low-cardinality set: ``gen_ai.request.model`` and a
stable ``agentkit.scope.org_id`` (or ``"none"`` when no scope is set).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.middlewares import tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _Call:
    name: str
    value: int | float
    tags: dict[str, str] = field(default_factory=dict)


class _RecordingMetrics:
    """Captures every counter + histogram emission."""

    def __init__(self) -> None:
        self.counters: list[_Call] = []
        self.histograms: list[_Call] = []

    def add_counter(
        self,
        name: str,
        value: int | float = 1,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.counters.append(_Call(name=name, value=value, tags=dict(tags or {})))

    def record_histogram(
        self,
        name: str,
        value: int | float,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.histograms.append(_Call(name=name, value=value, tags=dict(tags or {})))


# ── happy path: histograms land ──────────────────────────────────────────────


def test_chat_call_records_input_and_output_token_histograms():
    """A successful chat with usage produces two ``gen_ai.client.token.usage``
    samples — one tagged ``input`` and one tagged ``output``."""

    async def go() -> None:
        metrics = _RecordingMetrics()
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.002)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-tokens",
            Scope(),
            Budget(),
            Services(invoker=inv, metrics=metrics),
        )
        await inv.chat(ChatRequest([Message("user", "q")], "claude-sonnet-4-6"), ctx)

        token_hists = [h for h in metrics.histograms if h.name == "gen_ai.client.token.usage"]
        assert len(token_hists) == 2
        by_type = {h.tags["gen_ai.token.type"]: h for h in token_hists}
        assert by_type["input"].value == 100
        assert by_type["output"].value == 20
        # Tags include the low-cardinality model id
        for h in token_hists:
            assert h.tags["gen_ai.request.model"] == "claude-sonnet-4-6"

    _run(go())


def test_chat_call_records_operation_duration_histogram():
    """A successful chat produces a single ``gen_ai.client.operation.duration``
    sample, in SECONDS, with the same low-cardinality tag set."""

    async def go() -> None:
        metrics = _RecordingMetrics()
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-dur",
            Scope(),
            Budget(),
            Services(invoker=inv, metrics=metrics),
        )
        await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)

        dur = [h for h in metrics.histograms if h.name == "gen_ai.client.operation.duration"]
        assert len(dur) == 1
        # OTel convention: seconds (not ms). Value ought to be small + non-negative.
        assert dur[0].value >= 0
        assert dur[0].value < 5.0  # generous upper bound; in practice microseconds
        assert dur[0].tags["gen_ai.request.model"] == "m"

    _run(go())


def test_token_histograms_skipped_when_usage_is_zero():
    """No input/output tokens reported → no token-usage histograms.
    Duration still lands (the call still happened)."""

    async def go() -> None:
        metrics = _RecordingMetrics()
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext("r-zero", Scope(), Budget(), Services(invoker=inv, metrics=metrics))
        await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)

        token_hists = [h for h in metrics.histograms if h.name == "gen_ai.client.token.usage"]
        assert token_hists == []
        dur = [h for h in metrics.histograms if h.name == "gen_ai.client.operation.duration"]
        assert len(dur) == 1

    _run(go())


def test_scope_org_id_flows_into_metric_tags_when_set():
    """A Scope with org_id passes through into the low-cardinality tag set.
    Unset scope renders as ``"none"`` so the tag is always present."""

    async def go() -> None:
        metrics = _RecordingMetrics()
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=5, output_tokens=5, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-org",
            Scope(org_id=42),
            Budget(),
            Services(invoker=inv, metrics=metrics),
        )
        await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)

        dur = next(h for h in metrics.histograms if h.name == "gen_ai.client.operation.duration")
        assert dur.tags["agentkit.scope.org_id"] == "42"

    _run(go())


# ── error path: error counter ────────────────────────────────────────────────


def test_chat_call_raising_records_error_counter():
    """An exception during the chat stream produces one
    ``gen_ai.client.error.count`` increment tagged with the exception's
    class name, then re-raises so the caller still sees the failure."""

    async def go() -> None:
        metrics = _RecordingMetrics()
        # FakeLLM with fail_times keeps failing past the retry budget;
        # there's no retry middleware here so the first failure escapes.
        inv = Invoker(
            llm=FakeLLM("never", fail_times=10, fail_exc=TimeoutError("boom")),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-err",
            Scope(),
            Budget(),
            Services(invoker=inv, metrics=metrics),
        )
        with pytest.raises(TimeoutError):
            await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)

        errs = [c for c in metrics.counters if c.name == "gen_ai.client.error.count"]
        assert len(errs) == 1
        assert errs[0].value == 1
        assert errs[0].tags["error.type"] == "TimeoutError"
        assert errs[0].tags["gen_ai.request.model"] == "m"

    _run(go())
