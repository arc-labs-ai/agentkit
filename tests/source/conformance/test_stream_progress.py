"""Scripted API streams distinguish measured usage and model progress from status."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from _assertions import assert_money

from agentkit.core.runtime import calls
from agentkit.core.runtime.calls import CallRuntime, RetryPolicy
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef
from agentkit.models.providers.frontiers.calls import ApiAdapter

PRICES = PriceTable("test", "2026-11-12", {"test": {"input_uncached": "1", "output": "5"}})


def frames(mode, kind="content"):
    if mode == "openai_api":
        delta = {
            "status": {"type": "response.in_progress"},
            "content": {"type": "response.output_text.delta", "delta": "ok"},
            "reasoning": {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
            "tool": {"type": "response.function_call_arguments.delta", "delta": "{}"},
        }
        final = {"type": "response.completed", "response": {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }}
        return [], final if kind == "final" else delta[kind], [final]
    setup = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    delta = {
        "status": {"type": "ping"},
        "content": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        "reasoning": {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        "tool": {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": " "}},
    }
    final = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}
    return setup, final if kind == "final" else delta[kind], [final, {"type": "message_stop"}]


def encoded(frame):
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


async def ask(mode, stream, *, limits=None):
    credential = CredentialRef(lambda: {"OPENAI_API_KEY": "fake", "ANTHROPIC_API_KEY": "fake"})
    events = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    )) as client:
        adapter = ApiAdapter(client=client, base_url="https://model.invalid")
        async with CallRuntime(adapters={mode: adapter}, pricing=PRICES, retry=RetryPolicy(max_attempts=1)) as runtime:
            answer = await runtime.ask(
                Binding(mode, "test", credential), "q", call_id="stream-progress", limits=limits or CallLimits(), events=events.append,
            )
    return answer, events


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
@pytest.mark.parametrize("terminal", [False, True], ids=["eof-unknown", "terminal-exact-twin"])
async def test_stream_usage_requires_terminal_frame(mode, terminal):
    setup, delta, final = frames(mode)
    body = setup + [delta, delta] + (final if terminal else [])
    answer, events = await ask(mode, httpx.ByteStream(b"".join(map(encoded, body))))
    assert answer.attempts == 1
    assert answer.usage_complete is terminal
    assert answer.usage.input_uncached == (7 if terminal or mode == "anthropic_api" else 0)
    assert answer.usage.output == (3 if terminal else 0)
    usage = [event for event in events if event.kind == "usage"]
    assert len(usage) == 1
    if terminal:
        assert answer.outcome == "answered"
        assert_money(answer.cost_usd, "0.000022")
        assert events[-1].fields["cost_usd"] == "0.000022"
    else:
        assert answer.failure.classification == "transient_network"
        assert answer.cost_usd is None and answer.cost_source == "unpriced"
        assert events[-1].fields["cost_usd"] is None


class ClockedStream(httpx.AsyncByteStream):
    """Each frame hands control to the existing monitor, with no sleeps or timers."""

    def __init__(self, mode, kind):
        self.now = 0.0
        self.mode, self.kind = mode, kind
        self.checkpoint = asyncio.Event()
        self.resume = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        setup, delta, final = frames(self.mode, self.kind)
        # Status-only starts without a usage/content frame.
        if self.kind != "status":
            for frame in setup:
                yield encoded(frame)
        for stamp in (0.0, 0.75, 1.5, 2.25):
            self.now = stamp
            yield encoded(delta)
            self.checkpoint.set()
            await self.resume.wait()
            self.resume.clear()
        if self.mode == "anthropic_api":
            if self.kind == "status":
                for frame in setup:
                    yield encoded(frame)
            yield encoded({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}})
        for frame in final:
            yield encoded(frame)

    async def aclose(self):
        self.closed = True

    async def monitor_wait(self, tasks, *, timeout):
        checkpoint = asyncio.create_task(self.checkpoint.wait())
        try:
            await asyncio.wait(tasks | {checkpoint}, return_when=asyncio.FIRST_COMPLETED)
            self.checkpoint.clear()
            self.resume.set()
        finally:
            checkpoint.cancel()
            await asyncio.gather(checkpoint, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
@pytest.mark.parametrize("kind", ["status", "content", "reasoning", "tool", "final"])
async def test_idle_clock_requires_model_progress(monkeypatch, mode, kind):
    stream = ClockedStream(mode, kind)
    monkeypatch.setattr(calls, "time", SimpleNamespace(monotonic=lambda: stream.now))
    # Patch only this module's wait seam; the event loop keeps its real clock.
    monkeypatch.setattr(calls, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio) if name != "wait"},
        wait=stream.monitor_wait,
    ))
    answer, _ = await ask(mode, stream, limits=CallLimits(idle_s=1))
    assert stream.closed
    if kind == "status":
        assert answer.failure is not None
        assert answer.failure.classification == "timeout"
        assert answer.failure.mode_detail["limit"] == "idle"
        assert stream.now == 1.5
    else:
        assert answer.outcome == "answered", answer.failure
        assert answer.usage_complete
        assert stream.now == 2.25
