"""Streaming as the single primitive (ADR-0008): `stream()` is the one primitive, `chat ≡ collect(stream)`,
one middleware contract, incremental through the chain (retry commits on the first delta), and AgentLoop
surfaces real token deltas. Offline & deterministic."""

import asyncio

import pytest

from agentkit.agents import Agent
from agentkit.kernel.types import (
    ChatRequest,
    Delta,
    Message,
    Scope,
    ToolCall,
    Usage,
    assemble_deltas,
)
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Budget
from agentkit.testing import FakeLLM, Turn, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def test_stream_yields_multiple_incremental_deltas():
    llm = FakeLLM("hello world, this is streamed")  # > 8 chars → several chunks
    ctx = make_test_ctx(llm=llm, scope=Scope(1, 2), budget=Budget(max_cost_usd=10.0))
    req = ChatRequest(messages=[Message("user", "hi")], model="m")

    async def go():
        return [d async for d in ctx.invoker.stream(req, ctx)]

    deltas = _run(go())
    assert len(deltas) > 2  # genuinely incremental, not one blob
    assert all(isinstance(d, Delta) for d in deltas)
    assert "".join(d.text for d in deltas) == "hello world, this is streamed"
    assert (
        deltas[-1].usage is not None and deltas[-1].finish_reason == "stop"
    )  # terminal delta carries them


def test_chat_equals_collect_stream():
    """The core invariant: chat() is exactly collect(stream())."""
    req = ChatRequest(messages=[Message("user", "q")], model="m")
    ctx = make_test_ctx(
        llm=FakeLLM("a deterministic answer"), scope=Scope(1, 2), budget=Budget(max_cost_usd=10.0)
    )
    streamed = assemble_deltas(_run(_collect_list(ctx.invoker.stream(req, ctx))))
    direct = _run(
        ctx.invoker.chat(
            req,
            make_test_ctx(
                llm=FakeLLM("a deterministic answer"),
                scope=Scope(1, 2),
                budget=Budget(max_cost_usd=10.0),
            ),
        )
    )
    assert streamed.content == direct.content == "a deterministic answer"
    assert streamed.usage.total_tokens == direct.usage.total_tokens


async def _collect_list(stream):
    return [d async for d in stream]


def test_tool_calls_survive_streaming_collect():
    llm = FakeLLM.script([Turn(content="", tool_calls=(ToolCall("c1", "fetch", {"u": "x"}),))])
    ctx = make_test_ctx(llm=llm, scope=Scope(1, 2), budget=Budget(max_cost_usd=10.0))
    res = _run(ctx.invoker.chat(ChatRequest(messages=[Message("user", "go")], model="m"), ctx))
    assert res.finish_reason == "tool_calls" and res.tool_calls[0].name == "fetch"


# ---- incremental through the middleware chain -------------------------------------------------


def test_chain_streams_incrementally_and_meters_once():
    llm = FakeLLM("streamed through tracing and meter", usage=Usage(10, 5, 0.002))
    ctx = make_test_ctx(
        llm=llm,
        scope=Scope(1, 2),
        budget=Budget(max_cost_usd=10.0),
        chat_middleware=[tracing(), meter()],
    )
    req = ChatRequest(messages=[Message("user", "hi")], model="m")

    async def go():
        return [d.text async for d in ctx.invoker.stream(req, ctx) if d.text]

    chunks = _run(go())
    assert len(chunks) > 1  # tracing+meter pass deltas through incrementally
    assert (
        ctx.budget.spent_usd == pytest.approx(0.002) and ctx.budget.calls == 1
    )  # metered once, at stream-end


def test_retry_commits_on_first_delta_then_streams():
    # one pre-stream transient failure → retry re-invokes; the successful attempt streams incrementally
    llm = FakeLLM("recovered and streamed cleanly", fail_times=1, fail_exc=TimeoutError("t"))
    ctx = make_test_ctx(
        llm=llm,
        scope=Scope(1, 2),
        budget=Budget(max_cost_usd=10.0),
        chat_middleware=[retry(max_attempts=3, sleep=_nosleep)],
    )
    req = ChatRequest(messages=[Message("user", "hi")], model="m")

    async def go():
        return [d async for d in ctx.invoker.stream(req, ctx)]

    deltas = _run(go())
    assert llm.calls == 2  # retried the pre-stream failure once
    assert (
        "".join(d.text for d in deltas) == "recovered and streamed cleanly"
    )  # no attempt-1 partials leaked


async def _nosleep(_):
    pass


# ---- Agent (single-call) surfaces real tokens -------------------------------------------------


def test_agent_stream_surfaces_token_deltas():
    llm = FakeLLM("the final answer is forty two")  # multi-chunk
    loop = Agent(name="a", model="m")

    async def go():
        return [
            ev.text
            async for ev in loop.stream(
                "q", make_test_ctx(llm=llm, scope=Scope(1, 2), budget=Budget(max_cost_usd=10.0))
            )
            if ev.type == "message_delta"
        ]

    tokens = _run(go())
    assert len(tokens) > 1  # streamed token-by-token, not one blob
    assert "".join(tokens) == "the final answer is forty two"
