"""KEYSTONE — proves the unification: every cross-cutting concern works as a composed middleware
around the Invoker's two units of work (chat, tool), with no god method in sight."""

import asyncio

import pytest

from agentkit.adapters.store import InMemoryStore
from agentkit.capabilities import Guardrail
from agentkit.kernel.types import ChatRequest, LLMResult, Message, Scope, ToolRequest, Usage
from agentkit.middlewares import audit, egress, fallback, idempotent, meter, retry, tracing
from agentkit.runtime import Budget
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.testing import RecordingTracer as Tracer
from agentkit.tools import FunctionTool


def _run(coro):
    return asyncio.run(coro)


async def _nosleep(d):
    pass


# ---- chat chain: tracing + meter + retry compose -------------------------------------------


def test_chat_chain_traces_meters_and_retries():
    spans = []
    llm = FakeLLM("hello", fail_times=1, fail_exc=TimeoutError("t"), usage=Usage(100, 20, 0.002))
    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[tracing(), meter(), retry(max_attempts=3, sleep=_nosleep)],
        budget=Budget(max_cost_usd=1.0),
        trace=Tracer(exporter=spans.append),
        scope=Scope(1, 2),
        correlation_id="r",
    )
    res = _run(ctx.invoker.chat(ChatRequest(messages=[Message("user", "hi")], model="m"), ctx))

    assert res.content == "hello" and llm.calls == 2  # retried once (transient)
    assert ctx.budget.spent_usd == pytest.approx(0.002) and ctx.budget.calls == 1  # metered once
    chat_span = next(s for s in spans if s.name == "chat")
    assert chat_span.attrs.get("gen_ai.request.model") == "m"
    assert chat_span.attrs.get("gen_ai.usage.input_tokens") == 100


def test_meter_blocks_before_spend_and_is_catchable():
    from agentkit.runtime import MeterExceeded

    llm = FakeLLM("x", usage=Usage(0, 0, 0.01))
    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[meter()],
        budget=Budget(max_cost_usd=0.015),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="r",
    )

    def req():
        return ChatRequest(messages=[Message("user", "hi")], model="m")

    _run(ctx.invoker.chat(req(), ctx))  # 0.01 ok
    with pytest.raises(MeterExceeded):
        _run(ctx.invoker.chat(req(), ctx))  # → 0.02 > 0.015


def test_fallback_switches_model_on_transient_failure():
    class ModelLLM:
        async def chat(self, *, model, messages, tools=None, **kw):
            if model == "a":
                raise TimeoutError("a down")
            return LLMResult(content="ok", model=model, usage=Usage(1, 1, 0.0))

        async def stream(self, *, model, messages, tools=None, **kw):
            from agentkit.adapters.llm._mapping import result_to_delta

            yield result_to_delta(
                await self.chat(model=model, messages=messages, tools=tools, **kw)
            )

    ctx = make_test_ctx(
        llm=ModelLLM(),
        chat_middleware=[fallback(["a", "b"])],
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="r",
    )
    res = _run(
        ctx.invoker.chat(ChatRequest(messages=[Message("user", "hi")], model="ignored"), ctx)
    )
    assert res.content == "ok" and res.model == "b"  # fell over a → b, served model stamped


# ---- tool chain: egress + idempotent + audit compose ---------------------------------------


def test_tool_chain_audits_every_call_and_idempotently_dedupes():
    n = {"c": 0}
    store = InMemoryStore()
    tool = FunctionTool(
        "charge",
        lambda a, c: n.__setitem__("c", n["c"] + 1) or {"ok": 1},
        description="charge an account a given amount; mutates ledger",
        side_effecting=True,
    )
    ctx = make_test_ctx(
        llm=FakeLLM("ok"),
        tool_middleware=[tracing(), audit(), idempotent()],
        store=store,
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="r",
    )  # audit OUTER → records even on a hit
    req = ToolRequest("charge", {"amt": 5}, tool, side_effecting=True)

    _run(ctx.invoker.invoke_tool(req, ctx))
    _run(ctx.invoker.invoke_tool(req, ctx))  # same run+args → deduped execution
    assert n["c"] == 1  # executed once
    log = _run(store.list("audit:log"))
    assert len(log) == 2 and log[0]["tool"] == "charge"  # but audited both
    # the trail distinguishes the real side effect from the idempotent replay (audit fidelity)
    assert [r["decision"] for r in log] == ["executed", "deduped"]


def test_egress_blocks_disallowed_host_before_execution():
    n = {"c": 0}
    tool = FunctionTool(
        "fetch",
        lambda a, c: n.__setitem__("c", n["c"] + 1) or "ok",
        description="fetch a URL and return its body; egress-gated",
        side_effecting=True,
        url_arg="url",
    )
    ctx = make_test_ctx(
        llm=FakeLLM("ok"),
        tool_middleware=[egress(Guardrail(egress_allow=("example.com",)))],
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="r",
    )
    with pytest.raises(PermissionError):
        _run(
            ctx.invoker.invoke_tool(
                ToolRequest("fetch", {"url": "https://evil.com/x"}, tool, url_arg="url"), ctx
            )
        )
    assert n["c"] == 0  # blocked before the side effect
    out = _run(
        ctx.invoker.invoke_tool(
            ToolRequest("fetch", {"url": "https://api.example.com"}, tool, url_arg="url"), ctx
        )
    )
    assert out == "ok"


def test_idempotency_never_caches_failure():
    n = {"c": 0}

    def flaky(a, c):
        n["c"] += 1
        if n["c"] == 1:
            raise ValueError("boom")
        return "ok"

    store = InMemoryStore()
    tool = FunctionTool(
        "f",
        flaky,
        description="a flaky example tool used to test retry semantics",
        side_effecting=True,
    )
    ctx = make_test_ctx(
        llm=FakeLLM("ok"),
        tool_middleware=[idempotent()],
        store=store,
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="r",
    )
    req = ToolRequest("f", {"x": 1}, tool, side_effecting=True)
    with pytest.raises(ValueError):
        _run(ctx.invoker.invoke_tool(req, ctx))
    assert _run(ctx.invoker.invoke_tool(req, ctx)) == "ok" and n["c"] == 2  # failure not cached
