"""``output_coerce`` middleware — the in-chain response-coercion contract.

These tests pin the middleware-layer surface independently of the Agent:

1. With no adapter wired on the ``ctx``, the middleware is a strict
   no-op: deltas pass through unchanged and ``LLMResult.parsed`` is
   ``None``.
2. With an adapter on the ``RunContext`` and a valid model response,
   the assembled ``LLMResult.parsed`` is the typed instance.
3. With an adapter and an invalid response, the middleware re-raises
   ``OutputCoercionError`` so the outer retry policy sees the
   diagnostics.
4. Tool calls are passed through verbatim — a tool result is opaque
   to the schema adapter.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.capabilities.output_schema import OutputCoercionError, adapt
from agentkit.kernel.types import ChatRequest, Message
from agentkit.middlewares import output_coerce
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM, make_test_ctx

pydantic = pytest.importorskip("pydantic")


class Plan(pydantic.BaseModel):
    subject: str
    steps: list[str]


def _chat(llm, ctx, request):
    invoker = Invoker(llm=llm, chat_middleware=[output_coerce()])
    return asyncio.run(invoker.chat(request, ctx))


# ── 1. No adapter → no parsed ──────────────────────────────────────────────


def test_no_adapter_is_strict_noop() -> None:
    llm = FakeLLM('{"x": 1}')
    ctx = make_test_ctx(llm=llm)  # no _output_adapter on ctx
    request = ChatRequest(messages=[Message("user", "hi")], model="m")
    result = _chat(llm, ctx, request)
    assert result.content == '{"x": 1}'
    assert result.parsed is None


# ── 2. Adapter + valid → typed parsed ──────────────────────────────────────


def test_adapter_with_valid_response_yields_typed_parsed() -> None:
    valid = '{"subject": "ship", "steps": ["a", "b"]}'
    llm = FakeLLM(valid)
    ctx = make_test_ctx(llm=llm)
    ctx._output_adapter = adapt(Plan)  # type: ignore[attr-defined]
    request = ChatRequest(messages=[Message("user", "plan")], model="m")
    result = _chat(llm, ctx, request)
    assert isinstance(result.parsed, Plan)
    assert result.parsed.subject == "ship"
    assert result.parsed.steps == ["a", "b"]


# ── 3. Adapter + invalid → OutputCoercionError re-raised ────────────────────


def test_adapter_with_invalid_response_raises_coercion_error() -> None:
    llm = FakeLLM('{"subject": "x"}')  # missing required `steps`
    ctx = make_test_ctx(llm=llm)
    ctx._output_adapter = adapt(Plan)  # type: ignore[attr-defined]
    request = ChatRequest(messages=[Message("user", "plan")], model="m")
    with pytest.raises(OutputCoercionError) as exc_info:
        _chat(llm, ctx, request)
    # The error carries per-field diagnostics for the retry middleware to
    # reflect back to the model.
    assert exc_info.value.errors
    assert any("steps" in e for e in exc_info.value.errors)


# ── 4. Tool calls pass through unchanged ───────────────────────────────────


def test_tool_call_passes_through() -> None:
    """The middleware is chat-only — tool calls go through untouched even
    when an adapter is set on ctx."""
    from agentkit.kernel.types import ToolRequest

    class _T:
        async def run(self, args, ctx):  # noqa: ARG002
            return {"ok": True}

    llm = FakeLLM("unused")
    ctx = make_test_ctx(llm=llm)
    ctx._output_adapter = adapt(Plan)  # type: ignore[attr-defined]
    invoker = Invoker(llm=llm, tool_middleware=[output_coerce()])
    req = ToolRequest(name="t", arguments={}, tool=_T())
    out = asyncio.run(invoker.invoke_tool(req, ctx))
    assert out == {"ok": True}
