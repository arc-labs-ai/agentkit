from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import httpx
import pytest

from agentkit.core.execution.concurrency import CancellationToken
from agentkit.core.runtime.calls import CallRuntime, RetryPolicy
from agentkit.core.testing.fakes import FakeClaudeCli, FakeCodexCli, codex_turn
from agentkit.models.catalog.pricing import PriceTable, normalize_usage, price
from agentkit.models.contracts.schema import OutputSpec, clean_nulls, compile_schema
from agentkit.models.contracts.types import Binding, CredentialRef, Failure, ModelSettings, Turn
from agentkit.models.providers.cli.calls import CliAdapter
from agentkit.models.providers.frontiers.calls import ApiAdapter

MODES = ("claude_code", "codex", "anthropic_api", "openai_api")
PRICES = PriceTable(
    "test",
    "2026-10-13",
    {
        "test": {
            "input_uncached": "1",
            "input_cache_read": "0.1",
            "input_cache_write_5m": "1.25",
            "input_cache_write_1h": "2",
            "output": "5",
        }
    },
)
SCHEMA = {
    "type": "object",
    "properties": {
        "required": {"type": ["string", "null"]},
        "nested": {"anyOf": [{"$ref": "#/$defs/nested"}, {"type": "string"}]},
        "optional": {"type": "string"},
    },
    "required": ["required", "nested"],
    "additionalProperties": False,
    "$defs": {
        "nested": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "note": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
    },
}


class ScriptedAdapter:
    def __init__(self, *turns: Turn) -> None:
        self.turns = list(turns)
        self.requests = []

    async def turn(self, binding: Any, prompt: Any, **kwargs: Any) -> Turn:
        self.requests.append((binding, prompt, kwargs))
        if not self.turns:
            raise AssertionError("unexpected adapter call")
        return self.turns.pop(0)


def sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)


def real_adapter(mode: str, text: str, *, value: Any = None, commentary: str = "Working...") -> Any:
    if mode == "claude_code":
        payload = [
            {"type": "assistant", "message": {"id": "first", "content": [{"type": "text", "text": commentary}]}},
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": text}]}},
            {
                "type": "result",
                "subtype": "success",
                "result": text,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "structured_output": value,
                "session_id": "test-session",
                "total_cost_usd": 0.000035,
            },
        ]
        return CliAdapter(spawn=FakeClaudeCli.script(payload))
    if mode == "codex":
        return CliAdapter(
            spawn=FakeCodexCli.script(
                codex_turn(
                    text=text,
                    usage=(10, 0, 5),
                    items=[{"id": "commentary", "type": "agent_message", "text": commentary}],
                )
            )
        )
    if mode == "openai_api":
        events = [
            {"type": "response.output_text.delta", "delta": text},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": commentary}]},
                        {"type": "message", "content": [{"type": "output_text", "text": text}]},
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        ]
    else:
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=sse(events))))
    return ApiAdapter(client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_final_answer_through_actual_translators(mode: str) -> None:
    adapter = real_adapter(mode, "42")
    binding = Binding(mode, "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test", "ANTHROPIC_API_KEY": "test"}))
    events = []
    try:
        async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
            answer = await runtime.ask(binding, "answer", events=events.append, call_id="test")
        assert answer.outcome == "answered", answer.failure
        assert answer.text == "42"
        assert answer.usage.input_uncached == 10
        assert answer.usage.output == 5
        assert events[0].kind == "call_started" and events[-1].kind == "call_finished"
        assert not any("text" in e.fields for e in events)
    finally:
        if hasattr(adapter, "client"):
            await adapter.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_nested_nulls_through_actual_translators(mode: str) -> None:
    value = {"required": None, "nested": {"name": "ok", "note": None}, "optional": None}
    adapter = real_adapter(mode, json.dumps(value), value=value)
    binding = Binding(mode, "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test", "ANTHROPIC_API_KEY": "test"}))
    try:
        async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
            answer = await runtime.ask(binding, "JSON", OutputSpec(SCHEMA), call_id="structured")
        assert answer.outcome == "answered", answer.failure
        assert answer.value == {"required": None, "nested": {"name": "ok"}}
    finally:
        if hasattr(adapter, "client"):
            await adapter.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_one_correction_turn_and_canonical_bounds(mode: str) -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}}, "required": ["n"]}
    adapter = ScriptedAdapter(Turn(("preface", '{"n":1}')), Turn(('{"n":3}',)))
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding(mode, "test"), "JSON", OutputSpec(schema), call_id="repair")
    assert answer.value == {"n": 3} and answer.attempts == 2
    assert len(adapter.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["refusal", "truncation", "failure"])
async def test_terminal_outcomes_never_repair(mode: str, kind: str) -> None:
    turn = Turn(
        ("partial",),
        refused=kind == "refusal",
        truncated=kind == "truncation",
        failure=Failure("auth") if kind == "failure" else None,
    )
    adapter = ScriptedAdapter(turn)
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding(mode, "test"), "JSON", OutputSpec(SCHEMA), call_id="terminal")
    assert answer.outcome == {"refusal": "refused", "truncation": "truncated", "failure": "failed"}[kind]
    assert len(adapter.requests) == 1
    if kind != "truncation":
        assert answer.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_unpriced_is_refused_before_dispatch(mode: str) -> None:
    adapter = ScriptedAdapter()
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding(mode, "unknown-model"), "hello", call_id="unpriced")
    assert answer.cost_source == "unpriced" and answer.cost_usd is None
    assert answer.outcome == "failed" and not adapter.requests


@pytest.mark.asyncio
async def test_retry_after_minimum_and_quota_never_retries() -> None:
    events = []
    adapter = ScriptedAdapter(Turn(failure=Failure("rate_limited", True, 0.001)), Turn(("done",)))
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES, retry=RetryPolicy(base_s=0, cap_s=1))
    answer = await runtime.ask(Binding("openai_api", "test"), "hello", events=events.append, call_id="retry")
    assert answer.text == "done" and answer.attempts == 2
    assert next(e for e in events if e.kind == "retry").fields["wait_seconds"] >= 0.001
    adapter.turns = [Turn(failure=Failure("quota_exhausted"))]
    answer = await runtime.ask(Binding("openai_api", "test"), "hello", call_id="quota")
    assert answer.attempts == 1 and answer.failure.classification == "quota_exhausted"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_cancellation_unwinds_transport(mode: str) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    class Blocking:
        async def turn(self, *args: Any, **kwargs: Any) -> Turn:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return Turn()

    token = CancellationToken()
    events = []
    runtime = CallRuntime(adapters={mode: Blocking()}, pricing=PRICES)
    task = asyncio.create_task(
        runtime.ask(Binding(mode, "test"), "hello", cancel=token, events=events.append, call_id="cancel")
    )
    await started.wait()
    token.cancel()
    answer = await asyncio.wait_for(task, 1)
    assert cleaned.is_set() and answer.outcome == "cancelled"
    assert events[-1].fields["outcome"] == "cancelled"


@pytest.mark.parametrize("mode", MODES)
def test_schema_compilation_never_mutates_canonical(mode: str) -> None:
    snapshot = json.dumps(SCHEMA, sort_keys=True)
    compiled = compile_schema(SCHEMA, mode)
    assert json.dumps(SCHEMA, sort_keys=True) == snapshot
    if mode in ("openai_api", "codex"):
        assert compiled["required"] == list(SCHEMA["properties"])
    if mode == "claude_code":
        assert "definitions" in compiled and "$defs" not in compiled


def test_no_matching_union_preserves_invalid_value() -> None:
    bad = {"required": None, "nested": {"wrong": True, "note": None}, "optional": None}
    cleaned = clean_nulls(bad, SCHEMA)
    assert cleaned["nested"] == bad["nested"]


def test_usage_and_decimal_pricing_do_not_double_bill_cache_or_reasoning() -> None:
    u = normalize_usage(
        "openai_api",
        {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 10},
        },
    )
    assert u.input_uncached == 60 and u.reasoning == 10
    amount, source = price("test", u, PRICES)
    assert isinstance(amount, Decimal) and amount == Decimal("0.000164") and source == "price_table"


@pytest.mark.parametrize("mode", ("claude_code", "codex"))
@pytest.mark.asyncio
async def test_cli_sampling_refused_without_spawn(mode: str) -> None:
    runtime = CallRuntime(adapters={mode: ScriptedAdapter()}, pricing=PRICES)
    with pytest.raises(ValueError, match="temperature"):
        await runtime.ask(Binding(mode, "test"), "hello", settings=ModelSettings(temperature=0), call_id="settings")


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["wall", "cancel"])
async def test_tool_execution_shares_call_deadline_and_cancellation(cause: str) -> None:
    from agentkit.models.contracts.types import CallLimits

    entered, unwound = asyncio.Event(), asyncio.Event()

    class BlockingTools:
        def schemas(self):
            return []

        async def execute(self, *args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                unwound.set()

    adapter = ScriptedAdapter(Turn(tool_calls=({"id": "c", "name": "slow", "arguments": {}},)))
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES)
    token = CancellationToken()
    task = asyncio.create_task(
        runtime.run(
            Binding("openai_api", "test"),
            "Do work",
            BlockingTools(),
            call_id="tool-deadline",
            cancel=token,
            limits=CallLimits(wall_clock_s=0.1 if cause == "wall" else None),
        )
    )
    await entered.wait()
    if cause == "cancel":
        token.cancel()
    answer = await asyncio.wait_for(task, 1)
    assert unwound.is_set()
    assert answer.outcome == ("cancelled" if cause == "cancel" else "limit_reached")
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_cumulative_usage_and_cost_are_subtracted_only_once_on_resume() -> None:
    from agentkit.models.contracts.types import SessionHandle

    previous = SessionHandle("claude_code", "previous", reported_cost=Decimal("0.01"))
    claude = ScriptedAdapter(
        Turn(
            ("done",),
            raw_usage={"input_tokens": 10, "output_tokens": 2},
            reported_cost=Decimal("0.013"),
            reported_cost_cumulative=True,
        )
    )
    answer = await CallRuntime(adapters={"claude_code": claude}, pricing=PRICES).ask(
        Binding("claude_code", "test"), "next", resume=previous, call_id="claude-resume"
    )
    assert answer.cost_usd == Decimal("0.003")
    assert answer.usage.input_uncached == 10
    assert answer.session.reported_cost == Decimal("0.013")

    previous = SessionHandle("codex", "previous", raw_usage={"input_tokens": 100, "output_tokens": 10})
    codex = ScriptedAdapter(
        Turn(("done",), raw_usage={"input_tokens": 120, "output_tokens": 15}, usage_cumulative=True)
    )
    answer = await CallRuntime(adapters={"codex": codex}, pricing=PRICES).ask(
        Binding("codex", "test"), "next", resume=previous, call_id="codex-resume"
    )
    assert answer.usage.input_uncached == 20 and answer.usage.output == 5
    assert answer.cost_usd == Decimal("0.000045")


@pytest.mark.asyncio
async def test_retry_keeps_portable_history_and_native_usage_baseline() -> None:
    from agentkit.models.contracts.types import SessionHandle

    history = ({"role": "user", "content": "Remember code 123"},)
    previous = SessionHandle("openai_api", "previous", history=history)
    adapter = ScriptedAdapter(Turn(failure=Failure("transient_network", retryable=True)), Turn(("123",)))
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES, retry=RetryPolicy(base_s=0))
    answer = await runtime.ask(Binding("openai_api", "test"), "Which code?", resume=previous, call_id="retry-history")
    assert answer.outcome == "answered"
    assert adapter.requests[1][2]["session"].history[0] == history[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [(), ("",), ("  ",)])
async def test_empty_transport_turn_is_a_failure(messages) -> None:
    adapter = ScriptedAdapter(Turn(messages))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).ask(
        Binding("openai_api", "test"), "question", call_id="empty"
    )
    assert answer.outcome == "failed" and answer.failure is not None


@pytest.mark.asyncio
async def test_anthropic_stream_cancellation_retains_observed_usage() -> None:
    class Transport:
        async def _stream_events(self, *args):
            yield {"type": "message_start", "message": {"usage": {"input_tokens": 42}}}
            yield {"type": "message_delta", "usage": {"output_tokens": 7}}
            raise asyncio.CancelledError

    async def emit(event):
        pass

    turn = await ApiAdapter()._anthropic(Transport(), {}, emit)
    assert turn.raw_usage == {"input_tokens": 42, "output_tokens": 7}
    assert turn.failure.classification == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_tool_results_continue_in_native_history_and_context(mode: str) -> None:
    from agentkit.tools.execution.policy import ToolResult

    class Tools:
        def schemas(self):
            return []

        async def execute(self, name, args, ctx):
            return ToolResult(({"type": "text", "text": "tool evidence"},), is_error=True)

    wire_call = ({"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "lookup", "input": {}}]},)
    if mode == "openai_api":
        wire_call = ({"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"},)
    adapter = ScriptedAdapter(
        Turn(tool_calls=({"id": "c", "name": "lookup", "arguments": {}},), history=wire_call),
        Turn(failure=Failure("transient_network", retryable=True)),
        Turn(("recovered",)),
    )
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES, retry=RetryPolicy(max_attempts=2, base_s=0))
    answer = await runtime.run(Binding(mode, "test"), "lookup", Tools(), call_id="tool-loop")
    assert answer.outcome == "answered" and answer.attempts == 3
    history = adapter.requests[1][2]["session"].history
    assert "tool evidence" in json.dumps(history)
    assert any(m.role == "tool" and m.metadata["is_error"] for m in answer.context.messages)


@pytest.mark.asyncio
async def test_anthropic_empty_tool_arguments_and_reasoning_signature_survive_stream() -> None:
    class Transport:
        async def _stream_events(self, *args):
            for event in [
                {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "opaque"}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "signed"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "c", "name": "ping"},
                },
                {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
                {"type": "message_stop"},
            ]:
                yield event

    emitted = []
    turn = await ApiAdapter()._anthropic(Transport(), {}, emitted.append)
    assert turn.tool_calls[0]["arguments"] == {}
    assert turn.history[0]["content"][0]["signature"] == "signed"
    assert "opaque" not in repr(emitted)
    assert turn.raw_usage["output_tokens"] == 7


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_clock_s", float("nan")),
        ("wall_clock_s", float("inf")),
        ("max_cost_usd", Decimal("NaN")),
        ("max_cost_usd", Decimal("Infinity")),
    ],
)
def test_nonfinite_call_limits_are_rejected(field, value):
    from agentkit.models.contracts.types import CallLimits

    with pytest.raises(ValueError, match="finite"):
        CallLimits(**{field: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_stream_disconnect_preserves_usage_already_observed(mode):
    class Transport:
        async def _stream_events(self, *args):
            if mode == "openai_api":
                yield {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 42, "output_tokens": 7}, "output": []},
                }
            else:
                yield {"type": "message_start", "message": {"usage": {"input_tokens": 42, "output_tokens": 7}}}
            raise ConnectionError("connection lost")

    adapter = ApiAdapter()
    parser = adapter._openai if mode == "openai_api" else adapter._anthropic
    turn = await parser(Transport(), {}, lambda event: None)
    assert turn.raw_usage == {"input_tokens": 42, "output_tokens": 7}
    assert turn.failure is not None


@pytest.mark.asyncio
async def test_external_task_cancellation_rethrows_and_emits_terminal_event():
    started, unwound = asyncio.Event(), asyncio.Event()

    class Blocking:
        async def turn(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                unwound.set()

    events = []
    runtime = CallRuntime(adapters={"openai_api": Blocking()}, pricing=PRICES)
    task = asyncio.create_task(
        runtime.ask(Binding("openai_api", "test"), "q", call_id="external-cancel", events=events.append)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert unwound.is_set()
    assert events[-1].kind == "call_finished" and events[-1].fields["outcome"] == "cancelled"
    assert events[-1].fields["cost_usd"] is None and events[-1].fields["cost_source"] == "unpriced"


@pytest.mark.asyncio
async def test_cli_snapshot_failure_preserves_answer_and_records_degradation(monkeypatch):
    from agentkit.models.providers.cli.isolation import IsolatedHome

    def unavailable(home):
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(IsolatedHome, "snapshot", unavailable)
    async with CallRuntime(adapters={"claude_code": real_adapter("claude_code", "answer")}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "q",
            call_id="snapshot-failure",
        )
    assert answer.outcome == "answered" and answer.text == "answer"
    assert not answer.session.home_bundle
    assert any(d.setting == "native_session" for d in answer.degradations)
    assert answer.context.messages[-1].content == "answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_schema_correction_is_retained_in_session_and_context(mode):
    adapter = ScriptedAdapter(Turn(("not JSON",)), Turn(('{"answer":42}',)))
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES)
    output = OutputSpec(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    answer = await runtime.ask(Binding(mode, "test"), "q", output, call_id="correction-history")
    assert answer.value == {"answer": 42}
    assert any("Correct these validation errors" in m.content for m in answer.context.messages)
    correction_history = adapter.requests[1][2]["session"].history
    assert "not JSON" in json.dumps(correction_history)
    assert "Correct these validation errors" in json.dumps(correction_history)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["max_turns", "max_output_tokens", "max_cost_usd"])
async def test_call_limits_stop_continuation_before_second_dispatch(limit):
    from agentkit.models.contracts.types import CallLimits

    class Tools:
        def schemas(self):
            return []

        async def execute(self, *args):
            raise AssertionError("budget must stop before tool execution")

    value = Decimal("0.000001") if limit == "max_cost_usd" else 1
    # max_turns permits the first tool, then refuses the next dispatch.
    if limit == "max_turns":
        from agentkit.tools.execution.policy import ToolResult

        async def execute(*args):
            return ToolResult(({"type": "text", "text": "evidence"},))

        Tools.execute = execute
    turn = Turn(
        raw_usage={"input_tokens": 10, "output_tokens": 2}, tool_calls=({"id": "c", "name": "lookup", "arguments": {}},)
    )
    adapter = ScriptedAdapter(turn)
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).run(
        Binding("openai_api", "test"), "q", Tools(), call_id="bounded", limits=CallLimits(**{limit: value})
    )
    assert answer.outcome == "limit_reached"
    assert len(adapter.requests) == 1
    if limit == "max_output_tokens":
        assert adapter.requests[0][2]["settings"].max_output_tokens == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_tools_unavailable_fail_explicitly_and_preserve_usage(mode):
    adapter = ScriptedAdapter(
        Turn(
            raw_usage={"input_tokens": 7, "output_tokens": 1},
            tool_calls=({"id": "c", "name": "unknown", "arguments": {}},),
        )
    )
    answer = await CallRuntime(adapters={mode: adapter}, pricing=PRICES).run(
        Binding(mode, "test"), "q", call_id="missing-tools"
    )
    assert answer.failure.classification == "tool_error"
    assert answer.usage.input_uncached == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal,outcome", [("response.incomplete", "truncated"), ("response.failed", "failed")])
async def test_openai_terminal_status_preserves_usage(terminal, outcome):
    events = [
        {
            "type": terminal,
            "response": {
                "status": "incomplete" if outcome == "truncated" else "failed",
                "error": {"code": "invalid_request_error"},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        }
    ]
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=sse(events))))
    binding = Binding("openai_api", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test"}))
    try:
        answer = await CallRuntime(adapters={"openai_api": ApiAdapter(client=client)}, pricing=PRICES).ask(
            binding, "q", call_id="partial"
        )
        assert answer.outcome == outcome
        assert answer.usage.output == 3 and answer.usage.input_uncached == 7
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("usage,complete", [({}, True), ({"input_tokens": 7}, False)])
async def test_missing_or_incomplete_accounting_is_unknown_instead_of_free(usage, complete):
    events = []
    adapter = ScriptedAdapter(Turn(("answer",), raw_usage=usage, usage_complete=complete))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).ask(
        Binding("openai_api", "test"), "q", call_id="unknown-cost", events=events.append
    )
    assert answer.outcome == "answered" and answer.text == "answer"
    assert answer.cost_usd is None and answer.cost_source == "unpriced"
    assert answer.usage.input_uncached == usage.get("input_tokens", 0)
    assert events[-1].fields["cost_usd"] is None


@pytest.mark.asyncio
async def test_unknown_accounting_stops_a_dollar_bounded_continuation():
    from agentkit.models.contracts.types import CallLimits

    class Tools:
        def schemas(self):
            return []

        async def execute(self, *args):
            raise AssertionError("unknown spend cannot permit another operation")

    adapter = ScriptedAdapter(Turn(tool_calls=({"id": "c", "name": "lookup", "arguments": {}},)))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).run(
        Binding("openai_api", "test"),
        "q",
        Tools(),
        call_id="unknown-spend",
        limits=CallLimits(max_cost_usd=Decimal("1")),
    )
    assert answer.outcome == "limit_reached" and answer.cost_usd is None


@pytest.mark.asyncio
async def test_explicit_zero_usage_is_known_zero_cost():
    adapter = ScriptedAdapter(Turn(("answer",), raw_usage={"input_tokens": 0, "output_tokens": 0}))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).ask(
        Binding("openai_api", "test"), "q", call_id="zero-cost"
    )
    assert answer.cost_usd == Decimal(0) and answer.cost_source == "price_table"


@pytest.mark.asyncio
async def test_codex_native_deny_refuses_before_dispatch():
    from agentkit.models.contracts.constants import NativeToolPolicy

    adapter = ScriptedAdapter(Turn(("must not run",)))
    answer = await CallRuntime(
        adapters={"codex": adapter}, pricing=PRICES, native_tool_policy=NativeToolPolicy.DENY
    ).ask(Binding("codex", "test"), "q", call_id="native-deny")
    assert answer.outcome == "failed" and answer.failure.classification == "invalid_request"
    assert not adapter.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["warn", "allow"])
async def test_codex_native_warn_reduces_capabilities_and_reports_limit(policy):
    import tomllib

    from agentkit.tools import FunctionTool, ToolRegistry
    from agentkit.tools.execution.policy import ToolSet

    async def read_host_evidence() -> str:
        """Read evidence explicitly granted by the application host."""
        return "host evidence"

    tools = ToolSet(ToolRegistry.from_tools([FunctionTool.from_callable(read_host_evidence, side_effecting=False)]))
    adapter = real_adapter("codex", "done")
    adapter.native_tool_policy = policy
    binding = Binding("codex", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test"}))
    async with CallRuntime(adapters={"codex": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.run(binding, "q", tools, call_id="native-policy")
    assert answer.outcome == "answered"
    argv = adapter.spawn.argv()
    assert ("features.shell_tool=false" in argv) == (policy == "warn")
    assert ("features.unified_exec=false" in argv) == (policy == "warn")
    for feature in ("apps", "plugins", "skill_mcp_dependency_install"):
        assert (f"features.{feature}=false" in argv) == (policy == "warn")
    assert ('web_search="disabled"' in argv) == (policy == "warn")
    assert any(argument.startswith("mcp_servers.agentkit.url=") for argument in argv)
    hosted = next(argument for argument in argv if argument.startswith("mcp_servers.agentkit.tools="))
    assert tomllib.loads(hosted)["mcp_servers"]["agentkit"]["tools"] == {
        "read_host_evidence": {"approval_mode": "approve"},
    }
    assert any(d.setting == "native_tools" for d in answer.degradations) == (policy == "warn")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_cancelled_tool_batch_closes_pending_pairs_and_retains_confirmed_results(mode):
    from agentkit.tools.execution.policy import ToolResult

    entered, cleaned = asyncio.Event(), asyncio.Event()
    calls = tuple({"id": name, "name": name, "arguments": {}} for name in ["first", "blocked", "last"])
    adapter = ScriptedAdapter(
        Turn(tool_calls=calls, raw_usage={"input_tokens": 3, "output_tokens": 1}), Turn(("resumed",))
    )

    class Tools:
        def schemas(self):
            return []

        async def execute(self, name, *args):
            if name == "first":
                return ToolResult(({"type": "text", "text": "confirmed evidence"},))
            assert name == "blocked"
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    token = CancellationToken()
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES)
    task = asyncio.create_task(
        runtime.run(Binding(mode, "test"), "diagnose", Tools(), call_id="blocked-batch", cancel=token)
    )
    await entered.wait()
    token.cancel()
    interrupted = await task
    assert interrupted.outcome == "cancelled" and cleaned.is_set()
    results = {m.tool_call_id: m for m in interrupted.session.messages if m.role == "tool"}
    assert results["first"].content == "confirmed evidence"
    assert results["blocked"].metadata["interrupted"] and results["blocked"].metadata["is_error"]
    assert results["last"].metadata["not_executed"] and results["last"].metadata["is_error"]
    assert "confirmed evidence" in json.dumps(interrupted.session.history)
    resumed = await runtime.ask(
        Binding(mode, "test"), "Resume safely", resume=interrupted.session, call_id="safe-resume"
    )
    assert resumed.outcome == "answered"
    history = adapter.requests[-1][2]["session"].history
    if mode == "openai_api":
        assert {x["call_id"] for x in history if x.get("type") == "function_call"} == {
            x["call_id"] for x in history if x.get("type") == "function_call_output"
        }
    else:
        call_ids = {
            b["id"]
            for m in history
            for b in m.get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        result_ids = {
            b["tool_use_id"]
            for m in history
            for b in m.get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        assert call_ids == result_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_turn_limit_retains_latest_result_in_both_session_axes(mode):
    from agentkit.models.contracts.types import CallLimits
    from agentkit.tools.execution.policy import ToolResult

    class Tools:
        def schemas(self):
            return []

        async def execute(self, *args):
            return ToolResult(({"type": "text", "text": "durable diagnostic evidence"},))

    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "c", "name": "diagnose", "arguments": {}},),
            raw_usage={"input_tokens": 3, "output_tokens": 1},
        )
    )
    answer = await CallRuntime(adapters={mode: adapter}, pricing=PRICES).run(
        Binding(mode, "test"), "diagnose", Tools(), call_id="one-turn", limits=CallLimits(max_turns=1)
    )
    assert answer.outcome == "limit_reached"
    assert answer.session.messages[-1].content == "durable diagnostic evidence"
    assert "durable diagnostic evidence" in json.dumps(answer.session.history)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["codex", "claude_code"])
async def test_cli_resume_floors_only_compact_budget_boundary(mode, monkeypatch):
    """Exact provider costs may leave a finer-than-microdollar invocation cap."""
    from agentkit.core import runtime as runtime_exports
    from agentkit.core.contracts.types import Message
    from agentkit.models.contracts.types import CallLimits
    from agentkit.state.context import WorkingContext

    ceilings = []
    original_budget = runtime_exports.Budget

    def capture_budget(*args, **kwargs):
        ceilings.append(kwargs.get("max_cost_usd"))
        return original_budget(*args, **kwargs)

    monkeypatch.setattr(runtime_exports, "Budget", capture_budget)
    if mode == "codex":
        cli = FakeCodexCli.script(codex_turn(text="submitted", usage=(10, 0, 5)), repeat_last=True)
    else:
        cli = FakeClaudeCli.script(
            [
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "submitted",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "session_id": "precision-session",
                    "total_cost_usd": 0.0363984,
                },
            ],
            repeat_last=True,
        )
    adapter = CliAdapter(spawn=cli)
    context = WorkingContext()
    remaining = Decimal(".9136016")
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        first = await runtime.run(
            Binding(mode, "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "offline-fake-key"})),
            "Repair source",
            context=context,
            limits=CallLimits(max_cost_usd=Decimal(".95")),
            call_id="precision",
        )
        context.append(Message("user", "Private verifier rejected the initial submission; recover."))
        second = await runtime.run(
            Binding(mode, "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "offline-fake-key"})),
            "Repair source",
            context=context,
            resume=first.session,
            limits=CallLimits(max_cost_usd=remaining),
            call_id="precision/verify-1",
        )
    assert first.outcome == second.outcome == "answered"
    assert len(cli.invocations) == 2
    assert ceilings == [Decimal(".950000"), Decimal(".913601")]
    assert ceilings[1] <= remaining and remaining == Decimal(".9136016")
    assert second.cost_usd == (Decimal(".0363984") if mode == "claude_code" else Decimal(".000035"))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_fast_cooperative_tool_abort_beats_monitor_poll_and_retains_context(mode):
    """A joined tool can raise cooperative cancellation before the 50ms monitor observes the token."""
    from agentkit.core.contracts.types import Message, ToolCall
    from agentkit.core.execution.concurrency import Cancelled

    token = CancellationToken()
    cleaned = asyncio.Event()
    if mode == "openai_api":
        history = ({"type": "function_call", "call_id": "fast-tool", "name": "sample", "arguments": "{}"},)
    else:
        history = (
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "fast-tool", "name": "sample", "input": {}},
                ],
            },
        )
    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "fast-tool", "name": "sample", "arguments": {}},),
            canonical_messages=(Message("assistant", tool_calls=(ToolCall("fast-tool", "sample", {}),)),),
            history=history,
            raw_usage={"input_tokens": 10, "output_tokens": 2},
        )
    )

    class Tools:
        def schemas(self):
            return []

        async def execute(self, name, arguments, ctx):
            try:
                token.cancel()
                raise Cancelled("fast cooperative tool abort")
            finally:
                cleaned.set()

    events = []
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.run(
            Binding(mode, "test"),
            "Sample diagnostic",
            Tools(),
            cancel=token,
            events=events.append,
            call_id="fast-cooperative",
        )
    assert answer.outcome == "cancelled" and answer.failure.classification == "cancelled"
    assert cleaned.is_set() and len(adapter.requests) == 1
    assert answer.usage.input_uncached == 10 and answer.usage.output == 2
    assert answer.cost_usd == Decimal(".000020")
    result = next(message for message in answer.session.messages if message.role == "tool")
    assert result.tool_call_id == "fast-tool" and result.metadata["is_error"]
    assert "interrupted" in result.content.lower()
    assert events[-1].kind == "call_finished" and events[-1].fields["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_fast_cooperative_transport_abort_is_cancelled_and_unpriced():
    from agentkit.core.execution.concurrency import Cancelled

    token = CancellationToken()

    class Adapter:
        async def turn(self, *args, **kwargs):
            token.cancel()
            raise Cancelled("transport observed the token before monitor polling")

    async with CallRuntime(adapters={"openai_api": Adapter()}, pricing=PRICES) as runtime:
        answer = await runtime.run(Binding("openai_api", "test"), "Query", cancel=token, call_id="transport-abort")
    assert answer.outcome == "cancelled" and answer.failure.classification == "cancelled"
    assert answer.cost_usd is None and answer.cost_source == "unpriced"
