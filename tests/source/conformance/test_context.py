import json

import pytest

from agentkit.core.contracts.types import Message, MessageRole, Mode, ToolCall
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.adapters.messages import canonical_turn, history_for
from agentkit.models.contracts.constants import EventKind
from agentkit.models.contracts.types import Binding, Turn
from agentkit.state.context import WorkingContext
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.asyncio
async def test_context_is_attached_to_messages_agent_and_immutable_result_snapshot():
    context = WorkingContext()
    context.note("agent_id", "writer")
    adapter = ScriptedAdapter(Turn(("42",)))
    runtime = CallRuntime(adapters={Mode.OPENAI_API: adapter}, pricing=PRICES)
    answer = await runtime.ask(Binding(Mode.OPENAI_API, "test"), "question", context=context, call_id="call")
    assert answer.context.messages == tuple(context.messages)
    assert context.messages[0].metadata["agent_id"] == "writer"
    assert answer.session.messages == tuple(context.messages)
    context.append(Message(MessageRole.USER, "next"))
    assert len(answer.context.messages) == 2


@pytest.mark.asyncio
async def test_vendor_change_replays_conversation_and_records_reasoning_degradation():
    adapter = ScriptedAdapter(Turn(("first",)), Turn(("second",)))
    runtime = CallRuntime(adapters={Mode.OPENAI_API: adapter, Mode.ANTHROPIC_API: adapter}, pricing=PRICES)
    first = await runtime.ask(Binding(Mode.OPENAI_API, "test"), "Remember code 731", call_id="one")
    events = []
    second = await runtime.ask(
        Binding(Mode.ANTHROPIC_API, "test"), "What code?", resume=first.session, call_id="two", events=events.append
    )
    assert second.outcome == "answered"
    sent = adapter.requests[1][2]["session"]
    assert sent.mode == Mode.ANTHROPIC_API and sent.cli_session_id is None
    assert "731" in json.dumps(sent.history)
    assert any(e.kind == EventKind.DEGRADED for e in events)
    assert second.degradations[0].setting == "continuation"


def test_tool_and_image_messages_translate_without_raw_reasoning():
    messages = [
        Message(MessageRole.USER, "question"),
        Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("c", "image", {}),)),
        Message(
            MessageRole.TOOL,
            tool_call_id="c",
            content_blocks=({"type": "image", "mimeType": "image/png", "data": "eA=="},),
        ),
    ]
    anth = history_for(messages, Mode.ANTHROPIC_API)
    openai = history_for(messages, Mode.OPENAI_API)
    assert anth[-1]["content"][0]["content"][0]["type"] == "image"
    assert openai[-1]["output"][0]["type"] == "input_image"
    visible = canonical_turn(
        [
            {"role": "user", "content": "q"},
            {"type": "reasoning", "encrypted_content": "opaque"},
            {"type": "function_call", "call_id": "c", "name": "image", "arguments": "{}"},
        ],
        Mode.OPENAI_API,
    )
    assert visible[0].tool_calls[0].id == "c"
    assert "opaque" not in repr(visible)


def test_message_payloads_cannot_be_mutated_after_publication():
    blocks = ({"type": "text", "text": "hello"},)
    msg = Message(MessageRole.USER, content_blocks=blocks, metadata={"source": ["ref"]})
    blocks[0]["text"] = "changed"
    assert msg.content_blocks[0]["text"] == "hello"
    with pytest.raises(TypeError):
        msg.metadata["source"].append("evil")


@pytest.mark.asyncio
async def test_context_budget_rejects_before_transport_and_counts_prefix():
    from agentkit.state.context import PrefixContext

    context = WorkingContext(prefix=PrefixContext("system " * 100), limit=5)
    adapter = ScriptedAdapter(Turn(("must not run",)))
    runtime = CallRuntime(adapters={Mode.OPENAI_API: adapter}, pricing=PRICES)
    answer = await runtime.ask(Binding(Mode.OPENAI_API, "test"), "q", context=context, call_id="context-limit")
    assert answer.outcome == "limit_reached"
    assert answer.failure.mode_detail["limit"] == "context"
    assert not adapter.requests


@pytest.mark.asyncio
async def test_resume_preserves_prefix_and_grounding_when_switching_models():
    from agentkit.state.context import PrefixContext

    context = WorkingContext(
        prefix=PrefixContext("Keep a concise answer", (Message(MessageRole.USER, "Account code is 731"),), "Use JSON")
    )
    adapter = ScriptedAdapter(Turn(("first",)), Turn(("second",)))
    runtime = CallRuntime(adapters={Mode.OPENAI_API: adapter, Mode.ANTHROPIC_API: adapter}, pricing=PRICES)
    first = await runtime.ask(Binding(Mode.OPENAI_API, "test"), "Remember", context=context, call_id="prefix-one")
    second = await runtime.ask(
        Binding(Mode.ANTHROPIC_API, "test"), "Recall", resume=first.session, call_id="prefix-two"
    )
    sent = adapter.requests[1]
    assert "Keep a concise answer" in sent[1].system and "Use JSON" in sent[1].system
    assert "731" in json.dumps(sent[2]["session"].history)
    assert second.context.prefix == context.prefix


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API, Mode.CODEX])
async def test_same_model_resume_replays_caller_context_changes(mode):
    from agentkit.state.context import PrefixContext

    context = WorkingContext(prefix=PrefixContext("Original rules"))
    adapter = ScriptedAdapter(Turn(("first",), session_id="native-before"), Turn(("second",)))
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES)
    first = await runtime.ask(Binding(mode, "test"), "Remember", context=context, call_id="same-one")
    context.append(Message(MessageRole.USER, "New caller fact 812"))
    context.prefix = PrefixContext("Original rules", (Message(MessageRole.USER, "Fresh grounding 937"),))
    second = await runtime.ask(
        Binding(mode, "test"), "Recall", resume=first.session, context=context, call_id="same-two"
    )
    sent = adapter.requests[1][2]["session"]
    assert "812" in json.dumps(sent.history) and "937" in json.dumps(sent.history)
    assert sent.cli_session_id is None
    assert sent.prefix == context.prefix
    assert any(d.setting == "context_replay" for d in second.degradations)


def test_anthropic_tool_error_metadata_survives_portable_replay():
    message = Message(MessageRole.TOOL, "failed", tool_call_id="c", metadata={"is_error": True})
    assert history_for([message], Mode.ANTHROPIC_API)[0]["content"][0]["is_error"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_cli_host_tool_evidence_survives_api_switch(monkeypatch, target):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from agentkit.models.contracts.types import CredentialRef
    from agentkit.tools.execution.policy import ToolResult
    from tests.source.conformance.test_calls import real_adapter

    @asynccontextmanager
    async def fake_serve(tools, *, ctx, execution_observer, timeout_s=None, activity_observer=None):
        if activity_observer is not None:
            activity_observer(True)
        execution_observer(
            "lookup",
            {"code": "812"},
            ToolResult(
                (
                    {"type": "text", "text": "tool evidence 937"},
                    {"type": "image", "mimeType": "image/png", "data": "eA=="},
                ),
                is_error=True,
            ),
        )
        if activity_observer is not None:
            activity_observer(False)
        yield None

    monkeypatch.setattr("agentkit.tools.execution.policy.serve", fake_serve)
    api = ScriptedAdapter(Turn(("recalled",)))
    binding = Binding(Mode.CLAUDE_CODE, "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"}))
    async with CallRuntime(
        adapters={Mode.CLAUDE_CODE: real_adapter("claude_code", "done"), target: api}, pricing=PRICES
    ) as runtime:
        first = await runtime.run(binding, "lookup", SimpleNamespace(schemas=lambda: []), call_id="cli-tools")
        second = await runtime.ask(Binding(target, "test"), "recall", resume=first.session, call_id="api-tools")
    assert first.outcome == "answered" and second.outcome == "answered"
    evidence = next(m for m in first.context.messages if m.role == MessageRole.TOOL)
    assert evidence.content_blocks[1]["type"] == "image" and evidence.metadata["is_error"]
    history = api.requests[0][2]["session"].history
    encoded = json.dumps(history)
    assert "937" in encoded and "812" in encoded and "eA==" in encoded
    if target == Mode.ANTHROPIC_API:
        assert any(b.get("is_error") for m in history for b in m.get("content", []) if isinstance(b, dict))


def test_anthropic_replay_omits_blank_text_without_losing_tool_only_turns():
    messages = (
        Message(MessageRole.USER, "Inspect"),
        Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("receipt-1", "inspect", {"id": "812"}),)),
        Message(MessageRole.TOOL, "", tool_call_id="receipt-1", metadata={"is_error": True}),
        Message(MessageRole.ASSISTANT, "done"),
    )
    history = history_for(messages, Mode.ANTHROPIC_API)
    assert history[1]["content"] == [{"type": "tool_use", "id": "receipt-1", "name": "inspect", "input": {"id": "812"}}]
    assert history[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "receipt-1", "content": [], "is_error": True}
    ]
    assert history[3]["content"] == [{"type": "text", "text": "done"}]


def test_anthropic_opaque_history_normalization_preserves_signatures_images_and_result_ids():
    from agentkit.models.adapters.messages import anthropic_history

    history = (
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
                {"type": "text", "text": ""},
                {"type": "tool_use", "id": "c", "name": "chart", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c",
                    "is_error": True,
                    "content": [
                        {"type": "text", "text": "  "},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "eA=="}},
                    ],
                }
            ],
        },
    )
    encoded_before = json.dumps(history)
    normalized = anthropic_history(history)
    assert normalized[1]["content"][0] == history[1]["content"][0]
    assert normalized[1]["content"][1] == history[1]["content"][2]
    result = normalized[2]["content"][0]
    assert result["tool_use_id"] == "c" and result["is_error"]
    assert result["content"] == [history[2]["content"][0]["content"][1]]
    assert json.dumps(history) == encoded_before


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [Mode.CODEX, Mode.CLAUDE_CODE, Mode.OPENAI_API])
async def test_tool_evidence_handoff_serializes_valid_anthropic_request_without_active_tools(source):
    import httpx

    from agentkit.models.contracts.types import CredentialRef
    from agentkit.models.providers.frontiers.calls import ApiAdapter

    observed = []

    def handler(request):
        payload = json.loads(request.content)
        observed.append(payload)
        # Reproduce the documented provider rejection at the actual HTTP seam.
        blank = any(
            b.get("type") == "text" and not b.get("text", "").strip()
            for message in payload["messages"]
            for b in message.get("content", [])
            if isinstance(b, dict)
        )
        if blank:
            return httpx.Response(
                400,
                json={"error": {"type": "invalid_request_error", "message": "text content blocks must be non-empty"}},
            )
        frames = (
            {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "reviewed"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        )
        return httpx.Response(200, text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames))

    evidence = (
        Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("host-1", "read_source", {"id": "primary"}),)),
        Message(MessageRole.TOOL, "Primary source evidence 937", tool_call_id="host-1", metadata={"is_error": False}),
    )
    first_adapter = ScriptedAdapter(Turn(("research complete",), canonical_messages=evidence))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        runtime = CallRuntime(
            adapters={source: first_adapter, Mode.ANTHROPIC_API: ApiAdapter(client=client)}, pricing=PRICES
        )
        context = WorkingContext()
        if source == Mode.OPENAI_API:
            context.append(Message(MessageRole.USER, "Earlier investigation"))
            context.extend(evidence)
        first = await runtime.ask(Binding(source, "test"), "Research", context=context, call_id="research")
        final = await runtime.ask(
            Binding(Mode.ANTHROPIC_API, "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "synthetic"})),
            "Review evidence",
            resume=first.session,
            call_id="handoff",
        )
    assert final.outcome == "answered" and final.text == "reviewed"
    payload = observed[0]
    assert "tools" not in payload  # No previously granted tools are re-advertised.
    calls = [
        b
        for m in payload["messages"]
        for b in m.get("content", [])
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    results = [
        b
        for m in payload["messages"]
        for b in m.get("content", [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert calls == [{"type": "tool_use", "id": "host-1", "name": "read_source", "input": {"id": "primary"}}]
    assert results[0]["tool_use_id"] == "host-1"
    assert results[0]["content"] == [{"type": "text", "text": "Primary source evidence 937"}]
