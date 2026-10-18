"""One raw-byte image result across SDK and actual local MCP transports.

HTTP provider responses are scripted; MCP uses real authenticated loopback HTTP.
No provider credentials, paid requests, or CLI containment claims.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace

import httpx
import pytest

from agentkit.core import Mode
from agentkit.core.contracts.types import Message, MessageRole
from agentkit.core.serialization.messages import message_from_dict, message_to_dict
from agentkit.core.testing import make_test_ctx
from agentkit.integrations.mcp.serve import _tool_result
from agentkit.models.adapters.messages import history_for, tool_content
from agentkit.tools import ToolExecutionError, ToolRegistry, tool
from agentkit.tools.execution.policy import OverflowPolicy, ToolResult, ToolSet, _mcp_result, serve

# A complete 1x1 PNG, not merely a signature or arbitrary synthetic bytes.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


def picture(*, is_error=False, label="chart"):
    return ToolResult(
        ({"type": "text", "text": label}, {"type": "image", "mimeType": "image/png", "data": PNG}),
        structured={},
        is_error=is_error,
    )


@pytest.mark.parametrize("is_error", [False, True])
@pytest.mark.parametrize("boundary", ["openai_api", "anthropic_api", "mcp", "toolset_mcp"])
def test_same_raw_result_decodes_at_every_boundary(boundary, is_error):
    value = picture(is_error=is_error)
    if boundary in ("mcp", "toolset_mcp"):
        result = _tool_result(value) if boundary == "mcp" else _mcp_result("picture", value)
        wire = json.loads(result.model_dump_json(by_alias=True))
        assert wire["isError"] is is_error
        assert wire["structuredContent"] == {}
        blocks = wire["content"]
        assert blocks[0]["text"] == "chart"
        assert blocks[1]["mimeType"] == "image/png"
        encoded = blocks[1]["data"]
    else:
        blocks = json.loads(json.dumps(tool_content(value.content, Mode(boundary))))
        assert blocks[0]["text"] == "chart"
        image = blocks[1]
        if boundary == "openai_api":
            assert image["image_url"].startswith("data:image/png;base64,")
            encoded = image["image_url"].split(",", 1)[1]
        else:
            assert image["source"]["media_type"] == "image/png"
            encoded = image["source"]["data"]
    assert base64.b64decode(encoded, validate=True) == PNG
    assert value.content[1]["data"] == PNG


@pytest.mark.parametrize("defect", ["base64", "empty", "mime", "missing", "mutable"])
@pytest.mark.parametrize("boundary", ["openai_api", "anthropic_api", "mcp", "toolset_mcp"])
def test_malformed_images_are_explicit_safe_refusals(defect, boundary):
    block = dict(picture().content[1])
    if defect == "base64":
        block["data"] = "private-preencoded-data"
    elif defect == "empty":
        block["data"] = b""
    elif defect == "mime":
        block["mimeType"] = "private-invalid-mime"
    elif defect == "missing":
        del block["mimeType"]
    else:
        block["data"] = bytearray(PNG)
    value = ToolResult((block,))
    if boundary in ("mcp", "toolset_mcp"):
        result = _tool_result(value) if boundary == "mcp" else _mcp_result("picture", value)
        assert result.isError
        assert "invalid image block" in result.content[0].text
        assert "private" not in result.model_dump_json()
    else:
        with pytest.raises(ValueError, match="invalid image block") as caught:
            tool_content(value.content, Mode(boundary))
        assert "private" not in str(caught.value)


@pytest.mark.asyncio
async def test_encoded_size_exact_boundary_and_no_image_overflow_file(tmp_path):
    value = picture(label="é" * 100)
    encoded = [dict(value.content[0]), {**value.content[1], "data": base64.b64encode(PNG).decode("ascii")}]
    size = len(json.dumps({"content": encoded, "structuredContent": {}}, ensure_ascii=False).encode())
    tools = ToolSet(ToolRegistry(), overflow=OverflowPolicy(size, tmp_path))
    assert await tools.handle("picture", value) == value
    limited = replace(tools, overflow=OverflowPolicy(size - 1, tmp_path))
    with pytest.raises(ToolExecutionError, match="image result exceeds") as caught:
        await limited.handle("picture", value)
    assert caught.value.code == "tool_output_limit"
    assert not list(tmp_path.iterdir())


def test_image_message_durable_json_roundtrip_and_cross_provider_history():
    message = Message(
        MessageRole.TOOL, tool_call_id="provider-id", content_blocks=picture().content, metadata={"is_error": True}
    )
    restored = message_from_dict(json.loads(json.dumps(message_to_dict(message))))
    assert restored == message
    for mode in (Mode.OPENAI_API, Mode.ANTHROPIC_API):
        assert "provider-id" in json.dumps(history_for([restored], mode))


@pytest.mark.asyncio
@pytest.mark.parametrize("result_kind", ["image", "malformed", "oversized"])
async def test_real_local_mcp_image_concurrency_and_cross_call_refusal(result_kind, tmp_path):
    from agentkit.tools.execution.context import current_tool_invocation

    identities = []

    @tool(side_effecting=False)
    def plot(label: str) -> ToolResult:
        """Return the requested call-scoped image and label."""
        identities.append(current_tool_invocation())
        if result_kind == "malformed":
            return ToolResult(({"type": "image", "data": "private-invalid", "mimeType": "image/png"},))
        return picture(is_error=True, label=label if result_kind == "image" else "x" * 1000)

    left = serve(ToolSet(ToolRegistry.from_tools([plot]), overflow=OverflowPolicy(512, tmp_path)), ctx=make_test_ctx())
    right = serve(ToolSet(ToolRegistry.from_tools([plot]), overflow=OverflowPolicy(512, tmp_path)), ctx=make_test_ctx())
    async with left, right, httpx.AsyncClient(trust_env=False) as client:

        async def invoke(spec, label):
            request = {
                "jsonrpc": "2.0",
                "id": label,
                "method": "tools/call",
                "params": {
                    "name": "plot",
                    "arguments": {"label": label},
                    "_meta": {"claudecode/toolUseId": "provider-" + label},
                },
            }
            headers = {**spec.auth_headers, "accept": "application/json, text/event-stream"}
            response = await client.post(spec.url, headers=headers, json=request)
            assert response.status_code == 200
            result = response.json()["result"]
            assert result["isError"]
            if result_kind == "image":
                assert result["content"][0]["text"] == label
                assert base64.b64decode(result["content"][1]["data"], validate=True) == PNG
            else:
                assert len(result["content"]) == 1 and result["content"][0]["type"] == "text"
                assert result["structuredContent"]["code"] == (
                    "invalid_tool_content" if result_kind == "malformed" else "tool_output_limit"
                )
                assert "private" not in response.text and "saved to" not in response.text
            refused = await client.post(
                spec.url, headers={**headers, **(right if spec is left else left).auth_headers}, json=request
            )
            assert refused.status_code == 401

        await asyncio.gather(invoke(left, "left"), invoke(right, "right"))
    assert not list(tmp_path.iterdir())
    assert len(identities) == 2
    assert {x.provider_tool_call_id for x in identities} == {"provider-left", "provider-right"}
    assert {x.transport_request_id for x in identities} == {"left", "right"}
    assert len({x.host_invocation_id for x in identities}) == 2


def test_clean_png_has_valid_chunks_and_a_real_pixel():
    import struct
    import zlib

    assert PNG[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    chunks = {}
    while offset < len(PNG):
        size = struct.unpack(">I", PNG[offset : offset + 4])[0]
        kind, data = PNG[offset + 4 : offset + 8], PNG[offset + 8 : offset + 8 + size]
        assert struct.unpack(">I", PNG[offset + 8 + size : offset + 12 + size])[0] == zlib.crc32(kind + data)
        chunks[kind] = data
        offset += size + 12
    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (1, 1, 8, 6, 0, 0, 0)
    assert zlib.decompress(chunks[b"IDAT"]) == b"\0\xff\0\0\xff"
    assert chunks[b"IEND"] == b""


def frames(mode, *, first):
    if mode == Mode.OPENAI_API:
        output = (
            [{"type": "function_call", "call_id": "provider-image", "name": "plot", "arguments": "{}"}]
            if first
            else [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]
        )
        return [
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": output, "usage": {"input_tokens": 3, "output_tokens": 2}},
            }
        ]
    block = (
        {"type": "tool_use", "id": "provider-image", "name": "plot", "input": {}}
        if first
        else {"type": "text", "text": ""}
    )
    delta = {"type": "input_json_delta", "partial_json": "{}"} if first else {"type": "text_delta", "text": "done"}
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0, "content_block": block},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use" if first else "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
@pytest.mark.parametrize("result_kind", ["image", "image_error", "malformed", "oversized"])
async def test_actual_api_adapters_with_scripted_http_and_shared_runtime(mode, result_kind, tmp_path):
    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.catalog.pricing import PriceTable
    from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef
    from agentkit.models.providers.frontiers.calls import ApiAdapter
    from agentkit.tools.execution.context import current_tool_invocation

    requests, identities, events = [], [], []
    value = picture(is_error=result_kind == "image_error")
    if result_kind == "malformed":
        value = ToolResult(({"type": "image", "data": "private-base64", "mimeType": "image/png"},))
    elif result_kind == "oversized":
        value = picture(label="x" * 1000)

    @tool(side_effecting=False)
    def plot() -> ToolResult:
        """Return a host-controlled image for visual analysis."""
        identities.append(current_tool_invocation())
        return value

    def respond(request):
        requests.append(json.loads(request.content))
        body = b"".join(
            b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames(mode, first=len(requests) == 1)
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    pricing = PriceTable("image-test", "2026-10-18", {"test": {"input_uncached": "1", "output": "1"}})
    key = "ANTHROPIC_API_KEY" if mode == Mode.ANTHROPIC_API else "OPENAI_API_KEY"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as client:
        async with CallRuntime(adapters={mode: ApiAdapter(client=client)}, pricing=pricing) as runtime:
            answer = await runtime.run(
                Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-key"})),
                "Use plot to read the image.",
                ToolSet(ToolRegistry.from_tools([plot]), overflow=OverflowPolicy(512, tmp_path)),
                call_id="image-call",
                limits=CallLimits(max_turns=3),
                events=events.append,
            )
    assert answer.outcome == "answered"
    assert len(requests) == 2 and answer.attempts == 2
    assert len(identities) == 1
    assert identities[0].provider_tool_call_id == "provider-image"
    assert identities[0].transport_request_id is None
    assert identities[0].host_invocation_id != "provider-image"
    assert identities[0].call_id == "image-call"
    failed = result_kind in ("malformed", "oversized", "image_error")
    result_event = next(event for event in events if event.kind == "tool_result")
    assert result_event.fields["is_error"] is failed
    if mode == Mode.OPENAI_API:
        result = next(item for item in requests[1]["input"] if item.get("type") == "function_call_output")
        assert result["call_id"] == "provider-image"
        blocks = result["output"]
    else:
        result = next(
            block
            for item in requests[1]["messages"]
            if isinstance(item["content"], list)
            for block in item["content"]
            if block.get("type") == "tool_result"
        )
        assert result["tool_use_id"] == "provider-image" and result["is_error"] is failed
        blocks = result["content"]
    if result_kind in ("image", "image_error"):
        assert blocks[0]["text"] == "chart"
        encoded = blocks[1]["image_url"].split(",", 1)[1] if mode == Mode.OPENAI_API else blocks[1]["source"]["data"]
        assert base64.b64decode(encoded, validate=True) == PNG
    else:
        text = blocks if isinstance(blocks, str) else blocks[0]["text"]
        assert "invalid content" in text if result_kind == "malformed" else "image result exceeds" in text
        assert "private" not in text and "saved to" not in text
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("boundary", ["openai_api", "anthropic_api", "mcp", "toolset_mcp"])
@pytest.mark.parametrize("over", [False, True])
def test_raw_image_limit_has_a_clean_exact_boundary_twin(boundary, over):
    payload = b"x" * (20 * 1024 * 1024 + int(over))
    value = ToolResult(({"type": "image", "mimeType": "image/png", "data": payload},))
    # Synthetic bytes exercise the size guard only, not image validity.
    if boundary in ("mcp", "toolset_mcp"):
        result = _tool_result(value) if boundary == "mcp" else _mcp_result("image", value)
        assert result.isError is over
        if over:
            assert "20 MiB" in result.content[0].text
        else:
            assert base64.b64decode(result.content[0].data, validate=True) == payload
    elif over:
        with pytest.raises(ValueError, match="20 MiB"):
            tool_content(value.content, Mode(boundary))
    else:
        block = tool_content(value.content, Mode(boundary))[0]
        encoded = block["source"]["data"] if boundary == "anthropic_api" else block["image_url"].split(",", 1)[1]
        assert base64.b64decode(encoded, validate=True) == payload


@pytest.mark.parametrize("data", ["!not-base64!", "", 42, "eA=="])
def test_corrupt_or_invalid_transcript_image_refused_without_data_leak(data):
    with pytest.raises(ValueError, match="invalid image block"):
        message_from_dict(
            {"role": "tool", "content_blocks": [{"type": "image", "mimeType": "image/private", "data": data}]}
        )


@pytest.mark.asyncio
async def test_image_replay_bound_counts_base64_bytes():
    from agentkit.state.context import WorkingContext
    from agentkit.state.context.replay import prepare_bounded_replay

    message = Message(MessageRole.TOOL, content_blocks=picture().content, tool_call_id="provider-image")
    context = WorkingContext(messages=[message])
    size = len(json.dumps([message_to_dict(message)], ensure_ascii=False, separators=(",", ":")).encode())
    clean = await prepare_bounded_replay(context, max_bytes=size)
    planted = await prepare_bounded_replay(context, max_bytes=size - 1)
    assert clean.byte_size == planted.byte_size == size
    assert not clean.over_limit and planted.over_limit


@pytest.mark.parametrize("boundary", ["openai_api", "anthropic_api", "mcp", "toolset_mcp"])
def test_raw_bytes_in_non_image_blocks_are_refused_not_stringified(boundary):
    value = ToolResult(({"type": "unknown", "data": b"private-tool-bytes"},))
    if boundary in ("mcp", "toolset_mcp"):
        result = _tool_result(value) if boundary == "mcp" else _mcp_result("image", value)
        assert result.isError
        assert "private" not in result.model_dump_json()
    else:
        with pytest.raises(ValueError, match="invalid content") as caught:
            tool_content(value.content, Mode(boundary))
        assert "private" not in str(caught.value)
