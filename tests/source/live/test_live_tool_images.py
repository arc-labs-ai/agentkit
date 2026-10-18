"""Opt-in real API image-tool round trips; the providers are not scripted.

Set AGENTKIT_LIVE=1 and provide the selected provider's API key via environment.
This file makes paid calls. It does not exercise or certify CLI containment.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import struct
import zlib
from dataclasses import dataclass
from typing import Literal

import httpx
import pytest

from agentkit.core import Mode
from agentkit.core.runtime import CallRuntime, RetryPolicy
from agentkit.models import Binding, CallLimits, CredentialRef, NativeToolPolicy
from agentkit.models.providers.frontiers import ApiAdapter
from agentkit.tools import ToolRegistry, ToolResult, ToolSet, current_tool_invocation, tool


@dataclass(frozen=True)
class SwatchAnswer:
    color: Literal["red", "green", "blue"]


def _swatch(color: str) -> bytes:
    """Complete small PNG with one solid color; no color is disclosed in text."""
    rgb = {"red": b"\xff\0\0", "green": b"\0\xff\0", "blue": b"\0\0\xff"}[color]

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + rgb * 32) * 32))
        + chunk(b"IEND", b"")
    )


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("AGENTKIT_LIVE") != "1", reason="paid image tests require AGENTKIT_LIVE=1")
@pytest.mark.parametrize(
    ("mode", "model", "key"),
    [
        pytest.param(Mode.OPENAI_API, "gpt-4o-mini", "OPENAI_API_KEY", id="openai_api"),
        pytest.param(Mode.ANTHROPIC_API, "claude-haiku-4-5", "ANTHROPIC_API_KEY", id="anthropic_api"),
    ],
)
@pytest.mark.asyncio
async def test_real_api_accepts_host_tool_image_and_reports_usage(mode, model, key):
    if not os.environ.get(key):
        pytest.skip(f"{key} is not supplied")
    color = secrets.choice(("red", "green", "blue"))
    png = _swatch(color)
    result = ToolResult(
        (
            {"type": "text", "text": "Identify the solid color shown in this image."},
            {"type": "image", "mimeType": "image/png", "data": png},
        )
    )
    identities, observed_images, events = [], [], []

    @tool(side_effecting=False)
    def inspect_swatch() -> ToolResult:
        """Return the image to inspect when answering the color question."""
        identities.append(current_tool_invocation())
        return result

    async def inspect_wire(request: httpx.Request) -> None:
        # Inspect JSON in memory, never retain/log headers, keys or provider bodies.
        payload = json.loads(request.content)
        if mode == Mode.OPENAI_API:
            for item in payload.get("input", []):
                if item.get("type") != "function_call_output" or not isinstance(item.get("output"), list):
                    continue
                for block in item["output"]:
                    if block.get("type") == "input_image":
                        prefix, encoded = block["image_url"].split(",", 1)
                        assert prefix == "data:image/png;base64"
                        assert base64.b64decode(encoded, validate=True) == png
                        observed_images.append(True)
        else:
            for message in payload.get("messages", []):
                if not isinstance(message.get("content"), list):
                    continue
                for block in message["content"]:
                    if block.get("type") != "tool_result" or not isinstance(block.get("content"), list):
                        continue
                    assert not block.get("is_error", False)
                    for part in block["content"]:
                        if part.get("type") == "image":
                            assert part["source"]["media_type"] == "image/png"
                            assert base64.b64decode(part["source"]["data"], validate=True) == png
                            observed_images.append(True)

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=45,
        event_hooks={"request": [inspect_wire]},
    ) as client:
        async with CallRuntime(
            adapters={mode: ApiAdapter(client=client)},
            native_tool_policy=NativeToolPolicy.DENY,
            retry=RetryPolicy(max_attempts=1),
        ) as runtime:
            answer = await runtime.run(
                Binding(mode, model, CredentialRef(lambda: {key: os.environ[key]})),
                "Call inspect_swatch, inspect the returned image, and report its solid color. "
                "The color is selected privately by the host; do not guess without using the tool.",
                ToolSet(ToolRegistry.from_tools([inspect_swatch])),
                output=SwatchAnswer,
                limits=CallLimits(max_turns=3, max_output_tokens=256, wall_clock_s=60),
                events=events.append,
                call_id=f"live-image-{mode.value}",
            )
    # Avoid dumping raw provider diagnostics or complete request/answer objects on failure.
    assert answer.outcome == "answered", (
        str(answer.failure.classification) if answer.failure is not None else str(answer.outcome)
    )
    assert observed_images and identities, "no host image reached the real provider request"
    assert answer.value == SwatchAnswer(color), "provider did not identify the privately selected swatch"
    assert all(identity.provider_tool_call_id for identity in identities)
    assert all(identity.transport_request_id is None for identity in identities)
    assert all(identity.host_invocation_id != identity.provider_tool_call_id for identity in identities)
    assert len({identity.host_invocation_id for identity in identities}) == len(identities)
    assert answer.usage.input_uncached + answer.usage.input_cache_read > 0
    assert answer.usage.output > 0 and answer.usage_complete
    assert answer.cost_usd is not None and answer.cost_usd > 0
    assert sum(event.kind == "call_finished" for event in events) == 1
