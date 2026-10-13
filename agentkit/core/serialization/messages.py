"""Canonical message wire codec shared by checkpoints and transcript consumers.

The kernel owns the representation; persistence adapters own storage. Encoded
containers are independent mutable JSON shapes, while decoded records freeze
payloads through their constructors.
"""

from __future__ import annotations

from typing import Any

from agentkit.core.contracts.types import Message, ToolCall
from agentkit.core.serialization._immutable import thaw


def tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": thaw(call.arguments)}


def tool_call_from_dict(value: dict[str, Any]) -> ToolCall:
    return ToolCall(value["id"], value["name"], value.get("arguments", {}))


def message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "name": message.name,
        "tool_calls": [tool_call_to_dict(call) for call in message.tool_calls],
        "content_blocks": [thaw(block) for block in message.content_blocks],
        "metadata": thaw(message.metadata),
    }


def message_from_dict(value: dict[str, Any]) -> Message:
    return Message(
        role=value["role"],
        content=value.get("content", ""),
        name=value.get("name"),
        tool_call_id=value.get("tool_call_id"),
        tool_calls=tuple(tool_call_from_dict(call) for call in (value.get("tool_calls") or [])),
        content_blocks=tuple(value.get("content_blocks") or ()),
        metadata=value.get("metadata") or {},
    )
