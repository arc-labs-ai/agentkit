"""Canonical Message translations; opaque reasoning stays in SessionHandle.history."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agentkit.core.contracts.types import Message, MessageRole, Mode, ToolCall
from agentkit.core.serialization.content import encode_content


def tool_content(blocks: Any, mode: Mode) -> Any:
    blocks = encode_content(blocks)
    if mode == Mode.OPENAI_API and all(x.get("type") == "text" for x in blocks):
        return "\n".join(x.get("text", "") for x in blocks)
    out = []
    for block in blocks:
        kind = block.get("type")
        if kind == "image":
            if mode == Mode.ANTHROPIC_API:
                out.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": block["mimeType"], "data": block["data"]},
                    }
                )
            else:
                out.append({"type": "input_image", "image_url": f"data:{block['mimeType']};base64,{block['data']}"})
        elif kind == "text":
            if mode == Mode.ANTHROPIC_API and not block.get("text", "").strip():
                continue
            out.append({"type": "text" if mode == Mode.ANTHROPIC_API else "input_text", "text": block.get("text", "")})
        else:
            out.append(
                {"type": "text" if mode == Mode.ANTHROPIC_API else "input_text", "text": json.dumps(dict(block))}
            )
    return out


def history_for(messages: Sequence[Message], mode: Mode) -> tuple[dict[str, Any], ...]:
    """Translate portable conversation state when switching vendor or model."""
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
            continue  # supplied through the composed system prompt
        blocks = message.content_blocks or ({"type": "text", "text": message.content},)
        if mode == Mode.ANTHROPIC_API:
            if message.role == MessageRole.TOOL:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": tool_content(blocks, mode),
                                "is_error": bool(message.metadata.get("is_error", False)),
                            }
                        ],
                    }
                )
            else:
                content = tool_content(blocks, mode)
                content.extend(
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": dict(c.arguments)}
                    for c in message.tool_calls
                )
                if content:
                    out.append({"role": message.role, "content": content})
        else:
            if message.role == MessageRole.TOOL:
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": tool_content(blocks, mode),
                    }
                )
            else:
                if message.content or message.content_blocks:
                    if message.role == MessageRole.ASSISTANT:
                        content = (
                            [{"type": "output_text", "text": message.content}]
                            if not message.content_blocks
                            else [
                                {"type": "output_text", "text": b.get("text", "")}
                                for b in blocks
                                if b.get("type") == "text"
                            ]
                        )
                    else:
                        content = tool_content(blocks, mode)
                        if isinstance(content, str):
                            content = [{"type": "input_text", "text": content}]
                    out.append({"role": message.role, "content": content})
                out.extend(
                    {"type": "function_call", "call_id": c.id, "name": c.name, "arguments": json.dumps(c.arguments)}
                    for c in message.tool_calls
                )
    return tuple(out)


def anthropic_history(history: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Omit invalid blank text while preserving opaque blocks and tool pairing.

    Portable tool-only turns have no text; providers can also emit empty text
    blocks during streaming. Neither requires a fabricated text placeholder.
    Empty tool results remain real results with their IDs and error metadata.
    """
    out = []
    for item in history:
        content = item.get("content")
        if isinstance(content, (list, tuple)):
            cleaned = []
            for block in content:
                if block.get("type") == "text" and not block.get("text", "").strip():
                    continue
                if block.get("type") == "tool_result" and isinstance(block.get("content"), (list, tuple)):
                    block = {
                        **block,
                        "content": [
                            child
                            for child in block["content"]
                            if child.get("type") != "text" or child.get("text", "").strip()
                        ],
                    }
                cleaned.append(dict(block))
            if not cleaned:
                continue
            out.append({**item, "content": cleaned})
        elif isinstance(content, str) and not content.strip():
            continue
        else:
            out.append(dict(item))
    return tuple(out)


def canonical_turn(history: Sequence[Any], mode: Mode) -> tuple[Message, ...]:
    """Extract visible assistant messages and tool calls without copying reasoning."""
    out = []
    if mode == Mode.ANTHROPIC_API:
        if not history or history[-1].get("role") != "assistant":
            return ()
        blocks = history[-1].get("content") or []
        calls = tuple(ToolCall(b["id"], b["name"], b.get("input") or {}) for b in blocks if b.get("type") == "tool_use")
        out.append(
            Message(
                MessageRole.ASSISTANT,
                "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
                tool_calls=calls,
            )
        )
    else:
        # Input history is included before the output; take output items after
        # the last user/function result. Reasoning items are never canonicalized.
        start = 0
        for index, item in enumerate(history):
            if item.get("role") == "user" or item.get("type") == "function_call_output":
                start = index + 1
        openai_calls: list[ToolCall] = []
        for item in history[start:]:
            if item.get("type") == "function_call":
                openai_calls.append(ToolCall(item["call_id"], item["name"], json.loads(item.get("arguments") or "{}")))
            elif item.get("type") == "message":
                out.append(
                    Message(
                        MessageRole.ASSISTANT,
                        "".join(b.get("text", "") for b in item.get("content", []) if b.get("type") == "output_text"),
                    )
                )
        if openai_calls:
            out.append(Message(MessageRole.ASSISTANT, tool_calls=tuple(openai_calls)))
    return tuple(out)


def close_pending_tools(
    history: Sequence[Any],
    messages: Sequence[Message],
    mode: Mode,
    started: set[str],
) -> tuple[tuple[dict[str, Any], ...], tuple[Message, ...]]:
    """Make an interrupted API transcript resumable without claiming tool success.

    Confirmed canonical results are retained. Calls lacking a result receive
    explicit error evidence stating whether execution started; unknown side
    effects must be checked before retrying.
    """
    calls: dict[str, str] = {}
    completed = set()
    for item in history:
        if mode == Mode.OPENAI_API:
            if item.get("type") == "function_call":
                calls[item["call_id"]] = item["name"]
            elif item.get("type") == "function_call_output":
                completed.add(item["call_id"])
        elif mode == Mode.ANTHROPIC_API:
            content = item.get("content")
            for block in content if isinstance(content, (list, tuple)) else ():
                if block.get("type") == "tool_use":
                    calls[block["id"]] = block["name"]
                elif block.get("type") == "tool_result":
                    completed.add(block["tool_use_id"])
    results = {m.tool_call_id: m for m in messages if m.role == MessageRole.TOOL}
    appended = []
    out = [dict(item) for item in history]
    anthropic = []
    for identifier, name in calls.items():
        if identifier in completed:
            continue
        result = results.get(identifier)
        if result is None:
            interrupted = identifier in started
            text = (
                "Tool execution was interrupted without a confirmed result. Completion and side effects are unknown; "
                "verify state before retrying."
                if interrupted
                else "Tool was not executed because the call ended before execution."
            )
            result = Message(
                MessageRole.TOOL,
                text,
                name=name,
                tool_call_id=identifier,
                content_blocks=({"type": "text", "text": text},),
                metadata={"is_error": True, "interrupted": interrupted, "not_executed": not interrupted},
            )
            appended.append(result)
        blocks = result.content_blocks or ({"type": "text", "text": result.content},)
        if mode == Mode.ANTHROPIC_API:
            anthropic.append(
                {
                    "type": "tool_result",
                    "tool_use_id": identifier,
                    "content": tool_content(blocks, mode),
                    "is_error": bool(result.metadata.get("is_error")),
                }
            )
        else:
            out.append({"type": "function_call_output", "call_id": identifier, "output": tool_content(blocks, mode)})
    if anthropic:
        out.append({"role": "user", "content": anthropic})
    return tuple(out), tuple(appended)
