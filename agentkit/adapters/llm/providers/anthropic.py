"""Anthropic Messages API `LLMPort` (`/v1/messages`) — different request shape from OpenAI: `system` is a
top-level field, content is blocks, prompt caching is `cache_control` on a block. Thin async terminal (httpx);
`cache_hint` on the request → an ephemeral `cache_control` on the system prompt (cheap cache reads next turn).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from agentkit.adapters.llm._mapping import coerce_tool_args
from agentkit.adapters.llm.providers.base import HttpLLM, raise_if_error_frame
from agentkit.kernel.types import Delta, LLMResult, Message, ToolCall, Usage

# Anthropic REQUIRES max_tokens; OpenAI does not. When a caller leaves it unset, we must still send one,
# so responses are capped at this default (override per-call with `max_tokens=`, or per-client via
# `AnthropicLLM(default_max_tokens=…)`). Documented so the cap is never a silent surprise.
DEFAULT_MAX_TOKENS = 4096


def _to_msg(m: Message) -> dict[str, Any]:
    if m.role == "tool":  # a tool result → a user turn carrying a tool_result block
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            ],
        }
    if m.role == "assistant" and m.tool_calls:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
        # ``dict(tc.arguments)`` — ``ToolCall.arguments`` is a
        # ``MappingProxyType`` read-only view, and httpx's ``json=`` payload
        # serializer goes through stdlib ``json.dumps`` which doesn't handle
        # that type natively.
        blocks += [
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": dict(tc.arguments)}
            for tc in m.tool_calls
        ]
        return {"role": "assistant", "content": blocks}
    return {"role": m.role, "content": m.content}


def _to_tool(schema: Any) -> dict[str, Any]:
    return {
        "name": schema.name,
        "description": schema.description,
        "input_schema": schema.parameters or {},
    }


class AnthropicLLM(HttpLLM):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        version: str = "2023-06-01",
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        **kw: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kw)
        self._version = version
        self._default_max_tokens = default_max_tokens

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key or "",
            "anthropic-version": self._version,
            "content-type": "application/json",
            **self._extra_headers,
        }

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None,
        response_format: Any = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        return await self.chat(
            messages=[Message("system", system), Message("user", user)],
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any = None,
        response_format: Any = None,
        temperature: float = 0.0,
        max_tokens: Any = None,
        cache_hint: Any = None,
    ) -> LLMResult:
        model = model or self._default_model
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens or self._default_max_tokens,
            "messages": [_to_msg(m) for m in messages if m.role != "system"],
        }
        if system:
            payload["system"] = (
                [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                if cache_hint
                else system
            )  # cache_hint → prompt caching on the prefix
        if tools:
            payload["tools"] = [_to_tool(t) for t in tools]
        data = await self._post("/v1/messages", payload)

        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        tool_calls = tuple(
            ToolCall(id=b.get("id", ""), name=b.get("name", ""), arguments=b.get("input") or {})
            for b in data.get("content", [])
            if b.get("type") == "tool_use"
        )
        u = data.get("usage", {}) or {}
        usage = Usage(
            input_tokens=int(
                u.get("input_tokens", 0) or 0
            ),  # Anthropic input is already cache-fresh
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_read_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
        )
        usage = replace(usage, cost_usd=self._cost(model, usage))
        return LLMResult(
            content=text,
            model=data.get("model", model),
            provider="anthropic",
            finish_reason=data.get("stop_reason"),
            usage=usage,
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any = None,
        response_format: Any = None,
        temperature: float = 0.0,
        max_tokens: Any = None,
        cache_hint: Any = None,
    ) -> AsyncIterator[Delta]:
        """Incremental streaming over the Messages SSE (`stream:true`). Yields a text `Delta` per
        `text_delta`; tool-use input JSON arrives as `input_json_delta` fragments (accumulated), and token
        counts arrive split across `message_start` (input) and `message_delta` (output) — emitted on the
        final delta."""
        model = model or self._default_model
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens or self._default_max_tokens,
            "messages": [_to_msg(m) for m in messages if m.role != "system"],
            "stream": True,
        }
        if system:
            payload["system"] = (
                [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                if cache_hint
                else system
            )
        if tools:
            payload["tools"] = [_to_tool(t) for t in tools]

        served_model, finish = model, None
        in_tok = out_tok = cache_read = cache_write = 0
        blocks: dict[int, dict[str, Any]] = {}  # content blocks by index (text or tool_use)
        async for ev in self._stream_events("/v1/messages", payload):
            # An in-band error frame arrives INSIDE a 200 response, mid-stream.
            # Checked before the type dispatch so it cannot fall through every
            # branch and end the stream as if the answer were complete.
            raise_if_error_frame(ev)
            etype = ev.get("type")
            if etype == "message_start":
                msg = ev.get("message") or {}
                served_model = msg.get("model") or served_model
                u = msg.get("usage") or {}
                in_tok = int(u.get("input_tokens", 0) or 0)
                cache_read = int(u.get("cache_read_input_tokens", 0) or 0)
                cache_write = int(u.get("cache_creation_input_tokens", 0) or 0)
            elif etype == "content_block_start":
                cb = ev.get("content_block") or {}
                blocks[ev.get("index", 0)] = {
                    "type": cb.get("type"),
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "json": "",
                }
            elif etype == "content_block_delta":
                d = ev.get("delta") or {}
                if d.get("type") == "text_delta" and d.get("text"):
                    yield Delta(text=d["text"], model=served_model, provider="anthropic")
                elif d.get("type") == "input_json_delta":
                    blocks.setdefault(
                        ev.get("index", 0), {"type": "tool_use", "id": "", "name": "", "json": ""}
                    )
                    blocks[ev.get("index", 0)]["json"] += d.get("partial_json", "")
            elif etype == "message_delta":
                if (ev.get("delta") or {}).get("stop_reason"):
                    finish = ev["delta"]["stop_reason"]
                out_tok = int((ev.get("usage") or {}).get("output_tokens", out_tok) or out_tok)
        tool_calls = tuple(
            ToolCall(id=b["id"], name=b["name"], arguments=coerce_tool_args(b["json"]))
            for b in blocks.values()
            if b.get("type") == "tool_use"
        )
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        usage = replace(usage, cost_usd=self._cost(served_model, usage))
        yield Delta(
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish,
            model=served_model,
            provider="anthropic",
        )


def claude(*, api_key: str, model: str | None = "claude-sonnet-4-6", **kw: Any) -> AnthropicLLM:
    return AnthropicLLM(api_key=api_key, default_model=model, **kw)


__all__ = ["AnthropicLLM", "claude"]
