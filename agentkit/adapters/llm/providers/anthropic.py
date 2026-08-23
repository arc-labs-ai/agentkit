"""Anthropic Messages API `LLMPort` (`/v1/messages`) — different request shape from OpenAI: `system` is a
top-level field, content is blocks, prompt caching is `cache_control` on a block. Thin async terminal (httpx);
`cache_hint` on the request → an ephemeral `cache_control` on the system prompt (cheap cache reads next turn).
"""

from __future__ import annotations

import json
import warnings
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


def _response_format_instruction(response_format: Any) -> str | None:
    """Translate an OpenAI-style ``response_format`` into an Anthropic system instruction.

    The Messages API has NO ``response_format`` parameter. This adapter used to accept one and
    DROP it: measured, the request payload keys were
    ``['max_tokens', 'messages', 'model', 'temperature']`` — no trace of the caller's contract,
    no error, and a caller that asked for JSON got prose. That silence is the bug.

    Two honest options existed — translate, or refuse. Refusing outright would break callers that
    work today (``capabilities/eval/base.py`` sends ``{"type": "json_object"}`` to whatever judge
    model is wired, Anthropic included), so shapes with a faithful prompt-level equivalent are
    TRANSLATED into a system instruction, and the translation announces itself once per client via
    :meth:`AnthropicLLM._warn_prompt_level_json` because a prompt is best-effort where OpenAI's
    ``strict`` schema is server-enforced. Anything we cannot translate is REFUSED loudly.

    Returns ``None`` when nothing needs to be added (no format requested, or plain ``text``,
    which is already the default).
    """
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ValueError(
            "Anthropic has no response_format parameter; this adapter translates the OpenAI-style "
            f"dict shapes {{'type': 'text'|'json_object'|'json_schema'}} and cannot translate "
            f"{response_format!r}. Drop it, or use tools= for a schema-shaped answer."
        )
    kind = response_format.get("type")
    if kind == "text":
        return None  # already the default — nothing to say
    if kind == "json_object":
        return (
            "Respond with a single valid JSON object and nothing else: "
            "no prose before or after it, and no markdown code fences."
        )
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema is None:
            raise ValueError(
                "response_format {'type': 'json_schema'} needs json_schema.schema; got "
                f"{response_format!r}."
            )
        return (
            "Respond with a single valid JSON object and nothing else: no prose before or after "
            "it, and no markdown code fences. It must validate against this JSON Schema:\n"
            f"{json.dumps(schema, sort_keys=True)}"
        )
    raise ValueError(
        f"Anthropic has no response_format parameter and response_format type {kind!r} has no "
        "prompt-level translation. Use {'type': 'json_object'} or {'type': 'json_schema'}, or "
        "drop the parameter and use tools= for a schema-shaped answer."
    )


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
        # One-shot latch for the response_format downgrade notice. Per-instance (like
        # ``ModelRegistry._warned``) so a test can observe it on a fresh client and so a
        # per-turn agent loop doesn't emit the same warning on every call.
        self._warned_response_format = False

    def _warn_prompt_level_json(self) -> None:
        if self._warned_response_format:
            return
        self._warned_response_format = True
        warnings.warn(
            "Anthropic has no response_format parameter: this client translated it into a system "
            "instruction, which the model follows best-effort rather than the provider enforcing "
            "it. Validate the output (Agent(output=...) does) or use tools= for a schema-shaped "
            "answer. Previously this parameter was accepted and silently dropped.",
            UserWarning,
            stacklevel=3,
        )

    def _system_prompt(self, messages: Any, response_format: Any) -> str:
        """The payload's ``system`` text: the flattened system turns, plus the translated
        ``response_format`` instruction when one was requested (Anthropic has no such field —
        see :func:`_response_format_instruction`)."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        instruction = _response_format_instruction(response_format)
        if instruction is None:
            return system
        self._warn_prompt_level_json()
        return f"{system}\n\n{instruction}" if system else instruction

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
        system = self._system_prompt(messages, response_format)
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
        system = self._system_prompt(messages, response_format)
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
