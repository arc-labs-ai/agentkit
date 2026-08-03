"""OpenAI-compatible `LLMPort` over the Chat Completions API (`/chat/completions`) — the lingua franca:
OpenAI, DeepSeek, OpenRouter, Together, Groq, vLLM, … all speak it; `base_url` selects the provider.
Thin async terminal (httpx); resilience/observability come from the middleware chain.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from agentkit.adapters.llm._mapping import coerce_tool_args
from agentkit.adapters.llm.providers.base import HttpLLM
from agentkit.kernel.types import Delta, LLMResult, Message, ToolCall, Usage


def _to_msg(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # ToolCall.arguments is exposed as a MappingProxyType
                        # (read-only view); stdlib json refuses that type
                        # natively. Convert to a plain dict before dumping.
                        "arguments": json.dumps(dict(tc.arguments)),
                    },
                }
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content}


def _to_tool(schema: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters or {},
        },
    }


def _parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    out: list[ToolCall] = []
    for tc in message.get("tool_calls") or ():
        fn = tc.get("function", {})
        out.append(
            ToolCall(
                id=str(tc.get("id") or fn.get("name") or ""),
                name=fn.get("name", ""),
                arguments=coerce_tool_args(fn.get("arguments")),
            )
        )
    return tuple(out)


def _frag_to_toolcall(index: int, frag: dict[str, Any]) -> ToolCall:
    """Assemble a streamed tool call from its accumulated id/name/arguments fragments. When the provider
    omits an id, fall back to the stream index (unique per call) — not the name, which would collide for
    two parallel calls to the same tool."""
    return ToolCall(
        id=str(frag["id"] or f"call_{index}"),
        name=frag["name"],
        arguments=coerce_tool_args(frag["args"]),
    )


class OpenAICompatibleLLM(HttpLLM):
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_to_msg(m) for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = [_to_tool(t) for t in tools]
        data = await self._post("/chat/completions", payload)

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        u = data.get("usage", {}) or {}
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        prompt = int(u.get("prompt_tokens", 0) or 0)
        usage = Usage(
            input_tokens=max(0, prompt - cached),
            output_tokens=int(u.get("completion_tokens", 0) or 0),
            cache_read_tokens=cached,
        )
        usage = replace(usage, cost_usd=self._cost(model, usage))
        return LLMResult(
            content=message.get("content") or "",
            model=data.get("model", model),
            provider="openai_compatible",
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            tool_calls=_parse_tool_calls(message),
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
        """Incremental streaming over the Chat Completions SSE (`stream:true`). Yields a text `Delta` per
        content chunk as it arrives; tool-call fragments and usage are accumulated and emitted on the final
        delta.

        `stream_options.include_usage` asks the provider for token counts in the closing chunk. Token
        accounting on the streamed path is therefore best-effort: a backend that ignores that option (some
        self-hosted/OpenAI-compatible servers do) sends no usage, so the final delta carries a zero `Usage`
        and zero cost. The non-streaming `chat()` always reports usage from the response body — use it when
        exact cost accounting matters against such a backend."""
        model = model or self._default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_to_msg(m) for m in messages],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = [_to_tool(t) for t in tools]

        served_model, finish, usage = model, None, None
        frags: dict[int, dict[str, Any]] = {}  # tool-call fragments by index
        async for ev in self._stream_events("/chat/completions", payload):
            if ev.get("model"):
                served_model = ev["model"]
            for ch in ev.get("choices") or ():
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    yield Delta(
                        text=delta["content"], model=served_model, provider="openai_compatible"
                    )
                for tc in delta.get("tool_calls") or ():
                    frag = frags.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        frag["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        frag["name"] = fn["name"]
                    if fn.get("arguments"):
                        frag["args"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            u = ev.get("usage")
            if u:
                cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
                prompt = int(u.get("prompt_tokens", 0) or 0)
                usage = Usage(
                    input_tokens=max(0, prompt - cached),
                    output_tokens=int(u.get("completion_tokens", 0) or 0),
                    cache_read_tokens=cached,
                )
        tool_calls = tuple(_frag_to_toolcall(idx, f) for idx, f in sorted(frags.items()))
        final = usage or Usage()  # zero usage if the backend sent no usage chunk
        final = replace(final, cost_usd=self._cost(served_model, final))
        yield Delta(
            tool_calls=tool_calls,
            usage=final,
            finish_reason=finish,
            model=served_model,
            provider="openai_compatible",
        )


def openai(
    *,
    api_key: str,
    model: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    **kw: Any,
) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(api_key=api_key, base_url=base_url, default_model=model, **kw)


def deepseek(
    *,
    api_key: str,
    model: str | None = "deepseek-chat",
    base_url: str = "https://api.deepseek.com/v1",
    **kw: Any,
) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(api_key=api_key, base_url=base_url, default_model=model, **kw)


def openrouter(
    *,
    api_key: str,
    model: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    **kw: Any,
) -> OpenAICompatibleLLM:
    """One endpoint, every model (`anthropic/claude-...`, `openai/gpt-...`, `deepseek/...`)."""
    return OpenAICompatibleLLM(api_key=api_key, base_url=base_url, default_model=model, **kw)


__all__ = ["OpenAICompatibleLLM", "openai", "deepseek", "openrouter"]
