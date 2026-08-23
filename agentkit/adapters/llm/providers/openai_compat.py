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
from agentkit.adapters.llm.providers.base import HttpLLM, raise_if_error_frame
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
                        # ``ToolCall.arguments`` is a ``FrozenDict`` (a ``dict``
                        # SUBCLASS), not the ``MappingProxyType`` this comment
                        # used to name, and stdlib json encodes it natively —
                        # measured byte-identical to the plain dict's output, at
                        # 3.24 us against 3.11 us for the unwrapped form on a
                        # 3-key/2-level payload, i.e. noise on a path that is
                        # about to make an HTTPS round trip. So on every payload
                        # this function can actually be handed, the unwrap is a
                        # no-op.
                        #
                        # It is BELT-AND-BRACES now, not load-bearing, and this
                        # is a typed seam: ``m`` is a ``Message``, so every
                        # ``tc`` is agentkit's own ``ToolCall``, whose
                        # ``__post_init__`` runs ``deep_freeze``. That covers
                        # more than it used to — a ``MappingProxyType`` handed
                        # to ``ToolCall(arguments=...)`` is now NORMALISED into
                        # a ``FrozenDict``, nested proxies included (measured:
                        # ``ToolCall("c", "s", MappingProxyType({...})).arguments``
                        # is a ``FrozenDict`` and ``json.dumps`` of it succeeds),
                        # so the proxy case this comment used to cite is closed.
                        #
                        # One shape still slips through, and it is why the
                        # unwrap stays. ``deep_freeze`` rewrites dicts, lists and
                        # the stdlib proxy, and returns every OTHER ``Mapping``
                        # by identity — deliberately, rather than silently
                        # reconstructing a caller's own type. ``arguments`` is
                        # annotated ``dict[str, Any]``, so reaching that needs a
                        # caller mypy would already reject, but the runtime does
                        # not: measured, ``ToolCall("c", "s", ChainMap({...}))``
                        # stores the ``ChainMap`` verbatim and ``json.dumps`` of
                        # it raises ``Object of type ChainMap is not JSON
                        # serializable``, while ``json.dumps(dict(...))``
                        # succeeds. Here that TypeError would surface as a
                        # provider request that dies before it leaves the
                        # process, which is a bad way to learn it.
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
    """Parse the non-streamed `tool_calls` array.

    When the provider omits an id, the fallback is the POSITION, never the tool name — the exact
    rule `_frag_to_toolcall` states for the streamed path, which this function contradicted.
    Measured: two parallel calls to the same tool with no provider id produced ids
    `['search', 'search']`. One id for two calls means the `tool` result messages keyed by
    `tool_call_id` collide, so one tool's answer silently overwrites the other's and the model is
    told a result it never asked for."""
    out: list[ToolCall] = []
    for i, tc in enumerate(message.get("tool_calls") or ()):
        fn = tc.get("function", {})
        out.append(
            ToolCall(
                id=str(tc.get("id") or f"call_{i}"),
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
        open_slot = 0  # slot a fragment continues when the provider sends no `index`
        async for ev in self._stream_events("/chat/completions", payload):
            # See ``raise_if_error_frame``: an in-band error arrives inside a
            # 200 response and otherwise ends the stream as a complete answer.
            raise_if_error_frame(ev)
            if ev.get("model"):
                served_model = ev["model"]
            for ch in ev.get("choices") or ():
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    yield Delta(
                        text=delta["content"], model=served_model, provider="openai_compatible"
                    )
                for tc in delta.get("tool_calls") or ():
                    idx = tc.get("index")
                    if idx is None:
                        # `index` is how the spec tells parallel calls apart, and defaulting a
                        # missing one to 0 MERGED them: measured, two index-less fragments naming
                        # two distinct calls collapsed to one, `[('a2', 'search')]` instead of two.
                        # A fragment that carries an id or a name STARTS a call; one carrying only
                        # argument text continues the call most recently opened.
                        if tc.get("id") or (tc.get("function") or {}).get("name") or not frags:
                            open_slot = max(frags) + 1 if frags else 0
                        idx = open_slot
                    else:
                        open_slot = idx
                    frag = frags.setdefault(idx, {"id": "", "name": "", "args": ""})
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
