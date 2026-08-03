"""`CallableLLM` — adapt any injected single-call/tool-aware provider `fn` into an LLMPort."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

from agentkit.adapters.llm._mapping import (
    _duck_to_result,
    _extract_tool_calls,
    _flatten,
    _zero_cost,
    result_to_delta,
)
from agentkit.kernel.types import LLMResult, Message


class CallableLLM:
    """Adapt `fn(*, system, user, model, …)` (single call) and optionally
    `chat_fn(*, messages, model, tools, …)` (tool-aware) into an LLMPort. Sync fns run off the loop
    via `asyncio.to_thread`; coroutines are awaited. Provider tool-calls are normalized to `ToolCall`."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        to_result: Callable[[Any], LLMResult] | None = None,
        chat_fn: Callable[..., Any] | None = None,
        cost_fn: Callable[[Any, int, int], float] | None = None,
    ) -> None:
        self._fn = fn
        self._chat_fn = chat_fn
        # Cost is computed by an injected fn (a provider price table or
        # whatever the host provides) — never an observability-library
        # import in the core. Default: 0.0 (cost unknown without a source).
        self._cost_fn = cost_fn or _zero_cost
        self._to_result = to_result or (lambda raw: _duck_to_result(raw, self._cost_fn))

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
        kw = dict(
            system=system,
            user=user,
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (
            await self._fn(**kw)
            if inspect.iscoroutinefunction(self._fn)
            else await asyncio.to_thread(lambda: self._fn(**kw))
        )
        return self._to_result(raw)

    async def chat(
        self,
        *,
        messages: list[Message],
        model: Any,
        tools: Any = None,
        response_format: Any = None,
        temperature: float = 0.0,
        max_tokens: Any = None,
        cache_hint: Any = None,
    ) -> LLMResult:
        if self._chat_fn is None:  # no tool-aware fn → flatten + reuse complete
            system, user = _flatten(messages)
            return await self.complete(
                system=system,
                user=user,
                model=model,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        chat_fn = self._chat_fn
        assert chat_fn is not None  # guarded above; narrows for the type-checker
        kw = dict(
            messages=messages,
            model=model,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if cache_hint is not None:
            kw["cache_hint"] = cache_hint
        raw = (
            await chat_fn(**kw)
            if inspect.iscoroutinefunction(chat_fn)
            else await asyncio.to_thread(lambda: chat_fn(**kw))
        )
        res = raw if isinstance(raw, LLMResult) else self._to_result(raw)
        if not res.tool_calls:
            # LLMResult is frozen — rebuild with the extracted tool_calls
            # rather than mutating in place.
            res = replace(res, tool_calls=_extract_tool_calls(raw))
        return res

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
    ) -> AsyncIterator[Any]:
        # The wrapped provider (often a sync ModelSDKTools) returns a complete result → one terminal Delta.
        res = await self.chat(
            messages=messages,
            model=model,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_hint=cache_hint,
        )
        yield result_to_delta(res)
