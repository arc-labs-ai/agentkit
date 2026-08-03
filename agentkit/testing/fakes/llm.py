"""`FakeLLM`/`Turn` — deterministic offline LLM for tests (single answers or scripted tool loops)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from agentkit.adapters.llm._mapping import _flatten
from agentkit.kernel.types import (
    Delta,
    LLMResult,
    Message,
    ToolCall,
    ToolSchema,
    Usage,
    assemble_deltas,
)


@dataclass
class Turn:
    """One scripted reply for `FakeLLM.script` — a final answer, or a turn that requests tools."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None


class FakeLLM:
    """Deterministic offline LLM. dict {substring_of_user: content} | callable | str for single
    answers; `FakeLLM.script([Turn(...), …])` replays a multi-step tool loop across `chat` calls."""

    def __init__(
        self,
        responses: dict[str, str] | Callable[..., str] | str = "{}",
        *,
        usage: Usage | None = None,
        fail_times: int = 0,
        fail_exc: BaseException | None = None,
        delay: float = 0.0,
        turns: list[Turn] | None = None,
    ):
        self._responses = responses
        self._usage = usage or Usage(10, 5, 0.0001)
        self._fail_times = fail_times
        self._fail_exc = fail_exc or TimeoutError("temporary upstream timeout")
        self._delay = delay
        self._turns = list(turns) if turns is not None else None
        self._turn_idx = 0
        self.calls = 0

    @classmethod
    def script(cls, turns: list[Turn], **kw: Any) -> FakeLLM:
        return cls(turns=turns, **kw)

    def _content(self, system: str, user: str, model: str) -> str:
        if callable(self._responses):
            return self._responses(system=system, user=user, model=model)
        if isinstance(self._responses, dict):
            return next((v for k, v in self._responses.items() if k in user), "{}")
        return self._responses

    def _enter(self) -> None:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._fail_exc

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        self._enter()
        return LLMResult(
            content=self._content(system, user, model),
            model=model,
            provider="fake",
            finish_reason="stop",
            usage=self._usage,
        )

    async def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cache_hint: Any = None,
    ) -> AsyncIterator[Delta]:
        """The streaming primitive: emits content in small chunks then a terminal delta carrying
        usage/finish/tool_calls. `_enter()` raises BEFORE the first delta, so it's a pre-stream failure the
        `retry` middleware can re-invoke (honoring `fail_times`)."""
        if self._delay:
            await asyncio.sleep(self._delay)
        self._enter()
        if self._turns is not None:
            turn = self._turns[min(self._turn_idx, len(self._turns) - 1)]
            self._turn_idx += 1
            content, tool_calls = turn.content, tuple(turn.tool_calls)
            finish, usage = (
                ("tool_calls" if turn.tool_calls else "stop"),
                (turn.usage or self._usage),
            )
        else:
            system, user = _flatten(messages)
            content, tool_calls, finish, usage = (
                self._content(system, user, model),
                (),
                "stop",
                self._usage,
            )
        for i in range(0, len(content), 8):  # chunked → genuinely incremental
            yield Delta(text=content[i : i + 8], model=model, provider="fake")
        yield Delta(
            tool_calls=tool_calls, usage=usage, finish_reason=finish, model=model, provider="fake"
        )

    async def chat(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cache_hint: Any = None,
    ) -> LLMResult:
        return assemble_deltas(
            [
                d
                async for d in self.stream(  # chat ≡ collect(stream)
                    messages=messages,
                    model=model,
                    tools=tools,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cache_hint=cache_hint,
                )
            ]
        )
