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


class ScriptExhausted(BaseException):
    """A `FakeLLM.script` was asked for more turns than it has.

    WHY THIS EXISTS, AND WHY IT IS LOUD
    -----------------------------------
    This class used to be a `min()`. `stream` clamped the script index, so a
    2-turn script asked for 4 turns answered ``['one', 'two', 'two', 'two']``
    — measured, not hypothetical. A scripted LLM exists to drive an agent
    through an EXPECTED number of turns, so "the agent asked for more turns
    than the script has" is a statement about the code under test: it looped
    when it should have stopped. The clamp turned exactly that finding into a
    stable, plausible, wrong answer, and the test went green. The harness
    built to catch non-termination was the thing hiding it — the same shape as
    a broad ``except`` that reports success.

    WHY `BaseException` AND NOT `Exception`
    ---------------------------------------
    Because the failure it reports is a failure of the code that would be
    doing the catching. `agentkit` has ~20 deliberate ``except Exception``
    sites on the path between `FakeLLM.stream` and the test body — react
    reflects invalid output back to the model, `_invoke_tool_safe` turns every
    tool failure into a tool message, `resilience` classifies pre-stream
    failures into retry-vs-fail. Each one is correct for a real provider
    fault, and each one would convert "your loop does not terminate" into a
    retry or a reflected message and keep going. `asyncio.CancelledError` was
    promoted out of `Exception` in 3.8 for this exact reason; a test double's
    contract violation belongs in the same bucket. pytest reports a
    `BaseException` as a plain failure, so nothing is lost at the top.
    """


@dataclass
class Turn:
    """One scripted reply for `FakeLLM.script` — a final answer, or a turn that requests tools."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None


class FakeLLM:
    """Deterministic offline LLM. dict {substring_of_user: content} | callable | str for single
    answers; `FakeLLM.script([Turn(...), …])` replays a multi-step tool loop across `chat` calls.

    The two forms differ ON PURPOSE once they run out of material:

    * the single-answer forms (str / dict / callable) have nothing to run out
      of — they are a rule for turning a request into a reply, so call 1 and
      call 100 both answer. Repeating IS their meaning.
    * a script is a finite, ordered claim about how many turns the run takes.
      Asking for turn N+1 contradicts the claim, so it raises
      `ScriptExhausted`. Pass `repeat_last=True` for the deliberate
      never-terminating script — a single tool-call `Turn` used to drive a
      loop until `max_iterations` or a budget cuts it."""

    def __init__(
        self,
        responses: dict[str, str] | Callable[..., str] | str = "{}",
        *,
        usage: Usage | None = None,
        fail_times: int = 0,
        fail_exc: BaseException | None = None,
        delay: float = 0.0,
        turns: list[Turn] | None = None,
        repeat_last: bool = False,
    ):
        self._responses = responses
        self._usage = usage or Usage(10, 5, 0.0001)
        self._fail_times = fail_times
        self._fail_exc = fail_exc or TimeoutError("temporary upstream timeout")
        self._delay = delay
        self._turns = list(turns) if turns is not None else None
        self._repeat_last = repeat_last
        self._turn_idx = 0
        self.calls = 0

    @classmethod
    def script(cls, turns: list[Turn], *, repeat_last: bool = False, **kw: Any) -> FakeLLM:
        """Replay `turns` in order, one per `chat`/`stream` call, then raise `ScriptExhausted`.

        `repeat_last=True` replays the final turn forever instead. Spelled out
        as a real keyword rather than left to `**kw` so it shows up in a
        signature and an IDE completion — it is the escape hatch a reader
        needs at the moment the exception fires, and `**kw` hides it."""
        return cls(turns=turns, repeat_last=repeat_last, **kw)

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
            # `_enter()` already ran, so a `fail_times` failure never reaches
            # here: a call that raised produced no reply and must not burn a
            # turn. That keeps `fail_times=2` + a 3-turn script meaning "two
            # failures then the three scripted turns" rather than "two
            # failures that silently eat two thirds of the script".
            if self._turn_idx >= len(self._turns):
                if not self._repeat_last or not self._turns:
                    raise ScriptExhausted(_exhausted_message(len(self._turns), self._turn_idx))
                turn = self._turns[-1]
            else:
                turn = self._turns[self._turn_idx]
            # Incremented past the end too, so a `repeat_last` test can assert
            # HOW FAR past its script a loop went, not merely that it went.
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


def _exhausted_message(have: int, idx: int) -> str:
    """The whole value of raising is in this string, so it carries the two
    numbers a reader needs (how many turns the script had, which turn was
    asked for) and both ways out. A bare "script exhausted" would send someone
    to read `FakeLLM` instead of reading their own loop, which is where the
    bug almost always is."""
    return (
        f"FakeLLM script exhausted: the script has {have} turn(s), but the agent asked "
        f"for turn {idx + 1}. The agent kept calling the LLM after the script ran out — "
        "that is usually the defect under test (a loop that does not terminate, a stop "
        "condition that never fires), not a problem with the script. Either add the "
        f"missing Turn(...)s if {idx + 1} calls are genuinely expected, or pass "
        "FakeLLM.script([...], repeat_last=True) to replay the final turn forever when "
        "the test deliberately drives an unbounded loop until max_iterations or a budget "
        "stops it."
    )
