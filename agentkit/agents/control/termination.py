"""TerminationCondition — composable, async stop conditions for loops & multi-agent runs.

A condition is checked once per turn on the **delta** of new messages and returns a structured `Stop`
(or `None` to continue). Conditions compose with `|` (OR — stop if any) and `&` (AND — stop only when
all). Once terminated, a condition keeps returning its `Stop` until `reset()`. Conditions are **async**
by design, so one may `await` (e.g. an LLM judge via `FunctionalTermination`).

These are the *smart* stop layer; a **hard ceiling** (a loop's `max_iterations`, or the run `Budget`)
is always the never-hang backstop — termination conditions sit on top of it, never replace it.
"""

from __future__ import annotations

import inspect
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel._frozen import deep_freeze
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message


@dataclass(frozen=True)
class Stop:
    """Why a run stopped — a structured reason (fed into `AgentResult.evals['stop_reason']`).

    Frozen, because a condition LATCHES its ``Stop`` and hands the same instance
    to every caller on every subsequent turn. A consumer writing
    ``stop.reason = ...`` was rewriting the condition's own record of why it
    stopped, and every later read — including the policy's, including the
    trace's — saw the rewrite. Same reasoning as ``Suspended``: a value a
    caller reasons about must not be editable underneath the thing that
    produced it.

    ``detail`` used to be left as a plain writable dict, and this docstring
    used to defend that: a ``MappingProxyType`` cannot be deep-copied, and
    conditions ARE deep-copied per drive (``ReActCognition`` at both entry
    points, ``roundrobin._run_local_termination``) to keep counters run-local.
    That constraint was real, and clearing it is what ``_frozen`` was built for.
    ``__post_init__`` below now deep-freezes ``detail`` into a ``FrozenDict``,
    which deep-copies, pickles and json-dumps like the plain dict it subclasses.
    Measured on a latched ``Stop`` held by a live condition: ``deepcopy`` and a
    pickle round trip both return it intact, while ``stop.detail["x"] = 1`` and
    ``stop.detail["nested"]["b"] = 1`` both raise ``TypeError``. So the evidence
    is now as immutable as the ``reason``, and the whole verdict is a record
    rather than half of one.
    """

    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Frozen in name only until this ran: a termination verdict's detail is a record of WHY a
        run stopped, and a record that can be rewritten after the fact is not a record."""
        object.__setattr__(self, "detail", deep_freeze(self.detail))

    def __hash__(self) -> int:
        """A verdict is identified by WHY it stopped; `detail` is the evidence, and is
        excluded (see `_frozen.py`)."""
        return hash((self.reason,))


class TerminationCondition:
    """Base class. Subclass and implement `_evaluate(delta) -> Stop | None`.

    Statefulness: cumulative counters live on the instance; `reset()` clears them (call `super().reset()`).
    Once a `Stop` is produced, `__call__` returns it on every subsequent turn until `reset()`.
    """

    def __init__(self) -> None:
        self._stop: Stop | None = None

    @property
    def terminated(self) -> bool:
        return self._stop is not None

    async def __call__(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        # `ctx` is the LIVE run context, threaded so a condition that calls a model (FunctionalTermination /
        # judge_termination) runs on the actual run's budget/cancel/trace — not one captured at build time.
        if self._stop is None:
            self._stop = await self._evaluate(delta, ctx)
        return self._stop

    async def _evaluate(
        self, delta: Sequence[Message], ctx: Ctx | None = None
    ) -> Stop | None:  # pragma: no cover
        raise NotImplementedError

    async def reset(self) -> None:
        self._stop = None

    # OR / AND compose into one self-flattening composite (siblings of the same mode merge).
    def __or__(self, other: TerminationCondition) -> TerminationCondition:
        return _Composite(_flatten("any", self, other), mode="any")

    def __and__(self, other: TerminationCondition) -> TerminationCondition:
        return _Composite(_flatten("all", self, other), mode="all")


def _flatten(mode: str, *conds: TerminationCondition) -> list[TerminationCondition]:
    out: list[TerminationCondition] = []
    for c in conds:
        if isinstance(c, _Composite) and c.mode == mode:
            out.extend(c.conditions)
        else:
            out.append(c)
    return out


class _Composite(TerminationCondition):
    """OR (`mode="any"`) / AND (`mode="all"`) over child conditions. Each turn it advances **every**
    child (so each accumulates its own state), then stops per the mode."""

    def __init__(self, conditions: Sequence[TerminationCondition], *, mode: str) -> None:
        super().__init__()
        self.conditions = list(conditions)
        self.mode = mode

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        results = [
            await c(delta, ctx) for c in self.conditions
        ]  # advance each child (thread live ctx)
        if self.mode == "any":
            return next((r for r in results if r is not None), None)
        if all(c.terminated for c in self.conditions):  # "all": only when every child stopped
            return Stop("all", {"reasons": [c._stop.reason for c in self.conditions if c._stop]})
        return None

    async def reset(self) -> None:
        await super().reset()
        for c in self.conditions:
            await c.reset()


# --- catalog ----------------------------------------------------------------------------------------


class MaxMessages(TerminationCondition):
    """Stop after a cumulative number of messages seen across the deltas it's evaluated on. In a team
    (one new message per turn) this equals a turn count — i.e. `MaxMessages(N)` behaves like `MaxTurns(N)`
    there; in a loop it counts the assistant deltas. Use `MaxTurns` when you mean turns explicitly."""

    def __init__(self, max_messages: int) -> None:
        super().__init__()
        self.max = int(max_messages)
        self.count = 0

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        self.count += len(delta)
        return (
            Stop("max_messages", {"count": self.count, "max": self.max})
            if self.count >= self.max
            else None
        )

    async def reset(self) -> None:
        await super().reset()
        self.count = 0


class MaxTurns(TerminationCondition):
    """Stop after a number of turns (one per evaluation)."""

    def __init__(self, max_turns: int) -> None:
        super().__init__()
        self.max = int(max_turns)
        self.turn = 0

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        self.turn += 1
        return (
            Stop("max_turns", {"turn": self.turn, "max": self.max})
            if self.turn >= self.max
            else None
        )

    async def reset(self) -> None:
        await super().reset()
        self.turn = 0


class TextMention(TerminationCondition):
    """Stop when `text` appears in a message (optionally only from the given sources)."""

    def __init__(self, text: str, *, sources: Sequence[str] | None = None) -> None:
        super().__init__()
        self.text = text
        self.sources = set(sources) if sources else None

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        for m in delta:
            if self.sources is not None and (m.name or m.role) not in self.sources:
                continue
            if self.text in (m.content or ""):
                return Stop("text_mention", {"text": self.text})
        return None


class FunctionCall(TerminationCondition):
    """Stop when a tool/function with `name` is called."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        for m in delta:
            for tc in m.tool_calls or ():
                if tc.name == self.name:
                    return Stop("function_call", {"function": self.name})
        return None


class SourceMatch(TerminationCondition):
    """Stop when a message from one of `sources` (a message's `name`, else `role`) appears."""

    def __init__(self, *sources: str) -> None:
        super().__init__()
        self.sources = set(sources)

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        for m in delta:
            src = m.name or m.role
            if src in self.sources:
                return Stop("source_match", {"source": src})
        return None


class Timeout(TerminationCondition):
    """Stop after `seconds` of wall-clock since the first evaluation. `clock` is injectable for tests."""

    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()
        self.seconds = float(seconds)
        self._clock = clock
        self._start: float | None = None

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        now = self._clock()
        if self._start is None:
            self._start = now
        return (
            Stop("timeout", {"seconds": self.seconds})
            if (now - self._start) >= self.seconds
            else None
        )

    async def reset(self) -> None:
        await super().reset()
        self._start = None


class ExternalTermination(TerminationCondition):
    """Stop when flipped from outside via `.set()` (graceful externally-triggered stop).

    **Not cloned per drive.** Every other condition is deep-copied into a
    drive-local variable so two concurrent runs cannot race on
    ``MaxTurns.turn`` or ``Timeout._start``. Applying that to this one broke
    the only thing it does: the caller holds a handle, the RUN holds a copy,
    and ``set()`` on the handle never reached the loop — an external stop that
    could only ever work if it was flipped before the run started, which is
    not what "externally-triggered" means.

    So ``__deepcopy__`` returns ``self``. An external stop switch is by
    definition not per-run state; sharing it is the semantics, not a leak. The
    consequence is deliberate and worth knowing: one switch shared by two
    concurrent drives stops both, and either drive's ``reset()`` lowers it for
    both. Use one condition per run if you need to stop them independently.
    """

    def __init__(self) -> None:
        super().__init__()
        self._flag = False

    def __deepcopy__(self, memo: dict[int, Any]) -> ExternalTermination:
        memo[id(self)] = self
        return self

    def set(self) -> None:
        self._flag = True

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        return Stop("external") if self._flag else None

    async def reset(self) -> None:
        await super().reset()
        self._flag = False


class FunctionalTermination(TerminationCondition):
    """Stop when a custom predicate over the delta is true. The predicate may be sync OR async, and may
    optionally take a second `ctx` argument — async + ctx is the hook for an LLM judge ("is the task done?")
    that must run on the LIVE run context (see `judge_termination`)."""

    def __init__(
        self, predicate: Callable[..., bool | Awaitable[bool]], *, reason: str = "functional"
    ) -> None:
        super().__init__()
        self._pred = predicate
        self.reason = reason
        try:  # pass ctx only if the predicate accepts a 2nd arg
            self._arity = len(inspect.signature(predicate).parameters)
        except (ValueError, TypeError):  # builtins / C-callables → assume the 1-arg form
            self._arity = 1

    async def _evaluate(self, delta: Sequence[Message], ctx: Ctx | None = None) -> Stop | None:
        r = self._pred(delta, ctx) if self._arity >= 2 else self._pred(delta)
        if inspect.isawaitable(r):
            r = await r
        return Stop(self.reason) if r else None


def judge_termination(
    judge: Any,
    ctx: Ctx | None = None,
    *,
    reason: str = "judged_complete",
    question: str = "Is the user's task fully complete? Answer YES or NO.",
    yes: str = "YES",
) -> FunctionalTermination:
    """Build a `TerminationCondition` that asks `judge` (an `Agent`-like) whether the task is done, on the
    new-message delta each turn. Stops only on an explicit affirmative — anything else (incl. an error or
    ambiguity) continues, so the hard ceiling stays the real backstop.

    "Explicit affirmative" means the reply LEADS with ``yes`` — leading
    punctuation and whitespace are skipped, matching is case-insensitive, and a
    word boundary must follow. The substring test this used to do
    (``yes.upper() in out.upper()``) read both "Not yet — yesterday's draft is
    still open." and "There is no simple yes/no answer here." as affirmative,
    which is the exact opposite of each, and negation and hedging are what a
    judge produces most.

    Leading rather than anywhere, because the prompt asks the judge to *answer*
    YES or NO and the answer is the first thing it writes. The bias is
    deliberate and matches the paragraph above: "Answer: YES" reads as a
    non-stop and the loop continues one more turn, which the hard ceiling
    bounds anyway. A false stop has no such backstop — it silently truncates
    the work and returns it as complete.

    The judge runs on the **live** run ctx threaded in each turn (correct budget/cancel/trace even when the
    coordinator is reused across runs); the optional `ctx` here is only a fallback for the rare caller that
    evaluates the condition directly without threading one."""

    def _render(transcript: list[Message]) -> str:
        return "\n\n".join(f"[{m.name or m.role}] {m.content}" for m in transcript)

    async def _done(delta: Sequence[Message], live_ctx: Ctx | None = None) -> bool:
        run_ctx = live_ctx or ctx
        assert run_ctx is not None, "judge_termination needs a live Ctx"
        prompt = f"{question}\n\nLatest messages:\n{_render(list(delta))}"
        try:
            out = (await judge.run(prompt, run_ctx.child())).output or ""
        except Exception:  # noqa: BLE001 — a judge failure must never force a stop
            return False
        # ``\W*`` skips leading punctuation/whitespace ("**YES**", "  yes").
        # ``re.escape`` because ``yes=`` is caller-supplied and may hold regex
        # metacharacters; ``(?!\w)`` rather than ``\b`` so a multi-word phrase
        # like ``yes="task complete"`` anchors correctly at its end.
        return re.match(rf"\W*{re.escape(yes)}(?!\w)", out, re.IGNORECASE) is not None

    return FunctionalTermination(_done, reason=reason)


__all__ = [
    "ExternalTermination",
    "FunctionCall",
    "FunctionalTermination",
    "MaxMessages",
    "MaxTurns",
    "SourceMatch",
    "Stop",
    "TerminationCondition",
    "TextMention",
    "Timeout",
    "judge_termination",
]
