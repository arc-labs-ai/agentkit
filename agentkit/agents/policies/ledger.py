"""LedgerPolicy — stall-aware supervisor with task + progress ledgers.

Holds two ledgers: the **task ledger** (goal + facts + plan, stable, revised on
re-plan) and the **progress ledger** (re-derived each round — satisfied? looping?
who acts next?). Routes to the assessed ``next_speaker``; on a detected stall it
**re-plans** (bounded by ``max_replans``) instead of burning iterations.
``max_rounds`` is the hard backstop.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.agents.policies.roundrobin import _emit_policy_dispatch
from agentkit.agents.result import AgentResult
from agentkit.context import WorkingContext
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message, Usage

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent


def _ledger_dispatch(ctx: Ctx, *, policy: str, child: str, reason: str) -> None:
    """Anchor for the dispatch helper — exists at module scope so the formatter
    sees the imported helper used and keeps the import in place."""
    _emit_policy_dispatch(ctx, policy=policy, child=child, reason=reason)


@dataclass
class Task:
    """A first-class unit of work: goal + status, optionally decomposed into subtasks."""

    goal: str
    id: str = ""
    subtasks: list[Task] = field(default_factory=list)
    status: str = "pending"  # pending | active | blocked | done | abandoned
    owner: str | None = None
    result: Any = None

    def subtask(self, goal: str, **kw: Any) -> Task:
        t = Task(goal=goal, **kw)
        self.subtasks.append(t)
        return t

    def mark(self, status: str, *, result: Any = None, owner: str | None = None) -> Task:
        self.status = status
        if result is not None:
            self.result = result
        if owner is not None:
            self.owner = owner
        return self

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "abandoned")

    @property
    def done_count(self) -> int:
        return sum(1 for s in self.subtasks if s.status == "done")


@dataclass
class TaskLedger:
    """The plan: goal + known facts + the ordered steps. Stable; ``revise`` on re-plan."""

    goal: str
    facts: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)

    def revise(
        self, *, facts: list[str] | None = None, plan: list[str] | None = None
    ) -> TaskLedger:
        if facts:
            self.facts.extend(facts)
        if plan is not None:
            self.plan = list(plan)
        return self


@dataclass
class ProgressLedger:
    """Re-derived each round: is the run satisfied / looping / moving, and who acts next."""

    satisfied: bool = False
    in_a_loop: bool = False
    progress_being_made: bool = True
    next_speaker: str = ""
    instruction: str = ""


def _render(transcript: list[Message]) -> str:
    return "\n\n".join(f"[{m.name or m.role}] {m.content}" for m in transcript)


def heuristic_assessor(
    *, done_marker: str = "DONE"
) -> Callable[[TaskLedger, list[Message], list[str]], ProgressLedger]:
    """A deterministic default progress assessor: satisfied when the last reply
    contains ``done_marker``; ``in_a_loop`` when the last two assistant replies are
    identical (no progress); otherwise round-robins ``next_speaker``."""

    def assess(ledger: TaskLedger, transcript: list[Message], names: list[str]) -> ProgressLedger:
        replies = [m for m in transcript if m.role == "assistant"]
        last = replies[-1].content if replies else ""
        in_a_loop = len(replies) >= 2 and replies[-1].content == replies[-2].content
        nxt = names[len(replies) % len(names)] if names else ""
        return ProgressLedger(
            satisfied=done_marker in last,
            in_a_loop=in_a_loop,
            progress_being_made=not in_a_loop,
            next_speaker=nxt,
            instruction=ledger.plan[0] if ledger.plan else ledger.goal,
        )

    return assess


@dataclass
class LedgerPolicy:
    """Stall-aware supervisor: plan → assess progress → route to ``next_speaker``,
    or re-plan on stall — bounded by ``max_replans`` and the ``max_rounds`` ceiling."""

    assessor: Any = None  # (ledger, transcript, names) -> ProgressLedger (sync/async)
    planner: Any = None  # (goal) -> list[str] plan (sync/async); default → [goal]
    max_rounds: int = 20
    max_replans: int = 3
    name: str = "ledger"

    def __post_init__(self) -> None:
        self.assessor = self.assessor or heuristic_assessor()

    async def _make_plan(self, goal: str) -> list[str]:
        if self.planner is None:
            return [goal]
        plan = self.planner(goal)
        return list(await plan if inspect.isawaitable(plan) else plan)

    async def execute(
        self,
        coordinator: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AgentResult:
        children = getattr(coordinator.cognition, "children", None) or {}
        ledger = TaskLedger(goal=task, plan=await self._make_plan(task))
        transcript: list[Message] = [Message("user", task)]
        results: list[AgentResult] = []
        usage = Usage()
        names = list(children.keys())
        replans = 0
        stop_reason = "max_rounds"

        await ctx.emit("run_start", f"ledger {coordinator.name}", payload={"plan": ledger.plan})
        for _ in range(self.max_rounds):
            ctx.check_cancelled()
            prog = self.assessor(ledger, transcript, names)
            if inspect.isawaitable(prog):
                prog = await prog
            await ctx.emit(
                "progress",
                f"satisfied={prog.satisfied} loop={prog.in_a_loop}",
                payload={"next": prog.next_speaker, "replans": replans},
            )

            if prog.satisfied:
                stop_reason = "satisfied"
                break
            if prog.in_a_loop or not prog.progress_being_made:
                if replans >= self.max_replans:
                    stop_reason = "stalled"
                    break
                replans += 1
                ledger.revise(plan=await self._make_plan(task), facts=[f"replan #{replans}"])
                await ctx.emit("summary", f"re-planned (#{replans})", payload={"replans": replans})
                continue

            agent = children.get(prog.next_speaker)
            if agent is None and names:
                agent = children[names[len(results) % len(names)]]
            if agent is None:
                stop_reason = "no_children"
                break
            aname = getattr(agent, "name", prog.next_speaker or "agent")
            _ledger_dispatch(
                ctx,
                policy=type(self).__name__,
                child=aname,
                reason=f"ledger assessor picked {prog.next_speaker!r}",
            )
            prompt = _render(transcript) + (
                f"\n\n[instruction] {prog.instruction}" if prog.instruction else ""
            )
            res = await agent.run(prompt, ctx.child())
            results.append(res)
            usage = usage + res.usage
            transcript.append(Message("assistant", res.output, name=aname))

        await ctx.emit(
            "result",
            f"ledger finished: {stop_reason}",
            payload={"stop_reason": stop_reason, "replans": replans},
        )
        last = results[-1].output if results else ""
        return AgentResult(
            output=last,
            usage=usage,
            evals={
                "stop_reason": stop_reason,
                "ledger": ledger,
                "messages": transcript,
                "results": results,
                "replans": replans,
            },
        )


__all__ = [
    "LedgerPolicy",
    "ProgressLedger",
    "Task",
    "TaskLedger",
    "heuristic_assessor",
]
