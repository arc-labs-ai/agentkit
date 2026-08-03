"""PlanPolicy — supervisor for a plan of named-child Steps.

Groups run sequentially (lower group first); steps within a group run concurrently
(bounded by the tree semaphore). Merges ``Usage``. ``best_effort=False`` (the default)
cancels its group and raises on any step failure; ``best_effort=True`` isolates failures
into ``errors``.

A **human-gate step** (``Step.gate("review")``) is a coordinator-level suspend point:
reaching it checkpoints the accumulated results/errors/usage to ``ctx.store`` under a
``plan_policy:<run_id>`` key, then returns an ``AgentResult`` with
``stop_reason="awaiting_decision"`` and a ``Suspended`` in ``evals["suspended"]``. The
run resumes via :meth:`PlanPolicy.resume` — an ``approve`` decision continues at the next
group (and reclaims the checkpoint); a ``reject`` (or missing decision) returns a
terminal ``stop_reason="rejected"`` result without running further groups. This is the
``PlanPolicy`` counterpart of ``Workflow.human_gate`` / ``Workflow.resume``.

Use the plain leaf ``Agent`` for a single ReAct loop; ``PlanPolicy`` for variable,
phase-based plans dispatched across named children.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentkit.agents.policies.roundrobin import _emit_policy_dispatch
from agentkit.agents.result import AgentResult, Suspended
from agentkit.context import WorkingContext
from agentkit.kernel.concurrency import run_agents
from agentkit.kernel.errors import Failure
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Usage

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent


def _plan_dispatch(ctx: Ctx, *, policy: str, child: str, reason: str) -> None:
    """Anchor for the dispatch helper — exists at module scope so the formatter
    sees the imported helper used and keeps the import in place."""
    _emit_policy_dispatch(ctx, policy=policy, child=child, reason=reason)


def _ckpt_key(run_id: str) -> str:
    """Store key for a suspended PlanPolicy run's checkpoint (mirrors
    ``Workflow``'s namespace scheme so the two can coexist in one store)."""
    return f"plan_policy:{run_id}"


@dataclass(frozen=True)
class Step:
    """A planned step: dispatch ``input`` to the child named ``agent`` in ``group``.

    A step whose ``gate_name`` is set (with ``agent=None``) is a **human-gate**: reaching
    its group during execution suspends the plan and returns a ``Suspended`` result
    instead of dispatching. Build one with :meth:`gate`.

    ``agent`` defaults to ``None`` so gate steps are constructible via
    ``Step.gate(...)``; regular dispatch steps still use the same positional shape
    ``Step("researcher", "q1", group=0)`` and reject a ``None`` agent at dispatch time
    (``children[None]`` would raise ``KeyError``).
    """

    agent: str | None = None
    input: str = ""
    group: int = 0
    gate_name: str | None = None

    @classmethod
    def gate(cls, name: str, *, group: int = 0) -> Step:
        """Build a **human-gate step**. When ``PlanPolicy.execute`` reaches this step's
        group, execution checkpoints and returns a ``Suspended`` result; resume with
        :meth:`PlanPolicy.resume` and a decision dict ``{name: "approve" | "reject"}``.

        ``group`` places the gate between plan phases: for a
        ``[researchers @0, gate("review") @1, synth @2]`` plan the researchers all run,
        the gate suspends, and only an ``approve`` decision lets the synthesizer run.
        """
        return cls(agent=None, input="", group=group, gate_name=name)


@runtime_checkable
class Planner(Protocol):
    """Returns the ordered list of ``Step``\\ s for a goal. May be sync or async; the
    Policy detects via the result type."""

    def plan(self, goal: str, ctx: Ctx) -> list[Step]: ...


@dataclass
class StaticPlanner:
    """A trivial Planner that returns its preset ``steps`` verbatim."""

    steps: list[Step]

    def plan(self, goal: str, ctx: Ctx) -> list[Step]:
        return self.steps


@dataclass
class PlanPolicy:
    """Dispatches a plan of named-child steps. The plan comes from ``planner``
    (required when ``steps=`` is not supplied at ``execute`` time).

    ``best_effort=False`` (default) fail-fast: a failing step cancels its group and
    raises. ``best_effort=True``: each slot is a result OR a
    :class:`~agentkit.kernel.errors.Failure` wrapping the raised exception, so partial
    progress survives — failures land in ``AgentResult.evals['errors']`` as
    ``(child_name, failure)`` tuples (the ``Failure`` carries the originating exception
    on ``.cause``).

    **Human-gate**: a ``Step.gate("name")`` step suspends the plan at its group and
    checkpoints to ``ctx.store``; resume via :meth:`resume`. Mirrors
    ``Workflow.human_gate`` — the coordinator-level counterpart of the same primitive.
    """

    planner: Planner | None = None
    best_effort: bool = False
    name: str = "plan"

    async def execute(
        self,
        coordinator: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
        *,
        steps: list[Step] | None = None,
    ) -> AgentResult:
        children = getattr(coordinator.cognition, "children", None) or {}
        if steps is None:
            if self.planner is None:
                raise ValueError("PlanPolicy needs either `steps=` or a `planner=` to plan the run")
            # Honour the Planner Protocol's "may be sync or async"
            # contract. Iterating ``self.planner.plan(...)`` verbatim
            # would fail for an async planner: the coroutine object is
            # not iterable and the ``for`` loop below would raise
            # ``TypeError``. Mirrors ``LedgerPolicy._make_plan``.
            plan = self.planner.plan(task, ctx)
            steps = await plan if inspect.isawaitable(plan) else plan

        total = len(steps)
        # Drop one ``policy.dispatch`` event per planned step (in plan order),
        # naming the child + its 1-based step position so an operator can read
        # the trace as a plan: "step 1/3 → researcher, step 2/3 → critic, ...".
        # The event lands before the group fires so concurrent groups still
        # show all their steps' dispatches in plan order. Gate steps announce
        # under a synthetic ``gate:<name>`` identifier so the operator sees
        # the pause explicitly in the trace.
        for i, step in enumerate(steps, start=1):
            target = step.agent if step.gate_name is None else f"gate:{step.gate_name}"
            _plan_dispatch(
                ctx,
                policy=type(self).__name__,
                child=target or "<gate>",
                reason=f"plan step {i}/{total}",
            )

        return await self._run_groups(
            children,
            task,
            ctx,
            steps,
            results=[],
            errors=[],
            usage=Usage(),
            start_group=None,
        )

    async def resume(
        self,
        coordinator: Agent,
        decisions: dict[str, str],
        ctx: Ctx,
    ) -> AgentResult:
        """Resume a plan suspended at a human-gate step.

        Loads the checkpoint keyed by ``ctx.correlation_id`` from ``ctx.store``. If the
        gate's decision in ``decisions`` is ``"approve"``, the checkpoint is deleted and
        execution continues from the group AFTER the gate. Any other value — including a
        missing key — is treated as a rejection: the checkpoint is deleted (so a stale
        run can't be re-resumed) and an ``AgentResult`` with ``stop_reason="rejected"``
        is returned; no further groups run.

        Takes ``coordinator`` as its first argument (mirroring :meth:`execute`) because
        the children roster isn't part of the checkpoint — the caller re-supplies the
        coordinator on resume, matching how ``Workflow.resume`` re-supplies the graph.
        """
        if ctx.store is None:
            raise ValueError("PlanPolicy.resume requires ctx.store")

        run_id = ctx.correlation_id
        saved = await ctx.store.get(_ckpt_key(run_id))
        if not saved:
            raise ValueError(f"no suspended plan {run_id!r} to resume")

        task: str = saved["task"]
        steps: list[Step] = list(saved["steps"])
        results: list[AgentResult] = list(saved["results"])
        errors: list[Any] = list(saved["errors"])
        usage: Usage = saved["usage"]
        gate_group: int = saved["gate_group"]

        # Find the gate step at ``gate_group`` and look up its decision.
        gate_step = next(
            (s for s in steps if s.group == gate_group and s.gate_name is not None), None
        )
        if gate_step is None or gate_step.gate_name is None:
            # A checkpoint without a discoverable gate is corrupt; surface it as data
            # rather than falling through to a silent no-op resume.
            raise ValueError(
                f"checkpoint for run {run_id!r} references gate_group={gate_group} "
                "but no gate step lives there"
            )
        gate_name = gate_step.gate_name
        decision = decisions.get(gate_name)

        # Any non-approve decision (including missing) terminates the plan. Clear the
        # checkpoint on the way out so a stale run can't be re-resumed under the same id.
        if decision != "approve":
            await ctx.store.delete(_ckpt_key(run_id))
            last = results[-1].output if results else ""
            return AgentResult(
                output=last,
                usage=usage,
                evals={
                    "results": results,
                    "errors": errors,
                    "stop_reason": "rejected",
                    "gate": gate_name,
                    "decision": decision,
                },
            )

        # Approve → clear the checkpoint FIRST so a mid-resume crash doesn't leave a
        # phantom pending gate that would replay on the next resume attempt.
        await ctx.store.delete(_ckpt_key(run_id))

        children = getattr(coordinator.cognition, "children", None) or {}
        return await self._run_groups(
            children,
            task,
            ctx,
            steps,
            results=results,
            errors=errors,
            usage=usage,
            start_group=gate_group + 1,  # advance past the gate group
        )

    async def _run_groups(
        self,
        children: dict[str, Agent],
        task: str,
        ctx: Ctx,
        steps: list[Step],
        *,
        results: list[AgentResult],
        errors: list[Any],
        usage: Usage,
        start_group: int | None,
    ) -> AgentResult:
        """Core dispatch loop, shared by :meth:`execute` and :meth:`resume`. Iterates
        the plan's groups in ascending order (optionally skipping past ``start_group``).
        A gate step in the current group triggers a checkpoint + ``Suspended`` return.
        """
        groups = sorted({s.group for s in steps})
        if start_group is not None:
            groups = [g for g in groups if g >= start_group]

        for group in groups:
            group_steps = [s for s in steps if s.group == group]
            gate_step = next((s for s in group_steps if s.gate_name is not None), None)
            if gate_step is not None:
                # A group with any gate step suspends BEFORE any of its steps run —
                # gates and dispatch steps do not co-execute inside one group.
                gate_name = gate_step.gate_name
                assert gate_name is not None  # narrowing for mypy — see filter above
                run_id = ctx.correlation_id
                if ctx.store is not None:
                    await ctx.store.set(
                        _ckpt_key(run_id),
                        {
                            "task": task,
                            "steps": list(steps),
                            "results": list(results),
                            "errors": list(errors),
                            "usage": usage,
                            "gate_group": group,
                        },
                    )
                await ctx.emit(
                    "interrupt",
                    f"awaiting decision: {gate_name}",
                    payload={"gate": gate_name},
                )
                susp = Suspended(run_id=run_id, pending=(gate_name,), reason="awaiting_decision")
                last = results[-1].output if results else ""
                return AgentResult(
                    output=last,
                    usage=usage,
                    evals={
                        "results": results,
                        "errors": errors,
                        "stop_reason": "awaiting_decision",
                        "suspended": susp,
                    },
                )

            pairs = [(children[s.agent], s.input) for s in group_steps]  # type: ignore[index]
            outs = await run_agents(pairs, ctx, best_effort=self.best_effort)
            for step, res in zip(group_steps, outs, strict=False):
                # ``gather_best_effort`` wraps a raised exception into a
                # ``Failure`` (first-class error data with source/cause/
                # category) — that is now the correct guard for a failed
                # slot. ``isinstance(res, BaseException)`` would silently
                # miss it because ``Failure`` is a plain dataclass.
                if isinstance(res, Failure):
                    errors.append((step.agent, res))
                    continue
                results.append(res)
                usage = usage + res.usage

        last = results[-1].output if results else ""
        return AgentResult(
            output=last,
            usage=usage,
            evals={
                "results": results,
                "errors": errors,
                "stop_reason": "plan_complete",
            },
        )


__all__ = ["PlanPolicy", "Planner", "StaticPlanner", "Step"]
