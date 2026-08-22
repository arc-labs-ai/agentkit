"""PlanPolicy — supervisor for a plan of named-child Steps.

Groups run sequentially (lower group first); steps within a group run concurrently
(bounded by the tree semaphore). Merges ``Usage``. ``best_effort=False`` (the default)
cancels its group and raises on any step failure; ``best_effort=True`` isolates failures
into ``errors``.

A **human-gate step** (``Step.gate("review")``) is a coordinator-level suspend point:
reaching it checkpoints the accumulated results/errors/usage through the shared
``Checkpointer`` seam (``resolve_checkpointer``: an explicit ``checkpointer=`` on the
cognition, else ``ctx.checkpointer``, else a bridge over ``ctx.store``) at the plan's own
slot ``{run_id}:plan``, then returns an ``AgentResult`` with
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

from agentkit.agents.policies.roundrobin import _emit_policy_dispatch, _resolve_checkpointer
from agentkit.agents.result import AgentResult, Suspended, stop_reason_for
from agentkit.capabilities.checkpointer.persistence import (
    result_to_dict,
    usage_to_dict,
)
from agentkit.context import WorkingContext
from agentkit.kernel._json import dumps as _json_dumps
from agentkit.kernel.concurrency import run_agents
from agentkit.kernel.errors import Failure
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.resilience import ErrorClass
from agentkit.kernel.types import Usage

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.capabilities.checkpointer import Checkpointer


def _plan_dispatch(ctx: Ctx, *, policy: str, child: str, reason: str) -> None:
    """Anchor for the dispatch helper — exists at module scope so the formatter
    sees the imported helper used and keeps the import in place."""
    _emit_policy_dispatch(ctx, policy=policy, child=child, reason=reason)


def _ckpt_key(run_id: str) -> str:
    """LEGACY store key for a suspended plan, written before this policy moved
    onto the shared ``Checkpointer`` seam. Still READ on resume so a run that
    suspended across an upgrade can finish; no longer written."""
    return f"plan_policy:{run_id}"


def checkpoint_slot(run_id: str) -> str:
    """The durable slot for a suspended plan.

    Namespaced per producer, like every other slot: a coordinator's tool-looping
    children write at ``{run_id}:agent:{name}``, so a plan at ``{run_id}:plan``
    cannot be clobbered by a child completing (which deletes its own slot) and
    cannot clobber a nested coordinator's state.
    """
    return f"{run_id}:plan"


# Bump when the encoded shape changes incompatibly. A payload with no ``v`` is a
# pre-encoding record holding LIVE objects (see ``_decode_plan_state``).
_CKPT_VERSION = 1


def _warn_unpersisted_gate(gate: str, run_id: str) -> None:
    """Announce a plan suspend that cannot be resumed.

    Reaching a gate with no durable seam used to return a ``Suspended`` in
    silence: the caller got a run id and a pending gate name, and every
    ``resume`` for it would raise "no suspended plan". Mirrors
    ``Workflow``'s warning for the same situation.
    """
    import warnings

    warnings.warn(
        f"plan gate {gate!r} suspended run {run_id!r} but no durable seam is wired on the "
        "RunContext, so NOTHING WAS PERSISTED and PlanPolicy.resume() will raise "
        f'"no suspended plan {run_id!r} to resume". Wire either '
        "Services(checkpointer=Checkpointer(port=...)) or Services(store=...) to make this "
        "suspend resumable.",
        UserWarning,
        stacklevel=4,
    )


def _step_to_dict(s: Step) -> dict[str, Any]:
    return {"agent": s.agent, "input": s.input, "group": s.group, "gate_name": s.gate_name}


def _dict_to_step(d: dict[str, Any]) -> Step:
    return Step(
        agent=d.get("agent"),
        input=d.get("input", ""),
        group=int(d.get("group", 0)),
        gate_name=d.get("gate_name"),
    )


def _failure_to_dict(name: str | None, f: Failure) -> dict[str, Any]:
    """Encode a step failure. ``Failure.cause`` is a live exception and cannot
    cross a wire — the message it produced is kept, the traceback is not."""
    return {
        "agent": name,
        "category": str(getattr(f.category, "value", f.category)),
        "source": f.source,
        "message": f.message,
        "retriable": bool(f.retriable),
    }


def _dict_to_failure(d: dict[str, Any]) -> tuple[str | None, Failure]:
    return (
        d.get("agent"),
        Failure(
            category=ErrorClass(d.get("category", ErrorClass.UNKNOWN.value)),
            source=d.get("source", "PlanPolicy"),
            message=d.get("message", ""),
            retriable=bool(d.get("retriable", False)),
        ),
    )


def _encode_plan_state(
    *,
    task: str,
    steps: list[Step],
    results: list[AgentResult],
    errors: list[Any],
    usage: Usage,
    gate_group: int,
) -> dict[str, Any]:
    """Encode a suspended plan into a JSON-SAFE payload.

    Encoding always happens, even on an in-memory store that would happily hold
    the live dataclasses. Two reasons. It kept a real bug invisible: the old
    code put ``Step`` / ``Usage`` / ``AgentResult`` objects straight into
    ``ctx.store.set``, so every test passed on ``InMemoryStore`` while any
    durable store raised ``TypeError: Object of type Step is not JSON
    serializable`` — the human-gate feature simply did not work on the
    persistence people actually deploy. And encoding unconditionally keeps the
    in-memory tests honest about the wire contract, the same argument
    ``Checkpointer.snapshot`` makes for deep-copying state.

    A child result's ``evals`` / ``parsed`` can hold ANYTHING (a Pydantic model,
    a ``Message`` list from a nested coordinator). Rather than let the suspend
    itself explode — which loses the whole run, the worst possible outcome at a
    gate — those two fields are dropped, once, with a warning, if the payload
    will not serialize without them.
    """
    payload: dict[str, Any] = {
        "v": _CKPT_VERSION,
        "task": task,
        "steps": [_step_to_dict(s) for s in steps],
        "results": [result_to_dict(r) for r in results],
        "errors": [
            _failure_to_dict(n, f) if isinstance(f, Failure) else {"agent": n, "message": str(f)}
            for n, f in errors
        ],
        "usage": usage_to_dict(usage),
        "gate_group": gate_group,
    }
    if _serializable(payload):
        return payload

    import warnings

    for rec in payload["results"]:
        rec.pop("evals", None)
        rec.pop("parsed", None)
    warnings.warn(
        "a suspended plan carried child results whose `evals`/`parsed` could not be "
        "serialized, so those two fields were dropped from the checkpoint. The plan will "
        "resume and finish; the pre-gate results it hands back will have empty `evals` and "
        "`parsed=None`. Return JSON-safe values from `output=` parsers to keep them.",
        UserWarning,
        stacklevel=3,
    )
    return payload


def _serializable(payload: dict[str, Any]) -> bool:
    try:
        _json_dumps(payload)
    except (TypeError, ValueError):
        return False
    return True


def _decode_plan_state(
    state: dict[str, Any],
) -> tuple[str, list[Step], list[AgentResult], list[Any], Usage, int]:
    """Inverse of :func:`_encode_plan_state`, tolerating a PRE-encoding record.

    A payload with no ``"v"`` was written by the old code path and holds live
    objects — an in-memory store round-tripped them fine, so runs suspended
    before this change must still resume rather than crash on a dict lookup
    into a ``Step``.
    """
    from agentkit.capabilities.checkpointer.persistence import dict_to_result

    task = state["task"]
    gate_group = int(state["gate_group"])
    if state.get("v") is None:  # legacy: live objects, no encoding
        return (
            task,
            list(state["steps"]),
            list(state["results"]),
            list(state["errors"]),
            state["usage"],
            gate_group,
        )
    u = state.get("usage") or {}
    return (
        task,
        [_dict_to_step(d) for d in state.get("steps", [])],
        [dict_to_result(d) for d in state.get("results", [])],
        [_dict_to_failure(d) for d in state.get("errors", [])],
        Usage(u.get("input", 0), u.get("output", 0), u.get("cost", 0.0)),
        gate_group,
    )


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


class PlanShapeError(ValueError):
    """A plan that cannot be executed as written.

    Raised at the START of ``execute`` / ``resume``, before any child is
    dispatched, because every alternative is worse: a plan whose step 5 names a
    child that does not exist used to raise a bare ``KeyError('reseacher')``
    from inside the dispatch loop, AFTER steps 1-4 had run and spent money, and
    with their results unreachable — the accumulator is a local of
    ``_run_groups``. Under ``best_effort=True`` that also broke the mode's one
    promise, which is that partial progress survives.

    Subclasses ``ValueError`` so the existing ``except ValueError`` around plan
    construction keeps catching it.
    """


def _validate_plan(
    steps: list[Step], children: dict[str, Agent], *, best_effort: bool
) -> tuple[list[Step], list[tuple[str | None, Failure]]]:
    """Check a plan against the child roster BEFORE anything is dispatched.

    Returns the steps to execute plus per-step failures for the ones dropped.

    Two shapes are refused outright, in both modes, because there is no honest
    way to guess what was meant:

    * A **malformed step** — neither an ``agent`` nor a ``gate_name``. There is
      nothing to dispatch and nothing to wait for.
    * A **gate sharing a group with dispatch steps**. Reaching such a group
      suspends before any of its steps run (gates and work do not co-execute),
      and resume then continues at ``gate_group + 1`` — so those co-grouped
      steps were announced in the trace and then silently never ran, in either
      branch of the decision. Whether the work belongs before or after the
      human's decision is exactly what the plan failed to say, so the framework
      states the problem rather than picking one.

    An **unknown child name** is the one case that depends on the mode, because
    it is the one case that can legitimately be runtime data rather than a typo:
    a live ``Planner`` names the child it wants. Under ``best_effort=True`` it
    becomes a ``Failure`` in ``evals["errors"]`` and the rest of the plan runs;
    otherwise it is refused up front.
    """
    dropped: list[tuple[str | None, Failure]] = []
    keep: list[Step] = []
    roster = sorted(children)

    for group in sorted({s.group for s in steps}):
        in_group = [s for s in steps if s.group == group]
        gates = [s for s in in_group if s.gate_name is not None]
        work = [s for s in in_group if s.gate_name is None]
        if gates and work:
            raise PlanShapeError(
                f"plan group {group} mixes the human gate "
                f"{gates[0].gate_name!r} with {len(work)} dispatch step(s) "
                f"({', '.join(repr(s.agent) for s in work)}). A gate suspends its whole group "
                "before any step runs, and resume continues at the NEXT group, so those steps "
                "would never run. Move them to a group before the gate (to run first) or after "
                "it (to run only on approval)."
            )

    for step in steps:
        if step.gate_name is not None:
            keep.append(step)
            continue
        if step.agent is None:
            raise PlanShapeError(
                f"plan step {step!r} names neither an agent nor a gate — there is nothing to "
                "dispatch. Use Step('<child>', '<input>') or Step.gate('<name>')."
            )
        if step.agent not in children:
            msg = (
                f"plan step for child {step.agent!r} has no such child on the coordinator "
                f"(known: {roster or ['<none>']})"
            )
            if not best_effort:
                raise PlanShapeError(
                    msg + ". Fix the plan, or set best_effort=True to isolate unknown children "
                    "into evals['errors'] and run the rest."
                )
            # PERMANENT: retrying a name that is not on the roster cannot
            # succeed, and ``Failure.retriable`` is read by callers deciding
            # whether to re-dispatch.
            dropped.append(
                (
                    step.agent,
                    Failure(category=ErrorClass.PERMANENT, source="PlanPolicy", message=msg),
                )
            )
            continue
        keep.append(step)

    return keep, dropped


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

        # Validate against the roster BEFORE the dispatch events are emitted:
        # announcing a step that will never run puts a lie in the trace.
        steps, dropped = _validate_plan(steps, children, best_effort=self.best_effort)

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
            errors=list(dropped),
            usage=Usage(),
            start_group=None,
            cp=_resolve_checkpointer(coordinator, ctx),
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
        run_id = ctx.correlation_id
        cp = _resolve_checkpointer(coordinator, ctx)
        if cp is None:
            raise ValueError(
                "PlanPolicy.resume requires a durable seam: wire "
                "Services(checkpointer=Checkpointer(port=...)) or Services(store=...)"
            )

        # A record written by the pre-upgrade code lives under the old raw
        # store key. Read it as a fallback so a plan that suspended across the
        # upgrade can still finish, and remember which seam it came from so the
        # right one is cleared below.
        legacy = False
        found = await cp.resume(checkpoint_slot(run_id))
        saved: dict[str, Any] | None = found.state if found is not None else None
        if not saved and ctx.store is not None:
            saved = await ctx.store.get(_ckpt_key(run_id))
            legacy = saved is not None
        if not saved:
            raise ValueError(f"no suspended plan {run_id!r} to resume")

        task, steps, results, errors, usage, gate_group = _decode_plan_state(saved)

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
            await self._clear(cp, ctx, run_id, legacy=legacy)
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
                # A person declining is a deliberate terminal stop, not a
                # completed plan and not a failure. ``"terminated"`` is the
                # closed-taxonomy spelling; ``evals`` keeps the exact word.
                stop_reason=stop_reason_for("rejected"),
            )

        # Approve → clear the checkpoint FIRST so a mid-resume crash doesn't leave a
        # phantom pending gate that would replay on the next resume attempt.
        await self._clear(cp, ctx, run_id, legacy=legacy)

        # The roster is re-supplied by the caller on resume and may not be the
        # one the plan was built against — validate again rather than trusting
        # that the pre-suspend check still holds.
        children = getattr(coordinator.cognition, "children", None) or {}
        steps, dropped = _validate_plan(steps, children, best_effort=self.best_effort)
        errors.extend(dropped)
        return await self._run_groups(
            children,
            task,
            ctx,
            steps,
            results=results,
            errors=errors,
            usage=usage,
            start_group=gate_group + 1,  # advance past the gate group
            cp=cp,
        )

    @staticmethod
    async def _clear(
        cp: Checkpointer, ctx: Ctx, run_id: str, *, legacy: bool
    ) -> None:
        """Drop a consumed plan checkpoint from whichever seam held it."""
        await cp.delete(checkpoint_slot(run_id))
        if legacy and ctx.store is not None:
            await ctx.store.delete(_ckpt_key(run_id))

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
        cp: Checkpointer | None,
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
                if cp is None:
                    # No seam at all: the suspend is real but unrecoverable, and
                    # saying so beats handing back a run id whose every resume
                    # raises "no suspended plan".
                    _warn_unpersisted_gate(gate_name, run_id)
                else:
                    await cp.snapshot(
                        checkpoint_slot(run_id),
                        _encode_plan_state(
                            task=task,
                            steps=steps,
                            results=results,
                            errors=errors,
                            usage=usage,
                            gate_group=group,
                        ),
                        # SUSPENDED, not RUNNING: ``Checkpointer.resume``
                        # returns None for terminal statuses, and an
                        # auto-resume supervisor needs "waiting on a human"
                        # to be distinguishable from "engine in motion".
                        status=CheckpointStatus.SUSPENDED,
                        ctx=ctx,
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
                    # THE one that mattered: a plan parked on a gate must
                    # report ``is_suspended``/``is_resumable``, exactly like a
                    # tool call parked on an approval. While this defaulted to
                    # ``"complete"``, a caller branching on the typed field
                    # never prompted its human and never called ``resume``,
                    # even though the checkpoint was sitting in the store.
                    stop_reason=stop_reason_for("awaiting_decision"),
                )

            # ``children[...]`` is safe: ``_validate_plan`` ran before the first
            # dispatch and every surviving step names a child on the roster.
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
            stop_reason=stop_reason_for("plan_complete"),
        )


__all__ = ["PlanPolicy", "PlanShapeError", "Planner", "StaticPlanner", "Step", "checkpoint_slot"]
