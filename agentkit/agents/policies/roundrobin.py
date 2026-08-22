"""RoundRobinPolicy — children take turns on a shared transcript.

The coordinator Agent's ``children`` take fixed-rotation turns on a private flat
transcript (or the caller-supplied ``WorkingContext`` blackboard). Each turn renders
the transcript, runs the next child, appends its reply, emits a progress observation,
and evaluates the coordinator's termination condition on the new message. Termination
defaults to ``MaxTurns(len(children))`` when the coordinator carries none — the never-
hang backstop.

The Policy is intentionally stateless across runs: all state lives on the ``coordinator``
+ the ``context`` blackboard, so the same ``RoundRobinPolicy`` instance may be reused.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.agents.control.termination import MaxTurns, TerminationCondition
from agentkit.agents.result import AgentResult, stop_reason_for
from agentkit.capabilities.checkpointer import (
    Checkpointer,
    coord_state_from_dict,
    coord_state_to_dict,
    resolve_checkpointer,
)
from agentkit.context import WorkingContext
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message, Usage

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent


def _emit_policy_dispatch(ctx: Ctx, *, policy: str, child: str, reason: str) -> None:
    """Drop a ``policy.dispatch`` event on the currently-open span — best-effort.

    One event per child selection, naming the policy class, the selected child,
    and a human-readable reason ("round-robin index 2", "selector LLM picked Bob",
    "plan step 1/3", "ledger assessor picked Alice"). Gives operators a single
    timeline view of who-spoke-when across the coordinator loop without joining
    structured logs. Wrapped in suppress so a misbehaving tracer never breaks the
    coordinator run."""
    trace = getattr(ctx, "trace", None)
    if trace is None:
        return
    with contextlib.suppress(Exception):
        trace.add_event_to_current_span(
            "policy.dispatch",
            policy=policy,
            selected_child=child,
            selection_reason=reason,
        )


def render_transcript(transcript: list[Message], scratchpad: dict[str, Any] | None = None) -> str:
    """Render the transcript (+ scratchpad notes) as a single prompt string the next
    child sees. Shared between every policy that drives a coordinator loop."""
    body = "\n\n".join(f"[{m.name or m.role}] {m.content}" for m in transcript)
    if scratchpad:  # shared-blackboard notes, visible to all
        notes = "\n".join(f"- {k}: {v}" for k, v in scratchpad.items())
        return f"## Shared notes\n{notes}\n\n## Transcript\n{body}"
    return body


def _name_of(agent: Any, children: dict[str, Any]) -> str:
    """The transcript label for a child agent — its ``.name``, else its key in the
    children dict (so handoff/source-match read back a consistent name)."""
    name: str | None = getattr(agent, "name", None)
    if name:
        return name
    return next((k for k, a in children.items() if a is agent), "agent")


def _resolve_checkpointer(coordinator: Agent, ctx: Ctx) -> Checkpointer | None:
    """Per-coordinator ``cognition.checkpointer`` wins, then the shared order.

    Delegates to ``capabilities.checkpointer.resolve_checkpointer`` so this is
    not a THIRD resolution order alongside the tool loop's and ``Workflow``'s.
    It was: this function stopped at ``ctx.checkpointer`` and deliberately
    excluded the store bridge, on the stated grounds that "coordinator runs
    require a real Checkpointer for durability".

    That reasoning does not survive inspection. The bridge is exactly as
    durable as the store behind it (FileStore / Postgres / Redis); its only
    documented limitation is a single slot per run with no version history, and
    no policy reads history — they call ``resume`` for the latest and nothing
    else. Meanwhile the cost was real and silent: a ``Services(store=...)``
    wiring gave durable ReAct runs, durable Workflow gates, and coordinator
    runs that persisted NOTHING, with no warning. Measured: a completed
    coordinator run left zero keys in the store.

    Safe now in a way it would not have been before: the tool loop namespaces
    its slot per agent, so a coordinator writing at the run id and its children
    writing at ``{run_id}:agent:{name}`` no longer collide — which is what made
    sharing one resolution order across producers viable at all.
    """
    return resolve_checkpointer(ctx, getattr(coordinator.cognition, "checkpointer", None))


async def _replay_termination(
    termination: TerminationCondition | None,
    transcript: list[Message],
    ctx: Ctx,
) -> None:
    """Re-feed assistant deltas from a rehydrated transcript through the termination
    condition so cumulative counters catch up to where the run left off."""
    if termination is None:
        return
    await termination.reset()
    for m in transcript:
        if m.role != "assistant":
            continue
        stop = await termination([m], ctx)
        if stop is not None:
            return


def _coord_result(
    *,
    transcript: list[Message],
    results: list[AgentResult],
    usage: Usage,
    stop_reason: str,
) -> AgentResult:
    """Aggregate a coordinator run into a single ``AgentResult``. The transcript / per-
    child results / stop reason ride in ``evals`` so callers that care about the inner
    detail can introspect; ``output`` is the last assistant reply.

    ``stop_reason`` arrives free-form — the policy's own ceiling wording
    (``"max_turns"``) or a ``TerminationCondition``'s ``Stop.reason``, which may
    be anything a user wrote. It rides in ``evals`` verbatim AND is mapped onto
    the closed ``AgentResult.stop_reason`` taxonomy by
    :func:`~agentkit.agents.result.stop_reason_for`. Skipping that mapping left
    the typed field at its ``"complete"`` default, so a coordinator that ran out
    of turns reported the same terminal state as one that finished its work.
    """
    last = next(
        (m.content for m in reversed(transcript) if m.role == "assistant"),
        "",
    )
    return AgentResult(
        output=last,
        usage=usage,
        evals={
            "stop_reason": stop_reason,
            "messages": transcript,
            "results": results,
        },
        stop_reason=stop_reason_for(stop_reason),
    )


@dataclass
class RoundRobinPolicy:
    """Children speak in fixed rotation over a shared transcript. The coordinator
    Agent's ``termination`` is the smart stop on top of ``max_turns`` (the
    never-hang ceiling). When the coordinator carries no ``termination``, the
    default ``MaxTurns(len(children))`` makes the loop visit each child exactly
    once per turn.

    Optional knobs:
        max_turns: hard ceiling on the loop. Defaults to ``50`` — the never-hang
            backstop on top of the coordinator's smart termination.
        note_parser: ``Callable[[str], dict]`` — agent-side scratchpad write channel
            (e.g. ``marker_notes()``). Extracts notes from each child's reply onto
            the shared blackboard ``scratchpad``.
        compactor / compact_every: compact the blackboard's transcript every N
            turns. Only active when ``context`` is a shared blackboard (else there
            is no transcript-as-blackboard to compact).
    """

    name: str = "roundrobin"
    max_turns: int = 50
    note_parser: Callable[[str], dict[str, Any]] | None = None
    compactor: Any = None
    compact_every: int = 0
    _seq: list[str] = field(default_factory=list, init=False, repr=False)

    async def _select(
        self, turn: int, children: dict[str, Any], transcript: list[Message], ctx: Ctx
    ) -> Any:
        keys = list(children.keys())
        return children[keys[turn % len(keys)]]

    async def execute(
        self,
        coordinator: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AgentResult:
        cognition = coordinator.cognition
        children = getattr(cognition, "children", None) or {}
        if not children:
            raise ValueError(
                f"coordinator {coordinator.name!r}: cannot run a Policy with no children"
            )

        termination = getattr(cognition, "termination", None) or MaxTurns(len(children))
        run_id = ctx.correlation_id
        cpt = _resolve_checkpointer(coordinator, ctx)
        bb = context if context is not None and context.shared else context

        # ---- resume-from-checkpoint OR fresh init ------------------------------------
        saved = await cpt.resume(run_id) if cpt is not None else None
        if saved is not None and saved.status != "done":
            (
                start_turn,
                r_transcript,
                r_scratchpad,
                results,
                usage,
            ) = coord_state_from_dict(saved.state)
            if bb is not None:
                bb.messages.clear()
                bb.messages.extend(r_transcript)
                bb.scratchpad.clear()
                bb.scratchpad.update(r_scratchpad)
                transcript = bb.messages
            else:
                transcript = r_transcript
            await _replay_termination(termination, transcript, ctx)
        else:
            results = []
            usage = Usage()
            start_turn = 0
            if bb is not None:
                bb.append(Message("user", task))
                transcript = bb.messages
            else:
                transcript = [Message("user", task)]
            await termination.reset()

        stop_reason = "max_turns"
        terminated_early = False

        await ctx.emit(
            "run_start", f"coord {coordinator.name}", payload={"children": len(children)}
        )
        for turn in range(start_turn, self.max_turns):
            ctx.check_cancelled()
            agent = await self._select(turn, children, transcript, ctx)
            aname = _name_of(agent, children)
            keys = list(children.keys())
            _emit_policy_dispatch(
                ctx,
                policy=type(self).__name__,
                child=aname,
                reason=f"round-robin index {turn % len(keys)}",
            )

            prompt = render_transcript(transcript, bb.scratchpad if bb is not None else None)
            res = await agent.run(prompt, ctx.child())
            results.append(res)
            usage = usage + res.usage
            msg = Message("assistant", res.output, name=aname)
            transcript.append(msg)
            if bb is not None and self.note_parser is not None:
                for k, v in (self.note_parser(res.output) or {}).items():
                    bb.note(k, v)

            await ctx.emit(
                "summary",
                f"{aname}: {res.output[:80]}",
                agent=aname,
                payload={"turn": turn},
            )

            stop = await termination([msg], ctx)
            if stop is not None:
                stop_reason = stop.reason
                terminated_early = True
                if cpt is not None:
                    await cpt.snapshot(
                        run_id,
                        coord_state_to_dict(
                            turn=turn + 1,
                            transcript=list(transcript),
                            scratchpad=(bb.scratchpad if bb is not None else None),
                            results=results,
                            usage=usage,
                            stop_reason=stop_reason,
                            status="done",
                        ),
                        status=CheckpointStatus.DONE,
                    )
                break

            if cpt is not None:
                await cpt.snapshot(
                    run_id,
                    coord_state_to_dict(
                        turn=turn + 1,
                        transcript=list(transcript),
                        scratchpad=(bb.scratchpad if bb is not None else None),
                        results=results,
                        usage=usage,
                        stop_reason=None,
                        status="running",
                    ),
                    status=CheckpointStatus.RUNNING,
                )

            if (
                bb is not None
                and self.compactor is not None
                and self.compact_every
                and (turn + 1) % self.compact_every == 0
            ):
                bb.messages = await self.compactor.compact(bb.messages, ctx)
                transcript = bb.messages

        if not terminated_early and cpt is not None:
            await cpt.snapshot(
                run_id,
                coord_state_to_dict(
                    turn=self.max_turns,
                    transcript=list(transcript),
                    scratchpad=(bb.scratchpad if bb is not None else None),
                    results=results,
                    usage=usage,
                    stop_reason=stop_reason,
                    status="done",
                ),
                status=CheckpointStatus.DONE,
            )

        await ctx.emit(
            "result",
            f"coord finished: {stop_reason}",
            payload={"stop_reason": stop_reason},
        )
        return _coord_result(
            transcript=list(transcript),
            results=results,
            usage=usage,
            stop_reason=stop_reason,
        )


def marker_notes(marker: str = "NOTE:") -> Callable[[str], dict[str, Any]]:
    """Build a ``note_parser`` that extracts ``NOTE:key=value`` lines from a child's reply
    into scratchpad entries. The text-honest write channel for the shared blackboard."""

    def _parse(output: str) -> dict[str, Any]:
        notes: dict[str, Any] = {}
        for line in (output or "").splitlines():
            idx = line.find(marker)
            if idx == -1:
                continue
            rest = line[idx + len(marker) :].strip()
            if "=" in rest:
                key, _, value = rest.partition("=")
                notes[key.strip()] = value.strip()
        return notes

    return _parse


__all__ = ["RoundRobinPolicy", "marker_notes", "render_transcript"]
