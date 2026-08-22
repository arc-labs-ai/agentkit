"""SelectorPolicy — a Selector picks the next speaker each turn.

Same shared-transcript loop as ``RoundRobinPolicy``, but the next speaker is chosen
by a ``selector(transcript, agents)`` callable (sync or async, optionally ctx-aware).
Subsumes handoff/swarm (a selector that reads the last message for a target) and
LLM-driven selection (an async selector that asks a model). Returning an unknown/none
name falls back to round-robin so a flaky selector never stalls the run.

Also hosts ``llm_selector`` — the LLM-driven who-speaks-next selector — and
``handoff_selector`` — the marker-only legacy selector.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.agents.control.selector import Selector
from agentkit.agents.control.termination import MaxTurns, TerminationCondition
from agentkit.agents.policies.roundrobin import (
    _coord_result,
    _emit_policy_dispatch,
    _name_of,
    _replay_termination,
    _resolve_checkpointer,
    _run_local_termination,
    render_transcript,
)
from agentkit.agents.result import AgentResult
from agentkit.capabilities.checkpointer import coord_state_from_dict, coord_state_to_dict
from agentkit.context import WorkingContext
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message, Usage

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent


def _selector_dispatch(ctx: Ctx, *, policy: str, child: str, reason: str) -> None:
    """Thin re-export of ``_emit_policy_dispatch`` — anchored here so the import
    is unambiguously used inside this module's call sites below."""
    _emit_policy_dispatch(ctx, policy=policy, child=child, reason=reason)


@dataclass
class SelectorPolicy:
    """A ``Selector`` picks the next child each turn. Sync OR async; optionally
    ctx-aware (a third arg). Returns the next child's name or ``None`` to fall
    back to round-robin.

    Optional knobs:
        max_turns: hard ceiling on the loop. Defaults to ``50``.
        note_parser: ``Callable[[str], dict]`` — agent-side scratchpad write channel.
        compactor / compact_every: same as ``RoundRobinPolicy``.
    """

    selector: Selector
    name: str = "selector"
    max_turns: int = 50
    note_parser: Callable[[str], dict[str, Any]] | None = None
    compactor: Any = None
    compact_every: int = 0
    _selector_arity: int = field(default=2, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self._selector_arity = len(inspect.signature(self.selector).parameters)
        except (ValueError, TypeError):
            self._selector_arity = 2

    async def _select(
        self, turn: int, children: dict[str, Any], transcript: list[Message], ctx: Ctx
    ) -> tuple[Any, str]:
        """Return the next child + a human-readable dispatch reason.

        The selector may return an unknown name (or ``None``) — we fall back to
        round-robin so a flaky selector never stalls. The returned reason makes
        the policy.dispatch span event self-describing: operators see whether
        the selector chose explicitly OR the policy fell back."""
        agents = list(children.values())
        sel: Any = self.selector
        choice = (
            sel(transcript, agents, ctx) if self._selector_arity >= 3 else sel(transcript, agents)
        )
        if inspect.isawaitable(choice):
            choice = await choice
        if choice in children:
            return children[choice], f"selector returned {choice!r}"
        # Fallback: never stall.
        keys = list(children.keys())
        idx = turn % len(keys)
        return children[
            keys[idx]
        ], f"selector returned {choice!r}; fallback round-robin index {idx}"

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

        termination: TerminationCondition = _run_local_termination(
            cognition, MaxTurns(self.max_turns)
        )
        run_id = ctx.correlation_id
        cpt = _resolve_checkpointer(coordinator, ctx)
        bb = context if context is not None and context.shared else context

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
            agent, reason = await self._select(turn, children, transcript, ctx)
            aname = _name_of(agent, children)
            _selector_dispatch(ctx, policy=type(self).__name__, child=aname, reason=reason)

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


def handoff_selector(default: str, *, marker: str = "HANDOFF:") -> Selector:
    """Build a swarm-style handoff selector. The next speaker is named on the **last
    message** as ``HANDOFF:<name>``; absent a marker, ``default`` speaks.

    LEGACY. It does not check the named target against the roster, so an
    invented name reaches ``SelectorPolicy``, which discards it and falls back
    to round-robin by turn index — the pinned ``default`` never gets the turn.
    :func:`~agentkit.agents.control.handoff.route_by_handoff` validates, and is
    what new code should use. Kept as-is because callers depend on this exact
    (unvalidated) behaviour.
    """

    def _select(transcript: Sequence[Message], agents: Sequence[Any]) -> str | None:
        content = transcript[-1].content if transcript else ""
        idx = (content or "").rfind(marker)
        if idx != -1:
            rest = content[idx + len(marker) :].split()
            if rest:
                return rest[0]
        return default

    return _select


def _resolve_roster_name(reply: str, names: Sequence[str]) -> str | None:
    """Which roster name did this model reply name, if any?

    Three rules, in order, each fixing a way the naive
    ``next(n for n in names if n in out)`` got it wrong:

    1. **An exact reply wins.** The instruction asks for ONLY a name, so a
       stripped reply that equals a roster name is unambiguous — including when
       one name is a prefix of another. Substring scanning answered ``"bob"``
       for a reply of ``"bobby"`` off a ``["bob", "bobby"]`` roster.
    2. **Otherwise the LAST whole-word mention wins.** Scanning names in ROSTER
       order answered ``"alice"`` for ``"Not alice — bob should go next"``: the
       first roster entry that appeared anywhere in the reply, regardless of
       where or why. Reading the last mention follows the precedent
       ``parse_handoff`` already sets with its ``rfind`` — a model that reasons
       aloud and then commits does so at the end. Whole-word matching (rather
       than ``in``) is what stops ``"planner"`` from being read as ``"plan"``.
    3. **Ties at the same offset go to the longest name**, so ``"bobby"`` beats
       the ``"bob"`` inside it.

    Returns ``None`` when no roster name is mentioned, which the policy treats
    as "fall back to round-robin".
    """
    stripped = reply.strip()
    for n in names:
        if stripped == n:
            return n
    best: tuple[int, int, str] | None = None
    for n in names:
        if not n:
            continue
        last = None
        for m in re.finditer(rf"(?<!\w){re.escape(n)}(?!\w)", reply):
            last = m.start()
        if last is None:
            continue
        cand = (last, len(n), n)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best[2] if best is not None else None


def llm_selector(
    chooser: Any,
    *,
    roster: str = "",
    instruction: str = "Reply with ONLY the name of the agent who should speak next.",
) -> Callable[[Sequence[Message], list[Any], Ctx], Any]:
    """Build an async, ctx-aware selector that asks ``chooser`` (an ``Agent``-like with
    ``async run(prompt, ctx) -> result.output``) which agent goes next. Returns the
    first roster name that appears in the model's reply, or ``None`` (→ the policy
    falls back to round-robin)."""

    async def _select(transcript: Sequence[Message], agents: list[Any], ctx: Ctx) -> str | None:
        names = [getattr(a, "name", str(i)) for i, a in enumerate(agents)]
        prompt = (
            f"Agents: {', '.join(names)}\n"
            f"{roster}\n\n"
            f"Conversation so far:\n{render_transcript(list(transcript))}\n\n{instruction}"
        )
        out = (await chooser.run(prompt, ctx.child())).output or ""
        return _resolve_roster_name(out, names)

    return _select


__all__ = ["SelectorPolicy", "handoff_selector", "llm_selector"]
