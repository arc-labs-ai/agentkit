"""ReActCognition — chat ↔ tool-call loop with HITL + durable resume.

Holds the tool registry, iteration ceiling, termination condition,
guardrail, and checkpointer. Drives the classic ReAct loop: chat
call yields token deltas and optional tool calls; tool calls are
dispatched (with optional human gate); tool results re-enter the
prompt as ``tool`` messages; the loop continues until termination
fires, the model returns no tool calls, or the iteration ceiling
is hit.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.agents._agent_helpers import (
    _assistant,
    _final_events,
    _last_assistant,
    _parse_args,
    _to_text,
)
from agentkit.agents.control.elicitation import (
    Decision,
    Elicitation,
    coerce_decision,
    elicit,
    is_context_tainted,
    resolve_asker,
)
from agentkit.agents.control.gate import should_gate
from agentkit.agents.result import AgentResult, Suspended
from agentkit.capabilities.checkpointer import (
    Checkpointer,
    StoreBackedCheckpointStore,
    dict_to_tc,
    msg_to_dict,
    prefix_to_dict,
    rehydrate,
    tc_to_dict,
    usage_to_dict,
)
from agentkit.capabilities.output_schema import OutputCoercionError
from agentkit.kernel.concurrency import gather_bounded
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import (
    ChatRequest,
    Message,
    StreamEvent,
    ToolCall,
    ToolRequest,
    Usage,
    assemble_deltas,
)
from agentkit.tools import ToolShapeError

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.agents.control.termination import TerminationCondition
    from agentkit.context import WorkingContext


@dataclass(slots=True)
class ReActCognition:
    """Tool-loop cognition.

    Holds:
        tools: ``ToolRegistry`` (a plain list is wrapped automatically).
        max_iterations: hard ceiling on tool-loop turns.
        termination: smart-stop predicate evaluated on each assistant turn.
        guardrail: frames tool output as UNTRUSTED before it re-enters the prompt.
        checkpointer: per-cognition durable checkpoint store; ``None`` falls back
            to ``ctx.checkpointer`` then a legacy bridge over ``ctx.store``.

    Reads from the agent (chat-call shape): ``model``, ``prompt`` /
    ``request_builder``, ``response_format``, ``temperature``,
    ``max_tokens``, ``parse``, ``_output_adapter``.

    Emits: ``message_delta`` per chat token, ``tool_call`` /
    ``tool_result`` per tool invocation, ``interrupt`` when human
    approval is required, ``step`` at every iteration boundary, and
    exactly one terminal ``final`` event.
    """

    tools: Any  # ToolRegistry | list[FunctionTool|callable]
    max_iterations: int = 8
    termination: TerminationCondition | None = None
    guardrail: Any = None
    checkpointer: Checkpointer | None = None
    # Deadline, in seconds, on waiting for a human approval. Applies to the
    # PARK path (an ``Asker`` wired on the ctx) — the wait is bounded and an
    # expiry degrades the run instead of hanging it. Also stamped onto
    # ``Suspended.deadline_at`` on the return-and-resume path so an operator UI
    # can render a countdown and a late decision can be refused.
    # ``None`` (default) = wait indefinitely, exactly today's behaviour.
    approval_deadline_s: float | None = None
    name: str = field(default="react")

    def __post_init__(self) -> None:
        # Ergonomic: ReActCognition(tools=[get_weather, ...]) — wrap a plain list
        # into a ToolRegistry. ToolRegistry.from_tools auto-converts each plain
        # function into a FunctionTool.
        if isinstance(self.tools, (list, tuple)):
            from agentkit.tools import ToolRegistry

            self.tools = ToolRegistry.from_tools(self.tools)

    async def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]:
        request_builder = agent._resolve_request_builder()
        # The adapter travels on ``Call.meta`` via the explicit ``meta=``
        # param on ``invoker.stream`` — never on a private attribute of
        # the shared ``RunContext``.
        #
        # Clone the termination condition into a drive-local variable
        # so two concurrent drives sharing the same ReActCognition
        # instance (e.g. Coordinator dispatching two children with the
        # same child spec) don't race on ``MaxTurns.turn`` /
        # ``Timeout._start`` / ``ExternalTermination._flag``. Storing
        # the clone on ``self`` would still serialize the two drives
        # onto the same object under interleaved scheduling; a local
        # scoped to the drive frame is what actually delivers the
        # per-run isolation. ``Skill.as_agent`` deep-copies cognition;
        # this covers the parallel-dispatch path where cognitions are
        # shared without a Skill facade in between.
        import copy as _copy

        termination = _copy.deepcopy(self.termination) if self.termination is not None else None
        run_id = ctx.correlation_id

        saved = await self._load(ctx, run_id)
        if saved is not None and saved.status != "suspended":
            context, usage, start_i, repaired = rehydrate(saved.state)  # durable resume
            prompt_version = request_builder.prompt.version
        else:
            built = await request_builder.build(
                task, context, ctx, output_adapter=agent._output_adapter
            )
            prompt_version = built.prompt_version
            usage, start_i, repaired = Usage(), 0, False

        async for ev in self._iterate(
            agent, context, usage, start_i, repaired, ctx, run_id, prompt_version, termination
        ):
            yield ev

    async def resume(
        self,
        agent: Agent,
        run_id: str,
        decisions: Mapping[str, str | Decision],
        ctx: Ctx,
    ) -> AgentResult:
        """Resume a suspended tool-loop with per-call human decisions.

        Accepts BOTH shapes. A typed :class:`~agentkit.agents.control.elicitation.Decision`
        carries the actor and timestamp an audit trail needs; the legacy
        ``str`` form is coerced through ``coerce_decision`` so every existing
        caller keeps working from the same call site:

            ``"approve"``         → invoke with the model's args verbatim
            ``"reject"``/``"deny"`` → inject a DENIED tool message
            anything else         → parsed as a JSON ``arguments`` override

        The loop then continues from the post-tool position.

        A missing entry defaults to a denial, not an approval — an operator
        who answered three of four gates has not implicitly approved the
        fourth.
        """
        request_builder = agent._resolve_request_builder()
        saved = await self._load(ctx, run_id)
        if not saved or saved.status != "suspended":
            raise ValueError(f"no suspended run {run_id!r} to resume")
        context, usage, i, repaired = rehydrate(saved.state)
        pending = [dict_to_tc(d) for d in saved.state.get("pending", [])]
        prompt_version = request_builder.prompt.version

        # Cooperative cancellation before the dispatch loop — a token
        # tripped during a multi-approve resume (operator hit cancel
        # while decisions were in flight) must abort BEFORE any pending
        # tool fires; the per-iteration check below prevents dispatch
        # of any subsequent pending call as well.
        ctx.check_cancelled()

        import copy as _copy

        # Drive-local clone (mirrors ``drive``): resume also mutates
        # termination state via ``_iterate`` and must not leak counters
        # onto ``self.termination`` when the cognition is shared.
        termination = _copy.deepcopy(self.termination) if self.termination is not None else None

        # A deadline that passed while the run sat suspended. DEGRADE, don't
        # die: every pending call becomes ``expired`` and the loop continues,
        # so an operator who answers an hour late gets a run that completed
        # without their tool call rather than a silent success that acted on a
        # decision nobody was still entitled to make. Distinct wording from
        # DENIED so the transcript says which of the two happened.
        deadline_at = saved.state.get("deadline_at")
        expired = deadline_at is not None and time.time() > deadline_at
        if expired:
            await ctx.emit(
                "gate.check",
                render="resume arrived after the approval deadline; treating as expired",
                payload={"run_id": run_id, "deadline_at": deadline_at},
            )

        for tc in pending:
            ctx.check_cancelled()
            decision = (
                Decision(kind="expired", note="approval deadline passed while suspended")
                if expired
                else coerce_decision(decisions.get(tc.id, "reject"))
            )
            if decision.kind == "expired":
                context.append(
                    self._tool_message(
                        tc, "EXPIRED: the approval deadline passed; treated as not approved"
                    )
                )
                continue
            if not decision.approved:
                # The actor is named in the transcript the model sees, so a
                # later turn can reason about WHO refused — impossible when the
                # decision was a bare string.
                who = decision.actor or "human approval"
                context.append(self._tool_message(tc, f"DENIED: tool call rejected by {who}"))
                continue
            call = tc
            if decision.kind == "modify" and decision.value is not None:
                call = ToolCall(tc.id, tc.name, _parse_args(str(decision.value), tc.arguments))
            result = await self._invoke_tool_safe(ctx, call)
            context.append(self._tool_message(call, result))

        final: AgentResult | None = None
        async for ev in self._iterate(
            agent, context, usage, i + 1, repaired, ctx, run_id, prompt_version, termination
        ):
            if ev.type == "final":
                final = ev.result
        if final is None:  # pragma: no cover
            raise RuntimeError(f"resumed loop {run_id!r} produced no final result")
        return final

    async def _iterate(
        self,
        agent: Agent,
        context: WorkingContext,
        usage: Usage,
        start_i: int,
        repaired: bool,
        ctx: Ctx,
        run_id: str,
        prompt_version: str,
        termination: TerminationCondition | None,
    ) -> AsyncIterator[StreamEvent]:
        # ``termination`` is required — the caller (``drive`` / ``resume``)
        # owns cloning ``self.termination`` into a drive-local variable so
        # two concurrent runs sharing the same ReActCognition don't race
        # on ``MaxTurns.turn`` / ``Timeout._start`` /
        # ``ExternalTermination._flag``. Passing ``None`` explicitly
        # disables smart-stop for this run; reading ``self.termination``
        # here would silently re-share state and defeat the isolation.
        tool_schemas = self.tools.schemas()

        if termination is not None and start_i == 0:
            await termination.reset()

        for i in range(start_i, self.max_iterations):
            ctx.check_cancelled()

            # Pre-flight budget check. The post-call check further down is
            # what catches a ceiling crossed BY this run; this one catches a
            # ceiling that was ALREADY crossed when the loop started —
            # specifically, a resume against a budget nobody raised.
            #
            # Without it, ``guard()`` returns a not-ok verdict that the meter
            # middleware deliberately ignores (it cannot write a checkpoint,
            # so acting on the verdict is the cognition's job), the chat call
            # goes out anyway, and every retry of an exhausted run burns
            # another full call before noticing. Checked here, a resume
            # against an unraised ceiling costs nothing.
            if self._budget_exhausted(ctx):
                last = _last_assistant(context)
                await self._save(ctx, run_id, context, usage, i, repaired, status="suspended")
                async for ev in self._budget_final(ctx, last, usage, prompt_version):
                    yield ev
                return

            req = ChatRequest(
                messages=context.assembled(),
                model=agent.model or "",
                tools=tool_schemas or None,
                response_format=agent.response_format,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            )
            deltas = []
            try:
                async for d in ctx.invoker.stream(
                    req, ctx, meta={"output_adapter": agent._output_adapter}
                ):
                    if d.text:
                        # Forward the in-progress typed object verbatim. See
                        # ``SingleCallCognition.drive`` for the same three lines
                        # and ``StreamEvent.partial_output`` for the contract.
                        yield StreamEvent("message_delta", text=d.text, partial_output=d.partial)
                    deltas.append(d)
            except OutputCoercionError:
                # Same contract as ``SingleCallCognition.drive`` — see the long
                # comment there. ``output_coerce()``'s strict end-of-stream parse
                # must not escape past this loop's validate-and-repair branch
                # below, or a single malformed response aborts the whole run.
                if agent.parse is None:
                    raise
            res = assemble_deltas(deltas)
            usage = usage + res.usage

            # ── budget exhaustion: checkpoint BEFORE stopping ──────────────
            #
            # This is the whole point of ``Budget.on_exceeded="stop"``. Under
            # the default ``"raise"``, ``MeterExceeded`` comes out of the
            # ``invoker.stream`` call above and unwinds past every ``_save``
            # below — so the run aborts holding a checkpoint from the PREVIOUS
            # iteration, this turn's spend is unrecorded, and a resume
            # re-enters a budget that is still over its ceiling and raises
            # again on the first guard. Everything spent is unrecoverable.
            #
            # With ``"stop"`` the meter records the spend and returns a
            # verdict instead, and control reaches here — where the cognition
            # still holds the live context and can write a ``suspended``
            # checkpoint carrying the CURRENT state. The operator raises the
            # ceiling and resumes; nothing is lost.
            #
            # This is the POST-call check — the meter has just charged, so it
            # catches a ceiling crossed BY this call. The pre-flight at the top
            # of the loop catches one that was already crossed on entry. Both
            # are needed: without this one an overspend goes unnoticed until
            # the next iteration, and without the pre-flight every retry of an
            # already-stopped run buys one more call to rediscover it.
            #
            # Stopping here means at most one call's overshoot, which is the
            # same bound the ceiling always had (see ``Budget``'s "known
            # property" note).
            if self._budget_exhausted(ctx):
                context.append(_assistant(res))
                await self._save(ctx, run_id, context, usage, i + 1, repaired, status="suspended")
                async for ev in self._budget_final(ctx, res.content, usage, prompt_version):
                    yield ev
                return

            if termination is not None:  # smart stop on the assistant delta (drive-local clone)
                stop = await termination([_assistant(res)], ctx)
                if stop is not None:
                    context.append(_assistant(res))
                    await self._clear(ctx, run_id)
                    for ev in _final_events(
                        res.content,
                        usage,
                        partial=False,
                        reason=stop.reason,
                        prompt_version=prompt_version,
                    ):
                        yield ev
                    return

            if res.tool_calls:
                context.append(_assistant(res))
                # ``_needs_approval`` is pure — call it once here and
                # stamp a ``gate.check`` observation so the run's audit
                # trail records the HITL decision at every tool-call
                # boundary (not just the ones that actually gated).
                # Emit is defensive in ``RunContext.emit``; a
                # misbehaving observer never breaks the loop.
                gated = self._needs_approval(res.tool_calls, ctx)
                await ctx.emit(
                    "gate.check",
                    payload={"autonomy": str(ctx.autonomy), "gated": gated},
                )
                if gated and resolve_asker(ctx) is not None:
                    # ── PARK IN PLACE ─────────────────────────────────────
                    #
                    # An ``Asker`` is wired, so we await the person from
                    # inside this coroutine. Nothing unwinds: ``context``,
                    # ``usage``, the termination clone, and every local stay
                    # exactly where they are. That is the requirement a
                    # production caller holding live, unserialisable state
                    # could not meet through ``resume()``, which forces the
                    # whole loop to exit and be rebuilt from a snapshot.
                    #
                    # The return-and-resume path below is untouched and still
                    # runs whenever no asker is present, so callers that CAN
                    # serialise lose nothing.
                    for tc in res.tool_calls:
                        yield StreamEvent("interrupt", tool_call=tc)
                    async for ev in self._park(ctx, context, res.tool_calls, agent, run_id):
                        yield ev
                    await self._save(ctx, run_id, context, usage, i + 1, repaired)
                    yield StreamEvent("step", text=f"iteration:{i + 1}")
                    continue

                if gated:
                    deadline_at = (
                        None
                        if self.approval_deadline_s is None
                        else time.time() + self.approval_deadline_s
                    )
                    await self._save(
                        ctx,
                        run_id,
                        context,
                        usage,
                        i,
                        repaired,
                        status="suspended",
                        pending=res.tool_calls,
                        deadline_at=deadline_at,
                    )
                    for tc in res.tool_calls:
                        yield StreamEvent("interrupt", tool_call=tc)
                    susp = Suspended(
                        run_id=run_id,
                        pending=tuple(res.tool_calls),
                        # Absolute wall-clock expiry, so an operator UI renders
                        # a countdown and a decision arriving after it can be
                        # refused. ``None`` when no deadline is configured —
                        # today's unbounded behaviour, unchanged.
                        deadline_at=deadline_at,
                    )
                    yield StreamEvent(
                        "final",
                        usage=usage,
                        result=AgentResult(
                            output="",
                            usage=usage,
                            partial=True,
                            evals={"stop_reason": "awaiting_approval", "suspended": susp},
                            prompt_version=prompt_version,
                            # Typed so a reader can branch WITHOUT reaching into
                            # the ``evals`` bag: parked on a human, not broken.
                            stop_reason="suspended",
                        ),
                    )
                    return

                for tc in res.tool_calls:
                    yield StreamEvent("tool_call", tool_call=tc)
                results = await gather_bounded(
                    [self._invoke_tool_safe(ctx, tc) for tc in res.tool_calls],
                    sem=ctx.semaphore(),
                )
                for tc, r in zip(res.tool_calls, results, strict=False):
                    context.append(self._tool_message(tc, r))
                    yield StreamEvent("tool_result", tool_call=tc, tool_result=r)
                await self._save(ctx, run_id, context, usage, i + 1, repaired)
                yield StreamEvent("step", text=f"iteration:{i + 1}")
                continue

            # No tool calls → final answer (with optional one-shot validate-and-repair).
            if agent.parse is not None:
                try:
                    parsed = agent.parse(res.content)
                except Exception as exc:  # noqa: BLE001 — reflect invalid output back to the model
                    if not repaired:
                        repaired = True
                        context.append(_assistant(res))
                        context.append(
                            Message(
                                "user",
                                f"Your previous response was invalid ({exc}). "
                                f"Return only output matching the required format.",
                            )
                        )
                        await self._save(ctx, run_id, context, usage, i + 1, repaired)
                        yield StreamEvent("step", text="repair")
                        continue
                    await self._clear(ctx, run_id)
                    for ev in _final_events(
                        res.content,
                        usage,
                        partial=True,
                        reason="invalid_output",
                        prompt_version=prompt_version,
                        error=str(exc),
                    ):
                        yield ev
                    return
                await self._clear(ctx, run_id)
                context.append(Message("assistant", res.content))
                yield StreamEvent(
                    "final",
                    result=AgentResult(
                        output=res.content,
                        usage=usage,
                        parsed=parsed,
                        prompt_version=prompt_version,
                    ),
                    usage=usage,
                )
                return

            await self._clear(ctx, run_id)
            context.append(Message("assistant", res.content))
            yield StreamEvent(
                "final",
                result=AgentResult(output=res.content, usage=usage, prompt_version=prompt_version),
                usage=usage,
            )
            return

        # Tool-loop ceiling reached → partial.
        await self._clear(ctx, run_id)
        last = _last_assistant(context)
        for ev in _final_events(
            last,
            usage,
            partial=True,
            reason="max_iterations",
            prompt_version=prompt_version,
        ):
            yield ev

    # ---- human-in-the-loop: park in place ---------------------------------------------------

    async def _park(
        self,
        ctx: Any,
        context: Any,
        tool_calls: Sequence[ToolCall],
        agent: Agent,
        run_id: str,
    ) -> AsyncIterator[StreamEvent]:
        """Await a person for each gated tool call, without unwinding the loop.

        One :class:`Elicitation` per pending call, each carrying
        ``self.approval_deadline_s``. The four outcomes:

        * ``approve``  — run the tool with the model's arguments verbatim.
        * ``modify``   — run it with the person's replacement arguments.
        * ``deny``     — inject a DENIED tool message; the model sees the
                         refusal and reacts on its next turn.
        * ``expired``  — the deadline passed. DEGRADE, don't die: treated as a
                         denial so the loop continues, but the message says
                         "expired" so an operator can tell "someone said no"
                         from "nobody was there". That distinction is the
                         abandoned-tab case, and it is why expiry is an
                         ordinary recorded outcome rather than a hang.

        Emits ``tool_result`` per call, exactly like the ungated path, so a
        consumer's event handling is identical whether or not a human was
        involved.
        """
        for tc in tool_calls:
            request = Elicitation(
                id=tc.id,
                prompt=f"Approve tool {tc.name!r}?",
                kind="approval",
                tool_call=tc,
                deadline_s=self.approval_deadline_s,
                run_id=run_id,
                agent=agent.name,
            )
            decision = await elicit(ctx, request, context=context)
            if decision.approved:
                # Emitted only once the human said yes, but emitted — a
                # consumer that counts ``tool_call`` events to render "running
                # X…" must see the same sequence here as on the ungated path.
                yield StreamEvent("tool_call", tool_call=tc)
            if decision.kind == "expired":
                result: Any = (
                    f"EXPIRED: no human answered within {self.approval_deadline_s}s; "
                    "treated as not approved"
                )
                context.append(self._tool_message(tc, result))
            elif not decision.approved:
                note = f" ({decision.note})" if decision.note else ""
                result = f"DENIED: tool call rejected by {decision.actor or 'human'}{note}"
                context.append(self._tool_message(tc, result))
            else:
                call = tc
                if decision.kind == "modify" and decision.value is not None:
                    call = ToolCall(tc.id, tc.name, _parse_args(str(decision.value), tc.arguments))
                result = await self._invoke_tool_safe(ctx, call)
                context.append(self._tool_message(call, result))
            yield StreamEvent("tool_result", tool_call=tc, tool_result=result)

    # ---- budget exhaustion ------------------------------------------------------------------

    @staticmethod
    def _budget_exhausted(ctx: Any) -> bool:
        """Has a ceiling been crossed on a budget configured to stop rather than raise?

        Reads through ``getattr`` so a ``NullCtx`` or a structural test stub
        without a real ``Budget`` is simply "not exhausted" rather than an
        ``AttributeError`` inside the loop. A budget left on the default
        ``on_exceeded="raise"`` never reaches this check — it has already
        raised — so gating on the setting keeps the two modes from
        double-handling the same event.
        """
        budget = getattr(ctx, "budget", None)
        if budget is None or getattr(budget, "on_exceeded", "raise") != "stop":
            return False
        check = getattr(budget, "exhausted", None)
        return bool(check()) if callable(check) else False

    async def _budget_final(
        self, ctx: Any, content: str, usage: Usage, prompt_version: str
    ) -> AsyncIterator[StreamEvent]:
        """Terminal event for a budget-exhausted run.

        ``partial=True`` because the answer is unfinished, and
        ``stop_reason="budget_exhausted"`` — which
        ``AgentResult.is_resumable`` reports as recoverable, distinguishing it
        from both a completed run and a crash. The verdict's reason string
        lands in ``evals["error"]`` so an operator sees the actual numbers
        without re-deriving them.
        """
        budget = getattr(ctx, "budget", None)
        verdict = budget.verdict() if budget is not None else None
        await ctx.emit(
            "budget.exhausted",
            render="run stopped on a budget ceiling; checkpoint written",
            payload={"reason": verdict.reason if verdict else "", "calls": usage.total_tokens},
        )
        for ev in _final_events(
            content,
            usage,
            partial=True,
            reason="budget_exhausted",
            prompt_version=prompt_version,
            error=verdict.reason if verdict else "budget exhausted",
        ):
            yield ev

    # ---- tool dispatch helpers --------------------------------------------------------------

    def _needs_approval(self, calls: Sequence[ToolCall], ctx: Any) -> bool:
        """Hallucinated tool names must not crash the gate. A KeyError
        from ``self.tools.get`` would abort the loop before
        ``_invoke_tool_safe`` ever ran; treat a missing tool as "no
        gating needed" so the error surfaces in ``_invoke_tool_safe``
        where it can be reflected back to the model."""
        for c in calls:
            try:
                tool = self.tools.get(c.name)
            except KeyError:
                continue  # unknown tool — handled in _invoke_tool_safe
            if should_gate(
                ctx.autonomy,
                requires_approval=self.tools.requires_approval(c.name),
                key_step=getattr(tool, "side_effecting", False),
            ):
                return True
        return False

    def _tool_request(self, ctx: Any, tc: ToolCall) -> ToolRequest:
        tool = self.tools.get(tc.name)
        return ToolRequest(
            name=tc.name,
            arguments=tc.arguments,
            tool=tool,
            side_effecting=getattr(tool, "side_effecting", False),
            url_arg=getattr(tool, "url_arg", None),
        )

    async def _invoke_tool_safe(self, ctx: Any, tc: ToolCall) -> Any:
        """Invoke a tool and translate ANY failure into a structured error
        string the model can see in its next turn — instead of letting the
        whole loop crash on a single bad tool call.

        Catches the full failure surface — a hallucinated tool name
        (``KeyError`` from ``self.tools.get``), a timeout, a network
        blip, or a validator error would otherwise propagate and abort
        the agent mid-loop with no ``final`` event. The retry middleware
        (chat-chain) doesn't sit on the tool path, so the cognition owns
        reflect-and-recover for the whole error surface:

        - :class:`KeyError` from ``self.tools.get`` → "tool not found in
          registry" + the advertised tool list so the model can correct.
        - :class:`ToolShapeError` → quote the validator's structured
          ``errors`` so the model can retry-with-different-args.
        - Any other :class:`Exception` (timeout, network, internal tool
          bug) → "tool failed: ``{kind}: {msg}``" — the model sees
          enough to decide whether to retry the same call, pivot, or
          give up.

        :class:`BaseException` subclasses (CancelledError, KeyboardInterrupt,
        SystemExit) are NOT caught — those must continue to propagate so
        cancellation / shutdown still work."""
        try:
            request = self._tool_request(ctx, tc)
        except KeyError:
            # Hallucinated tool name — the model invented a tool that
            # doesn't exist. Tell it which tools ARE available so it
            # can recover.
            available = ", ".join(self.tools.names()) or "(none)"
            return f"ERROR: no tool named {tc.name!r}. Available tools: {available}."
        try:
            return await ctx.invoker.invoke_tool(request, ctx)
        except ToolShapeError as exc:
            details = "; ".join(exc.errors) if exc.errors else str(exc)
            return (
                "ERROR: tool returned a value that does not match its declared "
                f"output schema {exc.expected!r}. Details: {details}"
            )
        except Exception as exc:  # noqa: BLE001 — reflect ALL tool failures to the model
            return f"ERROR: tool {tc.name!r} failed: {type(exc).__name__}: {exc}"

    def _tool_message(self, tc: ToolCall, result: Any) -> Message:
        text = _to_text(result)
        if self.guardrail is not None and hasattr(self.guardrail, "wrap_tool_output"):
            text = self.guardrail.wrap_tool_output(text, source=tc.name)
        return Message("tool", content=text, tool_call_id=tc.id, name=tc.name)

    # ---- checkpointer wiring ----------------------------------------------------------------

    def _resolve_checkpointer(self, ctx: Any) -> Checkpointer | None:
        """Return the Checkpointer that should back this run.

        Resolution order: explicit ``self.checkpointer`` > ``ctx.checkpointer`` >
        a legacy bridge synthesized over ``ctx.store`` (so existing wirings that
        only inject a ``StorePort`` keep working). When all three are absent,
        durable resume is disabled and the helpers below become no-ops."""
        if self.checkpointer is not None:
            return self.checkpointer
        cp: Checkpointer | None = getattr(ctx, "checkpointer", None)
        if cp is not None:
            return cp
        store = getattr(ctx, "store", None)
        if store is not None:
            return Checkpointer(port=StoreBackedCheckpointStore(store))
        return None

    async def _load(self, ctx: Any, run_id: str) -> Any:
        cp = self._resolve_checkpointer(ctx)
        if cp is None:
            return None
        return await cp.resume(run_id)

    async def _clear(self, ctx: Any, run_id: str) -> None:
        cp = self._resolve_checkpointer(ctx)
        if cp is None:
            return
        await cp.delete(run_id)

    async def _save(
        self,
        ctx: Any,
        run_id: str,
        context: Any,
        usage: Any,
        next_i: int,
        repaired: bool,
        *,
        status: str = "running",
        pending: tuple[Any, ...] = (),
        deadline_at: float | None = None,
    ) -> None:
        """Snapshot the loop state. ``status`` is the suspended/running flag the
        resume gate reads; ``pending`` is the suspended-on-approval tool-call list
        ``resume()`` decisions are applied to.

        A context TAINTED by a secret is never snapshotted. Once a one-time
        code has entered ``context.messages`` — as an elicited value or a tool
        result derived from one — persisting the state would write the
        credential into Postgres, where it outlives by weeks the ten minutes it
        was valid for. Losing durability for the rest of that run is a real
        cost and the correct trade: an un-resumable run can be re-run, a leaked
        credential cannot be un-leaked.

        The refusal itself is enforced one layer down, in
        ``Checkpointer.snapshot`` — the taint marker rides along inside
        ``context.scratchpad``, which every producer serialises into its state
        blob, so the rule holds for the coordinator policies too rather than
        only for this loop. We emit the observation here because this is where
        the run context is in hand.
        """
        if is_context_tainted(context):
            await ctx.emit(
                "gate.check",
                render="checkpoint skipped: working context holds an elicited secret",
                payload={"run_id": run_id, "skipped": True},
            )
        cp = self._resolve_checkpointer(ctx)
        if cp is None:
            return
        await cp.snapshot(
            run_id,
            {
                "prefix": prefix_to_dict(context.prefix),
                "messages": [msg_to_dict(m) for m in context.messages],
                "scratchpad": context.scratchpad,
                "limit": context.limit,
                "shared": context.shared,
                "usage": usage_to_dict(usage),
                "iteration": next_i,
                "repaired": repaired,
                "pending": [tc_to_dict(t) for t in pending],
                # Absolute wall-clock expiry for a suspended run. Persisted
                # (not just returned on ``Suspended``) because the process that
                # resumes is usually not the process that suspended — an
                # in-memory deadline would be gone by then, which is exactly
                # the abandoned-tab case the deadline exists for.
                "deadline_at": deadline_at,
            },
            status=CheckpointStatus(status),
            ctx=ctx,
        )


__all__ = ["ReActCognition"]
