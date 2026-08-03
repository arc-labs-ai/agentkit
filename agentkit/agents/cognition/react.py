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

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.agents._agent_helpers import _assistant, _final_events, _to_text
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
        decisions: dict[str, str],
        ctx: Ctx,
    ) -> AgentResult:
        """Resume a suspended tool-loop with per-call human decisions.

        For each pending ``ToolCall``, the matching entry in
        ``decisions`` controls dispatch: ``"approve"`` invokes the
        tool with the model's args verbatim; ``"reject"`` / ``"deny"``
        injects a DENIED tool message; any other string is parsed as
        a JSON ``arguments`` override and invoked with the modified
        args. The loop then continues from the post-tool position.
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

        from agentkit.agents._agent_helpers import _parse_args

        # Drive-local clone (mirrors ``drive``): resume also mutates
        # termination state via ``_iterate`` and must not leak counters
        # onto ``self.termination`` when the cognition is shared.
        termination = _copy.deepcopy(self.termination) if self.termination is not None else None

        for tc in pending:
            ctx.check_cancelled()
            decision = decisions.get(tc.id, "reject")
            if decision in ("reject", "deny"):
                context.append(
                    self._tool_message(tc, "DENIED: tool call rejected by human approval")
                )
                continue
            call = (
                tc
                if decision == "approve"
                else ToolCall(tc.id, tc.name, _parse_args(decision, tc.arguments))
            )
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
            req = ChatRequest(
                messages=context.assembled(),
                model=agent.model or "",
                tools=tool_schemas or None,
                response_format=agent.response_format,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            )
            deltas = []
            async for d in ctx.invoker.stream(
                req, ctx, meta={"output_adapter": agent._output_adapter}
            ):
                if d.text:
                    yield StreamEvent("message_delta", text=d.text)
                deltas.append(d)
            res = assemble_deltas(deltas)
            usage = usage + res.usage

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
                if gated:
                    await self._save(
                        ctx,
                        run_id,
                        context,
                        usage,
                        i,
                        repaired,
                        status="suspended",
                        pending=res.tool_calls,
                    )
                    for tc in res.tool_calls:
                        yield StreamEvent("interrupt", tool_call=tc)
                    susp = Suspended(run_id=run_id, pending=tuple(res.tool_calls))
                    yield StreamEvent(
                        "final",
                        usage=usage,
                        result=AgentResult(
                            output="",
                            usage=usage,
                            partial=True,
                            evals={"stop_reason": "awaiting_approval", "suspended": susp},
                            prompt_version=prompt_version,
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
        last = next((m.content for m in reversed(context.messages) if m.role == "assistant"), "")
        for ev in _final_events(
            last,
            usage,
            partial=True,
            reason="max_iterations",
            prompt_version=prompt_version,
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
    ) -> None:
        """Snapshot the loop state. ``status`` is the suspended/running flag the
        resume gate reads; ``pending`` is the suspended-on-approval tool-call list
        ``resume()`` decisions are applied to."""
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
            },
            status=CheckpointStatus(status),
            ctx=ctx,
        )


__all__ = ["ReActCognition"]
