"""SingleCallCognition — one chat call + optional parse-and-repair.

The default cognition for ``Agent`` when neither ``tools`` nor
``children`` is set. Builds the request via the agent's
``RequestBuilder``, makes one streaming chat call, and either returns
the assistant's reply verbatim (when no ``parse`` is wired) or runs
``agent.parse`` and retries up to ``agent.max_repairs`` times by
reflecting the parser's exception back to the model.

State lives on the agent (``agent.parse``, ``agent.max_repairs``,
``agent.model``, ``agent.response_format``, …). The cognition is
pure behavior.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentkit.agents._agent_helpers import (
    _exhausted_ceiling,
    _final_events,
    _last_assistant,
)
from agentkit.agents.result import AgentResult
from agentkit.capabilities.output_schema import OutputCoercionError
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import (
    ChatRequest,
    Message,
    StreamEvent,
    Usage,
    assemble_deltas,
)

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.context import WorkingContext


@dataclass(slots=True)
class SingleCallCognition:
    """One streaming chat call + parse-and-repair retries.

    Reads from the agent: ``model``, ``prompt`` / ``request_builder``,
    ``parse``, ``max_repairs``, ``response_format``, ``temperature``,
    ``max_tokens``, ``_output_adapter``.

    Emits: ``message_delta`` for token streaming, then exactly one
    terminal ``final`` event. Never emits ``tool_call`` / ``interrupt``
    — that's ``ReActCognition``'s job.
    """

    name: str = "single_call"

    async def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]:
        request_builder = agent._resolve_request_builder()
        # The adapter travels on ``Call.meta`` via the explicit ``meta=``
        # param on ``invoker.stream``. The cognition also runs
        # ``agent.parse`` directly below — both paths stay consistent
        # when an adapter is wired.

        built = await request_builder.build(
            task, context, ctx, output_adapter=agent._output_adapter
        )
        prompt_version = built.prompt_version

        usage = Usage()
        max_iter = agent.max_repairs + 1
        single_call_attempt = 0


        for _i in range(max_iter):
            ctx.check_cancelled()

            # Pre-flight. Catches a ceiling ALREADY crossed on entry (a shared
            # Budget that a sibling agent exhausted, or a retry of a run
            # nobody raised the ceiling for) so we don't spend one more call
            # just to discover it. The post-call check below catches a ceiling
            # crossed BY this call.
            ceiling = _exhausted_ceiling(ctx)
            if ceiling is not None:
                for ev in _final_events(
                    _last_assistant(context),
                    usage,
                    partial=True,
                    reason="budget_exhausted",
                    prompt_version=prompt_version,
                    error=ceiling,
                ):
                    yield ev
                return

            req = ChatRequest(
                messages=context.assembled(),
                model=agent.model or "",
                tools=None,
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
                        # ``d.partial`` is the in-progress typed object the
                        # ``output_coerce`` middleware lifted onto this delta
                        # (None whenever no output schema is wired, or the
                        # middleware isn't in the chain). Forward it verbatim —
                        # the cognition neither parses nor interprets it, so a
                        # consumer reading only ``text`` is unaffected.
                        yield StreamEvent("message_delta", text=d.text, partial_output=d.partial)
                    deltas.append(d)
            except OutputCoercionError:
                # ``output_coerce()`` strict-parses at end-of-stream and
                # re-raises on failure. Left uncaught that would escape
                # PAST the reflect-and-retry loop below and abort the run
                # on the first malformed response — the exact case an
                # output schema exists to survive. Every delta was already
                # yielded before the middleware raised, so ``deltas`` is
                # complete: fall through and let ``agent.parse`` below
                # re-raise the same failure INSIDE the repair loop, where
                # it gets reflected back to the model.
                #
                # With no ``agent.parse`` there is no repair loop to fall
                # into and swallowing would silently drop the error, so
                # re-raise. (Unreachable via ``Agent``, which derives
                # ``parse`` from ``output=``; reachable when a caller wires
                # an adapter straight onto ``ctx._output_adapter``.)
                if agent.parse is None:
                    raise
            res = assemble_deltas(deltas)
            usage = usage + res.usage

            if agent.parse is None:
                context.append(Message("assistant", res.content))
                yield StreamEvent(
                    "final",
                    result=AgentResult(
                        output=res.content, usage=usage, prompt_version=prompt_version
                    ),
                    usage=usage,
                )
                return

            try:
                parsed = agent.parse(res.content)
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
            except Exception as exc:  # noqa: BLE001 — reflect invalid output back to the model
                if single_call_attempt >= agent.max_repairs:
                    context.append(Message("assistant", res.content))
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

                # A repair costs another call, so this is the one place a
                # post-call ceiling matters on this cognition. Checked BEFORE
                # re-prompting and never before returning an answer: a closing
                # call that lands exactly on the cap must still yield its
                # result, not a `budget_exhausted` for work already paid for.
                ceiling = _exhausted_ceiling(ctx)
                if ceiling is not None:
                    context.append(Message("assistant", res.content))
                    for ev in _final_events(
                        res.content,
                        usage,
                        partial=True,
                        reason="budget_exhausted",
                        prompt_version=prompt_version,
                        error=ceiling,
                    ):
                        yield ev
                    return

                context.append(Message("assistant", res.content))
                context.append(
                    Message(
                        "user",
                        f"Your previous response was invalid ({exc}). "
                        f"Return ONLY output matching the required format.",
                    )
                )
                single_call_attempt += 1


__all__ = ["SingleCallCognition"]
