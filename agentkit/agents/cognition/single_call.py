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

from agentkit.agents._agent_helpers import _final_events
from agentkit.agents.result import AgentResult
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
            req = ChatRequest(
                messages=context.assembled(),
                model=agent.model or "",
                tools=None,
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
