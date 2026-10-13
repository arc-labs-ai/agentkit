"""Bridge standardized call events into the existing EventBus and trace port."""

from __future__ import annotations

from typing import Any

from agentkit.core.events.bus import EventBus
from agentkit.models.contracts.constants import EventKind
from agentkit.models.contracts.types import CallEvent


class BusEventSink:
    """Use the existing replayable EventBus as a CallRuntime EventSink.

    EventBus is a bounded live fan-out with ring replay. Durable, lossless
    delivery belongs to an EventJournal or a host EventPublisher.
    """

    def __init__(self, bus: EventBus[CallEvent], *, close_on_finish: bool = False) -> None:
        self.bus, self.close_on_finish = bus, close_on_finish

    async def __call__(self, event: CallEvent) -> None:
        await self.bus.publish(event.call_id, event)
        if self.close_on_finish and event.kind == EventKind.CALL_FINISHED:
            await self.bus.close_stream(event.call_id)


class TraceEventSink:
    """GenAI spans through the existing TracePort, without a required OTel SDK."""

    def __init__(self, trace: Any) -> None:
        self.trace = trace
        self._spans: dict[str, tuple[Any, Any]] = {}

    async def __call__(self, event: CallEvent) -> None:
        fields = event.fields
        if event.kind == EventKind.CALL_STARTED:
            manager = self.trace.span(
                "invoke_agent",
                kind="agent",
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.provider.name": fields.get("mode"),
                    "gen_ai.request.model": fields.get("model"),
                    "agentkit.call.id": event.call_id,
                },
            )
            self._spans[event.call_id] = (manager, manager.__enter__())
        entry = self._spans.get(event.call_id)
        if entry is None:
            return
        manager, span = entry
        if event.kind == EventKind.USAGE:
            usage = fields.get("usage") or {}
            span.set(
                "gen_ai.usage.input_tokens",
                sum(
                    usage.get(k, 0)
                    for k in ("input_uncached", "input_cache_read", "input_cache_write_5m", "input_cache_write_1h")
                ),
            )
            span.set("gen_ai.usage.output_tokens", usage.get("output", 0))
        span.add_event(str(event.kind), **dict(fields))
        if event.kind == EventKind.CALL_FINISHED:
            span.set("agentkit.cost.usd", fields.get("cost_usd"))
            span.set("agentkit.cost.source", fields.get("cost_source"))
            manager.__exit__(None, None, None)
            del self._spans[event.call_id]
