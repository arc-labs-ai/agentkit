import asyncio
from dataclasses import replace

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.core.events import BusEventSink, EventDeliveryError, EventHub, MaskingSink
from agentkit.core.events.bus import EventBus
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.constants import EventKind, Outcome
from agentkit.models.contracts.types import Binding, CallEvent, Turn
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.asyncio
async def test_masking_sink_masks_then_bounds_nested_event_fields():
    received = []

    def mask(event):
        fields = dict(event.fields)
        fields["credential"] = "[redacted]"
        return replace(event, fields=fields)

    sink = MaskingSink(received.append, mask, max_field_bytes=12)
    await sink(CallEvent(EventKind.TOOL_CALL, "run", {"arguments": {"token": "secret", "body": "x" * 30}}))
    fields = received[0].fields
    assert fields["credential"] == "[redacted]"
    assert fields["arguments"]["token"] == "secret"
    assert fields["arguments"]["body"]["truncated"] is True
    assert fields["arguments"]["body"]["length"] == 30


@pytest.mark.asyncio
async def test_hub_persists_before_handlers_and_allows_event_triggered_followups():
    recorded = []
    handled = []

    class Journal:
        async def append(self, event):
            recorded.append(event)

        async def read(self, **kwargs):
            return recorded

    async with EventHub(journal=Journal()) as hub:

        async def followup(event):
            assert event in recorded
            handled.append(event.kind)
            if event.kind == EventKind.TOOL_RESULT:
                await hub.publish(replace(event, kind="harness.evidence_ready", sequence=2))

        unsubscribe = hub.subscribe("reactor", followup, kinds=(EventKind.TOOL_RESULT, "harness.*"))
        await hub.publish(CallEvent(EventKind.TOOL_RESULT, "run", {"is_error": False}, sequence=1))
        await hub.drain()
        assert handled == [EventKind.TOOL_RESULT, "harness.evidence_ready"]
        unsubscribe()
        await hub.publish(CallEvent(EventKind.USAGE, "run", sequence=3))
        assert len(handled) == 2


@pytest.mark.asyncio
async def test_event_subscriber_failure_is_visible_and_does_not_skip_other_subscribers():
    hub = EventHub()
    seen = []

    async def broken(event):
        raise ValueError("handler failure")

    async def other(event):
        seen.append(event)

    hub.subscribe("broken", broken)
    hub.subscribe("other", other)
    with pytest.raises(EventDeliveryError, match="broken"):
        await hub.publish(CallEvent(EventKind.USAGE, "run"))
    assert len(seen) == 1
    with pytest.raises(ExceptionGroup):
        await hub.drain()
    await hub.aclose()
    with pytest.raises(RuntimeError):
        await hub.publish(CallEvent(EventKind.USAGE, "run"))


@pytest.mark.asyncio
async def test_runtime_event_ids_and_existing_bus_replay():
    bus = EventBus()
    sink = BusEventSink(bus)
    runtime = CallRuntime(adapters={Mode.OPENAI_API: ScriptedAdapter(Turn(("42",)))}, pricing=PRICES)
    answer = await runtime.ask(Binding(Mode.OPENAI_API, "test"), "6 times 7", events=sink, call_id="run")
    assert answer.outcome is Outcome.ANSWERED
    subscriber = bus.subscribe("run", from_version=0)
    replay = []
    async for wrapped in subscriber:
        replay.append(wrapped.event)
        if wrapped.event.kind == EventKind.CALL_FINISHED:
            break
    await subscriber.aclose()
    await bus.close_stream("run")
    assert replay[0].kind == EventKind.CALL_STARTED and replay[-1].kind == EventKind.CALL_FINISHED
    assert [e.sequence for e in replay] == list(range(1, len(replay) + 1))
    assert len({e.event_id for e in replay}) == len(replay)
    assert all(e.timestamp and e.schema_version == "1" for e in replay)


@pytest.mark.asyncio
async def test_event_handlers_have_bounded_execution():
    hub = EventHub(handler_timeout_s=0.001)

    async def blocks(event):
        await asyncio.Event().wait()

    hub.subscribe("blocked", blocks)
    with pytest.raises(EventDeliveryError, match="TimeoutError"):
        await hub.publish(CallEvent(EventKind.USAGE, "run"))
    with pytest.raises(ExceptionGroup):
        await hub.aclose()
