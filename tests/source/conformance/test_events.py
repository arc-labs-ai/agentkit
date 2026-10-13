import asyncio
from dataclasses import replace

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.core.events import EventDeliveryError, EventHub, MaskingSink
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
async def test_isolated_event_subscriber_does_not_fail_publication_or_skip_others(caplog):
    hub = EventHub()
    seen = []

    async def broken(event):
        raise ValueError("handler failure")

    async def other(event):
        seen.append(event.kind)

    hub.subscribe("isolated", broken, isolate=True)
    hub.subscribe("other", other)
    await hub.publish(CallEvent(EventKind.USAGE, "isolated-run"))
    assert seen == [EventKind.USAGE]
    assert "isolated event subscriber failed" in caplog.text
    await hub.drain()
    await hub.aclose()


@pytest.mark.asyncio
async def test_runtime_events_are_published_through_event_hub():
    hub = EventHub()
    replay = []
    hub.subscribe("test", lambda event: _record(replay, event), call_id="run")
    runtime = CallRuntime(adapters={Mode.OPENAI_API: ScriptedAdapter(Turn(("42",)))}, pricing=PRICES)
    answer = await runtime.ask(Binding(Mode.OPENAI_API, "test"), "6 times 7", events=hub, call_id="run")
    assert answer.outcome is Outcome.ANSWERED
    assert replay[0].kind == EventKind.CALL_STARTED and replay[-1].kind == EventKind.CALL_FINISHED
    assert [e.sequence for e in replay] == list(range(1, len(replay) + 1))
    assert len({e.event_id for e in replay}) == len(replay)
    assert all(e.timestamp and e.schema_version == "1" for e in replay)
    await hub.aclose()


@pytest.mark.asyncio
async def test_hub_accepts_agent_event_prefix_before_runtime_source_sequence():
    hub = EventHub()
    seen = []
    hub.subscribe("collect", lambda event: _record(seen, event), call_id="prefixed")
    await hub.publish(CallEvent(EventKind.SKILL_LOADED, "prefixed", {"skill": "review"}))
    runtime = CallRuntime(adapters={Mode.OPENAI_API: ScriptedAdapter(Turn(("done",)))}, pricing=PRICES)
    answer = await runtime.ask(Binding(Mode.OPENAI_API, "test"), "continue", events=hub, call_id="prefixed")
    assert answer.outcome is Outcome.ANSWERED
    assert [event.sequence for event in seen] == list(range(1, len(seen) + 1))
    assert seen[0].kind == EventKind.SKILL_LOADED and seen[1].kind == EventKind.CALL_STARTED
    await hub.aclose()


async def _record(events, event):
    events.append(event)


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
