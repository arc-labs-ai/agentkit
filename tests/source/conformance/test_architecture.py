"""Harness-facing contracts: verification, authority and bounded event reactions."""

from dataclasses import replace
from decimal import Decimal

import pytest

from agentkit.agents.definitions.agent import AgentDefinition, DelegationPolicy
from agentkit.agents.definitions.instructions import InstructionSet
from agentkit.agents.definitions.store import MemoryAgentStore
from agentkit.agents.execution.runner import AgentRunner
from agentkit.core.events import EventDeliveryError, EventHub, TraceEventSink
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.constants import AuthKind, EventKind, ModelSelection, Outcome
from agentkit.models.contracts.types import Binding, CallEvent, CallLimits, Turn
from agentkit.models.providers.auth import AnthropicCredential
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.asyncio
async def test_verification_continues_with_context_and_aggregates_resources():
    adapter = ScriptedAdapter(
        Turn(("incomplete",), raw_usage={"input_tokens": 10, "output_tokens": 2}),
        Turn(("complete",), raw_usage={"input_tokens": 20, "output_tokens": 3}),
    )
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES))
    definition = AgentDefinition("worker", "1", "worker", Binding("openai_api", "test"))
    events = []
    answer = await runner.run_agent(
        definition,
        InstructionSet("finish", verifier=lambda answer: answer.text == "complete"),
        {},
        call_id="worker",
        events=events.append,
    )
    assert answer.outcome == Outcome.ANSWERED
    assert answer.messages == ("incomplete", "complete") and answer.attempts == 2
    assert answer.usage.input_uncached == 30 and answer.usage.output == 5
    assert adapter.requests[1][2]["session"] is not None
    assert "verifier rejected" in adapter.requests[1][1].user
    child_events = [event for event in events if event.call_id == "worker/verify-1"]
    assert child_events and all(event.parent_call_id == "worker" for event in child_events)


@pytest.mark.asyncio
async def test_verification_stops_repeated_no_progress_and_honors_async_verifier():
    adapter = ScriptedAdapter(Turn(("stuck",)), Turn(("stuck",)))
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES))
    definition = AgentDefinition("worker", "1", "worker", Binding("openai_api", "test"))

    async def incomplete(answer):
        return False

    answer = await runner.run_agent(
        definition,
        InstructionSet("finish", verifier=incomplete, no_progress_slices=1),
        {},
        call_id="worker",
    )
    assert answer.outcome == Outcome.LIMIT_REACHED and len(adapter.requests) == 2


@pytest.mark.asyncio
async def test_delegation_reserves_parent_budget_and_emits_parent_link():
    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "d", "name": "delegate", "arguments": {"agent_ref": "child@1", "objective": "help"}},),
            raw_usage={"input_tokens": 1, "output_tokens": 1},
        ),
        Turn(("child answer",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
        Turn(("parent answer",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    store = MemoryAgentStore()
    child = AgentDefinition("child", "1", "child")
    await store.put(child)
    parent = AgentDefinition(
        "parent",
        "1",
        "parent",
        Binding("openai_api", "test"),
        limits=CallLimits(max_cost_usd=Decimal("1"), max_turns=4),
        delegation=DelegationPolicy(("child@1",)),
    )
    events = []
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES), agents=store)
    answer = await runner.run_agent(parent, InstructionSet("finish"), {}, call_id="parent", events=events.append)
    assert answer.outcome == Outcome.ANSWERED
    assert adapter.requests[0][2]["limits"].max_cost_usd == Decimal("0.75")
    assert adapter.requests[1][2]["limits"].max_cost_usd == Decimal("0.25")
    assert all(e.parent_call_id == "parent" for e in events if e.call_id == "parent/delegate-1")


@pytest.mark.asyncio
async def test_parent_depth_policy_cannot_be_overridden_by_child():
    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "d", "name": "delegate", "arguments": {"agent_ref": "child@1", "objective": "help"}},),
            raw_usage={"input_tokens": 1, "output_tokens": 1},
        ),
        Turn(
            tool_calls=(
                {"id": "e", "name": "delegate", "arguments": {"agent_ref": "grandchild@1", "objective": "help"}},
            ),
            raw_usage={"input_tokens": 1, "output_tokens": 1},
        ),
        Turn(("bounded child",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
        Turn(("bounded parent",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    store = MemoryAgentStore()
    await store.put(AgentDefinition("grandchild", "1", "grandchild"))
    await store.put(
        AgentDefinition("child", "1", "child", delegation=DelegationPolicy(("grandchild@1",), max_depth=10))
    )
    parent = AgentDefinition(
        "parent",
        "1",
        "parent",
        Binding("openai_api", "test"),
        limits=CallLimits(max_cost_usd=Decimal("1"), max_turns=5),
        delegation=DelegationPolicy(("child@1",), max_depth=1),
    )
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES), agents=store)
    answer = await runner.run_agent(parent, InstructionSet("finish"), {}, call_id="parent")
    assert answer.outcome == Outcome.ANSWERED and len(adapter.requests) == 4
    assert "outside policy" in str(adapter.requests[2][2]["session"].history)


def test_public_auth_and_selection_are_validated_enums():
    definition = AgentDefinition("worker", "1", "worker")
    assert definition.model is ModelSelection.INHERIT
    assert AnthropicCredential("secret", "bearer").kind is AuthKind.BEARER
    with pytest.raises(ValueError):
        AnthropicCredential("secret", "mistyped")
    with pytest.raises(ValueError):
        AnthropicCredential("")
    with pytest.raises(ValueError):
        DelegationPolicy(budget_share=Decimal("1"))


@pytest.mark.asyncio
async def test_event_hub_assigns_unique_ids_and_rejects_duplicate_publications():
    received = []
    async with EventHub() as hub:

        async def handler(event):
            received.append(event)

        hub.subscribe("observer", handler)
        await hub.publish(CallEvent("harness.ready", "run"))
        await hub.publish(CallEvent("harness.ready", "run"))
        assert [event.event_id for event in received] == ["run:1", "run:2"]
        assert all(event.timestamp for event in received)
        with pytest.raises(EventDeliveryError, match="sequence"):
            await hub.publish(CallEvent("harness.ready", "run", sequence=2))


@pytest.mark.asyncio
async def test_journal_failure_prevents_delivery_and_allows_retry():
    delivered = []

    class Journal:
        broken = True

        async def append(self, event):
            if self.broken:
                raise OSError("unavailable")

    journal = Journal()
    async with EventHub(journal=journal) as hub:

        async def handler(event):
            delivered.append(event)

        hub.subscribe("observer", handler)
        with pytest.raises(OSError):
            await hub.publish(CallEvent(EventKind.USAGE, "run"))
        assert not delivered
        journal.broken = False
        await hub.publish(CallEvent(EventKind.USAGE, "run"))
        assert delivered[0].sequence == 1


@pytest.mark.asyncio
async def test_trace_bridge_records_usage_and_finishes_span():
    class Span:
        fields = {}
        events = []

        def set(self, name, value):
            self.fields[name] = value

        def add_event(self, name, **fields):
            self.events.append(name)

    class Manager:
        finished = False

        def __enter__(self):
            return span

        def __exit__(self, *args):
            self.finished = True

    class Trace:
        def span(self, *args, **kwargs):
            return manager

    span, manager = Span(), Manager()
    sink = TraceEventSink(Trace())
    await sink(CallEvent(EventKind.USAGE, "unknown"))
    await sink(CallEvent(EventKind.CALL_STARTED, "run", {"model": "test"}))
    await sink(CallEvent(EventKind.USAGE, "run", {"usage": {"input_uncached": 2, "input_cache_read": 3, "output": 4}}))
    await sink(CallEvent(EventKind.CALL_FINISHED, "run", {"cost_usd": "0.1"}))
    assert span.fields["gen_ai.usage.input_tokens"] == 5
    assert span.fields["gen_ai.usage.output_tokens"] == 4 and manager.finished
    assert not sink._spans


@pytest.mark.asyncio
async def test_triggered_skills_are_host_selected_digest_pinned_and_untrusted():
    from agentkit.agents.definitions.agent import TriggeredSkill
    from agentkit.agents.skills.resources import MemorySkillSource, SkillDocument, SkillRef

    ref = SkillRef("memory", "repair", "approved")
    document = SkillDocument(ref, "repair", "Repair guidance", "Untrusted repair body", {})
    adapter = ScriptedAdapter(Turn(("done",)), Turn(("done",)))
    runner = AgentRunner(
        CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES),
        sources={"memory": MemorySkillSource({"repair": document})},
    )
    definition = AgentDefinition(
        "worker",
        "1",
        "worker",
        Binding("openai_api", "test"),
        triggered_skills=(TriggeredSkill(ref, ("harness.repair_*",)),),
    )
    await runner.run_agent(definition, InstructionSet("finish"), {}, call_id="without")
    await runner.run_agent(
        definition,
        InstructionSet("finish"),
        {},
        call_id="with",
        triggering_events=(CallEvent("harness.repair_needed", "evidence"),),
    )
    assert document.body not in adapter.requests[0][1].user
    assert document.body in adapter.requests[1][1].user
    assert document.body not in adapter.requests[1][1].system
    changed = replace(document, ref=replace(ref, digest="changed"))
    runner.sources["memory"] = MemorySkillSource({"repair": changed})
    with pytest.raises(PermissionError):
        await runner.run_agent(
            definition,
            InstructionSet("finish"),
            {},
            call_id="changed",
            triggering_events=(CallEvent("harness.repair_needed", "evidence"),),
        )
    with pytest.raises(ValueError):
        TriggeredSkill(replace(ref, digest=None), ("*",))


@pytest.mark.asyncio
async def test_discoverable_skill_cannot_expand_tool_permissions():
    from agentkit.agents.skills.resources import MemorySkillSource, SkillDocument, SkillRef

    ref = SkillRef("memory", "repair", "approved")
    document = SkillDocument(ref, "repair", "Repair guidance", "body", {}, {"required_tools": ["dangerous"]})
    runner = AgentRunner(
        CallRuntime(adapters={"openai_api": ScriptedAdapter()}, pricing=PRICES),
        sources={"memory": MemorySkillSource({"repair": document})},
    )
    definition = AgentDefinition("worker", "1", "worker", Binding("openai_api", "test"), discoverable_skills=(ref,))
    with pytest.raises(PermissionError, match="unavailable tools"):
        await runner.run_agent(definition, InstructionSet("finish"), {}, call_id="worker")


@pytest.mark.asyncio
async def test_nested_event_queue_overflow_surfaces_without_deadlocking():
    hub = EventHub(queue_size=1)

    async def handler(event):
        if event.kind == "start":
            await hub.publish(CallEvent("next", "run"))
            await hub.publish(CallEvent("overflow", "run"))

    hub.subscribe("reactor", handler)
    with pytest.raises(EventDeliveryError):
        await hub.publish(CallEvent("start", "run"))
    with pytest.raises(ExceptionGroup):
        await hub.drain()
    await hub.aclose()
    with pytest.raises(ValueError):
        hub.subscribe("late", handler)


@pytest.mark.asyncio
async def test_skills_memory_resources_revalidate_and_sources_match():
    import hashlib

    from agentkit.agents.skills.resources import MemorySkillSource, SkillDocument, SkillRef, parse_skill

    content = "approved resource"
    ref = SkillRef("memory", "demo", "approved")
    doc = SkillDocument(
        ref, "demo", "description", "body", {"notes": (hashlib.sha256(content.encode()).hexdigest(), len(content))}
    )
    source = MemorySkillSource({"demo": doc}, {("demo", "notes"): content})
    assert await source.read(ref, "notes") == content
    assert await source.list() == (doc,)
    source.resources[("demo", "notes")] = "changed"
    with pytest.raises(PermissionError):
        await source.read(ref, "notes")
    with pytest.raises(PermissionError):
        await source.get(replace(ref, source="other"))
    for text in (
        "missing header",
        "---\nname: demo",
        "---\n[]\n---\nbody",
        "---\nname: Wrong\ndescription: x\n---\nbody",
    ):
        with pytest.raises(ValueError):
            parse_skill(text, SkillRef("memory", "demo"), {}, directory_name="demo")
    parsed = parse_skill(
        "---\nname: demo\ndescription: x\ncustom: preserve\n---\nbody",
        SkillRef("memory", "demo"),
        {},
        directory_name="demo",
    )
    assert parsed.unknown_fields == ("custom",)


@pytest.mark.asyncio
async def test_non_tool_minds_refuse_tools_instead_of_silently_ignoring():
    from agentkit.agents.minds import ChainOfThoughtMind, SingleCallMind
    from agentkit.tools import ToolRegistry
    from agentkit.tools.execution.policy import ToolSet

    async def evidence() -> str:
        """Read evidence for answering the current request."""
        return "evidence"

    tools = ToolSet(ToolRegistry.from_tools([evidence]))
    runtime = CallRuntime(adapters={"openai_api": ScriptedAdapter()}, pricing=PRICES)
    for mind in (ChainOfThoughtMind(), SingleCallMind()):
        with pytest.raises(ValueError, match="ReActMind"):
            await mind.run(runtime, Binding("openai_api", "test"), "task", tools)


@pytest.mark.asyncio
async def test_sequential_delegates_share_remaining_budget_with_explicit_child_cap():
    from agentkit.models.contracts.constants import CostSource

    call = {"name": "delegate", "arguments": {"agent_ref": "child@1", "objective": "help"}}
    adapter = ScriptedAdapter(
        Turn(tool_calls=({"id": "a", **call},), raw_usage={"input_tokens": 1}),
        Turn(("child one",), raw_usage={"input_tokens": 1}, reported_cost=Decimal("0.04")),
        Turn(tool_calls=({"id": "b", **call},), raw_usage={"input_tokens": 1}),
        Turn(("child two",), raw_usage={"input_tokens": 1}, reported_cost=Decimal("0.03")),
        Turn(("done",), raw_usage={"input_tokens": 1}),
    )
    store = MemoryAgentStore()
    await store.put(AgentDefinition("child", "1", "child", limits=CallLimits(max_cost_usd=Decimal("0.1"))))
    parent = AgentDefinition(
        "parent",
        "1",
        "parent",
        Binding("openai_api", "test"),
        limits=CallLimits(max_cost_usd=Decimal("1"), max_turns=5),
        delegation=DelegationPolicy(("child@1",)),
    )
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES), agents=store)
    answer = await runner.run_agent(parent, InstructionSet("finish"), {}, call_id="parent")
    assert answer.outcome == Outcome.ANSWERED
    assert [adapter.requests[i][2]["limits"].max_cost_usd for i in (1, 3)] == [Decimal("0.1"), Decimal("0.1")]
    assert answer.cost_usd == Decimal("0.070003")
    assert answer.cost_source != CostSource.UNPRICED


@pytest.mark.asyncio
async def test_concurrent_delegates_reserve_only_available_envelopes():
    import asyncio

    from agentkit.agents.minds import ReActMind
    from agentkit.core.contracts.types import Scope
    from agentkit.core.runtime import RunContext
    from agentkit.tools.execution.policy import ToolSet

    arrived = 0
    ready = asyncio.Event()
    caps = []

    class Adapter:
        async def turn(self, binding, prompt, **kwargs):
            nonlocal arrived
            caps.append(kwargs["limits"].max_cost_usd)
            arrived += 1
            if arrived == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 1)
            return Turn(("child",), raw_usage={"input_tokens": 1}, reported_cost=Decimal("0.02"))

    class ParallelMind:
        name = "parallel"
        results = []

        async def run(self, runtime, binding, prompt, tools: ToolSet, **kwargs):
            ctx = RunContext(scope=Scope(), correlation_id=kwargs["call_id"])
            self.results = await asyncio.gather(
                *[tools.execute("delegate", {"agent_ref": name, "objective": "help"}, ctx) for name in ("a@1", "b@1")]
            )
            return await runtime.ask(binding, prompt, **kwargs)

    mind = ParallelMind()
    store = MemoryAgentStore()
    for name in ("a", "b"):
        await store.put(AgentDefinition(name, "1", name, limits=CallLimits(max_cost_usd=Decimal("0.15"))))
    parent = AgentDefinition(
        "parent",
        "1",
        "parent",
        Binding("openai_api", "test"),
        mind="parallel",
        limits=CallLimits(max_cost_usd=Decimal("1")),
        delegation=DelegationPolicy(("a@1", "b@1"), max_concurrency=2),
    )
    runner = AgentRunner(
        CallRuntime(adapters={"openai_api": Adapter()}, pricing=PRICES),
        agents=store,
        minds={"parallel": mind, "react": ReActMind()},
    )
    answer = await runner.run_agent(parent, InstructionSet("finish"), {}, call_id="parent")
    assert answer.outcome == Outcome.ANSWERED and not any(result.is_error for result in mind.results)
    assert sorted(caps[:2]) == [Decimal("0.10"), Decimal("0.15")]
    assert sum(caps[:2]) == Decimal("0.25") and answer.cost_usd == Decimal("0.06")


@pytest.mark.asyncio
async def test_dispatched_child_exception_retains_unknown_spend_reservation():
    from agentkit.models.contracts.constants import CostSource

    class Runner(AgentRunner):
        child_caps = []

        async def _run_agent(self, definition, *args, **kwargs):
            if definition.id == "child":
                self.child_caps.append(definition.limits.max_cost_usd)
                raise RuntimeError("child failed after dispatch")
            return await super()._run_agent(definition, *args, **kwargs)

    call = {"name": "delegate", "arguments": {"agent_ref": "child@1", "objective": "help"}}
    adapter = ScriptedAdapter(
        Turn(tool_calls=({"id": "a", **call},), raw_usage={"input_tokens": 1}),
        Turn(tool_calls=({"id": "b", **call},), raw_usage={"input_tokens": 1}),
        Turn(tool_calls=({"id": "c", **call},), raw_usage={"input_tokens": 1}),
        Turn(("done",), raw_usage={"input_tokens": 1}),
    )
    store = MemoryAgentStore()
    await store.put(AgentDefinition("child", "1", "child", limits=CallLimits(max_cost_usd=Decimal("0.2"))))
    parent = AgentDefinition(
        "parent",
        "1",
        "parent",
        Binding("openai_api", "test"),
        limits=CallLimits(max_cost_usd=Decimal("1")),
        delegation=DelegationPolicy(("child@1",)),
    )
    runner = Runner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES), agents=store)
    answer = await runner.run_agent(parent, InstructionSet("finish"), {}, call_id="parent")
    assert runner.child_caps == [Decimal("0.2"), Decimal("0.05")]
    assert "delegation budget exhausted" in str(adapter.requests[-1][2]["session"].history)
    assert answer.cost_usd is None and answer.cost_source == CostSource.UNPRICED


@pytest.mark.asyncio
async def test_runtime_event_reactor_can_await_a_child_call_without_dispatch_deadlock():
    import asyncio

    adapter = ScriptedAdapter(
        Turn(("parent",), raw_usage={"input_tokens": 1}),
        Turn(("child",), raw_usage={"input_tokens": 1}),
    )
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES)
    observed = []
    async with EventHub(handler_timeout_s=1) as hub:

        async def reactor(event):
            observed.append(event)
            if event.call_id == "parent" and event.kind == EventKind.CALL_FINISHED:
                await hub.publish_reaction(event, "harness.ready", reaction_id="ready")
                child = await runtime.ask(
                    Binding("openai_api", "test"),
                    "child",
                    call_id="child",
                    parent_call_id="parent",
                    events=hub,
                )
                assert child.outcome == Outcome.ANSWERED

        hub.subscribe("reactor", reactor)
        await asyncio.wait_for(runtime.ask(Binding("openai_api", "test"), "parent", call_id="parent", events=hub), 2)
        await hub.drain()
    reactions = [event for event in observed if event.kind == "harness.ready"]
    assert reactions[0].call_id.startswith("parent/reaction-") and reactions[0].parent_call_id == "parent"
    assert len({event.event_id for event in observed}) == len(observed)
    assert any(event.call_id == "child" and event.kind == EventKind.CALL_FINISHED for event in observed)


@pytest.mark.asyncio
async def test_ambiguous_custom_event_does_not_poison_runtime_sequence():
    async with EventHub() as hub:
        await hub.publish(CallEvent(EventKind.CALL_STARTED, "run", sequence=1))
        with pytest.raises(EventDeliveryError, match="publish_reaction"):
            await hub.publish(CallEvent("harness.ready", "run"))
        await hub.publish(CallEvent(EventKind.CALL_FINISHED, "run", sequence=2))


@pytest.mark.asyncio
async def test_self_cancelled_handler_does_not_strand_followups_or_other_handlers():
    import asyncio

    seen = []
    hub = EventHub()

    async def cancels(event):
        if event.kind == "start":
            await hub.publish_reaction(event, "followup")
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    async def other(event):
        seen.append(event.kind)

    hub.subscribe("cancels", cancels)
    hub.subscribe("other", other)
    with pytest.raises(EventDeliveryError, match="CancelledError"):
        await hub.publish(CallEvent("start", "run"))
    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(hub.drain(), 1)
    assert seen == ["start", "followup"]
    await hub.aclose()


@pytest.mark.asyncio
async def test_real_worker_cancellation_clears_pending_acknowledgements():
    import asyncio

    entered = asyncio.Event()
    hub = EventHub()

    async def blocks(event):
        entered.set()
        await asyncio.Event().wait()

    hub.subscribe("blocker", blocks)
    first = asyncio.create_task(hub.publish(CallEvent("first", "run")))
    await entered.wait()
    second = asyncio.create_task(hub.publish(CallEvent("second", "run")))
    await asyncio.sleep(0)
    hub._worker.cancel()
    await asyncio.gather(first, second, hub._worker, return_exceptions=True)
    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(hub.drain(), 1)
    assert hub._queue.empty()


@pytest.mark.asyncio
async def test_tiny_agent_deadline_returns_a_limit_answer_before_dispatch():
    adapter = ScriptedAdapter(Turn(("unused",), raw_usage={"input_tokens": 1}))
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES))
    definition = AgentDefinition(
        "worker", "1", "worker", Binding("openai_api", "test"), limits=CallLimits(wall_clock_s=1e-20)
    )
    answer = await runner.run_agent(definition, InstructionSet("finish"), {}, call_id="worker")
    assert answer.outcome == Outcome.LIMIT_REACHED
    assert not adapter.requests


@pytest.mark.asyncio
async def test_verifier_slices_share_output_token_allowance():
    adapter = ScriptedAdapter(
        Turn(("incomplete",), raw_usage={"input_tokens": 1, "output_tokens": 4}),
        Turn(("complete",), raw_usage={"input_tokens": 1, "output_tokens": 2}),
    )
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES))
    definition = AgentDefinition(
        "worker", "1", "worker", Binding("openai_api", "test"), limits=CallLimits(max_output_tokens=6)
    )
    answer = await runner.run_agent(
        definition, InstructionSet("finish", verifier=lambda answer: answer.text == "complete"), {}, call_id="worker"
    )
    assert answer.outcome == Outcome.ANSWERED and answer.usage.output == 6
    assert adapter.requests[1][2]["limits"].max_output_tokens == 2
    adapter = ScriptedAdapter(Turn(("incomplete",), raw_usage={"input_tokens": 1, "output_tokens": 6}))
    runner = AgentRunner(CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES))
    answer = await runner.run_agent(
        definition, InstructionSet("finish", verifier=lambda _: False), {}, call_id="worker"
    )
    assert answer.outcome == Outcome.LIMIT_REACHED and len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_detached_event_descendant_restarts_idle_worker_before_drain():
    import asyncio

    release = asyncio.Event()
    tasks = []
    received = []
    async with EventHub() as hub:

        async def publish_later(parent):
            await release.wait()
            await hub.publish_reaction(parent, "later", reaction_id="detached")

        async def handler(event):
            received.append(event.kind)
            if event.kind == "start":
                tasks.append(asyncio.create_task(publish_later(event)))

        hub.subscribe("reactor", handler)
        await hub.publish(CallEvent("start", "run"))
        await hub.drain()
        assert hub._worker.done()
        release.set()
        await asyncio.gather(*tasks)
        await asyncio.wait_for(hub.drain(), 1)
        assert received == ["start", "later"]
