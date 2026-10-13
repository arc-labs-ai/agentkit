"""Verify the runnable usage path across memory, context, grants and execution."""

import asyncio

import pytest

from agentkit.core.contracts.types import Message, MessageRole, ToolCall
from agentkit.core.events import EventHub
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, Outcome
from agentkit.models.contracts.types import Turn
from agentkit.tools import InMemoryFiles
from tests.examples.support_harness import SupportHarness
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


def _order_call(order: str = "owned") -> Turn:
    return Turn(
        tool_calls=({"id": "lookup", "name": "inspect_order", "arguments": {"order_id": order}},),
        canonical_messages=(
            Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("lookup", "inspect_order", {"order_id": order}),)),
        ),
        raw_usage={"input_tokens": 10, "output_tokens": 2},
    )


def test_public_harness_retrieves_evidence_runs_tool_preserves_context_and_saves_note():
    async def run():
        files = InMemoryFiles()
        await files.create("/orgs/1/approved/refunds.md", "Refunds require approval before execution.")
        await files.create("/orgs/2/approved/refunds.md", "Other organization private policy.")
        adapter = ScriptedAdapter(
            _order_call(),
            Turn(("Duplicate captured; request refund approval.",), raw_usage={"input_tokens": 10, "output_tokens": 8}),
        )
        observations = []

        async def audit(event):
            observations.append(event)

        async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
            async with EventHub() as events:
                events.subscribe("audit", audit)
                harness = SupportHarness(
                    runtime,
                    Binding("openai_api", "test"),
                    org_id=1,
                    files=files,
                    orders={"owned": "Duplicate captured; no refund issued."},
                    events=events,
                )
                result = await harness.respond("How should we handle this payment?", order_id="owned")
                assert result.outcome == Outcome.ANSWERED
                evidence = harness.context.prefix.grounding[0].content
                assert "Refunds require approval" in evidence
                assert "Other organization" not in evidence
                assert any(
                    m.role == MessageRole.TOOL and "Duplicate captured" in m.content for m in harness.context.messages
                )
                await harness.save_note(result)
                assert "/orgs/1/notes/" in await files.view("/orgs/1/notes/")
                assert "Duplicate captured" not in harness.context.prefix.grounding[0].content
                host_view = harness.context
                host_view.note("review", "pending")
                assert harness.context.get("review") is None
                harness.replace_context(host_view)
                assert harness.context.get("review") == "pending"
                assert observations[0].kind == "call_started" and observations[-1].kind == "call_finished"

    asyncio.run(run())


def test_missing_approved_memory_stops_before_dispatch():
    async def run():
        adapter = ScriptedAdapter(Turn(("Should not execute",)))
        async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
            harness = SupportHarness(runtime, Binding("openai_api", "test"), org_id=1, files=InMemoryFiles(), orders={})
            with pytest.raises(LookupError, match="before model dispatch"):
                await harness.respond("Where is the refund?", order_id="owned")
            assert not adapter.requests

    asyncio.run(run())


def test_order_tool_cannot_read_outside_host_authorized_view():
    async def run():
        files = InMemoryFiles()
        await files.create("/orgs/1/approved/refunds.md", "Refunds require approval.")
        adapter = ScriptedAdapter(
            _order_call("other-org-order"),
            Turn(("Order inaccessible; escalate.",), raw_usage={"input_tokens": 10, "output_tokens": 8}),
        )
        async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
            harness = SupportHarness(
                runtime, Binding("openai_api", "test"), org_id=1, files=files, orders={"owned": "private"}
            )
            await harness.respond("Inspect an order", order_id="owned")
            results = [m for m in harness.context.messages if m.role == MessageRole.TOOL]
            assert results and results[0].metadata["is_error"]
            assert "private" not in results[0].content
            with pytest.raises(FileNotFoundError):
                await files.view("/orgs/1/notes/")

    asyncio.run(run())


@pytest.mark.asyncio
async def test_agent_output_contract_survives_verifier_continuation():
    from agentkit.agents.definitions.agent import AgentDefinition
    from agentkit.agents.definitions.instructions import InstructionSet
    from agentkit.agents.execution.runner import AgentRunner
    from agentkit.models.contracts.schema import OutputSpec
    from agentkit.state.context import WorkingContext

    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["review", "accepted"]}},
        "required": ["status"],
        "additionalProperties": False,
    }
    adapter = ScriptedAdapter(
        Turn(("first submission",), value={"status": "review"}, raw_usage={"input_tokens": 1, "output_tokens": 1}),
        Turn(("second submission",), value={"status": "review"}, raw_usage={"input_tokens": 1, "output_tokens": 1}),
        Turn(("accepted submission",), value={"status": "accepted"}, raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    context = WorkingContext()
    submissions = []

    def verify(answer):
        submissions.append(answer.value)
        if answer.value["status"] == "review":
            context.append(Message("user", "Review failed; submit the accepted typed receipt."))
            return False
        return True

    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        answer = await AgentRunner(runtime).run_agent(
            AgentDefinition(id="typed-review", version="1", description="Review a typed completion receipt"),
            InstructionSet("Return the reviewed receipt.", verifier=verify),
            {},
            binding=Binding("openai_api", "test"),
            output=OutputSpec(schema),
            context=context,
            call_id="typed-review",
        )
    assert answer.outcome == Outcome.ANSWERED and answer.value == {"status": "accepted"}
    assert submissions == [{"status": "review"}, {"status": "review"}, {"status": "accepted"}]
    assert len(adapter.requests) == 3
    assert all(request[2]["schema"] == schema for request in adapter.requests)
