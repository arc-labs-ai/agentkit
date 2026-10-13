"""Hosted reference advertisements follow each invocation's authority."""

from dataclasses import replace
from decimal import Decimal

import jsonschema
import pytest

from agentkit.agents import AgentDefinition, AgentRunner, DelegationPolicy, InstructionSet, ToolGrant
from agentkit.agents.definitions.store import MemoryAgentStore
from agentkit.agents.skills.resources import FileSkillSource, SkillRef
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.types import Binding, CallLimits, Turn
from agentkit.tools import FunctionTool, ToolRegistry
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.asyncio
async def test_skill_reference_enums_refresh_from_approved_manifests_without_exposing_other_skills(tmp_path):
    for name in ("alpha", "blocked"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Approved workflow instructions.\n---\nPrivate body {name}."
        )
        (directory / f"{name}.txt").write_text("Private resource contents.")
    source = FileSkillSource(tmp_path)
    first = await source.get(SkillRef("filesystem", "alpha"))
    adapter = ScriptedAdapter(*(Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}) for _ in range(2)))
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, sources={"filesystem": source})
        definition = AgentDefinition("worker", "1", "Read approved instructions.",
                                     discoverable_skills=(first.ref,))
        await runner.run_agent(definition, InstructionSet("Read the approved skill."), {},
                               binding=Binding("openai_api", "test"), call_id="first")
        old = {s.name: s for s in adapter.requests[0][2]["tools"].schemas()}
        assert old["load_skill"].parameters["properties"]["name"]["enum"] == ["filesystem:alpha"]
        assert old["read_skill_resource"].parameters["properties"]["path"]["enum"] == ["SKILL.md", "alpha.txt"]
        assert "blocked" not in str(old) and "Private body" not in str(old)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"name": "filesystem:blocked"}, old["load_skill"].parameters)
        (tmp_path / "alpha" / "new.txt").write_text("Newly approved resource.")
        with pytest.raises(PermissionError, match="changed since approval"):
            await runner.run_agent(definition, InstructionSet("Read."), {},
                                   binding=Binding("openai_api", "test"), call_id="stale")
        assert len(adapter.requests) == 1
        refreshed = await source.get(SkillRef("filesystem", "alpha"))
        await runner.run_agent(replace(definition, discoverable_skills=(refreshed.ref,)),
                               InstructionSet("Read."), {}, binding=Binding("openai_api", "test"), call_id="fresh")
        current = {s.name: s for s in adapter.requests[1][2]["tools"].schemas()}
        assert current["read_skill_resource"].parameters["properties"]["path"]["enum"] == [
            "SKILL.md", "alpha.txt", "new.txt",
        ]
        assert "new.txt" not in str(old), "new invocation mutated an earlier advertisement"


@pytest.mark.asyncio
async def test_delegation_reference_enum_follows_current_grant_and_never_lists_other_store_agents():
    store = MemoryAgentStore()
    for name in ("alpha", "beta", "blocked"):
        await store.put(AgentDefinition(name, "1", f"Agent {name}."))
    adapter = ScriptedAdapter(*(Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}) for _ in range(2)))
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, agents=store, tools=ToolRegistry())
        definition = AgentDefinition("coordinator", "1", "Coordinate authorized workers.",
                                     limits=CallLimits(max_cost_usd=Decimal("1")),
                                     delegation=DelegationPolicy(("alpha@1",)))
        for index, allowed in enumerate(("alpha@1", "beta@1")):
            await runner.run_agent(replace(definition, delegation=DelegationPolicy((allowed,))),
                                   InstructionSet("Delegate through the hosted tool."), {},
                                   binding=Binding("openai_api", "test"), call_id=f"delegate-{index}")
            schema = adapter.requests[index][2]["tools"].schemas()[0]
            assert schema.name == "delegate"
            assert schema.parameters["properties"]["agent_ref"]["enum"] == [allowed]
            assert allowed in schema.description and "blocked" not in schema.description
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({"agent_ref": "blocked@1", "objective": "help"}, schema.parameters)
        old = adapter.requests[0][2]["tools"].schemas()[0]
        assert old.parameters["properties"]["agent_ref"]["enum"] == ["alpha@1"]


@pytest.mark.asyncio
async def test_skill_advertisement_requires_current_tool_grants_before_model_dispatch(tmp_path):
    directory = tmp_path / "lookup"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        "---\nname: lookup\ndescription: Read authorized records.\n"
        "metadata:\n  required_tools: [read_record]\n---\nUse the approved lookup tool."
    )
    source = FileSkillSource(tmp_path)
    approved = await source.get(SkillRef("filesystem", "lookup"))

    async def read_record() -> str:
        """Read a record authorized by the host's tool policy."""
        return "record"

    registry = ToolRegistry.from_tools([FunctionTool.from_callable(read_record, side_effecting=False)])
    adapter = ScriptedAdapter(Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}))
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, tools=registry, sources={"filesystem": source})
        definition = AgentDefinition("worker", "1", "Read authorized records.",
                                     discoverable_skills=(approved.ref,))
        with pytest.raises(PermissionError, match="unavailable tools"):
            await runner.run_agent(definition, InstructionSet("Read."), {},
                                   binding=Binding("openai_api", "test"), call_id="denied")
        assert not adapter.requests
        await runner.run_agent(replace(definition, tools=ToolGrant(allow=("read_record",))),
                               InstructionSet("Read."), {}, binding=Binding("openai_api", "test"), call_id="allowed")
        schema = next(s for s in adapter.requests[0][2]["tools"].schemas() if s.name == "load_skill")
        assert schema.parameters["properties"]["name"]["enum"] == ["filesystem:lookup"]
