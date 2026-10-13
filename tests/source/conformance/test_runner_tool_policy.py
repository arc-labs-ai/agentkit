"""Filtering agent grants preserves policy and isolates deferred discovery."""

import pytest

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet, ToolGrant
from agentkit.core.runtime.calls import CallRuntime
from agentkit.core.testing import make_test_ctx
from agentkit.models.contracts.types import Binding, Turn
from agentkit.tools import ToolRegistry, tool
from agentkit.tools.execution.policy import OverflowPolicy, ToolSet
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.asyncio
async def test_runner_preserves_policy_filters_denied_tools_and_starts_fresh_discovery(tmp_path):
    called = []

    @tool(side_effecting=False)
    def allowed_record() -> str:
        """Read an authorized customer record."""
        called.append("allowed")
        return "record"

    @tool(side_effecting=False)
    def denied_record() -> str:
        """Read a forbidden customer record."""
        pytest.fail("denied tool body executed")

    seen = []
    policy = ToolSet(ToolRegistry.from_tools([allowed_record, denied_record]),
                     overflow=OverflowPolicy(512, tmp_path), images=False, timeout_s=0.2,
                     deferred=("allowed_record", "denied_record"),
                     result_hook=lambda name, value: seen.append(name) or value)
    await policy.execute("search_tools", {"query": "customer record"}, make_test_ctx())
    assert "allowed_record" in [schema.name for schema in policy.schemas()]

    class InspectPolicy(ScriptedAdapter):
        async def turn(self, binding, prompt, **kwargs):
            scoped = kwargs["tools"]
            assert scoped is not policy and scoped.images is False
            assert scoped.timeout_s == 0.2 and scoped.overflow is policy.overflow
            assert scoped.registry.names() == ["allowed_record"]
            assert "allowed_record" not in [schema.name for schema in scoped.schemas()]
            denied = await scoped.execute("denied_record", {}, make_test_ctx())
            assert denied.is_error
            await scoped.execute("search_tools", {"query": "customer record"}, make_test_ctx())
            result = await scoped.execute("call_loaded_tool", {
                "name": "allowed_record", "arguments_json": "{}"}, make_test_ctx())
            assert result.text == "record"
            return await super().turn(binding, prompt, **kwargs)

    adapter = InspectPolicy(*(Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}) for _ in range(2)))
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, tools=policy)
        definition = AgentDefinition("support", "1", "Read authorized customer records.",
                                     tools=ToolGrant(allow=("*",), deny=("denied_record",)))
        for number in range(2):
            answer = await runner.run_agent(definition, InstructionSet("Read the record."), {},
                                            binding=Binding("openai_api", "test"), call_id=f"scope-{number}")
            assert answer.outcome == "answered"
    assert called == ["allowed", "allowed"] and seen.count("allowed_record") == 2
