"""Actual API adapters over scripted HTTP; CLI rows assert strict refusal."""

import pytest

from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CredentialRef
from agentkit.models.providers.frontiers import ApiAdapter
from tests.examples.plugin_execution.configuration.providers import MODES, provider_binding
from tests.examples.plugin_execution.wiring.runtime import compose_call, execute


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_same_contributions_across_provider_modes(mode, scripted_client):
    call = compose_call("matrix")
    if mode.endswith("_api"):
        client, requests = scripted_client(mode)
        key = "ANTHROPIC_API_KEY" if mode == "anthropic_api" else "OPENAI_API_KEY"
        binding = Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"}))
        async with (
            client,
            CallRuntime(
                adapters={mode: ApiAdapter(client=client)},
                native_tool_policy="deny",
                unpriced_policy="proceed_unpriced",
            ) as runtime,
        ):
            result = await execute(runtime, binding, call)
        assert result.outcome == "answered", result.failure
        assert result.value.model_dump() == {"record_id": "record-1", "status": "ready"}
        assert call.effects == ["record-1"] and len(requests) == 2
        assert result.attempts == 2 and result.usage_complete and result.usage.output == 2
        assert result.cost_usd is None
        assert [tool["name"] for tool in requests[0]["tools"]] == ["lookup_record"]
        assert (
            all(tool["type"] == "function" for tool in requests[0]["tools"])
            if mode == "openai_api"
            else all("input_schema" in tool for tool in requests[0]["tools"])
        )
        assert "ready" in str(requests[1])
        kinds = [event["kind"] for event in call.lifecycle.events]
        assert kinds == ["call_started", "tool_requested", "permission_decision", "tool_finished", "call_finished"]
        tools = [event for event in call.lifecycle.events if event["kind"].startswith("tool_")]
        assert tools[0]["invocation_id"] == tools[1]["invocation_id"]
        assert tools[0]["provider_tool_call_id"] == "provider-call-1"
    else:
        binding = provider_binding(mode)
        binding = Binding(mode, binding.model, CredentialRef(lambda: pytest.fail("credential resolved")))
        async with CallRuntime(native_tool_policy="deny", unpriced_policy="proceed_unpriced") as runtime:
            result = await execute(runtime, binding, call)
        assert result.failure.classification == "execution_incompatible"
        assert result.attempts == 0 and not call.effects and not result.usage_complete
        assert result.cost_usd is None
        assert result.failure.mode_detail["dispatch_state"] == "not_started"
        assert [event["kind"] for event in call.lifecycle.events] == ["call_started", "call_finished"]
