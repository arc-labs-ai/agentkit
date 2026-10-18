"""Real shared host policy, scripted provider input; no paid calls."""

import asyncio

import pytest

from agentkit.core import CancellationToken
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CredentialRef
from agentkit.models.providers.frontiers import ApiAdapter
from agentkit.tools import PermissionDecision
from tests.examples.plugin_execution.wiring.runtime import compose_call, execute


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
@pytest.mark.parametrize("defect", ["denied", "unknown", "unattended"])
async def test_prohibited_invocations_have_no_effect(mode, defect, scripted_client):
    call = compose_call("refusal", granted_ids=frozenset() if defect == "denied" else frozenset({"record-1"}))
    if defect == "unattended":
        call.tools.permissions = (*call.tools.permissions, lambda request: PermissionDecision.ASK)
    client, _ = scripted_client(
        mode, tool_name="undeclared" if defect == "unknown" else "lookup_record", status="unavailable"
    )
    key = "ANTHROPIC_API_KEY" if mode == "anthropic_api" else "OPENAI_API_KEY"
    binding = Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"}))
    async with (
        client,
        CallRuntime(
            adapters={mode: ApiAdapter(client=client)}, native_tool_policy="deny", unpriced_policy="proceed_unpriced"
        ) as runtime,
    ):
        await execute(runtime, binding, call)
    assert not call.effects
    event = next(event for event in call.lifecycle.events if event["kind"] == "tool_finished")
    assert (
        event["code"]
        == {"denied": "permission_denied", "unattended": "permission_unattended", "unknown": "tool_error"}[defect]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_cancellation_joins_blocked_permission_and_settles_once(mode, scripted_client):
    call = compose_call("cancelled")
    entered, cleaned = asyncio.Event(), asyncio.Event()
    token = CancellationToken()

    async def blocked(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return PermissionDecision.ALLOW

    call.tools.permissions = (blocked,)
    client, requests = scripted_client(mode)
    key = "ANTHROPIC_API_KEY" if mode == "anthropic_api" else "OPENAI_API_KEY"
    async with (
        client,
        CallRuntime(
            adapters={mode: ApiAdapter(client=client)}, native_tool_policy="deny", unpriced_policy="proceed_unpriced"
        ) as runtime,
    ):
        pending = asyncio.create_task(
            execute(
                runtime, Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"})), call, cancel=token
            )
        )
        await entered.wait()
        token.cancel()
        result = await asyncio.wait_for(pending, 2)
    assert result.outcome == "cancelled" and cleaned.is_set() and not call.effects
    assert len(requests) == 1
    assert len([event for event in call.lifecycle.events if event["kind"] == "call_finished"]) == 1
