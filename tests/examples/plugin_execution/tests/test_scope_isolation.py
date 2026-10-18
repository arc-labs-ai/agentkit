"""Concurrent host scopes; transport-token isolation is tested in core conformance."""

import asyncio

import pytest

from agentkit.core import Scope
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CredentialRef
from agentkit.models.providers.frontiers import ApiAdapter
from tests.examples.plugin_execution.wiring.runtime import compose_call, execute


@pytest.mark.asyncio
async def test_concurrent_calls_cannot_read_the_other_collection(scripted_client):
    left = compose_call("left", scope=Scope(org_id=1))
    right = compose_call(
        "right", records={"record-2": "private"}, granted_ids=frozenset({"record-2"}), scope=Scope(org_id=2)
    )
    client_left, _ = scripted_client("openai_api")
    client_right, _ = scripted_client("openai_api", status="unavailable")
    binding = Binding("openai_api", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "synthetic-test-only"}))
    async with (
        client_left,
        client_right,
        CallRuntime(
            adapters={"openai_api": ApiAdapter(client=client_left)},
            native_tool_policy="deny",
            unpriced_policy="proceed_unpriced",
        ) as runtime_left,
        CallRuntime(
            adapters={"openai_api": ApiAdapter(client=client_right)},
            native_tool_policy="deny",
            unpriced_policy="proceed_unpriced",
        ) as runtime_right,
    ):
        await asyncio.gather(execute(runtime_left, binding, left), execute(runtime_right, binding, right))
    assert left.effects == ["record-1"] and not right.effects
    assert any(event.get("code") == "permission_denied" for event in right.lifecycle.events)
    assert all(event["call_id"] == "left" for event in left.lifecycle.events)
    assert all(event["call_id"] == "right" for event in right.lifecycle.events)
