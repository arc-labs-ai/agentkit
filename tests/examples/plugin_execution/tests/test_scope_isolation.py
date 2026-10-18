"""Concurrent host scopes; transport-token isolation is tested in core conformance.

The grant that matters here is not "may this tool run" but "may this call read
THIS record", and it comes from `ToolInvocationRequest.scope` — trusted host
context — rather than from anything the model said. Two calls in flight at once
are what make that checkable, and the property is claimed for all four modes,
so it is demonstrated in all four: the API modes through `ApiAdapter`, the CLI
modes through the MCP bridge `ToolSet.serve()` hands the CLI.

The CLI rows use a scripted binary. They prove the transport and the policy
path, not containment of a real CLI process — see `tests/conftest.py` and
`test_cli_containment.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.core import Scope
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CredentialRef
from agentkit.models.providers.cli import CliAdapter
from agentkit.models.providers.frontiers import ApiAdapter
from tests.examples.plugin_execution.configuration.providers import HOST_TOOL_PATH_REQUIREMENTS
from tests.examples.plugin_execution.wiring.runtime import compose_call, execute


def synthetic(mode):
    """A binding whose credential is a constant, so no real key can be used."""
    key = "ANTHROPIC_API_KEY" if mode in ("anthropic_api", "claude_code") else "OPENAI_API_KEY"
    return Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"}))


def two_tenants():
    """Two calls that differ only in tenant scope, grant and collection."""
    return (
        compose_call("left", scope=Scope(org_id=1)),
        compose_call(
            "right", records={"record-2": "private"}, granted_ids=frozenset({"record-2"}), scope=Scope(org_id=2)
        ),
    )


def assert_only_the_granted_tenant_read_its_record(left, right):
    """Right asked for left's record; only left's own collection was touched."""
    assert left.effects == ["record-1"] and not right.effects
    assert any(event.get("code") == "permission_denied" for event in right.lifecycle.events)
    assert all(event["call_id"] == "left" for event in left.lifecycle.events)
    assert all(event["call_id"] == "right" for event in right.lifecycle.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_concurrent_calls_cannot_read_the_other_collection(mode, scripted_client):
    """Each call's grant and collection must stay its own while both are in flight.

    Without this, a host would learn only in production that its tenant check
    was effectively global: a registry, callback or collection hoisted to
    module scope, or a permission hook that closed over the wrong call, serves
    one tenant's record to the other and still answers both correctly. The
    lifecycle assertions matter for the same reason — a shared observer that
    mixes two calls' events makes an audit trail that cannot be read back.
    """
    left, right = two_tenants()
    client_left, _ = scripted_client(mode)
    client_right, _ = scripted_client(mode, status="unavailable")
    async with (
        client_left,
        client_right,
        CallRuntime(
            adapters={mode: ApiAdapter(client=client_left)},
            native_tool_policy="deny",
            unpriced_policy="proceed_unpriced",
        ) as runtime_left,
        CallRuntime(
            adapters={mode: ApiAdapter(client=client_right)},
            native_tool_policy="deny",
            unpriced_policy="proceed_unpriced",
        ) as runtime_right,
    ):
        binding = synthetic(mode)
        await asyncio.gather(execute(runtime_left, binding, left), execute(runtime_right, binding, right))
    assert_only_the_granted_tenant_read_its_record(left, right)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
async def test_concurrent_cli_calls_cannot_read_the_other_collection(mode, scripted_cli):
    """Two concurrent CLI sessions get two separate grants, not one shared bridge.

    Each call stands up its own host-tool MCP server, and the scripted binaries
    reach them at the same time over their own authenticated endpoints. Without
    this, a bridge that cached a server, a registry or a context per PROCESS
    rather than per call would answer the second tenant out of the first
    tenant's scope — and every single-call CLI test would still pass, because
    the defect only appears once two calls overlap.

    Scripted binaries: this is the transport and the policy path, not evidence
    about real CLI processes.
    """
    left, right = two_tenants()
    cli_left = scripted_cli(mode)
    cli_right = scripted_cli(mode, status="unavailable")
    async with (
        CallRuntime(
            adapters={mode: CliAdapter(spawn=cli_left.spawn)},
            # Not `deny`: that policy requires containment the CLI modes cannot
            # verify, which refuses before dispatch. See the requirement's own
            # comment in `configuration/providers.py`.
            native_tool_policy="allow",
            unpriced_policy="proceed_unpriced",
        ) as runtime_left,
        CallRuntime(
            adapters={mode: CliAdapter(spawn=cli_right.spawn)},
            native_tool_policy="allow",
            unpriced_policy="proceed_unpriced",
        ) as runtime_right,
    ):
        binding = synthetic(mode)
        await asyncio.gather(
            execute(runtime_left, binding, left, execution=HOST_TOOL_PATH_REQUIREMENTS),
            execute(runtime_right, binding, right, execution=HOST_TOOL_PATH_REQUIREMENTS),
        )
    assert_only_the_granted_tenant_read_its_record(left, right)
    # Each bridge answered its own caller: one granted read, one refusal.
    assert not cli_left.results[0].isError
    assert cli_right.results[0].isError
    assert cli_right.results[0].structuredContent == {"code": "permission_denied"}
