"""Real shared host policy, scripted provider input; no paid calls.

The property under test is one property, stated once and demonstrated in all
four modes: a host permission decision is enforced BEFORE any tool effect,
whichever provider was selected. The two API modes reach the policy through
`ApiAdapter`, the two CLI modes through the MCP bridge `ToolSet.serve()` hands
the CLI. The decision itself is the same object in both — a CLI call arrives at
`ToolSet.execute` with this call's permission contributions and observers
attached, rather than at a second copy of the rule.

The CLI rows are driven by a scripted binary (`scripted_cli`). They prove the
TRANSPORT and the policy path behind it. They do NOT prove that a real
`claude`/`codex` process is contained — no process, native tool, sandbox or
credential exists here, and the example's strict profile refuses both CLI modes
outright, which `test_cli_containment.py` pins. Real-process characterization
lives in `tests/source/integration/execution`.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agentkit.core import CancellationToken
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CredentialRef
from agentkit.models.providers.cli import CliAdapter
from agentkit.models.providers.frontiers import ApiAdapter
from agentkit.tools import PermissionDecision
from tests.examples.plugin_execution.configuration.providers import HOST_TOOL_PATH_REQUIREMENTS
from tests.examples.plugin_execution.wiring.runtime import compose_call, execute

API_MODES = ["openai_api", "anthropic_api"]
CLI_MODES = ["claude_code", "codex"]
DEFECTS = ["denied", "unknown", "unattended"]
API_CODES = {"denied": "permission_denied", "unattended": "permission_unattended", "unknown": "tool_error"}
# A CLI mode reaches the host tool over MCP, where an UNDECLARED name is not
# advertised at all, so the server refuses it before the host policy path is
# entered and there is no host lifecycle phase to report. The API adapters
# route the same mistake through dispatch, which reports `tool_error`. Both are
# refusals before any handler; they differ in how early, and a test that
# flattened them would be asserting a shape rather than the guarantee.
CLI_CODES = {"denied": "permission_denied", "unattended": "permission_unattended", "unknown": None}


def synthetic(mode):
    """A binding whose credential is a constant, so no real key can be used."""
    key = "ANTHROPIC_API_KEY" if mode in ("anthropic_api", "claude_code") else "OPENAI_API_KEY"
    return Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"}))


def refusal_call(defect):
    """This call composed so that `defect` is the only thing wrong with it."""
    # Composed in, not assigned on: `ToolSet` is frozen, so its validation is an
    # invariant rather than advice, and a host adds a hook contribution through
    # the documented argument the way a second hook library would.
    return compose_call(
        "refusal",
        granted_ids=frozenset() if defect == "denied" else frozenset({"record-1"}),
        permissions=(lambda request: PermissionDecision.ASK,) if defect == "unattended" else (),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", API_MODES)
@pytest.mark.parametrize("defect", DEFECTS)
async def test_prohibited_invocations_have_no_effect(mode, defect, scripted_client):
    """A denied, undeclared or unapprovable call must leave the collection untouched.

    Without this, a host could only discover the ordering by inspecting its own
    data afterwards: a policy consulted after the handler, or an ASK that falls
    through to allow because nobody is attending, both produce a plausible
    answer and a record that was already read or written.
    """
    call = refusal_call(defect)
    client, _ = scripted_client(
        mode, tool_name="undeclared" if defect == "unknown" else "lookup_record", status="unavailable"
    )
    async with (
        client,
        CallRuntime(
            adapters={mode: ApiAdapter(client=client)}, native_tool_policy="deny", unpriced_policy="proceed_unpriced"
        ) as runtime,
    ):
        await execute(runtime, synthetic(mode), call)
    assert not call.effects
    event = next(event for event in call.lifecycle.events if event["kind"] == "tool_finished")
    assert event["code"] == API_CODES[defect]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", CLI_MODES)
async def test_an_allowed_cli_invocation_reaches_the_handler_and_the_model(mode, scripted_cli):
    """The clean twin of the CLI refusals: an allowed call must actually work.

    A refusal test passes just as happily when the bridge is broken and nothing
    ever reaches a handler, so the denials below mean nothing without this. It
    is also where a consumer sees what the bridge does and does not carry: the
    host result comes back as structured content the CLI can hand its model,
    the observer gets the same host invocation ID for both phases, and there is
    no provider tool-call ID to correlate on because MCP does not supply one.

    Scripted binary: this is the transport and the policy path, not evidence
    about a real CLI process.
    """
    call = compose_call("allowed")
    cli = scripted_cli(mode)
    async with CallRuntime(
        adapters={mode: CliAdapter(spawn=cli.spawn)},
        native_tool_policy="allow",
        unpriced_policy="proceed_unpriced",
    ) as runtime:
        result = await execute(runtime, synthetic(mode), call, execution=HOST_TOOL_PATH_REQUIREMENTS)
    assert result.outcome == "answered", result.failure
    assert result.value.model_dump() == {"record_id": "record-1", "status": "ready"}
    assert call.effects == ["record-1"]
    assert not cli.results[0].isError
    assert cli.results[0].structuredContent == {"record_id": "record-1", "status": "ready"}
    kinds = [event["kind"] for event in call.lifecycle.events]
    assert kinds == ["call_started", "tool_requested", "permission_decision", "tool_finished", "call_finished"]
    phases = [event for event in call.lifecycle.events if event["kind"].startswith("tool_")]
    assert phases[0]["invocation_id"] == phases[1]["invocation_id"]
    assert all(event["provider_tool_call_id"] is None for event in phases)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", CLI_MODES)
@pytest.mark.parametrize("defect", DEFECTS)
async def test_prohibited_cli_invocations_reach_no_handler_either(mode, defect, scripted_cli):
    """The same three refusals, carried over the CLI's MCP bridge instead of HTTP.

    This is the row a consumer reading only the example could not previously
    check. The CLI path is the one where a host could plausibly believe its
    policy had been left behind — the decision is made in the host process
    while the caller is a separate program talking MCP — so "nothing happened"
    has to be demonstrated there and not inferred from the API modes. Without
    it, a bridge that answered the CLI out of its own copy of the registry, or
    that ran the handler and then consulted the hook, would pass every test
    this example ships.

    Scripted binary: this is the transport and the policy path, not evidence
    about a real CLI process.
    """
    call = refusal_call(defect)
    cli = scripted_cli(mode, tool_name="undeclared" if defect == "unknown" else "lookup_record", status="unavailable")
    async with CallRuntime(
        adapters={mode: CliAdapter(spawn=cli.spawn)},
        # Not `deny`: that policy requires the containment the CLI modes cannot
        # verify, which refuses this call before dispatch. See
        # `HOST_TOOL_PATH_REQUIREMENTS`.
        native_tool_policy="allow",
        unpriced_policy="proceed_unpriced",
    ) as runtime:
        await execute(runtime, synthetic(mode), call, execution=HOST_TOOL_PATH_REQUIREMENTS)
    assert not call.effects
    # The CLI got an error result it can read, carrying the safe code and no
    # record data — the same refusal the API modes hand their model.
    result = cli.results[0]
    assert result.isError and (result.structuredContent or {}).get("code") == CLI_CODES[defect]
    finished = [event["code"] for event in call.lifecycle.events if event["kind"] == "tool_finished"]
    assert finished == ([] if CLI_CODES[defect] is None else [CLI_CODES[defect]])
    # Provider tool-call IDs are correlation data and the CLI bridge has none to
    # give; the host invocation ID is what the observer correlates on.
    tool_events = [event for event in call.lifecycle.events if event["kind"].startswith("tool_")]
    assert all(event["provider_tool_call_id"] is None for event in tool_events)
    assert all(event["call_id"] == "refusal" for event in call.lifecycle.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", API_MODES)
async def test_cancellation_joins_blocked_permission_and_settles_once(mode, scripted_client):
    """A run cancelled while authorization is pending must clean up and settle once.

    A permission hook that waits on a human can be waiting when the run is
    abandoned. Without this, the two failures are silent: the awaiting hook is
    dropped without its `finally` running, and the call reports a terminal
    outcome twice — once for the cancellation and once for the request that was
    already in flight — which double-counts in a host's own accounting.
    """
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

    # `replace`, not assignment: this test deliberately REPLACES the call's
    # permission policy rather than adding to it, and `replace` says exactly
    # that while re-running `ToolSet.__post_init__` — so the substituted policy
    # is validated like any other. Assignment would have skipped that, which is
    # why the dataclass is frozen.
    call.tools = replace(call.tools, permissions=(blocked,))
    client, requests = scripted_client(mode)
    async with (
        client,
        CallRuntime(
            adapters={mode: ApiAdapter(client=client)}, native_tool_policy="deny", unpriced_policy="proceed_unpriced"
        ) as runtime,
    ):
        pending = asyncio.create_task(execute(runtime, synthetic(mode), call, cancel=token))
        await entered.wait()
        token.cancel()
        result = await asyncio.wait_for(pending, 2)
    assert result.outcome == "cancelled" and cleaned.is_set() and not call.effects
    assert len(requests) == 1
    assert len([event for event in call.lifecycle.events if event["kind"] == "call_finished"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", CLI_MODES)
async def test_cancellation_joins_blocked_cli_permission_and_settles_once(mode, scripted_cli):
    """Cancelling across the CLI bridge still joins the hook and settles exactly once.

    Here the pending authorization has a second party: the scripted binary is
    blocked on an MCP response that will never come. Without this, a bridge
    could answer that request on the host's behalf to unblock the caller — a
    default-allow built out of cleanup — or leave the invocation dangling so
    the run reports an outcome while a hook is still running, and the host's
    own ledger ends up with two terminal records for one call.

    Scripted binary: this is the transport and the policy path, not evidence
    about a real CLI process.
    """
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

    # Replaced rather than added, for the reason the API twin above spells out.
    call.tools = replace(call.tools, permissions=(blocked,))
    cli = scripted_cli(mode)
    async with CallRuntime(
        adapters={mode: CliAdapter(spawn=cli.spawn)},
        native_tool_policy="allow",
        unpriced_policy="proceed_unpriced",
    ) as runtime:
        pending = asyncio.create_task(
            execute(runtime, synthetic(mode), call, cancel=token, execution=HOST_TOOL_PATH_REQUIREMENTS)
        )
        await entered.wait()
        token.cancel()
        result = await asyncio.wait_for(pending, 5)
    assert result.outcome == "cancelled" and cleaned.is_set() and not call.effects
    # No tool result was ever produced for the caller — not an allow, not a
    # denial. The invocation was torn down, and the double recorded that.
    assert not cli.results and cli.cancellations == ["lookup_record"]
    assert [event["code"] for event in call.lifecycle.events if event["kind"] == "tool_finished"] == ["cancelled"]
    assert len([event for event in call.lifecycle.events if event["kind"] == "call_finished"]) == 1
