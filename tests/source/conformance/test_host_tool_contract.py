"""Host policy tests; scripted transport coverage is not CLI confinement evidence."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agentkit.core.execution.concurrency import CancellationToken, Cancelled
from agentkit.core.testing import make_test_ctx
from agentkit.tools import ToolRegistry, tool
from agentkit.tools.execution.policy import ToolSet, serve


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge", [False, True], ids=["host", "mcp"])
@pytest.mark.parametrize("decision", ["allow", "deny", "ask"])
async def test_permission_hooks_and_identity_share_one_policy_path(bridge, decision):
    from mcp.shared.memory import create_connected_server_and_client_session

    from agentkit.tools import PermissionDecision, current_tool_invocation

    effects, events, checked = [], [], []

    @tool(side_effecting=True)
    def change_record(ctx, value: int) -> dict:
        """Change a record after host authorization."""
        effects.append((value, ctx.correlation_id, current_tool_invocation()))
        return {"value": value}

    async def permission(request):
        checked.append(request)
        return PermissionDecision(decision)

    tools = ToolSet(ToolRegistry.from_tools([change_record]), permissions=(permission,), observers=(events.append,))
    ctx = make_test_ctx()
    if bridge:
        spec = serve(tools, ctx=ctx)
        try:
            async with create_connected_server_and_client_session(spec.build_server()) as session:
                result = await session.call_tool("change_record", {"value": 7})
                error = result.isError
        finally:
            await spec.stop()
    else:
        result = await tools.execute("change_record", {"value": 7}, ctx)
        error = result.is_error
    assert error is (decision != "allow")
    assert len(effects) == (1 if decision == "allow" else 0)
    assert len(checked) == 1
    invocation = checked[0].invocation
    assert invocation.host_invocation_id and invocation.call_id == ctx.correlation_id
    assert [event.phase for event in events] == ["requested", "finished"]
    assert all(event.invocation == invocation for event in events)
    assert events[-1].code == {"allow": "ok", "deny": "permission_denied", "ask": "permission_unattended"}[decision]
    if effects:
        assert effects[0][2] == invocation
    assert current_tool_invocation() is None


@pytest.mark.asyncio
async def test_deny_is_final_and_approval_is_explicit():
    from agentkit.tools import PermissionDecision

    effects, prompts = [], []

    @tool(side_effecting=True, requires_approval=True)
    def mutate() -> str:
        """Perform an operation requiring explicit approval."""
        effects.append(True)
        return "ok"

    async def deny(request):
        return PermissionDecision.DENY

    async def ask(request):
        return PermissionDecision.ASK

    async def approve(request):
        prompts.append(request)
        return PermissionDecision.ALLOW

    reg = ToolRegistry.from_tools([mutate])
    blocked = ToolSet(reg, permissions=(ask, deny), approval=approve)
    assert (await blocked.execute("mutate", {}, make_test_ctx())).is_error
    assert not effects and not prompts
    assert not (await ToolSet(reg, approval=approve).execute("mutate", {}, make_test_ctx())).is_error
    assert len(effects) == len(prompts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["unknown", "invalid", "oversize"])
async def test_validation_precedes_permissions_and_effects(defect):
    from agentkit.tools import PermissionDecision

    effects, checked = [], []

    @tool(side_effecting=True)
    def write(value: int) -> str:
        """Write one validated integer to the record."""
        effects.append(value)
        return "ok"

    def permission(request):
        checked.append(request)
        return PermissionDecision.ALLOW

    tools = ToolSet(ToolRegistry.from_tools([write]), permissions=(permission,), max_input_bytes=128)
    result = await tools.execute(
        "missing" if defect == "unknown" else "write",
        {"value": "x" * 200 if defect == "oversize" else "invalid"},
        make_test_ctx(),
    )
    assert result.is_error and not effects and not checked


@pytest.mark.asyncio
async def test_custom_tool_schema_validated_and_arguments_frozen():
    from agentkit.core import ToolSchema

    effects = []

    class Custom:
        name, description = "custom", "Custom host implementation"
        schema = ToolSchema(
            "custom",
            description,
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
        output_schema = None
        requires_approval, side_effecting = False, True

        async def run(self, args, ctx):
            with pytest.raises(TypeError):
                args["value"] = 99
            effects.append(args["value"])
            return "ok"

    tools = ToolSet(ToolRegistry.from_tools([Custom()]))
    assert (await tools.execute("custom", {"value": "bad"}, make_test_ctx())).is_error
    assert not effects
    assert not (await tools.execute("custom", {"value": 3}, make_test_ctx())).is_error
    assert effects == [3]


@pytest.mark.asyncio
async def test_cancel_during_permission_prevents_effect_and_finishes_observers():
    from agentkit.tools import PermissionDecision, current_tool_invocation

    token, started = CancellationToken(), asyncio.Event()
    effects, events = [], []

    @tool(side_effecting=True)
    def write() -> str:
        """Write only while this call is authorized."""
        effects.append(True)
        return "ok"

    async def permission(request):
        started.set()
        await asyncio.Event().wait()
        return PermissionDecision.ALLOW

    tools = ToolSet(ToolRegistry.from_tools([write]), permissions=(permission,), observers=(events.append,))
    pending = asyncio.create_task(tools.execute("write", {}, make_test_ctx(cancel=token)))
    await started.wait()
    token.cancel()
    with pytest.raises(Cancelled):
        await asyncio.wait_for(pending, 1)
    assert not effects and events[-1].code == "cancelled"
    assert current_tool_invocation() is None


@pytest.mark.asyncio
async def test_parallel_invocations_do_not_share_identity_or_scope():
    from agentkit.tools import current_tool_invocation

    seen = []
    both = asyncio.Event()

    @tool(side_effecting=False)
    async def read(ctx) -> str:
        """Read from the current host-authorized scope."""
        identity = current_tool_invocation()
        seen.append((ctx.correlation_id, identity))
        if len(seen) == 2:
            both.set()
        await both.wait()
        assert current_tool_invocation() == identity
        return "ok"

    tools = ToolSet(ToolRegistry.from_tools([read]))
    contexts = [SimpleNamespace(correlation_id=name, check_cancelled=lambda: None) for name in ("a", "b")]
    results = await asyncio.gather(*(tools.execute("read", {}, ctx) for ctx in contexts))
    assert all(not result.is_error for result in results)
    assert {identity.call_id for _, identity in seen} == {"a", "b"}
    assert len({identity.host_invocation_id for _, identity in seen}) == 2


@pytest.mark.asyncio
async def test_late_result_after_cancellation_is_not_delivered():
    events = []
    token = CancellationToken()

    @tool(side_effecting=False)
    async def late() -> str:
        """Simulate a completed effect followed by call cancellation."""
        token.cancel()
        return "private late result"

    tools = ToolSet(ToolRegistry.from_tools([late]), observers=(events.append,))
    with pytest.raises(Cancelled):
        await tools.execute("late", {}, make_test_ctx(cancel=token))
    assert events[-1].code == "cancelled"
    assert "private late result" not in repr(events)


@pytest.mark.asyncio
async def test_cli_approval_transport_defers_to_configured_host_handler():
    from agentkit.tools import PermissionDecision

    @tool(side_effecting=True, requires_approval=True)
    def write() -> str:
        """Write only after the shared host approval callback."""
        return "ok"

    tools = ToolSet(ToolRegistry.from_tools([write]), approval=lambda request: PermissionDecision.ALLOW)
    spec = serve(tools, ctx=make_test_ctx())
    try:
        assert "mcp__agentkit__write" in spec.auto_approve
        assert not spec.requires_approval
    finally:
        await spec.stop()


@pytest.mark.asyncio
async def test_permission_failure_diagnostics_do_not_include_private_arguments():
    @tool(side_effecting=True)
    def write(value: str) -> str:
        """Write after the host checks resource permissions."""
        pytest.fail("effect must not execute")

    def broken(request):
        raise RuntimeError("PRIVATE-PERMISSION-CANARY")

    result = await ToolSet(ToolRegistry.from_tools([write]), permissions=(broken,)).execute(
        "write", {"value": "PRIVATE-ARGUMENT-CANARY"}, make_test_ctx()
    )
    assert result.is_error
    assert "PRIVATE" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["exception", "output"])
async def test_tool_failures_do_not_leak_raw_arguments_or_results(defect):
    from pydantic import BaseModel

    class Output(BaseModel):
        count: int

    @tool(side_effecting=False, output_schema=Output)
    def read() -> dict:
        """Read a record whose errors remain private to the host."""
        if defect == "exception":
            raise RuntimeError("PRIVATE-EXCEPTION-CANARY")
        return {"count": "PRIVATE-RESULT-CANARY"}

    result = await ToolSet(ToolRegistry.from_tools([read])).execute("read", {}, make_test_ctx())
    assert result.is_error and "PRIVATE" not in repr(result)


@pytest.mark.asyncio
async def test_explicit_safe_tool_failure_remains_actionable_without_exposing_its_cause():
    from agentkit.tools import ToolExecutionError

    @tool(side_effecting=False)
    def recover() -> str:
        """Report a safe recovery instruction supplied by the host."""
        try:
            raise RuntimeError("PRIVATE-CAUSE-CANARY")
        except RuntimeError as exc:
            raise ToolExecutionError("Retry with the same idempotency key", code="response_lost") from exc

    result = await ToolSet(ToolRegistry.from_tools([recover])).execute("recover", {}, make_test_ctx())
    assert result.is_error and "same idempotency key" in result.text
    assert result.structured["code"] == "response_lost" and "PRIVATE" not in repr(result)
