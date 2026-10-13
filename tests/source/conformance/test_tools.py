from __future__ import annotations

import asyncio
import json

import pytest

from agentkit.core.execution.concurrency import CancellationToken, Cancelled
from agentkit.core.testing import make_test_ctx
from agentkit.integrations.mcp import serve_registry
from agentkit.tools import ToolRegistry, tool
from agentkit.tools.execution.policy import OverflowPolicy, ToolResult, ToolSet, describe_limits, serve


@tool(side_effecting=False)
def lookup_record(key: str) -> dict:
    """Read the requested record from storage."""
    return {"key": key, "value": 42}


@pytest.mark.asyncio
async def test_deferred_mcp_tools_become_discoverable_and_callable_after_search():
    from mcp.shared.memory import create_connected_server_and_client_session

    ctx = make_test_ctx()
    tools = ToolSet(ToolRegistry.from_tools([lookup_record]), deferred=("lookup_record",))
    spec = serve(tools, ctx=ctx)
    try:
        assert "mcp__agentkit__lookup_record" in spec.auto_approve
        server = spec.build_server()
        assert server.create_initialization_options().capabilities.tools.listChanged
        async with create_connected_server_and_client_session(server) as session:
            assert [t.name for t in (await session.list_tools()).tools] == ["call_loaded_tool", "search_tools"]
            refused = await session.call_tool("lookup_record", {"key": "a"})
            assert refused.isError and "deferred" in refused.content[0].text
            searched = await session.call_tool("search_tools", {"query": "storage"})
            assert not searched.isError
            catalogue = json.loads(searched.content[0].text)
            assert catalogue["loaded"] == ["lookup_record"]
            assert catalogue["tools"][0]["inputSchema"]["properties"]["key"]["type"] == "string"
            assert [t.name for t in (await session.list_tools()).tools] == [
                "call_loaded_tool",
                "lookup_record",
                "search_tools",
            ]
            result = await session.call_tool("lookup_record", {"key": "a"})
            assert result.structuredContent == {"key": "a", "value": 42}
    finally:
        await spec.stop()


@pytest.mark.asyncio
async def test_dynamic_mcp_catalogue_cannot_advertise_an_unregistered_dispatch_target():
    from mcp.shared.memory import create_connected_server_and_client_session

    @tool(side_effecting=False)
    def unexpected() -> str:
        """A tool not declared in the dispatch registry."""
        return "not allowed"

    spec = serve_registry(
        ToolRegistry.from_tools([lookup_record]), name="test", ctx=make_test_ctx(), tools_provider=lambda: [unexpected]
    )
    try:
        async with create_connected_server_and_client_session(spec.build_server()) as session:
            with pytest.raises(Exception, match="unregistered"):
                await session.list_tools()
    finally:
        await spec.stop()


@pytest.mark.asyncio
async def test_tool_result_hook_runs_before_overflow_and_preserves_error_flag(tmp_path):
    names = []

    async def sanitize(name, value):
        names.append(name)
        return ToolResult(({"type": "text", "text": "x" * 1000},), is_error=True)

    tools = ToolSet(ToolRegistry(), overflow=OverflowPolicy(256, tmp_path), result_hook=sanitize)
    result = await tools.handle("external", {"secret": "drop this"})
    assert names == ["external"] and result.overflowed and result.is_error
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored["content"][0]["text"] == "x" * 1000 and "secret" not in stored
    assert len(result.text) < 256


@pytest.mark.asyncio
async def test_approval_required_tool_body_is_never_executed():
    @tool(side_effecting=True, requires_approval=True)
    def deploy() -> str:
        """Write to a production environment."""
        pytest.fail("unapproved body ran")

    tools = ToolSet(ToolRegistry.from_tools([deploy]))
    result = await tools.execute("deploy", {}, make_test_ctx())
    assert result.is_error and "host approval" in result.text


@pytest.mark.asyncio
async def test_tool_timeout_is_model_visible_and_cancels_body():
    stopped = asyncio.Event()

    @tool(side_effecting=False)
    async def stalled() -> str:
        """A tool that waits indefinitely."""
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return "unreachable"

    tools = ToolSet(ToolRegistry.from_tools([stalled]), timeout_s=0.01)
    result = await tools.execute("stalled", {}, make_test_ctx())
    assert result.is_error and "TimeoutError" in result.text
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_per_tool_deadline_overrides_toolset_default():
    @tool(side_effecting=False, timeout_s=0.2)
    async def slow_lookup() -> str:
        """Wait briefly for a remote read that has its own service deadline."""
        await asyncio.sleep(0.04)
        return "complete"

    result = await ToolSet(ToolRegistry.from_tools([slow_lookup]), timeout_s=0.01).execute(
        "slow_lookup", {}, make_test_ctx()
    )
    assert not result.is_error
    assert result.text == "complete"


@pytest.mark.asyncio
async def test_per_tool_deadline_expires_even_when_toolset_default_is_unbounded():
    @tool(side_effecting=False, timeout_s=0.01)
    async def stalled_lookup() -> str:
        """Wait forever to verify the declared per-tool deadline is enforced."""
        await asyncio.Event().wait()
        return "unreachable"

    result = await ToolSet(ToolRegistry.from_tools([stalled_lookup]), timeout_s=None).execute(
        "stalled_lookup", {}, make_test_ctx()
    )
    assert result.is_error and "TimeoutError" in result.text


@pytest.mark.asyncio
async def test_cancellation_propagates_instead_of_becoming_recoverable_tool_error():
    ctx = make_test_ctx(cancel=CancellationToken())
    ctx.cancel.cancel()
    tools = ToolSet(ToolRegistry.from_tools([lookup_record]))
    with pytest.raises(Cancelled):
        await tools.execute("lookup_record", {"key": "a"}, ctx)


@pytest.mark.asyncio
async def test_asyncio_cancellation_propagates_through_tool_execution():
    entered = asyncio.Event()

    @tool(side_effecting=False)
    async def stalled() -> str:
        """A cancellable tool that waits indefinitely."""
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    task = asyncio.create_task(ToolSet(ToolRegistry.from_tools([stalled])).execute("stalled", {}, make_test_ctx()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_output_validation_failure_is_visible_to_model():
    @tool(
        side_effecting=False,
        output_schema={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    )
    def invalid() -> dict:
        """A tool with a mismatched return value."""
        return {"n": "not an integer"}

    tools = ToolSet(ToolRegistry.from_tools([invalid]))
    assert (await tools.execute("invalid", {}, make_test_ctx())).is_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,text,error",
    [
        ("plain", "plain", False),
        (42, "42", False),
        ({"a": 1}, '{"a": 1}', False),
        ({"content": [{"type": "text", "text": "remote failed"}], "isError": True}, "remote failed", True),
    ],
)
async def test_results_share_normalized_text_and_error_semantics(value, text, error):
    result = await ToolSet(ToolRegistry()).handle("name", value)
    assert result.text == text and result.is_error is error


def test_descriptive_constraints_recurse_without_mutating_original_or_examples():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "maxItems": 2, "items": {"type": "integer", "minimum": 1}}},
        "tests.examples": [{"minimum": "literal"}],
        "anyOf": [{"type": "string", "pattern": "^[a-z]+$"}],
    }
    result = describe_limits(schema)
    assert "At most this many items: 2" in result["properties"]["items"]["description"]
    assert "At least: 1" in result["properties"]["items"]["items"]["description"]
    assert "Must match pattern" in result["anyOf"][0]["description"]
    assert "description" not in schema["properties"]["items"]
    assert result["tests.examples"] == schema["tests.examples"]


@pytest.mark.asyncio
async def test_loading_deferred_tool_returns_updated_schema_and_no_match_is_empty():
    tools = ToolSet(ToolRegistry.from_tools([lookup_record]), deferred=("lookup_record",))
    empty = await tools.execute("search_tools", {"query": "missing"}, make_test_ctx())
    assert empty.structured["loaded"] == [] and empty.structured["tools"] == []
    found = await tools.execute("search_tools", {"query": "record"}, make_test_ctx())
    assert found.structured["tools"][0]["name"] == "lookup_record"
    assert (await tools.execute("lookup_record", {"key": "a"}, make_test_ctx())).structured["value"] == 42


@pytest.mark.asyncio
async def test_stable_dispatcher_executes_discovered_tool_without_catalogue_refresh():
    from mcp.shared.memory import create_connected_server_and_client_session

    tools = ToolSet(ToolRegistry.from_tools([lookup_record]), deferred=("lookup_record",))
    spec = serve(tools, ctx=make_test_ctx())
    try:
        async with create_connected_server_and_client_session(spec.build_server()) as session:
            initial = [schema.name for schema in (await session.list_tools()).tools]
            assert "call_loaded_tool" in initial and "lookup_record" not in initial
            search = await session.call_tool("search_tools", {"query": "storage"})
            guidance = json.loads(search.content[0].text)["invoke"]
            assert guidance["tool"] == "call_loaded_tool"
            # No tools/list refresh is needed: the initial catalogue remains sufficient.
            result = await session.call_tool(
                "call_loaded_tool", {"name": "lookup_record", "arguments_json": '{"key":"remember"}'}
            )
            assert result.structuredContent == {"key": "remember", "value": 42}
    finally:
        await spec.stop()


@pytest.mark.asyncio
async def test_stable_dispatcher_cannot_execute_unloaded_or_eager_tools():
    @tool(side_effecting=False)
    def eager_record() -> str:
        """Read an eager record independently of deferred discovery."""
        pytest.fail("unauthorized dispatcher reached tool")

    tools = ToolSet(ToolRegistry.from_tools([lookup_record, eager_record]), deferred=("lookup_record",))
    for name in ("lookup_record", "eager_record", "unknown", "search_tools", "call_loaded_tool"):
        result = await tools.execute("call_loaded_tool", {"name": name, "arguments_json": "{}"}, make_test_ctx())
        assert result.is_error and "discovered" in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded", ["not json", "[]", '"text"', "null", '{"key": NaN}', '{"key": Infinity}', 123, "x" * 50_001]
)
async def test_stable_dispatcher_rejects_invalid_arguments_without_execution(encoded):
    tools = ToolSet(ToolRegistry.from_tools([lookup_record]), deferred=("lookup_record",))
    await tools.execute("search_tools", {"query": "record"}, make_test_ctx())
    result = await tools.execute(
        "call_loaded_tool", {"name": "lookup_record", "arguments_json": encoded}, make_test_ctx()
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_dispatcher_preserves_loaded_error_image_and_overflow_semantics(tmp_path):
    @tool(side_effecting=False)
    def loaded_error() -> str:
        """Read a loaded tool that reports an execution failure."""
        raise ValueError("recover with another input")

    @tool(side_effecting=False)
    def loaded_image() -> dict:
        """Read a loaded image result for visual analysis."""
        return {"content": [{"type": "image", "data": "eA==", "mimeType": "image/png"}]}

    @tool(side_effecting=False)
    def loaded_large() -> str:
        """Read a loaded result larger than its context limit."""
        return "x" * 4000

    names = ("loaded_error", "loaded_image", "loaded_large")
    tools = ToolSet(
        ToolRegistry.from_tools([loaded_error, loaded_image, loaded_large]),
        OverflowPolicy(2000, tmp_path),
        deferred=names,
    )
    await tools.execute("search_tools", {"query": "loaded"}, make_test_ctx())
    results = [
        await tools.execute("call_loaded_tool", {"name": name, "arguments_json": "{}"}, make_test_ctx())
        for name in names
    ]
    assert results[0].is_error and "recover" in results[0].text
    assert results[1].content[0]["type"] == "image"
    assert results[2].overflowed and len(results[2].text) < 2000


@pytest.mark.asyncio
async def test_serve_observers_are_per_invocation_and_preserve_arguments_results():
    from mcp.shared.memory import create_connected_server_and_client_session

    tools = ToolSet(ToolRegistry.from_tools([lookup_record]))
    observations, second_observations = [], []

    async def observe(name, arguments, result):
        observations.append((name, arguments, result))

    first = serve(tools, ctx=make_test_ctx(), name="first", execution_observer=observe)
    second = serve(
        tools, ctx=make_test_ctx(), name="second", execution_observer=lambda *args: second_observations.append(args)
    )
    try:
        async with create_connected_server_and_client_session(first.build_server()) as session:
            await session.call_tool("lookup_record", {"key": "one"})
        async with create_connected_server_and_client_session(second.build_server()) as session:
            await session.call_tool("lookup_record", {"key": "two"})
        assert len(observations) == len(second_observations) == 1
        name, arguments, result = observations[0]
        assert name == "lookup_record" and arguments == {"key": "one"}
        assert result.structured == {"key": "one", "value": 42}
        with pytest.raises(TypeError):
            arguments["key"] = "mutated"
        assert second_observations[0][1] == {"key": "two"}
    finally:
        await first.stop()
        await second.stop()


@pytest.mark.asyncio
async def test_serve_observer_captures_dispatcher_error_with_original_served_arguments():
    from mcp.shared.memory import create_connected_server_and_client_session

    tools = ToolSet(ToolRegistry.from_tools([lookup_record]), deferred=("lookup_record",))
    observed = []
    spec = serve(tools, ctx=make_test_ctx(), execution_observer=lambda *args: observed.append(args))
    try:
        async with create_connected_server_and_client_session(spec.build_server()) as session:
            args = {"name": "lookup_record", "arguments_json": "{}"}
            result = await session.call_tool("call_loaded_tool", args)
            assert result.isError
        assert observed[0][0] == "call_loaded_tool" and observed[0][1] == args
        assert observed[0][2].is_error
    finally:
        await spec.stop()


@pytest.mark.parametrize("name", ["search_tools", "call_loaded_tool"])
def test_deferred_discovery_names_cannot_shadow_registry_tools(name):
    from dataclasses import replace

    registry = ToolRegistry.from_tools([replace(lookup_record, name=name)])
    with pytest.raises(ValueError, match="reserved"):
        ToolSet(registry, deferred=(name,))


@pytest.mark.asyncio
async def test_serve_observer_keeps_image_content_and_overflow_flags(tmp_path):
    from mcp.shared.memory import create_connected_server_and_client_session

    @tool(side_effecting=False)
    def picture() -> dict:
        """Read an image result for visual model analysis."""
        return {"content": [{"type": "image", "data": "eA==", "mimeType": "image/png"}]}

    @tool(side_effecting=False)
    def large() -> str:
        """Read a large result that requires context overflow handling."""
        return "x" * 4000

    tools = ToolSet(ToolRegistry.from_tools([picture, large]), overflow=OverflowPolicy(256, tmp_path))
    observations = []
    spec = serve(tools, ctx=make_test_ctx(), execution_observer=lambda *args: observations.append(args))
    try:
        async with create_connected_server_and_client_session(spec.build_server()) as session:
            image = await session.call_tool("picture", {})
            overflow = await session.call_tool("large", {})
            assert image.content[0].type == "image" and not overflow.isError
        assert observations[0][2].content[0]["type"] == "image"
        assert observations[1][2].overflowed and "saved to" in observations[1][2].text
    finally:
        await spec.stop()


@pytest.mark.asyncio
async def test_cli_tool_approval_metadata_preserves_the_inner_host_gate():
    from mcp.shared.memory import create_connected_server_and_client_session

    called = []

    @tool(side_effecting=True, requires_approval=True)
    def privileged_write(value: str) -> str:
        """A write that still needs explicit host approval."""
        called.append(value)
        return "committed"

    tools = ToolSet(ToolRegistry.from_tools([privileged_write]), deferred=("privileged_write",))
    spec = serve(tools, ctx=make_test_ctx())
    try:
        assert "mcp__agentkit__privileged_write" in spec.requires_approval
        assert "mcp__agentkit__privileged_write" not in spec.auto_approve
        assert "mcp__agentkit__call_loaded_tool" in spec.auto_approve
        table = spec.codex_kwargs()["mcp_servers"]["agentkit"]
        assert table["tools"]["privileged_write"]["approval_mode"] == "prompt"
        assert table["tools"]["call_loaded_tool"]["approval_mode"] == "approve"
        async with create_connected_server_and_client_session(spec.build_server()) as session:
            await session.call_tool("search_tools", {"query": "privileged write"})
            result = await session.call_tool(
                "call_loaded_tool",
                {
                    "name": "privileged_write",
                    "arguments_json": '{"value":"denied"}',
                },
            )
            assert result.isError and "host approval" in result.content[0].text
            assert not called
    finally:
        await spec.stop()


@pytest.mark.asyncio
async def test_run_token_cancels_and_joins_an_inflight_tool_body():
    entered, cleaned = asyncio.Event(), asyncio.Event()
    token = CancellationToken()

    @tool(side_effecting=False)
    async def sample() -> str:
        """Collect diagnostic samples until the run is stopped."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            cleaned.set()
        return "unreachable"

    tools = ToolSet(ToolRegistry.from_tools([sample]), timeout_s=None)
    invocation = asyncio.create_task(tools.execute("sample", {}, make_test_ctx(cancel=token)))
    await asyncio.wait_for(entered.wait(), 1)
    token.cancel()
    with pytest.raises(Cancelled):
        await asyncio.wait_for(invocation, 1)
    assert cleaned.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_server", ["token", "server", "both", "during_cleanup"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_http_hosted_tool_cleanup_finishes_before_run_or_server_returns(stop_server, wrapped):
    import httpx

    entered, cleaning, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    token = CancellationToken()

    @tool(side_effecting=False)
    async def sample() -> str:
        """Collect diagnostic samples until the run is stopped."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await asyncio.sleep(0.03)
            cleaned.set()
        return "unreachable"

    registry = ToolRegistry.from_tools([sample])
    spec = (
        serve(ToolSet(registry, timeout_s=None), ctx=make_test_ctx(cancel=token), name="diagnostics")
        if wrapped
        else serve_registry(
            registry, name="diagnostics", ctx=make_test_ctx(cancel=token), json_response=True, timeout_s=None
        )
    )
    try:
        await spec.start()
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            request = asyncio.create_task(
                client.post(
                    spec.url,
                    headers={
                        **spec.auth_headers,
                        "Accept": "application/json, text/event-stream",
                        "MCP-Protocol-Version": "2025-11-25",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "sample", "arguments": {}},
                    },
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            if stop_server in ("token", "both", "during_cleanup"):
                token.cancel()
            if stop_server == "during_cleanup":
                await asyncio.wait_for(cleaning.wait(), 1)
            if stop_server in ("server", "both", "during_cleanup"):
                await asyncio.wait_for(spec.stop(), 2)
            response = await asyncio.wait_for(request, 2)
            assert response.status_code == 200 and response.json()["result"]["isError"]
            assert cleaned.is_set()
            assert not spec._invocations
    finally:
        await spec.stop()
