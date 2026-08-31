"""``serve_registry`` — a ``ToolRegistry`` handed to the Claude CLI over MCP.

``ClaudeCliCognition`` delegates the whole loop to the CLI, and the only way to
give that loop a tool agentkit owns is ``--mcp-config``. Everything needed to
DESCRIBE a tool was already here and already right — ``FunctionTool`` derives a
``ToolSchema`` from a signature and docstring, ``ToolRegistry`` holds them,
``ToolArgumentError`` refuses a bad call. Only the transport was missing.

Most of these tests drive a REAL MCP round trip over the in-memory transport the
``mcp`` package ships for exactly this (``create_connected_server_and_client_session``).
That is deliberate and it is what makes the "and the session survives" half of
each error test mean anything: the assertion is that a second call on the SAME
session succeeds after the first one failed. A test that called the handler
directly could not tell a tool error from a transport error, and that
distinction is the entire point of ``ToolArgumentError``.

The socket-level and subprocess-level cases are separate and few, because a
listener can only fail for its own reasons and those are the transport's
contract, not this module's.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp", reason="needs the `mcp` extra")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from agentkit.agents.control.safety import RunPolicy  # noqa: E402
from agentkit.integrations.mcp import McpServerSpec, serve_registry  # noqa: E402
from agentkit.integrations.mcp._transport import LoopbackMcpTransport  # noqa: E402
from agentkit.kernel.concurrency import CancellationToken  # noqa: E402
from agentkit.kernel.types import ToolSchema  # noqa: E402
from agentkit.testing import make_test_ctx  # noqa: E402
from agentkit.tools import FunctionTool, ToolRegistry, tool  # noqa: E402

# ── fixtures the whole file shares ─────────────────────────────────────────


@tool(side_effecting=False)
def run_check(name: str, strict: bool = False) -> str:
    """Run the named check and report whether it passed. Used by the tests as
    the ordinary, well-behaved tool."""
    return f"{name}:{'strict' if strict else 'lax'}:ok"


@tool(side_effecting=True, requires_approval=True, caps=("egress",))
def deploy(target: str) -> str:
    """Deploy the current build to the named target environment. Declares
    approval and an egress capability so both can be checked on the wire."""
    return f"deployed to {target}"


def _registry(*tools: Any) -> ToolRegistry:
    return ToolRegistry.from_tools(tools or (run_check,))


def _spec(*tools: Any, **kw: Any) -> McpServerSpec:
    kw.setdefault("ctx", make_test_ctx())
    kw.setdefault("name", "engine")
    return serve_registry(_registry(*tools), **kw)


@asynccontextmanager
async def _session(spec: McpServerSpec) -> AsyncIterator[Any]:
    """A client wired to the spec's server over the in-memory transport.

    An ``asynccontextmanager`` rather than a fixture: anyio's task-scoped
    cancel scopes require enter and exit in the SAME task, and pytest-asyncio
    can hand a fixture back to the test from a different one — the same reason
    ``conftest.py`` next door is shaped this way.
    """
    async with create_connected_server_and_client_session(spec.build_server()) as s:
        yield s


def _text(result: Any) -> str:
    return "".join(b.text for b in result.content if getattr(b, "text", None) is not None)


# ── 1. the config the CLI is actually handed ───────────────────────────────


def test_the_config_is_written_to_a_real_path_the_cli_can_read() -> None:
    """``--mcp-config`` takes a file or inline JSON, and a caller should be
    assembling neither by hand. The whole gap this closes is that nothing in
    agentkit could produce this document."""
    spec = _spec()

    assert spec.config_path.exists()
    config = json.loads(spec.config_path.read_text())
    assert config == {
        "mcpServers": {"engine": {"type": "http", "url": spec.url}}
    }
    assert spec.url is not None and spec.url.startswith("http://127.0.0.1:")


def test_the_tool_names_are_the_ones_the_cli_will_see() -> None:
    """``mcp__<server>__<tool>`` is what the model emits and what
    ``--allowed-tools`` matches, so it is wire contract. A caller that had to
    reconstruct this string would get it wrong the first time."""
    spec = _spec(run_check, deploy)

    assert spec.tool_names == ("mcp__engine__deploy", "mcp__engine__run_check")
    assert spec.mcp_names["run_check"] == "mcp__engine__run_check"


def test_cli_kwargs_pins_the_server_set_and_leaves_the_toolbox_alone() -> None:
    """``strict_mcp_config`` is included because a service wiring its own tools
    does not also want whatever ``~/.claude`` or the working directory happen
    to define. ``tools`` is NOT included by default: that flag disables the
    CLI's own Read/Grep/Bash, which is a decision about what the session can do
    at all rather than about MCP wiring."""
    spec = _spec()
    kwargs = spec.cli_kwargs()

    assert kwargs["mcp_config"] == (str(spec.config_path),)
    assert kwargs["strict_mcp_config"] is True
    assert "tools" not in kwargs
    assert spec.cli_kwargs(builtin_tools=False)["tools"] == ("",)


@pytest.mark.asyncio
async def test_strict_mcp_config_leaves_exactly_these_tools_and_no_others() -> None:
    """The spec's own requirement. The server advertises the registry and
    nothing else — no health tool, no introspection tool, nothing the CLI could
    call that the caller did not register."""
    spec = _spec(run_check, deploy)

    async with _session(spec) as s:
        listed = sorted(t.name for t in (await s.list_tools()).tools)

    assert listed == ["deploy", "run_check"]


# ── 2. the schema is NOT translated ────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_advertised_schema_is_tool_schema_parameters_byte_for_byte() -> None:
    """Both sides are JSON Schema. A translation step would be a second
    description of one thing and it would drift — the model would be shown a
    schema the tool does not actually validate against, and the mismatch shows
    up as a rejected call the model cannot diagnose.

    Byte equality rather than ``==``: two dicts can compare equal with
    different key ORDER, and the bytes are what the model is prompted with.
    """
    spec = _spec()
    assert isinstance(run_check.schema, ToolSchema)
    expected = json.dumps(dict(run_check.schema.parameters))

    async with _session(spec) as s:
        advertised = (await s.list_tools()).tools[0]

    assert json.dumps(advertised.inputSchema) == expected


@pytest.mark.asyncio
async def test_a_tool_with_no_schema_is_not_advertised() -> None:
    """``Tool.schema`` may be ``None`` — the Protocol calls that a
    loop-invisible tool. MCP has no way to express "callable but undescribed"
    (``inputSchema`` is required), so the honest translation is to leave it
    out rather than invent an empty schema the model would then call blind."""
    hidden = FunctionTool(
        name="hidden", fn=lambda args, ctx: "x", description="no schema", side_effecting=False
    )
    spec = _spec(run_check, hidden)

    async with _session(spec) as s:
        listed = [t.name for t in (await s.list_tools()).tools]

    assert listed == ["run_check"]
    assert "hidden" not in spec.mcp_names


# ── 3. a bad call is a TOOL error, and the session lives ───────────────────


@pytest.mark.asyncio
async def test_an_unexpected_argument_is_a_tool_error_the_model_can_read() -> None:
    """THE distinction this module exists to preserve. ``ToolArgumentError``
    is a bad call going IN, authored by the model, and the model is the only
    party that can fix it — so it has to come back as ``isError`` on the CALL,
    which is reflected into the transcript. A transport error instead kills the
    session, and the model never learns what it got wrong."""
    spec = _spec()

    async with _session(spec) as s:
        bad = await s.call_tool("run_check", {"name": "lint", "verbose": True})
        good = await s.call_tool("run_check", {"name": "lint"})

    assert bad.isError is True
    assert "verbose" in _text(bad) and "run_check" in _text(bad)
    assert good.isError is False, "the session did not survive a bad call"
    assert _text(good) == "lint:lax:ok"


@pytest.mark.asyncio
async def test_a_missing_required_argument_is_a_tool_error_the_model_can_read() -> None:
    """Same contract, other half. The message names the missing argument AND
    the accepted set, which is what makes the repair turn a repair rather than
    a guess."""
    spec = _spec()

    async with _session(spec) as s:
        bad = await s.call_tool("run_check", {})
        good = await s.call_tool("run_check", {"name": "types"})

    assert bad.isError is True
    assert "name" in _text(bad)
    assert good.isError is False and _text(good) == "types:lax:ok"


@pytest.mark.asyncio
async def test_an_ordinary_exception_fails_that_call_and_not_the_session() -> None:
    """A tool that raises is a failed CALL. Letting it out of the handler would
    end the MCP session and take every other tool with it — one broken tool
    would disable the whole registry mid-run."""

    @tool(side_effecting=False, name="boom")
    def boom(x: int = 0) -> str:
        """Raise on purpose so the handler's ordinary-exception path is
        exercised end to end over a real session."""
        raise RuntimeError("the database is on fire")

    spec = _spec(run_check, boom)

    async with _session(spec) as s:
        bad = await s.call_tool("boom", {})
        good = await s.call_tool("run_check", {"name": "after"})

    assert bad.isError is True
    assert "RuntimeError" in _text(bad) and "on fire" in _text(bad)
    assert good.isError is False, "one raising tool took the whole registry down"


@pytest.mark.asyncio
async def test_calling_a_tool_that_is_not_registered_is_a_tool_error() -> None:
    """The model can hallucinate a name. Refusing it as a call error keeps the
    session alive so the next call — the corrected one — can land."""
    spec = _spec()

    async with _session(spec) as s:
        bad = await s.call_tool("no_such_tool", {})
        good = await s.call_tool("run_check", {"name": "still-here"})

    assert bad.isError is True and "no_such_tool" in _text(bad)
    assert good.isError is False


# ── 4. the safety declarations travel ──────────────────────────────────────


@pytest.mark.asyncio
async def test_requires_approval_reaches_the_wire_and_the_auto_approve_list() -> None:
    """A tool declaring ``requires_approval`` must still pause once it moves
    behind MCP. Two halves: the annotation tells any client this is not a
    read, and ``auto_approve`` — the list a caller splats into
    ``allowed_tools`` — leaves it out, which is what actually makes the CLI
    prompt."""
    spec = _spec(run_check, deploy)

    async with _session(spec) as s:
        by_name = {t.name: t for t in (await s.list_tools()).tools}

    assert by_name["deploy"].annotations.readOnlyHint is False
    assert by_name["run_check"].annotations.readOnlyHint is True
    assert spec.requires_approval == ("mcp__engine__deploy",)
    assert spec.auto_approve == ("mcp__engine__run_check",)


@pytest.mark.asyncio
async def test_a_read_only_tool_that_needs_approval_does_not_look_free() -> None:
    """``readOnlyHint`` is precisely the hint a client consults to decide it may
    run something WITHOUT asking. So it cannot be a straight copy of
    ``side_effecting``: a read-only tool that nonetheless declares
    ``requires_approval`` — reading a customer record, say — would then
    advertise itself as free, and the declaration would stop meaning anything
    the moment the tool moved behind MCP. That is the exact case where an
    approval requirement matters most and is least visible."""

    @tool(side_effecting=False, requires_approval=True, name="read_record")
    def read_record(customer: str) -> str:
        """Read one customer record — a pure read that a human must still sign
        off, which is the combination the annotation has to survive."""
        return customer

    spec = _spec(read_record)

    async with _session(spec) as s:
        listed = (await s.list_tools()).tools[0]

    assert listed.annotations.readOnlyHint is False
    assert spec.auto_approve == ()
    assert spec.requires_approval == ("mcp__engine__read_record",)


@pytest.mark.asyncio
async def test_side_effecting_and_caps_travel_so_the_rule_of_two_still_applies() -> None:
    """``RunPolicy`` reads ``tool.caps``. If the tags stopped at the MCP
    boundary the lethal-trifecta check would silently stop applying exactly
    when tools are furthest from the caller — which is when it matters most.
    The tags ride in ``_meta`` (annotations are hints and clients are told not
    to trust them) and are re-exposed on the spec so the check can still run."""

    @tool(side_effecting=False, caps=("private_data",), name="read_secrets")
    def read_secrets(key: str) -> str:
        """Read a value out of the private store, tagged private_data so the
        trifecta check has something to see."""
        return key

    @tool(side_effecting=False, caps=("untrusted_content",), name="fetch_page")
    def fetch_page(url: str) -> str:
        """Fetch a page of untrusted web content, tagged accordingly."""
        return url

    spec = _spec(read_secrets, fetch_page, deploy)

    async with _session(spec) as s:
        meta = {t.name: t.meta for t in (await s.list_tools()).tools}

    assert meta["deploy"]["agentkit"] == {
        "side_effecting": True,
        "requires_approval": True,
        "caps": ["egress"],
    }
    assert spec.caps == ("egress", "private_data", "untrusted_content")
    verdict = RunPolicy().check(list(_registry(read_secrets, fetch_page, deploy).tools()))
    assert verdict.allowed is False, "the trifecta must still be visible through the spec"


# ── 5. names the CLI can actually address ──────────────────────────────────


@pytest.mark.asyncio
async def test_a_name_that_is_not_a_valid_mcp_identifier_is_sanitised() -> None:
    """Dots, spaces and unicode are not addressable in ``mcp__server__tool``.
    The model only ever sees the sanitised name, so the rename is invisible to
    it — but the CALLER needs the mapping to write ``allowed_tools``, which is
    why it is on the spec rather than swallowed."""

    @tool(side_effecting=False, name="run.check thé")
    def odd(x: int = 1) -> str:
        """A tool whose name contains a dot, a space and a non-ASCII letter,
        none of which survive into an MCP identifier."""
        return str(x)

    spec = _spec(odd)

    assert spec.mcp_names == {"run.check thé": "mcp__engine__run_check_th_"}
    async with _session(spec) as s:
        listed = [t.name for t in (await s.list_tools()).tools]
        called = await s.call_tool("run_check_th_", {"x": 7})

    assert listed == ["run_check_th_"]
    assert called.isError is False and _text(called) == "7"


def test_two_tools_that_collide_after_sanitising_refuse_to_start() -> None:
    """``ToolRegistry`` already refuses a name collision because a silent
    overwrite changes the implementation under an unchanged advertised name.
    Sanitising re-opens exactly that hole one layer down: two names the
    registry considered distinct can arrive at one MCP identifier, and the
    model would call one of them and get the other."""

    @tool(side_effecting=False, name="run.check")
    def dotted(x: int = 1) -> str:
        """One of the two colliding tools — distinguished only by punctuation
        that MCP identifiers cannot carry."""
        return "dotted"

    @tool(side_effecting=False, name="run check")
    def spaced(x: int = 1) -> str:
        """The other colliding tool — same identifier after sanitising, which
        is the whole point of this test."""
        return "spaced"

    with pytest.raises(ValueError, match="run_check"):
        _spec(dotted, spaced)


def test_a_server_name_that_is_not_addressable_is_refused_not_rewritten() -> None:
    """The tool names are rewritten because the model never typed them. The
    SERVER name the caller typed themselves and then hardcodes into
    ``allowed_tools`` strings and log greps — renaming it silently would
    break a string they are holding somewhere else."""
    with pytest.raises(ValueError, match="server name"):
        _spec(name="my engine")


# ── 6. the sizes nobody tests until they break ─────────────────────────────


@pytest.mark.asyncio
async def test_a_registry_with_no_tools_still_produces_a_usable_config() -> None:
    """An empty registry is a legitimate state — a feature flag turned
    everything off. Failing here would turn a configuration choice into a
    crash at CLI startup, and an empty tool list is exactly what it should
    mean."""
    spec = serve_registry(ToolRegistry(), name="engine", ctx=make_test_ctx())

    assert spec.tool_names == ()
    assert spec.config_path.exists()
    async with _session(spec) as s:
        assert (await s.list_tools()).tools == []


@pytest.mark.asyncio
async def test_a_hundred_tools_all_arrive_and_are_all_callable() -> None:
    """List responses are not paginated by this server, so a large registry is
    the case where a chunking bug would hide."""
    registry = ToolRegistry()
    for i in range(100):
        registry.register(
            FunctionTool(
                name=f"t{i:03d}",
                fn=lambda args, ctx, i=i: f"tool {i}",
                description=f"Tool number {i}, one of a hundred registered at once.",
                side_effecting=False,
                schema=ToolSchema(name=f"t{i:03d}", description="", parameters={"type": "object"}),
            )
        )
    spec = serve_registry(registry, name="engine", ctx=make_test_ctx())

    assert len(spec.tool_names) == 100
    async with _session(spec) as s:
        assert len((await s.list_tools()).tools) == 100
        last = await s.call_tool("t099", {})

    assert _text(last) == "tool 99"


# ── 7. whatever the tool returns has to become text ────────────────────────


@pytest.mark.asyncio
async def test_a_string_result_is_not_json_quoted() -> None:
    """The model reads this text. ``json.dumps`` on an already-textual result
    would wrap it in quotes and escape its newlines, which is noise in the
    transcript and changes what a downstream string comparison sees."""
    spec = _spec()

    async with _session(spec) as s:
        out = await s.call_tool("run_check", {"name": "a\nb"})

    assert _text(out) == "a\nb:lax:ok"


@pytest.mark.asyncio
async def test_a_non_json_serialisable_result_still_reaches_the_model() -> None:
    """A tool may legitimately return a live object. Raising here would turn a
    tool that RAN and SUCCEEDED into a failed call, and the side effect would
    already have happened — the worst possible place to fail."""

    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque handle>"

    @tool(side_effecting=False, name="opaque", output_schema=None)
    def opaque() -> Any:
        """Return an object json cannot encode, to prove the renderer degrades
        to a readable string instead of failing the call."""
        return Opaque()

    spec = _spec(opaque)

    async with _session(spec) as s:
        out = await s.call_tool("opaque", {})

    assert out.isError is False
    assert "<Opaque handle>" in _text(out)


@pytest.mark.asyncio
async def test_a_structured_result_arrives_as_json() -> None:
    """The complement: a dict or list is serialised rather than ``repr``'d, so
    the model gets something it can parse instead of Python syntax."""

    @tool(side_effecting=False, name="report")
    def report() -> Any:
        """Return a mapping so the JSON path of the renderer is covered."""
        return {"ok": True, "count": 2}

    spec = _spec(report)

    async with _session(spec) as s:
        out = await s.call_tool("report", {})

    assert json.loads(_text(out)) == {"ok": True, "count": 2}


# ── 8. a tool that never returns ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_hanging_tool_times_out_as_a_tool_error_and_the_session_lives() -> None:
    """Should the server have a timeout? Yes, opt-in. Without one a tool that
    never returns parks the CLI turn forever with no signal anywhere: the CLI
    is waiting on the MCP call, agentkit is waiting on the tool, and nothing
    times out. With one, the model is told the call timed out and can pick
    another route — which is only useful if the session survives, hence the
    second call here."""

    @tool(side_effecting=False, name="hang")
    async def hang() -> str:
        """Sleep far past any test timeout so the server's own deadline is the
        only thing that can end this call."""
        await asyncio.sleep(3600)
        return "never"

    spec = _spec(run_check, hang, timeout_s=0.05)

    async with _session(spec) as s:
        timed_out = await s.call_tool("hang", {})
        after = await s.call_tool("run_check", {"name": "after"})

    assert timed_out.isError is True
    assert "0.05" in _text(timed_out) and "hang" in _text(timed_out)
    assert after.isError is False


@pytest.mark.asyncio
async def test_there_is_no_timeout_unless_the_caller_asks_for_one() -> None:
    """A default deadline would silently kill a legitimately slow tool — a
    long build, a human-in-the-loop approval — and the failure would look like
    a flake. The caller knows which of those they have."""

    @tool(side_effecting=False, name="slow")
    async def slow() -> str:
        """Take long enough that any accidental sub-second default deadline
        would fire, but not long enough to slow the suite."""
        await asyncio.sleep(0.2)
        return "finished"

    spec = _spec(slow)
    assert spec.timeout_s is None

    async with _session(spec) as s:
        out = await s.call_tool("slow", {})

    assert out.isError is False and _text(out) == "finished"


# ── 9. cancellation and concurrency ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cancelled_run_refuses_calls_instead_of_running_the_tool() -> None:
    """The ctx is the run's cancellation seam and the CLI knows nothing about
    it. Once the run is cancelled, every remaining tool call has to be refused
    — reflecting it as an ordinary failure would invite the model to retry work
    somebody deliberately stopped."""
    token = CancellationToken()
    ran: list[str] = []

    @tool(side_effecting=True, name="record")
    def record(x: int = 0) -> str:
        """Record that it ran, so the test can prove a cancelled run never
        reaches the tool body rather than merely discarding its result."""
        ran.append("ran")
        return "ran"

    spec = _spec(record, ctx=make_test_ctx(cancel=token))

    async with _session(spec) as s:
        before = await s.call_tool("record", {})
        token.cancel()
        after = await s.call_tool("record", {})

    assert before.isError is False
    assert after.isError is True and "cancel" in _text(after).lower()
    assert ran == ["ran"], "the tool ran again after the run was cancelled"


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_cross_their_results() -> None:
    """The CLI can have several tool calls in flight at once. The handler holds
    no per-call state on the server object for exactly this reason; this test
    is what stops someone adding a ``self._current_args`` later."""

    @tool(side_effecting=False, name="echo")
    async def echo(value: str) -> str:
        """Yield to the loop mid-call so overlapping calls genuinely interleave
        rather than running to completion one at a time."""
        await asyncio.sleep(0.01)
        return value

    spec = _spec(echo)

    async with _session(spec) as s:
        results = await asyncio.gather(
            *(s.call_tool("echo", {"value": f"v{i}"}) for i in range(20))
        )

    assert [_text(r) for r in results] == [f"v{i}" for i in range(20)]
    assert spec.calls_seen == 20


# ── 10. the listener ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_server_binds_loopback_and_releases_the_port_on_exit() -> None:
    """No authentication, so loopback-only IS the containment — and a leaked
    listener collides with the next agent on this host."""
    spec = _spec()
    async with spec:
        assert spec.url == f"http://127.0.0.1:{spec.port}/mcp"
        assert spec.port > 0

    with socket.socket() as s:  # binding it again proves the listener is gone
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", spec.port))


@pytest.mark.asyncio
async def test_starting_twice_keeps_one_listener_and_stopping_twice_is_safe() -> None:
    """Both are real unwind paths: a caller who calls ``start()`` and then uses
    ``async with`` would otherwise get a second listener nothing ever closes,
    and an error handler that stops a server the ``async with`` already stopped
    should not raise on the way out of a failure."""
    spec = _spec()
    await spec.start()
    port = spec.port
    await spec.start()
    assert spec.port == port

    await spec.stop()
    await spec.stop()


@pytest.mark.asyncio
async def test_stopping_a_server_that_never_started_is_a_no_op() -> None:
    """A caller unwinding a half-built pipeline should not have to remember how
    far it got."""
    await _spec().stop()


def test_a_port_already_in_use_raises_an_os_error_the_caller_can_read() -> None:
    """A port collision surfaces from ``serve_registry`` itself, in the
    caller's own frame, naming the port.

    It has to, and not merely for tidiness: uvicorn's own bind-failure path is
    ``logger.error(...); sys.exit(3)`` raised inside the ``serve()`` task, and
    measured against a held port that ``SystemExit`` propagates through the
    event loop and out of ``asyncio.run`` — a library killing its host process
    over a port collision, while the awaiting caller sees a bare
    ``CancelledError`` with the port mentioned nowhere. Binding at reservation
    time is what turns that back into a catchable ``OSError``.
    """
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]

        with pytest.raises(OSError, match="[Aa]ddress already in use"):
            _spec(port=port)


@pytest.mark.asyncio
async def test_the_config_file_is_cleaned_up_when_the_server_stops() -> None:
    """A stale config file outlives the port it names, and the next reader gets
    a URL pointing at nothing — or, worse, at whatever bound that port next."""
    spec = _spec()
    path = spec.config_path
    async with spec:
        assert path.exists()

    assert not path.exists()


# ── 11. stdio ──────────────────────────────────────────────────────────────


def test_stdio_without_a_command_explains_why_it_cannot_work() -> None:
    """The CLI SPAWNS a stdio server, so the tools would be reconstructed in a
    fresh process — the in-process registry, its closures and its ``ctx`` do
    not survive that. Rather than serve a config the CLI will fail on, say so
    at build time and name the fix."""
    with pytest.raises(ValueError, match="command"):
        _spec(transport="stdio")


def test_the_stdio_config_names_the_command_the_cli_will_spawn() -> None:
    spec = _spec(transport="stdio", command=(sys.executable, "-m", "my.server"))

    config = json.loads(spec.config_path.read_text())
    assert config["mcpServers"]["engine"] == {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "my.server"],
        "env": {},
    }
    assert spec.url is None


@pytest.fixture(scope="module")
def e2e() -> dict[str, Any]:
    """One helper PROCESS, driving both real transports, shared by the tests
    below.

    Its own process rather than inline: spawning and tearing down MCP
    transports leaves anyio memory streams for the garbage collector, and this
    project runs warnings-as-errors, so the finalisation surfaces as an
    unraisable ``ResourceWarning`` collected at pytest's SESSION teardown —
    where a per-test filter cannot reach it and where it fails unrelated tests
    too. Silencing it globally would blind the suite to a class of real leak;
    the subprocess boundary contains it exactly.
    """
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tests.integrations.mcp._serve_registry_e2e"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=pathlib.Path(__file__).resolve().parents[3],
    )
    assert proc.returncode == 0, f"helper failed:\n{proc.stdout}\n{proc.stderr[-2000:]}"
    verdict: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return verdict


@pytest.mark.slow
def test_a_real_stdio_child_serves_the_registry_over_a_spawned_process(
    e2e: dict[str, Any],
) -> None:
    """The only thing that can confirm the stdio path speaks the protocol: a
    real subprocess, spawned exactly the way the CLI would spawn it, driven by
    agentkit's own ``MCPClient``. A framing bug only exists once there is a
    pipe between two processes."""
    assert e2e["tools"] == ["run_check"]
    assert e2e["ok"] == "lint:lax:ok"
    assert e2e["bad_is_error"] is True
    assert "verbose" in e2e["bad_text"]
    assert e2e["survived"] == "after:lax:ok"


@pytest.mark.slow
def test_the_http_listener_serves_the_same_registry_over_a_real_socket(
    e2e: dict[str, Any],
) -> None:
    """The in-process tests drive the MCP ``Server`` through an in-memory
    transport, which pins the protocol but never touches the Starlette mount,
    the session manager or uvicorn — and a wrong mount path or a
    stateful/stateless mismatch shows up only there, as "MCP server failed to
    connect" with nothing naming the cause."""
    assert e2e["http_tools"] == ["run_check"]
    assert e2e["http_ok"] == "http:lax:ok"
    assert e2e["http_bad_is_error"] is True
    assert e2e["http_survived"] == "later:lax:ok"
    assert e2e["http_calls_seen"] == 3


# ── 13. against the real binary ────────────────────────────────────────────
#
# Everything above proves agentkit speaks MCP correctly to agentkit. That is
# necessary and not sufficient: the CLI has its own opinions about
# ``--mcp-config`` — a document shape, a transport name, a startup deadline —
# and each of them was wrong at least once while this was being built.

real_cli = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


@real_cli
@pytest.mark.slow
def test_the_real_cli_accepts_the_config_and_calls_the_served_tool(tmp_path: Any) -> None:
    """The end of the gap this closes: a registry in one process, a config file
    agentkit wrote, and the real binary finding it, connecting, and running the
    tool body back here.

    The marker file is the load-bearing assertion. ``calls_seen`` and the
    model's prose both go up when the tool is called, but only the marker
    separates "the CLI reached OUR server" from "the model answered plausibly
    without one" — and those two produce transcripts that read the same.
    """
    marker = tmp_path / "called.txt"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tests.integrations.mcp._serve_registry_cli_e2e", str(marker)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=pathlib.Path(__file__).resolve().parents[3],
    )
    assert proc.returncode == 0, f"helper failed:\n{proc.stdout}\n{proc.stderr[-2000:]}"
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])

    assert verdict["stop_reason"] == "complete", verdict["evals"]
    # strict_mcp_config: exactly the server we declared, none of the ones the
    # working directory or ~/.claude happen to define.
    assert verdict["mcp_servers"] == [{"name": "engine", "status": "connected"}]
    assert verdict["tool_names"] == ["mcp__engine__run_check"]
    assert verdict["calls_seen"] >= 1, "the CLI never reached the server"
    assert marker.read_text() == "lint", "the tool body did not run in the serving process"


# ── 13. review pass: gaps the first suite left open ────────────────────────
#
# Every test below was written against a mutant the original 35 did not kill.
# Two are regression tests for defects that mutation exposed.


@pytest.mark.asyncio
async def test_a_tool_that_raises_its_own_timeout_error_keeps_its_message() -> None:
    """A ``TimeoutError`` OUT of the tool is not this server's deadline firing.

    The two used to be indistinguishable, because ``asyncio.wait_for`` collapses
    them into one exception and the handler reported every one of them as the
    deadline. With the default ``timeout_s=None`` — no deadline configured at
    all — a tool whose upstream read timed out was reported to the model as
    ``did not return within Nones and was abandoned``: false twice over (nothing
    was abandoned, there was no deadline) and it threw away the one message
    naming what actually timed out. The model cannot repair a call it has been
    lied to about.
    """

    @tool(side_effecting=False, name="call_upstream")
    def call_upstream(url: str) -> str:
        """Read the named upstream URL, and fail the way a real client fails
        when the far end stops answering."""
        raise TimeoutError("upstream read timed out after 3s")

    spec = _spec(call_upstream)
    assert spec.timeout_s is None

    async with _session(spec) as s:
        out = await s.call_tool("call_upstream", {"url": "http://x"})

    body = _text(out)
    assert out.isError is True
    assert "upstream read timed out after 3s" in body, "the tool's own diagnosis was discarded"
    assert "None" not in body, f"reported a deadline that was never configured: {body}"
    assert "abandoned" not in body


@pytest.mark.asyncio
async def test_a_deadline_and_a_tool_raised_timeout_do_not_read_the_same() -> None:
    """The other half of the pair: when a deadline IS set and DOES fire, the
    message still has to say so. Fixing the case above by deleting the deadline
    message would pass that test and lose this one."""

    @tool(side_effecting=False, name="hangs")
    async def hangs() -> str:
        """Sleep past any deadline the caller could reasonably set, so only the
        server's own timeout can end this call."""
        await asyncio.sleep(3600)
        return "never"

    @tool(side_effecting=False, name="raises")
    def raises() -> str:
        """Fail immediately with a TimeoutError of its own, inside a server that
        also has a deadline configured."""
        raise TimeoutError("lock acquisition timed out")

    spec = _spec(hangs, raises, timeout_s=0.05)

    async with _session(spec) as s:
        by_deadline = _text(await s.call_tool("hangs", {}))
        by_tool = _text(await s.call_tool("raises", {}))

    assert "0.05" in by_deadline and "abandoned" in by_deadline
    assert "lock acquisition timed out" in by_tool
    assert "0.05" not in by_tool, f"the tool's own failure was blamed on the deadline: {by_tool}"


@pytest.mark.asyncio
async def test_a_run_cancelled_while_the_tool_runs_reads_as_a_cancellation() -> None:
    """Cancelling mid-call is not an ordinary tool failure.

    ``Cancelled`` subclasses ``RuntimeError``, so without its own clause it
    lands in the generic handler and reaches the model as
    ``tool 'x' failed: Cancelled: ...`` — which reads as a transient error and
    invites a retry of work somebody deliberately stopped. The dedicated clause
    existed but nothing asserted it: degrading it back to the generic message
    left all 35 original tests passing.
    """
    token = CancellationToken()
    ctx = make_test_ctx(cancel=token)

    @tool(side_effecting=True, name="long_write")
    async def long_write() -> str:
        """Start a write, notice the run was cancelled underneath it, and stop
        rather than finish."""
        token.cancel()
        ctx.check_cancelled()
        return "wrote"

    spec = _spec(long_write, ctx=ctx)

    async with _session(spec) as s:
        out = await s.call_tool("long_write", {})

    body = _text(out)
    assert out.isError is True
    assert "cancelled" in body.lower()
    assert "was running" in body, f"did not say the cancellation caught a call in flight: {body}"
    assert "failed:" not in body, f"a deliberate stop was reported as a tool failure: {body}"


@pytest.mark.asyncio
async def test_start_does_not_return_until_the_listener_actually_answers() -> None:
    """``start()`` waiting for uvicorn to finish starting is the entire reason
    the transport was extracted rather than copied.

    ``reserve()`` binds the socket but does not listen on it; uvicorn calls
    ``listen()`` during startup. Returning from ``start()`` before that hands
    the CLI a URL that does not answer, which surfaces thirty seconds later as
    "MCP server failed to connect" with nothing to debug. Nothing asserted it:
    deleting the wait loop left the whole 209-test MCP and Claude-CLI suite
    green. So connect, with no retry and no sleep, the instant ``start()``
    returns.
    """
    spec = _spec()
    async with spec:
        with socket.create_connection((spec.host, spec.port), timeout=5.0) as probe:
            assert probe.getpeername()[1] == spec.port


@pytest.mark.asyncio
async def test_a_listener_that_stops_before_serving_raises_instead_of_spinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of that wait loop.

    A bare ``await self._task`` re-raises a ``serve()`` that FAILED, but a
    ``serve()`` that simply RETURNS leaves ``started`` False forever and the
    loop spins until something else's timeout kills the process, naming the
    wrong thing. The named ``RuntimeError`` is what makes that legible, and no
    test could reach it.
    """
    import uvicorn

    class _ServerThatNeverServes:
        def __init__(self, config: Any) -> None:
            self.started = False
            self.should_exit = False

        async def serve(self, sockets: Any = None) -> None:
            return None

    monkeypatch.setattr(uvicorn, "Server", _ServerThatNeverServes)

    transport = LoopbackMcpTransport()
    try:
        with pytest.raises(RuntimeError, match="stopped before it began serving"):
            await transport.start(object())
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_re_entering_a_stopped_spec_puts_its_config_file_back() -> None:
    """``start()`` and ``stop()`` are both documented as idempotent, so a retry
    loop that re-enters ``async with spec`` looks like it should work — and it
    did, half way. The listener came back on the same port and ``spec.url``
    answered, but ``stop()`` had deleted the config file and nothing rewrote it,
    so ``cli_kwargs()`` named a path that no longer existed: a live server the
    CLI could not be pointed at, reported as success.
    """
    spec = _spec()
    async with spec:
        first = json.loads(spec.config_path.read_text())
    assert not spec.config_path.exists()

    async with spec:
        assert spec.config_path.exists(), "restarted a listener with no config to reach it by"
        assert json.loads(spec.config_path.read_text()) == first
        # and it is that file the CLI would be handed, not some other path
        assert spec.cli_kwargs()["mcp_config"] == (str(spec.config_path),)
        with socket.create_connection((spec.host, spec.port), timeout=5.0):
            pass

    assert not spec.config_path.exists()


def test_the_config_file_is_not_world_readable() -> None:
    """The document names a loopback endpoint with no authentication in front of
    it. Not a security boundary — anything that can scan the port range finds
    the server anyway — but there is no reason to publish the address to every
    other account on a shared build host."""
    spec = _spec()
    mode = spec.config_path.stat().st_mode & 0o777
    assert mode == 0o600, f"config readable beyond its owner: {oct(mode)}"


def test_the_stdio_config_carries_the_env_the_child_is_spawned_with() -> None:
    """The child is a fresh interpreter spawned by the CLI with the CLI's
    environment, so ``env=`` is often the only way it can import the package
    that defines the tools. It was threaded through and never asserted: replacing
    it with a hard-coded ``{}`` left every test passing, and the failure would
    have surfaced as an ImportError inside a process the caller does not own."""
    spec = _spec(
        transport="stdio",
        command=(sys.executable, "-m", "my.server"),
        env={"PYTHONPATH": "/srv/app", "AGENTKIT_ENV": "prod"},
    )

    entry = json.loads(spec.config_path.read_text())["mcpServers"]["engine"]
    assert entry["env"] == {"PYTHONPATH": "/srv/app", "AGENTKIT_ENV": "prod"}
