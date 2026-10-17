"""``serve_registry`` — the registry-as-MCP-server surface, minus the fence.

The credential in front of this server has its own suite next door
(``test_transport_auth.py``): generation, the 401, the 0600 config file, the
``O_NOFOLLOW`` refusal, the stdio ``auth='bearer'`` refusal. Nothing here
repeats any of that. What is left is everything the module's own docstring
calls load-bearing and nothing was holding:

* a bad CALL comes back as an ``isError`` tool result, never as an exception —
  an exception out of the handler is a transport error, which ends the session
  and takes every other tool in the registry with it;
* one tool raising, timing out or being cancelled is one failed call, and the
  three are told apart in the text the model reads, because "abandoned by this
  server's deadline" and "your own upstream read timed out" call for different
  next turns;
* what gets ADVERTISED is the tool's own ``ToolSchema.parameters`` and the
  tool's own declarations — ``readOnlyHint`` in particular must not say "free"
  about a tool that declared ``requires_approval``, or the declaration stops
  meaning anything the moment the tool moves behind MCP;
* two names that sanitise to one MCP identifier are refused rather than
  shadowed, because the model would call one and reach the other.

Driven in process throughout: the handler via ``_dispatch``, the protocol via
the ``mcp`` package's in-memory session pair, the ASGI app via a hand-built
scope. A real uvicorn listener is deliberately NOT started — the SDK leaves
anyio streams for the collector, which this suite's warnings-as-errors setting
turns into a failure in whatever test the GC lands on, and the real-socket
checks live in the ``_*_e2e.py`` helper processes for exactly that reason.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("mcp", reason="needs the `mcp` extra")

from mcp import types as mcp_types
from mcp.shared.memory import create_connected_server_and_client_session

from agentkit.core.contracts.types import ToolSchema
from agentkit.core.execution.concurrency import CancellationToken, Cancelled
from agentkit.core.testing import make_test_ctx
from agentkit.integrations.mcp.serve import (
    META_KEY,
    McpServerSpec,
    serve_registry,
    serve_registry_stdio,
    stdio_command,
)
from agentkit.tools import ToolArgumentError, ToolRegistry, tool
from agentkit.tools.execution.policy import ToolResult

_EMPTY_OBJECT: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


@tool(side_effecting=False)
def run_check(name: str) -> str:
    """Run the named check and report the verdict. The ordinary tool the
    advertisement tests read back off the wire."""
    return f"{name}:ok"


@dataclass
class _Stub:
    """A ``Tool`` that does exactly what a test tells it to.

    A stub rather than ``FunctionTool`` because several of the decisions under
    test are only reachable through shapes ``@tool`` will not build: a name
    that sanitises to nothing, a ``schema is None`` loop-invisible tool, a body
    that raises ``Cancelled``, a body that returns a ``ToolResult`` carrying
    image bytes. Satisfies the ``Tool`` Protocol structurally, which is what
    ``ToolRegistry.register`` checks.
    """

    name: str = "probe"
    description: str = "Probe the thing and report what it found."
    schema: ToolSchema | None = field(
        default_factory=lambda: ToolSchema(name="probe", description="Probe it.", parameters=dict(_EMPTY_OBJECT))
    )
    output_schema: Any = None
    side_effecting: bool = False
    requires_approval: bool = False
    idempotent: bool = False
    caps: tuple[str, ...] = ()
    returns: Any = None
    raises: BaseException | None = None
    sleep_s: float = 0.0
    calls: list[Mapping[str, Any]] = field(default_factory=list)

    async def run(self, args: Mapping[str, Any], ctx: Any) -> Any:
        self.calls.append(dict(args))
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if self.raises is not None:
            raise self.raises
        return self.returns


def _spec(tools: Sequence[Any] | ToolRegistry | None = None, **kw: Any) -> McpServerSpec:
    kw.setdefault("name", "engine")
    kw.setdefault("ctx", make_test_ctx())
    return serve_registry(tools if tools is not None else [run_check], **kw)


def _text(result: mcp_types.CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, mcp_types.TextContent))


# ── what gets advertised ───────────────────────────────────────────────────


def test_a_tool_that_declares_approval_is_never_advertised_as_free_to_run() -> None:
    """``readOnlyHint`` is precisely the hint a client consults to decide it may
    run something without asking. A read-only tool that nonetheless declares
    ``requires_approval`` must not advertise itself as free, or the declaration
    stops meaning anything the moment the tool moves behind MCP — the one
    moment it matters most, because the party deciding is no longer agentkit.
    """
    gated = _Stub(name="read_prod", side_effecting=False, requires_approval=True)
    advertised = _spec([gated])._advertised[0]

    assert advertised.annotations is not None
    assert advertised.annotations.readOnlyHint is False, "approval-gated tool advertised as free"
    assert advertised.annotations.destructiveHint is False, "it really is read-only; both halves travel"


def test_the_declarations_travel_in_meta_as_well_as_in_the_hints() -> None:
    """The MCP spec tells clients not to make decisions on annotation hints
    from an untrusted server, and three booleans are lossy anyway — ``caps``
    has no annotation at all. ``_meta`` is where a cooperating agentkit-side
    consumer (``RunPolicy``'s Rule-of-Two check, an audit trail) reads the
    declarations back instead of inferring them.
    """
    tooled = _Stub(name="deploy", side_effecting=True, requires_approval=True, caps=("net", "fs"))
    advertised = _spec([tooled])._advertised[0]

    assert advertised.meta == {
        META_KEY: {"side_effecting": True, "requires_approval": True, "caps": ["net", "fs"]}
    }


@pytest.mark.asyncio
async def test_the_schema_the_model_is_shown_is_the_tools_own_not_a_re_derivation() -> None:
    """The reason the module uses the lowlevel server rather than ``FastMCP``.
    ``FastMCP`` re-derives ``inputSchema`` from a Python signature via pydantic;
    the schema already exists, and advertising a second description of it means
    the model is eventually shown a schema the tool does not validate against —
    which reads to the model as an unexplained rejection it cannot repair.
    Asserted off the WIRE, because that is the copy the model sees.
    """
    registry = ToolRegistry.from_tools([run_check])
    expected = dict(registry.get("run_check").schema.parameters)  # type: ignore[union-attr]
    spec = _spec(registry)

    async with create_connected_server_and_client_session(spec.build_server()) as session:
        listed = (await session.list_tools()).tools

    assert [t.name for t in listed] == ["run_check"]
    assert listed[0].inputSchema == expected
    assert json.dumps(listed[0].inputSchema, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_a_tool_with_no_schema_is_dropped_rather_than_advertised_blind() -> None:
    """MCP has no way to say "callable but undescribed" — ``inputSchema`` is
    required — so an invented empty schema would advertise something the model
    then calls blind. The ``Tool`` Protocol calls a schema-less tool
    loop-invisible, and that is honoured here by omission.
    """
    spec = _spec([_Stub(name="invisible", schema=None), run_check])

    assert spec.tool_names == ("mcp__engine__run_check",)
    assert [t.name for t in spec._advertised] == ["run_check"]
    assert dict(spec.mcp_names) == {"run_check": "mcp__engine__run_check"}


def test_two_names_that_sanitise_to_one_identifier_are_refused_not_shadowed() -> None:
    """``ToolRegistry`` refuses a duplicate name because a silent overwrite
    changes the implementation behind an unchanged advertised name. Sanitising
    re-opens exactly that hole one layer down: ``run.check`` and ``run check``
    are two distinct registrations that arrive at one MCP identifier, and the
    model would call one and reach the other with no signal anywhere.
    """
    with pytest.raises(ValueError, match="collision") as exc:
        _spec([_Stub(name="run.check"), _Stub(name="run check")])

    message = str(exc.value)
    assert "run.check" in message and "run check" in message, "the error must name both registrations"
    assert "run_check" in message, "and the identifier they collide on"


def test_a_name_with_nothing_addressable_in_it_is_refused_at_wiring_time() -> None:
    """Sanitising cannot invent an identifier out of nothing, and an empty MCP
    name is a tool the CLI addresses as ``mcp__engine__`` — a string that
    collides with the next such tool and names no tool at all. Refused where
    the wiring is rather than serving a nameless entry.
    """
    with pytest.raises(ValueError, match="no MCP-addressable characters"):
        _spec([_Stub(name="")])


def test_a_bare_sequence_of_tools_still_gets_the_collision_guarantee() -> None:
    """A caller wiring a CLI session has a list in hand, so a list is accepted
    rather than forcing ``ToolRegistry.from_tools(...)`` at every call site —
    and ``from_tools`` is exactly what happens to it, so a duplicate name still
    raises instead of one tool silently shadowing the other.
    """
    assert _spec([run_check]).tool_names == ("mcp__engine__run_check",)

    with pytest.raises(ValueError, match="already registered"):
        _spec([_Stub(name="twice"), _Stub(name="twice")])


def test_approval_and_auto_approve_partition_the_catalogue_for_the_cli_flags() -> None:
    """``requires_approval`` has to survive the move behind MCP as a list of
    QUALIFIED names, because that is the spelling the CLI's permission flags
    take. A tool missing from both tuples is a tool the caller cannot address
    in either flag, and the overlap would be a grant and a gate on one name.
    """
    spec = _spec([_Stub(name="read_logs"), _Stub(name="deploy", side_effecting=True, requires_approval=True)])

    assert spec.requires_approval == ("mcp__engine__deploy",)
    assert spec.auto_approve == ("mcp__engine__read_logs",)
    assert set(spec.tool_names) == set(spec.requires_approval) | set(spec.auto_approve)
    assert not set(spec.requires_approval) & set(spec.auto_approve)


def test_the_caps_of_every_served_tool_are_unioned_onto_the_spec() -> None:
    """``caps`` is what ``RunPolicy``'s Rule-of-Two check reads, and it has to
    describe the SERVER: a caller holding the spec cannot see the individual
    tools any more, so a capability declared by one tool and dropped here is a
    capability the policy stops knowing about."""
    spec = _spec([_Stub(name="a", caps=("net",)), _Stub(name="b", caps=("fs", "net"))])

    assert spec.caps == ("fs", "net")


# ── a bad call is a tool error, never a transport error ────────────────────


@pytest.mark.asyncio
async def test_a_bad_call_is_reflected_to_the_model_instead_of_ending_the_session() -> None:
    """THE case the module exists to get right. ``ToolArgumentError`` already
    names the tool, the offending arguments and the accepted set — which is
    what turns the next turn into a repair rather than a guess. Raised out of
    the handler it would instead be a transport error that ends the session and
    takes every other tool in the registry with it, and the model would be told
    the tool system is broken rather than that its call was wrong.
    """
    boom = ToolArgumentError("run_check", unexpected=("verbose",), accepted=("name",))
    spec = _spec([_Stub(name="run_check", raises=boom)])

    result = await spec._dispatch("run_check", {"name": "lint", "verbose": True})

    assert result.isError is True
    assert _text(result) == str(boom)
    assert "verbose" in _text(result) and "name" in _text(result)


@pytest.mark.asyncio
async def test_an_unknown_tool_name_names_the_catalogue_it_is_not_in() -> None:
    """A misaddressed call is still a CALL, so it is answered rather than
    raised — and answered with the available names, because the alternative is
    a model retrying the same wrong spelling with nothing to correct against.
    """
    spec = _spec([_Stub(name="probe")])

    result = await spec._dispatch("prbe", {})

    assert result.isError is True
    assert "'prbe'" in _text(result) and "probe" in _text(result) and "engine" in _text(result)


@pytest.mark.asyncio
async def test_one_tool_raising_is_one_failed_call_and_not_a_dead_registry() -> None:
    """The registry has to survive a tool body blowing up: the next call on the
    same session must still work. An exception escaping the handler would
    disable every OTHER tool too, which is the failure the blanket ``except``
    in ``_invoke`` is there for.
    """
    broken = _Stub(name="broken", raises=RuntimeError("disk on fire"))
    fine = _Stub(name="fine", returns="still here")
    spec = _spec([broken, fine])

    failed = await spec._dispatch("broken", {})
    assert failed.isError is True
    assert "RuntimeError" in _text(failed) and "disk on fire" in _text(failed)

    survived = await spec._dispatch("fine", {})
    assert survived.isError is False and _text(survived) == "still here"
    assert spec.calls_seen == 2, "both calls were counted, including the one that failed"


@pytest.mark.asyncio
async def test_a_call_after_the_run_was_cancelled_never_reaches_the_tool_body() -> None:
    """Once the run is cancelled every remaining call has to be refused BEFORE
    the body runs. The CLI knows nothing about agentkit's cancellation seam and
    will keep issuing calls; a side-effecting tool that fires after somebody
    pressed stop is the failure this check exists to prevent, and "refused" has
    to be observable as the body not having run, not just as error text.
    """
    cancel = CancellationToken()
    tooled = _Stub(name="deploy", side_effecting=True)
    spec = _spec([tooled], ctx=make_test_ctx(cancel=cancel))
    cancel.cancel()

    result = await spec._dispatch("deploy", {})

    assert result.isError is True
    assert "cancelled" in _text(result) and "was not run" in _text(result)
    assert tooled.calls == [], "the body ran for a run that had already been stopped"


@pytest.mark.asyncio
async def test_a_tool_cancelled_mid_run_is_not_reported_as_an_ordinary_failure() -> None:
    """``Cancelled`` is a ``RuntimeError`` subclass, so without its own clause
    it is swallowed by the generic one and reported as a plain tool failure —
    which reads to the model as an invitation to retry work somebody
    deliberately stopped.
    """
    spec = _spec([_Stub(name="build", raises=Cancelled("run cancelled"))])

    result = await spec._dispatch("build", {})

    assert result.isError is True
    assert "the run was cancelled while" in _text(result)
    assert "failed: Cancelled" not in _text(result), "reported as an ordinary failure"


@pytest.mark.asyncio
async def test_this_servers_deadline_says_it_abandoned_the_tool() -> None:
    """With a deadline configured and expired, the model needs to be told the
    tool was abandoned while still running — the work may well have landed, so
    this is not the same advice as "it failed"."""
    spec = _spec([_Stub(name="slow", sleep_s=5)], timeout_s=0.01)

    result = await spec._dispatch("slow", {})

    assert result.isError is True
    assert _text(result) == "tool 'slow' did not return within 0.01s and was abandoned"


@pytest.mark.asyncio
async def test_a_timeouterror_the_tool_raised_itself_keeps_the_tools_own_message() -> None:
    """REGRESSION. ``asyncio.wait_for`` collapses "our deadline expired" and
    "the tool raised TimeoutError" into one indistinguishable exception, and
    the handler then reported every one of them as this server's deadline. With
    ``timeout_s=None`` that produced the literal text "did not return within
    Nones and was abandoned" for a tool that had in fact failed fast with a
    perfectly good message of its own, which was discarded. Both halves are
    pinned: no deadline at all, and a deadline that has not expired.
    """
    raised = TimeoutError("upstream read timed out after 2s")

    unbounded = await _spec([_Stub(name="fetch", raises=raised)])._dispatch("fetch", {})
    assert unbounded.isError is True
    assert _text(unbounded) == "tool 'fetch' failed: TimeoutError: upstream read timed out after 2s"
    assert "Nones" not in _text(unbounded)

    bounded = await _spec([_Stub(name="fetch", raises=raised)], timeout_s=30)._dispatch("fetch", {})
    assert _text(bounded) == "tool 'fetch' failed: TimeoutError: upstream read timed out after 2s"
    assert "did not return within" not in _text(bounded), "blamed a deadline that never fired"


# ── what the model reads back ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_string_result_reaches_the_model_unquoted() -> None:
    """A ``str`` is already the answer. ``json.dumps`` would wrap it in quotes
    and escape its newlines, which is noise in the transcript and changes what
    a downstream string comparison sees."""
    spec = _spec([_Stub(name="probe", returns="line one\nline two")])

    result = await spec._dispatch("probe", {})

    assert _text(result) == "line one\nline two"
    assert result.isError is False


@pytest.mark.asyncio
async def test_a_structured_result_arrives_as_json_rather_than_python_repr() -> None:
    """Anything JSON can encode is encoded, so the model gets something it can
    parse. Python ``repr`` syntax — single quotes, ``True``, ``None`` — is not
    JSON and a model asked to parse it will get it wrong some of the time."""
    spec = _spec([_Stub(name="probe", returns={"ok": True, "items": [1, 2]})])

    assert json.loads(_text(await spec._dispatch("probe", {}))) == {"ok": True, "items": [1, 2]}


@pytest.mark.asyncio
async def test_a_none_result_is_an_empty_block_rather_than_the_word_none() -> None:
    """A tool that returns nothing returned nothing. "null" or "None" in the
    transcript is a value the model will reason about as if the tool had
    produced one."""
    spec = _spec([_Stub(name="probe", returns=None)])

    assert _text(await spec._dispatch("probe", {})) == ""


@pytest.mark.asyncio
async def test_an_unserialisable_result_is_still_a_successful_call() -> None:
    """The worst available outcome. The tool already RAN and its side effect
    has already happened, so turning a successful call into a failed one at the
    serialisation step would have the model retry a write that already landed.
    ``repr`` is a poor answer; a false failure is a much worse one.
    """
    unencodable = object()
    spec = _spec([_Stub(name="probe", returns=unencodable)])

    result = await spec._dispatch("probe", {})

    assert result.isError is False, "a serialisation problem must not fail a completed call"
    assert _text(result) == repr(unencodable)


@pytest.mark.asyncio
async def test_image_bytes_are_base64_encoded_only_at_the_transport_boundary() -> None:
    """``ToolResult`` keeps image bytes in memory; MCP image data is base64
    text on the wire. Encoding anywhere earlier would make every in-process
    consumer carry a transport detail, and not encoding here sends bytes down a
    JSON channel that cannot hold them."""
    payload = b"\x89PNG\r\n\x1a\nnot-really"
    body = ToolResult(({"type": "text", "text": "chart"}, {"type": "image", "data": payload, "mimeType": "image/png"}))
    spec = _spec([_Stub(name="plot", returns=body)])

    result = await spec._dispatch("plot", {})

    assert result.isError is False
    kinds = [block.type for block in result.content]
    assert kinds == ["text", "image"], "the text block beside the image must survive too"
    image = result.content[1]
    assert isinstance(image, mcp_types.ImageContent)
    assert image.mimeType == "image/png"
    import base64

    assert base64.b64decode(image.data) == payload


@pytest.mark.asyncio
async def test_an_image_block_that_is_not_bytes_is_an_error_result_not_a_crash() -> None:
    """A malformed block has to come back as a refusal the model can read. A
    raised ``TypeError`` here would be a transport error, i.e. the whole
    registry going down over one tool's bad return value."""
    body = ToolResult(({"type": "image", "data": "already-a-string", "mimeType": "image/png"},))
    spec = _spec([_Stub(name="plot", returns=body)])

    result = await spec._dispatch("plot", {})

    assert result.isError is True
    assert "invalid image block" in _text(result)


@pytest.mark.asyncio
async def test_an_oversized_image_is_refused_in_words_rather_than_sent() -> None:
    """Over the MCP response limit the send fails somewhere inside the
    transport, which surfaces as a dead session rather than as anything the
    model can act on. Refusing it as a tool error keeps the diagnosis next to
    the call that caused it."""
    body = ToolResult(({"type": "image", "data": b"\0" * (20 * 1024 * 1024 + 1), "mimeType": "image/png"},))
    spec = _spec([_Stub(name="plot", returns=body)])

    result = await spec._dispatch("plot", {})

    assert result.isError is True
    assert "20 MiB" in _text(result)


@pytest.mark.asyncio
async def test_a_block_type_this_transport_cannot_carry_stays_visible_as_text() -> None:
    """Dropping an unsupported block silently is the one unacceptable option:
    the model is then answering about a result it was only shown part of, with
    nothing saying so."""
    body = ToolResult(({"type": "audio", "data": "zzz", "mimeType": "audio/wav"},))
    spec = _spec([_Stub(name="listen", returns=body)])

    rendered = _text(await spec._dispatch("listen", {}))

    assert "audio" in rendered and "audio/wav" in rendered


@pytest.mark.asyncio
async def test_a_result_with_no_blocks_at_all_still_carries_one_text_block() -> None:
    """``content=[]`` is not a valid ``CallToolResult`` for every client, and a
    tool legitimately returning nothing must not be the thing that breaks the
    session."""
    spec = _spec([_Stub(name="quiet", returns=ToolResult(()))])

    result = await spec._dispatch("quiet", {})

    assert len(result.content) == 1
    assert _text(result) == ""


@pytest.mark.asyncio
async def test_a_result_hook_rewrites_the_answer_and_may_be_async() -> None:
    """The hook is the seam a host uses to redact or truncate before anything
    reaches the model, so it has to run on the way OUT — after the tool, before
    the block is built — and it has to be allowed to await, because redaction
    that needs a lookup is the normal case."""
    seen: list[str] = []

    async def hook(name: str, value: Any) -> Any:
        seen.append(name)
        return f"redacted({value})"

    spec = _spec([_Stub(name="probe", returns="secret")], result_hook=hook)

    assert _text(await spec._dispatch("probe", {})) == "redacted(secret)"
    assert seen == ["probe"], "the hook is told which tool produced the value"


@pytest.mark.asyncio
async def test_a_hook_that_returns_a_call_tool_result_is_passed_through_whole() -> None:
    """A host that wants to set ``isError`` itself — or emit blocks this module
    would not build — must be able to. Re-wrapping its result would discard the
    flag and flatten the blocks it chose."""
    answer = mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text="no")], isError=True)
    spec = _spec([_Stub(name="probe", returns="yes")], result_hook=lambda name, value: answer)

    result = await spec._dispatch("probe", {})

    assert result is answer
    assert result.isError is True


# ── shutdown ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_call_arriving_during_shutdown_is_refused_without_running_it() -> None:
    """``stop()`` is not instantaneous and the CLI keeps issuing calls across
    it. Starting a tool body while the server is tearing down is a side effect
    nobody will be around to observe the result of."""
    tooled = _Stub(name="deploy", side_effecting=True)
    spec = _spec([tooled])
    await spec.stop()

    result = await spec._dispatch("deploy", {})

    assert result.isError is True
    assert "stopping" in _text(result)
    assert tooled.calls == []


@pytest.mark.asyncio
async def test_stopping_the_server_turns_an_inflight_call_into_an_error_result() -> None:
    """A call cancelled by shutdown must come back as a result, not as a
    ``CancelledError`` propagating out of the handler — the latter is a
    transport error raised while the transport is already going away, which
    loses the one piece of information worth having: that the tool was running
    when the server stopped.
    """
    spec = _spec([_Stub(name="slow", sleep_s=5)])
    call = asyncio.create_task(spec._dispatch("slow", {}))
    await asyncio.sleep(0.05)
    assert spec._invocations, "the invocation is tracked so shutdown can reach it"

    await spec.stop()
    result = await call

    assert result.isError is True
    assert "stopped while tool was running" in _text(result)


@pytest.mark.asyncio
async def test_a_call_cancelled_by_its_own_caller_stays_a_cancellation() -> None:
    """The mirror of the test above, and the reason the shutdown check is a
    check rather than a blanket ``except``. When the MCP session cancels a
    request — the peer went away, the enclosing task group is unwinding — the
    cancellation has to keep propagating, because converting it into a tidy
    ``isError`` result would leave the session believing a request it abandoned
    was answered.
    """
    spec = _spec([_Stub(name="slow", sleep_s=5)])
    call = asyncio.create_task(spec._dispatch("slow", {}))
    await asyncio.sleep(0.05)

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert not spec._invocations, "the abandoned invocation was left in the tracking set"


@pytest.mark.asyncio
async def test_a_cancelled_stop_still_releases_the_credential_before_it_rethrows() -> None:
    """``stop()`` shields its own cleanup. A caller cancelled mid-shutdown must
    not leave the config file behind: under ``auth='bearer'`` that file is a
    live credential naming a port something else may bind next, and the one
    path it was lost on was the path where something had already gone wrong and
    nobody was coming back to tidy up. The cancellation is still delivered —
    afterwards.
    """
    release = asyncio.Event()
    spec = _spec()
    assert spec._transport is not None

    async def blocked_stop() -> None:
        await release.wait()

    spec._transport.stop = blocked_stop  # type: ignore[method-assign]
    stopping = asyncio.create_task(spec.stop())
    await asyncio.sleep(0.05)

    stopping.cancel()
    await asyncio.sleep(0.05)
    assert not stopping.done(), "cancellation reached the cleanup it was supposed to shield"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stopping, 5)
    assert not spec.config_path.exists(), "a cancelled stop left the bearer token on disk"


# ── the config document and the CLI wiring ─────────────────────────────────


def test_cli_kwargs_shuts_out_ambient_mcp_config_without_touching_builtin_tools() -> None:
    """``strict_mcp_config`` keeps the CLI from also loading whatever MCP
    servers the working directory or the user's home configuration define — the
    difference between "these tools" and "these tools plus a teammate's
    .mcp.json". Disabling Read, Grep and Bash is a SEPARATE statement about
    what the session can do at all, so a flag named after MCP wiring must not
    quietly make it.
    """
    spec = _spec()

    default = spec.cli_kwargs()
    assert default == {"mcp_config": (str(spec.config_path),), "strict_mcp_config": True}
    assert "tools" not in default, "MCP wiring silently disabled the CLI's own tools"

    only_ours = spec.cli_kwargs(builtin_tools=False)
    assert only_ours["tools"] == ("",)
    assert only_ours["strict_mcp_config"] is True


def test_codex_kwargs_is_the_same_server_spelled_for_the_other_cli() -> None:
    """Codex reads MCP servers as ``config.toml`` keys and takes a bearer token
    through an environment variable it names rather than a header, so the
    projection is not a rename. The method exists so a reader who found
    ``cli_kwargs`` finds the counterpart in the same place instead of
    concluding there is none — which is only true if it really delegates.
    """
    spec = _spec()

    kwargs = spec.codex_kwargs(tool_timeout_s=12)
    table = kwargs["mcp_servers"]["engine"]

    assert table["url"] == spec.url
    assert table["tool_timeout_sec"] == 12
    assert table["bearer_token_env_var"] in kwargs["env"], "the token has to reach the child's env"
    assert kwargs["env"][table["bearer_token_env_var"]] == spec.auth_headers["Authorization"].removeprefix("Bearer ")


def test_a_stdio_server_has_no_url_to_hand_anyone() -> None:
    """Returning a plausible-looking URL for a server that has none is a lie a
    caller can paste into a health check, a log line or another config file,
    and it fails later and somewhere else."""
    assert _spec(transport="stdio", command=("python", "-m", "x")).url is None
    assert _spec().url is not None


def test_a_stdio_spec_writes_the_argv_the_cli_will_actually_spawn() -> None:
    """The CLI SPAWNS a stdio server, so this process's registry, closures and
    ctx do not cross the boundary — the document's command is the only thing
    that reconstructs them, and it is read out of the FILE because that is the
    copy the CLI gets."""
    spec = _spec(transport="stdio", command=("py", "-m", "mine"), env={"A": "1"})
    entry = json.loads(spec.config_path.read_text())["mcpServers"]["engine"]

    assert entry == {"type": "stdio", "command": "py", "args": ["-m", "mine"], "env": {"A": "1"}}


def test_stdio_without_a_command_is_refused_because_the_caller_cannot_be_rebuilt() -> None:
    """A stdio spec with no command is a server the CLI cannot start, and the
    failure surfaces as "MCP server failed to connect" naming nothing. The
    refusal names the way out, because "pass command=" is not guessable from
    the symptom."""
    with pytest.raises(ValueError, match="needs command=") as exc:
        _spec(transport="stdio")

    assert "serve_registry_stdio" in str(exc.value), "the error must name what the command should run"


def test_a_server_name_the_cli_cannot_address_is_refused_at_wiring_time() -> None:
    """Tool names are rewritten because the model never typed them. The server
    name the CALLER typed, and then hardcoded into ``allowed_tools`` strings
    and log greps — renaming it silently would break a string they are holding
    somewhere else, so this one raises instead."""
    for bad in ("my server", "engine.one", "", "e" * 65):
        with pytest.raises(ValueError, match="not addressable"):
            _spec(name=bad)


@pytest.mark.asyncio
async def test_re_entering_a_spec_puts_back_the_config_file_stop_deleted() -> None:
    """``start()`` and ``stop()`` are both documented idempotent, so a second
    ``async with spec`` is a supported retry. It used to hand back a spec that
    was serving perfectly well and whose ``cli_kwargs()`` named a path that no
    longer existed — a live server the CLI could not be pointed at, reported as
    success. Checked over stdio, where ``start()`` does nothing else at all.
    """
    spec = _spec(transport="stdio", command=("py", "-m", "mine"))

    async with spec:
        assert spec.config_path.exists()
    assert not spec.config_path.exists(), "a stale config outlives the server it names"

    async with spec:
        assert spec.config_path.exists(), "the retry was pointed at a file that was gone"
    assert not spec.config_path.exists()


def test_the_file_is_removed_even_when_the_directory_belongs_to_the_caller(tmp_path: Any) -> None:
    """The directory is theirs — they may well have chosen a persistent one,
    which is the point of ``config_path=``. The FILE is ours, we wrote it, and
    what we wrote into it is a secret, so it goes and the directory stays."""
    path = tmp_path / "nested" / "engine.mcp.json"
    spec = _spec(transport="stdio", command=("py", "-m", "x"), config_path=path)
    assert path.exists()

    asyncio.run(spec.stop())

    assert not path.exists()
    assert path.parent.is_dir(), "a caller-owned directory must survive its spec"


def test_calls_seen_separates_a_server_never_reached_from_a_model_that_never_called() -> None:
    """A session that produced nothing is two very different bugs — the CLI
    never connected, or it connected and the model did not use the tools — and
    they are indistinguishable from the outside without this counter. Blaming
    the model is the expensive guess."""
    spec = _spec()
    assert spec.calls_seen == 0

    asyncio.run(spec._dispatch("nope", {}))

    assert spec.calls_seen == 1, "a misaddressed call still reached the server"


# ── the dynamic catalogue and the ASGI edges ───────────────────────────────


@pytest.mark.asyncio
async def test_a_catalogue_that_changed_during_a_call_is_announced_to_the_client() -> None:
    """A provider-backed catalogue is only useful if the client is told it
    moved: the CLI caches ``tools/list``, so a tool that appeared during a call
    stays invisible until something invalidates that cache. The notification is
    sent from inside the call that changed it, which is the only moment the
    server knows both the before and the after.
    """
    visible = [_Stub(name="search")]
    hidden = _Stub(name="loaded")
    registry = ToolRegistry.from_tools([visible[0], hidden])

    def provider() -> list[Any]:
        return list(visible)

    def reveal(name: str, value: Any) -> Any:
        if hidden not in visible:
            visible.append(hidden)
        return value

    spec = serve_registry(
        registry, name="engine", ctx=make_test_ctx(), tools_provider=provider, result_hook=reveal
    )
    notified: list[str] = []

    async def record(message: Any) -> None:
        root = getattr(message, "root", None)
        if isinstance(root, mcp_types.ToolListChangedNotification):
            notified.append(root.method)

    try:
        server = spec.build_server()
        assert server.create_initialization_options().capabilities.tools.listChanged, (
            "a server that can change its catalogue must say so at initialisation"
        )
        async with create_connected_server_and_client_session(server, message_handler=record) as session:
            assert [t.name for t in (await session.list_tools()).tools] == ["search"]
            await session.call_tool("search", {})
            await asyncio.sleep(0.05)
            assert [t.name for t in (await session.list_tools()).tools] == ["loaded", "search"]
            assert notified == ["notifications/tools/list_changed"]

            # A call that changes nothing must say nothing. An announcement per
            # call would make the CLI re-list on every single tool invocation,
            # which is the cost the before/after comparison exists to avoid.
            await session.call_tool("search", {})
            await asyncio.sleep(0.05)
            assert notified == ["notifications/tools/list_changed"], "announced a catalogue that did not move"
    finally:
        await spec.stop()


@pytest.mark.asyncio
async def test_a_get_against_a_json_only_server_is_refused_rather_than_opened() -> None:
    """``json_response=True`` says this server answers one request with one
    JSON body. The streamable-HTTP manager would otherwise answer a ``GET`` by
    opening an SSE stream nothing is going to write to, which parks the caller
    forever instead of telling it the method is wrong.
    """
    spec = _spec(json_response=True)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", spec.port),
    }
    await spec._asgi_app()(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 405
    assert (b"allow", b"POST") in [(k.lower(), v) for k, v in start["headers"]]


@pytest.mark.asyncio
async def test_every_other_request_reaches_the_session_manager_untouched() -> None:
    """The 405 above is conditional, and the condition is the point: a blanket
    ``GET`` refusal would break the default server, where the SSE ``GET`` is
    how the stream is opened. The route is also a CLASS rather than a function
    because Starlette calls a function endpoint with one ``Request`` — the
    three-argument ASGI signature would never have been reached, and the server
    would answer every request with a ``TypeError``.
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    reached: list[str] = []

    async def record(self: Any, scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["method"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(_: dict[str, Any]) -> None:
        return None

    def scope(method: str) -> dict[str, Any]:
        return {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": "/mcp",
            "raw_path": b"/mcp",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 0),
        }

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(StreamableHTTPSessionManager, "handle_request", record)
        streaming = _spec()._asgi_app()
        await streaming(scope("POST"), receive, send)
        await streaming(scope("GET"), receive, send)

    assert reached == ["POST", "GET"], "a request the server must answer never reached the MCP app"


@pytest.mark.asyncio
async def test_readiness_refuses_to_probe_a_server_with_no_endpoint() -> None:
    """A stdio server is spawned by the CLI and has no address to probe, so a
    readiness check that silently returned "ready" would be a green light
    nobody earned."""
    spec = _spec(transport="stdio", command=("py", "-m", "x"))

    with pytest.raises(ValueError, match="requires the HTTP transport"):
        await spec.check_ready()


@pytest.mark.asyncio
async def test_readiness_reads_the_catalogue_out_of_an_sse_reply_as_well_as_json() -> None:
    """The streamable-HTTP endpoint answers ``tools/list`` as ``text/event-stream``
    unless ``json_response`` is set, so a probe that only understood
    ``response.json()`` reported every default server as unready. The request
    also has to carry the credential and the protocol version, or the probe
    fails for reasons that have nothing to do with readiness.
    """
    import httpx

    spec = _spec()
    captured: dict[str, Any] = {}
    catalogue = {"tools": [{"name": "run_check"}]}

    class _Response:
        def __init__(self, *, headers: dict[str, str], text: str) -> None:
            self.headers = headers
            self.text = text

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return json.loads(self.text)

    class _Client:
        def __init__(self, **_: Any) -> None:
            self.reply = _Response(
                headers={"content-type": "text/event-stream"},
                text=f"event: message\ndata: {json.dumps({'result': catalogue})}\n\n",
            )

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: Any) -> _Response:
            captured.update(url=url, headers=headers, body=json)
            return self.reply

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "AsyncClient", _Client)
        assert await spec.check_ready() == catalogue

    assert captured["url"] == spec.url
    assert captured["body"]["method"] == "tools/list"
    assert captured["headers"]["Authorization"] == spec.auth_headers["Authorization"]
    assert captured["headers"]["MCP-Protocol-Version"]
    assert "text/event-stream" in captured["headers"]["Accept"]


@pytest.mark.asyncio
async def test_a_protocol_error_in_the_readiness_reply_is_not_reported_as_ready() -> None:
    """A 200 carrying a JSON-RPC ``error`` is a server that answered and
    refused. Treating the empty ``result`` as an empty catalogue would start a
    model against a server whose tools it will never see."""
    import httpx

    spec = _spec()

    class _Response:
        headers = {"content-type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"error": {"code": -32600, "message": "nope"}}

    class _Client:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, *_: Any, **__: Any) -> _Response:
            return _Response()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "AsyncClient", _Client)
        with pytest.raises(RuntimeError, match="protocol error"):
            await spec.check_ready()


# ── the stdio child ────────────────────────────────────────────────────────


def test_the_stdio_child_serves_the_registry_and_authors_no_config() -> None:
    """The process the CLI SPAWNED cannot also be the process that tells the
    CLI how to spawn it, so this half writes nothing: a config file written
    here would be a second, stale description of a server already running, and
    its placeholder path is deliberately one that cannot be opened. Driven over
    a pipe that is already at EOF, so the loop sets up and shuts down without a
    subprocess.

    ``start()`` and ``_write_config()`` stay SAFE on this spec rather than
    being forbidden — the context-manager shape is shared with the HTTP path so
    a caller does not branch on the transport — and the filesystem assertions
    live out here in the sync body, because that placeholder path must never be
    opened and ``ruff``'s ASYNC240 is right that a blocking stat has no place
    inside an async test either way.
    """
    import contextlib

    import anyio
    import mcp.server.stdio as stdio_mod

    @contextlib.asynccontextmanager
    async def closed_pipe() -> Any:
        to_server, from_client = anyio.create_memory_object_stream[Any](1)
        to_client, from_server = anyio.create_memory_object_stream[Any](1)
        await to_server.aclose()
        try:
            yield from_client, to_client
        finally:
            await from_client.aclose()
            await to_client.aclose()
            await from_server.aclose()

    built: list[McpServerSpec] = []
    original = McpServerSpec.build_server

    def capture(self: McpServerSpec) -> Any:
        built.append(self)
        return original(self)

    async def drive() -> None:
        await serve_registry_stdio(ToolRegistry.from_tools([run_check]), name="engine", ctx=make_test_ctx())
        built[0]._write_config()
        await built[0].start()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stdio_mod, "stdio_server", closed_pipe)
        patch.setattr(McpServerSpec, "build_server", capture)
        asyncio.run(drive())

    child = built[0]
    assert [t.name for t in child._advertised] == ["run_check"], "the child serves the registry it was given"
    assert child._document == {}, "the spawned process must not author the document that spawned it"
    assert str(child.config_path) == "<stdio: no config file>", "a placeholder that looks openable invites a read"
    assert not child.config_path.exists(), "the child authored a config it must not own"


def test_the_stdio_child_is_spawned_with_this_interpreter_not_whatever_python_means() -> None:
    """The CLI spawns the child with the CALLER's ``PATH``, not the
    virtualenv's, so a bare ``python`` there is whatever the system ships —
    which will not have agentkit installed, and the failure is "MCP server
    failed to connect" with no mention of an interpreter."""
    import sys

    assert stdio_command("pkg.server") == (sys.executable, "-m", "pkg.server")
    assert stdio_command("pkg.server")[0] != "python"


@pytest.mark.asyncio
async def test_a_readiness_reply_with_no_data_frame_is_not_reported_as_ready() -> None:
    """The counterpart of the protocol-error test, and the one that used to pass.
    A reply declared `text/event-stream` with no `data:` frame fell back to the
    literal string `"{}"` — which parses, carries no `error`, and yields an empty
    catalogue. So a malformed reply reported READY, and the caller found out via
    a `KeyError` on `ready["tools"]` somewhere else entirely, long after the probe
    that existed to catch it had said yes."""
    import httpx

    spec = _spec()

    class _Response:
        headers = {"content-type": "text/event-stream"}
        text = "event: message\n\n"  # a frame, but no `data:` line

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, *_: Any, **__: Any) -> _Response:
            return _Response()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "AsyncClient", _Client)
        with pytest.raises(RuntimeError, match="no `data:` frame"):
            await spec.check_ready()


@pytest.mark.asyncio
async def test_a_readiness_reply_with_neither_error_nor_result_is_not_reported_as_ready() -> None:
    """`payload.get("result") or {}` turned a reply that answered nothing into an
    empty catalogue, which is indistinguishable from a healthy server that has no
    tools. A readiness probe that cannot tell those apart is not a readiness
    probe."""
    import httpx

    spec = _spec()

    class _Response:
        headers = {"content-type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"jsonrpc": "2.0", "id": 1}

    class _Client:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, *_: Any, **__: Any) -> _Response:
            return _Response()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "AsyncClient", _Client)
        with pytest.raises(RuntimeError, match="neither an `error` nor a `result`"):
            await spec.check_ready()
