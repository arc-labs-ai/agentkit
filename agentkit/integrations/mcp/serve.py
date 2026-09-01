"""``serve_registry`` — hand a ``ToolRegistry`` to the Claude CLI over MCP.

:class:`~agentkit.agents.cognition.claude_cli.ClaudeCliCognition` delegates the
whole loop to the CLI, and the CLI's only seam for a tool it did not ship with
is ``--mcp-config``. Everything needed to DESCRIBE an agentkit tool was already
here and already correct — ``FunctionTool`` derives a ``ToolSchema`` from a
signature and a docstring, ``ToolRegistry`` holds them and refuses a name
collision, ``ToolArgumentError`` refuses a bad call with a message the model can
act on. Only the transport was missing, and a caller was left assembling MCP
JSON by hand::

    spec = serve_registry(registry, name="engine", ctx=ctx)
    async with spec:
        cognition = ClaudeCliCognition(
            model="claude-sonnet-4-6",
            **spec.cli_kwargs(builtin_tools=False),   # only OUR tools
        )
        result = await Agent(name="dev", cognition=cognition).run(task, ctx)

Inside the session the tools appear as ``mcp__engine__<tool>``.

Four decisions here are load-bearing rather than stylistic, and each is
defended at its site below:

* ``ToolSchema.parameters`` becomes MCP ``inputSchema`` **unchanged**. Both are
  JSON Schema; a translation layer would be a second description of one thing
  and would drift.
* A :class:`~agentkit.tools.errors.ToolArgumentError` becomes an MCP **tool**
  error, never a transport error. A bad call is reflected to the model, which
  authored it and is the only party that can fix it; a transport error ends the
  session instead and the model never learns what it got wrong.
* ``requires_approval``, ``side_effecting`` and ``caps`` all travel, so
  ``RunPolicy``'s Rule-of-Two check and the CLI's permission prompt keep
  applying to tools that moved behind MCP — which is exactly when they matter
  most.
* The config is written to a real file, because ``--mcp-config`` takes a path
  or inline JSON and a caller should be building neither.

Requires the ``mcp`` extra (``pip install "arc-agentkit[mcp]"``).

The HTTP transport is **authenticated by default**: the listener carries a
generated bearer token and ``cli_kwargs()`` keeps carrying everything needed to
address it, so nothing about the wiring above changes. That default is the one
behavioural change worth calling out, and it is a change of default rather than
of capability — ``auth="none"`` is still there and is now something a caller
says out loud.

A served registry is where this matters most, and more than it does for the
sibling ``ApprovalServer``. An approval server answers prompts; a served
``ToolRegistry`` is whatever the application put in it. The listener sits in the
same network namespace as the CLI's ``Bash``, which routinely runs a build, a
package install or a test suite out of a repository the agent was pointed at, so
"only loopback can reach it" and "only the runner can reach it" are not the same
statement. A verdict reachable by anything sharing the namespace is a verdict no
longer produced only by the runner that was supposed to produce it: it does not
have to be attacked to be worthless, it only has to be reachable. See
:mod:`agentkit.integrations.mcp._transport` for the fence and for why it is a
bearer token rather than a Unix socket.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from agentkit.integrations.mcp._transport import (
    LoopbackMcpTransport,
    McpAuth,
    http_mcp_config,
    qualified_tool_name,
    validated_auth,
)
from agentkit.kernel._json import dumps as _json_dumps
from agentkit.kernel.concurrency import Cancelled
from agentkit.kernel.protocols import Ctx
from agentkit.tools.base import Tool
from agentkit.tools.errors import ToolArgumentError
from agentkit.tools.registry import ToolRegistry

_MCP_INSTALL_HINT = (
    'MCP integration requires the `mcp` package. Install with: pip install "arc-agentkit[mcp]"'
)

try:
    from mcp import types as mcp_types
    from mcp.server.lowlevel.server import Server as LowLevelServer
except ImportError as exc:  # pragma: no cover — fires only when the extra is missing
    raise ImportError(_MCP_INSTALL_HINT) from exc


Transport = Literal["http", "stdio"]

# MCP's own tool-name grammar (SEP-986) is ``[A-Za-z0-9._-]{1,128}``, but the
# CLI addresses a served tool as ``mcp__<server>__<tool>`` and callers then put
# that string into ``--allowed-tools``, shell arguments and log greps. Dropping
# the dot keeps the qualified name a single glob-free, quote-free token, which
# is a narrower rule than MCP's and never an invalid one.
_NOT_ADDRESSABLE = re.compile(r"[^A-Za-z0-9_-]")
_SERVER_NAME = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")

# The key our per-tool metadata hides under in MCP ``_meta``. Namespaced
# because ``_meta`` is a shared bag and an unqualified ``caps`` would collide
# with whatever the next server-side middleware wants to attach.
META_KEY = "agentkit"


def _mcp_tool_name(name: str) -> str:
    """The addressable spelling of a tool name.

    The model only ever sees this form — it comes back in ``tools/list`` — so
    the rename is invisible to the party issuing calls. It is NOT invisible to
    the caller, who writes the qualified name into ``allowed_tools`` by hand,
    which is why :attr:`McpServerSpec.mcp_names` publishes the mapping instead
    of leaving it implicit.
    """
    out = _NOT_ADDRESSABLE.sub("_", name)[:128]
    if not out:
        raise ValueError(f"tool name {name!r} has no MCP-addressable characters at all")
    return out


def _render(value: Any) -> str:
    """Whatever the tool returned, as the text the model will read.

    A ``str`` passes through unquoted: it is already the answer, and
    ``json.dumps`` would wrap it in quotes and escape its newlines, which is
    noise in the transcript and changes what a downstream string comparison
    sees.

    Anything JSON can encode is encoded, so the model gets something parseable
    rather than Python ``repr`` syntax. Anything JSON cannot encode falls back
    to ``repr`` and MUST NOT raise: the tool already RAN, its side effect has
    already happened, and turning a successful call into a failed one at the
    serialisation step is the worst available outcome — the model would retry a
    write that already landed.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return _json_dumps(value)
    except (TypeError, ValueError):
        return repr(value)


def _text_result(text: str, *, is_error: bool) -> mcp_types.CallToolResult:
    """One text block, matching what the CLI actually reads.

    ``structuredContent`` is deliberately absent. The sibling
    :class:`~agentkit.integrations.mcp.approvals.ApprovalServer` learned the
    hard way that the CLI rejects a permission result carrying one; a tool
    result is more forgiving, but there is no second description of the answer
    worth maintaining here either.
    """
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        isError=is_error,
    )


def _tools_of(registry: ToolRegistry | Sequence[Tool] | Iterable[Tool]) -> list[Tool]:
    """Accept a registry or a bare sequence of tools.

    A ``ToolRegistry`` is the normal case and carries the collision guarantee.
    A sequence is accepted because the caller building a CLI session often has
    a list in hand and would otherwise write ``ToolRegistry.from_tools(...)``
    at every call site — and ``from_tools`` is exactly what happens to it here,
    so a duplicate name still raises rather than silently shadowing.
    """
    if isinstance(registry, ToolRegistry):
        return registry.tools()
    return ToolRegistry.from_tools(registry).tools()


def _advertise(tool: Tool, *, mcp_name: str) -> mcp_types.Tool:
    """One agentkit ``Tool`` as one MCP tool definition.

    ``inputSchema`` is ``ToolSchema.parameters`` passed through with a single
    ``dict()`` unwrap of the ``FrozenDict`` — no key renaming, no shape
    massaging, no defaults filled in. Both sides are JSON Schema, so a
    translation step would be a second description of one thing: it would
    drift, and the drift shows up as the model being shown a schema the tool
    does not validate against, which reads to it as an unexplained rejection.

    ``readOnlyHint`` is ``not (side_effecting or requires_approval)`` rather
    than plain ``not side_effecting``. ``readOnlyHint`` is precisely the hint a
    client consults to decide it may run something without asking, so a
    read-only tool that nonetheless declares ``requires_approval`` must not
    advertise itself as free — otherwise the declaration stops meaning anything
    the moment the tool moves behind MCP.

    ``side_effecting``, ``requires_approval`` and ``caps`` are ALSO repeated in
    ``_meta`` verbatim. The annotations are hints, and the MCP spec tells
    clients not to make decisions on hints from untrusted servers; ``_meta``
    is where a cooperating agentkit-side consumer (``RunPolicy``, an audit
    trail) can read the declarations back without inferring them from three
    booleans that were lossy on the way out.
    """
    assert tool.schema is not None  # filtered by the caller; see _plan
    side_effecting = bool(tool.side_effecting)
    requires_approval = bool(tool.requires_approval)
    caps = tuple(getattr(tool, "caps", ()) or ())
    return mcp_types.Tool(
        name=mcp_name,
        description=tool.description,
        inputSchema=dict(tool.schema.parameters),
        annotations=mcp_types.ToolAnnotations(
            readOnlyHint=not (side_effecting or requires_approval),
            destructiveHint=side_effecting,
            idempotentHint=bool(getattr(tool, "idempotent", False)),
        ),
        _meta={
            META_KEY: {
                "side_effecting": side_effecting,
                "requires_approval": requires_approval,
                "caps": list(caps),
            }
        },
    )


def _plan(tools: Sequence[Tool]) -> dict[str, Tool]:
    """Map MCP name → tool, refusing a collision instead of shadowing.

    ``ToolRegistry`` already refuses a duplicate name, because a silent
    overwrite changes the implementation behind an unchanged advertised name.
    Sanitising re-opens exactly that hole one layer down: ``run.check`` and
    ``run check`` are two distinct registrations that arrive at one MCP
    identifier, and the model would call one and get the other with no signal
    anywhere. So the same policy applies here, one layer lower.

    A tool with ``schema is None`` is dropped rather than advertised. The
    ``Tool`` Protocol calls that loop-invisible, and MCP has no way to say
    "callable but undescribed" — ``inputSchema`` is required — so inventing an
    empty schema would advertise something the model would then call blind.
    """
    plan: dict[str, Tool] = {}
    for tool in tools:
        if tool.schema is None:
            continue
        mcp_name = _mcp_tool_name(tool.name)
        if mcp_name in plan:
            raise ValueError(
                f"MCP tool name collision: {plan[mcp_name].name!r} and {tool.name!r} both "
                f"become {mcp_name!r} once non-addressable characters are replaced. "
                "Rename one — the model would otherwise call one and reach the other."
            )
        plan[mcp_name] = tool
    return plan


@dataclass
class McpServerSpec:
    """Everything a caller needs to point the CLI at a served ``ToolRegistry``.

    Built by :func:`serve_registry`; not constructed directly. Owns the config
    file and (for ``transport="http"``) the listener, so it is also the async
    context manager that starts and stops them::

        async with spec:
            ...

    ``config_path`` is a real file because ``--mcp-config`` takes a path or
    inline JSON, and the inline form that
    :class:`~agentkit.integrations.mcp.approvals.ApprovalServer` uses is only
    viable there because that document is one line. A registry's document is
    not, and a service that has to embed it in an argv is a service one tool
    away from an "argument list too long".
    """

    name: str
    config_path: Path
    transport: Transport
    auth: McpAuth
    tool_names: tuple[str, ...]
    mcp_names: Mapping[str, str]
    requires_approval: tuple[str, ...]
    auto_approve: tuple[str, ...]
    caps: tuple[str, ...]
    timeout_s: float | None
    host: str
    port: int

    _tools: Mapping[str, Tool] = field(repr=False, default_factory=dict)
    _ctx: Any = field(repr=False, default=None)
    _advertised: list[Any] = field(repr=False, default_factory=list)
    _document: dict[str, Any] = field(repr=False, default_factory=dict)
    _owns_config_dir: bool = field(repr=False, default=True)
    _transport: LoopbackMcpTransport | None = field(default=None, repr=False)
    _calls: int = field(default=0, repr=False, init=False)

    # ---- wiring ----------------------------------------------------------------------------

    @property
    def url(self) -> str | None:
        """The MCP endpoint, or ``None`` for a stdio server — there is no URL
        to have, and returning a plausible-looking one would be a lie a caller
        could paste somewhere."""
        if self.transport != "http":
            return None
        return f"http://{self.host}:{self.port}/mcp"

    @property
    def auth_headers(self) -> Mapping[str, str]:
        """The headers a client must send to be answered, empty under
        ``auth="none"``.

        ``cli_kwargs()`` already carries everything the CLI needs, because a
        caller assembling MCP JSON by hand is the gap this module exists to
        close. A client that is not the CLI — agentkit's own
        ``StreamableHttpServer``, a probe, a health check — needs the same
        thing one layer down, and this is it.

        A mapping rather than the raw token, and there is deliberately no
        ``spec.token``: a bare token invites concatenation into a URL, and a
        credential in a URL is the one placement the fence refuses outright
        because URLs are logged.
        """
        if self._transport is None:
            return {}
        return MappingProxyType(self._transport.auth_headers)

    @property
    def calls_seen(self) -> int:
        """How many tool calls this server has handled. A session that reports
        zero either never called a tool or never reached the server — worth
        being able to tell apart before blaming the model."""
        return self._calls

    def cli_kwargs(self, *, builtin_tools: bool = True) -> dict[str, Any]:
        """The ``ClaudeCliCognition`` fields that wire this server in.

        ``strict_mcp_config=True`` is always included: without it the CLI also
        loads whatever MCP servers the working directory or the user's home
        configuration happen to define, which is not what a service wiring its
        own registry is asking for. It is the difference between "these tools"
        and "these tools plus a teammate's ``.mcp.json``".

        ``builtin_tools`` is a SEPARATE decision and defaults to leaving the
        CLI alone. ``tools=("",)`` disables Read, Grep, Bash and the rest —
        that is a statement about what the session can do at all, not about MCP
        wiring, and a flag named ``strict_mcp_config`` should not quietly make
        it. Pass ``builtin_tools=False`` when the registry really is meant to
        be the entire toolbox.
        """
        kwargs: dict[str, Any] = {
            "mcp_config": (str(self.config_path),),
            "strict_mcp_config": True,
        }
        if not builtin_tools:
            kwargs["tools"] = ("",)
        return kwargs

    def codex_kwargs(self, *, tool_timeout_s: float | None = None) -> dict[str, Any]:
        """The ``CodexCliCognition`` fields that wire this server in.

        The same server, spelled the way the other CLI reads it: Codex takes MCP
        servers as ``config.toml`` keys (``-c mcp_servers.<name>.<field>=…``)
        rather than as a ``--mcp-config`` document, and an authenticated HTTP
        server's token travels in an environment variable it names rather than
        in a header. Both are handled by
        :func:`agentkit.integrations.codex_cli.as_codex_mcp`; this method exists
        so a reader who found ``cli_kwargs`` finds the counterpart in the same
        place instead of concluding there is none.

        There is no ``builtin_tools=`` switch, unlike ``cli_kwargs``. Codex has
        no tool allow-list — every session has ``shell`` — so "only OUR tools"
        is not a thing a flag can say there, and a parameter that quietly did
        nothing would be worse than its absence. What contains the CLI's own
        tools in a Codex session is ``sandbox=``.
        """
        from agentkit.integrations.codex_cli import as_codex_mcp

        return as_codex_mcp(self, tool_timeout_s=tool_timeout_s)

    # ---- the server ------------------------------------------------------------------------

    def build_server(self) -> Any:
        """The lowlevel MCP ``Server`` this spec serves. Public so the wire
        format can be pinned without opening a socket — which is how the
        byte-equality of ``inputSchema`` is asserted.

        Lowlevel rather than ``FastMCP`` on purpose, and this is the choice the
        whole module rests on. ``FastMCP`` derives ``inputSchema`` from a Python
        signature via pydantic and validates arguments against that derived
        model BEFORE the handler runs. Both are exactly wrong here: the schema
        already exists (``ToolSchema.parameters``) and re-deriving it would
        advertise something else, and pydantic's rejection would replace
        ``ToolArgumentError``'s message — which names the tool, the offending
        arguments and the accepted set — with a generic validation dump.

        ``validate_input=False`` for the same reason: the lowlevel server would
        otherwise jsonschema-check the arguments and return its own message.
        agentkit's tools already validate their own calls, and their diagnosis
        is the better one.
        """
        server: Any = LowLevelServer(self.name)
        advertised = self._advertised

        @server.list_tools()  # type: ignore[misc, no-untyped-call, untyped-decorator]
        async def _list() -> list[Any]:
            return advertised

        @server.call_tool(validate_input=False)  # type: ignore[misc, no-untyped-call, untyped-decorator]
        async def _call(name: str, arguments: dict[str, Any]) -> Any:
            return await self._dispatch(name, arguments)

        return server

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
        """One MCP tool call → one ``Tool.run`` → one text block.

        Never raises. Everything below returns an ``isError`` RESULT instead,
        and the difference is the whole point of the module: an exception out
        of this handler is a TRANSPORT error, which ends the MCP session and
        takes every other tool in the registry with it. One tool raising would
        disable the whole registry mid-run, and the model would be told the
        tool system is broken rather than that its call failed.

        No per-call state is kept on ``self`` beyond the counter, and that is
        deliberate rather than incidental: the CLI issues several tool calls
        concurrently, so a ``self._current_arguments`` would hand one call's
        arguments to another's handler.
        """
        self._calls += 1
        tool = self._tools.get(name)
        if tool is None:
            return _text_result(
                f"no tool named {name!r} on MCP server {self.name!r}. "
                f"Available: {sorted(self._tools) or '<none>'}",
                is_error=True,
            )

        # The run's cancellation seam, which the CLI knows nothing about. Once
        # the run is cancelled every remaining call has to be refused BEFORE
        # the tool body runs — a side-effecting tool that fires after someone
        # pressed stop is the failure this check exists to prevent.
        try:
            self._ctx.check_cancelled()
        except Cancelled as exc:
            return _text_result(f"the run was cancelled; {name!r} was not run ({exc})", is_error=True)

        # ``asyncio.timeout`` rather than ``asyncio.wait_for`` so that OUR
        # deadline expiring can be told apart from the tool raising a
        # ``TimeoutError`` of its own — an upstream read, a lock acquisition, a
        # subprocess wait. ``wait_for`` collapses both into one indistinguishable
        # exception, and the handler below then reported every one of them as
        # this server's deadline. With ``timeout_s=None`` that produced the
        # literal text "did not return within Nones and was abandoned" for a
        # tool that had in fact failed fast with a perfectly good message of its
        # own, which was then discarded. ``deadline.expired()`` is True only when
        # the scope below is what did the cancelling.
        deadline: asyncio.Timeout | None = None
        try:
            if self.timeout_s is None:
                result = await tool.run(arguments, self._ctx)
            else:
                async with asyncio.timeout(self.timeout_s) as deadline:
                    result = await tool.run(arguments, self._ctx)
        except ToolArgumentError as exc:
            # THE case this module exists to get right. A bad call is reflected
            # to the model verbatim: the message already names the tool, the
            # offending arguments and the accepted set, which is what turns the
            # next turn into a repair rather than a guess.
            return _text_result(str(exc), is_error=True)
        except TimeoutError as exc:
            if deadline is not None and deadline.expired():
                return _text_result(
                    f"tool {name!r} did not return within {self.timeout_s}s and was abandoned",
                    is_error=True,
                )
            # The tool raised it, not us. Its message names what actually timed
            # out; ours would name a deadline that never fired.
            return _text_result(f"tool {name!r} failed: TimeoutError: {exc}", is_error=True)
        except Cancelled as exc:
            # A ``RuntimeError`` subclass, so it would otherwise be swallowed
            # by the generic clause below and reported to the model as an
            # ordinary failure — an invitation to retry work somebody
            # deliberately stopped.
            return _text_result(f"the run was cancelled while {name!r} was running ({exc})", is_error=True)
        except Exception as exc:  # noqa: BLE001 — see the docstring
            return _text_result(f"tool {name!r} failed: {type(exc).__name__}: {exc}", is_error=True)

        return _text_result(_render(result), is_error=False)

    # ---- lifecycle -------------------------------------------------------------------------

    async def __aenter__(self) -> McpServerSpec:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Serve. Idempotent, and a no-op for ``transport="stdio"``.

        Nothing to start for stdio because the CLI owns that process: it reads
        the config, spawns the command, and speaks to its stdin. The context
        manager is still the right shape there so a caller does not have to
        branch on the transport just to get the config file cleaned up.

        The config file is put back if it is missing, which is what makes a
        second ``async with spec`` mean what it looks like. ``stop()`` deletes
        it, and both ``start()`` and ``stop()`` are documented as idempotent, so
        re-entering used to hand back a spec that was listening perfectly well
        and whose ``cli_kwargs()`` named a path that no longer existed — a live
        server the CLI could not be pointed at, reported as success.
        """
        self._write_config()
        if self.transport != "http" or self._transport is None:
            return
        await self._transport.start(self._asgi_app())
        self.port = self._transport.port

    async def stop(self) -> None:
        """Release the port and delete the config file. Idempotent, and safe on
        a spec that never started.

        The config file goes because a stale one outlives the port it names,
        and the next reader gets a URL pointing at nothing — or, worse, at
        whatever bound that port next. Under ``auth="bearer"`` it is worse
        still: the document carries the token, so a config that outlives its
        listener is a live credential lying on disk naming a port something
        else may bind.

        Which is why the removal is in a ``finally``. Without it, a listener
        whose shutdown raised — a uvicorn task that would not join, an fd
        already closed underneath us — took the deletion with it and left
        exactly that file behind, on the one path where something had already
        gone wrong and nobody was going to come back and tidy up.

        The FILE goes even when the directory is the caller's. The directory is
        theirs (they may well have chosen a persistent one, which is the point
        of passing ``config_path=``); the file is ours, we wrote it, and what we
        wrote into it is a secret.
        """
        try:
            if self._transport is not None:
                await self._transport.stop()
        finally:
            if self._owns_config_dir:
                shutil.rmtree(self.config_path.parent, ignore_errors=True)
            else:
                self.config_path.unlink(missing_ok=True)

    def _write_config(self) -> None:
        """Materialise the ``--mcp-config`` document. Idempotent.

        Called once by :func:`serve_registry` so the spec is complete before
        anything is serving, and again by :meth:`start` so a restart does not
        leave the CLI pointed at a file ``stop()`` removed. A spec built by
        :func:`serve_registry_stdio` has no document and writes nothing — that
        process was spawned BY the config and cannot also author it.
        """
        if not self._document:
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # 0600 was hygiene when the document only named a loopback endpoint:
        # anything that can scan the port range finds the server anyway, and
        # there was simply no reason to publish the address. Under
        # ``auth="bearer"`` it stopped being hygiene and became the fence's
        # other half — the CLI reads the token OUT of this file, so anything
        # that can read this file holds the credential.
        #
        # Which is why the mode is set on the DESCRIPTOR, before a byte of the
        # document is written, rather than by a ``chmod`` afterwards.
        # ``write_text`` creates at ``0o666 & ~umask`` — 0644 under the common
        # umask — so a later ``chmod`` leaves a window in which the live token
        # is on disk world-readable. Measured, not theorised: a polling thread
        # read the real credential out of the file at 0644 before the chmod
        # landed. The default ``config_path`` is inside a ``mkdtemp`` whose 0700
        # hid it, but ``config_path=`` is a supported argument and ``stop()``
        # reasons explicitly about callers who point it at a directory that
        # persists — which is exactly where the window is reachable. The window
        # also reopened on every ``start()``, because this method runs again.
        #
        # ``O_CREAT``'s mode argument only applies when the file is created, so
        # ``fchmod`` covers the restart case where it already exists at some
        # other mode. Both act on the descriptor, so neither can be raced by a
        # symlink swapped in AFTER the open.
        #
        # ``O_NOFOLLOW`` covers the case that defence does not: a symlink
        # already sitting at ``config_path`` when the open runs. Without it,
        # ``O_CREAT`` follows the link and writes the bearer token through to
        # whatever it points at, at whatever mode that file already had — so
        # the credential lands somewhere the caller never named and the
        # ``fchmod`` tightens a descriptor that was never the risk. The window
        # is real wherever the path is attacker-influenced: a predictable name
        # under a shared ``/tmp``, or a caller-supplied ``config_path`` in a
        # directory something else can write.
        #
        # Refusing is the only safe answer, and it costs a caller who
        # deliberately symlinked their config path an ``OSError`` naming it —
        # which is the right trade, because there is no way to tell that caller
        # apart from an attacker who got there first.
        fd = os.open(
            self.config_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", closefd=False) as handle:
                handle.write(json.dumps(self._document, indent=2))
        finally:
            os.close(fd)

    def _asgi_app(self) -> Any:
        """The streamable-HTTP app around the lowlevel server.

        ``stateless=True`` because each tool call is a complete request: there
        is no session state worth resuming, and a stateful manager would keep
        one session per CLI process for no benefit. Same reasoning, and the
        same setting, as ``ApprovalServer``.
        """
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Route

        manager = StreamableHTTPSessionManager(app=self.build_server(), stateless=True)

        class _Asgi:
            """A CLASS, not a function, because Starlette routes a callable
            with a function type as a request/response endpoint and only an
            object as a raw ASGI app — the three-argument signature would
            otherwise be called with one ``Request``."""

            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                await manager.handle_request(scope, receive, send)

        return Starlette(
            routes=[Route("/mcp", endpoint=_Asgi())],
            lifespan=lambda _app: manager.run(),
        )


def serve_registry(
    registry: ToolRegistry | Sequence[Tool],
    *,
    name: str,
    ctx: Ctx,
    transport: Transport = "http",
    auth: McpAuth | None = None,
    timeout_s: float | None = None,
    config_path: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    command: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> McpServerSpec:
    """Describe ``registry`` as an MCP server the Claude CLI can be pointed at.

    Synchronous on purpose: it reserves the port and writes the config file, so
    the returned spec is complete enough to build a ``ClaudeCliCognition``
    before anything is serving. ``async with spec`` then starts the listener.

    ``ctx`` is handed to every ``tool.run(args, ctx)``. It is captured once,
    which also makes it the cancellation seam — cancelling that ctx refuses
    every subsequent call rather than letting the CLI keep driving tools for a
    run somebody stopped.

    ``timeout_s`` bounds a single tool call and defaults to ``None``. A default
    deadline would silently kill a legitimately slow tool — a long build, a
    human-in-the-loop approval — and the failure would look like a flake. But
    without one a tool that never returns parks the CLI turn forever with no
    signal anywhere: the CLI waits on the MCP call, agentkit waits on the tool,
    and nothing times out. The caller is the only party that knows which of
    those they have.

    ``transport="stdio"`` needs ``command``: the CLI SPAWNS a stdio server, so
    the tools are reconstructed in a fresh process and this process's registry,
    its closures and its ``ctx`` do not survive the boundary. The command is
    whatever re-creates them on the other side, and it should call
    :func:`serve_registry_stdio`.

    ``auth`` defaults to whatever the chosen transport can actually enforce:
    ``"bearer"`` over HTTP, ``"none"`` over stdio. It resolves per transport
    rather than being one constant because the two boundaries are different
    objects, not two settings of one. Over HTTP the port is reachable by
    anything sharing the namespace and a token is the only thing that narrows
    it. Over stdio there is no request to carry a header at all: the CLI
    spawned the process and holds both ends of the pipe, which IS the check,
    and demanding an explicit ``auth="none"`` there would be ceremony for a
    boundary the OS already holds.

    Which is why ``transport="stdio", auth="bearer"`` RAISES here rather than
    quietly handing back an unauthenticated spec. A caller who asked for a
    credential and got a server without one has a guard that looks wired and
    enforces nothing — the exact failure this package refuses everywhere else,
    and the only reason to prefer a wiring-time ``ValueError`` over a
    convenient degradation.

    ``auth="none"`` is the old unauthenticated loopback listener, unchanged and
    still supported. It is a defensible choice on a host nobody else shares; it
    stopped being defensible as the thing you get by saying nothing.
    """
    if not _SERVER_NAME.fullmatch(name):
        # The tool names are rewritten because the model never typed them; the
        # server name the caller typed themselves and then hardcodes into
        # ``allowed_tools`` strings and log greps. Renaming it silently would
        # break a string they are holding somewhere else.
        raise ValueError(
            f"MCP server name {name!r} is not addressable: the CLI spells a served tool "
            "mcp__<server>__<tool>, so the server name must match [A-Za-z0-9_-]{1,64}"
        )
    if transport == "stdio" and not command:
        raise ValueError(
            "serve_registry(transport='stdio') needs command=: the CLI spawns a stdio "
            "server itself, so the tools are rebuilt in a fresh process and this "
            "process's registry, closures and ctx cannot cross that boundary. Pass the "
            "argv that re-creates them — a module whose main calls serve_registry_stdio "
            "— or use transport='http', which keeps the tools in THIS process."
        )

    # Resolved before anything is built, so the refusal below lands in the
    # caller's own frame next to the wiring that caused it.
    mode: McpAuth = (
        ("bearer" if transport == "http" else "none")
        if auth is None
        else validated_auth(auth, where="serve_registry")
    )
    if transport == "stdio" and mode == "bearer":
        raise ValueError(
            "serve_registry(transport='stdio', auth='bearer') cannot be provided: a stdio "
            "server has no request to carry a credential — the CLI spawns the process and "
            "holds both ends of the pipe, and that pipe is the boundary. Pass auth='none' "
            "to say so explicitly, or transport='http', where the token is a real fence "
            "in front of a port anything on the host can otherwise reach."
        )

    plan = _plan(_tools_of(registry))
    advertised = [_advertise(tool, mcp_name=n) for n, tool in sorted(plan.items())]
    mcp_names = {tool.name: qualified_tool_name(name, n) for n, tool in plan.items()}

    approval = tuple(
        sorted(qualified_tool_name(name, n) for n, t in plan.items() if t.requires_approval)
    )
    every = tuple(sorted(qualified_tool_name(name, n) for n in plan))
    caps: set[str] = set()
    for tool in plan.values():
        caps.update(getattr(tool, "caps", ()) or ())

    listener: LoopbackMcpTransport | None = None
    if transport == "http":
        listener = LoopbackMcpTransport(host=host, port=port, authenticated=mode == "bearer")
        # Reserve BEFORE writing the config: the file has to name the port, and
        # picking one that something else then takes is the failure mode the
        # reservation exists to remove. See ``LoopbackMcpTransport.reserve``.
        port = listener.reserve()
        # The token comes off the listener that will enforce it, not out of a
        # second generator here. One object holds the credential, so the
        # document and the fence cannot disagree — and a spec that re-enters
        # ``async with`` rewrites the same document, because the listener it
        # reads from is the same object.
        document = http_mcp_config(name, listener.url, token=listener.token)
    else:
        document = {
            "mcpServers": {
                name: {
                    "type": "stdio",
                    "command": command[0],
                    "args": list(command[1:]),
                    "env": dict(env or {}),
                }
            }
        }

    owns_dir = config_path is None
    if config_path is None:
        path = Path(tempfile.mkdtemp(prefix="agentkit-mcp-")) / f"{name}.mcp.json"
    else:
        path = Path(config_path)

    spec = McpServerSpec(
        name=name,
        config_path=path,
        transport=transport,
        auth=mode,
        tool_names=every,
        mcp_names=MappingProxyType(mcp_names),
        requires_approval=approval,
        auto_approve=tuple(n for n in every if n not in approval),
        caps=tuple(sorted(caps)),
        timeout_s=timeout_s,
        host=host,
        port=port,
        _tools=plan,
        _ctx=ctx,
        _advertised=advertised,
        _document=document,
        _owns_config_dir=owns_dir,
        _transport=listener,
    )
    # Written HERE rather than in ``start()`` alone: the spec has to be complete
    # enough to build a ``ClaudeCliCognition`` before anything is serving, and
    # ``cli_kwargs()`` names this path.
    spec._write_config()
    return spec


async def serve_registry_stdio(
    registry: ToolRegistry | Sequence[Tool],
    *,
    name: str,
    ctx: Ctx,
    timeout_s: float | None = None,
) -> None:
    """Serve ``registry`` on THIS process's stdin/stdout until the peer closes.

    The other half of ``transport="stdio"``: the parent writes a config naming
    a command, the CLI spawns that command, and the command's ``main`` is this.
    Everything about how a tool is advertised and how a bad call is reported is
    shared with the HTTP path — this differs only in which pipe carries it,
    which is the whole reason both go through :meth:`McpServerSpec.build_server`.

    No config file is written here. The process that will be spawned cannot
    also be the process that tells the CLI how to spawn it.
    """
    from mcp.server.stdio import stdio_server

    spec = McpServerSpec(
        name=name,
        # Never read on this path — nothing writes a config in the child. The
        # placeholder is honest about that: a path that does not exist is
        # better than one that looks openable.
        config_path=Path("<stdio: no config file>"),
        transport="stdio",
        # Nothing to authenticate to: the CLI spawned this process and holds
        # both ends of the pipe. Saying "bearer" here would be a claim about a
        # fence that does not exist on this path.
        auth="none",
        tool_names=(),
        mcp_names=MappingProxyType({}),
        requires_approval=(),
        auto_approve=(),
        caps=(),
        timeout_s=timeout_s,
        host="",
        port=0,
        _owns_config_dir=False,
    )
    plan = _plan(_tools_of(registry))
    spec._tools = plan
    spec._ctx = ctx
    spec._advertised = [_advertise(t, mcp_name=n) for n, t in sorted(plan.items())]

    server = spec.build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def stdio_command(module: str) -> tuple[str, ...]:
    """The argv for a child that serves a registry over stdio.

    ``sys.executable`` rather than ``"python"``: the CLI spawns this with the
    caller's ``PATH``, not the virtualenv's, and a bare ``python`` there is
    whatever the system ships — which will not have agentkit installed.
    """
    return (sys.executable, "-m", module)


__all__ = [
    "META_KEY",
    "McpServerSpec",
    "serve_registry",
    "serve_registry_stdio",
    "stdio_command",
]
