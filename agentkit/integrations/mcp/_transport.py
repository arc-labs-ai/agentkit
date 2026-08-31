"""The loopback listener both of agentkit's MCP *servers* run on.

agentkit is the callee twice over — :class:`~agentkit.integrations.mcp.approvals.ApprovalServer`
answers the Claude CLI's permission prompts, and
:func:`~agentkit.integrations.mcp.serve.serve_registry` hands it a ``ToolRegistry``
— and both need exactly the same four things: a loopback port that is genuinely
theirs, a uvicorn task that is up before the CLI is told about it, a shutdown
that releases the port, and the ``mcp__<server>__<tool>`` spelling the CLI's
flags expect.

This module exists because the second server was written and the first already
had all of it. Two copies of a bind-and-wait loop is two places to fix the
"handed the CLI a URL that was not listening yet" bug, and only one of them
would have been fixed.

**Loopback is not the fence.** It used to be the whole argument, and it holds
exactly as long as nothing untrusted shares the host — which is the one thing
``ClaudeCliCognition`` guarantees is false. The CLI runs ``Bash``: a build, a
package install, a test suite, a script out of a repository the agent was
pointed at, all inside the same network namespace as the server holding the
agent's own tools. The trust boundary the loopback argument assumes is
precisely the boundary the tool set is designed to cross, so every listener
here now carries a generated bearer token by default and the request is refused
at the ASGI layer, before the MCP app exists for it. ``host`` still defaults to
``127.0.0.1`` and no caller in this package overrides it: the token is the
authorisation check, loopback is defence in depth, and neither replaces the
other. Do not bind a routable address.

``hook_settings`` in :mod:`agentkit.integrations.claude_cli.hooks` solves the
same problem better, with an ``AF_UNIX`` socket in a ``0700`` directory and the
filesystem as the check. That is not available here and the reason is the
client, not the taste: measured against ``claude`` 2.1.251, ``--transport``
accepts only ``stdio, sse, http``; ``unixSocket`` is not an ``--mcp-config``
key (the ``unixSocket`` strings in the binary are its own sandbox plumbing);
and a ``headers`` entry in the config document DOES arrive — the
``Authorization`` header was observed on the CLI's first ``server/discover``
probe against an instrumented listener. A ``unix`` transport is deliberately
NOT offered: no client in this repository could address one — agentkit's own
``StreamableHttpServer`` takes a URL and the ``mcp`` package's streamable-HTTP
client gives no seam for an ``httpx`` UDS transport — so it would be a fence
around a field nobody can stand in, testable only against a hand-rolled socket
and reachable by nothing that exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import socket
import weakref
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qsl

_UVICORN_INSTALL_HINT = (
    "serving MCP from agentkit needs the 'mcp' extra: pip install 'arc-agentkit[mcp]'"
)

#: How a caller names the fence in front of a server. ``"none"`` is the old
#: unauthenticated loopback listener, kept because it is a defensible choice on
#: a host nobody else shares — and spelled out because it stopped being what
#: you get by saying nothing.
McpAuth = Literal["bearer", "none"]

_AUTH_MODES: tuple[str, ...] = ("bearer", "none")

# 32 bytes of ``secrets`` entropy, which ``token_urlsafe`` renders as 43
# characters. Sized against the threat that motivates the token at all: a
# process sharing the namespace that can issue guesses as fast as the loop
# accepts them. 256 bits is not a number that yields to that.
_TOKEN_BYTES = 32

# Query parameters that name a credential. Refused on the NAME, not the value,
# so the PATTERN is what gets stopped: a caller who reaches for it is told the
# first time rather than the first time a URL turns up in an access log, a
# proxy log, a shell history or an exception carrying the request line.
_CREDENTIAL_QUERY_KEYS = frozenset(
    {"access_token", "api_key", "apikey", "authorization", "bearer", "key", "token"}
)


def validated_auth(auth: str, *, where: str) -> McpAuth:
    """``auth`` as a mode, or a ``ValueError`` naming the alternatives.

    The same fail-closed reflex ``ApprovalServer.autonomy`` already has, and
    for the same reason: an unrecognised value must not fall through to the
    permissive branch. ``auth="bearer "`` out of a YAML file, or ``auth="off"``
    from someone guessing the spelling, would otherwise take the fence down
    with nothing said anywhere — a guard that looks wired and enforces nothing.
    ``Literal`` catches the typo under mypy; this catches it for the callers
    who build their wiring out of configuration, which is most of them.
    """
    if auth not in _AUTH_MODES:
        raise ValueError(
            f"{where}: auth={auth!r} is not an authentication mode — use one of "
            f"{', '.join(_AUTH_MODES)}. An unrecognised mode would leave the "
            "listener unauthenticated, which is the one failure this fence must not have"
        )
    return auth  # type: ignore[return-value]


def bearer_headers(token: str | None) -> dict[str, str]:
    """The request headers that address an authenticated server, or ``{}``.

    Spelled once, because both the ``--mcp-config`` document and the mapping a
    non-CLI client is handed have to agree byte for byte, and they are built in
    different modules. The header name and the ``Bearer `` prefix being
    literals in two places is how one of them ends up lowercase.
    """
    return {"Authorization": f"Bearer {token}"} if token else {}


def require_bearer(app: Any, token: str) -> Any:
    """Wrap ``app`` so an unaddressed request is refused before it reaches it.

    A rejected CALLER is a transport-level rejection, and that is the whole
    design of this function rather than an implementation detail. A bad *call*
    is reflected to the model on purpose — ``ToolArgumentError`` becomes an
    ``isError`` tool result precisely so the model can repair it. A bad
    *caller* must not be: a tool-shaped refusal is something the model reads,
    reasons about and tries to route around, and it is very good at routing
    around things it has been shown. The 401 is answered by an ASGI callable
    that never constructs an MCP session, so there is no tool result to read.

    An empty ``token`` raises rather than building a fence that lets
    ``Authorization: Bearer `` through — ``compare_digest(b"", b"")`` is true,
    so an accidentally-empty credential is not a closed door with a bad key,
    it is an open one that logs nothing.
    """
    if not token:
        raise ValueError(
            "require_bearer: an empty token would authenticate every caller that "
            "sends 'Authorization: Bearer ' — pass a real credential, or do not "
            "wrap the app at all"
        )

    async def guarded(scope: Any, receive: Any, send: Any) -> None:
        # ``lifespan`` is the ONE scope that passes through unchecked, and it is
        # allow-listed by name rather than by "not http".
        # ``StreamableHTTPSessionManager`` starts its task group in the lifespan,
        # so a fence that answered every scope type would leave the manager
        # un-started and every AUTHENTICATED call would then fail inside it — an
        # outage that reads exactly like a credential bug and is not one.
        #
        # Everything else is refused. "Not http" was the obvious spelling and it
        # was a fail-open: uvicorn's ``ws="auto"`` turns on websocket upgrades
        # the moment ``websockets`` or ``wsproto`` is importable — neither is an
        # agentkit dependency, so whether the fence held depended on an
        # unrelated package in the caller's environment — and a ``websocket``
        # scope went straight past the token to the app. Measured: an
        # unauthenticated upgrade against an authenticated listener answered
        # ``HTTP/1.1 101 Switching Protocols``. It was survivable only because
        # the app behind it happens to have no websocket route today, which
        # makes the inner app's routing table the authorisation check. This
        # class is documented as where a third server gets its fence from, and
        # that server would have inherited none for this scope.
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await app(scope, receive, send)
            return
        if scope_type != "http":
            # A close before ``websocket.accept`` is the portable denial:
            # uvicorn turns it into ``403 Forbidden`` and never upgrades.
            # Anything else unrecognised is answered by returning without
            # touching ``app`` — silence is the fail-CLOSED outcome for a scope
            # whose reply shape we do not know.
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return

        query = scope.get("query_string") or b""
        if query:
            keys = {k.lower() for k, _ in parse_qsl(query.decode("latin-1"), keep_blank_values=True)}
            if keys & _CREDENTIAL_QUERY_KEYS:
                # 400 rather than 401: the credential is not missing, the
                # request is malformed. Refused even when the value is
                # CORRECT — the point is to stop the placement, and a
                # placement that works once is a placement that ships.
                await _refuse(
                    send,
                    status=400,
                    error="credential_in_url",
                    detail=(
                        "credentials in the query string are refused because URLs are "
                        "logged; send the token in the Authorization header instead"
                    ),
                )
                return

        expected = token.encode()
        for name, value in scope.get("headers") or ():
            if name.lower() != b"authorization":
                continue
            scheme, _, credential = value.partition(b" ")
            # The scheme comparison is a plain one on purpose: "bearer" is not
            # a secret, and its length leaks nothing. Only the credential goes
            # through ``compare_digest``, which is the line that matters —
            # ``==`` on bytes short-circuits at the first differing byte, and
            # over loopback an attacker owns the timing well enough to turn
            # that into a byte-at-a-time recovery of the token.
            if scheme.lower() == b"bearer" and secrets.compare_digest(credential, expected):
                await app(scope, receive, send)
                return
            # The FIRST ``Authorization`` header decides. Scanning on past a
            # bad one would let a caller send a spray of guesses in one request
            # and, worse, would make what this server accepts depend on header
            # order — which no proxy in the path is obliged to preserve.
            break

        await _refuse(
            send,
            status=401,
            error="unauthorized",
            detail=(
                "this agentkit MCP server requires the bearer token from its "
                "--mcp-config document; read the Authorization header out of that file "
                "(cli_kwargs() / auth_headers already carry it) rather than assembling one"
            ),
        )

    return guarded


async def _refuse(send: Any, *, status: int, error: str, detail: str) -> None:
    """One refusal, rendered the same way every time.

    A JSON body rather than a bare status line because of who reads it. The CLI
    surfaces a failed MCP connection as "MCP server failed to connect" and
    nothing else, so this body is the only place the cause can be written down
    for the operator who goes looking — and the thing they most need told is
    that the credential is in the config document already.

    ``json.dumps`` rather than an f-string: ``detail`` is prose that will be
    edited, and a quote or a backslash in it would otherwise emit a body no
    client can parse — a refusal that reads as a broken server rather than as a
    refusal.

    ``WWW-Authenticate`` is the HTTP contract for a 401 and is what makes the
    rejection legible to a client that never read this file.
    """
    body = json.dumps({"error": error, "detail": detail}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if status == 401:
        headers.append((b"www-authenticate", b'Bearer realm="agentkit-mcp"'))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def qualified_tool_name(server: str, tool: str) -> str:
    """The name the CLI knows a served tool by.

    ``mcp__<server>__<tool>`` is wire contract, not a label: it is what
    ``--permission-prompt-tool`` names, what ``--allowed-tools`` matches, and
    what the model emits in a tool call. Spelled once here so a change to the
    separator cannot land in one caller and not the other.
    """
    return f"mcp__{server}__{tool}"


def http_mcp_config(server: str, url: str, *, token: str | None = None) -> dict[str, Any]:
    """The ``--mcp-config`` document for one loopback HTTP server.

    ``headers`` is the CLI's own key and it is the reason the fence is a bearer
    token rather than a Unix socket: it is the only credential seam the binary
    offers, and it was verified to reach the wire (see the module docstring)
    rather than assumed from the schema. Omitted entirely when there is no
    token, so an unauthenticated document stays byte-identical to the one this
    function produced before there was such a thing.
    """
    entry: dict[str, Any] = {"type": "http", "url": url}
    if token:
        entry["headers"] = bearer_headers(token)
    return {"mcpServers": {server: entry}}


@dataclass
class LoopbackMcpTransport:
    """A uvicorn server on a background task, bound to loopback.

    ``port=0`` means "let the OS choose", resolved by :meth:`reserve`.

    ``authenticated`` defaults to **True**, and the default is the point. This
    class is where a future third server on this seam gets its fence from, and
    the two current ones were unauthenticated for exactly as long as nothing
    made the safe thing the thing you get by saying nothing. The token is
    generated here and there is no parameter to supply one: a caller-supplied
    token is a token that becomes a constant in somebody's config file, checked
    in, and shared by every worker on the fleet.

    It is generated ONCE, at construction, and does not rotate when the
    listener is stopped and started again. ``start()``/``stop()`` are both
    documented idempotent and re-entering is a supported retry shape; a rotated
    token would invalidate a config document the caller already read and hand
    them a "MCP server failed to connect" naming nothing. The credential's
    lifetime is this object's lifetime.
    """

    host: str = "127.0.0.1"
    port: int = 0
    authenticated: bool = True

    # ``repr=False`` is load-bearing, not tidiness: a dataclass repr lands in
    # tracebacks, log lines and pytest failure output, and a transport that
    # renders its own token there defeats the 0600 on the config file the
    # moment anything raises.
    _token: str = field(default="", init=False, repr=False)
    _socket: socket.socket | None = field(default=None, init=False, repr=False)
    # ``weakref.finalize`` is generic in the stubs — over the callable's
    # ParamSpec and the referent — and a plain class at runtime, which only
    # type-checks because ``from __future__ import annotations`` keeps this
    # line from ever being evaluated.
    _closer: weakref.finalize[[], LoopbackMcpTransport] | None = field(
        default=None, init=False, repr=False
    )
    _server: Any = field(default=None, init=False, repr=False)
    _task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.authenticated:
            self._token = secrets.token_urlsafe(_TOKEN_BYTES)

    @property
    def running(self) -> bool:
        return self._task is not None

    @property
    def token(self) -> str | None:
        """The credential this listener requires, or ``None`` when it requires
        none. A property rather than a public field so nothing can assign one
        after construction — a token swapped mid-run is a listener whose config
        file no longer opens it."""
        return self._token or None

    @property
    def auth_headers(self) -> dict[str, str]:
        """The headers a client must send. What a caller uses INSTEAD of the
        raw token: a bare token invites concatenation into a URL, and a
        credential in a URL is the one placement :func:`require_bearer`
        refuses outright."""
        return bearer_headers(self._token)

    @property
    def url(self) -> str:
        """The MCP endpoint. ``/mcp`` is the streamable-http mount both
        FastMCP and the lowlevel session manager use."""
        return f"http://{self.host}:{self.port}/mcp"

    def reserve(self) -> int:
        """Bind the port NOW, synchronously, and hold the socket until
        :meth:`start` hands it to uvicorn.

        Two problems solved by binding here rather than letting uvicorn do it.

        The first is that ``--mcp-config`` wants a URL written to a FILE before
        the CLI is spawned, so the port has to be known before anything is
        serving on it. The previous shape asked the OS for a free port, closed
        the socket, and bound it again seconds later — a race that a second
        agent starting on the same host wins often enough to matter. Holding
        the socket closes that window entirely.

        The second is worse and is why this is not merely tidier. uvicorn's own
        bind failure path is ``logger.error(...); sys.exit(3)``, raised inside
        the ``serve()`` task. Measured against a port held by another listener,
        that ``SystemExit`` propagates through the event loop and out of
        ``asyncio.run`` — a library taking down its host process over a port
        collision, with the awaiting caller getting a bare ``CancelledError``
        and no mention of the port anywhere. Binding here raises the actual
        ``OSError: [Errno 48] Address already in use`` in the caller's own
        frame, where it can be caught and read.
        """
        if self._socket is not None:
            return self.port
        sock = socket.socket()
        # SO_REUSEADDR so a restart does not trip over its own predecessor's
        # TIME_WAIT. It does NOT let two live listeners share the port — an
        # actual collision still raises below, which is the point.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            sock.close()
            # Re-raise with the endpoint in the message. The bare errno does
            # NOT carry it — measured, ``str(OSError)`` for a collision is
            # exactly ``"[Errno 48] Address already in use"`` — and a process
            # running several of these servers gets one indistinguishable line
            # per collision with nothing saying which port. ``errno``,
            # ``strerror`` and the exception type all survive, so anything
            # matching on those still works; only the message grows.
            raise OSError(
                exc.errno,
                f"{exc.strerror}: could not bind {self.host}:{self.port}",
            ) from exc
        self._socket = sock
        # Reserving takes an fd that ``stop()`` is normally responsible for.
        # A caller who builds a spec and then raises before ``async with`` never
        # reaches ``stop()``, and the fd would leak for the life of the process
        # — measured as an unclosed-socket ResourceWarning at interpreter exit,
        # which under a warnings-as-errors suite fails whatever test happens to
        # be running when the GC gets to it. The finalizer holds ``sock`` (not
        # ``self``), so it does not keep this object alive; it only guarantees
        # the fd goes back when the object does.
        self._closer = weakref.finalize(self, sock.close)
        self.port = sock.getsockname()[1]
        return self.port

    async def start(self, app: Any) -> None:
        """Serve ``app`` on the reserved port. Idempotent.

        Idempotent rather than an error because both servers here expose
        ``start()`` publicly alongside an ``async with``, and a caller who does
        both should not get a second listener on a second port that nothing
        will ever shut down.
        """
        if self._task is not None:
            return
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover — exercised by the extra-less env
            raise ImportError(_UVICORN_INSTALL_HINT) from exc

        self.reserve()
        # The fence goes on HERE rather than at each call site, so "did this
        # server remember to authenticate" is not a question that can have two
        # answers. Both servers in this package reach the network through this
        # one line, and a third one will too.
        served = require_bearer(app, self._token) if self._token else app
        config = uvicorn.Config(
            served,
            host=self.host,
            port=self.port,
            log_level="error",  # the app's own logs are the interesting ones
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
        # Wait for serving to actually begin: handing the CLI a URL that is not
        # answering yet turns into a startup timeout thirty seconds later,
        # reported as "MCP server failed to connect" with nothing to debug.
        while not self._server.started:
            if self._task.done():
                # Re-raise whatever killed it. The bare ``await`` alone is not
                # enough: a ``serve()`` that RETURNS instead of raising leaves
                # ``started`` False forever and this loop spins until the test
                # timeout kills the process, naming the wrong thing.
                await self._task
                raise RuntimeError(
                    f"the MCP listener on {self.url} stopped before it began serving"
                )
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        """Shut down and release the port. Idempotent, and safe on a transport
        that never started — a caller unwinding a half-built server should not
        have to remember which half it got to.

        The wait is what makes ``stop()`` mean "the port is free": returning
        while uvicorn is still closing sockets leaves the next agent on this
        host racing a listener that is nearly gone. The explicit ``close()``
        afterwards covers the reserve-but-never-start path, where uvicorn never
        took ownership of the socket at all.
        """
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=5.0)
            self._task = None
        self._server = None
        if self._closer is not None:
            self._closer()  # closes the reserved socket; a no-op if it already ran
            self._closer = None
        self._socket = None


__all__ = [
    "LoopbackMcpTransport",
    "McpAuth",
    "bearer_headers",
    "http_mcp_config",
    "qualified_tool_name",
    "require_bearer",
    "validated_auth",
]
