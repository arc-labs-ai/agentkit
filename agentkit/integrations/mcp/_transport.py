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

.. warning::
   Everything here binds ``127.0.0.1`` with **no authentication**: anything
   able to reach the port can call the tools behind it. Loopback-only is the
   containment, which is why ``host`` defaults to ``127.0.0.1`` and why no
   caller in this package overrides it. Do not bind a routable address.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import weakref
from dataclasses import dataclass, field
from typing import Any

_UVICORN_INSTALL_HINT = (
    "serving MCP from agentkit needs the 'mcp' extra: pip install 'arc-agentkit[mcp]'"
)


def qualified_tool_name(server: str, tool: str) -> str:
    """The name the CLI knows a served tool by.

    ``mcp__<server>__<tool>`` is wire contract, not a label: it is what
    ``--permission-prompt-tool`` names, what ``--allowed-tools`` matches, and
    what the model emits in a tool call. Spelled once here so a change to the
    separator cannot land in one caller and not the other.
    """
    return f"mcp__{server}__{tool}"


def http_mcp_config(server: str, url: str) -> dict[str, Any]:
    """The ``--mcp-config`` document for one loopback HTTP server."""
    return {"mcpServers": {server: {"type": "http", "url": url}}}


@dataclass
class LoopbackMcpTransport:
    """A uvicorn server on a background task, bound to loopback.

    ``port=0`` means "let the OS choose", resolved by :meth:`reserve`.
    """

    host: str = "127.0.0.1"
    port: int = 0

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

    @property
    def running(self) -> bool:
        return self._task is not None

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
        config = uvicorn.Config(
            app,
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
    "http_mcp_config",
    "qualified_tool_name",
]
