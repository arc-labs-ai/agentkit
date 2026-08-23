"""MCPClient — async context manager wrapping an ``mcp.ClientSession``.

Owns the connection lifecycle (transport open, session initialize, session
close) so callers get a single ``async with`` block instead of the two-level
context stack the raw MCP SDK exposes.

Two transports today:

- ``StdioServer`` for locally-spawned subprocess servers (the majority of
  the community servers land here).
- ``StreamableHttpServer`` for HTTP-based servers with SSE responses (the
  modern remote/hosted shape).

Every ``mcp`` import is guarded so ``agentkit`` still imports cleanly
without the extra installed; the informative ``ImportError`` fires only
when a caller reaches for this module's public surface.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from agentkit.kernel._frozen import deep_freeze

_MCP_INSTALL_HINT = 'MCP integration requires the `mcp` package. Install with: pip install "arc-agentkit[mcp]"'

try:
    from mcp import ClientSession
    from mcp import types as mcp_types
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError as exc:  # pragma: no cover — fires only when extra is missing
    raise ImportError(_MCP_INSTALL_HINT) from exc

# The streamable-HTTP transport was renamed AND RESIGNATURED upstream:
# ``streamablehttp_client(url, headers=..., timeout=...)`` became
# ``streamable_http_client(url, http_client=...)``, moving ownership of the
# ``httpx.AsyncClient`` — and everything configured on it — to the caller.
#
# The old spelling still works but emits a ``DeprecationWarning``, which any
# caller running with ``-W error`` (this project's own suite included) turns
# into a hard failure raised from inside the transport, where it reads as a
# connection problem rather than a deprecation. Prefer the new one; keep the
# old as the floor of the supported ``mcp>=1.28`` range.
#
# ``_OWNS_HTTP_CLIENT`` records which we got, so the single call site builds the
# right arguments instead of a second import guard living down there.
# Typed ``Any`` deliberately: the two spellings have INCOMPATIBLE signatures
# (that is the whole point of the migration), so mypy cannot unify them under
# one name and would reject the fallback branch's keyword arguments against the
# new signature. The branch below is guarded by ``_OWNS_HTTP_CLIENT`` at
# runtime, which is where the distinction actually lives.
_http_transport: Any
try:
    from mcp.client.streamable_http import (  # type: ignore[attr-defined]
        streamable_http_client as _http_transport,
    )

    _OWNS_HTTP_CLIENT = True
except ImportError:  # pragma: no cover — older ``mcp`` within the pinned range
    from mcp.client.streamable_http import streamablehttp_client as _http_transport

    _OWNS_HTTP_CLIENT = False

if TYPE_CHECKING:
    from agentkit.kernel.protocols import Ctx


# ─────────────────────────────────────────────────────────────────────────────
# Transport configs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StdioServer:
    """Configuration for a stdio-transport MCP server.

    ``command`` is the executable the client will spawn (e.g. ``uvx``,
    ``npx``, or a compiled binary). ``args`` are the argv tail. ``env``
    overlays on ``os.environ`` inside the subprocess. ``cwd`` becomes
    the subprocess working directory.
    """

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | Path | None = None

    def __post_init__(self) -> None:
        """Frozen in name only until this ran. `env` is handed to a subprocess; a later in-place
        edit would change what the NEXT spawn inherits while the config object still reads as
        the one that was reviewed."""
        object.__setattr__(self, "env", deep_freeze(self.env))

    def __hash__(self) -> int:
        """Identity is the process this spawns: command, args, cwd. `env` is excluded
        (see `_frozen.py` for why a frozen payload is still unhashable). `cwd` is
        normalised through `str` because it may be a `Path`."""
        return hash((self.command, self.args, str(self.cwd) if self.cwd is not None else None))


@dataclass(frozen=True)
class StreamableHttpServer:
    """Configuration for the modern streamable HTTP MCP transport.

    ``url`` points at the server endpoint; ``headers`` may carry auth
    tokens. ``timeout_s`` bounds the request timeout at the transport
    layer.
    """

    url: str
    headers: dict[str, str] | None = None
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Frozen in name only until this ran. `headers` can carry credentials; a value that can
        be edited after review is one that can be edited after the review that approved it."""
        object.__setattr__(self, "headers", deep_freeze(self.headers))

    def __hash__(self) -> int:
        """Identity is the endpoint: url and timeout. `headers` are excluded twice over —
        unhashable (see `_frozen.py`), and they can carry credentials, which have no
        business influencing a bucket."""
        return hash((self.url, self.timeout_s))


# ─────────────────────────────────────────────────────────────────────────────
# Small in-memory TTL cache for list_tools / list_resources
#
# The MCP spec allows servers to advertise a cache TTL on list responses;
# the current ``mcp>=1.28,<2`` Python client doesn't yet surface that field
# on ``ListToolsResult`` / ``ListResourcesResult``, so we apply a fixed
# ``_DEFAULT_TTL_S`` on the client side. Callers who want an unbounded
# cache lifetime can query with ``force_refresh=True`` to bypass it.
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TTL_S = 30.0


@dataclass(slots=True)
class _TTLEntry:
    value: Any
    expires_at: float

    def is_fresh(self, now: float) -> bool:
        return now < self.expires_at


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MCPClient:
    """Async context manager wrapping an ``mcp.ClientSession``.

    Enter the context to open the transport + initialize the session; exit
    to tear everything down in reverse order. The wrapped session is
    accessible via ``client.session`` after entry.
    """

    server: StdioServer | StreamableHttpServer
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _session: ClientSession | None = field(default=None, init=False, repr=False)
    _tools_cache: _TTLEntry | None = field(default=None, init=False, repr=False)
    _resources_cache: _TTLEntry | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> MCPClient:
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        try:
            if isinstance(self.server, StdioServer):
                params = StdioServerParameters(
                    command=self.server.command,
                    args=list(self.server.args),
                    env=self.server.env,
                    cwd=self.server.cwd,
                )
                transport = await self._exit_stack.enter_async_context(stdio_client(params))
                read_stream, write_stream = transport
            elif isinstance(self.server, StreamableHttpServer):
                if _OWNS_HTTP_CLIENT:
                    import httpx

                    # The transport no longer takes headers or a timeout, so
                    # they move onto a client we own. It is entered on the exit
                    # stack ABOVE the transport, so unwinding tears the
                    # transport down FIRST and it never writes into a closed
                    # client.
                    http_client = await self._exit_stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=self.server.headers or {},
                            timeout=self.server.timeout_s,
                        )
                    )
                    opened = _http_transport(self.server.url, http_client=http_client)
                else:  # pragma: no cover — older ``mcp`` within the pinned range
                    opened = _http_transport(
                        url=self.server.url,
                        headers=self.server.headers,
                        timeout=self.server.timeout_s,
                    )
                transport = await self._exit_stack.enter_async_context(opened)
                # streamablehttp yields a 3-tuple (read, write, get_session_id);
                # only the first two feed ClientSession.
                read_stream, write_stream, *_ = transport
            else:  # pragma: no cover — unreachable, mypy exhausts the union
                raise TypeError(f"unknown MCP server config: {type(self.server).__name__}")
            session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._session = session
        except BaseException:
            # Roll back the partially-opened stack so a failed enter doesn't
            # leak the subprocess / http transport.
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        stack = self._exit_stack
        self._exit_stack = None
        self._session = None
        self._tools_cache = None
        self._resources_cache = None
        if stack is not None:
            await stack.__aexit__(exc_type, exc, tb)

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPClient not entered — use `async with MCPClient(...) as c:`")
        return self._session

    # ---- tool surface ---------------------------------------------------------------------

    async def list_tools(self, *, force_refresh: bool = False) -> list[mcp_types.Tool]:
        """Return the server's advertised tools, honoring a short client-side TTL cache.

        The cache is only trusted while the session is live — after
        ``__aexit__`` we always fall through to ``self.session``, which
        raises ``RuntimeError("MCPClient not entered")``. This prevents
        the cache from handing back tool metadata after the transport
        has been torn down.
        """
        now = monotonic()
        if (
            not force_refresh
            and self._session is not None
            and self._tools_cache is not None
            and self._tools_cache.is_fresh(now)
        ):
            return list(self._tools_cache.value)
        result = await self.session.list_tools()
        tools = list(result.tools)
        self._tools_cache = _TTLEntry(value=tools, expires_at=now + _DEFAULT_TTL_S)
        return list(tools)

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        ctx: Ctx | None = None,
    ) -> mcp_types.CallToolResult:
        """Invoke a tool by name; cooperative-cancellation-aware.

        ``ctx.check_cancelled()`` fires before dispatch so a run cancelled
        while queued never reaches the server. Mid-call cancellation
        happens at the transport level when the client session is closed
        (the caller's ``async with MCPClient(...)`` exit).
        """
        if ctx is not None:
            ctx.check_cancelled()
        return await self.session.call_tool(name, args)

    # ---- resource surface -----------------------------------------------------------------

    async def list_resources(self, *, force_refresh: bool = False) -> list[mcp_types.Resource]:
        # Same defensive check as list_tools — the cache is trusted only
        # while the session is live.
        now = monotonic()
        if (
            not force_refresh
            and self._session is not None
            and self._resources_cache is not None
            and self._resources_cache.is_fresh(now)
        ):
            return list(self._resources_cache.value)
        result = await self.session.list_resources()
        resources = list(result.resources)
        self._resources_cache = _TTLEntry(value=resources, expires_at=now + _DEFAULT_TTL_S)
        return list(resources)

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI. Concatenates all text contents; drops
        binary blobs (base64-encoding those is the caller's concern when
        it matters — this method returns the human/model-facing text)."""
        # pydantic AnyUrl accepts str; the SDK coerces internally.
        result = await self.session.read_resource(uri)  # type: ignore[arg-type]
        parts: list[str] = []
        for content in result.contents:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    # ---- prompt surface -------------------------------------------------------------------

    async def list_prompts(self) -> list[mcp_types.Prompt]:
        result = await self.session.list_prompts()
        return list(result.prompts)

    async def get_prompt(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Fetch a rendered prompt from the server.

        MCP prompts are returned as a list of PromptMessage; we flatten
        the ``TextContent`` blocks across every message into a single
        string (role information is dropped since agentkit's ``Prompt``
        is role-less).
        """
        str_args = {k: str(v) for k, v in (args or {}).items()}
        result = await self.session.get_prompt(name, str_args)
        parts: list[str] = []
        for msg in result.messages:
            content = msg.content
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n\n".join(parts)


__all__ = ["MCPClient", "StdioServer", "StreamableHttpServer"]
