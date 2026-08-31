"""Helper process: drive a stdio-served ``ToolRegistry`` with agentkit's own client.

Run as ``python -m tests.integrations.mcp._serve_registry_e2e``; prints one JSON
line describing what happened.

Two processes, both ours, and that is the point. The PARENT (``main`` below)
speaks MCP as a client via ``MCPClient(StdioServer(...))``; the CHILD (``--serve``,
spawned by that client exactly the way the Claude CLI would spawn it) serves the
registry with ``serve_registry_stdio``. Nothing else can confirm the stdio path
actually speaks the protocol — a signature can be right and the framing still
wrong, and the framing only exists once there is a real pipe between two real
processes.

Its own process rather than inline in the suite for the same reason
``_http_transport_e2e.py`` next door is: spawning and tearing down MCP
transports leaves anyio memory streams for the garbage collector, and this
project runs warnings-as-errors, so the finalisation surfaces as an unraisable
``ResourceWarning`` collected at pytest's SESSION teardown — where a per-test
filter cannot reach it, and where it also fails unrelated tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from agentkit.integrations.mcp import (
    MCPClient,
    StdioServer,
    StreamableHttpServer,
    serve_registry,
    serve_registry_stdio,
)
from agentkit.testing import make_test_ctx
from agentkit.tools import ToolRegistry, tool


@tool(side_effecting=False)
def run_check(name: str, strict: bool = False) -> str:
    """Run the named check and report whether it passed. The single tool the
    child serves, kept identical to the one the in-process tests use."""
    return f"{name}:{'strict' if strict else 'lax'}:ok"


def _registry() -> ToolRegistry:
    return ToolRegistry.from_tools([run_check])


async def _serve() -> None:
    await serve_registry_stdio(_registry(), name="engine", ctx=make_test_ctx())


def _text(result: Any) -> str:
    return "".join(b.text for b in result.content if getattr(b, "text", None) is not None)


def _flatten(exc: BaseException) -> list[str]:
    """Every leaf of a possibly-nested ``ExceptionGroup``, as text.

    ``anyio`` task groups wrap whatever the transport raised, and the wrapper's
    own ``str`` is "unhandled errors in a TaskGroup (1 sub-exception)" — which
    says nothing about the 401 that is the entire point of the assertion. The
    parent test needs to see the status code, not the shape of the plumbing it
    arrived through.
    """
    if isinstance(exc, BaseExceptionGroup):
        return [line for sub in exc.exceptions for line in _flatten(sub)]
    return [f"{type(exc).__name__}: {exc}"]


async def main() -> dict[str, Any]:
    server = StdioServer(command=sys.executable, args=("-m", __spec__.name, "--serve"))
    verdict: dict[str, Any] = {}
    async with MCPClient(server) as client:
        verdict["tools"] = [t.name for t in await client.list_tools()]
        verdict["ok"] = _text(await client.call_tool("run_check", {"name": "lint"}))

        # A bad call has to come back as a TOOL error over a real pipe, and the
        # session has to keep working afterwards — the two halves have to be
        # observed on the SAME connection or the claim is untested.
        bad = await client.call_tool("run_check", {"name": "lint", "verbose": True})
        verdict["bad_is_error"] = bool(bad.isError)
        verdict["bad_text"] = _text(bad)
        verdict["survived"] = _text(await client.call_tool("run_check", {"name": "after"}))

    verdict.update(await _over_http())
    return verdict


async def _over_http() -> dict[str, Any]:
    """The same registry over a real socket.

    The in-process tests drive the MCP ``Server`` through the in-memory
    transport, which pins the protocol but never touches the Starlette app, the
    session manager or uvicorn. Those are where a wrong mount path or a
    stateful/stateless mismatch lives, and each fails as "MCP server failed to
    connect" with nothing pointing at the cause.
    """
    spec = serve_registry(_registry(), name="engine", ctx=make_test_ctx())
    out: dict[str, Any] = {}
    async with spec:
        # The config file is what the CLI is handed, so read the URL and the
        # credential back OUT of it rather than off the object: a spec whose
        # file disagrees with its own attributes would pass every in-process
        # test and fail the CLI.
        entry = json.loads(spec.config_path.read_text())["mcpServers"]["engine"]
        url, headers = entry["url"], entry["headers"]
        async with MCPClient(
            StreamableHttpServer(url=url, headers=dict(headers), timeout_s=10.0)
        ) as client:
            out["http_tools"] = [t.name for t in await client.list_tools()]
            out["http_ok"] = _text(await client.call_tool("run_check", {"name": "http"}))
            bad = await client.call_tool("run_check", {"nope": 1})
            out["http_bad_is_error"] = bool(bad.isError)
            out["http_survived"] = _text(
                await client.call_tool("run_check", {"name": "later"})
            )
        out["http_calls_seen_before_refusal"] = spec.calls_seen

        # What a caller sees if they point a PLAIN client at an authenticated
        # server. It has to be an exception out of the transport rather than an
        # ``isError`` result, and ``calls_seen`` has to be unmoved afterwards —
        # that pair is the difference between "a bad caller" and "a bad call",
        # and only a real socket can tell them apart.
        try:
            async with MCPClient(StreamableHttpServer(url=url, timeout_s=10.0)) as blind:
                await blind.list_tools()
        except BaseException as exc:  # noqa: BLE001 — the TYPE is the finding
            out["unauthenticated_error"] = " | ".join(_flatten(exc))
        else:
            out["unauthenticated_error"] = ""
    out["http_calls_seen"] = spec.calls_seen
    return out


if __name__ == "__main__":
    if "--serve" in sys.argv:
        asyncio.run(_serve())
    else:
        print(json.dumps(asyncio.run(main())))
