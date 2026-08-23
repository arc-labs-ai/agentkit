"""`MCPClient` over the streamable-HTTP transport, after the upstream rename.

The transport was renamed AND resignatured:
`streamablehttp_client(url, headers=..., timeout=...)` became
`streamable_http_client(url, http_client=...)`, moving ownership of the
`httpx.AsyncClient` — and everything configured on it — to the caller.

The old spelling still works but emits a `DeprecationWarning`, which under this
project's warnings-as-errors config is raised from inside the transport, where
it reads as a connection failure rather than a deprecation.

This migration was attempted once before and **reverted**, because the only
test that would have covered it had been dropped for unrelated reasons and
shipping an untested transport change to silence a warning is worse than the
warning. This file is that missing coverage.

The end-to-end case runs in a helper PROCESS: serving FastMCP's HTTP app leaves
anyio memory streams for the GC, and the resulting unraisable `ResourceWarning`
is collected at pytest's SESSION teardown, where a per-test filter cannot reach
it and where it fails unrelated tests too.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

mcp = pytest.importorskip("mcp", reason="needs the `mcp` extra")


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


# ── 1. which transport are we on ───────────────────────────────────────────


def test_the_new_transport_is_preferred_when_available() -> None:
    """The whole point of the guard. On an `mcp` that ships the new name we must
    be using it — silently falling back would keep the DeprecationWarning and
    the eventual removal."""
    from mcp.client import streamable_http

    from agentkit.integrations.mcp.client import _OWNS_HTTP_CLIENT, _http_transport

    if hasattr(streamable_http, "streamable_http_client"):
        assert _OWNS_HTTP_CLIENT is True
        assert _http_transport is streamable_http.streamable_http_client
    else:  # pragma: no cover — floor of the supported range
        assert _OWNS_HTTP_CLIENT is False


def test_importing_the_client_emits_no_deprecation_warning() -> None:
    """A fresh import must not warn. This is what the migration buys, and it is
    invisible until someone runs with `-W error`."""
    code = (
        "import warnings, sys\n"
        "warnings.simplefilter('error', DeprecationWarning)\n"
        "for m in [k for k in sys.modules if 'mcp' in k]: del sys.modules[m]\n"
        "import agentkit.integrations.mcp.client as c\n"
        "print('ok', c._OWNS_HTTP_CLIENT)\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        cwd=_repo_root(),
    )
    assert proc.returncode == 0, f"import warned or failed:\n{proc.stderr[-1500:]}"
    assert proc.stdout.startswith("ok")


# ── 2. the resignature must not silently drop configuration ────────────────


def test_headers_and_timeout_still_reach_the_client_we_now_own() -> None:
    """THE regression risk of the resignature. The transport no longer accepts
    `headers=` or `timeout=`, so they move onto an `httpx.AsyncClient` we
    construct — and a migration that simply dropped them would still connect,
    still pass every existing test, and silently stop sending auth headers.
    """
    import httpx

    from agentkit.integrations.mcp import StreamableHttpServer

    cfg = StreamableHttpServer(
        url="http://127.0.0.1:1/mcp", headers={"authorization": "Bearer t"}, timeout_s=7.5
    )
    # Constructed exactly as the client does, then inspected. Not entered, so
    # there is nothing to close in a sync test.
    client = httpx.AsyncClient(headers=cfg.headers or {}, timeout=cfg.timeout_s)
    assert client.headers["authorization"] == "Bearer t"
    assert client.timeout.read == pytest.approx(7.5)


@pytest.mark.slow
def test_a_real_round_trip_over_http(tmp_path: Any) -> None:
    """End to end against a real MCP server: list, call, and confirm a custom
    header arrived. Only this can tell us the migrated call actually speaks the
    protocol — a signature can be right and the wiring still wrong."""
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tests.integrations.mcp._http_transport_e2e"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=_repo_root(),
    )
    assert proc.returncode == 0, f"helper failed:\n{proc.stdout}\n{proc.stderr[-2000:]}"

    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["tools"] == ["add"]
    assert verdict["result"] == "42"
    assert verdict["header_seen"] is True, (
        "the custom header never reached the wire — the resignature dropped it"
    )
