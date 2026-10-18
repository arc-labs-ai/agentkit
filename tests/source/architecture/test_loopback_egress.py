"""Loopback host-tool traffic must never consult an ambient proxy.

WHY THIS IS AN ARCHITECTURE TEST
--------------------------------
`httpx` trusts `HTTP_PROXY` / `ALL_PROXY` by default and does NOT exempt
localhost. So a client built without `trust_env=False` sends a request for
`http://127.0.0.1:<port>/mcp` through whatever those variables name — and that
channel carries the bridge's bearer token in an `Authorization` header, plus
every tool argument and every tool result. Measured before the fix: `proxied=
True` with two mounts for exactly that URL.

The server side protects the same token properly — a `0600` `O_NOFOLLOW` write,
credential-first teardown, 401 bodies that never echo it — so an environment
variable undoing all of that is the shape of hole nobody notices until they
read a proxy log.

A per-call-site test cannot prevent the next one, because the next one is a
client nobody has written yet. This walks the MCP integration instead, so a new
`httpx.AsyncClient(...)` there fails here rather than in production. The scan is
syntactic on purpose: the point is to catch the SPELLING that is easy to omit.

Deliberately scoped to the MCP integration. Provider egress is a different
question — a corporate proxy to reach `api.anthropic.com` is legitimate — so
`frontiers/base.py` still trusts the environment, and that is a decision rather
than an oversight.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[3] / "agentkit"
LOOPBACK_AREA = PACKAGE / "integrations" / "mcp"


def _async_clients(root: pathlib.Path) -> list[tuple[str, int, bool]]:
    """Every `httpx.AsyncClient(...)` under `root`, and whether it pins `trust_env`."""
    found: list[tuple[str, int, bool]] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name != "AsyncClient":
                continue
            pinned = any(
                kw.arg == "trust_env" and isinstance(kw.value, ast.Constant) and kw.value.value is False
                for kw in node.keywords
            )
            try:
                label = str(path.relative_to(PACKAGE.parent))
            except ValueError:
                # The scan is callable on any tree, which is how its own
                # sharpness gets checked without mutating shared source.
                label = str(path)
            found.append((label, node.lineno, pinned))
    return found


def test_the_scan_finds_the_clients_it_is_meant_to_police() -> None:
    """A guard that silently matches nothing passes forever. If the construction
    moves or is wrapped in a helper, this fails and says to re-aim the scan
    rather than letting it rot into a no-op."""
    found = _async_clients(LOOPBACK_AREA)
    assert found, (
        "no httpx.AsyncClient construction found under integrations/mcp — has the client "
        "moved behind a helper? Re-aim this scan at it."
    )


def test_no_loopback_client_trusts_the_ambient_environment() -> None:
    """The invariant. A loopback client that trusts the environment will send the
    bridge's bearer token wherever `HTTP_PROXY` points."""
    offenders = [
        f"{path}:{line} builds httpx.AsyncClient without trust_env=False"
        for path, line, pinned in _async_clients(LOOPBACK_AREA)
        if not pinned
    ]
    assert not offenders, (
        "these clients talk to a loopback MCP server while honouring ambient proxy "
        "variables, so the channel's bearer token and every tool argument and result can "
        "be routed off-box by an environment variable:\n  " + "\n  ".join(offenders)
    )
