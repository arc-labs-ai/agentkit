"""The credential in front of agentkit's MCP *servers*.

Both servers in this package — ``ApprovalServer`` and ``serve_registry`` — used
to bind ``127.0.0.1`` with nothing in front of them, and the containment
argument was "it is only loopback". That argument holds exactly as long as
nothing untrusted shares the host, and the whole point of
``ClaudeCliCognition`` is that the CLI runs ``Bash``: a build, a package
install, a test suite out of a repository the agent was pointed at, inside the
same network namespace as the server holding the agent's own tools. The trust
boundary loopback assumes is precisely the boundary the tool set is designed to
cross.

``hook_settings`` next door already does the better thing — an ``AF_UNIX``
socket in a ``0700`` directory, with the filesystem as the authorisation check
— and it is the model these tests hold the MCP servers to, adapted to the one
mechanism the Claude CLI can actually address. Measured against ``claude``
2.1.251: ``--transport`` accepts only ``stdio, sse, http`` (no unix option),
``unixSocket`` is not an ``--mcp-config`` key, and a ``headers`` entry in the
config document DOES arrive on the wire — the ``Authorization`` header was
observed on the CLI's very first ``server/discover`` probe. So the fence is a
generated bearer token on the loopback listener.

Three properties carry the weight here and each has its own section below:

* the credential is GENERATED, never supplied — a caller-supplied token is a
  token that becomes a constant in somebody's config file;
* a rejected caller is a TRANSPORT-level rejection, so the request never
  becomes a tool result the model could read and route around;
* asking for a credential where none can be provided RAISES at wiring time,
  because a guard that silently degrades to unauthenticated is the exact shape
  of bug this package refuses everywhere else.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import secrets
import socket
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="needs the `mcp` extra")

from agentkit.integrations.mcp import ApprovalServer, serve_registry  # noqa: E402
from agentkit.integrations.mcp import _transport as transport_mod  # noqa: E402
from agentkit.integrations.mcp._transport import (  # noqa: E402
    LoopbackMcpTransport,
    require_bearer,
)
from agentkit.testing import make_test_ctx  # noqa: E402
from agentkit.tools import ToolRegistry, tool  # noqa: E402


@tool(side_effecting=False)
def run_check(name: str) -> str:
    """Run the named check and report the verdict. The one tool these tests
    serve; identical in shape to the one the sibling suite uses."""
    return f"{name}:ok"


def _spec(**kw: Any) -> Any:
    kw.setdefault("ctx", make_test_ctx())
    kw.setdefault("name", "engine")
    return serve_registry(ToolRegistry.from_tools([run_check]), **kw)


def _entry(spec: Any) -> dict[str, Any]:
    """The server entry as the CLI will read it — out of the FILE, not off the
    object. A spec whose document disagrees with its own attributes passes
    every attribute assertion and still fails the binary."""
    document: dict[str, Any] = json.loads(spec.config_path.read_text())
    return dict(document["mcpServers"][spec.name])


# ── a recording inner app, so "never reached" is observable ────────────────


class _Sentinel:
    """An ASGI app that records that it was entered and answers 200.

    The point of every rejection test below is that this object stays
    untouched: a transport-level refusal must not reach the MCP session
    manager at all, because anything the session manager answers is a
    tool-shaped result the model would be shown and could try to route around.
    """

    def __init__(self) -> None:
        self.scopes: list[dict[str, Any]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})


async def _request(
    app: Any,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    query: bytes = b"",
) -> list[dict[str, Any]]:
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": query,
        "root_path": "",
        "headers": list(headers or []),
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8000),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _status(sent: list[dict[str, Any]]) -> int:
    return int(next(m for m in sent if m["type"] == "http.response.start")["status"])


def _headers(sent: list[dict[str, Any]]) -> dict[bytes, bytes]:
    start = next(m for m in sent if m["type"] == "http.response.start")
    return {k.lower(): v for k, v in start["headers"]}


def _body(sent: list[dict[str, Any]]) -> str:
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()


# ── 1. the credential is generated, never supplied ─────────────────────────


def test_a_served_registry_is_authenticated_without_the_caller_asking() -> None:
    """Unauthenticated had to stop being the DEFAULT or stop being SILENT, and
    the shipped state was neither: a warning in a module docstring is read by
    the person maintaining the module, not by the person wiring a worker. The
    default moved."""
    spec = _spec()

    assert spec.auth == "bearer"
    header = _entry(spec)["headers"]["Authorization"]
    assert header.startswith("Bearer ")
    assert len(header.removeprefix("Bearer ")) >= 32


def test_there_is_no_seam_through_which_a_caller_can_supply_the_token() -> None:
    """A caller-supplied token is a token that will be a constant in somebody's
    config file, checked in, and shared between every worker on the fleet. The
    absence of the parameter is the enforcement — a runtime check could be
    argued past, a missing keyword cannot."""
    for fn in (serve_registry, LoopbackMcpTransport.__init__):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"token", "secret", "bearer", "credential"}, fn


def test_two_servers_in_one_process_get_independent_credentials() -> None:
    """One process routinely runs several of these — an approval gate and a
    served registry, or two registries for two agents. A shared token would
    make each one's fence the union of both, so reaching either server would
    be enough to reach the other."""
    one, two = _spec(), _spec(name="other")

    assert _entry(one)["headers"] != _entry(two)["headers"]


def test_the_same_spec_keeps_its_credential_across_stop_and_start() -> None:
    """DECIDED, and the decision is written down: the token is generated once
    per spec and does NOT rotate on restart.

    ``start()`` and ``stop()`` are both documented idempotent, and re-entering
    ``async with spec`` is a supported retry shape. A rotated token would
    invalidate a config document the caller has already read — ``cli_kwargs()``
    hands out a PATH, and a CLI process holding the old header would fail with
    "MCP server failed to connect" naming nothing. The credential's lifetime is
    the spec's lifetime, which is the run's.
    """
    spec = _spec()
    first = _entry(spec)

    async def cycle() -> None:
        async with spec:
            pass
        async with spec:
            pass

    asyncio.run(cycle())
    spec._write_config()
    assert _entry(spec) == first


# ── 2. unauthenticated survives, but only by name ──────────────────────────


def test_the_old_loopback_behaviour_is_still_available_and_now_has_a_name() -> None:
    """Loopback-with-nothing-in-front is a defensible choice on a host nobody
    else shares. It stops being defensible when it is what you get by saying
    nothing, so it kept working and acquired a spelling."""
    spec = _spec(auth="none")

    assert spec.auth == "none"
    assert "headers" not in _entry(spec)
    assert spec.auth_headers == {}


def test_a_misspelt_auth_mode_is_refused_rather_than_read_as_unauthenticated() -> None:
    """The same fail-closed reflex ``ApprovalServer.autonomy`` already has. A
    string that is not a mode must not fall through to the permissive branch:
    ``auth="bearer "`` or ``auth="none "`` out of a YAML file would otherwise
    take the fence down without a word anywhere."""
    with pytest.raises(ValueError, match="auth="):
        _spec(auth="bear")


# ── 3. refusal, never fallback ─────────────────────────────────────────────


def test_asking_for_a_credential_on_stdio_raises_at_wiring_time() -> None:
    """A stdio server has no request to carry a header: the CLI spawned the
    process and owns both ends of the pipe, which IS the boundary. Quietly
    handing back an unauthenticated spec would be a guard that looks wired and
    enforces nothing — the exact shape this package refuses everywhere else —
    so it raises where the wiring is, and names what to do instead."""
    with pytest.raises(ValueError, match="stdio") as exc:
        _spec(transport="stdio", command=("python", "-m", "x"), auth="bearer")

    assert "auth='none'" in str(exc.value), "the error must name the way out"


def test_stdio_without_asking_is_unauthenticated_because_the_pipe_is_the_fence() -> None:
    """The default resolves per transport rather than being one constant: over
    stdio there is nothing to authenticate to, and demanding an explicit
    ``auth='none'`` there would be ceremony for a boundary the OS already
    holds."""
    spec = _spec(transport="stdio", command=("python", "-m", "x"))

    assert spec.auth == "none"
    assert spec.auth_headers == {}


# ── 4. a rejected CALLER is a transport rejection, not a tool error ─────────


@pytest.mark.asyncio
async def test_a_properly_addressed_call_passes_the_fence_untouched() -> None:
    inner = _Sentinel()
    app = require_bearer(inner, "s3kr1t")

    sent = await _request(app, headers=[(b"authorization", b"Bearer s3kr1t")])

    assert _status(sent) == 200
    assert _body(sent) == "inner"
    assert len(inner.scopes) == 1


@pytest.mark.asyncio
async def test_a_call_with_no_credential_never_reaches_the_mcp_app() -> None:
    """401 at the ASGI layer, before the session manager exists for this
    request. A bad CALLER is not a bad CALL: reflecting it as a tool result —
    the way ``ToolArgumentError`` is deliberately reflected — would teach the
    model that the security boundary is a thing to work around, and it is very
    good at working around things it is shown."""
    inner = _Sentinel()
    app = require_bearer(inner, "s3kr1t")

    sent = await _request(app)

    assert _status(sent) == 401
    assert _headers(sent)[b"www-authenticate"].startswith(b"Bearer")
    assert inner.scopes == [], "the request reached the tool app"
    # Parseable, because the CLI reports a failed MCP connection as "MCP server
    # failed to connect" and nothing else — this body is the only place the
    # operator can find out the credential is already in their config document,
    # and a body no client can parse reads as a broken server rather than a
    # refusal.
    assert json.loads(_body(sent))["error"] == "unauthorized"
    assert "auth_headers" in json.loads(_body(sent))["detail"]


@pytest.mark.asyncio
async def test_a_wrong_credential_is_refused_the_same_way_as_a_missing_one() -> None:
    inner = _Sentinel()
    app = require_bearer(inner, "s3kr1t")

    for header in (b"Bearer wrong", b"Bearer ", b"Basic s3kr1t", b"s3kr1t", b""):
        sent = await _request(app, headers=[(b"authorization", header)])
        assert _status(sent) == 401, header

    assert inner.scopes == []


@pytest.mark.asyncio
async def test_a_credential_in_the_query_string_is_refused_even_when_correct() -> None:
    """URLs get logged — access logs, proxy logs, shell history, an exception
    with the request line in it. A query-string credential is refused on the
    NAME of the parameter rather than on its value, so a caller who reaches for
    the pattern is stopped the first time rather than the first time it leaks.
    400 rather than 401: the credential is not missing, the request is
    malformed."""
    inner = _Sentinel()
    app = require_bearer(inner, "s3kr1t")

    sent = await _request(
        app,
        headers=[(b"authorization", b"Bearer s3kr1t")],
        query=b"access_token=s3kr1t",
    )

    assert _status(sent) == 400
    assert json.loads(_body(sent))["error"] == "credential_in_url"
    assert "Authorization" in json.loads(_body(sent))["detail"]
    assert inner.scopes == []


@pytest.mark.asyncio
async def test_the_credential_is_compared_in_constant_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``==`` on ``str``/``bytes`` short-circuits at the first differing byte.
    Over loopback the attacker owns the timing — no network jitter, no
    scheduling noise it cannot average out, and it can issue the probes as fast
    as the loop will accept them — which turns a 43-character secret into 43
    sequential searches over a small alphabet. ``secrets.compare_digest`` is
    the only comparison allowed to touch the token."""
    seen: list[tuple[Any, Any]] = []
    real = secrets.compare_digest

    def spy(a: Any, b: Any) -> bool:
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(secrets, "compare_digest", spy)

    app = require_bearer(_Sentinel(), "s3kr1t")
    sent = await _request(app, headers=[(b"authorization", b"Bearer s3kr1t")])

    assert _status(sent) == 200
    assert seen, "the token was compared without secrets.compare_digest"


@pytest.mark.asyncio
async def test_the_lifespan_scope_passes_through_the_fence() -> None:
    """``StreamableHTTPSessionManager`` starts its task group in the lifespan.
    A fence that answered 401 to every scope type would leave the manager
    un-started and every authenticated call would then fail inside it — an
    outage that reads exactly like a credential bug and is not one."""
    seen: list[str] = []

    class _Lifespan:
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            seen.append(scope["type"])

    await require_bearer(_Lifespan(), "s3kr1t")({"type": "lifespan"}, None, None)

    assert seen == ["lifespan"]


# ── 5. the config file is a credential file now ────────────────────────────


def test_the_config_file_carrying_a_credential_is_still_owner_only() -> None:
    """It was 0600 when it only named a loopback endpoint, on the argument that
    there is no reason to publish the address. Now it carries the token, so the
    mode stopped being hygiene and became the fence's other half — the CLI
    reads the token out of this file, and anything that can read this file has
    the token."""
    spec = _spec()

    assert spec.config_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_stop_removes_the_config_even_when_the_listener_fails_to_stop() -> None:
    """The config and the listener have to go TOGETHER. A config that outlives
    its listener used to be a stale address; now it is a live credential lying
    on disk naming a port something else may bind next."""
    spec = _spec()
    path = spec.config_path

    async def boom() -> None:
        raise OSError("the listener would not let go")

    assert spec._transport is not None
    spec._transport.stop = boom  # type: ignore[method-assign]

    with pytest.raises(OSError, match="would not let go"):
        await spec.stop()

    assert not path.exists(), "a credential outlived the listener it names"


def test_a_caller_supplied_config_path_still_has_its_credential_removed(
    tmp_path: Path,
) -> None:
    """The directory is the caller's and stays. The FILE is ours — we wrote it,
    and what we wrote into it is a secret — so it goes on ``stop()`` like any
    other. Leaving a token behind in a directory the caller chose (and may well
    have chosen because it persists) is the leak this closes.

    Synchronous, with the filesystem assertions outside the coroutine: blocking
    ``pathlib`` calls on the event loop are what ruff's ASYNC240 exists to
    catch, and the suite is held to it too.
    """
    path = tmp_path / "engine.mcp.json"
    spec = _spec(config_path=path)
    assert "headers" in _entry(spec)

    async def cycle() -> None:
        async with spec:
            pass

    asyncio.run(cycle())

    assert not path.exists()
    assert tmp_path.is_dir(), "the caller's directory is not ours to remove"


# ── 6. the approval server gets the same fence ─────────────────────────────


@pytest.mark.asyncio
async def test_the_approval_server_is_authenticated_by_default() -> None:
    """Worse here than for a served registry in one way and better in another:
    an approval server only answers prompts, but a verdict reachable by
    anything sharing the namespace is a verdict no longer produced only by the
    reviewer who was supposed to produce it. It does not have to be attacked to
    be worthless; it only has to be reachable."""
    server = ApprovalServer(autonomy="auto")

    assert server.auth == "bearer"
    async with server:
        document = json.loads(server.mcp_config)
        entry = document["mcpServers"]["agentkit_approvals"]
        assert entry["headers"]["Authorization"].startswith("Bearer ")
        assert server.cli_kwargs()["mcp_config"] == (server.mcp_config,)


@pytest.mark.asyncio
async def test_the_approval_server_keeps_its_credential_across_a_restart() -> None:
    """Same decision as the served registry, and it has to be the same one:
    ``start()`` used to build a fresh listener each time, so a token living on
    the listener would rotate every restart while ``mcp_config`` looked
    unchanged to the caller reading it."""
    server = ApprovalServer(autonomy="auto")
    async with server:
        first = json.loads(server.mcp_config)["mcpServers"]["agentkit_approvals"]["headers"]
    async with server:
        again = json.loads(server.mcp_config)["mcpServers"]["agentkit_approvals"]["headers"]

    assert first == again


@pytest.mark.asyncio
async def test_the_approval_server_can_be_unauthenticated_by_name() -> None:
    server = ApprovalServer(autonomy="auto", auth="none")
    async with server:
        assert "headers" not in json.loads(server.mcp_config)["mcpServers"]["agentkit_approvals"]


def test_a_misspelt_auth_mode_on_the_approval_server_raises_at_construction() -> None:
    """Construction, not first prompt. Every other refusal in ``__post_init__``
    is there because the alternative is discovering the misconfiguration once a
    run is in flight and a person is waiting."""
    with pytest.raises(ValueError, match="auth="):
        ApprovalServer(autonomy="auto", auth="none ")  # type: ignore[arg-type]


# ── 7. what a caller uses instead of assembling a header ───────────────────


def test_auth_headers_is_the_seam_for_a_client_that_is_not_the_cli() -> None:
    """``cli_kwargs()`` already carries everything the CLI needs, because a
    caller assembling MCP JSON by hand is the gap this module exists to close.
    A non-CLI client — agentkit's own ``StreamableHttpServer``, say — needs the
    same thing one layer down, and the answer is a mapping rather than the raw
    token: a bare ``spec.token`` invites string concatenation into a URL, which
    is the one placement the fence refuses."""
    spec = _spec()

    assert not hasattr(spec, "token")
    assert spec.auth_headers == _entry(spec)["headers"]
    assert list(spec.auth_headers) == ["Authorization"]


def test_the_listener_does_not_print_its_own_credential() -> None:
    """A ``repr`` lands in tracebacks, log lines and pytest failure output. A
    transport that renders its own token there defeats the 0600 on the config
    file the moment anything raises."""
    listener = LoopbackMcpTransport(authenticated=True)

    assert listener.token is not None
    assert listener.token not in repr(listener)


def test_the_shared_listener_defaults_to_authenticated() -> None:
    """The default in the shared primitive is where a third caller's fence
    comes from. Defaulting to ``False`` here would mean the next server built
    on this class is unauthenticated unless somebody remembered — which is how
    the two current ones got that way."""
    assert transport_mod.LoopbackMcpTransport().token is not None


# ── 8. the fence is on the listener, so neither server can forget it ───────


# ── 9. review pass: gaps the first suite left open ─────────────────────────
#
# Everything above is the behaviour the change set out to build. What follows
# came from attacking it: each of these was a mutant that survived the suite as
# shipped, or a fail-open found by driving hostile scopes and races at the code
# and watching what it actually did.


@pytest.mark.asyncio
async def test_a_websocket_scope_is_refused_rather_than_handed_to_the_app() -> None:
    """The fence used to pass every scope that was not ``http`` straight
    through, which made "not an HTTP request" mean "not this fence's problem".

    It is, because uvicorn is configured with ``ws="auto"`` and that turns
    websocket upgrades ON as soon as ``websockets`` or ``wsproto`` is
    importable. Neither is an agentkit dependency, so whether this fence held
    depended on an unrelated package in the caller's environment. Measured with
    ``websockets`` on the path: an unauthenticated upgrade against an
    AUTHENTICATED listener answered ``HTTP/1.1 101 Switching Protocols`` and
    the scope reached the served app carrying no credential at all.

    It was survivable only because the app behind it happens to have no
    websocket route today — which puts the authorisation decision in the inner
    app's routing table rather than in the fence. ``LoopbackMcpTransport`` is
    documented as where a third server on this seam gets its fence from, and
    that server would have inherited none for this scope.
    """
    inner = _Sentinel()
    app = require_bearer(inner, "s3kr1t")
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    await app({"type": "websocket", "headers": [], "query_string": b""}, receive, send)

    assert inner.scopes == [], "an unauthenticated websocket reached the app"
    # A close BEFORE accept is the portable denial; uvicorn renders it as 403
    # and never completes the upgrade.
    assert sent == [{"type": "websocket.close", "code": 1008}]


@pytest.mark.asyncio
async def test_only_the_lifespan_scope_is_allow_listed_through_the_fence() -> None:
    """``lifespan`` passes by NAME, not by "everything except http".

    The distinction is the whole finding above: an allow-list of one is a fence
    that a new ASGI scope type cannot silently widen, and a deny-list of one is
    a fence that every new scope type walks around.
    """
    for scope_type in ("websocket", "http.disconnect", "", None, "HTTP"):
        inner = _Sentinel()
        app = require_bearer(inner, "s3kr1t")

        async def send(message: dict[str, Any]) -> None:
            return None

        async def receive() -> dict[str, Any]:
            return {"type": "x"}

        await app({"type": scope_type, "headers": [], "query_string": b""}, receive, send)

        assert inner.scopes == [], f"scope {scope_type!r} reached the app unauthenticated"


def test_an_empty_token_is_refused_rather_than_authenticating_everybody() -> None:
    """``compare_digest(b"", b"")`` is TRUE, so a fence built on an empty token
    admits anyone who sends ``Authorization: Bearer `` — an open door rather
    than a closed one with a bad key, and it logs nothing either way.

    ``start()`` guards its own call site, but this function is exported and the
    failure is silent, so the refusal belongs where the fence is built.
    """
    with pytest.raises(ValueError, match="empty token"):
        require_bearer(_Sentinel(), "")


@pytest.mark.asyncio
async def test_every_name_that_spells_a_credential_in_a_url_is_refused() -> None:
    """The suite proved ``access_token``; the other six names in the set were
    never exercised, and a mutant that shrank the set to ``{"access_token"}``
    passed the whole suite. The point of refusing on the NAME is to stop the
    PATTERN, which is only true if the pattern is more than one spelling."""
    inner = _Sentinel()
    app = require_bearer(inner, "s3kr1t")

    for name in ("access_token", "api_key", "apikey", "authorization", "bearer", "key", "token"):
        sent = await _request(
            app,
            headers=[(b"authorization", b"Bearer s3kr1t")],
            query=f"{name}=s3kr1t".encode(),
        )
        assert _status(sent) == 400, name
        assert json.loads(_body(sent))["error"] == "credential_in_url", name

    # Case-folded, and a bare flag with no value still names the pattern.
    for query in (b"TOKEN=s3kr1t", b"Api_Key=s3kr1t", b"token"):
        assert _status(await _request(app, query=query)) == 400, query

    assert inner.scopes == []


@pytest.mark.asyncio
async def test_only_the_first_authorization_header_is_allowed_to_decide() -> None:
    """Five lines of comment explain why the loop ``break``s, and nothing
    tested it: turning that ``break`` into a ``continue`` passed the entire
    suite.

    Two things go wrong without it. A caller can spray guesses in a single
    request — one connection, many candidate credentials, and a rate limit
    counting requests counts one. And what the server accepts starts depending
    on header ORDER, which nothing in an HTTP path is obliged to preserve, so
    the same request can be answered differently by two hops.
    """
    inner = _Sentinel()
    app = require_bearer(inner, "s3kr1t")

    sent = await _request(
        app,
        headers=[(b"authorization", b"Bearer wrong"), (b"authorization", b"Bearer s3kr1t")],
    )

    assert _status(sent) == 401, "a spray of guesses in one request found the token"
    assert inner.scopes == []


@pytest.mark.asyncio
async def test_the_refusal_does_not_hand_the_caller_the_credential_it_is_missing() -> None:
    """The 401 body exists to tell an operator where the token already is,
    because the CLI reports a failed MCP connection and nothing else. It must
    not tell them what the token IS.

    This is the most damaging regression the change could have, and nothing
    caught it: an f-string interpolating ``token`` into ``detail`` passed every
    test in the suite. A 401 body goes to whoever asked — including the caller
    that had no credential — and lands in their logs.
    """
    token = "s3kr1t-4bcdefgh"
    app = require_bearer(_Sentinel(), token)

    for headers in ([], [(b"authorization", b"Bearer wrong")]):
        sent = await _request(app, headers=headers)
        assert _status(sent) == 401
        assert token not in _body(sent), "the refusal published the credential"

    query_refusal = await _request(app, query=b"token=" + token.encode())
    assert _status(query_refusal) == 400
    assert token not in _body(query_refusal), "the refusal echoed the credential back"


def test_the_credential_is_never_on_disk_readable_by_anyone_but_its_owner(
    tmp_path: Path,
) -> None:
    """``write_text`` then ``chmod`` is a window, not an atom.

    ``write_text`` creates at ``0o666 & ~umask`` — 0644 under the common umask
    — so the live token sat on disk world-readable until the ``chmod`` landed.
    Measured rather than argued: a polling thread read the real credential out
    of the file at 0644, and it matched the running server's. The default
    ``config_path`` hides inside a 0700 ``mkdtemp``, but ``config_path=`` is a
    supported argument and ``stop()`` reasons explicitly about callers who
    point it at a directory that persists — which is exactly where the window
    is reachable. It reopened on every ``start()``, because the config is
    rewritten there.

    Asserting the FINAL mode cannot see this; only watching the file appear
    can. The mode is set on the descriptor now, before any bytes are written.
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    path = shared / "engine.mcp.json"

    modes: list[int] = []
    exposed: list[str] = []
    stop = threading.Event()

    def racer() -> None:
        """Another account on the host, polling for the file to appear."""
        while not stop.is_set():
            try:
                mode = stat.S_IMODE(path.stat().st_mode)
            except FileNotFoundError:
                continue
            modes.append(mode)
            if mode & 0o077:
                exposed.append(path.read_text())
                return

    watcher = threading.Thread(target=racer)
    watcher.start()
    try:
        spec = _spec(config_path=path)
        # Rewritten on every start(), so the window would reopen there too.
        spec._write_config()
    finally:
        stop.set()
        watcher.join(timeout=5.0)

    assert exposed == [], "the live credential was world-readable before the chmod"
    assert modes, "the racer never saw the file at all — the test proves nothing"
    assert all(m == 0o600 for m in modes), f"observed modes: {[oct(m) for m in modes]}"
    assert spec.auth_headers["Authorization"].startswith("Bearer ")


def test_a_config_file_left_over_at_a_loose_mode_is_narrowed_before_it_is_filled(
    tmp_path: Path,
) -> None:
    """``O_CREAT``'s mode argument applies only when the file is CREATED. A
    restart into a path that already exists — a previous run's file, or one an
    attacker pre-created in a directory they can write — would otherwise keep
    whatever mode it had and be filled with a fresh token anyway."""
    path = tmp_path / "engine.mcp.json"
    path.write_text("{}")
    path.chmod(0o666)

    spec = _spec(config_path=path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert spec.auth_headers["Authorization"] in path.read_text()


@pytest.mark.asyncio
async def test_reading_the_config_early_does_not_silently_freeze_the_port() -> None:
    """Making the listener outlive ``stop()`` also made it lazily CACHED, and a
    cached listener froze ``host``/``port`` at whatever they were the first time
    anything touched ``mcp_config``, ``auth_headers`` or ``cli_kwargs()``.

    Before the change ``start()`` built the listener fresh, so assigning
    ``server.port`` afterwards was honoured. After it, the assignment was
    ignored and the server bound an ephemeral port instead — silently, while
    the caller believed they had pinned one. Measured: asked for a specific
    port after reading ``mcp_config``, got an OS-chosen one.

    Re-syncing beats rebuilding: rebuilding would rotate the token out from
    under a config document the caller may already be holding, which is the
    thing the caching was introduced to prevent.
    """
    server = ApprovalServer(autonomy="auto")
    _ = server.mcp_config  # builds and caches the listener at port 0

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        wanted = int(probe.getsockname()[1])

    server.port = wanted
    async with server:
        assert server.port == wanted, "a port assigned after an early read was ignored"
        assert server.url.endswith(f":{wanted}/mcp")


@pytest.mark.asyncio
async def test_an_authenticated_listener_refuses_before_the_app_sees_anything() -> None:
    """The wrapping happens in ``start()`` rather than at each call site, so
    "did this server remember to authenticate" is not a question that can have
    two answers."""
    inner = _Sentinel()
    listener = LoopbackMcpTransport(authenticated=True)
    await listener.start(inner)
    try:
        assert listener.token is not None
        reader, writer = await asyncio.open_connection(listener.host, listener.port)
        writer.write(
            b"POST /mcp HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=10.0)
        writer.close()
    finally:
        await listener.stop()

    assert raw.startswith(b"HTTP/1.1 401")
    assert b"www-authenticate: Bearer" in raw
    # The lifespan scope DOES reach it — that is the pass-through the session
    # manager needs. What must not reach it is the request.
    assert [s["type"] for s in inner.scopes] == ["lifespan"]


@pytest.mark.asyncio
async def test_an_upgrade_request_does_not_get_a_socket_out_of_an_authenticated_listener() -> None:
    """The same refusal, over a real socket, for the scope that used to walk
    around it.

    In-process the fence now answers ``websocket.close``; what an attacker
    actually experiences is the status line, and it has to be a refusal rather
    than ``101 Switching Protocols``. Whether uvicorn even offers the upgrade
    depends on ``websockets``/``wsproto`` being importable — they are not
    agentkit dependencies, so this assertion is written to hold either way: the
    handshake must not complete, and the app must not see the scope.
    """
    inner = _Sentinel()
    listener = LoopbackMcpTransport(authenticated=True)
    await listener.start(inner)
    try:
        reader, writer = await asyncio.open_connection(listener.host, listener.port)
        writer.write(
            b"GET /mcp HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(4096), timeout=10.0)
        writer.close()
    finally:
        await listener.stop()

    assert not raw.startswith(b"HTTP/1.1 101"), "an unauthenticated caller got a socket"
    assert [s["type"] for s in inner.scopes] == ["lifespan"]
