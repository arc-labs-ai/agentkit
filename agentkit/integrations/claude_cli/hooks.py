"""``hook_settings`` — make the tool middleware chain apply to the CLI's own tools.

``ClaudeCliCognition`` hands the whole agent loop to the ``claude`` binary, and
the binary runs ``Write`` / ``Edit`` / ``Bash`` / ``WebFetch`` itself. None of
that goes through the :class:`~agentkit.runtime.invoker.Invoker`, so **every**
tool-chain middleware stops applying — ``egress`` (default-deny URL checking),
``guard``, ``Guardrail.check_url`` (SSRF and allowlist), and anything else the
app put in that list. The chain is still wired, still constructed, still in the
app's config; it simply never runs. ``_charge_meters`` in the cognition says
the same thing about metering and calls it what it is: "how a documented safety
mechanism ends up doing nothing". Metering was then patched by hand. Nothing
else was.

The CLI's seam for this is ``PreToolUse`` hooks in a settings file, and
``ClaudeCliCognition`` already takes ``settings`` → ``--settings``. So::

    settings = hook_settings(
        middleware=tool_chain,      # the same list the Invoker would take
        ctx=ctx,
        tools=("Write", "Edit", "Bash"),
    )
    cognition = ClaudeCliCognition(settings=settings.path, ...)

The stated cost of this design is that "the refusal now lives in a generated
hook script, which is a second execution path to keep correct" — and a second
execution path that drifts from the first is exactly how this class of bug
arrives. Everything below is arranged so there is no second execution path.

**How the hook reaches the chain.** The generated script is a dumb pipe: it
reads the CLI's payload off stdin, writes it to a Unix socket, and prints
whatever comes back. It never imports agentkit and contains no policy. The
decision is made by ``chain(middleware, terminal)`` — the *same function over
the same objects* the ``Invoker`` composes — running in the parent process, on
the parent's event loop, against the live ``ctx``. There is one implementation
of the refusal, not two.

The alternative — a subprocess that re-enters Python and rebuilds the ``ctx``
— was measured against it. macOS, CPython 3.13, 25 calls per row, three runs;
medians vary with machine load, so the ranges are the honest form::

    the decision itself (socket round trip + chain, in process)   0.24 – 0.43 ms
    this hook, end to end, spawned the way the CLI spawns it        43 – 67 ms
    floor: the same shell + interpreter doing literally nothing     22 – 35 ms
    alternative: a subprocess that only IMPORTS agentkit           185 – 330 ms

So the decision costs a third of a millisecond and the socket adds nothing
measurable; what a hook costs is the process the CLI spawns, and this one lands
within ~20 ms of the floor for doing that at all. The subprocess design starts
at 4–6× the whole hook — before it has rebuilt a ``ctx``, and it cannot rebuild
a faithful one anyway, which is the next section.

**What is serialisable.** Nothing has to be. A ``ctx`` carries an ``Invoker``,
a store, a cancel token and live meters; a subprocess hook could carry none of
them and would have to reconstruct a lookalike, which is the drift. Because the
chain runs in the parent, the only things that cross the process boundary are
the tool name and its arguments (JSON — which is all the CLI has anyway) and an
allow/deny back.

What limits this mechanism is not serialisation, it is *when the hook fires*.
``PreToolUse`` runs **before** the CLI's tool and agentkit never sees a result,
so a middleware can only participate if its entire contribution happens in
``on_request``:

===================  ===========================================================
works                ``Egress`` (SSRF + allowlist — the ``url_arg`` mapping in
                     ``_NATIVE_TOOLS`` is what makes it check anything at all),
                     and any ``BaseMiddleware`` that overrides only
                     ``on_request`` and refuses by raising ``Blocked``.
accepted, but inert  ``SecurityMiddleware`` / ``security()``. It is accepted
                     here (it overrides only ``on_request``) and it does
                     nothing: its first line is ``if ctx.operation !=
                     Operation.MODEL_CALL: return``, and every call this
                     mechanism makes is a TOOL_CALL. It guards the *prompt*, on
                     the chat chain, which the CLI does not route through here
                     either. Handing the whole app chain over is still right —
                     this row exists so nobody reads a wired
                     ``SecurityMiddleware`` as CLI tool-call coverage. There is
                     no check that catches this class in general: "overrides
                     on_request but returns early for tool calls" is a property
                     of the body, not the signature.
cannot work          ``Audit`` (records in ``on_response``/``on_error``; there
                     is no result and no error to record — a ``PostToolUse``
                     hook is where that belongs), ``memoize`` / ``idempotent``
                     (a hit means "return the cached result instead of running",
                     and a ``PreToolUse`` hook can only allow or refuse, never
                     supply a result), ``retry`` / ``fallback`` (re-invoke
                     ``next``), ``tracing`` and every other raw ``(call, next)``
                     middleware (opaque — we cannot tell whether it touches the
                     result), ``MeterMiddleware`` (usage is the CLI's, and
                     ``_charge_meters`` already books it).
out of scope         ``RunPolicy`` — a pre-run check over a whole tool SET, not
                     a per-call middleware. Call it at wiring time as before.
===================  ===========================================================

Everything in the second row is a ``TypeError`` at generation time, not a
no-op at run time. That distinction is the entire reason this module exists: a
chain that silently does not apply is the bug being fixed, so a chain that
*cannot* apply must not be accepted quietly.

**Failure policy: fail closed, without taking the session down.** These are
different things and the requirement is both. Every failure — a middleware that
raises an ordinary exception, a chain that blows its deadline, a script that
cannot reach the socket — becomes a **deny for that one tool call**, delivered
in the CLI's own structured shape with exit code 0. The model is told why and
adapts; the session continues. The alternative spellings are all worse: a
non-zero exit the CLI does not recognise is a *non-blocking* error, which
prints and then **runs the tool**; and exit 2 blocks the call but reports it as
a broken hook rather than a decision.

The one failure this cannot cover honestly: if the generated script is deleted
or made unreadable mid-session, the CLI cannot execute the hook at all and
treats that as a non-blocking error — the tool runs. Nothing inside the hook
can fix a hook that never starts. The mitigations are that the script lives in
a ``0700`` directory this process created and removes, and that ``--settings``
scopes it to one session; the honest framing is that this is defence in depth
layered on ``--tools`` / ``--disallowed-tools`` / ``permission_mode``, not a
replacement for them.

**Timeouts are layered inside-out** so the innermost one answers first and can
still produce a reason: the chain gets ``timeout_s``; the script gets
``timeout_s + 2s`` — as a ``SIGALRM`` over the *whole* script, not just its
socket, because reading stdin is outside the socket's deadline and is exactly
where a wedged hook would hang; the CLI's own hook ``timeout`` gets
``timeout_s + 4s``. The CLI's must be the loosest, because a hook the CLI kills
is a non-blocking error and the tool runs — precisely the fail-open being
closed here, and the reason the script bounds itself rather than trusting the
layer above it to bound it kindly.

**The chain can only SUBTRACT.** A pass returns ``{}`` — no
``permissionDecision`` at all — and the CLI's normal permission flow decides.
Returning ``"allow"`` would *bypass* that flow: the CLI's own changelog carries
a fix for "PreToolUse auto-allow hooks bypassing tool restrictions". A
middleware chain is a refusal mechanism; letting it grant would make wiring one
in strictly more dangerous than not.

Unix-only: the transport is an ``AF_UNIX`` socket in a ``0700`` directory,
which makes the OS the authorisation check. (``ApprovalServer`` next door binds
loopback TCP with no authentication and documents loopback as the containment;
this is the same shape with a better fence, and it is available here because
the hook always runs on the same host as the parent.)

Nothing here needs an extra: it is ``asyncio`` and ``socket``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shlex
import shutil
import socket
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentkit.kernel.middleware import (
    BaseMiddleware,
    Blocked,
    Call,
    Handler,
    chain,
    collect_one,
)
from agentkit.kernel.types import ToolRequest

HOOK_EVENT = "PreToolUse"
SCRIPT_NAME = "pretooluse_hook.py"
SETTINGS_NAME = "settings.json"
SOCKET_NAME = "hook.sock"

# The CLI's native tools, mapped onto the two ``ToolRequest`` fields the tool
# chain actually reads. Without this table ``egress()`` would sit in the chain
# with ``url_arg=None`` and check nothing — inert in exactly the way
# ``Egress.__init__`` refuses to be at construction time. ``side_effecting``
# follows the CLI's own split between tools that change the world and tools
# that only look at it.
_NATIVE_TOOLS: dict[str, tuple[bool, str | None]] = {
    # name              side_effecting   url_arg
    "Task": (True, None),
    "Bash": (True, None),
    "BashOutput": (False, None),
    "KillShell": (True, None),
    "Glob": (False, None),
    "Grep": (False, None),
    "Read": (False, None),
    "Edit": (True, None),
    "Write": (True, None),
    "NotebookEdit": (True, None),
    "WebFetch": (False, "url"),
    "WebSearch": (False, None),
    "TodoWrite": (True, None),
    "ExitPlanMode": (False, None),
    "SlashCommand": (True, None),
}

# The matcher is a regex over the tool name, so a name has to be regex-inert.
# Anything else would either fail to compile inside the CLI or, worse, compile
# into a pattern that quietly matches the wrong set.
_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")

# ``sun_path`` is 104 bytes on macOS / 108 on Linux, and ``TMPDIR`` on macOS is
# already ~50 of them. Honour ``TMPDIR`` when the result fits and fall back to
# ``/tmp`` when it does not, rather than binding a path the OS truncates.
_SUN_PATH_MAX = 100

# The asyncio stream limit for one hook payload. Raised well above the 64 KiB
# default because ``tool_input`` carries ``Write``'s whole ``content`` and
# ``Edit``'s whole ``new_string``: a payload that overruns the limit raises,
# and a raise here is a DENY, so a stingy limit would refuse large-but-ordinary
# writes for a reason that has nothing to do with policy. Still bounded — the
# peer is a local script, but an unbounded read is a memory bug waiting for a
# bad one.
_PAYLOAD_LIMIT = 4 * 1024 * 1024


class _NeverRuns:
    """The ``ToolRequest.tool`` slot for a call agentkit will never execute.

    The CLI owns the execution; the chain's terminal here yields a sentinel
    instead of running anything. A plain ``None`` would make a middleware that
    reached for ``request.tool.run`` fail with ``AttributeError`` somewhere
    confusing, so the slot is filled with something that says what happened.
    """

    name = "claude-cli"

    async def run(self, arguments: Mapping[str, Any], ctx: Any) -> Any:
        raise RuntimeError(
            "a PreToolUse hook decides whether the Claude CLI may run this tool; "
            "agentkit never executes it, so there is nothing to run here"
        )


_NEVER_RUNS = _NeverRuns()

# What the chain's terminal yields. The terminal is never a real execution, so
# the value only has to be distinguishable in a debugger.
_NOT_EXECUTED = object()


@dataclass(frozen=True)
class HookDecision:
    """One hook round trip. Frozen for the same reason ``PolicyVerdict`` is: a
    decision that can be edited after the fact is not an audit trail."""

    tool: str
    allowed: bool
    reason: str = ""


def _deny(reason: str) -> dict[str, Any]:
    """The CLI's refusal shape. ``permissionDecisionReason`` reaches the MODEL,
    which can adapt — so a usable sentence ("that path is out of scope") is
    worth more than "denied"."""
    return {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT,
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _pass() -> dict[str, Any]:
    """No opinion — deliberately NOT ``permissionDecision: "allow"``. See the
    module docstring: the chain can only subtract."""
    return {}


# The generated script. Substitution is by ``str.replace`` on two sentinels
# rather than ``str.format`` / f-string, because the body is dense with braces
# and a formatting bug here would be a hook that silently fails to parse.
_HOOK_SCRIPT = '''\
# Generated per session by agentkit's hook_settings(). Do not edit; it is
# rewritten every run and deleted with its socket.
#
# A dumb pipe on purpose. It carries the CLI's PreToolUse payload to the
# process that owns the middleware chain and prints the answer back. No policy
# lives here, so there is no second implementation of the refusal to drift from
# the first. It imports only the standard library, and runs under -I -S so a
# PYTHONPATH or sitecustomize in the CLI's environment cannot inject code into
# the guard.
import json
import signal
import socket
import sys

SOCKET_PATH = __SOCKET_PATH__
TIMEOUT_S = __TIMEOUT_S__


def deny(reason):
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def too_slow(signum, frame):
    raise TimeoutError("no answer within %ss" % TIMEOUT_S)


answer = None
try:
    # One alarm over the WHOLE script, not just the socket: the socket timeout
    # does not cover reading stdin, and a hook that hangs is the single failure
    # the CLI resolves by killing it and RUNNING the tool. Bounding ourselves
    # means we refuse first.
    signal.signal(signal.SIGALRM, too_slow)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT_S)
    payload = sys.stdin.read()
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(TIMEOUT_S)
    buf = b""
    try:
        conn.connect(SOCKET_PATH)
        conn.sendall(payload.encode("utf-8") + b"\\n")
        conn.shutdown(socket.SHUT_WR)
        while not buf.endswith(b"\\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        conn.close()
    answer = buf.decode("utf-8").strip() or deny(
        "the tool chain closed the connection without answering"
    )
except BaseException as exc:  # a guard that crashes must still refuse
    answer = deny(
        "the agentkit hook could not reach the middleware chain (%s: %s)"
        % (type(exc).__name__, exc)
    )
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)

# Exit 0 with a structured decision, always. A non-zero exit code the CLI does
# not recognise is a NON-blocking error: it is printed and the tool RUNS.
sys.stdout.write(answer)
sys.exit(0)
'''


class HookSettings:
    """A live ``PreToolUse`` endpoint plus the settings file that points at it.

    Built by :func:`hook_settings`; the constructor is not the public seam.
    ``path`` is what goes to ``ClaudeCliCognition(settings=...)``. The object is
    an async context manager, and ``aclose()`` removes the whole directory —
    script, socket and settings together, so a stale settings file can never
    outlive the listener it names. (``ApprovalServer`` avoids a temp file
    entirely for that reason. It can, because ``--mcp-config`` takes inline
    JSON; ``--settings`` does too, but the hook *command* needs a script on
    disk regardless, so the directory exists either way and the settings file
    may as well live in it and die with it. The residual risk — a settings file
    that outlives its socket — resolves to a deny here, not to an unguarded
    call.)
    """

    def __init__(
        self,
        *,
        ctx: Any,
        middleware: Sequence[Any],
        tools: tuple[str, ...],
        directory: Path,
        sock: socket.socket,
        timeout_s: float,
    ) -> None:
        self.ctx = ctx
        self.middleware: tuple[Any, ...] = tuple(middleware)
        self.tools = tools
        self.timeout_s = timeout_s
        self._directory = directory
        # The composed chain, built ONCE and shared by every hook call — the
        # same object graph an ``Invoker`` would hold. ``_terminal`` stands in
        # for the execution the CLI is about to do itself.
        self._handler: Handler = chain(list(middleware), self._terminal)
        self._decisions: list[HookDecision] = []
        self._sock = sock
        self._server: asyncio.AbstractServer | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    # ---- paths ---------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """The ``--settings`` file. What the caller hands the cognition."""
        return self._directory / SETTINGS_NAME

    @property
    def script_path(self) -> Path:
        return self._directory / SCRIPT_NAME

    @property
    def socket_path(self) -> Path:
        return self._directory / SOCKET_NAME

    @property
    def decisions(self) -> tuple[HookDecision, ...]:
        """Every decision this endpoint made, oldest first. A run that reports
        zero either never touched a guarded tool or never reached the hook —
        worth being able to tell apart, and the only in-process evidence that
        the guard was live at all."""
        return tuple(self._decisions)

    # ---- lifecycle -----------------------------------------------------------------------

    async def __aenter__(self) -> HookSettings:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _serve(self) -> None:
        """Hand the already-bound socket to the loop. Idempotent."""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._serve_forever())

    async def _serve_forever(self) -> None:
        server = await asyncio.start_unix_server(
            self._handle, sock=self._sock, limit=_PAYLOAD_LIMIT
        )
        self._server = server
        async with server:
            await server.serve_forever()

    async def aclose(self) -> None:
        """Stop listening and remove the directory. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        with contextlib.suppress(Exception):
            self._sock.close()
        # The directory holds the script, the socket and the settings file.
        # Removing it as a unit is what makes "the settings file outlived its
        # listener" unrepresentable rather than merely unlikely.
        shutil.rmtree(self._directory, ignore_errors=True)

    def close(self) -> None:
        """Synchronous teardown for a caller that is not in a loop — removes
        the directory (so nothing stale is left on disk) and stops the
        listener the next time the loop runs. Prefer ``aclose()``."""
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._server is not None:
            self._server.close()
            self._server = None
        with contextlib.suppress(Exception):
            self._sock.close()
        shutil.rmtree(self._directory, ignore_errors=True)

    # ---- the wire ------------------------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """One connection = one tool call. Concurrency is the loop's problem
        and it is already solved: each connection gets its own task and its own
        ``Call``, and nothing is stashed on the shared ``ctx``, so N parallel
        CLI tool calls produce N independent decisions.

        A payload this cannot fully read is a **deny**, and the two ways it can
        be half-read are the interesting ones. Both used to be waved through:
        a missing ``tool_name`` became ``ToolRequest(name="")`` and a
        ``tool_input`` that was not an object became ``{}``. Neither raises, so
        both reached the chain as a *well-formed call about nothing* — and a
        guard keyed on the tool name (``if ctx.request.name == "Bash"``) or on
        an argument (``Egress`` reads ``arguments[url_arg]``, and a missing URL
        is not a blocked URL) then passed it. A pass emits no
        ``permissionDecision``, so under ``bypassPermissions`` the CLI ran the
        tool with the guard having inspected nothing. That is the silent no-op
        this module exists to remove, reintroduced one layer down, so the
        fields are validated before they become a ``Call`` rather than after.
        """
        tool_name = ""
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=self.timeout_s)
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise TypeError(f"expected a JSON object, got {type(payload).__name__}")
            name = payload.get("tool_name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "no tool_name in the payload, so there is no call to check — a guard "
                    "keyed on the tool name would have seen an empty name and passed"
                )
            tool_name = name
            arguments = payload.get("tool_input")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise TypeError(
                    f"tool_input is a {type(arguments).__name__}, not an object, so the "
                    f"arguments the guard is supposed to inspect cannot be read"
                )
            body = await self._decide(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001 — an unreadable payload is a refusal
            reason = (
                f"the agentkit hook could not read the CLI's payload "
                f"({type(exc).__name__}: {exc})"
            )
            # Recorded, not just answered: `decisions` is the only in-process
            # evidence the guard was live, and a call refused because it was
            # unreadable is one an operator especially wants to see.
            self._decisions.append(HookDecision(tool=tool_name, allowed=False, reason=reason))
            body = _deny(reason)
        try:
            writer.write(json.dumps(body).encode("utf-8") + b"\n")
            await writer.drain()
        except Exception:  # noqa: BLE001 — the script's own timeout covers a lost peer
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    # ---- the decision --------------------------------------------------------------------

    async def _terminal(self, call: Call) -> AsyncIterator[Any]:
        """The chain's terminal. In an ``Invoker`` this is ``tool.run``; here
        the CLI is about to do that itself, so the terminal exists only to give
        the middlewares something to wrap. Reaching it means nothing refused."""
        yield _NOT_EXECUTED

    async def _decide(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """One CLI tool call → one pass through the real chain → allow or deny.

        Never raises. This runs inside a connection handler whose only output
        is a JSON line; an exception escaping here would leave the script with
        nothing to print, and a hook that prints nothing is a hook the CLI
        reads as "no opinion" — the fail-open this whole module exists to
        close. Every failure becomes a deny with the reason attached.
        """
        side_effecting, url_arg = _NATIVE_TOOLS.get(tool_name, (True, None))
        # An unknown tool defaults to ``side_effecting=True``: the conservative
        # reading, since a middleware keyed on that flag should treat a tool it
        # has never heard of as the dangerous kind.
        request = ToolRequest(
            name=tool_name,
            arguments=dict(tool_input),
            tool=_NEVER_RUNS,
            side_effecting=side_effecting,
            url_arg=url_arg,
        )
        call = Call("tool", request, self.ctx, meta={"claude_cli_hook": True})
        before = _fingerprint(request)

        reason: str | None = None
        try:
            await asyncio.wait_for(collect_one(self._handler(call)), timeout=self.timeout_s)
        except Blocked as exc:
            # The typed refusal. Its reason is a policy statement written for a
            # reader, so it goes to the model verbatim.
            reason = exc.reason
        except TimeoutError:
            # Also reached if a middleware raises TimeoutError itself, which is
            # indistinguishable from here and lands on the same answer anyway.
            reason = (
                f"the agentkit tool chain did not answer within {self.timeout_s}s, "
                f"so the {tool_name} call was refused rather than allowed unchecked"
            )
        except Exception as exc:  # noqa: BLE001 — see the docstring
            reason = (
                f"the agentkit tool chain failed while checking {tool_name} "
                f"({type(exc).__name__}: {exc}); a guard that cannot answer refuses"
            )
        else:
            if _fingerprint(call.request) != before:
                # A middleware rewrote the unit of work. ``MiddlewareContext``
                # allows that and the ``Invoker`` honours it, but a PreToolUse
                # hook has no channel for "run it with these arguments
                # instead" — the CLI would run the ORIGINAL call. Silently
                # dropping the rewrite would execute exactly what the
                # middleware was trying to prevent, so the rewrite refuses.
                reason = (
                    f"a middleware rewrote the {tool_name} call, and a PreToolUse hook can only "
                    f"allow or refuse — the CLI would have run the original arguments, so the "
                    f"rewrite is treated as a refusal"
                )

        allowed = reason is None
        self._decisions.append(HookDecision(tool=tool_name, allowed=allowed, reason=reason or ""))
        await self._emit(tool_name, allowed, reason or "")
        return _pass() if allowed else _deny(reason or "")

    async def _emit(self, tool_name: str, allowed: bool, reason: str) -> None:
        """Put the decision on the run's observer stream. Best-effort and
        never raises: a refusal that happened is more important than a UI event
        that did not, and ``_decide`` has a never-raises contract to keep."""
        emit = getattr(self.ctx, "emit", None)
        if emit is None:
            return
        with contextlib.suppress(Exception):
            await emit(
                "gate.check",
                f"claude-cli {tool_name}: {'allowed' if allowed else 'refused'}",
                payload={"tool": tool_name, "allowed": allowed, "reason": reason},
            )


def _fingerprint(request: Any) -> str:
    """A stable rendering of the parts of a ``ToolRequest`` a rewrite could
    change. ``default=repr`` because the CLI's arguments are JSON to begin
    with, and anything a middleware substituted that is not is a change we
    want to notice rather than crash on."""
    return json.dumps(
        {"name": request.name, "arguments": request.arguments},
        sort_keys=True,
        default=repr,
    )


def _validated_middleware(middleware: Sequence[Any]) -> tuple[Any, ...]:
    """Reject at wiring time anything this mechanism cannot actually run.

    The rule is narrow and checkable: a ``PreToolUse`` hook fires before the
    tool and never sees a result, so a middleware may only contribute through
    ``on_request``. A ``BaseMiddleware`` that overrides ``on_response`` or
    ``on_error`` would have those phases run against a sentinel — ``Audit``
    would record a hash of nothing and call it "executed" — and a raw
    ``(call, next)`` middleware is an opaque callable we cannot inspect at all,
    so ``memoize``'s cache hit would short-circuit into an ALLOW of a call it
    meant to replace.

    Accepting them and quietly doing nothing is the failure mode this whole
    module was written to remove, so it is an error here instead.
    """
    mws = tuple(middleware)
    if not mws:
        raise ValueError(
            "hook_settings() was given no middlewares, so the generated hook could never "
            "refuse anything — a settings file that enforces nothing is the silent no-op "
            "this feature exists to prevent. Pass the tool chain, or do not pass --settings."
        )
    for mw in mws:
        if not isinstance(mw, BaseMiddleware):
            raise TypeError(
                f"{_name(mw)} is a raw (call, next) middleware. A PreToolUse hook runs before "
                f"the CLI's tool and never sees a result, so only a BaseMiddleware that "
                f"overrides on_request can apply here; a raw middleware may re-invoke, skip or "
                f"rewrite the result and none of that is expressible. Drop it from the hook "
                f"chain (it still applies to agentkit's own tool calls through the Invoker)."
            )
        overridden = [
            phase
            for phase in ("on_response", "on_error")
            if getattr(type(mw), phase) is not getattr(BaseMiddleware, phase)
        ]
        if overridden:
            raise TypeError(
                f"{_name(mw)} overrides {' and '.join(overridden)}, which a PreToolUse hook can "
                f"never reach: the CLI executes the tool itself, so there is no result and no "
                f"error to hand back. Only on_request applies here. Keep it in the Invoker's "
                f"tool chain, and use a PostToolUse hook for anything that needs the result."
            )
    return mws


def _name(mw: Any) -> str:
    return f"{type(mw).__name__}()" if isinstance(mw, BaseMiddleware) else repr(mw)


def _validated_tools(tools: Sequence[str], *, allow_unknown: bool) -> tuple[str, ...]:
    """A typo'd tool name is a matcher that never fires: a guard that looks
    wired and enforces nothing. So an unrecognised name is an error, with
    ``allow_unknown_tools=True`` as the escape hatch for a CLI that has grown a
    tool this table has not."""
    names = tuple(tools)
    if not names:
        raise ValueError(
            "hook_settings() needs at least one tool to guard. An empty tuple generates a "
            "matcher that fires for nothing, which is a settings file that looks like a guard "
            "and is not one."
        )
    for name in names:
        if not _TOOL_NAME.match(name):
            raise ValueError(
                f"{name!r} is not a usable tool name: the CLI matches these as a regex, so a "
                f"name outside [A-Za-z][A-Za-z0-9_]* either fails to compile or matches the "
                f"wrong set of tools."
            )
        if not allow_unknown and name not in _NATIVE_TOOLS and not name.startswith("mcp__"):
            raise ValueError(
                f"{name!r} is not a tool the Claude CLI is known to have (did you mean one of "
                f"{', '.join(sorted(_NATIVE_TOOLS))}?). A name the CLI never emits produces a "
                f"hook that never runs. Pass allow_unknown_tools=True if the CLI has genuinely "
                f"grown this tool."
            )
    return names


def _settings_body(base: Mapping[str, Any] | None, hooks: dict[str, Any]) -> dict[str, Any]:
    """Merge what is provably disjoint; refuse the rest.

    A caller's own ``hooks`` block and ours cannot be combined safely from
    here: concatenating two ``PreToolUse`` arrays runs both matchers (theirs
    may allow what ours refuses, and ``allow`` wins), while replacing theirs
    drops hooks they are relying on — and either way it happens silently, in a
    settings file nobody reads. Every other key is copied through untouched,
    because nothing about ``env`` or ``model`` interacts with a hook.
    """
    body = dict(base or {})
    if "hooks" in body:
        raise ValueError(
            "hook_settings() will not merge into settings that already define hooks: combining "
            "two PreToolUse arrays either runs both (and an 'allow' from the caller's hook "
            "overrides this chain's refusal) or drops one of them, silently. Generate the "
            "settings here and add the caller's other keys via base=, or wire the caller's "
            "hooks into the chain instead."
        )
    body["hooks"] = hooks
    return body


def _work_dir() -> Path:
    """A ``0700`` directory holding the script, the socket and the settings.

    ``0700`` is the authorisation check: an ``AF_UNIX`` socket is protected by
    the filesystem, so no other user on the host can answer this agent's
    permission decisions. ``mkdtemp`` gives that mode by default.
    """
    directory = Path(tempfile.mkdtemp(prefix="agentkit-hook-"))
    if len(str(directory / SOCKET_NAME)) > _SUN_PATH_MAX:
        # ``sun_path`` is a fixed 104/108-byte field and a long TMPDIR
        # overruns it — the bind would fail, or worse, bind a truncated path.
        shutil.rmtree(directory, ignore_errors=True)
        directory = Path(tempfile.mkdtemp(prefix="agentkit-hook-", dir="/tmp"))  # noqa: S108
    return directory


def hook_settings(
    *,
    middleware: Sequence[Any],
    ctx: Any,
    tools: Sequence[str],
    timeout_s: float = 5.0,
    base: Mapping[str, Any] | None = None,
    allow_unknown_tools: bool = False,
) -> HookSettings:
    """Generate Claude Code settings whose ``PreToolUse`` hook runs ``middleware``.

    ``middleware`` is the app's tool chain — the same list handed to
    ``Invoker(tool_middleware=...)``. Only middlewares that can actually apply
    before execution are accepted; see the module docstring's table, and
    ``_validated_middleware`` for why the rest are a ``TypeError`` here rather
    than a no-op later.

    ``tools`` names the CLI tools to guard. ``timeout_s`` bounds one chain
    evaluation; the script and the CLI get looser deadlines derived from it so
    the innermost one answers first with a usable reason.

    ``base`` supplies non-hook settings keys to carry through (``env``,
    ``model``, …). Settings that already define ``hooks`` are refused rather
    than merged.

    Returns immediately with a **listening** endpoint: the socket is bound and
    listening synchronously before this returns, so ``.path`` is never a
    promise — there is no window in which the CLI could run a hook against a
    socket that does not exist yet. Serving is then handed to the running
    event loop, which is why this must be called from inside one.

    The caller owns the lifetime::

        async with hook_settings(middleware=chain, ctx=ctx, tools=("Write",)) as settings:
            await Agent(..., cognition=ClaudeCliCognition(settings=settings.path)).run(task, ctx)

    ``aclose()`` removes the directory, taking the script, the socket and the
    settings file with it.
    """
    if not hasattr(socket, "AF_UNIX"):  # pragma: no cover — exercised only on Windows
        raise RuntimeError(
            "hook_settings() needs AF_UNIX sockets, which this platform does not have. The "
            "loopback-TCP alternative is deliberately not offered: it would let any local "
            "process answer this agent's permission decisions."
        )
    try:
        asyncio.get_running_loop()
    except RuntimeError as exc:
        raise RuntimeError(
            "hook_settings() must be called from inside a running event loop: the hook is "
            "answered by the chain in THIS process, which is the whole point of the design."
        ) from exc

    mws = _validated_middleware(middleware)
    names = _validated_tools(tools, allow_unknown=allow_unknown_tools)

    directory = _work_dir()
    socket_path = directory / SOCKET_NAME
    script_path = directory / SCRIPT_NAME

    # Bind SYNCHRONOUSLY, before anything writes a path into the settings file.
    # ``listen()`` means the kernel queues connections from the moment this
    # returns, so the settings file never names a socket that is not there yet
    # — even if the loop has not scheduled the serving task.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(socket_path))
        sock.listen(128)
        sock.setblocking(False)
    except OSError:
        sock.close()
        shutil.rmtree(directory, ignore_errors=True)
        raise

    # Everything past the bind holds two resources — the listening socket and
    # the directory — so any failure has to give both back. Without this, a
    # settings shape this function REFUSES (base= carrying its own hooks) would
    # leak a bound socket and a temp directory on the way out: a wiring error
    # that costs a file descriptor is a wiring error people stop reporting.
    try:
        # Three deadlines, loosest outermost. The CLI's must be the loosest of
        # all: a hook the CLI kills is a non-blocking error, which prints and
        # then RUNS the tool. Letting our own deadlines fire first turns that
        # fail-open into a deny with a reason.
        script_timeout = timeout_s + 2.0
        cli_timeout = int(timeout_s + 4.0) + 1

        script_path.write_text(
            _HOOK_SCRIPT.replace("__SOCKET_PATH__", json.dumps(str(socket_path))).replace(
                "__TIMEOUT_S__", repr(float(script_timeout))
            ),
            encoding="utf-8",
        )
        script_path.chmod(0o600)

        # ``-I -S``: no PYTHONPATH, no user site dir, no sitecustomize. The
        # hook runs inside a CLI session whose environment this process does
        # not own, so isolating the interpreter keeps that environment from
        # injecting code into the guard. It also costs ~9 ms less to start
        # (46 ms vs 55 ms measured), which matters per tool call.
        command = shlex.join([sys.executable, "-I", "-S", str(script_path)])
        body = _settings_body(
            base,
            {
                HOOK_EVENT: [
                    {
                        # Anchored: a bare ``Write`` matcher would also fire
                        # for a future ``WriteAll``, which is a guard applying
                        # somewhere nobody chose.
                        "matcher": f"^({'|'.join(names)})$",
                        "hooks": [{"type": "command", "command": command, "timeout": cli_timeout}],
                    }
                ]
            },
        )
        (directory / SETTINGS_NAME).write_text(json.dumps(body, indent=2), encoding="utf-8")
    except BaseException:
        sock.close()
        shutil.rmtree(directory, ignore_errors=True)
        raise

    settings = HookSettings(
        ctx=ctx,
        middleware=mws,
        tools=names,
        directory=directory,
        sock=sock,
        timeout_s=timeout_s,
    )
    settings._serve()
    return settings


__all__ = ["HOOK_EVENT", "HookDecision", "HookSettings", "hook_settings"]
