"""Machinery shared by every "delegate the loop to a local CLI" cognition.

Two of them ship — :mod:`agentkit.agents.cognition.claude_cli` and
:mod:`agentkit.agents.cognition.codex_cli` — and they are genuinely different
programs: different flags, different event vocabularies, different containment
models. What they are NOT different about is the plumbing around the
subprocess, and that is what lives here.

Everything in this module was written for the ``claude`` cognition first and
then needed, unchanged, by the ``codex`` one. Each was copied for exactly as
long as it took to notice: a second ``_terminate`` that forgets the SIGKILL
escalation, or a second ``_parse_line`` that catches ``JSONDecodeError``
instead of ``ValueError``, is not a cosmetic duplication — the ``ValueError``
one in particular is a shipped bug this codebase has already paid for once (one
undecodable byte on stdout ended a whole run and reported $0.00 to the budget).
A single definition is the only way the fix stays fixed for both.

Nothing here knows a flag name or an event type. Anything that does belongs in
the per-CLI module, because that is the half where the two really do differ and
a shared "mostly the same" abstraction would be worse than two honest copies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentkit.kernel.protocols import Ctx

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent


# The transport seam's type. Deliberately ``...``-argumented: the spawn sites
# pass different ``stdin=`` modes and future flags may add keywords, and a
# double that had to re-declare the exact signature would break on every one of
# them for no gain — it forwards ``**kwargs`` anyway.
CliSpawn = Callable[..., Awaitable["asyncio.subprocess.Process"]]


# ─────────────────────────────────────────────────────────────────────────────
# Class-level concurrency guard.
#
# The upstream Claude Code SDK has a known hang at ~200 parallel subprocesses
# (SDK issue #728) because the subprocess close path serializes on shared IPC.
# We bound spawn concurrency per (binary, config_dir) tuple so that two
# distinct CLI installations (or two isolated config dirs) don't share a
# single semaphore. The default (8) keeps a healthy margin under the observed
# hang threshold; callers who know their environment can bump it via
# ``max_concurrent``.
#
# Keying on the BINARY is what lets ``claude`` and ``codex`` sessions run
# alongside each other without competing for one bound — two different
# programs, two different subprocess tables, no shared failure mode.
#
# ``weakref.WeakValueDictionary`` lets the semaphore get GC'd when no cognition
# referencing that key remains — no permanent global state accumulated across
# long-running processes.
# ─────────────────────────────────────────────────────────────────────────────
_SEMAPHORES: weakref.WeakValueDictionary[tuple[str, str | None, int], asyncio.BoundedSemaphore] = (
    weakref.WeakValueDictionary()
)


def _get_semaphore(bin_: str, config_dir: str | None, max_concurrent: int) -> asyncio.BoundedSemaphore:
    """Return the shared BoundedSemaphore for this (bin, config_dir, max) triple.

    A ``WeakValueDictionary`` alone doesn't hold the semaphore alive — the
    caller must keep the returned reference for the duration of the acquire.
    That's the intended lifetime: as soon as no cognition still has an in-flight
    ``drive`` referencing it, the entry drops.
    """
    key = (bin_, config_dir, max_concurrent)
    sem = _SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.BoundedSemaphore(max_concurrent)
        _SEMAPHORES[key] = sem
    return sem


def _middleware_name(mw: object) -> str:
    """A name a reader can match against the chain they wrote.

    The two middleware shapes name themselves differently. ``BaseMiddleware``
    instances (``Egress``, ``Audit``, ``MeterMiddleware``) have a useful class
    name. The raw ``(call, next)`` ones are closures that are ALL called ``mw``
    — ``memoize()``, ``retry()`` and ``tracing()`` return functions whose
    ``__name__`` is literally ``"mw"``, which in a warning would read as three
    identical entries. Their ``__qualname__`` is ``"memoize.<locals>.mw"``, and
    the half before ``.<locals>`` is the factory the caller actually typed.

    ``idempotent()`` therefore reports as ``memoize`` — it delegates to it. That
    is a small inaccuracy kept on purpose: the alternative is a registry of
    factory identities that drifts the first time someone writes a middleware
    of their own, and "memoize" still points at the right module.
    """
    qualname = getattr(mw, "__qualname__", None)
    if isinstance(qualname, str) and qualname:
        return qualname.split(".<locals>.")[0].rsplit(".", 1)[-1]
    return type(mw).__name__


def _tool_middleware_names(ctx: Ctx | None) -> tuple[str, ...]:
    """The tool chain's middleware names, de-duplicated, in chain order.

    Read defensively off ``ctx.invoker`` because a context legitimately has no
    invoker (``make_test_ctx()`` with no LLM, a bare structural stub) and a
    missing collaborator must not turn a safety warning into an AttributeError
    raised out of ``drive``.

    Only the TOOL chain. The chat chain is bypassed too, but it was never going
    to see a tool call, so naming ``compaction`` in a warning about tool calls
    would be a name the reader cannot act on.
    """
    invoker = getattr(ctx, "invoker", None)
    chain = getattr(invoker, "tool_middleware", None) or ()
    # ``dict.fromkeys`` rather than ``set``: chain ORDER is what the caller
    # wrote, and two ``retry()`` entries should read as one name, not two.
    return tuple(dict.fromkeys(_middleware_name(mw) for mw in chain))


def _coerce_structured(agent: Agent | None, value: Any) -> tuple[Any, str | None]:
    """Turn a CLI's validated JSON into the type the agent declared.

    The CLI validates against the schema and hands back a plain dict. An agent
    that declared ``output=Invoice`` wants an ``Invoice``, so the value goes
    back through the same ``SchemaAdapter`` that produced the schema — one
    round trip, one definition of the type.

    Returns ``(parsed, error)``. A coercion failure is reported, never raised:
    the run happened and its text is real, so the caller gets a terminal event
    with ``partial=True`` and an explanation rather than an exception thrown
    from inside a generator.

    With no adapter (an explicit ``json_schema=`` on an agent that declares no
    ``output=``) the validated dict IS the parsed value — there is no Python
    type to build.
    """
    adapter = getattr(agent, "_output_adapter", None)
    if adapter is None:
        return value, None
    try:
        return adapter.validate(value), None
    except Exception as exc:  # noqa: BLE001 — reported as data, see docstring
        # ``OutputCoercionError.__str__`` summarises ("1 error(s)"); the
        # per-field diagnostics live on ``.errors`` and are the only part a
        # caller can act on, so they go into the message.
        detail = "; ".join(str(e) for e in getattr(exc, "errors", ()) or ())
        return None, f"{type(exc).__name__}: {exc}" + (f" — {detail}" if detail else "")


@dataclass(frozen=True, slots=True)
class _CliCall:
    """The minimum a ``Meter`` needs from a "call" it is charging.

    ``Budget.charge`` ignores its ``call`` entirely, but ``Quota`` reads
    ``call.ctx.scope.key()`` to partition per tenant — so a bare ``None`` would
    charge the run budget and crash the quota. ``request`` is present and
    ``None`` because a custom meter may look for it; there is no ``ChatRequest``
    here, and inventing one would be a lie.
    """

    ctx: Any
    request: Any = None


def _parse_line(line: bytes) -> dict[str, Any] | None:
    """Parse one stdout line as a JSON object; skip blank / non-JSON lines.

    A CLI occasionally emits a blank warmup line or a non-JSON diagnostic;
    those must NOT crash the loop — return ``None`` and let the caller move
    on.

    ``ValueError``, not ``JSONDecodeError``. ``json.loads`` on **bytes** sniffs
    the encoding first (RFC 4627 / :func:`json.detect_encoding`), so a line
    opening with ``b"\\xff\\xfe"`` is read as a UTF-16-LE BOM and a line
    opening with ``b"\\xef\\xbb\\xbf"`` as a UTF-8 one — and when the rest does
    not decode, what comes back is a ``UnicodeDecodeError``, which is a
    ``ValueError`` but is NOT a ``JSONDecodeError``. It escaped this function
    and hit ``drive``'s ``except BaseException``, so ONE undecodable byte on
    stdout ended the whole run: ``stop_reason="parse_failed"``, no output, and
    — because the reader never got past that line — the terminal payload was
    never seen, so a completed run reported ``$0.00`` to the budget. Measured.
    Reachable without anything exotic: both binaries emit human diagnostics on
    the same streams, and filenames are arbitrary bytes, so a warning naming a
    latin-1 path is a non-UTF-8 line on stdout — exactly the case this
    docstring already promised to survive. Both exceptions mean the same thing
    here — "this line is not a JSON object" — and both must be skipped.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except ValueError:  # JSONDecodeError *and* UnicodeDecodeError
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _reraise_if_not_an_exception(fatal_exc: BaseException | None) -> None:
    """Finish the sentence an outer ``except BaseException`` starts.

    That arm widens past ``Exception`` so a ``KeyboardInterrupt`` or a
    ``SystemExit`` mid-run still produces the terminal event these cognitions
    guarantee — its comment says "before propagating". It did not propagate.
    Ctrl-C during ``agent.run()`` on a CLI cognition came back as a tidy
    ``AgentResult(partial=True, stop_reason="failed")`` with
    ``evals["error"] == "KeyboardInterrupt: "``, the interpreter kept running,
    and the caller's own ``except KeyboardInterrupt`` never fired. Measured;
    ``asyncio.CancelledError`` had exactly this bug and was fixed the same way.

    ``Exception`` is the dividing line and not a hand-written list because it
    is the same line the language draws: things outside it are not "the
    operation failed", they are "this program is being taken down" or "a
    contract has been violated" — ``SystemExit``, ``KeyboardInterrupt``, and
    ``agentkit.testing.ScriptExhausted``, which is a ``BaseException``
    precisely so it survives sites like this one. A run's failure modes are all
    ``Exception``s (``FileNotFoundError`` on a missing binary, ``PermissionError``,
    a parse bug), so the reported-as-data contract is untouched: every one of
    them still arrives as a ``final`` event and nothing more.
    """
    if fatal_exc is not None and not isinstance(fatal_exc, Exception):
        raise fatal_exc


async def _terminate(proc: asyncio.subprocess.Process, grace_s: float) -> None:
    """Send SIGTERM, wait ``grace_s`` for a clean exit, then SIGKILL if the
    process is still alive. Never raises: this runs from a ``finally`` and
    must not mask the original ``Cancelled``.
    """
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()


__all__ = [
    "CliSpawn",
    "_CliCall",
    "_coerce_structured",
    "_get_semaphore",
    "_middleware_name",
    "_parse_line",
    "_reraise_if_not_an_exception",
    "_terminate",
    "_tool_middleware_names",
]
