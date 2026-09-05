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
import math
import os
import signal
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import StreamEvent, Usage

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


def _coerce_structured(
    agent: Agent | None, value: Any
) -> tuple[Any, StructuredOutputFailure | None]:
    """Turn a CLI's validated JSON into the type the agent declared.

    The CLI validates against the schema and hands back a plain dict. An agent
    that declared ``output=Invoice`` wants an ``Invoice``, so the value goes
    back through the same ``SchemaAdapter`` that produced the schema — one
    round trip, one definition of the type.

    Returns ``(parsed, failure)``. A coercion failure is reported, never raised:
    the run happened and its text is real, so the caller gets a terminal event
    with ``partial=True`` and an explanation rather than an exception thrown
    from inside a generator.

    The failure is a :class:`StructuredOutputFailure` rather than a string. It
    used to be a sentence built here by joining the adapter's per-field
    diagnostics, which threw away the structure an application needs — the
    field paths — and left anyone who wanted them parsing English that a
    library upgrade can reword.

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
        # ``OutputCoercionError.__str__`` only summarises ("4 error(s)"); the
        # per-field diagnostics live on ``.errors`` and are the only part a
        # caller can act on, so they are parsed into typed violations.
        return None, StructuredOutputFailure.from_coercion_error(exc)


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


async def _charge_meters(ctx: Ctx | None, usage: Usage, *, enabled: bool) -> str | None:
    """Put a CLI run's spend on the framework's books. Returns an error note.

    Both cognitions bypass the ``Invoker``, so the ``meter()`` middleware never
    sees this usage and every meter on the context stays at zero. That is how a
    documented safety mechanism ends up doing nothing — the same failure
    ``ActorBudget`` had — so the charge happens here instead.

    **Nothing raises out of this.** The spend already happened, the run already
    produced an answer, and the terminal-event guarantee says the caller gets
    that answer; a ceiling crossed on the LAST call is recorded and reported,
    not converted into a lost result. A custom meter that misbehaves is
    contained for the same reason. The note comes back as data so ``_finalise``
    can put it in ``evals`` rather than lose it.

    Shared rather than written twice. It WAS written twice — the two copies
    were byte-identical apart from their docstrings, which is the state a
    metering path should never be left in: a fix applied to one adapter and not
    the other does not fail a test or a type check, it just silently stops
    charging for one of the two CLIs. There is nothing adapter-specific in the
    logic; the only input either one contributed was its own ``meter_spend``,
    which is why that arrives as ``enabled``.

    ``enabled`` is keyword-only and has no default, so a call site has to state
    its ``meter_spend`` rather than inherit a guess about it.
    """
    if not enabled or ctx is None:
        return None
    call = _CliCall(ctx=ctx)
    note: str | None = None
    for meter in getattr(ctx, "all_meters", None) or []:
        try:
            await meter.charge(call, usage)
        except Exception as exc:  # noqa: BLE001 — see docstring
            note = f"{type(exc).__name__}: {exc}"
    actor = getattr(ctx, "actor_budget", None)
    if actor is not None:
        with contextlib.suppress(Exception):
            actor.charge(tokens=usage.total_tokens, cost_usd=usage.cost_usd, steps=1)
    return note


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


# Whether this OS can signal a process GROUP. POSIX-only; on Windows both
# names are absent and every group operation below degrades to signalling the
# direct child, which is what the code did everywhere before groups existed.
_HAS_PROCESS_GROUPS = hasattr(os, "killpg") and hasattr(os, "getpgid")

# Passed as ``start_new_session=`` so the child becomes the leader of a NEW
# session and process group, which is the half that makes ``os.killpg`` below
# safe: without it ``os.getpgid(child)`` returns OUR OWN group and a group kill
# would take down the calling service.
#
# A plain bool rather than a ``**kwargs`` dict so the spawn call stays typed —
# ``**dict[str, bool]`` cannot be checked against ``create_subprocess_exec``'s
# overloads and produced a dozen mypy errors per call site. ``False`` is the
# parameter's own default and is accepted everywhere, so the Windows path
# passes it harmlessly rather than branching around it.
START_NEW_SESSION: bool = _HAS_PROCESS_GROUPS


def _child_pgid(proc: asyncio.subprocess.Process) -> int | None:
    """The child's process-group id, or ``None`` when there isn't one to signal.

    ``None`` for three distinct reasons, all of which must degrade to
    signalling the direct child rather than raising out of a ``finally``:

    * the platform has no process groups (Windows);
    * the process is a test double — ``_ReplayProcess`` has no ``pid`` at all,
      and inventing one would make the doubles signal a real pid;
    * the child is already reaped, so ``getpgid`` raises ``ProcessLookupError``.

    The pid is also sanity-checked against our own group. That check is the
    difference between a bug and a catastrophe: if a caller supplies a
    ``spawn=`` that forgets ``start_new_session``, the child sits in the
    SERVICE's group, and killing it would kill the service. Measured — see the
    process-group test: ``start_new_session=True`` alone still orphans
    grandchildren, so both halves have to be present or neither is.
    """
    if not _HAS_PROCESS_GROUPS:
        return None
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None
    if pgid == os.getpgrp():
        # Not a group of its own — signalling it would signal us.
        return None
    return pgid


def _signal_tree(proc: asyncio.subprocess.Process, pgid: int | None, sig: int) -> None:
    """Deliver ``sig`` to the whole tree, best-effort, never raising.

    Both halves, deliberately. The group call is what reaches the
    grandchildren — a ``claude`` running ``Bash(npm install)``, a ``codex``
    running a test suite — and the direct call is what still works for a test
    double and on a platform with no groups. Sending SIGTERM twice to the same
    child is harmless; sending it to neither is the leak this exists to close.
    """
    if pgid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, sig)
    with contextlib.suppress(ProcessLookupError, OSError):
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()


async def _terminate(
    proc: asyncio.subprocess.Process,
    grace_s: float,
    *,
    process_group: bool = True,
) -> None:
    """Stop the child AND everything it started. Never raises: this runs from a
    ``finally`` and must not mask the original ``Cancelled``.

    SIGTERM the group, wait ``grace_s`` for the direct child, then SIGKILL the
    group. The final sweep is not a fallback for a stubborn child — it runs
    even when the child exited politely, because the child exiting says nothing
    about the ``npm install`` it forked. That was the leak: ``proc.terminate()``
    signals ONE pid, so a cancelled run left the expensive half of the work
    running, holding its own copy of the pipe, until it finished on its own.
    Measured before this change; the grandchild survived every time.

    ``process_group=False`` opts a caller out entirely (signal only the direct
    child). Nothing in-tree passes it; it exists so a caller who deliberately
    shares a group with the child is not forced into a group kill.

    **The pgid is captured BEFORE the first signal**, while the leader is
    certainly alive, because ``proc.wait()`` reaps the leader and frees the id
    for reuse. A reaped-then-recycled pgid is a real if narrow race — the OS
    would have to allocate the same id to a new group leader inside the
    microseconds between the reap and the sweep — and it is the same race every
    process supervisor accepts for the same reason: the alternative is leaking
    the tree on every cancel.
    """
    pgid = _child_pgid(proc) if process_group else None
    _signal_tree(proc, pgid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except TimeoutError:
        pass
    _signal_tree(proc, pgid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        await proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Environment policy.
#
# Both cognitions used to hand the child ``os.environ.copy()`` and nothing
# else. That is defensible as a default and indefensible as the ONLY option,
# because of what the binaries actually do with what they find. Measured
# against claude 2.1.236, which says it in its own words on stderr:
#
#     ⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY or another
#       auth source is set and takes precedence over your claude.ai login
#
# So a service that sets ``config_dir=`` per tenant — whose entire purpose is
# to run as that tenant's signed-in profile — silently runs on whatever
# machine-wide API key happens to be exported instead. The isolation the caller
# asked for is precisely the thing that does not happen, and the only signal is
# a stderr line the cognition surfaces only on failure.
#
# codex 0.152.1 measured the OTHER way on the same test: a bogus
# ``OPENAI_API_KEY`` was ignored in favour of the ``CODEX_HOME`` login. The
# policy is still worth having there — precedence is configurable, the reverse
# is a CLI upgrade away, and "the child gets exactly what we chose" is the
# property a reproducible service wants either way — but the urgency is
# Claude's, and pretending both are equally on fire would be inventing a bug.
# ─────────────────────────────────────────────────────────────────────────────

EnvPolicy = Literal["inherit", "profile", "isolated"]

# What survives ``isolated``. Not a security boundary — a child that runs shell
# commands can read the filesystem regardless — but the difference between a
# reproducible spawn and one that inherits whatever the operator's shell had.
#
# Every name here earns its place by breaking something when absent: PATH finds
# the binary, HOME/XDG find its config, the TLS and proxy variables are how a
# corporate network is reachable at all, and dropping those turns "isolated"
# into "does not work behind a proxy", which is how a policy gets switched off
# wholesale instead of being fixed.
_ISOLATED_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    # Corporate TLS interception and egress. Absent these, an isolated spawn
    # fails to reach the provider at all on a managed network.
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    # Windows: a child cannot start without these.
    "SystemRoot",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
)


def _build_child_env(
    *,
    policy: EnvPolicy,
    credential_vars: tuple[str, ...],
    overrides: Mapping[str, str] | None = None,
    passthrough: tuple[str, ...] = _ISOLATED_PASSTHROUGH,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """The child's environment, and the names this policy removed.

    Returns the removed names as well as the env because they are the only
    thing worth warning about: "we stripped ANTHROPIC_API_KEY so your
    ``config_dir`` profile is actually in force" is actionable, and a silent
    strip is the same class of invisible behaviour as the silent inherit it
    replaces.

    * ``inherit``  — the parent's environment, verbatim. The historical
      behaviour, kept reachable by name so a caller who wants it says so.
    * ``profile``  — the parent's environment MINUS ``credential_vars``, so the
      CLI falls through to its own login/config directory. Everything else the
      run depends on (PATH, proxies, locale) is untouched, which is what makes
      this a safe default to imply from ``config_dir=``.
    * ``isolated`` — only ``passthrough`` plus ``overrides``. Nothing about the
      calling process leaks in, so two machines running the same task hand the
      CLI the same environment.

    ``overrides`` are applied LAST in every policy, including over a name this
    policy just removed. That ordering is the point: ``profile`` with an
    explicit ``env={"ANTHROPIC_API_KEY": tenant_key}`` is how a caller states
    "this key, not the ambient one", and a strip that outranked an explicit
    assignment would make the safe policy unusable with per-tenant credentials.
    """
    overrides = dict(overrides or {})
    if policy == "isolated":
        env = {name: os.environ[name] for name in passthrough if name in os.environ}
        # Nothing was "removed" from a set that was never copied; report the
        # credentials that existed and did not survive, since that is the fact
        # a caller debugging an auth error needs.
        removed = tuple(n for n in credential_vars if n in os.environ and n not in overrides)
    else:
        env = os.environ.copy()
        removed = ()
        if policy == "profile":
            removed = tuple(n for n in credential_vars if n in env and n not in overrides)
            for name in removed:
                env.pop(name, None)
    env.update(overrides)
    return env, removed


# ─────────────────────────────────────────────────────────────────────────────
# Structured output: refuse a schema before it costs a subprocess.
# ─────────────────────────────────────────────────────────────────────────────


class InvalidSchemaError(ValueError):
    """A requested output schema cannot be sent to a CLI.

    ``ValueError`` because that is what the constructors already raise for a
    configuration the CLI would refuse, and a caller catching ``ValueError``
    around cognition setup should not have to learn a new type to keep working.
    """


def _validate_json_schema(schema: Any, *, who: str) -> dict[str, Any]:
    """Check a schema well enough to spawn, and return it.

    NOT a JSON Schema validator, and deliberately not: bringing one in would
    cost the zero-dependency core its main property, and the failures worth
    catching here are structural rather than subtle. Both binaries reject a bad
    schema — ``claude`` with "--json-schema is not a valid JSON Schema",
    ``codex`` on the file it was handed — but they do it two to five seconds
    into a spawn, from inside a subprocess, with the message on stderr, which
    this cognition surfaces only on a failed run. The same mistake caught at
    the call site costs nothing and names itself.

    ``who`` prefixes the message so a reader knows which cognition refused.
    """
    if not isinstance(schema, dict):
        raise InvalidSchemaError(
            f"{who}: json_schema must be a JSON Schema object (a dict), got "
            f"{type(schema).__name__}"
        )
    if not schema:
        raise InvalidSchemaError(
            f"{who}: json_schema is empty. Pass None to run without structured output "
            "rather than an empty schema, which constrains nothing and which the CLI "
            "rejects."
        )
    try:
        json.dumps(schema)
    except (TypeError, ValueError) as exc:
        # A Pydantic ``FieldInfo``, a ``type``, a set — the shapes that reach a
        # schema dict when it was assembled by hand rather than by an adapter.
        raise InvalidSchemaError(
            f"{who}: json_schema is not JSON-serialisable ({exc}). Both CLIs take the "
            "schema as JSON, so every value in it has to be a JSON scalar, list or dict."
        ) from None
    declared = schema.get("type")
    if declared is not None and not isinstance(declared, str | list):
        raise InvalidSchemaError(
            f"{who}: json_schema['type'] must be a string or a list of strings, got "
            f"{type(declared).__name__}"
        )
    props = schema.get("properties")
    if props is not None and not isinstance(props, dict):
        raise InvalidSchemaError(
            f"{who}: json_schema['properties'] must be an object, got {type(props).__name__}"
        )
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            raise InvalidSchemaError(
                f"{who}: json_schema['required'] must be a list of property names, got "
                f"{required!r}"
            )
        # The one semantic check, because it is the mistake that actually
        # happens: a field renamed in ``properties`` and not in ``required``
        # produces a schema no output can ever satisfy, so the CLI burns its
        # whole structured-output retry budget and returns
        # ``error_max_structured_output_retries`` — a failure that reads as the
        # model's fault and is not.
        if isinstance(props, dict):
            orphans = [r for r in required if r not in props]
            if orphans:
                raise InvalidSchemaError(
                    f"{who}: json_schema requires {orphans} but does not define "
                    f"{'them' if len(orphans) > 1 else 'it'} in 'properties' "
                    f"(defined: {sorted(props)}). No output can satisfy this schema, so the "
                    "CLI would exhaust its structured-output retries and blame the model."
                )
    return schema


# ─────────────────────────────────────────────────────────────────────────────
# Liveness.
#
# A caller can already wrap ``drive`` in ``asyncio.wait_for`` and both
# cognitions honour it — the cancel path is tested against real subprocesses.
# What that cannot do is tell the difference between the four ways a CLI run
# stops making progress, and the difference is the whole diagnosis:
#
#   the binary never came up          → auth, config, a bad --flag
#   it came up and said nothing       → the provider is hanging
#   it went quiet mid-answer          → a tool call is stuck
#   it is simply taking too long      → the work is too big
#
# One ``TimeoutError`` for all four leaves an operator to guess. Measured, and
# the reason this exists: ``claude -p`` with an invalid ``ANTHROPIC_API_KEY``
# produced no stdout, no stderr and no exit for over 45 seconds — the process
# was alive, the run looked active, and nothing in the stream said otherwise.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CliTimeouts:
    """Four liveness bounds on one CLI run, in seconds. ``None`` disables one.

    ``startup``
        Spawn to the CLI's FIRST line of stdout. Local work only — boot, read
        config, resolve auth — so it cannot false-positive on model latency,
        which is why it is the one bound that defaults to ON. Both binaries
        announce themselves first (``system/init`` for claude,
        ``thread.started`` for codex) within a few seconds; the 120s default is
        roughly forty times the observed warm-up, so it fires for hangs and
        nothing else.

    ``first_event``
        Spawn to the first line that produces a ``StreamEvent`` — the first
        thing a UI could show. Separates "the CLI is up but the provider is
        not answering" from "the CLI never came up".

    ``idle``
        The longest gap ALLOWED BETWEEN consecutive stdout lines.

        **This is the one that bites.** The CLI runs its own tools in its own
        process, so a ``Bash(npm install)`` or a long test suite is minutes of
        legitimate silence on stdout. An ``idle`` tuned to model latency will
        kill working runs. Set it above the longest tool call the session can
        make, or leave it off.

    ``total``
        Wall clock for the whole run, measured from the spawn.

    Defaults are deliberately asymmetric: ``startup`` on, the rest off. A
    default that silently kills a working run is worse than the hang it
    prevents, and only ``startup`` is knowably safe without knowing what the
    session does. :meth:`production` is the opinionated preset for a service
    that has made that judgement.
    """

    startup: float | None = 120.0
    first_event: float | None = None
    idle: float | None = None
    total: float | None = None

    def __post_init__(self) -> None:
        for name in ("startup", "first_event", "idle", "total"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(
                    f"CliTimeouts.{name} must be a positive number of seconds or None "
                    f"(got {value!r}). Zero would time out before the process could be "
                    "scheduled; None is how a bound is disabled."
                )

    @classmethod
    def production(cls) -> CliTimeouts:
        """All four bounds set, for a service that would rather fail than hang.

        ``idle=600`` because the bound has to clear the longest TOOL call the
        session can make, not the longest model turn — see the warning on
        ``idle``. Ten minutes clears a dependency install; it does not clear a
        full integration suite, and a session that runs one needs its own
        number.
        """
        return cls(startup=60.0, first_event=180.0, idle=600.0, total=3600.0)

    @classmethod
    def off(cls) -> CliTimeouts:
        """No bounds at all — the behaviour before this existed."""
        return cls(startup=None, first_event=None, idle=None, total=None)


class CliTimedOut(Exception):
    """A liveness bound was crossed. ``reason`` is the typed stop reason.

    Raised out of :func:`_iter_stdout` and converted to a terminal event by the
    caller; it never escapes ``drive``, which reports failures as data.
    """

    def __init__(self, reason: str, limit_s: float) -> None:
        super().__init__(f"{reason} after {limit_s:g}s")
        self.reason = reason
        self.limit_s = limit_s


# Longest single stdout line the reader will assemble.
#
# ``asyncio.create_subprocess_exec`` defaults its ``StreamReader`` to 64 KiB,
# and both CLIs speak newline-delimited JSON where ONE line can carry a whole
# tool result — the contents of a file the agent just read. Past 64 KiB the
# reader raises ``ValueError: Separator is not found, and chunk exceed the
# limit`` and the run dies. Measured against codex 0.152.1 on "read every file
# under agentkit/ and summarise each": ``failed`` / ``parse_failed`` after
# three events, with the CLI itself exiting 0 and the answer thrown away.
#
# That error names nothing an operator could act on, which is what made it
# expensive: it reads like a bug in the parser rather than a buffer that was
# sized for a different protocol.
#
# 8 MiB is a ceiling, not a reservation — the buffer grows only as far as a
# line actually needs — so the cost of the headroom is nothing until it is
# used, and it stays a real backstop against a runaway line eating memory.
_STDOUT_LINE_LIMIT = 8 * 1024 * 1024


class CliLineTooLong(Exception):
    """One stdout line exceeded ``_STDOUT_LINE_LIMIT`` and could not be read.

    A named type because of what it replaced. ``StreamReader.readline`` reports
    this as a bare ``ValueError: Separator is not found, and chunk exceed the
    limit``, which the adapters classified as ``parse_failed`` — so the operator
    got a message that names no limit, no payload and no fix, and reads like a
    bug in agentkit's parser rather than a buffer sized for a different
    protocol. Measured against real codex on "read every file under agentkit/
    and summarise each": ``failed`` / ``parse_failed`` after three events, with
    the CLI itself exiting 0 and holding the answer.

    ``limit_bytes`` is in the message rather than in a separate ``evals`` field
    because the adapters already surface ``evals["error"]`` as
    ``"<type>: <message>"``, so putting the number there makes it reachable
    without threading a new field through two ``_finalise`` implementations.

    The message deliberately does NOT tell the reader to raise the limit. It is
    a private module constant, so "raise it" would be advice to patch library
    internals; the ceiling is a real backstop against a runaway line eating
    memory, and the actionable fix is on the payload side. If a legitimate
    workload needs more, that wants a public field with a considered default,
    not an operator editing this file.
    """

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(
            f"the CLI emitted a single stdout line larger than the {limit_bytes:,}-byte "
            "reader limit, so neither that line nor the rest of the run could be "
            "assembled. That is ONE NDJSON payload, not the whole stream: in practice "
            "it is a tool result carrying a very large file. Have the agent read the "
            "file in parts, or narrow what the tool returns."
        )
        self.limit_bytes = limit_bytes


class _CliDeadline:
    """Which bound expires next, and what to call it when it does.

    Split from the read loop because both cognitions need the identical
    bookkeeping and the phase transitions are the part that is easy to get
    subtly wrong: ``startup`` stops applying once ANY line arrives,
    ``first_event`` once a line produces an event (not merely a line — a
    ``system/init`` is the CLI talking to itself), and ``idle`` only starts
    applying after the first line, or every run would be measured for silence
    against a clock that started before the process did.
    """

    __slots__ = ("_idle_from", "_now", "_seen_event", "_seen_line", "_started", "_t")

    def __init__(self, timeouts: CliTimeouts, *, now: Callable[[], float] | None = None) -> None:
        self._t = timeouts
        self._now = now or time.monotonic
        self._started = self._now()
        self._idle_from = self._started
        self._seen_line = False
        self._seen_event = False

    def note_line(self) -> None:
        self._idle_from = self._now()
        self._seen_line = True

    def note_event(self) -> None:
        self._seen_event = True

    def next_deadline(self) -> tuple[str, float, float] | None:
        """``(reason, seconds_left, configured_limit)``, or ``None`` if nothing
        is currently bounded.

        ``None`` is the fast path and it matters: with only ``startup`` set —
        the default — every line after the first is unbounded, so the read loop
        can await the iterator directly instead of wrapping each line in a
        ``Task``. A token-streamed run is thousands of lines, and a task per
        line to enforce a bound nobody configured is pure overhead.
        """
        candidates: list[tuple[str, float, float]] = []
        if self._t.total is not None:
            candidates.append(("total_timeout", self._started + self._t.total, self._t.total))
        if not self._seen_line and self._t.startup is not None:
            candidates.append(("startup_timeout", self._started + self._t.startup, self._t.startup))
        if not self._seen_event and self._t.first_event is not None:
            candidates.append(
                ("first_event_timeout", self._started + self._t.first_event, self._t.first_event)
            )
        if self._seen_line and self._t.idle is not None:
            candidates.append(("idle_timeout", self._idle_from + self._t.idle, self._t.idle))
        if not candidates:
            return None
        reason, at, limit = min(candidates, key=lambda c: c[1])
        return reason, max(0.0, at - self._now()), limit


# How often the read loop surfaces so its caller can poll the cooperative
# cancellation token, when there is a ``Ctx`` carrying one.
#
# Not a field on :class:`CliTimeouts`, deliberately. Those are BOUNDS — things
# that end a run — and ``CliTimeouts.off()`` sets every one of them to ``None``.
# A cancellation cadence living there would mean "no timeouts" silently also
# meant "the stop button takes as long as the CLI feels like", which is exactly
# the bug this constant exists to fix.
#
# 250ms is chosen against the concurrency ceiling this module is built around
# (~200 live subprocesses): four wakeups per second per process is ~800/s at
# full load, which is nothing, and a quarter second is well inside what a
# person experiences as an immediate response to a stop button.
_CANCEL_POLL_S = 0.25


async def _iter_stdout(
    stdout: Any,
    deadline: _CliDeadline,
    *,
    poll_s: float | None = None,
) -> AsyncIterator[bytes | None]:
    """Iterate a process's stdout under :class:`CliTimeouts`.

    Raises :class:`CliTimedOut` when a bound is crossed; ends normally at EOF.
    The caller terminates the process — this function does not, because the
    two cognitions differ on what a dead process means (a Claude session's
    process IS its conversation; a Codex thread survives on disk) and a helper
    that killed things would take that decision away from them.

    ``poll_s`` makes the loop surface periodically with ``None`` — "nothing was
    read yet" — so the caller can check something the stream cannot tell it,
    which in practice is the cooperative cancellation token. Without it a
    caller only gets to look between LINES, so a CLI that has gone quiet
    mid-``Bash(npm install)`` cannot be stopped until it speaks again. Measured
    before this existed: a tripped token went unnoticed indefinitely against a
    silent process, and 83s against a real one.

    It never LENGTHENS a wait — a configured bound still fires exactly when it
    was set to — and a tick does not count as progress, so it cannot hold the
    ``idle`` bound open on a process that has genuinely stopped talking.

    **A tick cancels the pending read, and that is safe HERE for a reason worth
    stating.** ``asyncio.StreamReader`` keeps its buffer on the reader, and
    ``readline`` only drains it once a whole line is in hand, so a cancelled
    read leaves the bytes where they were and the next call finds them.
    Measured directly: 5000 lines dribbled out of a real subprocess with 73
    reads cancelled mid-flight arrived complete and in order. This is a
    property of the line protocol, NOT of the reader in general —
    ``readexactly`` accumulates into a local and DOES lose data when cancelled
    — so this loop must keep reading lines and nothing else.
    """
    iterator = stdout.__aiter__()
    while True:
        nxt = deadline.next_deadline()
        wait_s = nxt[1] if nxt is not None else None
        # A tick only happens when the poll cadence is the SOONER of the two,
        # which is what keeps a configured bound firing on its own schedule.
        ticking = poll_s is not None and (wait_s is None or poll_s < wait_s)
        if ticking:
            wait_s = poll_s
        try:
            if wait_s is None:
                # The fast path, and worth keeping: with nothing bounded and
                # nothing to poll for, a token-streamed run is thousands of
                # lines and this is the difference between 0.08µs and 1.5µs
                # each.
                line = await iterator.__anext__()
            else:
                line = await asyncio.wait_for(iterator.__anext__(), wait_s)
        except StopAsyncIteration:
            return
        except ValueError as exc:
            # ``StreamReader.readline`` raises a bare ``ValueError`` for exactly
            # one thing: the line outgrew the reader's buffer. (It converts the
            # internal ``LimitOverrunError`` to ``ValueError(e.args[0])``, and
            # that branch is the only ``ValueError`` in the method.) Naming it
            # here is what lets the adapters report a limit instead of a
            # sentence about separators.
            #
            # The limit is read off the READER rather than assumed from
            # ``_STDOUT_LINE_LIMIT``, because a caller-supplied ``spawn=`` may
            # have built the pipe with a different one — and a message quoting a
            # number the reader is not actually enforcing is worse than no
            # number at all. ``_limit`` is private, which is the trade: it is
            # the only place the true value exists, it has been stable across
            # CPython versions, and the fallback keeps a rename from turning a
            # diagnostic into a crash.
            enforced = getattr(stdout, "_limit", None)
            raise CliLineTooLong(
                enforced if isinstance(enforced, int) and enforced > 0 else _STDOUT_LINE_LIMIT
            ) from exc
        except TimeoutError:
            if ticking:
                yield None  # nothing read; let the caller look around
                continue
            assert nxt is not None  # only reachable from the bounded branch
            raise CliTimedOut(nxt[0], nxt[2]) from None
        deadline.note_line()
        yield line



# How long the diagnostics are worth waiting for once the run itself is over.
# Generous, because on every healthy path this returns instantly: the process
# has exited, its pipe is at EOF, and the read completes without suspending.
# The bound only engages when something is still HOLDING the pipe.
_STDERR_DRAIN_S = 2.0



# Above this many BYTES, a text payload goes to a file instead of into argv.
#
# The kernel copies argv AND the environment into the new process image, and
# that copy is bounded twice over:
#
#   darwin   ARG_MAX = 1,048,576 bytes for argv + envp COMBINED. Measured on
#            2026-09-04: one 1,000,000-byte argument spawns fine, twenty
#            100,000-byte ones do not — it is a shared pot, not a per-argument
#            cap, and your environment is spending from it too.
#   Linux    the same total, PLUS MAX_ARG_STRLEN = 32 pages = 131,072 bytes for
#            any SINGLE argument. That is the tighter of the two and the one to
#            design against: a payload that spawns happily on a Mac can be
#            rejected outright on the CI runner.
#
# 32 KiB leaves 4x headroom under the per-argument cap, and leaves the shared
# pot free for the payloads that have NO file transport to escape to —
# ``--agents`` and ``--json-schema``, which the CLI will only accept inline.
#
# BYTES, not characters: ``len(text)`` on a str with any non-ASCII in it
# understates what execve actually copies, by up to 4x.
_ARGV_TEXT_LIMIT = 32 * 1024


def _too_big_for_argv(text: str) -> bool:
    """Would ``text`` be reckless to pass as a command-line argument?"""
    return len(text.encode("utf-8", errors="replace")) > _ARGV_TEXT_LIMIT


async def _drain_stderr(stderr: Any, timeout_s: float = _STDERR_DRAIN_S) -> bytes:
    """Read stderr to EOF, but never wait forever for it.

    ``await proc.stderr.read()`` waits for EOF, and EOF needs EVERY writer to
    close the pipe — not just the process we spawned. A CLI that leaves a
    helper behind holding an inherited stderr therefore hangs this read
    forever, and because it runs from the ``finally`` of the driver it hangs
    the whole run: past the liveness bounds, past cancellation, past anything
    the caller can do. Measured against codex 0.152.1, which spawns a
    ``codex-code-mode-host`` helper: the run reached EOF on stdout, entered the
    finally, and never came out. Roughly one run in three, which is exactly the
    profile of a bug that gets blamed on the model.

    Read in CHUNKS rather than as one bounded ``read()``. ``read()`` with no
    size accumulates into a local, so cancelling it discards everything it had
    — and this is the diagnostic text explaining a failure, which is the worst
    possible moment to return nothing. ``read(n)`` returns as soon as any data
    is available, so a timeout costs at most the chunk in flight.

    Never raises: a run's terminal event must not depend on the quality of its
    stderr.
    """
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            chunk = await asyncio.wait_for(stderr.read(65536), remaining)
        except (TimeoutError, asyncio.CancelledError):
            break
        except Exception:
            break
        if not chunk:
            break  # EOF, the normal ending
        chunks.append(chunk)
    return b"".join(chunks)


def _timeout_stop_reason(
    exc: CliTimedOut, stderr_bytes: bytes, auth_markers: tuple[str, ...]
) -> str:
    """Refine a crossed bound into a more specific reason where stderr allows.

    Only ``startup_timeout`` is ever refined, and only to
    ``authentication_failed``. That narrowness is the point: this is a substring
    match on human-facing text a vendor can reword at any time, so it may only
    ever make an ALREADY-FAILED run's label more useful. It cannot turn a
    working run into a failure, and it cannot mask a different failure — the
    fallback is the bound's own name, which is always correct.

    It earns its place because it names the single most common cause of the
    single worst symptom. Measured: ``claude -p`` with an invalid
    ``ANTHROPIC_API_KEY`` emitted no stdout and no exit for 45+ seconds; the run
    looked active, and ``startup_timeout`` alone would send an operator looking
    at the network or the model when the answer was a credential.

    The markers must be phrases a CLI prints when auth has FAILED, never ones it
    prints while working. The Claude CLI's "ANTHROPIC_API_KEY ... takes
    precedence over your claude.ai login" is the trap: it appears on perfectly
    successful runs, and using it here would relabel every unrelated startup
    hang as an auth problem.
    """
    if exc.reason != "startup_timeout" or not stderr_bytes:
        return exc.reason
    haystack = stderr_bytes.decode("utf-8", errors="replace").lower()
    if any(marker in haystack for marker in auth_markers):
        return "authentication_failed"
    return exc.reason



# ─────────────────────────────────────────────────────────────────────────────
# Structured-output failures, as data.
#
# The information was always there and always destroyed on the way out. The
# adapters produce per-field diagnostics — ``OutputCoercionError.errors`` is a
# list of ``"path: message"`` strings for every flavour (Pydantic, dataclass,
# attrs, raw JSON Schema) — and this module joined them into one sentence and
# put it in ``evals["structured_output_error"]``. A caller wanting to show a
# user WHICH field was wrong, or to count failures by path across a fleet, had
# to parse English back out of a string that a library upgrade can reword.
# ─────────────────────────────────────────────────────────────────────────────

# Root sentinel used by ``JsonSchemaAdapter`` when a violation is not
# attributable to one field ("'z' is a required property" belongs to the
# object, not to ``z``).
_ROOT_PATHS = ("<root>", "")

# Caps. ``evals`` is deep-frozen, checkpointed and often logged, and a fan-out
# over a large list can produce hundreds of violations that all say the same
# thing. Twenty is well past the point a repair prompt stays useful.
_MAX_VIOLATIONS = 20
_MAX_RAW_CHARS = 800


@dataclass(frozen=True, slots=True)
class SchemaViolation:
    """One field that did not satisfy the declared schema.

    ``path`` is JSONPath-ish (``$.lines[0].qty``) rather than the adapters'
    dotted form (``lines.0.qty``) because the dotted form is ambiguous the
    moment a mapping has a numeric-looking key, and because ``$``-rooted paths
    are what a caller can hand to ``jq``, a UI, or an error-grouping key.
    """

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass(frozen=True, slots=True)
class StructuredOutputFailure:
    """Why a run that declared an output type did not produce one.

    Four kinds, and they are not interchangeable — each has a different fix:

    ``missing``
        The CLI returned no structured payload at all despite being given a
        schema. The docs for both binaries are explicit that this is a failure
        rather than "the model chose prose".
    ``undecodable``
        There was a payload and it was not JSON.
    ``schema_mismatch``
        It was JSON and it did not fit the declared type. The only kind that
        carries ``violations``.
    ``retries_exhausted``
        The CLI validated against the schema itself, re-prompted itself, and
        gave up. agentkit never saw the intermediate attempts.

    Lives in ``evals["structured_output_failure"]`` **as a dict**, not as this
    object: ``evals`` is deep-frozen, checkpointed and serialised, and putting a
    dataclass in it would make a result that cannot round-trip through JSON.
    :meth:`of` reads it back into this type when a caller wants the typed view.
    """

    kind: str
    detail: str
    violations: tuple[SchemaViolation, ...] = ()
    raw_excerpt: str | None = None
    truncated: bool = False

    def __str__(self) -> str:
        if not self.violations:
            return self.detail
        return f"{self.detail} — " + "; ".join(str(v) for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe form that goes into ``evals``."""
        out: dict[str, Any] = {
            "kind": self.kind,
            "detail": self.detail,
            "violations": [{"path": v.path, "message": v.message} for v in self.violations],
        }
        if self.raw_excerpt is not None:
            out["raw_excerpt"] = self.raw_excerpt
        if self.truncated:
            out["truncated"] = True
        return out

    @classmethod
    def of(cls, evals: Mapping[str, Any]) -> StructuredOutputFailure | None:
        """The typed view of ``evals["structured_output_failure"]``, or ``None``.

        The read side of the contract, so an application branches on a type
        instead of re-parsing the sentence it was handed::

            failure = StructuredOutputFailure.of(result.evals)
            if failure is not None:
                for v in failure.violations:
                    form.mark_invalid(v.path, v.message)
        """
        raw = evals.get("structured_output_failure")
        if not isinstance(raw, Mapping):
            return None
        violations = tuple(
            SchemaViolation(path=str(v.get("path", "$")), message=str(v.get("message", "")))
            for v in raw.get("violations", ())
            if isinstance(v, Mapping)
        )
        excerpt = raw.get("raw_excerpt")
        return cls(
            kind=str(raw.get("kind", "schema_mismatch")),
            detail=str(raw.get("detail", "")),
            violations=violations,
            raw_excerpt=str(excerpt) if excerpt is not None else None,
            truncated=bool(raw.get("truncated", False)),
        )

    @classmethod
    def from_coercion_error(cls, exc: BaseException) -> StructuredOutputFailure:
        """Build one from an ``OutputCoercionError`` without importing it.

        Read structurally (``getattr``) rather than by type for the same reason
        the rest of this module does: a test double or a caller's own adapter
        may raise something that merely quacks like one, and a failure to
        recognise it would replace per-field diagnostics with a bare class name
        — which is precisely the outcome this type exists to end.
        """
        errors = [str(e) for e in (getattr(exc, "errors", None) or ())]
        violations = tuple(_parse_violation(e) for e in errors[:_MAX_VIOLATIONS])
        raw = getattr(exc, "raw", None)
        excerpt: str | None = None
        truncated = len(errors) > _MAX_VIOLATIONS
        if raw is not None:
            text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            if len(text) > _MAX_RAW_CHARS:
                text = text[:_MAX_RAW_CHARS] + "…"
                truncated = True
            excerpt = text
        return cls(
            kind="schema_mismatch",
            detail=f"{type(exc).__name__}: {exc}",
            violations=violations,
            raw_excerpt=excerpt,
            truncated=truncated,
        )

    def repair_prompt(self) -> str:
        """Compact, model-readable feedback naming exactly what to change.

        Deliberately NOT wired into an automatic retry here — see the
        cognitions' structured-output notes. The Claude CLI already re-prompts
        ITSELF against ``--json-schema`` and reports
        ``error_max_structured_output_retries`` when it gives up, so an
        agentkit-side loop on top of that is a second retry layer over an opaque
        one: double the spend, double the latency, and usage accounting that no
        longer maps to attempts. This method is for the caller who wants to run
        the repair turn deliberately, and for putting a useful message in front
        of a person.
        """
        if not self.violations:
            return (
                "Your previous response did not produce the required structured "
                f"output ({self.detail}). Return ONLY a JSON object matching the "
                "declared schema."
            )
        lines = "\n".join(f"- {v.path}: {v.message}" for v in self.violations)
        more = "\n- (further violations omitted)" if self.truncated else ""
        return (
            "Your previous response did not match the required schema. Fix exactly "
            f"these fields and return ONLY the corrected JSON object:\n{lines}{more}"
        )


def _parse_violation(entry: str) -> SchemaViolation:
    """``"lines.0.qty: Input should be…"`` → ``$.lines[0].qty`` + the message.

    Splits on the FIRST ``": "`` only, because messages contain colons
    ("Input should be a valid integer, unable to parse string as an integer"
    does not, but ``"got {'a': 1}"`` does) and splitting on the last one moves
    half the message into the path.

    An entry with no separator at all becomes a root-scoped violation carrying
    the whole string. That is the honest reading — an adapter that did not
    attribute its complaint to a field has told us the OBJECT is wrong — and it
    is better than inventing a path or dropping the diagnostic.
    """
    head, sep, tail = entry.partition(": ")
    if not sep:
        return SchemaViolation(path="$", message=entry.strip())
    return SchemaViolation(path=_json_path(head.strip()), message=tail.strip())


def _json_path(dotted: str) -> str:
    """``lines.0.qty`` → ``$.lines[0].qty``; the root sentinels → ``$``."""
    if dotted in _ROOT_PATHS:
        return "$"
    out = "$"
    for segment in dotted.split("."):
        if segment.isdigit():
            out += f"[{segment}]"
        else:
            out += f".{segment}"
    return out



# ─────────────────────────────────────────────────────────────────────────────
# The process lifecycle, once.
#
# Measured before extracting it: ``ClaudeCliCognition.drive`` and
# ``CodexCliCognition._run_once`` were 60% identical, 85 code lines in common,
# and the common part was exactly the delicate part — the spawn under the
# concurrency guard, the stdin write, the read loop, the cancellation and
# timeout branches, the ``finally`` that terminates the process group and
# drains stderr, and the ``should_reraise_cancel`` dance that makes an outer
# ``wait_for`` still raise ``TimeoutError``.
#
# That is the code where a divergence is invisible and expensive. This module's
# own docstring already lists three bugs that came from copying it: a
# ``_terminate`` that forgot the SIGKILL escalation, a ``_parse_line`` that
# caught the wrong exception type, a terminal event that swallowed
# ``KeyboardInterrupt``. Each was fixed on one side and not the other.
#
# What is NOT here, deliberately: argv construction, the event vocabulary,
# capabilities, and the persistent-session loops. ``ClaudeCliSession._turn``
# ends a turn at a ``result`` payload rather than at EOF and multiplexes a
# control protocol on the same stream; ``CodexCliSession`` re-spawns per turn.
# Forcing those together would be the "mostly the same abstraction" this
# module's docstring warns is worse than two honest copies.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _CliLaunch:
    """Everything the spawn needs that the adapter alone can compute."""

    argv: list[str]
    env: dict[str, str]
    cwd: str | None = None
    # Written to the child's stdin, which is then closed. Both CLIs take the
    # prompt this way; the ENCODING differs (claude wants one stream-json user
    # turn, codex wants the raw text) and that is the adapter's business.
    stdin_payload: bytes = b""


@dataclass(slots=True)
class _CliRunOutcome:
    """What happened to the process, for the adapter's ``_finalise``.

    Mutable and caller-owned because the driver is an async GENERATOR: it
    yields events as they arrive and cannot also return a value. The caller
    constructs this, passes it in, and reads it after the loop.
    """

    cancelled: bool = False
    timed_out: CliTimedOut | None = None
    fatal_exc: BaseException | None = None
    should_reraise_cancel: bool = False
    stderr_bytes: bytes = b""
    spawned: bool = False
    # ``-1`` = never spawned; ``None`` = still alive. Both mean "the exit code
    # says nothing about this turn", which is what ``_finalise`` expects.
    return_code: int | None = -1


async def _run_cli_process(
    *,
    prepare: Callable[[], _CliLaunch],
    handle: Callable[[dict[str, Any]], AsyncIterator[StreamEvent]],
    spawn: CliSpawn | None,
    semaphore: asyncio.BoundedSemaphore,
    timeouts: CliTimeouts,
    terminate_grace_s: float,
    ctx: Ctx | None,
    outcome: _CliRunOutcome,
) -> AsyncIterator[StreamEvent]:
    """Spawn one CLI, stream its events, and record how it ended.

    Never raises for an ordinary failure — everything lands on ``outcome`` so
    the adapter can still emit its one terminal ``final`` event. The two
    exceptions to that are the two that must propagate, and they propagate
    AFTER the caller has emitted that event: see ``should_reraise_cancel`` and
    :func:`_reraise_if_not_an_exception`.

    ``prepare`` is a callback rather than pre-computed arguments so that argv
    and env resolution happens INSIDE the guarded block. Both adapters can
    throw there — a malformed ``Prompt`` template, a budget already exhausted,
    a schema that will not render — and those are failures a caller must
    receive as a terminal event like any other, not as an exception from a
    generator that never yielded.

    ``handle`` turns one parsed stdout payload into zero or more events, and
    owns the adapter's own state folding. The driver stays out of the event
    vocabulary entirely; all it does with an event is note it against the
    ``first_event`` deadline and pass it on.
    """
    proc: asyncio.subprocess.Process | None = None
    try:
        launch = prepare()
        async with semaphore:
            proc = await (spawn or asyncio.create_subprocess_exec)(
                *launch.argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=launch.cwd,
                env=launch.env,
                start_new_session=START_NEW_SESSION,
                limit=_STDOUT_LINE_LIMIT,
            )
            outcome.spawned = True
            assert proc.stdout is not None  # PIPE
            assert proc.stdin is not None  # PIPE
            # The prompt, then EOF. Closing stdin is not a tidy-up: it is how
            # both CLIs are told the input is complete, and without it a
            # one-shot run waits for a turn that is never coming.
            proc.stdin.write(launch.stdin_payload)
            await proc.stdin.drain()
            with contextlib.suppress(Exception):
                proc.stdin.close()

            deadline = _CliDeadline(timeouts)
            # Poll only when there is something to poll FOR. No ctx means no
            # cancellation token, so the read stays on the allocation-free
            # fast path rather than paying a wakeup cadence nobody reads.
            poll_s = _CANCEL_POLL_S if ctx is not None else None
            try:
                async for line in _iter_stdout(proc.stdout, deadline, poll_s=poll_s):
                    if ctx is not None:
                        try:
                            ctx.check_cancelled()
                        except Exception:
                            outcome.cancelled = True
                            break
                    if line is None:
                        continue  # a poll tick — the check above was the point
                    payload = _parse_line(line)
                    if payload is None:
                        continue
                    async for event in handle(payload):
                        deadline.note_event()
                        yield event
            except CliTimedOut as exc:
                outcome.timed_out = exc
            finally:
                if (
                    outcome.cancelled or outcome.timed_out is not None
                ) and proc.returncode is None:
                    # The GROUP, not the pid. A cancelled or timed-out run is
                    # exactly the one whose CLI may have forked something
                    # expensive and stopped reporting on it.
                    await _terminate(proc, terminate_grace_s)
                if proc.stderr is not None:
                    outcome.stderr_bytes = await _drain_stderr(proc.stderr)
                with contextlib.suppress(Exception):
                    # Bounded for the same reason as the read above: this runs
                    # from a ``finally`` and a ``finally`` that can hang turns
                    # every failure into a hang.
                    await asyncio.wait_for(proc.wait(), _STDERR_DRAIN_S)
    except asyncio.CancelledError:
        outcome.cancelled = True
        outcome.should_reraise_cancel = True
        if proc is not None and proc.returncode is None:
            await _terminate(proc, terminate_grace_s)
    except BaseException as exc:  # noqa: BLE001 — terminal-event guarantee
        # Widened past ``Exception`` so a ``KeyboardInterrupt`` or ``SystemExit``
        # mid-run still produces the adapter's terminal event. The caller
        # re-raises it afterwards via ``_reraise_if_not_an_exception``; that
        # second half was missing once, and Ctrl-C came back as a tidy
        # ``AgentResult`` while the interpreter kept running.
        outcome.fatal_exc = exc
    outcome.return_code = proc.returncode if proc is not None else -1



# The pre-flight working-directory check, and the matcher that reads its
# result back. One definition, because the STRING is load-bearing: both
# ``_finalise`` implementations classify the failure by
# ``str(exc).startswith(...)`` to tell ``working_dir_missing`` apart from
# ``spawn_failed``, and there were four copies of the literal across the two
# modules. A typo in the raiser or in either matcher would not fail a type
# check or a lint — it would silently reclassify a mistyped path as a missing
# binary, which is the one distinction this branch exists to draw.
_WORKING_DIR_MISSING = "working_dir does not exist:"


def _require_working_dir(path: Any) -> None:
    """Raise ``FileNotFoundError`` if ``path`` is set and is not there.

    Checked before the spawn rather than left to ``create_subprocess_exec``,
    which raises ``FileNotFoundError`` for a missing ``cwd`` AND for a missing
    binary — the same exception for two problems with different fixes.
    """
    if path is not None and not Path(path).exists():
        raise FileNotFoundError(f"{_WORKING_DIR_MISSING} {path}")


def _is_working_dir_missing(exc: BaseException | None) -> bool:
    """Did :func:`_require_working_dir` raise this?"""
    return isinstance(exc, FileNotFoundError) and str(exc).startswith(_WORKING_DIR_MISSING)



# ─────────────────────────────────────────────────────────────────────────────
# Secrets.
#
# Two different exposures, and they need different answers.
#
# ARGV. Both cognitions accept configuration that can carry a credential and
# render it into the argument list: ``claude --mcp-config '<inline JSON>'`` and
# ``--settings '<inline JSON>'``, ``codex -c model_providers.x.api_key=...``.
# An argument list is world-readable — any user on the box can ``ps`` it — so
# that is a credential disclosed to every local account. This is the same
# problem as the prompt-in-argv one, except the payload IS the secret. Where
# the CLI accepts a file instead (claude does, for both), the blob is written
# to a 0600 file and the PATH is passed. Where it does not (codex's ``-c``),
# the caller is warned and pointed at the mechanism that does work.
#
# DIAGNOSTICS. ``evals["stderr"]`` is the CLI's stderr verbatim, and an
# ``AgentResult`` is checkpointed, logged and fanned out to observers. A CLI
# that prints a failing request's ``Authorization`` header — which is ordinary
# diagnostic behaviour — would have that header persisted. Redaction here is
# defence in depth: it cannot be complete, so it must not be relied on as a
# boundary, but a stored credential is worth catching cheaply.
# ─────────────────────────────────────────────────────────────────────────────

# High-confidence credential shapes. Ordered most-specific first so a token
# that matches two patterns is replaced by the more informative label.
#
# The bar for adding one is that a FALSE POSITIVE must be tolerable: this text
# is what an operator reads to diagnose a failed run, and a redactor that eats
# ordinary words makes the diagnostic useless — which is how redaction gets
# switched off wholesale. Every pattern below therefore requires a
# credential-shaped prefix or a long opaque run, never a bare word.
_SECRET_PATTERNS: tuple[tuple[Any, str], ...] = tuple(
    (__import__("re").compile(pattern), label)
    for pattern, label in (
        (r"sk-ant-[A-Za-z0-9_\-]{12,}", "[redacted:anthropic-key]"),
        (r"sk-[A-Za-z0-9_\-]{16,}", "[redacted:api-key]"),
        (r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}", "[redacted:github-token]"),
        (r"github_pat_[A-Za-z0-9_]{20,}", "[redacted:github-token]"),
        (r"xox[baprs]-[A-Za-z0-9-]{10,}", "[redacted:slack-token]"),
        (r"AKIA[0-9A-Z]{16}", "[redacted:aws-key-id]"),
        # JWT: three base64url runs separated by dots, starting with the
        # ``eyJ`` that every ``{"`` header encodes to.
        (r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", "[redacted:jwt]"),
    )
)

# The catch-all, for credential formats no list can enumerate. Only the VALUE
# is replaced — the key stays, because "which secret leaked" is exactly what an
# operator needs and is not itself sensitive.
_SECRET_ASSIGNMENT = __import__("re").compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|bearer|"
    r"secret|password|passwd|token)\b(\s*[:=]\s*|\s+)"
    r"(?:Bearer\s+)?([A-Za-z0-9._\-+/]{12,}={0,2})"
)


def _redact_secrets(text: str) -> str:
    """Replace credential-shaped runs in text destined for storage.

    Defence in depth, explicitly NOT a boundary. It cannot know every
    credential format, so a caller must not conclude that redacted output is
    safe to publish — it is a cheap way to stop the common shapes reaching a
    checkpoint, a log aggregator or an observer fan-out, and nothing more.

    Structure-preserving on purpose: the KEY in ``api_key=…`` survives and only
    the value is replaced. An operator reading a failed run needs to know which
    credential was involved, and the name of a secret is not the secret.
    """
    if not text:
        return text
    for pattern, label in _SECRET_PATTERNS:
        text = pattern.sub(label, text)
    redacted: str = _SECRET_ASSIGNMENT.sub(
        lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text
    )
    return redacted


def _looks_secret(key: str, value: Any) -> bool:
    """Would ``key=value`` put a credential in an argument list?

    Keyed on the NAME rather than the value, because a config key called
    ``api_key`` is a credential whatever it happens to hold, and because
    guessing from the value alone is how a redactor starts eating model names.
    """
    lowered = key.lower()
    if not any(
        marker in lowered
        for marker in ("api_key", "apikey", "token", "secret", "password", "passwd", "credential")
    ):
        return False
    # ``*_env_var`` keys name an environment variable to read the secret FROM;
    # that is the documented safe pattern, not a leak.
    return not lowered.endswith(("_env_var", "_envvar", "_env"))



def _make_scratch() -> str:
    """A 0700 directory for this run's generated files."""
    import tempfile

    path = tempfile.mkdtemp(prefix="agentkit-cli-")
    os.chmod(path, 0o700)
    return path


def _write_private(directory: str, name: str, text: str) -> Path:
    """Write ``text`` to ``directory/name`` readable only by this user.

    ``0o600`` set BEFORE the content is written, via ``os.open`` rather than
    ``Path.write_text`` + ``chmod``. The two-step version leaves a window in
    which the file exists with the process umask's permissions and already has
    the secret in it — short, but this is the file that exists precisely
    because the alternative (an argument list) was world-readable, and closing
    one disclosure by opening a smaller one is not a fix.
    """
    path = Path(directory) / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _is_inline_json(value: str) -> bool:
    """Is this an inline JSON blob rather than a path?

    Both CLIs accept either for these options. The distinction is structural
    and unambiguous — a filesystem path does not start with ``{`` — so no
    filesystem probe is needed, which matters because the value may name a file
    that does not exist yet or is not readable from here.
    """
    return value.lstrip().startswith("{")



# ─────────────────────────────────────────────────────────────────────────────
# Reading a payload written by somebody else.
#
# Both parsers used ``payload.get("x") or {}`` throughout, which guards a
# MISSING field and a ``None`` one and nothing else. Handed a field of the
# wrong type — ``{"type": "assistant", "message": true}`` — the very next
# ``.get`` raised ``AttributeError`` out of the parser, ``drive`` reported
# ``parse_failed``, and the run ended having lost its terminal payload and
# charged $0.00 to the budget. That is the failure mode this module's docstring
# already describes for one undecodable byte; a wrong-typed field is the same
# bug through a different door.
#
# Found by the property tests, not by review: nobody writes
# ``{"message": true}`` into an example. The parsers' own comment promises
# "forward-compat is don't crash", and these are what make that true rather
# than aspirational.
# ─────────────────────────────────────────────────────────────────────────────


def _as_dict(value: Any) -> dict[str, Any]:
    """``value`` if it is a mapping, else an empty dict."""
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    """``value`` if it is a list, else an empty list.

    A ``str`` is deliberately NOT treated as a sequence: iterating one yields
    characters, and a parser looping over a string field would emit an event
    per letter rather than skipping a malformed payload.
    """
    return list(value) if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    """A whole number, or ``default``.

    ``bool`` is excluded because ``isinstance(True, int)`` is ``True`` in
    Python and a token count of ``True`` would silently become ``1``.
    Negatives are clamped at zero by the callers that need it, not here — this
    is a type gate, and ``duration_ms`` may legitimately be any integer.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return default if (math.isnan(value) or math.isinf(value)) else int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _as_cost(value: Any) -> float:
    """A number a ledger can add up: finite and non-negative, or ``0.0``.

    NaN is the one that matters. It is not a wrong number — it is a value that
    makes every subsequent comparison in the ledger false, so a budget can no
    longer answer whether it is over. A negative cost is clamped for the mirror
    reason: it would CREDIT the budget, handing headroom back to a run that
    just spent money.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if not isinstance(value, str):
            return 0.0
        try:
            value = float(value)
        except ValueError:
            return 0.0
    number = float(value)
    if math.isnan(number) or math.isinf(number) or number < 0.0:
        return 0.0
    return number



__all__ = [
    "CliSpawn",
    "CliLineTooLong",
    "CliTimedOut",
    "CliTimeouts",
    "EnvPolicy",
    "InvalidSchemaError",
    "SchemaViolation",
    "StructuredOutputFailure",
    "START_NEW_SESSION",
    "_CliCall",
    "_CliLaunch",
    "_CliRunOutcome",
    "_as_cost",
    "_as_dict",
    "_as_int",
    "_as_list",
    "_build_child_env",
    "_CANCEL_POLL_S",
    "_ARGV_TEXT_LIMIT",
    "_charge_meters",
    "_drain_stderr",
    "_STDOUT_LINE_LIMIT",
    "_too_big_for_argv",
    "_coerce_structured",
    "_CliDeadline",
    "_get_semaphore",
    "_is_inline_json",
    "_is_working_dir_missing",
    "_iter_stdout",
    "_require_working_dir",
    "_timeout_stop_reason",
    "_middleware_name",
    "_looks_secret",
    "_make_scratch",
    "_parse_line",
    "_redact_secrets",
    "_reraise_if_not_an_exception",
    "_terminate",
    "_tool_middleware_names",
    "_run_cli_process",
    "_validate_json_schema",
    "_write_private",
]
