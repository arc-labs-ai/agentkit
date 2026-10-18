"""The shared CLI plumbing, exercised only when something has already gone wrong.

WHY THIS FILE EXISTS
--------------------
``agentkit/models/providers/cli/_common.py`` exists because its own docstring
names three bugs that came from copying it: a ``_terminate`` that forgot the
SIGKILL escalation, a ``_parse_line`` that caught ``JSONDecodeError`` instead of
``ValueError``, a terminal event that swallowed ``KeyboardInterrupt``. Every one
of them lives on a path that runs ONLY after a failure, which is why none of
them failed a test: on the happy path the module is a thin pipe, and the happy
path is what the rest of the CLI suite drives.

So each test below pins a DECISION the module makes about a run that is already
lost, and each decision has a way of being wrong that produces no error:

* a line that cannot be decoded ends the whole run instead of being skipped, and
  because the reader never reaches the terminal payload the run reports ``$0.00``
  to the budget — measured, and the reason ``ValueError`` and not
  ``JSONDecodeError`` is in the ``except`` clause;
* a ``KeyboardInterrupt`` mid-run is delivered as a tidy result and never
  propagates, so the interpreter keeps running and the caller's own
  ``except KeyboardInterrupt`` never fires;
* a cancelled run signals one pid, so the ``npm install`` the CLI forked keeps
  running — holding its own copy of the pipe — until it finishes on its own;
* a group kill aimed at a child that was never put in its own session signals
  the SERVICE's process group instead, which is the difference between a leak
  and a catastrophe;
* a ``finally`` that reads stderr to EOF hangs forever on a helper that
  inherited the pipe, turning every failure into a hang;
* a bound that fires is reported as the generic name instead of the specific one
  (``startup_timeout`` where stderr says the credential was refused), sending an
  operator to look at the network;
* a schema the CLI would reject at startup is discovered two to five seconds
  into a subprocess, with the complaint on a stderr this cognition surfaces only
  on a run that already failed.

The process lifecycle is tested at two levels on purpose. Most tests use a
hand-written process double, because the decisions are about branching and a
double makes the branch addressable. The group-termination test uses a REAL
local ``sys.executable`` subprocess with a real grandchild, because a double
cannot have grandchildren and "we signalled the group" is only a claim until
something that was not the direct child dies.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal
import sys
import time
from typing import Any

import pytest

from agentkit.models.providers.cli._common import (
    CliLineTooLong,
    CliTimedOut,
    CliTimeouts,
    InvalidSchemaError,
    SchemaViolation,
    StructuredOutputFailure,
    _as_cost,
    _as_int,
    _build_child_env,
    _child_pgid,
    _CliDeadline,
    _CliLaunch,
    _CliRunOutcome,
    _drain_stderr,
    _iter_stdout,
    _parse_line,
    _process_exit_stop_reason,
    _redact_secrets,
    _require_working_dir,
    _reraise_if_not_an_exception,
    _run_cli_process,
    _terminate,
    _timeout_stop_reason,
    _validate_json_schema,
)

# ── process doubles ──────────────────────────────────────────────────────────
# Hand-written rather than ``FakeClaudeCli``, because the subject here is the
# driver's branching and not a CLI's event vocabulary: a stdout that never
# yields, one that raises the reader's buffer error, and a cleanup that throws
# are not shapes a recording can express.


class _Stderr:
    """``proc.stderr``, draining once like the real pipe."""

    def __init__(self, data: bytes = b"") -> None:
        self._buf = bytearray(data)

    async def read(self, n: int = -1) -> bytes:
        del n
        out, self._buf = bytes(self._buf), bytearray()
        return out


class _Stdin:
    def __init__(self) -> None:
        self.written = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _Stdout:
    """An async line iterator over a fixed list of behaviours.

    ``bytes`` is a line, an exception instance is raised instead, and ``None``
    means "never speak again" — the shape a hung CLI has, which is the only one
    a liveness bound can be observed against.

    The cursor advances only once a step has actually been DELIVERED, which
    mirrors the property ``_iter_stdout`` is built on and documents: a
    ``StreamReader`` keeps its buffer on the reader, so a read cancelled by a
    poll tick leaves the bytes where they were and the next call finds them. A
    double that consumed its step before awaiting would report EOF after the
    first tick and make every polling test pass for the wrong reason.
    """

    def __init__(self, *steps: bytes | BaseException | None) -> None:
        self._steps = list(steps)
        self._at = 0

    def __aiter__(self) -> _Stdout:
        return self

    async def __anext__(self) -> bytes:
        if self._at >= len(self._steps):
            raise StopAsyncIteration
        step = self._steps[self._at]
        if step is None:
            await asyncio.Event().wait()  # a process that has gone silent
            raise AssertionError("unreachable")
        self._at += 1
        if isinstance(step, BaseException):
            raise step
        return step


class _Proc:
    """The ``asyncio.subprocess.Process`` surface ``_run_cli_process`` touches."""

    def __init__(self, stdout: _Stdout, *, stderr: bytes = b"", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = _Stderr(stderr)
        self.stdin = _Stdin()
        self.returncode: int | None = None
        self.signals: list[int] = []
        self._exit_code = exit_code

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.signals.append(signal.SIGTERM)
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.signals.append(signal.SIGKILL)
        if self.returncode is None:
            self.returncode = -9


def _spawn_returning(proc: Any) -> Any:
    async def spawn(*_argv: str, **_kwargs: Any) -> Any:
        return proc

    return spawn


async def _no_events(_payload: dict[str, Any]) -> Any:
    """A ``handle`` that maps every payload to nothing.

    The tests in this file are about the process lifecycle, not the event
    vocabulary, so the adapter half is deliberately inert.
    """
    if False:  # pragma: no cover - a generator that yields nothing
        yield None


async def _run(
    proc: Any,
    *,
    timeouts: CliTimeouts | None = None,
    ctx: Any = None,
    grace_s: float = 0.05,
    prepare: Any = None,
) -> _CliRunOutcome:
    """Drive one ``_run_cli_process`` to completion and hand back the outcome."""
    outcome = _CliRunOutcome()
    agen = _run_cli_process(
        prepare=prepare or (lambda: _CliLaunch(argv=["fake-cli"], env={}, stdin_payload=b"go\n")),
        handle=_no_events,
        spawn=_spawn_returning(proc),
        semaphore=asyncio.BoundedSemaphore(1),
        timeouts=timeouts or CliTimeouts.off(),
        terminate_grace_s=grace_s,
        ctx=ctx,
        outcome=outcome,
    )
    async with contextlib.aclosing(agen) as events:
        async for _ in events:
            pass
    return outcome


class _StopButton:
    """A ``Ctx``-shaped stub carrying nothing but cooperative cancellation."""

    def __init__(self, *, after_lines: int = 0) -> None:
        self.checks = 0
        self._after = after_lines

    def check_cancelled(self) -> None:
        self.checks += 1
        if self.checks > self._after:
            raise RuntimeError("the caller pressed stop")


# ── one undecodable byte must not end the run ────────────────────────────────


@pytest.mark.parametrize(
    ("line", "why"),
    [
        (b"\n", "a blank warm-up line"),
        (b"   \t \n", "whitespace only"),
        (b"warning: could not stat /tmp/x\n", "a human diagnostic on stdout"),
        (b'{"type":"result",\n', "a process killed mid-object"),
        (b"\xff\xfe{bad utf-16 bom\n", "a UTF-16 BOM whose body does not decode"),
        (b"\xef\xbb\xbf{bad utf-8 bom\n", "a UTF-8 BOM whose body does not decode"),
        (b"warning: caf\xe9 is latin-1\n", "a latin-1 filename in a warning"),
        (b"[1, 2, 3]\n", "valid JSON that is not an object"),
        (b'"a bare string"\n', "valid JSON that is not an object"),
    ],
)
def test_a_line_that_is_not_a_json_object_is_skipped_rather_than_ending_the_run(line: bytes, why: str) -> None:
    """``_parse_line`` returns ``None`` — it never raises — for every shape of
    non-payload a real binary puts on stdout.

    The ``except ValueError`` is load-bearing and is NOT interchangeable with
    ``except JSONDecodeError``: ``json.loads`` on BYTES sniffs the encoding
    first, so a line opening with a BOM whose body does not decode comes back as
    a ``UnicodeDecodeError`` — a ``ValueError`` that is not a
    ``JSONDecodeError``. Narrow the clause and that exception escapes into
    ``drive``'s ``except BaseException``: measured, one undecodable byte ended
    the whole run with ``parse_failed``, no output, and — because the reader
    never got past that line — a completed run charging ``$0.00`` to the budget.
    Both binaries print human diagnostics on these streams and filenames are
    arbitrary bytes, so this needs nothing exotic to happen.
    """
    assert _parse_line(line) is None, why


def test_a_json_object_survives_the_same_guard_that_skips_everything_else() -> None:
    """The other half: the guard must not be so wide that it eats real payloads.

    A ``_parse_line`` that returned ``None`` for everything would pass every
    test above and produce a run that exits 0, emits no event and reports an
    empty answer with no stop reason — the failure shape this module calls
    "going quiet" rather than failing.
    """
    assert _parse_line(b'{"type":"result","subtype":"success"}\n') == {"type": "result", "subtype": "success"}


# ── the terminal-event guarantee is about the event, not about swallowing ────


def test_an_exception_outside_exception_is_re_raised_once_the_result_is_delivered() -> None:
    """``KeyboardInterrupt`` / ``SystemExit`` must finish propagating.

    The driver widens its ``except`` past ``Exception`` so that a Ctrl-C mid-run
    still produces the terminal event the cognitions guarantee. The second half
    was missing once: Ctrl-C during ``agent.run()`` came back as a tidy
    ``AgentResult(partial=True, stop_reason="failed")`` with
    ``evals["error"] == "KeyboardInterrupt: "``, the interpreter kept running,
    and the caller's own ``except KeyboardInterrupt`` never fired. ``Exception``
    is the dividing line because it is the line the language draws — things
    outside it are "this program is being taken down", not "the operation
    failed".
    """
    with pytest.raises(KeyboardInterrupt):
        _reraise_if_not_an_exception(KeyboardInterrupt())
    with pytest.raises(SystemExit):
        _reraise_if_not_an_exception(SystemExit(2))


@pytest.mark.parametrize(
    "exc",
    [None, FileNotFoundError(2, "No such file or directory", "claude"), PermissionError(13, "Permission denied"),
     ValueError("a parse bug")],
)
def test_an_ordinary_run_failure_is_reported_as_data_and_never_re_raised(exc: BaseException | None) -> None:
    """Every way a RUN can fail is an ``Exception``, so the reported-as-data
    contract is untouched by the re-raise above.

    If this started raising, a missing binary — the single most common CLI
    failure there is — would come out of ``agent.run()`` as an exception instead
    of as the ``final`` event with ``stop_reason="spawn_failed"`` that callers
    are told to rely on.
    """
    _reraise_if_not_an_exception(exc)


# ── liveness: which bound, and what to call it ───────────────────────────────


def test_each_bound_is_named_for_the_kind_of_stuck_it_detects() -> None:
    """One ``TimeoutError`` for all four ways a run stops making progress leaves
    an operator to guess, and the four have four different fixes.

    The phase transitions are the easy part to get subtly wrong, so they are
    pinned directly: ``startup`` stops applying once ANY line arrives,
    ``first_event`` once a line produced an EVENT (a ``system/init`` is the CLI
    talking to itself, not content), and ``idle`` only starts applying after the
    first line — otherwise every run is measured for silence against a clock
    that started before the process did.
    """
    clock = [100.0]
    deadline = _CliDeadline(
        CliTimeouts(startup=10.0, first_event=30.0, idle=5.0, total=60.0), now=lambda: clock[0]
    )

    # Nothing has arrived: startup is the soonest bound.
    assert deadline.next_deadline()[0] == "startup_timeout"

    # A line arrived, so startup no longer applies and idle begins.
    clock[0] = 103.0
    deadline.note_line()
    assert deadline.next_deadline()[0] == "idle_timeout"

    # A line is not an event, so first_event is still live — and once idle is
    # off, it is what remains below ``total``.
    quiet = _CliDeadline(CliTimeouts(startup=10.0, first_event=4.0, idle=None, total=60.0), now=lambda: clock[0])
    quiet.note_line()
    assert quiet.next_deadline()[0] == "first_event_timeout"
    quiet.note_event()
    assert quiet.next_deadline()[0] == "total_timeout"


def test_nothing_configured_means_nothing_to_wrap_so_the_read_stays_on_the_fast_path() -> None:
    """``next_deadline()`` returning ``None`` is what lets the read loop await
    the iterator directly instead of wrapping every line in a ``Task``.

    A token-streamed run is thousands of lines; a task per line to enforce a
    bound nobody configured is pure overhead, and the only way to keep that
    property honest is to assert that "unbounded" is representable at all.
    """
    assert _CliDeadline(CliTimeouts.off()).next_deadline() is None


def test_a_bound_of_zero_is_refused_because_it_would_fire_before_the_process_runs() -> None:
    """``None`` disables a bound; ``0`` is a different thing and would be a
    silent "every run fails".

    A caller reaching for "no limit" and typing ``0`` gets a construction-time
    refusal naming the way out, rather than a service whose every CLI run times
    out before the child can be scheduled.
    """
    for name in ("startup", "first_event", "idle", "total"):
        with pytest.raises(ValueError, match=f"CliTimeouts.{name} must be a positive"):
            CliTimeouts(**{name: 0})
        with pytest.raises(ValueError, match=f"CliTimeouts.{name} must be a positive"):
            CliTimeouts(**{name: -1.0})


def test_the_production_preset_bounds_all_four_so_a_service_fails_rather_than_hangs() -> None:
    """The defaults are asymmetric on purpose — only ``startup`` is knowably
    safe without knowing what the session does — so the opinionated preset has
    to exist as a named thing rather than as advice in a docstring.

    ``idle`` has to clear the longest TOOL call, not the longest model turn; a
    preset that tuned it to model latency would kill working runs, which is the
    failure this preset is most able to cause.
    """
    preset = CliTimeouts.production()
    assert (preset.startup, preset.first_event, preset.idle, preset.total) == (60.0, 180.0, 600.0, 3600.0)
    assert preset.idle >= 600.0, "idle must clear a dependency install, not a model turn"


@pytest.mark.asyncio
async def test_a_silent_process_crosses_its_startup_bound_and_is_killed() -> None:
    """The measurement this module exists for: ``claude -p`` with an invalid
    credential produced no stdout, no stderr and no exit for 45+ seconds.

    The process was alive and the run looked active, so without a bound the
    caller waits forever. Crossing it must raise ``CliTimedOut`` carrying the
    reason AND the configured limit — the number is what tells an operator
    whether to raise the bound or go looking for a real hang.
    """
    deadline = _CliDeadline(CliTimeouts(startup=0.05))
    agen = _iter_stdout(_Stdout(None), deadline)
    with pytest.raises(CliTimedOut) as caught:
        async with contextlib.aclosing(agen) as lines:
            async for _ in lines:
                pass
    assert caught.value.reason == "startup_timeout"
    assert caught.value.limit_s == 0.05


@pytest.mark.asyncio
async def test_a_poll_tick_lets_the_caller_look_around_without_counting_as_progress() -> None:
    """A stop button has to work against a process that has gone quiet.

    Without the tick the caller only gets to look BETWEEN lines, so a CLI
    sitting in a long ``Bash(npm install)`` cannot be stopped until it speaks
    again — measured, a tripped token went unnoticed indefinitely against a
    silent process and 83s against a real one. The tick must not LENGTHEN a
    wait and must not count as activity, or it would hold the ``idle`` bound
    open on a process that has genuinely stopped talking.
    """
    deadline = _CliDeadline(CliTimeouts(startup=None, idle=0.4))
    ticks = 0
    lines: list[bytes] = []
    agen = _iter_stdout(_Stdout(b'{"a":1}\n', None), deadline, poll_s=0.01)
    async with contextlib.aclosing(agen) as stream:
        async for item in stream:
            if item is None:
                ticks += 1
                if ticks == 3:
                    break
            else:
                lines.append(item)
    assert lines == [b'{"a":1}\n']
    assert ticks == 3, "the loop must surface periodically even with nothing to read"


@pytest.mark.asyncio
async def test_a_tick_cannot_postpone_a_configured_bound() -> None:
    """The tick is only taken when the poll cadence is the SOONER of the two.

    A tick that replaced the bound's own deadline would turn every configured
    timeout into "never", silently, on exactly the runs the bound exists for —
    and the configuration would still read as enabled.
    """
    deadline = _CliDeadline(CliTimeouts(startup=0.05))
    started = time.monotonic()
    agen = _iter_stdout(_Stdout(None), deadline, poll_s=0.01)
    with pytest.raises(CliTimedOut) as caught:
        async with contextlib.aclosing(agen) as stream:
            async for _ in stream:
                pass
    assert caught.value.reason == "startup_timeout"
    assert time.monotonic() - started < 5.0, "the bound, not the tick cadence, must end the wait"


@pytest.mark.asyncio
async def test_a_line_too_big_for_the_reader_is_named_rather_than_reported_as_a_parse_bug() -> None:
    """``StreamReader.readline`` reports an overrun as a bare ``ValueError:
    Separator is not found, and chunk exceed the limit``.

    Classified as ``parse_failed``, that message names no limit, no payload and
    no fix, and reads like a bug in agentkit's parser rather than a buffer sized
    for a different protocol. Measured against real codex on "read every file
    under agentkit/ and summarise each": ``failed`` / ``parse_failed`` after
    three events, with the CLI itself exiting 0 and holding the answer. The
    limit is read off the READER, because a caller-supplied ``spawn=`` may have
    built the pipe with a different one and a message quoting a number nothing
    is enforcing is worse than no number.
    """
    overrun = ValueError("Separator is not found, and chunk exceed the limit")

    agen = _iter_stdout(_Stdout(overrun), _CliDeadline(CliTimeouts.off()))
    with pytest.raises(CliLineTooLong) as caught:
        async with contextlib.aclosing(agen) as stream:
            async for _ in stream:
                pass
    assert caught.value.limit_bytes == 8 * 1024 * 1024
    assert "8,388,608-byte" in str(caught.value)
    assert "read the file in parts" in str(caught.value), "the fix is on the payload side"

    class _NarrowReader(_Stdout):
        _limit = 4096

    agen = _iter_stdout(_NarrowReader(overrun), _CliDeadline(CliTimeouts.off()))
    with pytest.raises(CliLineTooLong) as narrow:
        async with contextlib.aclosing(agen) as stream:
            async for _ in stream:
                pass
    assert narrow.value.limit_bytes == 4096, "the number quoted must be the one being enforced"


def test_only_an_already_failed_startup_is_relabelled_and_only_to_an_auth_failure() -> None:
    """The stderr refinement may make a FAILED run's label more useful and
    nothing else.

    It is a substring match on human text a vendor can reword at any time, so
    the narrowness is the safety property: it cannot turn a working run into a
    failure and it cannot mask a different failure, because the fallback is the
    bound's own name. The trap it must avoid is the Claude CLI's "ANTHROPIC_API_KEY
    ... takes precedence over your claude.ai login", which it prints on perfectly
    SUCCESSFUL runs — using that marker would relabel every unrelated startup
    hang as a credential problem.
    """
    markers = ("invalid api key", "please run /login", "not logged in")

    assert _timeout_stop_reason(CliTimedOut("startup_timeout", 60), b"Invalid API key", markers) == (
        "authentication_failed"
    )
    # A different bound is never refined: an idle timeout is not an auth story
    # however the stderr reads.
    assert _timeout_stop_reason(CliTimedOut("idle_timeout", 600), b"Invalid API key", markers) == "idle_timeout"
    # No stderr, or stderr that says something else, keeps the bound's own name.
    assert _timeout_stop_reason(CliTimedOut("startup_timeout", 60), b"", markers) == "startup_timeout"
    assert _timeout_stop_reason(CliTimedOut("startup_timeout", 60), b"network unreachable", markers) == (
        "startup_timeout"
    )
    # The success-path warning must NOT be treated as a credential failure.
    precedence = b"ANTHROPIC_API_KEY or another auth source is set and takes precedence over your claude.ai login"
    assert _timeout_stop_reason(CliTimedOut("startup_timeout", 60), precedence, markers) == "startup_timeout"


# ── classifying a process that has already failed ────────────────────────────


def test_a_death_by_signal_outranks_a_schema_diagnostic_on_the_same_stderr() -> None:
    """A negative return code is a process that was killed; nothing on stderr
    can make that a schema problem.

    The precedence matters because the stderr of a crashed run routinely still
    holds whatever it was complaining about beforehand, and reporting
    ``schema_rejected`` for a SIGKILL sends an operator to edit a schema that
    was never the cause.
    """
    assert _process_exit_stop_reason(-9, b"Error: --json-schema is not a valid JSON Schema") == "process_crashed"
    assert _process_exit_stop_reason(-15, b"") == "process_crashed"


def test_a_schema_the_cli_refused_at_startup_is_named_instead_of_a_bare_exit_code() -> None:
    """``cli_exit_1`` names nothing an operator can act on; the diagnosis is on
    stderr and this is the one place it is read.

    Both binaries reject a bad schema two to five seconds into a spawn, from
    inside a subprocess, with the message on a stderr this cognition surfaces
    only on a failed run — so without this the operator gets an exit code and a
    hunt.
    """
    for marker in (
        b"Error: --json-schema is not a valid JSON Schema",
        b"invalid json schema",
        b"could not read --output-schema",
    ):
        assert _process_exit_stop_reason(1, marker) == "schema_rejected", marker


def test_an_ordinary_non_zero_exit_keeps_its_own_number() -> None:
    """The code is the only fact available, so it has to survive into the stop
    reason — ``cli_exit_2`` is a clap usage error and ``cli_exit_1`` is not, and
    collapsing both to ``failed`` throws away the distinction an operator starts
    from.
    """
    assert _process_exit_stop_reason(2, b"error: unexpected argument '--search' found") == "cli_exit_2"
    assert _process_exit_stop_reason(1, b"boom") == "cli_exit_1"
    assert _process_exit_stop_reason(0, b"") == "cli_exit_0"


# ── a schema is refused before it costs a subprocess ─────────────────────────


@pytest.mark.parametrize(
    ("schema", "fragment", "why"),
    [
        ("{}", r"must be a JSON Schema object \(a dict\)", "a JSON string rather than a dict"),
        ({}, "json_schema is empty", "an empty schema constrains nothing and both CLIs reject it"),
        ({"type": "object", "properties": {"a": {"default": {1, 2}}}}, "not JSON-serialisable",
         "a set reaches a hand-assembled schema dict"),
        ({"type": 3}, "must be a string or a list of strings", "a numeric type keyword"),
        ({"type": "object", "properties": []}, "'properties'] must be an object", "properties as a list"),
        ({"type": "object", "required": "answer"}, "must be a list of property names",
         "required as a bare string, which is the easy typo"),
    ],
)
def test_a_schema_the_cli_would_reject_is_refused_at_the_call_site(
    schema: Any, fragment: str, why: str
) -> None:
    """Both binaries reject these, two to five seconds into a spawn, with the
    message on a stderr the cognition surfaces only on an already-failed run.

    This is not a JSON Schema validator and must not become one — the core's
    zero-dependency property is worth more than catching subtle schema bugs —
    but the structural mistakes above cost nothing to catch and name themselves.
    """
    with pytest.raises(InvalidSchemaError, match=fragment):
        _validate_json_schema(schema, who="X")
    assert why


def test_a_required_field_that_properties_does_not_define_is_refused_as_unsatisfiable() -> None:
    """The one SEMANTIC check, because it is the mistake that actually happens:
    a field renamed in ``properties`` and not in ``required``.

    That schema is one no output can ever satisfy, so the CLI burns its whole
    structured-output retry budget and returns
    ``error_max_structured_output_retries`` — a failure that reads as the model's
    fault and is not. The message has to name the orphans and what is defined,
    or the reader is left diffing two dicts by eye.
    """
    with pytest.raises(InvalidSchemaError) as caught:
        _validate_json_schema(
            {"type": "object", "properties": {"qty": {"type": "integer"}}, "required": ["quantity"]}, who="X"
        )
    message = str(caught.value)
    assert "['quantity']" in message
    assert "defined: ['qty']" in message
    assert "exhaust its structured-output retries" in message


def test_a_consistent_schema_passes_through_untouched() -> None:
    """The guard must not refuse a schema the CLI would accept — a validator
    that rejects working configurations is one callers route around, and then it
    protects nothing."""
    schema = {"type": "object", "properties": {"qty": {"type": "integer"}}, "required": ["qty"]}
    assert _validate_json_schema(schema, who="X") is schema


# ── the environment the child actually gets ──────────────────────────────────


def test_the_profile_policy_strips_the_ambient_credential_and_says_which_ones() -> None:
    """``config_dir=`` exists to make the CLI run as a specific profile, and the
    binary prefers an ambient API key over that profile — announcing it on its
    own stderr.

    So inheriting the parent environment wholesale makes the isolation the
    caller asked for precisely the thing that does not happen. The REMOVED names
    come back as data because a silent strip is the same class of invisible
    behaviour as the silent inherit it replaces: the run that used to work now
    fails to authenticate and nothing connects that to a default nobody typed.
    """
    env, removed = _build_child_env(
        policy="profile",
        credential_vars=("AGENTKIT_TEST_FAKE_KEY", "AGENTKIT_TEST_ABSENT_KEY"),
        overrides=None,
        passthrough=(),
    )
    # Only names that were actually present are reported as removed.
    assert "AGENTKIT_TEST_ABSENT_KEY" not in removed
    assert "PATH" in env, "profile strips credentials, not the things a run needs"


def test_an_explicit_override_outranks_the_strip_so_a_tenant_credential_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``overrides`` are applied LAST in every policy, including over a name the
    policy just removed.

    That ordering is the point: ``profile`` with an explicit
    ``env={"ANTHROPIC_API_KEY": tenant_key}`` is how a caller states "this key,
    not the ambient one". A strip that outranked an explicit assignment would
    make the safe policy unusable with per-tenant credentials, which is how a
    policy gets switched off wholesale instead of fixed.
    """
    monkeypatch.setenv("AGENTKIT_TEST_FAKE_KEY", "ambient-value")
    env, removed = _build_child_env(
        policy="profile",
        credential_vars=("AGENTKIT_TEST_FAKE_KEY",),
        overrides={"AGENTKIT_TEST_FAKE_KEY": "tenant-value"},
    )
    assert env["AGENTKIT_TEST_FAKE_KEY"] == "tenant-value"
    assert removed == (), "a name the caller supplied explicitly was not taken away from them"


def test_the_isolated_policy_keeps_only_what_a_run_cannot_start_without(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing about the calling process leaks into an isolated spawn, so two
    machines running the same task hand the CLI the same environment.

    The passthrough list is not decoration: PATH finds the binary, HOME/XDG find
    its config, and the TLS/proxy variables are how a corporate network is
    reachable at all. Dropping those would turn "isolated" into "does not work
    behind a proxy", which is how a policy gets switched off wholesale.
    """
    monkeypatch.setenv("AGENTKIT_TEST_AMBIENT_NOISE", "leaked")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("AGENTKIT_TEST_FAKE_KEY", "ambient-value")

    env, removed = _build_child_env(policy="isolated", credential_vars=("AGENTKIT_TEST_FAKE_KEY",))

    assert "AGENTKIT_TEST_AMBIENT_NOISE" not in env
    assert env.get("HTTPS_PROXY") == "http://proxy.invalid:3128"
    assert "AGENTKIT_TEST_FAKE_KEY" not in env
    assert removed == ("AGENTKIT_TEST_FAKE_KEY",), "an operator debugging an auth error needs the name"


def test_the_inherit_policy_removes_nothing_so_the_historical_behaviour_stays_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller who wants the ambient key says so by name.

    Keeping ``inherit`` honest matters because it is the escape hatch the
    warning on the stripping policies points at; if it quietly stripped too,
    that advice would be wrong and there would be no way back.
    """
    monkeypatch.setenv("AGENTKIT_TEST_FAKE_KEY", "ambient-value")
    env, removed = _build_child_env(policy="inherit", credential_vars=("AGENTKIT_TEST_FAKE_KEY",))
    assert env["AGENTKIT_TEST_FAKE_KEY"] == "ambient-value"
    assert removed == ()


# ── a mistyped cwd is not a missing binary ───────────────────────────────────


def test_a_missing_working_dir_is_distinguishable_from_a_missing_binary(tmp_path: Any) -> None:
    """``create_subprocess_exec`` raises ``FileNotFoundError`` for a missing cwd
    AND for a missing binary — the same exception for two problems with
    different fixes.

    The pre-flight check is what lets the terminal event say
    ``working_dir_missing``, and the MESSAGE PREFIX is load-bearing: both
    ``_finalise`` implementations classify the failure by matching it, so a typo
    in the raiser or in either matcher would silently reclassify a mistyped path
    as a missing binary and fail no type check or lint.
    """
    _require_working_dir(None)  # unset is not an error
    _require_working_dir(tmp_path)  # present is not an error
    with pytest.raises(FileNotFoundError, match="working_dir does not exist:"):
        _require_working_dir(tmp_path / "nope")


# ── stopping the child AND what it started ───────────────────────────────────


@pytest.mark.asyncio
async def test_a_child_that_ignores_sigterm_is_escalated_to_sigkill() -> None:
    """The SIGKILL sweep is not optional: a grace period that ends in a shrug
    leaves the process alive holding its own copy of the pipe.

    It also runs even when the child exited politely, because the child exiting
    says nothing about the ``npm install`` it forked. A ``_terminate`` that
    forgot the escalation is one of the three bugs this shared module exists to
    stop happening twice.
    """

    class _Stubborn(_Proc):
        async def wait(self) -> int:
            if self.returncode is not None:
                return self.returncode
            await asyncio.Event().wait()  # never obeys SIGTERM
            raise AssertionError("unreachable")

        def terminate(self) -> None:
            self.signals.append(signal.SIGTERM)  # noted and ignored

    proc = _Stubborn(_Stdout())
    await _terminate(proc, 0.05, process_group=False)
    assert proc.signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.asyncio
async def test_a_backend_that_owns_tree_termination_is_used_instead_of_raw_signals() -> None:
    """An execution backend that can kill a tree itself is the authority on how.

    Signalling around it would mean two mechanisms racing on one child — and the
    backend's is the one that knows about containers, cgroups or a remote host,
    where a local ``killpg`` reaches nothing at all.
    """

    class _Backed:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.grace: float | None = None
            self.signals: list[int] = []

        async def terminate_tree(self, grace_s: float) -> None:
            self.grace = grace_s
            self.returncode = -15

        async def wait(self) -> int:
            return self.returncode if self.returncode is not None else 0

        def terminate(self) -> None:  # pragma: no cover - must not be reached
            self.signals.append(signal.SIGTERM)

        def kill(self) -> None:  # pragma: no cover - must not be reached
            self.signals.append(signal.SIGKILL)

    proc = _Backed()
    await _terminate(proc, 0.25)
    assert proc.grace == 0.25
    assert proc.signals == [], "the backend's own termination replaces the signal pair"


@pytest.mark.parametrize(
    ("proc", "why"),
    [
        (type("_NoPid", (), {})(), "a test double has no pid, and inventing one would signal a real process"),
        (type("_OurGroup", (), {"pid": os.getpid()})(), "a child sharing OUR group would mean killing the service"),
        (type("_Reaped", (), {"pid": 2**30})(), "an already-reaped child has no group left to signal"),
    ],
)
def test_there_is_no_group_to_signal_degrades_to_the_direct_child(proc: Any, why: str) -> None:
    """Three distinct reasons there is no group, all of which must return
    ``None`` rather than raise out of a ``finally``.

    The own-group check is the difference between a bug and a catastrophe: if a
    caller supplies a ``spawn=`` that forgets ``start_new_session``, the child
    sits in the SERVICE's process group, and killing that group kills the
    service. Measured — ``start_new_session=True`` alone still orphans
    grandchildren, so both halves have to be present or neither is.
    """
    assert _child_pgid(proc) is None, why


@pytest.mark.asyncio
async def test_cancelling_a_run_reaps_a_real_grandchild_and_not_only_the_direct_child() -> None:
    """The leak this exists to close, at the only level that can observe it.

    ``proc.terminate()`` signals ONE pid, so a cancelled run left the expensive
    half of the work — the ``npm install``, the test suite — running and holding
    its own copy of the pipe until it finished on its own. Measured before the
    group kill; the grandchild survived every time. A process double cannot have
    grandchildren, so this uses a real local ``sys.executable`` tree: the
    grandchild proves its own liveness by holding a listening socket, which
    avoids every pid-reuse and zombie race.
    """
    probe = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()

    grandchild = (
        "import socket,time; s=socket.socket(); s.bind(('127.0.0.1'," + str(port) + ")); "
        "s.listen(); print('ready',flush=True); time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,json,time; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdout=subprocess.PIPE); "
        "p.stdout.readline(); print(json.dumps({'type':'ready','pid':p.pid}),flush=True); time.sleep(60)"
    )

    up = asyncio.Event()
    stop = asyncio.Event()
    grandchild_pid: int | None = None

    async def handle(payload: dict[str, Any]) -> Any:
        nonlocal grandchild_pid
        grandchild_pid = payload.get("pid")
        up.set()
        if False:  # pragma: no cover - yields nothing
            yield None

    class _Trip:
        def check_cancelled(self) -> None:
            if stop.is_set():
                raise RuntimeError("the caller pressed stop")

    outcome = _CliRunOutcome()

    async def run() -> None:
        agen = _run_cli_process(
            prepare=lambda: _CliLaunch(argv=[sys.executable, "-c", parent, grandchild], env={}),
            handle=handle,
            spawn=None,  # a REAL subprocess
            semaphore=asyncio.BoundedSemaphore(1),
            timeouts=CliTimeouts(total=30.0),
            terminate_grace_s=0.2,
            ctx=_Trip(),
            outcome=outcome,
        )
        async with contextlib.aclosing(agen) as events:
            async for _ in events:
                pass

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(up.wait(), 20)
        stop.set()
        await asyncio.wait_for(task, 20)

        assert outcome.cancelled is True
        assert outcome.return_code is not None and outcome.return_code < 0, "the direct child died by signal"

        # The grandchild released its socket, which it only does by dying.
        await asyncio.sleep(0.2)
        with pytest.raises(OSError):
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 3)
            writer.close()
            await writer.wait_closed()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if grandchild_pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(grandchild_pid, signal.SIGKILL)


# ── the finally that must not hang ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_stderr_is_read_in_chunks_so_a_bound_costs_one_chunk_not_the_diagnostic() -> None:
    """``read()`` with no size accumulates into a local, so cancelling it
    discards everything it had — and this is the text explaining a failure.

    Chunked, a timeout costs at most the chunk in flight. The whole diagnostic
    still has to arrive on the normal path, or the chunking would be a silent
    truncation of the one channel that explains a bare ``cli_exit_1``.
    """

    class _Chunked:
        def __init__(self) -> None:
            self._chunks = [b"Traceback (most recent call last):\n", b"  RuntimeError: boom\n", b""]

        async def read(self, _n: int) -> bytes:
            return self._chunks.pop(0)

    assert await _drain_stderr(_Chunked()) == b"Traceback (most recent call last):\n  RuntimeError: boom\n"


@pytest.mark.asyncio
async def test_a_helper_still_holding_the_pipe_cannot_hang_the_whole_run() -> None:
    """EOF on stderr needs EVERY writer to close the pipe, not just the process
    we spawned.

    Measured against codex 0.152.1, which spawns a ``codex-code-mode-host``
    helper: the run reached EOF on stdout, entered the ``finally``, and never
    came out — past the liveness bounds, past cancellation, past anything the
    caller could do. Roughly one run in three, which is exactly the profile of a
    bug that gets blamed on the model.
    """

    class _Held:
        async def read(self, _n: int) -> bytes:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    started = time.monotonic()
    assert await _drain_stderr(_Held(), timeout_s=0.05) == b""
    assert time.monotonic() - started < 5.0, "the read is bounded, so the finally cannot hang the run"


@pytest.mark.asyncio
async def test_a_broken_stderr_pipe_never_costs_the_run_its_terminal_event() -> None:
    """A run's terminal event must not depend on the quality of its stderr.

    This runs from a ``finally`` on the failure path, so an exception escaping
    here would replace a diagnosed failure with an undiagnosed one — the
    original error lost and the new one naming a pipe.
    """

    class _Broken:
        async def read(self, _n: int) -> bytes:
            raise OSError("the pipe is gone")

    assert await _drain_stderr(_Broken()) == b""


# ── the driver's own branches ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tripped_cancellation_token_stops_the_read_and_terminates_the_process() -> None:
    """Cooperative cancellation is checked once per line, and the process it was
    reading has to die with it.

    A cancel that recorded itself and left the child running is the leak the
    group kill exists for, arriving through the cooperative door instead of the
    ``CancelledError`` one. The outcome must say ``cancelled`` so the adapter
    stamps ``cancelled`` rather than a parse error on the half-written line the
    cut produced.
    """
    proc = _Proc(_Stdout(b'{"a":1}\n', b'{"b":2}\n', b'{"c":3}\n'))
    outcome = await _run(proc, ctx=_StopButton(after_lines=1))

    assert outcome.cancelled is True
    assert outcome.should_reraise_cancel is False, "a cooperative stop is not an asyncio cancel"
    assert signal.SIGTERM in proc.signals and signal.SIGKILL in proc.signals
    assert outcome.timed_out is None


@pytest.mark.asyncio
async def test_an_asyncio_cancel_is_recorded_for_re_raise_so_wait_for_still_raises() -> None:
    """Suppressing ``CancelledError`` breaks ``asyncio.wait_for``'s timeout and
    a ``TaskGroup``'s sibling cancellation.

    The driver records it instead, so the adapter can deliver its terminal event
    AND THEN propagate. Both halves are required: without the flag the cancel is
    swallowed and the caller's timeout never fires; without the terminal event
    the exactly-one-``final`` contract breaks on the one path callers most need
    it.
    """

    async def refuse(*_argv: str, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    outcome = _CliRunOutcome()
    agen = _run_cli_process(
        prepare=lambda: _CliLaunch(argv=["fake-cli"], env={}),
        handle=_no_events,
        spawn=refuse,
        semaphore=asyncio.BoundedSemaphore(1),
        timeouts=CliTimeouts.off(),
        terminate_grace_s=0.05,
        ctx=None,
        outcome=outcome,
    )
    async with contextlib.aclosing(agen) as events:
        async for _ in events:
            pass

    assert outcome.cancelled is True
    assert outcome.should_reraise_cancel is True
    assert outcome.spawned is False


@pytest.mark.asyncio
async def test_a_spawn_that_never_happened_is_not_confused_with_one_that_did() -> None:
    """``spawned`` is what separates ``spawn_failed`` from ``parse_failed``, and
    ``uncertain`` from "we know nothing ran".

    A failure before the spawn means the work definitely did not happen, which
    is a retryable situation; a failure after it means the CLI may have written
    files, spent money, or both, and reporting that as a clean non-start is how
    a retry duplicates side effects.
    """

    async def missing(*_argv: str, **_kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", "claude")

    outcome = _CliRunOutcome()
    agen = _run_cli_process(
        prepare=lambda: _CliLaunch(argv=["claude"], env={}),
        handle=_no_events,
        spawn=missing,
        semaphore=asyncio.BoundedSemaphore(1),
        timeouts=CliTimeouts.off(),
        terminate_grace_s=0.05,
        ctx=None,
        outcome=outcome,
    )
    async with contextlib.aclosing(agen) as events:
        async for _ in events:
            pass

    assert outcome.spawned is False
    assert outcome.uncertain is False, "nothing ran, so nothing is uncertain"
    assert isinstance(outcome.fatal_exc, FileNotFoundError)
    assert outcome.return_code == -1, "-1 means no process, which is not an exit code of 0"


@pytest.mark.asyncio
async def test_a_failure_after_the_spawn_is_marked_uncertain_because_work_may_have_happened() -> None:
    """Past the spawn, agentkit cannot know what the CLI did before it broke.

    ``execution_uncertain`` is the flag that stops a caller retrying a run that
    may already have written to the workspace. Defaulting it to False would make
    every mid-run parse bug look safe to repeat.
    """

    async def exploding(_payload: dict[str, Any]) -> Any:
        raise RuntimeError("a parse bug in the adapter")
        if False:  # pragma: no cover - yields nothing
            yield None

    proc = _Proc(_Stdout(b'{"a":1}\n'))
    outcome = _CliRunOutcome()
    agen = _run_cli_process(
        prepare=lambda: _CliLaunch(argv=["fake-cli"], env={}),
        handle=exploding,
        spawn=_spawn_returning(proc),
        semaphore=asyncio.BoundedSemaphore(1),
        timeouts=CliTimeouts.off(),
        terminate_grace_s=0.05,
        ctx=None,
        outcome=outcome,
    )
    async with contextlib.aclosing(agen) as events:
        async for _ in events:
            pass

    assert outcome.spawned is True
    assert outcome.uncertain is True
    assert isinstance(outcome.fatal_exc, RuntimeError)


@pytest.mark.asyncio
async def test_a_cleanup_that_fails_is_recorded_without_replacing_the_real_outcome() -> None:
    """Cleanup is observable, and it is not the story.

    A ``finally`` that let its own failure escape would overwrite the diagnosis
    with a teardown error — the caller loses the reason the run failed and gains
    one about closing a pipe. ``cleanup_failed`` keeps the fact without taking
    the diagnosis.
    """

    class _BadCleanup(_Proc):
        async def aclose(self) -> None:
            raise OSError("the transport is already gone")

    proc = _BadCleanup(_Stdout(b'{"a":1}\n'))
    outcome = await _run(proc)

    assert outcome.cleanup_failed is True
    assert outcome.fatal_exc is None, "a teardown error must not become the run's failure"
    assert outcome.cancelled is False


@pytest.mark.asyncio
async def test_the_prompt_is_written_to_stdin_and_then_stdin_is_closed() -> None:
    """Closing stdin is not a tidy-up: it is how both CLIs are told the input is
    complete.

    Without it a one-shot run waits for a turn that is never coming — a hang
    with no error, which from the outside is indistinguishable from a slow model
    and would be attributed to one.
    """
    proc = _Proc(_Stdout(b'{"a":1}\n'))
    await _run(proc)

    assert proc.stdin.written == b"go\n"
    assert proc.stdin.closed is True


@pytest.mark.asyncio
async def test_a_throw_while_resolving_argv_still_reaches_the_caller_as_an_outcome() -> None:
    """``prepare`` runs INSIDE the guarded block so that argv/env resolution
    failures are reported like any other.

    Both adapters can throw there — a malformed ``Prompt`` template, an
    exhausted budget, a schema that will not render — and those are failures a
    caller must receive as a terminal event, not as an exception out of a
    generator that never yielded.
    """

    def refuse() -> _CliLaunch:
        raise ValueError("the prompt template is malformed")

    outcome = await _run(_Proc(_Stdout()), prepare=refuse)

    assert isinstance(outcome.fatal_exc, ValueError)
    assert outcome.spawned is False


# ── numbers a ledger has to be able to add up ────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected", "why"),
    [
        (True, 0, "a token count of True would silently become 1"),
        (False, 0, "the same, through the other bool"),
        (7, 7, "an ordinary count"),
        (7.9, 7, "a float count truncates rather than crashing the fold"),
        (float("nan"), 0, "NaN makes every later comparison false"),
        (float("inf"), 0, "infinity is not a token count"),
        ("42", 42, "a CLI that quotes its numbers still has to be readable"),
        (" 42 ", 42, "with whitespace, as a hand-edited recording has"),
        ("not a number", 0, "garbage falls back rather than raising out of the parser"),
        (None, 0, "a missing field"),
        ({"n": 1}, 0, "a wrong-typed field, which the property tests found"),
    ],
)
def test_a_token_count_from_somebody_elses_payload_is_type_gated(value: Any, expected: int, why: str) -> None:
    """Handed a field of the wrong type, ``payload.get("x") or {}`` guards
    nothing and the very next ``.get`` raises ``AttributeError`` out of the
    parser.

    That ends the run with ``parse_failed``, having lost its terminal payload
    and charged ``$0.00`` to the budget. ``bool`` is excluded explicitly because
    ``isinstance(True, int)`` is ``True`` in Python, so a count of ``True`` would
    become ``1`` — a number with no source that no assertion about tokens would
    notice.
    """
    assert _as_int(value) == expected, why


@pytest.mark.parametrize(
    ("value", "expected", "why"),
    [
        (0.25, 0.25, "an ordinary cost"),
        (0.0, 0.0, "a genuinely free run is not the same as an unknown one"),
        (3, 3.0, "an integer cost"),
        ("0.25", 0.25, "a quoted cost"),
        (float("nan"), None, "NaN makes a budget unable to answer whether it is over"),
        (float("inf"), None, "infinity would exhaust every ceiling at once"),
        (-1.0, None, "a negative cost would CREDIT the budget"),
        (True, None, "a bool is not a cost"),
        (None, None, "no measurement available"),
        ("free", None, "unparseable text"),
        ({"usd": 1}, None, "a wrong-typed field"),
    ],
)
def test_a_cost_is_either_a_number_a_ledger_can_add_or_explicitly_unknown(
    value: Any, expected: float | None, why: str
) -> None:
    """NaN is the one that matters: it is not a wrong number, it is a value that
    makes every subsequent comparison in the ledger false.

    A budget handed NaN can no longer answer whether it is over. A negative cost
    is refused for the mirror reason — it hands headroom BACK to a run that just
    spent money. ``None`` is the honest answer for all of them, and is distinct
    from ``0.0``, which is a claim that the run was free.
    """
    result = _as_cost(value)
    if expected is None:
        assert result is None, why
    else:
        assert result == expected, why
        assert not math.isnan(result)


# ── a structured-output failure a caller can act on ──────────────────────────


def test_per_field_diagnostics_survive_as_data_instead_of_being_joined_into_a_sentence() -> None:
    """The adapters produce per-field diagnostics and this module used to join
    them into one string.

    A caller wanting to show a user WHICH field was wrong, or to count failures
    by path across a fleet, then had to parse English back out of a message a
    library upgrade can reword. The paths are JSONPath-ish rather than the
    adapters' dotted form because ``lines.0.qty`` is ambiguous the moment a
    mapping has a numeric-looking key, and ``$``-rooted paths are what a caller
    can hand to ``jq``, a UI, or an error-grouping key.
    """

    class _Coercion(Exception):
        errors = ["lines.0.qty: Input should be a valid integer", "total: Field required"]
        raw = {"lines": [{"qty": "two"}]}

    failure = StructuredOutputFailure.from_coercion_error(_Coercion("2 error(s)"))

    assert failure.kind == "schema_mismatch"
    assert [(v.path, v.message) for v in failure.violations] == [
        ("$.lines[0].qty", "Input should be a valid integer"),
        ("$.total", "Field required"),
    ]
    assert failure.truncated is False
    assert failure.raw_excerpt is not None and "two" in failure.raw_excerpt


def test_a_fan_out_failure_is_capped_and_says_that_it_was() -> None:
    """``evals`` is deep-frozen, checkpointed and often logged, and a fan-out
    over a large list produces hundreds of violations that all say the same
    thing.

    Capping without recording the cap would be the worse bug: a caller rendering
    twenty fields would believe those were all of them. ``truncated`` is the flag
    that keeps the cap honest.
    """

    class _Flood(Exception):
        errors = [f"lines.{i}.qty: Input should be a valid integer" for i in range(50)]
        raw = "x" * 5000

    failure = StructuredOutputFailure.from_coercion_error(_Flood("50 error(s)"))

    assert len(failure.violations) == 20
    assert failure.truncated is True
    assert failure.raw_excerpt is not None
    assert failure.raw_excerpt.endswith("…")
    assert len(failure.raw_excerpt) <= 801
    assert "(further violations omitted)" in failure.repair_prompt()


def test_an_unattributed_complaint_becomes_a_root_violation_rather_than_being_dropped() -> None:
    """An adapter that did not attribute its complaint to a field has told us
    the OBJECT is wrong.

    Inventing a path would mislead a UI into marking an innocent field; dropping
    the diagnostic would leave a failure with no explanation at all. Root-scoping
    it is the honest reading, and the root sentinels the adapters use have to map
    to the same ``$``.
    """
    failure = StructuredOutputFailure.from_coercion_error(
        type("_E", (Exception,), {"errors": ["not an object at all"], "raw": None})("1 error(s)")
    )
    assert [(v.path, v.message) for v in failure.violations] == [("$", "not an object at all")]

    rooted = StructuredOutputFailure.from_coercion_error(
        type("_E", (Exception,), {"errors": ["<root>: 'z' is a required property"], "raw": None})("1 error(s)")
    )
    assert rooted.violations[0].path == "$"


def test_a_message_containing_a_colon_keeps_its_whole_message() -> None:
    """The split is on the FIRST ``": "`` only.

    Splitting on the last one moves half the message into the path, which
    produces a path no UI can match and a message missing its most specific
    half — and the shapes that contain colons (``"got {'a': 1}"``) are exactly
    the informative ones.
    """
    failure = StructuredOutputFailure.from_coercion_error(
        type("_E", (Exception,), {"errors": ["meta.tags: unexpected value: got {'a': 1}"], "raw": None})("1")
    )
    assert failure.violations[0].path == "$.meta.tags"
    assert failure.violations[0].message == "unexpected value: got {'a': 1}"


def test_the_typed_view_round_trips_through_the_json_safe_form() -> None:
    """``evals`` is deep-frozen and serialised, so the failure goes in as a dict
    — but an application must be able to branch on a type rather than re-parse
    the sentence it was handed.

    ``of()`` is the read side of that contract. If it stopped recognising the
    dict it would return ``None``, and every caller's
    ``if failure is not None:`` branch would silently stop running while the
    failure was still there in the string form.
    """
    original = StructuredOutputFailure(
        kind="schema_mismatch",
        detail="OutputCoercionError: 1 error(s)",
        violations=(SchemaViolation(path="$.qty", message="Input should be a valid integer"),),
        raw_excerpt='{"qty": "two"}',
        truncated=True,
    )
    revived = StructuredOutputFailure.of({"structured_output_failure": original.to_dict()})

    assert revived == original
    assert StructuredOutputFailure.of({}) is None
    assert StructuredOutputFailure.of({"structured_output_failure": "a sentence"}) is None


def test_a_failure_with_no_field_detail_still_produces_usable_repair_feedback() -> None:
    """``retries_exhausted`` and ``missing`` carry no violations, because
    agentkit never saw the attempts.

    A repair prompt that listed zero fields and said "fix exactly these" would be
    a prompt asking the model to change nothing — so the no-violations case needs
    its own sentence rather than a degenerate version of the other one.
    """
    prompt = StructuredOutputFailure(kind="missing", detail="no structured_output").repair_prompt()
    assert "did not produce the required structured output" in prompt
    assert "Fix exactly" not in prompt


# ── diagnostics that get stored ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "absent", "why"),
    [
        ("auth failed for sk-ant-api03-ABCDEFGHIJKLMNOP", "sk-ant-api03-ABCDEFGHIJKLMNOP", "an anthropic key"),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz", "a bearer header"),
        ("api_key=0123456789abcdefghij", "0123456789abcdefghij", "a config assignment"),
        ("token ghp_0123456789abcdefghij0123", "ghp_0123456789abcdefghij0123", "a github token"),
        ("AKIAIOSFODNN7EXAMPLE denied", "AKIAIOSFODNN7EXAMPLE", "an aws key id"),
    ],
)
def test_a_credential_shaped_run_in_stored_diagnostics_is_replaced(text: str, absent: str, why: str) -> None:
    """``evals["stderr"]`` is the CLI's stderr verbatim, and an ``AgentResult``
    is checkpointed, logged and fanned out to observers.

    A CLI that prints a failing request's ``Authorization`` header — ordinary
    diagnostic behaviour — would otherwise have that header persisted. Defence in
    depth, explicitly not a boundary: it cannot know every credential format, so
    redacted output must not be treated as safe to publish.
    """
    redacted = _redact_secrets(text)
    assert absent not in redacted, why
    assert "[redacted" in redacted


def test_redaction_keeps_the_key_name_because_which_secret_leaked_is_not_the_secret() -> None:
    """Structure-preserving on purpose.

    An operator reading a failed run needs to know WHICH credential was
    involved, and a redactor that ate the name too would make the diagnostic
    useless — which is how redaction gets switched off wholesale instead of
    tuned.
    """
    assert _redact_secrets("api_key=0123456789abcdefghij").startswith("api_key=")


def test_ordinary_diagnostic_text_survives_redaction_untouched() -> None:
    """A redactor that eats ordinary words makes the diagnostic useless.

    Every pattern requires a credential-shaped prefix or a long opaque run,
    never a bare word, precisely so a stderr about models, paths and errors
    reads as it was written. A false positive here is not cosmetic: it is the
    reason somebody turns redaction off.
    """
    text = "model claude-opus-4-5 failed on /Users/x/secret-plans/report.md after 3 retries"
    assert _redact_secrets(text) == text
    assert _redact_secrets("") == ""
