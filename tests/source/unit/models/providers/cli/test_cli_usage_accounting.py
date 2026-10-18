"""What a CLI run's TOKEN counts mean when the run never reached its usage payload.

WHY THIS FILE EXISTS
--------------------
``ProviderUsage.cost_usd`` was changed from ``0.0`` to ``None`` because a
dataclass default was making a positive claim on behalf of code that had said
nothing: an un-populated usage reported ``cost_source=REPORTED`` and charged
``$0.00``, so a ten-minute Opus run that timed out looked free. That fix is in
and the failure suites pin it.

``input_tokens`` and ``output_tokens`` got no such fix, because they cannot
have one: they are ``int``, and ``0`` is both "the provider billed nothing" and
"nobody ever looked". Measured against the codex driver with a stdout that
emits 200 ``item.completed`` payloads and then goes quiet under ``idle=0.05``::

    stop_reason        failed / idle_timeout
    partial            True
    usage              ProviderUsage(input_tokens=0, output_tokens=0,
                                     cost_usd=None, cost_source=UNPRICED)
    evals[raw_usage]   {}

So the money is honest and the tokens are a stated zero for work that really
happened. The consequence is downstream and was measured the same way: the CLI
adapter turns that into ``Turn(raw_usage={}, usage_complete=False)``,
``normalize_usage`` turns ``{}`` into zeros, and
``core/runtime/calls.py``'s ``remaining = limits.max_output_tokens -
usage.output`` therefore hands the NEXT attempt the full token allowance that
200 turns have already spent. ``max_cost_usd`` has ``budget_unknown`` to fall
back on; the token ceiling has no equivalent and does not know it is guessing.

What each test below pins, and what breaks without it:

* the reproduction itself, so that "a timed-out run reports zero tokens" is a
  statement the suite makes rather than one a live run discovers;
* ``usage_complete=False`` on every way a run can be cut short. That flag is the
  ONLY thing standing between a stated zero and a believed zero, and it is
  derived by hand in two separate ``_finalise`` implementations — the exact
  duplication ``_common.py`` exists to end. A path that reports zeros WITHOUT it
  is the real bug, and nothing pinned the set of paths end to end;
* the one case where the flag is NOT set and the zeros are reported as final: a
  terminal payload that arrives carrying no ``usage`` object at all. Both
  parsers read it with ``_as_dict(payload.get("usage"))``, which defaults to
  ``{}``, so a CLI that omits the field produces ``saw_terminal=True``,
  ``partial=False`` and a zero-token run presented as complete;
* that ``_charge_meters`` tells a caller's meter ``cost_usd=None`` rather than
  zero, and that the token integers it hands over are the part the type cannot
  qualify. That is the hand-off, recorded here so the next reader does not
  "fix" it by refusing the charge — ``_charge_meters`` MUST still charge (a
  meter that never hears about a crashed run never exhausts), which the codex
  failure suite already asserts in its own words: "a meter must not be told the
  run was free".

The process doubles are hand-written rather than ``FakeCodexCli`` for the reason
the rest of this directory gives: a stdout that stops mid-stream and never
speaks again is not a shape a recording can express, and it is the only shape a
liveness bound can be observed against. No test here waits: every bound is
0.05s and every double returns without suspending until it is deliberately
silent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.contracts.types import CostSource, ProviderUsage, StreamEvent
from agentkit.models.providers.cli._common import (
    CliResult,
    CliTimeouts,
    _charge_meters,
    _usage_was_measured,
)
from agentkit.models.providers.cli.codex.driver import CodexCliCognition

# ── stubs ────────────────────────────────────────────────────────────────────


@dataclass
class _Agent:
    """The only attributes the codex cognition reads off an agent."""

    prompt: Any = None
    model: str | None = None
    _output_adapter: Any = None


class _Stderr:
    def __init__(self, data: bytes = b"") -> None:
        self._buf = bytearray(data)

    async def read(self, n: int = -1) -> bytes:
        del n
        out, self._buf = bytes(self._buf), bytearray()
        return out


class _Stdin:
    def write(self, data: bytes) -> None:
        del data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _Stdout:
    """Emit the given lines, then either end the stream or go silent forever.

    ``silent_after`` is the whole point: a CLI that is killed by a wall-clock
    bound has an OPEN stdout with nothing more coming, which is a different
    shape from EOF and the only one a bound can fire against. The cursor
    advances only once a line has been delivered, mirroring the ``StreamReader``
    property ``_iter_stdout`` is built on.
    """

    def __init__(self, *lines: bytes, silent_after: bool = True) -> None:
        self._lines = list(lines)
        self._at = 0
        self._silent_after = silent_after

    def __aiter__(self) -> _Stdout:
        return self

    async def __anext__(self) -> bytes:
        if self._at < len(self._lines):
            line = self._lines[self._at]
            self._at += 1
            return line
        if self._silent_after:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        raise StopAsyncIteration


class _Proc:
    """The ``asyncio.subprocess.Process`` surface the shared driver touches."""

    def __init__(self, stdout: _Stdout, *, exit_code: int = 0, stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.stderr = _Stderr(stderr)
        self.stdin = _Stdin()
        self.returncode: int | None = None
        self._exit_code = exit_code

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9


def _spawning(proc: Any) -> Any:
    async def spawn(*_argv: str, **_kwargs: Any) -> Any:
        return proc

    return spawn


class _StopButton:
    """A ``Ctx``-shaped stub carrying nothing but cooperative cancellation.

    The driver polls it once per stdout line, which is the only cancellation
    route through a double whose reads never await.
    """

    def __init__(self, *, after_lines: int = 0) -> None:
        self.checks = 0
        self._after = after_lines

    def check_cancelled(self) -> None:
        self.checks += 1
        if self.checks > self._after:
            raise RuntimeError("the caller pressed stop")


class _Meter:
    """A caller-supplied meter, recording the usage it was handed verbatim."""

    def __init__(self) -> None:
        self.charges: list[ProviderUsage] = []

    async def charge(self, _call: Any, usage: ProviderUsage) -> None:
        self.charges.append(usage)


class _Envelope:
    """A per-actor ceiling in tokens, dollars and steps.

    Records what it was TOLD as well as accumulating it, because "the ledger did
    not move" and "the ledger was handed a zero" are the two readings this file
    is about and a bare counter cannot separate them.
    """

    def __init__(self) -> None:
        self.charges: list[tuple[int, Any, int]] = []
        self.tokens = 0
        self.steps = 0

    def charge(self, *, tokens: int, cost_usd: Any, steps: int) -> None:
        self.charges.append((tokens, cost_usd, steps))
        self.tokens += tokens
        self.steps += steps


class _Ctx:
    """The two attributes ``_charge_meters`` reaches for, and nothing else."""

    def __init__(self, *, all_meters: tuple[Any, ...] = (), actor_budget: Any = None) -> None:
        self.all_meters = all_meters
        self.actor_budget = actor_budget

    def check_cancelled(self) -> None:
        return None


# ── drivers ──────────────────────────────────────────────────────────────────

_THREAD_STARTED = b'{"type":"thread.started","thread_id":"thread-1"}\n'


def _real_work(n: int) -> tuple[bytes, ...]:
    """``n`` turns' worth of completed items — work that provably happened."""
    return tuple(
        b'{"type":"item.completed","item":{"id":"m%d","type":"agent_message",'
        b'"text":"turn %d did real work"}}\n' % (i, i)
        for i in range(n)
    )


async def _drive(cognition: CodexCliCognition, *, ctx: Any = None) -> Any:
    """One ``drive``'s single terminal result, asserting the exactly-one contract."""
    events: list[StreamEvent] = [ev async for ev in cognition.drive(_Agent(), "do the work", ctx, None)]
    types = [e.type for e in events]
    assert types.count("final") == 1, types
    assert types[-1] == "final", types
    return events[-1].result


# ── the reproduction ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_wall_clock_bound_that_fires_mid_stream_reports_zero_tokens_for_work_it_watched_happen() -> None:
    """The live symptom, offline: 200 real turns in, usage all zeros.

    Both CLIs report usage ONLY on their terminal payload — ``result`` for
    claude, ``turn.completed`` for codex — so a process killed before that
    arrives genuinely never tells us the token counts. The zeros are therefore
    not a parsing bug; they are the ``int`` default of a field that has no way
    to say "unknown", and this test exists so that fact is asserted rather than
    assumed. If the token counts ever stop being zero here, something started
    inventing them.

    ``cost_usd is None`` / ``UNPRICED`` is the half that IS honest, and it is
    asserted alongside so the two can never drift back together: a change that
    reinstated a ``0.0`` cost default would pass a test that only looked at
    tokens.
    """
    proc = _Proc(_Stdout(_THREAD_STARTED, *_real_work(200)))
    cognition = CodexCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=None, idle=0.05))

    result = await _drive(cognition)

    assert result.evals["stop_reason"] == "idle_timeout", "precondition: a bound fired, the CLI did not finish"
    assert result.output, "precondition: the stream carried real answer text, so work provably happened"
    assert result.usage.total_tokens == 0, "the defect: a stated zero for 200 turns of real work"
    assert result.usage.cost_usd is None, "the money half stays honest — unknown, not free"
    assert result.usage.cost_source is CostSource.UNPRICED
    assert result.evals["raw_usage"] == {}, (
        "the raw dict is EMPTY rather than a dict of zeros, which is what lets "
        "the adapter tell a missing measurement from a measured one"
    )
    assert not _usage_was_measured(result.usage)


# ── the flag that keeps a stated zero from becoming a believed one ───────────


@pytest.mark.parametrize(
    ("bound", "lines", "expected_reason"),
    [
        # Nothing on stdout at all: the binary never came up.
        (CliTimeouts(startup=0.05), (), "startup_timeout"),
        # A line arrived, so ``startup`` is satisfied — but ``thread.started``
        # is the CLI talking to itself and produces no ``StreamEvent``, which is
        # the only shape that can reach the ``first_event`` bound. Feeding it an
        # ``item.completed`` would satisfy this bound too and, with every other
        # bound off, hang on the silence instead of timing out.
        (CliTimeouts(startup=None, first_event=0.05), (_THREAD_STARTED,), "first_event_timeout"),
        # Spoke, answered, then went quiet mid-run.
        (CliTimeouts(startup=None, idle=0.05), (_THREAD_STARTED, *_real_work(3)), "idle_timeout"),
        # Still talking, just taking too long overall.
        (CliTimeouts(startup=None, total=0.05), (_THREAD_STARTED, *_real_work(3)), "total_timeout"),
    ],
)
@pytest.mark.asyncio
async def test_each_liveness_bound_marks_its_own_usage_incomplete_and_names_itself(
    bound: CliTimeouts, lines: tuple[bytes, ...], expected_reason: str
) -> None:
    """All four bounds, through the driver, to a terminal result.

    ``_CliDeadline`` is tested in isolation and the per-CLI suites drive
    ``startup`` and ``idle``, but nothing drove ``first_event`` or ``total`` to a
    RESULT, and nothing anywhere asserted what the usage of a bounded run looks
    like. Both halves matter together: ``partial=True`` is what the adapter turns
    into ``usage_complete=False``, and that flag is the only thing that stops
    ``raw_usage`` being read as a measured zero. A bound that fired while
    leaving ``partial`` unset would produce a run that positively reports no
    spend — which is the defect this file is named for, one branch away.

    ``timeout_kind`` is asserted separately from ``stop_reason`` because
    ``_timeout_stop_reason`` may relabel the latter (a startup hang whose stderr
    says the credential was refused becomes ``authentication_failed``); the
    bound that actually fired must survive that relabelling, or an operator
    cannot tell which clock ran out.
    """
    proc = _Proc(_Stdout(*lines))
    cognition = CodexCliCognition(spawn=_spawning(proc), timeouts=bound)

    result = await _drive(cognition)

    assert result.evals["timeout_kind"] == expected_reason
    assert result.evals["stop_reason"] == expected_reason
    assert result.partial is True, "partial is what becomes usage_complete=False"
    assert result.usage.total_tokens == 0 and result.usage.cost_usd is None
    assert not _usage_was_measured(result.usage)


@pytest.mark.asyncio
async def test_a_stop_button_a_crash_and_a_failed_spawn_all_mark_their_usage_incomplete() -> None:
    """The three non-timeout ways a run ends without its usage payload.

    Cancellation, signal death and a binary that is not on PATH reach
    ``_finalise`` through three different branches of the stop-reason priority,
    and each one produces the same zero-token usage. They are grouped here
    because the property is identical and the branches are not: a fix to one
    that missed another would leave a silent zero reachable through whichever
    was forgotten. ``spawn_failed`` is the one case where zero tokens is also
    the TRUTH — nothing ran — and it still must not be reported as a
    measurement, because the caller cannot tell the two apart from the numbers.
    """
    cancelled = await _drive(
        CodexCliCognition(spawn=_spawning(_Proc(_Stdout(_THREAD_STARTED, *_real_work(3), silent_after=False)))),
        ctx=_StopButton(after_lines=1),
    )
    assert cancelled.evals["stop_reason"] == "cancelled"

    crashed = await _drive(
        CodexCliCognition(
            spawn=_spawning(
                _Proc(_Stdout(_THREAD_STARTED, *_real_work(3), silent_after=False), exit_code=-9)
            )
        )
    )
    assert crashed.evals["stop_reason"] == "process_crashed"

    async def missing(*_argv: str, **_kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", "codex")

    never_ran = await _drive(CodexCliCognition(spawn=missing))
    assert never_ran.evals["stop_reason"] == "spawn_failed"

    for result in (cancelled, crashed, never_ran):
        reason = result.evals["stop_reason"]
        assert result.partial is True, reason
        assert result.usage.total_tokens == 0 and result.usage.cost_usd is None, reason
        assert not _usage_was_measured(result.usage), reason


# ── the path the flag does NOT cover ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_terminal_payload_carrying_no_usage_object_leaves_the_zeros_unqualified() -> None:
    """``turn.completed`` with no ``usage`` field: the stream ENDED properly and
    still told us nothing about tokens.

    Both parsers read the field with ``_as_dict(payload.get("usage"))``, which
    answers ``{}`` for a payload that omits it — so this is not a malformed
    stream, it is a complete one with a missing measurement, and
    ``saw_terminal=True`` carries it straight past every "the run was cut short"
    branch in ``_finalise``. Measured: ``partial=False``,
    ``stop_reason=None``, ``usage.total_tokens == 0``. The money survives
    (``cost_usd is None``, because ``_priced`` checks ``raw_usage`` for the token
    keys before pricing anything); the tokens do not, and nothing on the result
    says so.

    The assertion is on the PREDICATE and on ``raw_usage``, not on ``partial``,
    deliberately. Whether a usage-less terminal payload should also mark the run
    partial is the per-CLI ``_finalise``'s decision and is owned by the driver
    modules; what ``_common.py`` can guarantee — and what this pins — is that
    the condition is detectable from the usage alone, so neither driver has to
    invent its own test for it. Pinning ``partial is False`` here would instead
    turn the driver-side fix into a test failure.
    """
    proc = _Proc(
        _Stdout(
            _THREAD_STARTED,
            b'{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"the answer"}}\n',
            b'{"type":"turn.completed"}\n',
            silent_after=False,
        )
    )

    result = await _drive(CodexCliCognition(spawn=_spawning(proc)))

    assert result.output == "the answer", "precondition: the CLI answered and ended its own stream"
    assert result.evals["raw_usage"] == {}, "precondition: no usage object arrived with the terminal payload"
    assert result.usage.total_tokens == 0
    assert result.usage.cost_usd is None, "the cost guard already refuses to price an unobserved usage"
    assert not _usage_was_measured(result.usage), (
        "the zeros are detectable as un-populated from the usage alone, which is "
        "what a driver needs in order to stop reporting them as final"
    )


# ── the predicate itself ─────────────────────────────────────────────────────


def test_an_unpopulated_usage_is_told_apart_from_a_genuinely_cheap_one() -> None:
    """One definition of "these integers are a measurement", because three
    places currently decide it three different ways.

    ``codex._priced`` asks whether ``raw_usage`` contains the two token keys,
    the CLI adapter asks whether ``result.partial`` is set, and
    ``core/runtime/calls.py`` asks whether ``turn.raw_usage`` is truthy. All
    three are answering the same question about the same object and none of them
    can be reused by the others, which is the hand-copied-judgement pattern this
    module exists to end — the module docstring lists three shipped bugs that
    came from exactly it.

    The predicate's edge is the interesting part: ZERO TOTAL TOKENS WITH AN
    UNKNOWN COST is the un-populated default and nothing else. A real turn
    consumes input tokens — a model cannot be asked nothing — and a run that
    reached its terminal payload came back either with tokens or with a cost. So
    a usage carrying tokens is a measurement even when its cost is unknown
    (every codex run, which the CLI never prices), and a usage carrying a cost
    is a measurement even if some other field is zero (a cache-only turn).
    """
    assert not _usage_was_measured(ProviderUsage()), "the default is what a timed-out run leaves behind"
    assert not _usage_was_measured(ProviderUsage(cost_source=CostSource.UNPRICED))

    assert _usage_was_measured(ProviderUsage(input_tokens=1)), "a model cannot be asked zero tokens"
    assert _usage_was_measured(ProviderUsage(output_tokens=1))
    assert _usage_was_measured(ProviderUsage(cost_usd=0.0)), (
        "a REPORTED zero is a measurement: the provider was asked and said free"
    )
    assert _usage_was_measured(ProviderUsage(input_tokens=200, cache_read_tokens=1000)), (
        "an unpriced codex run has real tokens and no cost, and must not be "
        "mistaken for a run nobody measured"
    )


def test_cache_only_token_counts_do_not_by_themselves_pass_for_a_measurement() -> None:
    """``cache_read_tokens`` without ``input_tokens`` and without a cost is
    still the shape of a payload nobody filled in.

    Stated separately because it is the predicate's one deliberate false
    negative and it must be a decision rather than an accident. ``total_tokens``
    is ``input + output`` only, so a usage whose sole non-zero field is a cache
    counter reads as un-populated — and that is the safe direction: calling an
    unknown measurement "unknown" costs a caller a precision they never had,
    while calling an un-populated default a measurement is the defect this whole
    file is about. Any real payload carrying cache counters carries
    ``input_tokens`` too — in the Claude stream-json fixture under
    ``tests/source/fixtures/cli/claude`` every ``assistant`` usage object has
    both.
    """
    assert not _usage_was_measured(ProviderUsage(cache_read_tokens=90_000))
    assert _usage_was_measured(ProviderUsage(cache_read_tokens=90_000, input_tokens=4))


# ── a result must say whether it is final ────────────────────────────────────


def test_a_cli_result_cannot_be_built_without_stating_whether_the_run_completed() -> None:
    """``partial`` had a default of ``False``, which is the same defect class as
    the ``cost_usd = 0.0`` default that was just removed from ``ProviderUsage``.

    It is not a cosmetic default. The CLI adapter reads ``result.partial`` to
    decide BOTH ``Turn.usage_complete`` and whether to replace the token dict
    with ``{}``, so a ``CliResult`` built without mentioning it claims "this run
    finished and its usage is final" on behalf of code that said nothing — and
    the usage most likely to be attached to such a result is the un-populated
    one, because that is what a path which forgot to set ``partial`` is likely
    to have forgotten to populate too. Both shipped drivers pass it explicitly
    today; this makes that impossible to stop doing silently, which is the only
    reason the ``ProviderUsage`` fix stayed fixed.
    """
    with pytest.raises(TypeError):
        CliResult(output="half an answer", usage=ProviderUsage())  # type: ignore[call-arg]

    stated = CliResult(output="half an answer", usage=ProviderUsage(), partial=True)
    assert stated.partial is True


# ── what a caller's meter is told, and what it cannot be told ───────────────


@pytest.mark.asyncio
async def test_a_meter_hears_unknown_money_and_an_unqualifiable_zero_for_the_tokens() -> None:
    """The hand-off, written down so nobody closes it the wrong way.

    ``_charge_meters`` MUST still charge an unmeasured usage. The codex failure
    suite says why in its own assertion — "a meter must not be told the run was
    free" — and the actor-envelope suite says the other half: a loop on crashed
    runs that charged nothing would advance no ceiling at all and so could never
    be stopped by one. Refusing the charge is therefore not the fix, however
    tempting ``usage == ProviderUsage()`` makes it look.

    What the charge CAN say honestly is bounded by the protocol. Money has an
    unknown value and uses it (``cost_usd=None``, which the envelope is
    documented to skip rather than add). Tokens do not: ``total_tokens`` is an
    ``int``, so the envelope is handed ``0`` and cannot distinguish it from a
    run that genuinely consumed nothing. That asymmetry is the remaining defect
    and it lives in the ``ProviderUsage`` contract, not here — this test pins the
    boundary so the next reader sees exactly where it is instead of re-deriving
    it from a timed-out run.
    """
    meter, envelope = _Meter(), _Envelope()
    unmeasured = ProviderUsage()

    note = await _charge_meters(_Ctx(all_meters=(meter,), actor_budget=envelope), unmeasured, enabled=True)

    assert note is None, "an unmeasured usage is not a meter malfunction"
    assert len(meter.charges) == 1, "the meter must hear about the run at all"
    assert meter.charges[0].cost_usd is None, "unknown money stays unknown"
    assert meter.charges[0].cost_source is CostSource.UNPRICED
    assert not _usage_was_measured(meter.charges[0])
    assert envelope.charges == [(0, None, 1)], (
        "the step is charged because a step really happened; the token figure is "
        "a zero the envelope has no way to read as 'unknown'"
    )
    assert envelope.tokens == 0, (
        "the measured consequence: an actor looping on timed-out runs never "
        "approaches its token ceiling, while its step ceiling does close"
    )
