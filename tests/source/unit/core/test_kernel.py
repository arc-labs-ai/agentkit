"""The retry chokepoint: backoff, the attempt budget, and the breaker's gate.

WHY THIS FILE EXISTS
--------------------
`core/execution/resilience.py` is the single place that decides whether a failed
provider call is tried again, how long the caller waits first, and when a
dependency is taken out of rotation. Everything in it is cheap to break
silently, because a broken retry layer still returns the right answer on the
happy path:

* backoff that never fires looks identical to backoff that fires — until a
  retry storm takes the dependency down. Measured once the sleep had been
  refactored into an unreachable `else` branch: 3 attempts, `sleep calls: []`,
  and no test noticed.
* a probe slot that is consumed and never handed back looks identical to a
  healthy breaker — until a single 401 refuses a recovered dependency for the
  rest of the process lifetime.
* a lock that is not taken looks identical to a lock that is taken — until two
  threads both become "the" half-open probe and stampede the dependency the
  gate exists to protect.

So each test below asserts an OBSERVABLE consequence of the invariant (how many
times `fn` ran, how long the caller was asked to wait, which calls the gate
admitted), not the shape of the code. Every clock, sleep and rng is injected:
nothing here sleeps, and the whole file runs in milliseconds.

The substring classifier's verdicts are not re-litigated here —
`tests/source/unit/models/adapters/test_transport_classifier_agreement.py`
owns the agreement between this module's `classify` and the structured
`models.contracts.failures.classify`. This file owns what the retry loop DOES
with a verdict once it has one.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from agentkit.core.contracts.types import ChatRequest, Message, MessageRole, ToolRequest, ToolSchema
from agentkit.core.execution.resilience import (
    CircuitBreaker,
    CircuitOpen,
    ErrorClass,
    backoff_delay,
    classify,
    run_with_resilience,
)

# ── test doubles ─────────────────────────────────────────────────────────────


class _CeilingRng:
    """A deterministic stand-in for `random`: always returns the full ceiling.

    Full-jitter backoff draws uniformly from `[0, ceiling]`, so a real rng makes
    the delay unassertable. Returning the ceiling pins the SCHEDULE (the upper
    bound each attempt is allowed to draw from) while leaving the production
    arithmetic completely untouched.
    """

    def __init__(self) -> None:
        self.ceilings: list[float] = []

    def uniform(self, low: float, high: float) -> float:
        assert low == 0, "full jitter draws from zero; a non-zero floor removes the jitter"
        self.ceilings.append(high)
        return high


class _RecordingSleep:
    """An async sleep that records what it was asked to wait and returns at once.

    The recording IS the assertion for most of this file: "did the caller back
    off, and for how long" is the whole retry contract, and it is invisible from
    the return value of `run_with_resilience`.
    """

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class _Clock:
    """A hand-wound monotonic clock, so cooldowns elapse without elapsing."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _transient() -> BaseException:
    """An exception the substring classifier reads as TRANSIENT."""
    exc = TimeoutError("upstream timed out")
    assert classify(exc) is ErrorClass.TRANSIENT, "precondition: this fixture must be retryable"
    return exc


def _permanent() -> BaseException:
    """An exception the substring classifier reads as PERMANENT (a 401)."""
    exc = RuntimeError("401 unauthorized: invalid api key")
    assert classify(exc) is ErrorClass.PERMANENT, "precondition: this fixture must fail fast"
    return exc


def _unknown() -> BaseException:
    """An exception the substring classifier can say nothing about."""
    exc = RuntimeError("zzz")
    assert classify(exc) is ErrorClass.UNKNOWN, "precondition: this fixture must be unclassifiable"
    return exc


def _raiser(exc: BaseException) -> tuple[Any, list[int]]:
    """An async callable that always raises `exc`, plus its call log."""
    calls: list[int] = []

    async def fn() -> Any:
        calls.append(len(calls) + 1)
        raise exc

    return fn, calls


# ── backoff: the delay has to actually happen, and stay bounded ──────────────


@pytest.mark.asyncio
async def test_a_retry_waits_between_attempts_and_never_after_the_last_one() -> None:
    """The retry loop must back off ON the retrying path.

    This is not hypothetical tidiness: refactoring the function into
    try/except/else moved the `await asleep(...)` below `return result` in the
    `else` block, where it is unreachable. Every retry became a hot loop with
    zero delay — the retry storm `backoff_delay` exists to prevent — and
    nothing failed, because the call still eventually raised the same error.
    `ruff` does not flag it and coverage cannot see it, so the number of sleeps
    and their values are asserted directly.

    The exact values also pin the SHAPE of the schedule: the jitter ceiling
    doubles per attempt (`base * 2**(attempt-1)`), so a backoff that stopped
    growing — the other way a storm gets through — fails here too. And the
    wait count is `attempts - 1`: sleeping after the final attempt would charge
    the caller a delay for a retry that never happens.
    """
    sleeper = _RecordingSleep()
    rng = _CeilingRng()
    fn, calls = _raiser(_transient())

    with pytest.raises(TimeoutError):
        await run_with_resilience(fn, max_attempts=3, sleep=sleeper, rng=rng)

    assert len(calls) == 3, "the attempt budget was not spent"
    assert sleeper.waits == [0.5, 1.0], "the retry did not back off, or stopped doubling"
    assert rng.ceilings == [0.5, 1.0], "the delay was not drawn as full jitter under the ceiling"


def test_a_huge_attempt_number_saturates_the_backoff_instead_of_raising() -> None:
    """A retry helper must not crash on its own arithmetic.

    `2 ** (attempt - 1)` is an arbitrary-precision int, and converting one above
    `2**1024` to float RAISES: measured, `backoff_delay(1025)` gave
    `OverflowError: int too large to convert to float`. That turns a
    recoverable error into an unrecoverable one at the exact moment the system
    is already failing — a long-lived caller with a high attempt counter gets a
    crash out of the component that exists to absorb crashes.

    The clamp is exact rather than approximate: `base * 2**1023` already exceeds
    any sane cap, so saturating the exponent cannot change the delay anyone ever
    observes. Both halves are asserted — no exception, and the value still
    lands on the cap.
    """
    rng = _CeilingRng()

    assert backoff_delay(1025, cap=30.0, rng=rng) == 30.0
    assert backoff_delay(10**6, cap=30.0, rng=rng) == 30.0
    assert rng.ceilings == [30.0, 30.0], "a saturated attempt must still be capped, not infinite"


@pytest.mark.parametrize("attempt", [-5, 0, 1, 2, 10, 60, 1024, 4096])
def test_the_backoff_is_always_inside_zero_and_the_cap(attempt: int) -> None:
    """The cap is the promise the caller plans around: it is what bounds how
    long a run can be stalled by retries, and a delay above it (or below zero,
    which `asyncio.sleep` accepts silently) is unbounded wall-clock time nobody
    budgeted for. Driven with the REAL rng, so the jitter draw itself is in
    scope, and across attempts on both sides of the saturation clamp — the
    extremes are exactly where an off-by-one in the exponent escapes.
    """
    for _ in range(50):
        delay = backoff_delay(attempt, base=0.5, cap=2.0)
        assert 0.0 <= delay <= 2.0, f"attempt {attempt} drew {delay}, outside [0, cap]"


# ── the attempt budget ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("max_attempts", [1, 2, 3, 5])
async def test_max_attempts_is_the_exact_number_of_calls(max_attempts: int) -> None:
    """`max_attempts` is a budget, not a hint, and it is counted in CALLS —
    the first try is attempt one, not a free extra.

    Both directions are failures with teeth. One call too many multiplies the
    load a failing dependency sees and the cost of a paid API call; one too few
    turns a configured 3-attempt retry into a 2-attempt retry, so the flake the
    policy was sized for still fails the run. The off-by-one is invisible in
    normal operation, which is why the count is pinned at both ends of the range
    — including `max_attempts=1`, which must mean "no retries at all".
    """
    sleeper = _RecordingSleep()
    fn, calls = _raiser(_transient())

    with pytest.raises(TimeoutError):
        await run_with_resilience(fn, max_attempts=max_attempts, sleep=sleeper)

    assert len(calls) == max_attempts
    assert len(sleeper.waits) == max_attempts - 1, "one backoff per RETRY, not per attempt"


@pytest.mark.asyncio
async def test_a_permanent_verdict_makes_exactly_one_attempt_and_never_sleeps() -> None:
    """Failing fast is the whole point of classification.

    A 401, a 403 or a content-filter rejection cannot succeed on a second
    attempt, so retrying one burns the attempt budget, the latency budget and —
    for a metered provider — money, and it delays the only useful outcome
    (surfacing the real error) by the full backoff. The original exception must
    also propagate unchanged: wrapping it would cost the caller the diagnosis.
    """
    sleeper = _RecordingSleep()
    exc = _permanent()
    fn, calls = _raiser(exc)

    with pytest.raises(RuntimeError) as excinfo:
        await run_with_resilience(fn, max_attempts=5, sleep=sleeper)

    assert excinfo.value is exc, "the permanent failure must reach the caller as itself"
    assert len(calls) == 1, "a permanent failure was retried"
    assert sleeper.waits == [], "the caller paid a backoff for a retry that never happened"


@pytest.mark.asyncio
async def test_an_unclassifiable_failure_is_still_retried() -> None:
    """UNKNOWN must retry, and this is the control on the test above.

    Only PERMANENT fails fast. The conservative choice on a verdict the
    substring classifier cannot make is to try again: one wasted retry costs a
    backoff, while failing fast on a transient error costs the whole call. If
    "fail fast on PERMANENT" ever widens to "fail fast unless proven
    TRANSIENT", every provider error the message matcher has not seen before
    silently stops being retried — and substring matching guarantees there will
    be such errors.
    """
    sleeper = _RecordingSleep()
    fn, calls = _raiser(_unknown())

    with pytest.raises(RuntimeError):
        await run_with_resilience(fn, max_attempts=3, sleep=sleeper)

    assert len(calls) == 3, "an unknown failure was treated as permanent"


@pytest.mark.asyncio
async def test_a_custom_classifier_may_report_the_verdict_as_its_string_value() -> None:
    """`classify_fn` is the documented escape hatch — the module's own docstring
    sends callers to it, because substring matching is brittle by nature — and
    `ErrorClass` is a `StrEnum`, so its members ARE their string values.

    The retry loop compared the verdict with `is`, which is identity, and a
    `StrEnum` member is not identical to an equal string. So a hand-written
    classifier returning `"permanent"` (the obvious thing to write in untyped
    code, and what survives any JSON / config round trip) was read as
    not-permanent and the failure was RETRIED — measured: three attempts on a
    401 that the caller had explicitly classified as permanent. Silent, and
    precisely the budget-burning retry this module exists to prevent;
    `mypy --strict` catches it in a typed codebase and nothing catches it
    anywhere else.
    """
    sleeper = _RecordingSleep()
    fn, calls = _raiser(_transient())  # a message the DEFAULT classifier calls transient

    with pytest.raises(TimeoutError):
        await run_with_resilience(
            fn,
            max_attempts=3,
            sleep=sleeper,
            classify_fn=lambda _exc: ErrorClass.PERMANENT.value,  # type: ignore[arg-type,return-value]
        )

    assert len(calls) == 1, "a caller's own PERMANENT verdict was ignored and the call was retried"
    assert sleeper.waits == []


@pytest.mark.asyncio
async def test_a_cancelled_call_is_not_retried_and_reaches_the_caller_untouched() -> None:
    """Cancellation is an instruction, not a failure.

    `CancelledError` is a `BaseException`, so it bypasses the `except Exception`
    that classifies and retries — and it must keep bypassing it. Retrying a
    cancelled call ignores the caller's cancellation and keeps a dependency
    busy after the work has been abandoned; worse, under `asyncio` a swallowed
    `CancelledError` breaks task cancellation itself, so a shutdown hangs
    instead of unwinding.
    """
    sleeper = _RecordingSleep()
    fn, calls = _raiser(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await run_with_resilience(fn, max_attempts=3, sleep=sleeper)

    assert len(calls) == 1, "a cancelled call was retried"
    assert sleeper.waits == [], "a cancelled caller was made to wait"


# ── the breaker: equality, the lock, and the single-probe gate ───────────────


def test_two_identically_configured_breakers_compare_equal() -> None:
    """A breaker is configuration plus state, and both are compared.

    Its mutex and its remembered last error are IMPLEMENTATION, excluded from
    `==` and `repr`. Letting the lock into equality makes every breaker unequal
    to every other breaker — including one that is identical in every way a
    caller can observe — which silently breaks anything that compares, dedupes
    or caches configuration (a registry keyed by its dependency policy, a test
    asserting a built runtime matches its expected shape). It also drags a
    `_thread.lock` into `repr`, so a breaker cannot be logged cleanly.
    """
    clock = _Clock()
    one = CircuitBreaker("upstream", fail_threshold=3, cooldown=15.0, clock=clock)
    two = CircuitBreaker("upstream", fail_threshold=3, cooldown=15.0, clock=clock)

    assert one == two, "a non-observable field leaked into equality"
    assert "_lock=" not in repr(one), "the mutex leaked into repr"

    one.record_failure(_transient())
    assert one != two, "state is observable and MUST still be compared"


def test_counting_a_failure_holds_the_lock_for_the_whole_transition() -> None:
    """Every transition is a read-modify-write on shared fields, and the
    documented deployment is ONE breaker per dependency shared across a thread
    pool. Without the lock two threads can both read `open`, both flip it to
    `half_open`, and both become "the" single probe — sending a herd at the
    dependency the gate exists to protect from exactly that. The window is
    small, so it will not reproduce on demand; what can be checked is that the
    mutex is genuinely HELD while the breaker is mid-transition.

    The injected clock is the instrument: `record_failure` calls it inside the
    critical section, on the branch that opens the breaker. From there, another
    thread must not be able to take the lock. Deterministic and instant — no
    sleeps, no racing.
    """
    acquired_elsewhere: list[bool] = []
    breaker: CircuitBreaker | None = None

    def clock() -> float:
        def probe() -> None:
            assert breaker is not None
            got = breaker._lock.acquire(blocking=False)
            acquired_elsewhere.append(got)
            if got:
                breaker._lock.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "the probe thread must not block; it asks without waiting"
        return 1000.0

    breaker = CircuitBreaker("upstream", fail_threshold=1, clock=clock)
    breaker.record_failure(_transient())

    assert breaker.state == "open", "the transition under test did not happen"
    assert acquired_elsewhere == [False], "another thread entered a transition already in progress"


def test_a_probe_that_never_reports_is_presumed_lost_after_one_cooldown() -> None:
    """`half_open` IS the single-probe gate, so the probe reporting back is
    load-bearing — and it is an assumption the callers cannot honour. The retry
    loop deliberately skips `record_failure` on a PERMANENT error, and a
    `BaseException` escapes its `except Exception` entirely. Measured before the
    guard: one 401 on the post-cooldown probe left `state == "half_open"` and
    every later call — against a fully healthy dependency, 10,000 simulated
    seconds later — raised `CircuitOpen` for the rest of the process lifetime.
    A breaker that never closes again is strictly worse than no breaker.

    So after one cooldown with no verdict the probe is presumed lost and a
    FRESH one is admitted — still exactly one at a time, which is the
    no-stampede property the state exists for, and the assertions below pin
    both halves.
    """
    clock = _Clock()
    breaker = CircuitBreaker("upstream", fail_threshold=1, cooldown=30.0, clock=clock)
    breaker.record_failure(_transient())
    assert breaker.state == "open"

    clock.advance(31)
    assert breaker.allow() is True and breaker.state == "half_open"
    assert breaker.allow() is False, "a second caller slipped past the single-probe gate"

    clock.advance(29)  # the probe is silent, but one cooldown has NOT passed
    assert breaker.allow() is False, "a replacement probe was admitted before the cooldown elapsed"

    clock.advance(2)  # now it has
    assert breaker.allow() is True, "a silent probe wedged the gate forever"
    assert breaker.allow() is False, "presuming the probe lost must still admit only ONE"


def test_releasing_the_probe_restarts_the_cooldown_so_the_breaker_still_brakes() -> None:
    """The neutral release returns the slot to OPEN, and the cooldown has to
    start over.

    The first version did not restart it, reasoning that a permanently-failing
    dependency should keep surfacing its real 401 rather than hide behind a
    `CircuitOpen`. The effect was a breaker that does not brake: an
    already-elapsed cooldown makes EVERY caller a fresh probe, not just the
    next one, so twenty consecutive calls spanning two milliseconds all reached
    the dead dependency. The premise was wrong anyway — the refusal is raised
    `from` the real error, so nothing was being masked; the rate limiting was
    simply absent.
    """
    clock = _Clock()
    breaker = CircuitBreaker("upstream", fail_threshold=1, cooldown=60.0, clock=clock)
    breaker.record_failure(_permanent())
    clock.advance(61)
    assert breaker.allow() is True and breaker.state == "half_open"

    breaker.release_probe()
    assert breaker.state == "open", "the slot was not returned"

    admitted = 0
    for _ in range(20):  # a herd arriving right behind the released probe
        clock.advance(0.0001)
        admitted += breaker.allow()
    assert admitted == 0, "the released probe left the gate wide open"

    clock.advance(61)
    assert breaker.allow() is True, "the restarted cooldown must still expire"


def test_the_neutral_release_grants_no_health_credit() -> None:
    """`release_probe` is the third edge precisely because neither of the other
    two fits: `record_failure` would fold a contract failure (a 401) into the
    consecutive-failure count, breaking the deliberate rule that it is not an
    upstream health signal, and `record_success` would grant credit for a call
    that never proved the dependency recovered.

    So it must consume nothing and grant nothing. Resetting the counter here
    reads a 401 as a RECOVERY: the breaker arrives back at OPEN with a clean
    bill of health, and the next real failure has to climb the whole threshold
    again before the dependency is taken out of rotation — the failure count is
    the only memory the breaker has, and this is the one edge that is supposed
    to leave it alone. The counter is private but it is also the state the
    dataclass compares and reprs, so it is read directly rather than inferred.
    """
    clock = _Clock()
    breaker = CircuitBreaker("upstream", fail_threshold=3, cooldown=10.0, clock=clock)
    for _ in range(3):
        breaker.record_failure(_transient())
    clock.advance(11)
    assert breaker.allow() is True and breaker.state == "half_open"

    accumulated = breaker._fails
    assert accumulated == 3, "precondition: the breaker has three failures on record"

    breaker.release_probe()

    assert breaker.state == "open"
    assert breaker._fails == accumulated, "a neutral release granted health credit"


# ── the breaker inside the retry loop ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_permanent_failure_on_the_probe_returns_the_slot_instead_of_wedging_it() -> None:
    """The two rules the retry loop has to honour at once: a PERMANENT error is
    not a dependency health signal (so it must not be counted), but the call
    still CONSUMED the half-open probe slot (so it must be released).

    Doing the first and forgetting the second is what shipped: one 401 on the
    post-cooldown probe refused a healthy dependency for the rest of the
    process lifetime, because the only exits from `half_open` were the two
    methods that call was deliberately skipping. The state after the call is
    therefore asserted directly — reaching OPEN is the proof the slot came
    back, and a later successful probe is the proof the breaker can still
    recover.
    """
    clock = _Clock()
    breaker = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=clock)
    breaker.record_failure(_transient())
    clock.advance(11)

    sleeper = _RecordingSleep()
    fn, calls = _raiser(_permanent())
    with pytest.raises(RuntimeError):
        await run_with_resilience(fn, breaker=breaker, max_attempts=3, sleep=sleeper)

    assert len(calls) == 1, "the permanent failure was retried"
    assert breaker.state == "open", "the probe slot was consumed and never handed back"

    clock.advance(11)

    async def healthy() -> str:
        return "recovered"

    assert await run_with_resilience(healthy, breaker=breaker, max_attempts=3, sleep=sleeper) == "recovered"
    assert breaker.state == "closed", "the breaker could no longer recover"


@pytest.mark.asyncio
async def test_a_permanent_failure_never_trips_the_breaker() -> None:
    """A 401 / 403 / content-filter rejection is a contract failure between the
    caller and the provider, not evidence that the dependency is unhealthy.
    Counting one toward the threshold takes a perfectly healthy dependency out
    of rotation for a whole cooldown because the CALLER sent a bad request —
    and with a shared breaker it does that to every other caller too.

    The threshold is one here, so a single counted failure would be visible
    immediately.
    """
    clock = _Clock()
    breaker = CircuitBreaker("upstream", fail_threshold=1, cooldown=60.0, clock=clock)
    sleeper = _RecordingSleep()

    for _ in range(3):
        fn, _calls = _raiser(_permanent())
        with pytest.raises(RuntimeError):
            await run_with_resilience(fn, breaker=breaker, max_attempts=2, sleep=sleeper)

    assert breaker.state == "closed", "a contract failure opened the breaker"
    assert breaker.allow() is True, "a healthy dependency was taken out of rotation"


@pytest.mark.asyncio
async def test_a_refused_call_still_names_the_failure_that_opened_the_breaker() -> None:
    """A refusal has to carry a diagnosis.

    The retry loop's own `last` is a LOCAL, so a separate invocation that is
    refused by an already-open breaker had nothing to chain from, and the caller
    saw a bare `CircuitOpen: circuit open: upstream` — the breaker name and
    nothing else. That is the postmortem for an outage arriving with the cause
    deleted, and it is the common case: the call that trips the breaker is
    rarely the call whose traceback anyone reads.

    So the breaker remembers the failure that opened it and the refusal is
    raised `from` it. This also covers the other half of the contract: a
    refused call must not reach the dependency at all.
    """
    clock = _Clock()
    breaker = CircuitBreaker("upstream", fail_threshold=1, cooldown=60.0, clock=clock)
    sleeper = _RecordingSleep()

    boom = _transient()
    fn, calls = _raiser(boom)
    with pytest.raises(TimeoutError):
        await run_with_resilience(fn, breaker=breaker, max_attempts=1, sleep=sleeper)
    assert breaker.state == "open" and len(calls) == 1

    refused, refused_calls = _raiser(_transient())
    with pytest.raises(CircuitOpen) as excinfo:
        await run_with_resilience(refused, breaker=breaker, max_attempts=3, sleep=sleeper)

    assert refused_calls == [], "a refused call still reached the dependency"
    assert excinfo.value.__cause__ is boom, "the refusal lost the failure that opened the breaker"


@pytest.mark.asyncio
async def test_a_refusal_from_a_nested_breaker_hands_the_probe_slot_back() -> None:
    """A `CircuitOpen` raised by the call never reached THIS dependency, so it
    is the textbook outcome that says nothing about its health — and the probe
    slot it consumed has to come back.

    Nesting is the documented shape: one breaker per dependency, and
    `run_with_resilience` composes (`recurrence.py` wraps it), so a call guarded
    by a shared provider breaker can run inside a per-model one. When the inner
    gate refuses, the re-raise bypasses both `record_failure` and
    `release_probe`, which is the same wedge a PERMANENT probe failure used to
    cause: the outer breaker sits in `half_open` holding a slot nothing will
    ever release. Measured before the fix: the outer breaker stayed
    `half_open`, so a fully recovered dependency was refused for another whole
    cooldown — the lost-probe guard is what caps the damage, and it is a
    backstop, not the contract.

    Counting it instead would be worse than wedging: it folds a DIFFERENT
    dependency's outage into this one's failure count.
    """
    clock = _Clock()
    sleeper = _RecordingSleep()

    inner = CircuitBreaker("inner", fail_threshold=1, cooldown=600.0, clock=clock)
    inner.record_failure(_transient())
    assert inner.allow() is False, "precondition: the inner gate refuses"

    outer = CircuitBreaker("outer", fail_threshold=2, cooldown=30.0, clock=clock)
    outer.record_failure(_transient())
    outer.record_failure(_transient())
    clock.advance(31)

    dependency_calls: list[int] = []

    async def through_the_inner_gate() -> Any:
        async def dependency() -> str:
            dependency_calls.append(1)
            return "unreachable"

        return await run_with_resilience(dependency, breaker=inner, max_attempts=3, sleep=sleeper)

    with pytest.raises(CircuitOpen):
        await run_with_resilience(through_the_inner_gate, breaker=outer, max_attempts=3, sleep=sleeper)

    assert dependency_calls == [], "precondition: the inner gate refused before any call"
    assert outer._fails == 2, "another dependency's outage was counted against this one"
    assert outer.state == "open", "the inner refusal wedged the outer breaker in half_open"

    clock.advance(31)
    assert outer.allow() is True, "the outer breaker could no longer admit a probe"

    with pytest.raises(CircuitOpen):  # and with no outer gate at all, it simply propagates
        await run_with_resilience(through_the_inner_gate, max_attempts=3, sleep=sleeper)


@pytest.mark.asyncio
async def test_cancelling_the_probe_hands_the_slot_back_without_counting_it() -> None:
    """Cancellation is the other half of the `half_open` wedge.

    `CancelledError` and `SystemExit` are `BaseException`s, so they escape the
    `except Exception` that reports to the breaker entirely — and the call that
    was cancelled was the single post-cooldown probe. Left unreported, the
    breaker holds a slot nobody will ever release; reported as a FAILURE, a
    shutdown or a caller's own timeout is recorded as evidence that the
    dependency is unhealthy, so the next deployment starts with a breaker
    already half-tripped by its predecessor's teardown.

    Neither, so: the slot goes back, the count does not move, and the
    cancellation unwinds untouched.
    """
    clock = _Clock()
    breaker = CircuitBreaker("upstream", fail_threshold=1, cooldown=20.0, clock=clock)
    breaker.record_failure(_transient())
    clock.advance(21)

    sleeper = _RecordingSleep()
    fn, calls = _raiser(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await run_with_resilience(fn, breaker=breaker, max_attempts=3, sleep=sleeper)

    assert len(calls) == 1, "a cancelled probe was retried"
    assert breaker.state == "open", "the cancelled probe held the slot"
    assert breaker._fails == 1, "cancellation was counted as a dependency failure"


def test_a_refusal_reads_as_transient_so_a_caller_can_degrade() -> None:
    """`CircuitOpen` is the breaker telling the caller "not now", not "never".

    The caller decides between degrading and propagating, and an outer retry
    layer that classified it PERMANENT would turn a cooldown-length brownout
    into a failed run. Reading it as TRANSIENT is also what makes a refusal
    count as a retryable failure to the composed `Failure` taxonomy, so the two
    layers do not disagree about the same exception.
    """
    assert classify(CircuitOpen("circuit open: upstream")) is ErrorClass.TRANSIENT


def test_a_straggler_failure_does_not_push_the_cooldown_out() -> None:
    """A breaker that never gets to probe is a breaker that never closes.

    `record_success` already refuses to act on a call admitted before the trip,
    and documents why at length: that call "is evidence about the past, not
    about whether the dependency has recovered". `record_failure` did the
    opposite. Every straggler draining out of the pool after the trip still
    satisfied `_fails >= fail_threshold`, so it reset `_opened_at` to the moment
    IT happened to finish — pushing admission a full cooldown further out each
    time.

    Measured before the fix with `cooldown=15` and ten in-flight calls draining
    over 20s: a probe was refused at +34s and first admitted at +36s. A slowly
    draining pool extends that without bound, so a dependency that recovered
    during the cooldown stays shut out by the backlog of its own outage — the
    exact stampede-avoidance mechanism starving the recovery it exists to find.

    The half-open probe is the opposite case and is asserted separately below:
    that one call IS admitted under the current state, so its verdict is about
    now and must restart the clock.
    """
    now = 0.0

    def clock() -> float:
        return now

    breaker = CircuitBreaker("upstream", fail_threshold=2, cooldown=15.0, clock=clock)
    breaker.record_failure(RuntimeError("first"))
    breaker.record_failure(RuntimeError("tripped it"))
    assert breaker.state == "open"

    # Stragglers admitted before the trip, finishing over the next 20 seconds.
    for moment in (5.0, 10.0, 15.0, 20.0):
        now = moment
        breaker.record_failure(RuntimeError("admitted before the trip"))

    now = 15.0001
    assert breaker.allow(), (
        "the cooldown was measured from the last straggler instead of from the trip, "
        f"so a recovered dependency is still refused at +{now}s"
    )
    assert breaker.state == "half_open", "the post-cooldown admission must be the single probe"


def test_a_probe_failure_does_restart_the_cooldown() -> None:
    """The twin of the test above, and the reason the fix is a state check rather
    than "stop resetting the clock". The half-open probe is the one call admitted
    under the CURRENT state, so its failure is the only current evidence the
    breaker ever gets — it must reopen the gate and start a fresh cooldown, or a
    dependency that is still down gets probed in a tight loop."""
    now = 0.0

    def clock() -> float:
        return now

    breaker = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=clock)
    breaker.record_failure(RuntimeError("trip"))
    now = 10.0
    assert breaker.allow() and breaker.state == "half_open"

    breaker.record_failure(RuntimeError("the probe failed"))
    assert breaker.state == "open"
    now = 19.9
    assert not breaker.allow(), "a failed probe must buy a full fresh cooldown"
    now = 20.1
    assert breaker.allow(), "and exactly one, measured from the probe's own failure"


# ─────────────────────────────────────────────────────────────────────────────
# Hashability of the request/schema values.
#
# Each of these three types carries a decoded-JSON payload — a schema body, a
# transcript, a tool's arguments — and each deliberately leaves it OUT of
# `__hash__`. Two different things go wrong if one creeps back in, and only the
# first of them raises:
#
#   * `dict` and `list` are unhashable, so a type that folds its payload into
#     its hash cannot go in a set or be a dict key AT ALL once the payload is
#     non-empty. The empty-payload case still works, which is exactly why a test
#     built on a bare instance proves nothing.
#   * even where it is hashable, the cost becomes O(payload). `ChatRequest`
#     hashes `len(self.messages)` rather than the messages: in an agent loop the
#     transcript is the thing that grows, so hashing the whole thing makes every
#     cache probe more expensive as the run goes on.
#
# These were parked invariants whose test file was deleted. They name this file,
# so they live here.
# ─────────────────────────────────────────────────────────────────────────────


def test_a_tool_schema_stays_hashable_however_deep_its_json_schema_is() -> None:
    """The JSON Schema body is a `dict`, so folding it into the hash makes every
    tool with actual parameters unhashable — and tools with parameters are the
    only interesting kind. Identity is the name and description; the body is
    payload, and two schemas differing only in body are the same tool advertised
    differently."""
    nested = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "opts": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        "required": ["q"],
    }
    schema = ToolSchema("lookup", "Look a fact up", nested)

    assert hash(schema) == hash(ToolSchema("lookup", "Look a fact up", {})), (
        "the schema body must not reach the hash: identity is the name and description"
    )
    # Hash and equality deliberately disagree, and that is the contract: the
    # hash is a cheap BUCKET that ignores the payload, while `__eq__` stays
    # exact. Same bucket, still distinct objects — which is what lets a set hold
    # both without the hash ever touching the schema body.
    other = ToolSchema("lookup", "Look a fact up", {"type": "object"})
    assert hash(schema) == hash(other) and schema != other
    assert len({schema, other}) == 2
    assert {schema: "ok"}[schema] == "ok"


def test_a_chat_request_hash_does_not_grow_with_the_transcript() -> None:
    """`ChatRequest` hashes `len(self.messages)`, not the messages. In an agent
    loop the transcript is precisely the part that grows, so hashing it makes
    every cache probe cost more as the run goes on — and `list[Message]` is
    unhashable, so folding it in does not merely slow the probe down, it stops
    the type being usable as a key at all."""
    one = ChatRequest([Message(MessageRole.USER, "hi")], "m")
    many = ChatRequest([Message(MessageRole.USER, f"turn {i}") for i in range(200)], "m")

    assert hash(one) == hash(ChatRequest([Message(MessageRole.USER, "different text")], "m")), (
        "identity must not depend on message CONTENT, only on how many there are"
    )
    assert hash(one) != hash(many), "the message COUNT is deliberately part of identity"
    assert {one, many} == {one, many} and len({one, many}) == 2
    assert {many: "ok"}[many] == "ok", "a long transcript must still be usable as a key"


def test_a_tool_request_stays_hashable_however_nested_its_arguments() -> None:
    """`arguments` is decoded JSON straight off the wire, so it is routinely a
    nested `dict` — unhashable. Identity here is the tool, whether it has effects
    and which argument carries a URL; the arguments themselves are the payload
    that identity is checked BEFORE, which is the whole reason they are excluded."""
    request = ToolRequest("fetch", {"url": "https://x.test", "headers": {"a": ["b", "c"]}}, tool=None, url_arg="url")

    assert hash(request) == hash(ToolRequest("fetch", {}, tool=None, url_arg="url")), (
        "the decoded arguments must not reach the hash"
    )
    # Same bucket, still distinct — see the ToolSchema test for why hash and
    # equality are allowed to disagree here.
    other = ToolRequest("fetch", {"url": "https://y.test"}, tool=None, url_arg="url")
    assert hash(request) == hash(other) and request != other
    assert {request: "ok"}[request] == "ok"


def test_a_spend_cap_cannot_be_given_as_a_float() -> None:
    """Money is `Decimal` and the annotation was the only thing saying so.

    `math.isfinite` accepts a float, so `CallLimits(max_cost_usd=0.1)`
    constructed happily and the mistake surfaced far away and much later: the
    delegation path multiplies the cap by a `Decimal` budget share, so the run
    worked until the first delegation and then raised `TypeError: unsupported
    operand type(s) for *: 'float' and 'decimal.Decimal'` — with no mention of
    the caller's cap anywhere in it.

    Refused rather than coerced, and that is the decision worth pinning. A float
    spend cap is how a limit silently drifts: `0.1 + 0.2` is already
    `0.30000000000000004` before it reaches here, so quietly accepting it would
    preserve the drift while hiding it. `int` and `str` convert exactly, so they
    are accepted — the caller just has to say which they meant.
    """
    from decimal import Decimal

    from agentkit.models.contracts.types import CallLimits

    with pytest.raises(TypeError, match="must be a Decimal, not a float"):
        CallLimits(max_cost_usd=0.1)

    for exact in (Decimal("0.10"), 1, "0.10"):
        limits = CallLimits(max_cost_usd=exact)
        assert isinstance(limits.max_cost_usd, Decimal), f"{exact!r} should convert exactly"
        # The arithmetic the delegation path actually does, which is what the
        # late TypeError came from.
        assert limits.max_cost_usd * Decimal("0.25") == limits.max_cost_usd / 4

    assert CallLimits(max_cost_usd="0.10").max_cost_usd == Decimal("0.10")
