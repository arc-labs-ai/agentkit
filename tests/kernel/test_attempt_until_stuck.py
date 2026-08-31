"""``attempt_until_stuck`` — the bound is RECURRENCE, not a count.

The invariant every test here defends, stated once: three attempts producing
three different failure signatures is progress, and two producing the same one
is not. A count cannot tell those apart, so these tests pin the two halves
separately — a repeat must stop the loop *with attempts remaining*, and
distinct signatures must be allowed to keep going up to the backstop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.errors import Failure
from agentkit.kernel.recurrence import Stuck, attempt_until_stuck
from agentkit.kernel.resilience import ErrorClass, run_with_resilience
from agentkit.testing import make_test_ctx


def _scripted(outcomes: list[Any]) -> tuple[Any, list[int]]:
    """An async ``fn`` that yields ``outcomes`` in order, plus a call counter
    the assertions read to prove the loop STOPPED rather than merely returned."""
    calls: list[int] = []

    async def fn() -> Any:
        calls.append(len(calls) + 1)
        return outcomes[len(calls) - 1]

    return fn, calls


async def _no_sleep(_seconds: float) -> None:
    return None


# ── the headline case ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_three_distinct_signatures_then_success_is_progress():
    """Three different failures then a win. A ``max_attempts=3`` retry loop
    would have given up one attempt short; recurrence-bounding lets it run
    because nothing ever repeated."""
    fn, calls = _scripted(["a", "b", "c", "done"])
    out = await attempt_until_stuck(
        fn,
        fingerprint=lambda o: None if o == "done" else o,
        on_repeat="stop",
        max_attempts=6,
    )
    assert out == "done"
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_a_repeat_stops_early_with_attempts_remaining():
    """The other half: a repeated signature ends the loop at attempt 2 even
    though 8 more were budgeted. If this ever calls ``fn`` a third time the
    bound has silently reverted to a count."""
    fn, calls = _scripted(["same", "same", "same", "same"])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=10)
    assert isinstance(out, Failure)
    assert len(calls) == 2
    assert out.category is ErrorClass.PERMANENT and out.retriable is False
    assert out.source == "attempt_until_stuck"


@pytest.mark.asyncio
async def test_max_attempts_is_a_backstop_not_the_bound():
    """Signatures that keep differing run until the backstop, and exhaustion is
    reported as UNKNOWN/retriable — the run was still moving, it just ran out
    of budget, which is a different finding from being stuck."""
    fn, calls = _scripted(["a", "b", "c", "d", "e"])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=3)
    assert isinstance(out, Failure)
    assert len(calls) == 3
    assert out.category is ErrorClass.UNKNOWN and out.retriable is True
    assert out.partial_output == "c"


# ── the two on_repeat modes ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_repeat_escalate_raises_stuck_carrying_the_failure():
    fn, _calls = _scripted(["x", "x"])
    with pytest.raises(Stuck) as excinfo:
        await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="escalate", max_attempts=9)
    failure = excinfo.value.failure
    assert isinstance(failure, Failure)
    assert failure.category is ErrorClass.PERMANENT and failure.retriable is False
    assert failure.partial_output == "x"


@pytest.mark.asyncio
async def test_on_repeat_stop_returns_the_same_failure_it_would_have_raised():
    """The two modes must differ only in delivery. A drift here (a different
    category on the quiet path) would make ``on_repeat`` a semantics switch
    instead of a delivery switch."""
    raised: Failure | None = None
    fn, _ = _scripted(["x", "x"])
    try:
        await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="escalate", max_attempts=9)
    except Stuck as exc:
        raised = exc.failure
    fn2, _ = _scripted(["x", "x"])
    returned = await attempt_until_stuck(fn2, fingerprint=lambda o: o, on_repeat="stop", max_attempts=9)
    assert raised == returned


@pytest.mark.asyncio
async def test_exhaustion_returns_a_failure_even_under_escalate():
    """``on_repeat`` governs the REPEAT only. Exhausting a backstop while every
    signature still differs is progress, and progress is not escalated."""
    fn, _ = _scripted(["a", "b", "c"])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="escalate", max_attempts=3)
    assert isinstance(out, Failure) and out.category is ErrorClass.UNKNOWN


# ── the interesting case: alternation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_alternating_signatures_are_a_cycle_not_progress():
    """A,B,A,B forever. A naive "compare with the previous signature" never
    fires here and burns the whole budget on two dead ends; comparing against
    EVERY signature seen stops at the third attempt, the moment A recurs."""
    fn, calls = _scripted(["A", "B", "A", "B", "A", "B"])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=6)
    assert isinstance(out, Failure)
    assert len(calls) == 3


# ── composition with run_with_resilience ─────────────────────────────────────


@pytest.mark.asyncio
async def test_composes_with_run_with_resilience_transient_inside_semantic_outside():
    """The division of labour, end to end: the inner loop absorbs a raised
    transient fault and the outer loop never sees it (so it does not burn a
    semantic attempt); the outer loop stops the run when the *completed*
    outcomes start repeating."""
    inner_calls: list[str] = []
    outcomes = iter(["nope", "nope"])

    async def flaky() -> str:
        inner_calls.append("call")
        if len(inner_calls) == 1:
            raise TimeoutError("upstream timed out")
        return next(outcomes)

    async def attempt() -> Any:
        return await run_with_resilience(flaky, max_attempts=3, sleep=_no_sleep)

    out = await attempt_until_stuck(attempt, fingerprint=lambda o: o, on_repeat="stop", max_attempts=5)
    assert isinstance(out, Failure)
    # 1 raised transient + 2 completed-but-identical outcomes.
    assert len(inner_calls) == 3


@pytest.mark.asyncio
async def test_a_raising_attempt_is_never_fingerprinted():
    """The seam between the two helpers. Exceptions are faults and belong to
    ``run_with_resilience``; this loop only reasons about attempts that
    COMPLETED. Fingerprinting a raise would make both helpers retry the same
    error and silently square the attempt count."""
    seen: list[Any] = []

    async def fn() -> Any:
        raise ValueError("boom")

    def fp(outcome: Any) -> Any:
        seen.append(outcome)
        return outcome

    with pytest.raises(ValueError, match="boom"):
        await attempt_until_stuck(fn, fingerprint=fp, on_repeat="stop", max_attempts=4)
    assert seen == []


# ── edges ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_success_on_the_first_attempt_never_loops():
    fn, calls = _scripted(["done"])
    out = await attempt_until_stuck(fn, fingerprint=lambda _o: None, on_repeat="escalate", max_attempts=4)
    assert out == "done" and len(calls) == 1


@pytest.mark.asyncio
async def test_max_attempts_one_is_a_legal_single_shot():
    fn, calls = _scripted(["a", "b"])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=1)
    assert isinstance(out, Failure) and out.category is ErrorClass.UNKNOWN
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [0, -1, -100])
@pytest.mark.asyncio
async def test_non_positive_max_attempts_is_a_programming_error(bad: int):
    """Zero attempts must not be reported as a ``Failure``: a caller would log
    an upstream failure for work that never ran."""
    fn, calls = _scripted(["a"])
    with pytest.raises(ValueError, match="max_attempts"):
        await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=bad)
    assert calls == []


@pytest.mark.asyncio
async def test_a_raising_fingerprint_propagates_rather_than_degrading():
    """Swallowing this would give every attempt a fresh "unknown" signature —
    i.e. a loop that can never detect a repeat and always runs to the backstop.
    A silently count-bounded loop is exactly what this helper exists to avoid."""
    fn, _ = _scripted(["a", "b"])

    def fp(_o: Any) -> Any:
        raise KeyError("failure_signature")

    with pytest.raises(KeyError):
        await attempt_until_stuck(fn, fingerprint=fp, on_repeat="stop", max_attempts=4)


@pytest.mark.asyncio
async def test_unhashable_signatures_are_compared_structurally():
    """An LLM-derived signature is naturally a dict/list. Requiring hashability
    would push every caller into hand-serialising it; equal-but-not-identical
    dicts must read as the same signature."""
    fn, calls = _scripted([{"missing": ["a", "b"]}, {"missing": ["a", "b"]}, {"missing": ["c"]}])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=5)
    assert isinstance(out, Failure) and len(calls) == 2


@pytest.mark.asyncio
async def test_structurally_distinct_unhashable_signatures_keep_going():
    """The other side of the same coin — structural comparison must not
    over-collapse distinct dicts into one 'repeat'."""
    fn, calls = _scripted([{"missing": ["a"]}, {"missing": ["b"]}, "done"])
    out = await attempt_until_stuck(
        fn, fingerprint=lambda o: None if o == "done" else o, on_repeat="stop", max_attempts=5
    )
    assert out == "done" and len(calls) == 3


@pytest.mark.asyncio
async def test_an_async_fingerprint_is_awaited():
    """The signature is often itself a model call ("did this answer the
    goal?"), so it has to be allowed to be async — same treatment
    ``LedgerPolicy`` gives its assessor."""
    fn, calls = _scripted(["a", "a"])

    async def fp(outcome: Any) -> Any:
        await asyncio.sleep(0)
        return outcome

    out = await attempt_until_stuck(fn, fingerprint=fp, on_repeat="stop", max_attempts=5)
    assert isinstance(out, Failure) and len(calls) == 2


@pytest.mark.asyncio
async def test_cancellation_is_checked_between_attempts():
    """Everything in the framework that loops checks the token at the loop top.
    Without it a cancelled run keeps paying for attempts until the backstop."""
    token = CancellationToken()
    ctx = make_test_ctx(cancel=token)
    calls: list[int] = []

    async def fn() -> Any:
        calls.append(1)
        token.cancel()  # cancelled DURING attempt 1
        return "a"

    with pytest.raises(Cancelled):
        await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=5, ctx=ctx)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_an_already_cancelled_run_burns_no_attempt():
    token = CancellationToken()
    token.cancel()
    ctx = make_test_ctx(cancel=token)
    fn, calls = _scripted(["a"])
    with pytest.raises(Cancelled):
        await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=5, ctx=ctx)
    assert calls == []


@pytest.mark.asyncio
async def test_the_stuck_failure_carries_the_last_outcome_as_partial_output():
    """The last attempt still produced an answer. Dropping it would throw away
    the only artifact of the whole loop."""
    fn, _ = _scripted([{"text": "half an answer", "sig": "q"}, {"text": "half an answer", "sig": "q"}])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o["sig"], on_repeat="stop", max_attempts=5)
    assert isinstance(out, Failure)
    assert out.partial_output == {"text": "half an answer", "sig": "q"}


# ── review additions: gaps the first suite left open ─────────────────────────


@pytest.mark.asyncio
async def test_a_hash_collision_is_not_a_repeat():
    """``stable_hash`` is a CACHE key basis, and its lossiest branch is fail-safe there and
    fail-dangerous here: an object it cannot describe structurally degrades to the bare type
    name, so every instance of that class collides. Trusting the hash alone reported
    PERMANENT/non-retriable "you are going in circles" after two attempts that produced two
    genuinely different answers — a healthy run killed on a claim the loop had not proven.
    The hash indexes; ``==`` decides."""

    class Sig:
        __slots__ = ("v",)

        def __init__(self, v: str) -> None:
            self.v = v

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Sig) and self.v == other.v

        def __hash__(self) -> int:  # pragma: no cover - not used, keeps Sig a normal value
            return hash(self.v)

    from agentkit.kernel.resilience import stable_hash

    # The precondition: these three DO collide under the shared hash basis.
    assert stable_hash(Sig("A")) == stable_hash(Sig("B")) == stable_hash(Sig("C"))

    fn, calls = _scripted([Sig("A"), Sig("B"), Sig("C"), "done"])
    out = await attempt_until_stuck(
        fn, fingerprint=lambda o: None if o == "done" else o, on_repeat="stop", max_attempts=6
    )
    assert out == "done"
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_colliding_but_unequal_json_shapes_are_not_a_repeat():
    """JSON has one sequence type, so a tuple and a list hash identically. They are not the
    same signature, and the loop must not say they are."""
    from agentkit.kernel.resilience import stable_hash

    assert stable_hash(("a",)) == stable_hash(["a"])

    fn, calls = _scripted([("a",), ["a"], "done"])
    out = await attempt_until_stuck(
        fn, fingerprint=lambda o: None if o == "done" else o, on_repeat="stop", max_attempts=5
    )
    assert out == "done"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_an_unconfirmable_signature_exhausts_rather_than_escalating():
    """The other side of the confirmation guard. A signature with no value equality (plain
    object, identity ``__eq__``) can no longer be *falsely* called stuck — which means its
    genuine repeats are not detected either, and the run falls through to the backstop.
    That is the intended direction: exhaustion is UNKNOWN/retriable and honest about running
    out of budget, where a false repeat is PERMANENT and terminal."""

    class Opaque:
        __slots__ = ("v",)

        def __init__(self, v: str) -> None:
            self.v = v

    fn, calls = _scripted([Opaque("same"), Opaque("same"), Opaque("same")])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="escalate", max_attempts=3)
    assert isinstance(out, Failure)
    assert out.category is ErrorClass.UNKNOWN and out.retriable is True
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_a_comparison_that_raises_counts_as_not_the_same():
    """Elementwise-comparing signatures (the ``numpy`` shape) return a non-bool from ``==``.
    Letting that escape would turn a stall check into a crash; treating it as "not the same"
    keeps the run bounded by the backstop."""

    class Elementwise:
        __slots__ = ()

        def __eq__(self, other: object) -> Any:
            raise ValueError("truth value of an array is ambiguous")

        def __hash__(self) -> int:  # pragma: no cover
            return 0

    fn, calls = _scripted([Elementwise(), Elementwise(), Elementwise()])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=3)
    assert isinstance(out, Failure) and out.category is ErrorClass.UNKNOWN
    assert len(calls) == 3


@pytest.mark.parametrize("bad", ["raise", "Escalate", "", "stop "])
@pytest.mark.asyncio
async def test_an_unrecognised_on_repeat_is_a_programming_error(bad: str):
    """It used to fall through the ``== "escalate"`` test and behave as ``"stop"``: the caller
    asked for a raise, got a PERMANENT ``Failure`` handed back as an ordinary return value, and
    code that only handles ``Stuck`` treats the stall as the answer. Silently resolving a typo
    to the NON-raising branch is the one direction a delivery switch must never fail in."""
    fn, calls = _scripted(["x", "x"])
    with pytest.raises(ValueError, match="on_repeat"):
        await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat=bad, max_attempts=5)  # type: ignore[arg-type]
    assert calls == [], "an argument error must be raised before any attempt is paid for"


@pytest.mark.asyncio
async def test_an_async_fingerprint_returning_none_is_success():
    """``None`` is the success signal and it must be read AFTER the await. Checking before
    would see a coroutine — never ``None`` — so an async fingerprint could never report done:
    the run would loop, repeat, and be declared PERMANENTLY stuck on a SUCCESSFUL outcome,
    leaving an un-awaited coroutine behind each time."""
    fn, calls = _scripted(["a", "done"])

    async def fp(outcome: Any) -> Any:
        await asyncio.sleep(0)
        return None if outcome == "done" else outcome

    out = await attempt_until_stuck(fn, fingerprint=fp, on_repeat="escalate", max_attempts=5)
    assert out == "done"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_the_failure_names_both_ends_of_the_cycle():
    """Why the history is a dict of attempt numbers and not a bare set. On an A,B,A stall
    "attempt 3 repeats attempt 1" tells an operator it is an oscillation rather than a wedge,
    and that distinction changes what they go and look at. Untested, the attempt number is
    free to rot into a placeholder."""
    fn, _ = _scripted(["A", "B", "A"])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=9)
    assert isinstance(out, Failure)
    assert "attempt 3 repeats attempt 1" in out.message
    assert "2 distinct signatures" in out.message


@pytest.mark.asyncio
async def test_the_exhaustion_failure_is_attributable_and_counts_signatures():
    fn, _ = _scripted(["a", "b", "c"])
    out = await attempt_until_stuck(fn, fingerprint=lambda o: o, on_repeat="stop", max_attempts=3)
    assert isinstance(out, Failure)
    assert out.source == "attempt_until_stuck"
    assert "3 distinct signatures" in out.message


@pytest.mark.asyncio
async def test_stuck_is_part_of_the_framework_taxonomy():
    """An ``AgentkitError`` and not a new root, so a framework-wide ``except AgentkitError``
    still catches it. A parallel taxonomy would slip straight through every existing handler."""
    from agentkit.kernel.errors import AgentkitError

    assert issubclass(Stuck, AgentkitError)
    fn, _ = _scripted(["x", "x"])
    with pytest.raises(AgentkitError) as excinfo:
        await attempt_until_stuck(fn, fingerprint=lambda o: o, max_attempts=5)
    assert isinstance(excinfo.value, Stuck)
    assert str(excinfo.value) == excinfo.value.failure.message


@pytest.mark.asyncio
async def test_escalate_is_the_default_delivery():
    """The default is the loud mode. If it ever silently became ``stop`` every caller that
    relies on the raise would start reading a ``Failure`` as an answer."""
    fn, _ = _scripted(["x", "x"])
    with pytest.raises(Stuck):
        await attempt_until_stuck(fn, fingerprint=lambda o: o, max_attempts=5)
