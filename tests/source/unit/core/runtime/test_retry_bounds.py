"""The bounds on `RetryPolicy`: how long a run may be stalled, and by whom.

WHY THIS FILE EXISTS
--------------------
`CallRuntime`'s retry loop is the OTHER half of the retry contract. The
chokepoint in `core/execution/resilience.py` decides whether an exception is
worth another try; this loop decides it for a `Failure`, and it is the one that
owns the two bounds a caller actually plans around:

* `cap_s` — no single wait may exceed it;
* `max_elapsed_s` — the retries together may not outrun it.

Both are ceilings on WALL-CLOCK TIME inside a call the caller is awaiting, and
a ceiling that is not enforced is not a ceiling: a provider is free to send
`Retry-After: 3600`, and honouring that inside one `ask()` hangs the caller for
an hour. The loop therefore refuses to retry rather than overshoot either
bound, and that refusal is the thing most easily lost in a refactor — it looks
like a missing feature, not like a bug.

The third property here is that `Failure.retryable` is READ, never re-derived
from the classification. An adapter that knows this particular 429 is
permanent (a reached spend cap, a CLI process that already consumed its one
chance) says so by clearing the flag, and a loop that decides for itself from
the class ignores it and burns the budget anyway.

Nothing here sleeps: `_sleep_turn` is replaced with a recorder, which is also
what makes the waits assertable at all — they are otherwise invisible except as
elapsed time.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.core.runtime import calls as calls_module
from agentkit.core.runtime.calls import CallRuntime, RetryPolicy
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.constants import EventKind, FailureClass
from agentkit.models.contracts.types import Binding, Failure, Turn

PRICES = PriceTable(
    "test",
    "2026-10-13",
    {"test": {"input_uncached": "1", "input_cache_read": "0.1", "output": "5"}},
)


class _Adapter:
    """Replays scripted turns and records how many times it was dispatched."""

    def __init__(self, *turns: Turn, repeat_last: bool = False) -> None:
        self.turns = list(turns)
        self.repeat_last = repeat_last
        self.calls = 0

    async def turn(self, binding: Any, prompt: Any, **kwargs: Any) -> Turn:
        self.calls += 1
        if len(self.turns) > 1 or not self.repeat_last:
            return self.turns.pop(0)
        return self.turns[0]


class _SleepSpy:
    """Stands in for `_sleep_turn`, recording the wait instead of taking it."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> Turn:
        self.waits.append(seconds)
        return Turn()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> _SleepSpy:
    spy = _SleepSpy()
    monkeypatch.setattr(calls_module, "_sleep_turn", spy)
    return spy


async def _ask(runtime: CallRuntime, call_id: str) -> tuple[Any, list[Any]]:
    events: list[Any] = []
    async with runtime:
        answer = await runtime.ask(
            Binding("openai_api", "test"), "hello", events=events.append, call_id=call_id
        )
    return answer, events


def _retries(events: list[Any]) -> list[Any]:
    return [event for event in events if event.kind == EventKind.RETRY]


@pytest.mark.asyncio
async def test_max_attempts_is_the_exact_number_of_dispatches(no_sleep: _SleepSpy) -> None:
    """`max_attempts` counts dispatches, including the first one.

    A dependency that is down answers every attempt the same way, so this
    number is the only thing standing between one failing call and an unbounded
    retry loop against a provider that charges per request. One attempt too
    many doubles the load a struggling dependency sees from every caller at
    once; one too few silently shortens a policy someone sized deliberately.
    Both are invisible on the happy path, so the count is pinned against the
    adapter itself rather than inferred from the reported `attempts`.
    """
    adapter = _Adapter(Turn(failure=Failure(FailureClass.OVERLOADED, True)), repeat_last=True)
    runtime = CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=4, base_s=0, cap_s=0, max_elapsed_s=120),
    )

    answer, events = await _ask(runtime, "budget")

    assert adapter.calls == 4, "the attempt budget was not spent exactly"
    assert answer.attempts == 4
    assert len(_retries(events)) == 3, "one RETRY event per retry, not per attempt"
    assert answer.outcome == "failed"
    assert answer.failure is not None and answer.failure.classification == FailureClass.OVERLOADED


@pytest.mark.asyncio
async def test_no_single_backoff_exceeds_the_cap(no_sleep: _SleepSpy) -> None:
    """`cap_s` is the ceiling on ONE wait, and the exponential schedule reaches
    it fast — the base doubles per retry, so by the fourth attempt an
    unclamped 1000-second base is 8000 seconds inside a call the caller is
    awaiting. The cap is what makes the backoff's growth safe to configure at
    all; without it, the only way to bound a run is to lower `base_s` and give
    up the storm protection the exponential schedule provides.

    The base here is absurd on purpose: every emitted wait must still land
    under the cap, and the RETRY events must report the same number the loop
    actually waited, or an operator reading the event stream is being told a
    different story from the one the caller experienced.
    """
    adapter = _Adapter(Turn(failure=Failure(FailureClass.RATE_LIMITED, True)), repeat_last=True)
    runtime = CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=4, base_s=1000.0, cap_s=0.05, max_elapsed_s=120),
    )

    answer, events = await _ask(runtime, "cap")

    assert adapter.calls == 4, "a wait above the cap must shorten the wait, not the attempts"
    assert no_sleep.waits, "the retries did not back off at all"
    assert all(0.0 <= wait <= 0.05 for wait in no_sleep.waits), f"a wait escaped the cap: {no_sleep.waits}"
    assert [event.fields["wait_seconds"] for event in _retries(events)] == no_sleep.waits
    assert answer.attempts == 4


@pytest.mark.asyncio
async def test_a_retry_after_longer_than_the_cap_ends_the_run_instead_of_waiting(
    no_sleep: _SleepSpy,
) -> None:
    """A server-sent `Retry-After` is a hint, not a licence to block.

    It is the one wait the caller does not control: the provider picks the
    number, and `Retry-After: 3600` is a documented, ordinary response to a
    reached rate-limit window. Honouring it inside a call someone is awaiting
    converts a fast failure into an hour-long hang — and the hang is worse than
    the failure, because the caller's own deadline can no longer fire anything
    useful.

    So when the hint exceeds `cap_s` the loop stops retrying and returns the
    failure. The caller gets the real classification, with `retry_after_s`
    attached, and decides for itself whether to come back later.
    """
    adapter = _Adapter(
        Turn(failure=Failure(FailureClass.RATE_LIMITED, True, 3600.0)),
        Turn(("never reached",)),
    )
    runtime = CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=3, base_s=0, cap_s=1.0, max_elapsed_s=120),
    )

    answer, events = await _ask(runtime, "retry-after-over-cap")

    assert no_sleep.waits == [], "the caller was made to wait longer than the cap allows"
    assert adapter.calls == 1 and answer.attempts == 1
    assert _retries(events) == [], "a retry was announced and then not performed"
    assert answer.failure is not None
    assert answer.failure.classification == FailureClass.RATE_LIMITED
    assert answer.failure.retry_after_s == 3600.0, "the hint must still reach the caller"


@pytest.mark.asyncio
async def test_a_wait_that_would_outrun_max_elapsed_is_not_taken(no_sleep: _SleepSpy) -> None:
    """`max_elapsed_s` bounds the retries TOGETHER, and it is checked against
    the wait that is about to be taken — not only against time already spent.

    Checking it only before the sleep is the subtle version of the bug: every
    individual wait is under the cap and elapsed time is still inside the
    budget, so each retry looks legal, and the LAST one sails past the deadline
    because nobody asked whether it would fit. The budget exists so a caller
    can promise its own user a bound; a retry that overshoots it has already
    broken that promise by the time it finishes.
    """
    adapter = _Adapter(
        Turn(failure=Failure(FailureClass.TRANSIENT_NETWORK, True, 5.0)),
        Turn(("never reached",)),
    )
    runtime = CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=3, base_s=0, cap_s=30.0, max_elapsed_s=1.0),
    )

    answer, events = await _ask(runtime, "elapsed")

    assert no_sleep.waits == [], "a wait longer than the whole elapsed budget was taken"
    assert adapter.calls == 1 and answer.attempts == 1
    assert _retries(events) == []
    assert answer.failure is not None and answer.failure.classification == FailureClass.TRANSIENT_NETWORK


@pytest.mark.asyncio
async def test_a_retry_after_inside_the_cap_is_waited_in_full(no_sleep: _SleepSpy) -> None:
    """The control on the two refusals above: a hint that FITS must be honoured
    exactly, and it must win over the computed backoff.

    The provider is the only party that knows when its rate-limit window
    reopens, so retrying earlier than it asked is a request guaranteed to fail
    — and on a provider that counts rejected requests, one that makes things
    worse. Taking the maximum of the hint and the jittered backoff is what
    makes "bounded" compatible with "respectful"; dropping the hint leaves a
    sub-millisecond backoff hammering a closed window, which is exactly how a
    bounded retry turns into a storm.
    """
    adapter = _Adapter(
        Turn(failure=Failure(FailureClass.RATE_LIMITED, True, 5.0)),
        Turn(("done",)),
    )
    runtime = CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=3, base_s=0, cap_s=30.0, max_elapsed_s=120),
    )

    answer, events = await _ask(runtime, "retry-after-honoured")

    assert no_sleep.waits == [5.0], "the provider's own wait was ignored"
    assert adapter.calls == 2 and answer.attempts == 2
    assert answer.text == "done"
    retry = _retries(events)
    assert [event.fields["source"] for event in retry] == ["retry-after"], (
        "the event stream must say the wait came from the provider, not from the backoff"
    )
    assert retry[0].fields["wait_seconds"] == 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "classification",
    [
        # Classes whose default IS retryable — the point of the test. A loop
        # that re-derives retryability from the class ignores the flag here.
        FailureClass.RATE_LIMITED,
        FailureClass.OVERLOADED,
        FailureClass.TRANSIENT_NETWORK,
        # ...and the classes that are never retryable, as the control.
        FailureClass.QUOTA_EXHAUSTED,
        FailureClass.INVALID_REQUEST,
        FailureClass.CONTEXT_EXCEEDED,
        FailureClass.REFUSED,
        FailureClass.UNKNOWN,
    ],
)
async def test_a_failure_marked_not_retryable_is_never_retried(
    classification: FailureClass, no_sleep: _SleepSpy
) -> None:
    """`Failure.retryable` is the decision; the classification is only a label.

    The adapter is the layer that knows the particulars — a 429 that is really
    a reached spend cap, a CLI process that already consumed its single
    allowance, an overload reported on a request that is not idempotent. It says
    so by clearing the flag, and the loop must take it at face value. Deciding
    for itself from the class is what burns the budget on an account that can
    never succeed, which is the failure mode the quota classification exists to
    prevent in the first place.

    Parametrised deliberately over the classes whose DEFAULT is retryable: on
    those, re-deriving the verdict and reading the flag differ, and only this
    direction catches it.
    """
    adapter = _Adapter(
        Turn(failure=Failure(classification, False)),
        Turn(("never reached",)),
    )
    runtime = CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=3, base_s=0, cap_s=0, max_elapsed_s=120),
    )

    answer, events = await _ask(runtime, f"not-retryable-{classification}")

    assert adapter.calls == 1, "a failure the adapter declared final was retried"
    assert answer.attempts == 1
    assert no_sleep.waits == []
    assert _retries(events) == []
    assert answer.failure is not None and answer.failure.classification == classification
