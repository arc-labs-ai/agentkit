"""The per-actor spend envelope, charged on the path that bypasses the Invoker.

WHY THIS FILE EXISTS
--------------------
A CLI cognition delegates the whole loop to a subprocess, so it never passes
through the ``Invoker`` and the ``meter()`` middleware never sees its usage.
``_charge_meters`` is the hand-written replacement for that middleware, and the
per-actor envelope is the SECOND ledger it has to move: the shared meters are a
run budget and a tenant quota, while the envelope is one actor's own ceiling in
tokens, dollars AND steps.

``_common.py``'s own docstring records that this is a hole ``ActorBudget``
has already shipped once — the shared meters were charged and the per-actor one
was not, so an actor whose entire purpose was a spend limit ran without one.
The defect is silent in both directions: nothing raises, the run returns a
perfectly ordinary answer, and the envelope simply stays at zero. An envelope
that never advances never becomes exhausted, so no later consultation of it can
stop anything — which is why these tests assert on the envelope's STATE after
the charge and not merely that a method was called.

The helper is driven directly rather than through either cognition. It is
shared code with exactly one adapter-specific input (``enabled``, which is the
caller's ``meter_spend``), and the claude driver's own budget suite already
covers the wiring; what is untested is the helper's contract itself, which the
codex cognition depends on identically.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from agentkit.core.contracts.types import ProviderUsage
from agentkit.models.providers.cli._common import _charge_meters


class _Envelope:
    """A per-actor ceiling in all three units the envelope caps.

    Deliberately NOT a bare recorder. A recorder proves a call happened; it
    cannot distinguish an envelope that advanced from one that was handed zeros,
    and "exhausting it stops nothing" is a statement about the ceiling, not
    about the call.
    """

    def __init__(
        self, *, max_tokens: int = 10_000, max_cost_usd: str = "1.00", max_steps: int = 10
    ) -> None:
        self.max_tokens = max_tokens
        self.max_cost_usd = Decimal(max_cost_usd)
        self.max_steps = max_steps
        self.tokens = 0
        # ``Decimal`` and not ``float``: this is money, and a float ledger
        # drifts a spend cap by fractions of a cent per charge until the
        # ceiling it is compared against is no longer the one that was set.
        self.cost_usd = Decimal(0)
        self.steps = 0
        self.charges: list[tuple[int, Any, int]] = []

    def charge(self, *, tokens: int, cost_usd: Any, steps: int) -> None:
        self.charges.append((tokens, cost_usd, steps))
        self.tokens += tokens
        if cost_usd is not None:
            self.cost_usd += Decimal(str(cost_usd))
        self.steps += steps

    @property
    def exhausted(self) -> bool:
        return (
            self.tokens >= self.max_tokens
            or self.cost_usd >= self.max_cost_usd
            or self.steps >= self.max_steps
        )


class _RefusingEnvelope(_Envelope):
    """An envelope whose ceiling this very charge crosses, and which says so by
    raising — the commonest refusal, and the one that must not cost the caller
    an answer that has already been paid for."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def charge(self, *, tokens: int, cost_usd: Any, steps: int) -> None:
        self.attempts += 1
        raise RuntimeError("actor envelope ceiling crossed by this call")


class _Ctx:
    """The two attributes ``_charge_meters`` reads off a context, and nothing
    else: it reaches for both with ``getattr``, so a test double that defines
    more would hide a lookup that had started using something absent in
    production."""

    def __init__(self, *, actor_budget: Any = None, all_meters: tuple[Any, ...] = ()) -> None:
        self.actor_budget = actor_budget
        self.all_meters = all_meters


class _Meter:
    def __init__(self) -> None:
        self.charges: list[Any] = []

    async def charge(self, call: Any, usage: ProviderUsage) -> None:
        self.charges.append((call, usage))


def _usage(
    *, input_tokens: int = 100, output_tokens: int = 50, cost: str | None = "0.25"
) -> ProviderUsage:
    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=Decimal(cost) if cost is not None else None,
    )


@pytest.mark.asyncio
async def test_the_envelope_advances_so_that_exhausting_it_can_stop_the_next_step() -> None:
    """The whole point of the envelope is that it RUNS OUT. An envelope that is
    never charged is a ceiling nothing ever approaches: every consultation of it
    reports headroom, so an actor given a $1 limit runs until something else
    stops it. The assertion is therefore on the envelope's own verdict after
    enough spend to cross it — a charge that happened but moved nothing would
    satisfy a call-count assertion and fail this one."""
    envelope = _Envelope(max_steps=3)
    ctx = _Ctx(actor_budget=envelope)

    for _ in range(3):
        assert await _charge_meters(ctx, _usage(), enabled=True) is None

    assert envelope.steps == 3
    assert envelope.exhausted, "three charged steps against a three-step ceiling is exhausted"


@pytest.mark.asyncio
async def test_the_envelope_is_charged_in_tokens_dollars_and_steps_because_it_caps_all_three() -> None:
    """The envelope has three independent ceilings, and a charge that moved only
    one of them would leave the other two inert — an actor capped at 10 steps
    could burn its whole token allowance inside one of them, or vice versa. The
    token figure is the TOTAL (prompt plus completion), because the ceiling is
    on what the actor consumed and the split between the two is not the
    envelope's business."""
    envelope = _Envelope()
    ctx = _Ctx(actor_budget=envelope)

    await _charge_meters(ctx, _usage(input_tokens=100, output_tokens=50, cost="0.25"), enabled=True)

    assert envelope.charges == [(150, Decimal("0.25"), 1)]
    assert envelope.cost_usd == Decimal("0.25")
    assert isinstance(envelope.cost_usd, Decimal), (
        "a spend cap compared against a float silently drifts; the charge has "
        "to arrive as exact money"
    )


@pytest.mark.asyncio
async def test_a_run_whose_cost_is_unknown_still_advances_the_token_and_step_ceilings() -> None:
    """``cost_usd`` is ``None`` whenever the CLI never reached its terminal
    payload — a truncated stream, a cancel, a crash. ``None`` means unknown and
    must not be read as zero, but it also must not cost the envelope the two
    ceilings that ARE known: an actor looping on crashed runs would otherwise
    advance nothing at all and never hit its step limit."""
    envelope = _Envelope(max_steps=2)
    ctx = _Ctx(actor_budget=envelope)

    await _charge_meters(ctx, _usage(cost=None), enabled=True)
    await _charge_meters(ctx, _usage(cost=None), enabled=True)

    assert [c[1] for c in envelope.charges] == [None, None], "unknown stayed unknown"
    assert envelope.tokens == 300
    assert envelope.exhausted, "the step ceiling still closed on an unpriced run"


@pytest.mark.asyncio
async def test_an_envelope_that_refuses_the_charge_never_costs_the_caller_the_answer() -> None:
    """The commonest refusal is a ceiling crossed by THIS call: the money is
    already spent and the subprocess has already produced its answer. Letting
    the refusal propagate would throw away a paid-for result out of the middle
    of a generator, which is the one thing the terminal-event guarantee forbids.
    The envelope is also charged AFTER the shared meters for the same reason —
    a raising envelope must not be able to skip the run budget."""
    envelope, meter = _RefusingEnvelope(), _Meter()
    ctx = _Ctx(actor_budget=envelope, all_meters=(meter,))

    note = await _charge_meters(ctx, _usage(), enabled=True)

    assert envelope.attempts == 1
    assert len(meter.charges) == 1, "the shared ledger moved even though the envelope refused"
    assert note is None, (
        "the envelope's own refusal is suppressed rather than reported: a "
        "ceiling the actor crossed is not a meter malfunction"
    )


@pytest.mark.asyncio
async def test_a_context_that_carries_no_envelope_is_not_an_error() -> None:
    """Most runs have no per-actor envelope at all — it is an opt-in second
    ledger. The lookup is a ``getattr`` with a default for that reason, and a
    missing envelope has to be a no-op rather than an ``AttributeError`` that
    would turn a working run into ``spawn_failed``."""
    meter = _Meter()

    assert await _charge_meters(_Ctx(all_meters=(meter,)), _usage(), enabled=True) is None
    assert await _charge_meters(object(), _usage(), enabled=True) is None
    assert len(meter.charges) == 1


@pytest.mark.asyncio
async def test_the_opt_out_takes_the_envelope_off_the_books_as_well() -> None:
    """``meter_spend=False`` is the documented way to take a run off the books,
    and a flag that silenced the shared meters while still moving one actor's
    envelope would leave that actor paying for a run its owner asked to be
    invisible to the accounting."""
    envelope, meter = _Envelope(), _Meter()
    ctx = _Ctx(actor_budget=envelope, all_meters=(meter,))

    assert await _charge_meters(ctx, _usage(), enabled=False) is None

    assert envelope.charges == []
    assert meter.charges == []
    assert not envelope.exhausted
