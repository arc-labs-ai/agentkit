"""Property-based tests for the money ledger — invariants over ALL inputs.

Example-based tests check the numbers somebody thought of. That is exactly how
this suite shipped a hole: the "every sane spelling" ceiling test used ``1.5``
and ``2``, both of which happen to be *exactly* representable in binary, so a
mutation replacing ``Decimal(str(v))`` with ``Decimal(v)`` survived. A test
using ``0.01`` would have killed it instantly — nobody picked ``0.01``.

Hypothesis picks the numbers nobody would, and shrinks a failure to the
smallest one that still breaks. For a ledger, that is the difference between
"the cases we imagined are exact" and "it is exact".

Each property below is a law the ledger must obey, not a scenario.
"""

from __future__ import annotations

import asyncio
from decimal import ROUND_HALF_UP, Decimal

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from agentkit.kernel.middleware import Call
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.runtime import Budget
from agentkit.runtime.context import RunContext
from agentkit.runtime.meter import MONEY_SCALE, to_money

# Amounts a provider could plausibly bill: non-negative, finite, and at or
# inside the ledger's scale. `allow_subnormal=False` keeps Hypothesis from
# exploring denormals, which are not money and only test the float parser.
COSTS = st.floats(
    min_value=0.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)

# Exact decimal amounts at the ledger's own scale — the shape a careful caller
# passes, and the shape `to_money` promises to preserve bit-for-bit.
EXACT = st.decimals(
    min_value=Decimal("0"),
    max_value=Decimal("1000"),
    places=MONEY_SCALE,
    allow_nan=False,
    allow_infinity=False,
)


def _call() -> Call:
    ctx = RunContext("prop", Scope(1, 1))
    return Call("chat", ChatRequest(messages=[Message("user", "hi")], model="m"), ctx)


def _charge_all(budget: Budget, amounts: list[Decimal]) -> None:
    async def go() -> None:
        for amount in amounts:
            await budget.charge(_call(), Usage(0, 0, float(amount)))

    asyncio.run(go())


# ── to_money: the coercion boundary ──────────────────────────────────────────


@given(EXACT)
def test_to_money_is_identity_on_values_already_at_scale(amount: Decimal) -> None:
    """An amount already at the ledger's scale must survive untouched.

    This is what makes ``to_money`` safe to call repeatedly — every charge
    path runs through it, and a coercion that nudged already-exact values
    would compound across a run.
    """
    assert to_money(amount) == amount


@given(EXACT)
def test_to_money_is_idempotent(amount: Decimal) -> None:
    """``to_money(to_money(x)) == to_money(x)`` for every x. A coercion that
    is not idempotent drifts a little more on each pass through the meter."""
    once = to_money(amount)
    assert to_money(once) == once


@given(COSTS)
def test_to_money_never_exceeds_the_declared_scale(cost: float) -> None:
    """The scale is a promise the rest of the system relies on: the checkpoint
    format, the invoice, and the equality in every reconciliation test."""
    assert -to_money(cost).as_tuple().exponent <= MONEY_SCALE


@given(COSTS)
def test_to_money_agrees_with_the_decimal_spelling_of_the_float(cost: float) -> None:
    """Pins the ``Decimal(str(x))`` choice as a property rather than an
    example. ``Decimal(0.01)`` is the binary expansion; ``Decimal(str(0.01))``
    is the number the caller wrote. They differ for most non-dyadic values,
    and Hypothesis will find one immediately if this regresses."""
    quantum = Decimal(1).scaleb(-MONEY_SCALE)
    assert to_money(cost) == Decimal(str(cost)).quantize(quantum, rounding=ROUND_HALF_UP)


@given(EXACT)
def test_to_money_strict_accepts_anything_at_scale(amount: Decimal) -> None:
    """Strict mode must never refuse a value that is already representable —
    otherwise a legitimate ceiling becomes unconstructable."""
    assert to_money(amount, strict=True) == amount


# ── the ledger: sums must be exact and order-independent ─────────────────────


@given(st.lists(EXACT, min_size=1, max_size=40))
@settings(max_examples=60, deadline=None)
def test_the_ledger_equals_the_exact_sum_of_its_charges(amounts: list[Decimal]) -> None:
    """The defining property. A float ledger fails this for most inputs; the
    100x$0.01 example test is just one point on it."""
    budget = Budget()
    _charge_all(budget, amounts)
    assert budget.spent() == sum(amounts, Decimal(0))


@given(st.lists(EXACT, min_size=2, max_size=20))
@settings(max_examples=60, deadline=None)
def test_charge_order_does_not_change_the_total(amounts: list[Decimal]) -> None:
    """Addition is commutative for money and is NOT for floats at the margin.
    Concurrent agents charge a shared Budget in nondeterministic order, so an
    order-dependent total would make a multi-agent run's cost irreproducible —
    and irreproducible costs cannot be reconciled or tested."""
    forward, backward = Budget(), Budget()
    _charge_all(forward, amounts)
    _charge_all(backward, list(reversed(amounts)))
    assert forward.spent() == backward.spent()


@given(st.lists(EXACT, min_size=1, max_size=20))
@settings(max_examples=60, deadline=None)
def test_the_float_mirror_never_drifts_from_the_ledger(amounts: list[Decimal]) -> None:
    """``spent_usd`` is documented as a float RENDERING of ``spent()``. If the
    two can disagree, every doc and dashboard reading the mirror is lying."""
    budget = Budget()
    _charge_all(budget, amounts)
    assert budget.spent_usd == float(budget.spent())


@given(st.lists(EXACT, min_size=1, max_size=20))
@settings(max_examples=60, deadline=None)
def test_spending_is_monotonic(amounts: list[Decimal]) -> None:
    """A charge never reduces the total. Sounds trivial; it is exactly what a
    sign error or a stray ``=`` instead of ``+=`` breaks, and it is the
    invariant every ceiling check depends on."""
    budget = Budget()
    previous = Decimal(0)

    async def go() -> None:
        nonlocal previous
        for amount in amounts:
            await budget.charge(_call(), Usage(0, 0, float(amount)))
            assert budget.spent() >= previous
            previous = budget.spent()

    asyncio.run(go())


@given(EXACT, st.lists(EXACT, min_size=1, max_size=15))
@settings(max_examples=60, deadline=None)
def test_remaining_is_always_ceiling_minus_spent_and_never_negative(
    ceiling: Decimal, amounts: list[Decimal]
) -> None:
    """Two claims at once: the arithmetic is exact, and headroom is clamped at
    zero so a caller can divide by it or render it without special-casing."""
    # ``on_exceeded="stop"`` because the property is about the ARITHMETIC, and
    # under the default raise-mode Hypothesis immediately (and correctly)
    # finds ceiling=0 + charge=1 and gets a MeterExceeded instead of a number.
    budget = Budget(max_cost_usd=ceiling, on_exceeded="stop")
    _charge_all(budget, amounts)
    remaining = budget.remaining()
    assert remaining is not None
    assert remaining >= 0
    assert remaining == max(Decimal(0), ceiling - budget.spent())


@given(EXACT, st.lists(EXACT, min_size=1, max_size=15))
@settings(max_examples=60, deadline=None)
def test_exhausted_agrees_with_the_arithmetic(
    ceiling: Decimal, amounts: list[Decimal]
) -> None:
    """``exhausted()`` is what the cognitions branch on to stop a run, so it
    must never disagree with the numbers a human would check by hand."""
    budget = Budget(max_cost_usd=ceiling, on_exceeded="stop")
    _charge_all(budget, amounts)
    assert budget.exhausted() == (budget.spent() > ceiling)


# ── invoicing ────────────────────────────────────────────────────────────────


@given(EXACT)
def test_spent_cents_is_within_half_a_cent_of_the_exact_total(amount: Decimal) -> None:
    """Rounding must be to the NEAREST cent. Truncation also satisfies "within
    a cent", so the bound is deliberately half a cent — that is the property
    ``ROUND_DOWN`` violates, and it is the one that costs real money."""
    budget = Budget(spent_usd=float(amount))
    assert abs(budget.spent_cents() - budget.spent()) <= Decimal("0.005")


@given(EXACT)
def test_spent_cents_has_exactly_two_decimal_places(amount: Decimal) -> None:
    budget = Budget(spent_usd=float(amount))
    assert -budget.spent_cents().as_tuple().exponent == 2


# ── Usage: the accumulator is a monoid ───────────────────────────────────────

TOKENS = st.integers(min_value=0, max_value=10**7)
USAGES = st.builds(
    Usage,
    input_tokens=TOKENS,
    output_tokens=TOKENS,
    cost_usd=COSTS,
    cache_read_tokens=TOKENS,
    cache_write_tokens=TOKENS,
)


def _tokens(u: Usage) -> tuple[int, int, int, int]:
    return (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens)


@given(USAGES, USAGES, USAGES)
def test_usage_token_addition_is_associative(a: Usage, b: Usage, c: Usage) -> None:
    """``Budget`` folds an arbitrary interleaving of child usages, so
    associativity is what makes a tree's token total well-defined.

    Scoped to the TOKEN fields. ``cost_usd`` is deliberately excluded: see
    ``test_usage_cost_is_quantized_not_exact`` — it is a float that
    re-rounds on every addition, which makes it neither associative nor
    identity-preserving. That is a documented property of the mirror, not of
    the ledger.
    """
    assert _tokens((a + b) + c) == _tokens(a + (b + c))


@given(USAGES, USAGES)
def test_usage_addition_is_commutative_in_token_counts(a: Usage, b: Usage) -> None:
    """Children complete in nondeterministic order."""
    left, right = a + b, b + a
    assert left.input_tokens == right.input_tokens
    assert left.output_tokens == right.output_tokens
    assert left.cache_read_tokens == right.cache_read_tokens
    assert left.cache_write_tokens == right.cache_write_tokens


@given(USAGES)
def test_the_empty_usage_is_a_token_identity(a: Usage) -> None:
    """``Usage()`` is the seed every accumulator starts from; if it were not
    an identity for the counts, every run's totals would be off by whatever
    it contributed."""
    assert _tokens(a + Usage()) == _tokens(a)
    assert _tokens(Usage() + a) == _tokens(a)


@given(USAGES)
def test_usage_cost_is_quantized_not_exact(a: Usage) -> None:
    """Pins a real wart so nobody mistakes ``usage.cost_usd`` for the ledger.

    ``Usage.__add__`` does ``round(x + y, 6)``, so adding the EMPTY usage can
    still change the value:

        >>> Usage(cost_usd=1.4296875) + Usage()
        Usage(..., cost_usd=1.429688, ...)

    Two consequences a reader must not be surprised by:

    * ``cost_usd`` is neither associative nor identity-preserving under
      addition — it is a running approximation, quantized on every step.
    * ``round()`` is banker's rounding (HALF_EVEN) while the ``Budget``
      ledger quantizes HALF_UP, so on an exact tie the two disagree by one
      unit in the last place: ``round(5e-7, 6) == 0.0`` but the ledger
      records ``0.000001``.

    Neither matters in practice, because ``pricing.cost()`` already emits
    6-decimal values and ``Budget.spent()`` — not this field — is the
    authority a run reconciles against. It matters a great deal if someone
    starts summing ``usage.cost_usd`` and expecting cents to balance, which
    is exactly why it is written down here rather than left to be
    rediscovered.
    """
    combined = a + Usage()
    assert abs(combined.cost_usd - a.cost_usd) <= 1e-6
    # ...and the quantization is to the ledger's scale, not something else.
    assert combined.cost_usd == round(a.cost_usd, 6)


@given(USAGES)
def test_total_tokens_is_input_plus_output(a: Usage) -> None:
    assume(a.input_tokens + a.output_tokens < 2**62)
    assert a.total_tokens == a.input_tokens + a.output_tokens
