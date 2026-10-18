"""Two bounds a run is only as good as: the fan-out pool, and the money.

WHY THIS FILE EXISTS
--------------------
Both halves of this file guard a number whose wrong value produces NO error.

* ``ToolSet.max_concurrency`` becomes an ``asyncio.Semaphore``. A pool of zero
  permits is a perfectly valid semaphore that admits nobody, so a ``ToolSet``
  built with ``0`` constructs, advertises its schemas, answers
  ``effective_timeout_s`` and then wedges the first fan-out forever — not
  slowly, not with a diagnostic, just never. The construction-time refusal is
  the only place this is cheap to catch, because by the time it matters the
  symptom is a hung run with no stack to read.

* A spend ceiling is ``Decimal`` on purpose. Money handled as a float drifts a
  cap by fractions of a cent per charge until the ceiling being compared
  against is no longer the one that was configured. The trap is that the drift
  is invisible to the obvious assertion: ``Decimal("1.00") == 1.0`` is ``True``
  in Python, so a test comparing a ledger against a float literal passes while
  enforcing nothing about the type. Every money assertion here therefore checks
  the TYPE as well as the value, and compares exact decimals against exact
  decimals.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from agentkit.agents.definitions.agent import DelegationPolicy
from agentkit.models.contracts.types import CallLimits
from agentkit.tools.execution.policy import OverflowPolicy, ToolSet
from agentkit.tools.registry.registry import ToolRegistry


@pytest.mark.parametrize(
    ("max_concurrency", "why"),
    [
        (0, "a pool with no permits admits nobody, ever"),
        (-1, "asyncio.Semaphore refuses a negative value, but only when it is built"),
        (-100, "no magnitude of negative makes it a pool"),
    ],
)
def test_a_tool_concurrency_that_could_never_admit_a_call_is_refused_at_construction(
    max_concurrency: int, why: str
) -> None:
    """A fan-out cap below one is not a tight limit, it is a closed door.

    ``max_concurrency`` exists only to size the semaphore the tool loop fans
    out through, and zero permits is a semaphore that is permanently locked:
    the first ``async with sem`` parks and nothing in the system will ever
    release it, because no holder exists. The run does not fail — it stops,
    with no exception, no log line and no stack naming the configuration that
    did it, until an outer deadline fires somewhere unrelated.

    Refused HERE because ``ToolSet`` is where the number is a value a caller
    typed. One step later it is a semaphore, and a semaphore cannot tell you
    who sized it.
    """
    with pytest.raises(ValueError, match="concurrency"):
        ToolSet(registry=ToolRegistry(), max_concurrency=max_concurrency)


def test_one_permit_is_the_smallest_pool_that_can_still_run_something() -> None:
    """The boundary, asserted so the refusal above cannot be widened into
    "positive means at least two". Serialised execution is a legitimate
    configuration — a registry of tools that must not overlap — and a bound
    that rejected it would force callers into a pool size they did not want."""
    tools = ToolSet(registry=ToolRegistry(), max_concurrency=1)

    assert tools.max_concurrency == 1
    assert not asyncio.Semaphore(tools.max_concurrency).locked(), (
        "the surviving value has to be usable as a pool that admits its first caller"
    )


def test_the_refusal_is_what_a_zero_permit_pool_would_otherwise_do() -> None:
    """The evidence behind the bound, kept next to it so the reason survives
    the next person who reads ``if self.max_concurrency < 1`` and wonders
    whether zero is merely unusual. A zero-permit semaphore is already locked
    before anybody asks, and an acquire against it never completes — which is
    why the only safe moment to object is construction."""
    closed = asyncio.Semaphore(0)

    assert closed.locked(), "nobody holds this; it was born with nothing to give"

    async def never_gets_in() -> None:
        async with closed:
            raise AssertionError("a zero-permit pool admitted a caller")

    with pytest.raises(TimeoutError):
        asyncio.run(asyncio.wait_for(never_gets_in(), timeout=0.05))


def test_an_overflow_bound_below_the_floor_is_refused_for_the_same_reason() -> None:
    """The sibling bound on the same policy object, and the same failure shape
    one layer along: a ``max_bytes`` smaller than the sentence that explains the
    spill would overflow every result into a temp file and hand the model a
    message it cannot act on. Asserted here because both bounds live in
    ``__post_init__`` and a refactor that loses one tends to lose both."""
    with pytest.raises(ValueError, match="256 bytes"):
        OverflowPolicy(max_bytes=255)

    assert OverflowPolicy(max_bytes=256).max_bytes == 256


def test_a_delegated_budget_split_is_exact_money_and_not_a_float_approximation() -> None:
    """A parent splits its ceiling with its children, and the two halves have
    to add back up to exactly what it was given. In binary floating point they
    do not: the share is a fraction of a cent out, every delegation compounds
    it, and the error lands in whichever direction nobody checked — a child
    authorised slightly more than the parent reserved, or a parent left with a
    cap it can never spend down to.

    Asserted on the TYPE as well as the value because the value assertion alone
    proves nothing: ``Decimal("0.075") == 0.075`` is ``True``, so a float share
    would satisfy an equality check and still be a float.
    """
    ceiling = Decimal("0.10")
    share = DelegationPolicy().budget_share

    assert isinstance(share, Decimal), "a float share makes every split downstream inexact"

    child = ceiling * share
    parent = ceiling - child

    assert isinstance(child, Decimal) and isinstance(parent, Decimal)
    assert child == Decimal("0.025")
    assert parent == Decimal("0.075")
    assert parent + child == ceiling, (
        "the split has to be lossless: what the parent keeps plus what it "
        "reserves for children IS the ceiling it was given"
    )


def test_a_spend_ceiling_that_is_not_a_real_number_is_refused_before_a_run_starts() -> None:
    """``CallLimits`` is the only place a cap is checked before it becomes the
    thing a run is compared against. A non-finite ceiling is the dangerous
    direction: ``Decimal("Infinity")`` compares greater than every spend, so a
    typo in configuration reads as "no limit" rather than as an error, and the
    run it was meant to bound has none. Zero and negatives are refused for the
    mirror reason — they can never be satisfied, so the run is refused before
    it does any work rather than after."""
    for bad in (Decimal("Infinity"), Decimal("-Infinity"), Decimal("NaN"), Decimal(0), Decimal(-1)):
        with pytest.raises(ValueError, match="finite and positive"):
            CallLimits(max_cost_usd=bad)

    assert CallLimits(max_cost_usd=Decimal("0.01")).max_cost_usd == Decimal("0.01")
