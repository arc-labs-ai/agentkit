"""ActorBudget — per-actor four-axis envelope with reserve/charge/settle.

Coverage:
  1. Construction stores maxes + zeroes usage and reservation
  2. ``charge`` accumulates ``used_*`` per axis
  3. ``remaining_*`` reflects max - used - reserved
  4. ``exhausted`` trips when any axis hits zero
  5. ``can_spawn_child`` precheck doesn't mutate
  6. ``reserve_for_child`` carves a slice (available drops)
  7. ``reserve_for_child`` raises ``BudgetExhausted`` if oversized
  8. ``settle_child`` releases reservation + caps charged-against-parent
  9. ``tighten`` lowers caps; clamps to ``used + reserved`` floor
  10. ``BudgetExhausted`` carries the constraining axis
  11. wall-clock auto-tracks via injected clock
"""

from __future__ import annotations

import pytest

from agentkit.agents.control.budget import ActorBudget, BudgetExhausted


def _budget(
    *,
    tokens: int = 1000,
    cost: float = 1.0,
    steps: int = 10,
    wall: float = 60.0,
    clock=None,
) -> ActorBudget:
    kw = {
        "max_tokens": tokens,
        "max_cost_usd": cost,
        "max_steps": steps,
        "max_wall_seconds": wall,
    }
    if clock is not None:
        kw["clock"] = clock
    return ActorBudget(**kw)


def test_construction_zeros_usage_and_reservations():
    b = _budget(tokens=100, cost=0.5, steps=5, wall=30.0)
    assert b.max_tokens == 100 and b.max_cost_usd == pytest.approx(0.5)
    assert b.max_steps == 5 and b.max_wall_seconds == pytest.approx(30.0)
    assert b.used_tokens == 0 and b.used_cost_usd == 0.0 and b.used_steps == 0
    assert b.reserved_tokens == 0 and b.reserved_cost_usd == 0.0 and b.reserved_steps == 0


def test_charge_accumulates_per_axis():
    b = _budget()
    b.charge(tokens=200, cost_usd=0.1, steps=1)
    b.charge(tokens=300, cost_usd=0.2, steps=2)
    assert b.used_tokens == 500
    assert b.used_cost_usd == pytest.approx(0.3)
    assert b.used_steps == 3


def test_remaining_deducts_used_and_reserved():
    b = _budget(tokens=1000, cost=1.0, steps=10)
    b.charge(tokens=200, cost_usd=0.2, steps=1)
    b.reserve_for_child(tokens=300, cost_usd=0.3, steps=2)
    assert b.remaining_tokens() == 1000 - 200 - 300
    assert b.remaining_cost_usd() == pytest.approx(0.5)
    assert b.remaining_steps() == 7


def test_exhausted_trips_when_any_axis_at_zero():
    b = _budget(tokens=10, cost=10.0, steps=10)
    b.charge(tokens=10, cost_usd=0.0, steps=0)
    assert b.remaining_tokens() == 0
    assert b.exhausted() is True  # tokens axis at zero is enough


def test_can_spawn_child_is_pure():
    b = _budget(tokens=1000, cost=1.0, steps=10)
    assert b.can_spawn_child(request_tokens=500, request_cost_usd=0.5, request_steps=5) is True
    # State unchanged.
    assert b.reserved_tokens == 0 and b.reserved_cost_usd == 0.0 and b.reserved_steps == 0


def test_reserve_for_child_carves_a_slice():
    b = _budget(tokens=1000, cost=1.0, steps=10)
    b.reserve_for_child(tokens=400, cost_usd=0.4, steps=4)
    assert b.reserved_tokens == 400
    assert b.reserved_cost_usd == pytest.approx(0.4)
    assert b.reserved_steps == 4
    assert b.remaining_tokens() == 600


def test_reserve_for_child_raises_when_oversized():
    b = _budget(tokens=1000, cost=1.0, steps=10)
    b.reserve_for_child(tokens=800, cost_usd=0.0, steps=0)
    with pytest.raises(BudgetExhausted) as info:
        b.reserve_for_child(tokens=300, cost_usd=0.0, steps=0)
    # The constraining axis is exposed for callers that want to react.
    assert info.value.axis == "tokens"


def test_settle_child_caps_charge_at_reservation():
    """A child that over-spent doesn't bleed past its reservation —
    the parent's books only reflect what was allotted."""
    b = _budget(tokens=1000, cost=1.0, steps=10)
    b.reserve_for_child(tokens=500, cost_usd=0.5, steps=5)
    b.settle_child(
        reserved_tokens=500,
        reserved_cost_usd=0.5,
        reserved_steps=5,
        used_tokens=600,  # over-spend
        used_cost_usd=0.6,
        used_steps=6,
    )
    # Charge capped at reservation, not actual spend.
    assert b.used_tokens == 500
    assert b.used_cost_usd == pytest.approx(0.5)
    assert b.used_steps == 5
    # Reservation released.
    assert b.reserved_tokens == 0


def test_settle_child_releases_unused_reservation():
    """A child that under-spent returns the unused slice."""
    b = _budget(tokens=1000, cost=1.0, steps=10)
    b.reserve_for_child(tokens=500, cost_usd=0.0, steps=0)
    b.settle_child(
        reserved_tokens=500,
        reserved_cost_usd=0.0,
        reserved_steps=0,
        used_tokens=200,  # child used much less than reserved
        used_cost_usd=0.0,
        used_steps=0,
    )
    assert b.used_tokens == 200
    assert b.reserved_tokens == 0
    # Headroom: max - used (200) - reserved (0) = 800
    assert b.remaining_tokens() == 800


def test_tighten_lowers_caps():
    b = _budget(tokens=1000, cost=1.0, steps=10, wall=60.0)
    b.tighten(new_max_tokens=500, new_max_cost_usd=0.5, new_max_steps=5, new_max_wall_seconds=30.0)
    assert b.max_tokens == 500
    assert b.max_cost_usd == pytest.approx(0.5)
    assert b.max_steps == 5
    assert b.max_wall_seconds == pytest.approx(30.0)


def test_tighten_never_raises_existing_cap():
    b = _budget(tokens=100)
    b.tighten(new_max_tokens=10_000)  # trying to raise
    assert b.max_tokens == 100  # no change


def test_tighten_clamps_to_used_plus_reserved():
    """A proposed cap below current commitments would retroactively
    exhaust the budget. tighten() clamps to ``used + reserved`` so
    the agent observes "no further work" instead of an impossible
    negative remaining."""
    b = _budget(tokens=1000)
    b.charge(tokens=300, cost_usd=0.0, steps=0)
    b.reserve_for_child(tokens=200, cost_usd=0.0, steps=0)
    # Try to tighten BELOW used (300) + reserved (200) = 500.
    b.tighten(new_max_tokens=100)
    # Cap clamped up to the floor, not down to 100.
    assert b.max_tokens == 500


def test_wall_clock_auto_tracks_via_injected_clock():
    """The wall-clock axis decreases as time passes (controlled via the
    injected ``clock`` callable). No explicit charge needed."""
    fake_time = [1000.0]

    def fake_clock() -> float:
        return fake_time[0]

    b = _budget(wall=10.0, clock=fake_clock)
    assert b.remaining_wall_seconds() == pytest.approx(10.0)
    fake_time[0] += 3.0
    assert b.remaining_wall_seconds() == pytest.approx(7.0)
    fake_time[0] += 100.0  # past the cap
    assert b.remaining_wall_seconds() == 0.0
    assert b.exhausted() is True


def test_budget_exhausted_carries_axis():
    b = _budget(tokens=100, cost=10.0, steps=100)
    with pytest.raises(BudgetExhausted) as info:
        b.reserve_for_child(tokens=200, cost_usd=0.0, steps=0)
    assert info.value.axis == "tokens"
    assert "tokens=200" in str(info.value)
