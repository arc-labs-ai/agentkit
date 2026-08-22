"""ActorBudget — per-agent envelope with reserve/charge/settle.

Complements ``agentkit.runtime.meter.Budget`` (one global envelope
per RUN). ``ActorBudget`` is the per-actor companion: one envelope
per agent, with reservation accounting for child spawns. The two
compose — run-wide ``Budget`` caps total cost; per-agent
``ActorBudget`` slices that cap across the tree so a runaway
descendant can't burn the whole envelope.

Four axes (tokens / cost_usd / steps / wall_seconds): LLM-driven
multi-agent runs can exhaust on any of them — token caps miss
tight-loop failures (spawn-then-stop burns steps not tokens), step
caps miss long blocking I/O, wall caps miss runaway concurrency.

Lifecycle of a child's slice:

    1. ``can_spawn_child`` — precheck.
    2. ``reserve_for_child`` — slice moves "available" → "reserved".
    3. Child runs, charges ``used_*`` on its own envelope.
    4. ``settle_child`` — reservation collapses into parent's
       ``used_*``, CAPPED at the reservation. Overspend stays on
       the child's books; the parent's slice is enforced.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal

from agentkit.runtime.meter import to_money


class BudgetExhausted(RuntimeError):
    """Raised when a budget axis would go negative.

    Carries the axis name so callers can react differently (a
    token-exhaustion is a "wind down gracefully" signal; a
    wall-clock-exhaustion may warrant a hard cancel).
    """

    def __init__(self, axis: str, message: str) -> None:
        super().__init__(message)
        self.axis = axis


def _monotonic_seconds() -> float:
    """Default wall-clock source. Monotonic so paused/sleeping
    processes don't accidentally "consume" wall budget."""
    return time.monotonic()


# The cost axis is an EXACT Decimal ledger, mirroring the run-scoped
# ``runtime.meter.Budget``. It was float, with an ``== 0.0`` exhaustion check,
# and that does not survive float arithmetic:
#
#     ActorBudget(max_cost_usd=1.0) charged ten times at 0.10
#       -> used 0.9999999999999999, remaining 1.11e-16, exhausted() False
#
# The dollar was gone and the loop kept going. ``remaining_*`` clamps at zero
# with ``max(0.0, ...)``, which catches an OVERSHOOT but not an undershoot, so
# the residue slipped through as "budget left". It was also inconsistent —
# 0.1 + 0.2 against a 0.3 cap happens to land exactly on zero — so the bug
# depended on which numbers a caller picked.
#
# A threshold hid that; exactness removes it. ``max_cost_usd`` /
# ``used_cost_usd`` / ``reserved_cost_usd`` remain readable FLOAT MIRRORS of
# the Decimal ledger, re-derived after every mutation, so ``run_agents`` and
# any caller reading them keep working untouched. ``*_cost()`` are the exact
# accessors to reconcile against.
#
# Wall-clock keeps a threshold: it is genuinely a float clock reading, and a
# nanosecond is far below any clock's resolution.
_WALL_EPSILON = 1e-9


class ActorBudget:
    """Per-agent four-axis budget with reservation accounting.

    Construct with the agent's caps. The wall-clock axis is rooted
    at construction time (``_started_at``) — the budget knows when
    "now - started_at" exceeds ``max_wall_seconds`` even without
    explicit ticks.

    Every money-bearing parameter accepts anything ``to_money`` accepts —
    ``float`` / ``int`` / ``str`` / ``Decimal`` — so a caller holding an exact
    amount (``run_agents`` slices in ``Decimal``) is not forced through float
    and back. Values are normalised on the way in.

    Concurrency: not thread-safe; assumed owned by a single agent's
    loop. Cross-agent budget interactions (parent's spawn /
    settlement of children) happen at well-defined transitions and
    the framework runs those serially on the parent's event loop.
    """

    __slots__ = (
        "_clock",
        "_max_cost",
        "_reserved_cost",
        "_started_at",
        "_used_cost",
        "max_cost_usd",
        "max_steps",
        "max_tokens",
        "max_wall_seconds",
        "reserved_cost_usd",
        "reserved_steps",
        "reserved_tokens",
        "used_cost_usd",
        "used_steps",
        "used_tokens",
    )

    def __init__(
        self,
        *,
        max_tokens: int,
        max_cost_usd: float | int | str | Decimal,
        max_steps: int,
        max_wall_seconds: float,
        clock: Callable[[], float] = _monotonic_seconds,
    ) -> None:
        self.max_tokens = max_tokens
        self._max_cost = to_money(max_cost_usd)
        self._used_cost = Decimal(0)
        self._reserved_cost = Decimal(0)
        self.max_steps = max_steps
        self.max_wall_seconds = max_wall_seconds
        self.used_tokens = 0
        self.used_cost_usd = 0.0  # float MIRROR — see the module note
        self.used_steps = 0
        self.reserved_tokens = 0
        self.reserved_cost_usd = 0.0  # float MIRROR
        self.reserved_steps = 0
        self._clock = clock
        self._started_at = clock()
        self._sync_cost_mirrors()

    # ── the exact cost ledger, and its float mirrors ─────────────────

    def _sync_cost_mirrors(self) -> None:
        """Re-derive the float attributes from the Decimal ledger.

        Keeping mirrors rather than changing the attribute types means
        ``run_agents`` (which does float share arithmetic over
        ``remaining_cost_usd()``) and every existing reader are untouched.
        """
        self.max_cost_usd = float(self._max_cost)
        self.used_cost_usd = float(self._used_cost)
        self.reserved_cost_usd = float(self._reserved_cost)

    def max_cost(self) -> Decimal:
        """The cap, exactly."""
        return self._max_cost

    def used_cost(self) -> Decimal:
        """Actual spend, exactly. THIS is the number to reconcile against."""
        return self._used_cost

    def reserved_cost(self) -> Decimal:
        """Currently-held child reservations, exactly."""
        return self._reserved_cost

    def remaining_cost(self) -> Decimal:
        """Headroom, exactly. Clamped at zero."""
        return max(Decimal(0), self._max_cost - self._used_cost - self._reserved_cost)

    # ── available / remaining ────────────────────────────────────────

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens - self.reserved_tokens)

    def remaining_cost_usd(self) -> float:
        """Headroom as a float — the shape ``run_agents`` and existing callers
        expect. :meth:`remaining_cost` is the exact one."""
        return float(self.remaining_cost())

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.used_steps - self.reserved_steps)

    def remaining_wall_seconds(self) -> float:
        elapsed = self._clock() - self._started_at
        return max(0.0, self.max_wall_seconds - elapsed)

    def exhausted(self) -> bool:
        """True if any axis has nothing left to give. The agent loop
        checks this at the top of every iteration."""
        return (
            self.remaining_tokens() <= 0
            or self.remaining_cost() <= 0
            or self.remaining_steps() <= 0
            or self.remaining_wall_seconds() <= _WALL_EPSILON
        )

    # ── spend ────────────────────────────────────────────────────────

    def charge(
        self,
        *,
        tokens: int = 0,
        cost_usd: float | int | str | Decimal = 0.0,
        steps: int = 0,
    ) -> None:
        """Record actual spend. Soft-exceeds the cap (lets the
        in-flight tool call complete) — the next ``exhausted()``
        check trips the loop. Never raises so a recently-charged
        call doesn't surface as a crash; the loop checks
        ``exhausted()`` and stops cleanly.
        """
        self.used_tokens += tokens
        self._used_cost += to_money(cost_usd)
        self.used_steps += steps
        self._sync_cost_mirrors()

    # ── child spawn / settlement ────────────────────────────────────

    def can_spawn_child(
        self,
        *,
        request_tokens: int,
        request_cost_usd: float | int | str | Decimal,
        request_steps: int,
    ) -> bool:
        """Precheck — would the requested slice fit?

        Pure: no state mutation. Pair with ``reserve_for_child`` for
        the actual carve-out.
        """
        return (
            self.remaining_tokens() >= request_tokens
            and self.remaining_cost() >= to_money(request_cost_usd)
            and self.remaining_steps() >= request_steps
        )

    def reserve_for_child(
        self,
        *,
        tokens: int,
        cost_usd: float | int | str | Decimal,
        steps: int,
    ) -> None:
        """Move a slice from "available" to "reserved".

        Raises ``BudgetExhausted`` if the request doesn't fit (after
        a ``can_spawn_child`` precheck this shouldn't fire, but the
        explicit raise here is the enforcement seam for races
        between concurrent spawn attempts).
        """
        if not self.can_spawn_child(
            request_tokens=tokens,
            request_cost_usd=cost_usd,
            request_steps=steps,
        ):
            axis = self._tightest_axis(tokens, cost_usd, steps)
            raise BudgetExhausted(
                axis,
                f"insufficient budget to spawn child requesting "
                f"tokens={tokens} cost=${float(cost_usd):.4f} steps={steps}; "
                f"remaining tokens={self.remaining_tokens()} "
                f"cost=${self.remaining_cost_usd():.4f} "
                f"steps={self.remaining_steps()}",
            )
        self.reserved_tokens += tokens
        self._reserved_cost += to_money(cost_usd)
        self.reserved_steps += steps
        self._sync_cost_mirrors()

    def settle_child(
        self,
        *,
        reserved_tokens: int,
        reserved_cost_usd: float | int | str | Decimal,
        reserved_steps: int,
        used_tokens: int,
        used_cost_usd: float | int | str | Decimal,
        used_steps: int,
    ) -> None:
        """Settle a child's reservation against its actual usage.

        Releases the reserved slice and accumulates the child's
        actual spend into the parent's ``used_*``. Capped at the
        reservation so an over-spending child can't silently exceed
        its slice on the parent's books — the parent observes "this
        child spent up to its allowance"; the over-spend stays on
        the child's own metrics.

        Idempotent if reservations are tracked correctly upstream —
        but the framework calls this exactly once per child
        ``DoneSignal``.
        """
        # Release the reserved hold.
        reserved_cost = to_money(reserved_cost_usd)
        self.reserved_tokens = max(0, self.reserved_tokens - reserved_tokens)
        self._reserved_cost = max(Decimal(0), self._reserved_cost - reserved_cost)
        self.reserved_steps = max(0, self.reserved_steps - reserved_steps)
        # Charge against the parent's used_* but never more than what
        # was reserved (over-spending is the child's problem).
        self.used_tokens += min(used_tokens, reserved_tokens)
        self._used_cost += min(to_money(used_cost_usd), reserved_cost)
        self.used_steps += min(used_steps, reserved_steps)
        self._sync_cost_mirrors()

    # ── live-tightening (parent-initiated BudgetReduced) ─────────────

    def tighten(
        self,
        *,
        new_max_tokens: int | None = None,
        new_max_cost_usd: float | int | str | Decimal | None = None,
        new_max_steps: int | None = None,
        new_max_wall_seconds: float | None = None,
    ) -> None:
        """Lower (never raise) the caps. The parent uses this when a
        ``BudgetReducedSignal`` lands on a child mid-flight.

        Caps never go BELOW current ``used_*`` (that would
        retroactively exhaust). The function silently clamps to
        ``used_*`` if the proposed new cap is lower — better to
        truncate to "no further work" than to claim the agent has
        already over-spent.
        """
        if new_max_tokens is not None and new_max_tokens < self.max_tokens:
            self.max_tokens = max(self.used_tokens + self.reserved_tokens, new_max_tokens)
        if new_max_cost_usd is not None:
            proposed = to_money(new_max_cost_usd)
            if proposed < self._max_cost:
                # Never below what is already committed — lowering the cap must
                # not retroactively make spent-plus-reserved look like an
                # overdraft. Compared in Decimal so a float-representable cap
                # cannot land a hair under the committed total and clamp wrong.
                self._max_cost = max(self._used_cost + self._reserved_cost, proposed)
                self._sync_cost_mirrors()
        if new_max_steps is not None and new_max_steps < self.max_steps:
            self.max_steps = max(self.used_steps + self.reserved_steps, new_max_steps)
        if new_max_wall_seconds is not None and new_max_wall_seconds < self.max_wall_seconds:
            self.max_wall_seconds = new_max_wall_seconds

    # ── helpers ──────────────────────────────────────────────────────

    def _tightest_axis(
        self, request_tokens: int, request_cost_usd: float | int | str | Decimal, request_steps: int
    ) -> str:
        """Which axis would block a spawn first? Diagnostic for
        ``BudgetExhausted`` so the error message points at the real cause.

        The cost slack is computed in ``Decimal`` and then compared as a float.
        Two reasons: a caller may hand us either a float or a ``Decimal``
        (``run_agents`` now slices exactly, in Decimal), and mixing the two
        raises ``TypeError`` — ``float - Decimal`` is unsupported, which turned
        a diagnostic into a crash on the very path it exists to explain. The
        comparison itself is a float because the three axes are different units
        and only their relative slack matters here.
        """
        slack = {
            "tokens": float(self.remaining_tokens() - request_tokens),
            "cost_usd": float(self.remaining_cost() - to_money(request_cost_usd)),
            "steps": float(self.remaining_steps() - request_steps),
        }
        return min(slack, key=lambda k: slack[k])


__all__ = ["BudgetExhausted", "ActorBudget"]
