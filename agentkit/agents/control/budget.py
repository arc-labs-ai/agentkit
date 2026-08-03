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


class ActorBudget:
    """Per-agent four-axis budget with reservation accounting.

    Construct with the agent's caps. The wall-clock axis is rooted
    at construction time (``_started_at``) — the budget knows when
    "now - started_at" exceeds ``max_wall_seconds`` even without
    explicit ticks.

    Concurrency: not thread-safe; assumed owned by a single agent's
    loop. Cross-agent budget interactions (parent's spawn /
    settlement of children) happen at well-defined transitions and
    the framework runs those serially on the parent's event loop.
    """

    __slots__ = (
        "_clock",
        "_started_at",
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
        max_cost_usd: float,
        max_steps: int,
        max_wall_seconds: float,
        clock: Callable[[], float] = _monotonic_seconds,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.max_steps = max_steps
        self.max_wall_seconds = max_wall_seconds
        self.used_tokens = 0
        self.used_cost_usd = 0.0
        self.used_steps = 0
        self.reserved_tokens = 0
        self.reserved_cost_usd = 0.0
        self.reserved_steps = 0
        self._clock = clock
        self._started_at = clock()

    # ── available / remaining ────────────────────────────────────────

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens - self.reserved_tokens)

    def remaining_cost_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.used_cost_usd - self.reserved_cost_usd)

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.used_steps - self.reserved_steps)

    def remaining_wall_seconds(self) -> float:
        elapsed = self._clock() - self._started_at
        return max(0.0, self.max_wall_seconds - elapsed)

    def exhausted(self) -> bool:
        """True if any axis has nothing left to give. The agent loop
        checks this at the top of every iteration."""
        return (
            self.remaining_tokens() == 0
            or self.remaining_cost_usd() == 0.0
            or self.remaining_steps() == 0
            or self.remaining_wall_seconds() == 0.0
        )

    # ── spend ────────────────────────────────────────────────────────

    def charge(
        self,
        *,
        tokens: int = 0,
        cost_usd: float = 0.0,
        steps: int = 0,
    ) -> None:
        """Record actual spend. Soft-exceeds the cap (lets the
        in-flight tool call complete) — the next ``exhausted()``
        check trips the loop. Never raises so a recently-charged
        call doesn't surface as a crash; the loop checks
        ``exhausted()`` and stops cleanly.
        """
        self.used_tokens += tokens
        self.used_cost_usd += cost_usd
        self.used_steps += steps

    # ── child spawn / settlement ────────────────────────────────────

    def can_spawn_child(
        self,
        *,
        request_tokens: int,
        request_cost_usd: float,
        request_steps: int,
    ) -> bool:
        """Precheck — would the requested slice fit?

        Pure: no state mutation. Pair with ``reserve_for_child`` for
        the actual carve-out.
        """
        return (
            self.remaining_tokens() >= request_tokens
            and self.remaining_cost_usd() >= request_cost_usd
            and self.remaining_steps() >= request_steps
        )

    def reserve_for_child(
        self,
        *,
        tokens: int,
        cost_usd: float,
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
                f"tokens={tokens} cost=${cost_usd:.4f} steps={steps}; "
                f"remaining tokens={self.remaining_tokens()} "
                f"cost=${self.remaining_cost_usd():.4f} "
                f"steps={self.remaining_steps()}",
            )
        self.reserved_tokens += tokens
        self.reserved_cost_usd += cost_usd
        self.reserved_steps += steps

    def settle_child(
        self,
        *,
        reserved_tokens: int,
        reserved_cost_usd: float,
        reserved_steps: int,
        used_tokens: int,
        used_cost_usd: float,
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
        self.reserved_tokens = max(0, self.reserved_tokens - reserved_tokens)
        self.reserved_cost_usd = max(0.0, self.reserved_cost_usd - reserved_cost_usd)
        self.reserved_steps = max(0, self.reserved_steps - reserved_steps)
        # Charge against the parent's used_* but never more than what
        # was reserved (over-spending is the child's problem).
        self.used_tokens += min(used_tokens, reserved_tokens)
        self.used_cost_usd += min(used_cost_usd, reserved_cost_usd)
        self.used_steps += min(used_steps, reserved_steps)

    # ── live-tightening (parent-initiated BudgetReduced) ─────────────

    def tighten(
        self,
        *,
        new_max_tokens: int | None = None,
        new_max_cost_usd: float | None = None,
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
        if new_max_cost_usd is not None and new_max_cost_usd < self.max_cost_usd:
            self.max_cost_usd = max(self.used_cost_usd + self.reserved_cost_usd, new_max_cost_usd)
        if new_max_steps is not None and new_max_steps < self.max_steps:
            self.max_steps = max(self.used_steps + self.reserved_steps, new_max_steps)
        if new_max_wall_seconds is not None and new_max_wall_seconds < self.max_wall_seconds:
            self.max_wall_seconds = new_max_wall_seconds

    # ── helpers ──────────────────────────────────────────────────────

    def _tightest_axis(
        self, request_tokens: int, request_cost_usd: float, request_steps: int
    ) -> str:
        """Which axis would block a spawn first? Diagnostic for
        ``BudgetExhausted`` so the error message points at the real
        cause."""
        slack = {
            "tokens": self.remaining_tokens() - request_tokens,
            "cost_usd": self.remaining_cost_usd() - request_cost_usd,
            "steps": self.remaining_steps() - request_steps,
        }
        return min(slack, key=lambda k: slack[k])


__all__ = ["BudgetExhausted", "ActorBudget"]
