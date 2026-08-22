"""meter middleware — guard every meter before the work, charge them after.

A `BaseMiddleware` (transform/observe): `on_request` guards the per-run `Budget` and any per-tenant
`Quota` (`ctx.run.all_meters`) before any spend; `on_response` charges them on results carrying `usage`
(chat). Tool calls (no usage) pass the guard and aren't charged.

After charging, `on_response` also drops a `budget.checkpoint` event on the currently-open chat span
with the post-charge spent + remaining USD. Operators can read the trace to spot the call that
crossed a budget threshold without correlating against an external metric.
"""

from __future__ import annotations

import contextlib
from typing import Any

from agentkit.kernel.middleware import BaseMiddleware, MiddlewareContext


class MeterMiddleware(BaseMiddleware):
    async def on_request(self, ctx: MiddlewareContext) -> None:
        for m in ctx.run.all_meters:
            await m.guard(ctx.call)  # over a ceiling → MeterExceeded, before any spend

    async def on_response(self, ctx: MiddlewareContext, result: Any) -> Any:
        usage = getattr(result, "usage", None)
        if usage is not None:
            for m in ctx.run.all_meters:
                # The return value is a ``Charge`` verdict on first-party
                # meters and ``None`` on a custom meter written against the
                # older ``-> None`` signature. The middleware deliberately
                # does NOT act on it: a middleware cannot write a checkpoint
                # or shape a terminal event, and inventing a second
                # stop-the-run mechanism here would compete with the
                # cognition's. Under ``on_exceeded="raise"`` the meter raises
                # and this is moot; under ``"stop"`` the cognition reads
                # ``ctx.budget.exhausted()`` between units of work, where it
                # still holds the state a clean stop needs.
                await m.charge(ctx.call, usage)
            # The per-actor envelope. NOT in ``all_meters``: ``ActorBudget`` is
            # not a ``Meter`` (four axes, a sync ``charge``, no guard/charge
            # protocol), so it has to be charged explicitly — and it never was.
            # The consequence was a documented safety mechanism that did
            # nothing: $3.00 of measured spend against a $1.00 ActorBudget cap
            # left ``used_cost`` at zero and ``exhausted()`` False, because the
            # only thing that ever touched the envelope was ``run_agents``
            # reserving and then releasing slices with zero usage.
            #
            # ``ActorBudget.charge`` is sync and documented never to raise: it
            # soft-exceeds the cap so the in-flight call completes, and the
            # cognition's next pre-flight check stops the loop.
            actor = getattr(ctx.run, "actor_budget", None)
            if actor is not None:
                actor.charge(
                    tokens=usage.total_tokens,
                    cost_usd=usage.cost_usd,
                    steps=1,  # one model call = one step on the actor's books
                )
            # Best-effort budget headroom event. We use ``budget.remaining_usd()``
            # which returns None when no ceiling is configured — only emit when a
            # real ceiling is set so a dev-default zero-budget call doesn't carry
            # noisy zeros. A misbehaving tracer must never break the run, so the
            # call is wrapped in ``contextlib.suppress(Exception)``.
            budget = getattr(ctx.run, "budget", None)
            trace = getattr(ctx.run, "trace", None)
            if budget is not None and trace is not None:
                remaining = budget.remaining_usd()
                if remaining is not None:
                    with contextlib.suppress(Exception):
                        trace.add_event_to_current_span(
                            "budget.checkpoint",
                            spent_usd=budget.spent_usd,
                            remaining_usd=remaining,
                            calls=budget.calls,
                        )
        return result


def meter() -> MeterMiddleware:
    """Factory: build a fresh ``MeterMiddleware``. The lowercase name
    matches the rest of the middleware factories (``tracing()``,
    ``retry()``, …) used in typical chat / tool chains."""
    return MeterMiddleware()
