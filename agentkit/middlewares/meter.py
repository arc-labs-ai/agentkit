"""meter middleware — guard every meter before the work, charge them after.

A `BaseMiddleware` (transform/observe): `on_request` guards the per-run `Budget` and any per-tenant
`Quota` (`ctx.run.all_meters`) before any spend; `on_response` charges them on results carrying `usage`
(chat). Tool calls (no usage) pass the guard and aren't charged.

After charging, `on_response` also drops a `budget.checkpoint` event on the currently-open chat span
with the post-charge spent + remaining USD. Operators can read the trace to spot the call that
crossed a budget threshold without correlating against an external metric.

Two things the generic `BaseMiddleware` compilation could not express, so this class owns its own
`as_middleware()` (see there): charging an ABANDONED stream, and NOT charging a cache hit.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from agentkit.kernel.middleware import (
    BaseMiddleware,
    Call,
    Handler,
    Middleware,
    MiddlewareContext,
    _assemble,
    _drive,
)


class MeterMiddleware(BaseMiddleware):
    async def on_request(self, ctx: MiddlewareContext) -> None:
        for m in ctx.run.all_meters:
            await m.guard(ctx.call)  # over a ceiling → MeterExceeded, before any spend

    async def on_response(self, ctx: MiddlewareContext, result: Any) -> Any:
        # A memoize/idempotent HIT re-emits the stored result verbatim,
        # ``usage`` and all — so a meter that charges on "any result carrying
        # usage" bills the same provider call once per hit. Measured: four
        # identical chats behind ``memoize`` made ``provider_calls=1`` but drove
        # ``budget.spent_usd`` 0.25 → 1.0, and a fifth raised ``MeterExceeded``
        # on $0.75 of money that was never spent — a cache turned into a
        # budget-exhaustion bug, which is the exact opposite of its purpose.
        # ``call.meta["cache_hit"]`` is the signal memoize already sets for this
        # (``Audit`` reads it to mark a record "deduped"); reading it here costs
        # nothing and needs no new plumbing. Tracing sits OUTSIDE meter in the
        # documented chain, so a hit is still traced — it is just not billed.
        if ctx.call.meta.get("cache_hit"):
            return result
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

    def as_middleware(self) -> Middleware:
        """Compile to the `(call, next)` primitive — with a settle path that
        also runs when the stream is ABANDONED.

        ``BaseMiddleware.as_middleware`` wraps the inner stream in
        ``except Exception``, so ``on_response`` is reached only on normal
        exhaustion. ``GeneratorExit`` (a consumer that ``break``s out of the
        ``async for``, or any early ``aclose()``) and ``CancelledError`` are
        ``BaseException``s: they fly straight past that handler and the charge
        never happens. Measured: a consumer breaking after one delta produced
        ``provider stream calls: 2`` while the budget still read
        ``spent=0.25 calls=1`` from the one fully-consumed call. The provider
        billed those tokens; the budget never saw them, so every ceiling in the
        run was quietly wrong — and an abandon-happy caller could spend without
        limit.

        Design of the settle path:

        * Normal exhaustion charges exactly as before, ``MeterExceeded``
          included — that is the ceiling doing its job.
        * An abandoned / cancelled / failed stream charges only what actually
          ARRIVED (``if items``), because that is the only spend we can honestly
          attest to. A pre-stream failure yielded nothing, so it is not charged
          and does not inflate ``budget.calls`` — which preserves today's
          behaviour for the retry-inside-meter chain.
        * That path also swallows ``MeterExceeded``. ``Budget.charge`` updates
          the ledger BEFORE it evaluates the ceiling, so the money is recorded
          either way; raising out of a ``GeneratorExit`` unwind would replace
          the close signal with an unrelated error in a consumer that has
          already walked away. The next ``on_request`` guard raises instead.
        """

        async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
            mctx = MiddlewareContext(call)
            # Driven through ``_drive`` (not awaited directly) so the phases keep
            # BaseMiddleware's contract: a subclass may write them as async
            # generators yielding ``Observation``s.
            await _drive(call, self.on_request(mctx), default=None)  # raises before any spend
            items: list[Any] = []
            try:
                async for item in nxt(call):
                    items.append(item)
                    yield item  # observe-only: deltas pass through untouched
            except BaseException:
                if items:
                    with contextlib.suppress(Exception):
                        await _drive(call, self.on_response(mctx, _assemble(call, items)), default=None)
                raise
            await _drive(call, self.on_response(mctx, _assemble(call, items)), default=None)

        return mw


def meter() -> MeterMiddleware:
    """Factory: build a fresh ``MeterMiddleware``. The lowercase name
    matches the rest of the middleware factories (``tracing()``,
    ``retry()``, …) used in typical chat / tool chains."""
    return MeterMiddleware()
