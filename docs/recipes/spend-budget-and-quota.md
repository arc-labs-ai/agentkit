# How do I cap a run's spend?

## When you'd want this

Any run that talks to a real LLM can, in principle, overspend. The
answer isn't "watch a dashboard" — it's a hard, in-loop ceiling that
halts the run when it's crossed. agentkit ships two meters that share
one Protocol:

- **`Budget`** — per-run cost / call ceiling. One instance per run.
- **`Quota`** — per-tenant rolling window (RPM, TPM, USD/window),
  keyed by `Scope.key()`. One shared instance across all runs of a
  tenant.

Both hook in through the same `meter()` middleware in the chat chain.
By default overspend raises `MeterExceeded` — the invoker unwinds and
the loop halts. With `on_exceeded="stop"` it returns a **verdict**
instead, and the run stops *after* writing a checkpoint, so the spend
is recoverable. See
[Making exhaustion recoverable](#making-exhaustion-recoverable).

Money is `Decimal`, not `float`. Binary floating point cannot
represent `0.01`, so a hundred one-cent charges summed as floats land
at `1.0000000000000007` and a metered run cannot be reconciled to the
cent. `Budget` keeps an exact ledger — read it with `budget.spent()`
(`Decimal`) or `budget.spent_cents()` for invoicing.
`budget.spent_usd` is still there and is a float *mirror* of the
ledger, kept in sync after every charge: fine for display, not for
summing.

!!! note "Assumes `ANTHROPIC_API_KEY` in the environment"
    Wired via `providers.claude(...)` on the `Invoker`. Swap for
    `providers.openai` (and set `OPENAI_API_KEY`) if that's what you
    have — the rest of the wiring is unchanged.

## Working code

```python
"""Requires ANTHROPIC_API_KEY in the environment.

Demonstrates the halt path with `max_calls=1` — a `max_cost_usd`
ceiling is set generously so cost is not what stops the loop; the
per-call ceiling reliably trips on the second `invoker.chat(...)`."""

import asyncio
import os

from agentkit import Budget, ChatRequest, Message, MeterExceeded, Quota, Scope
from agentkit.adapters.llm import providers
from agentkit.middlewares import meter
from agentkit.runtime import Invoker, RunContext, Services


async def main() -> None:
    llm = providers.claude(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )
    services = Services(invoker=Invoker(llm=llm, chat_middleware=[meter()]))

    ctx = RunContext(
        correlation_id="run-1",
        scope=Scope(org_id="acme", domain_id="research"),
        budget=Budget(max_cost_usd=0.10, max_calls=1),  # halts on the second call
        meters=[Quota(max_rpm=3)],                       # extra per-tenant window
        services=services,
    )

    req = ChatRequest(
        messages=[Message("user", "One short sentence about octopus cognition.")],
        model="claude-sonnet-4-6",
    )
    for i in range(1, 6):
        try:
            await ctx.invoker.chat(req, ctx)
            print(f"call {i}: OK    spent=${ctx.budget.spent_usd:.4f}")
        except MeterExceeded as exc:
            print(f"call {i}: STOP  {exc}")
            break


if __name__ == "__main__":
    asyncio.run(main())
```

## How it works

The `meter()` middleware calls `guard(...)` on every entry in
`ctx.all_meters` before the invoker runs the call, and `charge(...)`
after. Both hit an async lock so totals are invariant under concurrent
workers.

`Budget._verdict` uses **strict greater-than** (`spent > max`) — a call
whose cost lands exactly on the ceiling completes; the next one trips
`MeterExceeded`. The same applies to `max_calls`: the first call
lands on `calls == 1`, then the second call's `guard` sees `1 > 1`
false, `charge` bumps to `2`, and the third call's `guard` trips.
That's why `max_calls=1` above stops the loop on the second iteration,
not the first. `Budget` is also the run's depth + concurrency
authority: `max_depth` caps how deep the agent tree can spawn, and
`semaphore()` bounds tree-wide concurrency (default 8).

`Quota` is a rolling window keyed by `Scope.key()`. It counts requests
made (not just completed), so a burst that overshoots RPM is caught on
the way in, not after the fact. The in-memory implementation is fine
for a single process; a multi-process deployment wants a Redis-backed
Quota (write your own `Meter` impl behind the same Protocol).

## Making exhaustion recoverable

The default `on_exceeded="raise"` has a sharp edge worth knowing
about. `MeterExceeded` is raised from inside `charge()`, which runs in
the meter middleware's `on_response`, which is inside
`ctx.invoker.stream(...)` — so it unwinds **past every checkpoint
write in the loop**. A tool-loop run that crosses its ceiling on the
second turn aborts holding the *first* turn's checkpoint: marked
`running` rather than `suspended`, and recording only half the money
that actually left the account.

`on_exceeded="stop"` fixes that. The meter records the spend and
returns a `Charge` verdict; control reaches the cognition, which still
holds the live context and writes a current `suspended` checkpoint
before ending the run:

```python
budget = Budget(max_cost_usd="1.00", on_exceeded="stop")
result = await agent.run(task, ctx)

if result.stop_reason == "budget_exhausted":
    assert result.is_resumable                  # a checkpoint exists, and it's current
    raise_the_ceiling_and_requeue(ctx.correlation_id)
```

Both cognitions also **pre-flight** the budget at the top of each
iteration, so retrying a run whose ceiling nobody raised costs nothing
— it stops before making a call rather than after. And
`budget.max_cost_usd = 10.0` after construction really does raise the
ceiling; the normalised value re-derives on assignment.

Raising is still the **default**, deliberately. Flipping it would
silently change control flow in every existing wiring: a run that used
to abort would continue past its ceiling in any caller that ignores
the return value — a worse failure than the one being fixed.

The verdict itself is useful even when you keep raising:

```python
verdict = await budget.charge(call, usage)
verdict.ok            # False when a ceiling was crossed
verdict.reason        # "cost $1.2 > $1"
verdict.spent         # Decimal — exact (Budget.spent_usd is the float mirror)
verdict.remaining     # Decimal | None
verdict.usage         # cumulative Usage — input/output/cache tokens, not just cost
verdict.raise_if_exceeded()   # back to the old control flow, at one site
```

## Token counts, not just cost

`Budget.usage` accumulates the whole `Usage` — input, output,
cache-read and cache-write tokens. Because `Budget` is shared **by
reference** across `ctx.child()`, a whole agent tree rolls up into
one object:

```python
budget.usage.input_tokens        # across every agent in the tree
budget.usage.cache_read_tokens   # cache effectiveness, for free
budget.usage.total_tokens
```

Applications used to re-aggregate this from spans or from their own
callbacks. They no longer need to.

## Per-agent budgets

For multi-agent runs, `ActorBudget` gives each child agent its own
four-axis envelope (`tokens`, `cost_usd`, `steps`, `wall_seconds`) with
reservation accounting. When `run_agents(...)` fans children out under
a parent's `ActorBudget`, each child gets a `1/N` slice reserved before
dispatch; overspend on one child can't drain the parent's envelope
because `settle_child(...)` caps at the reservation.

```python
from agentkit.agents.control.budget import ActorBudget

ctx.actor_budget = ActorBudget(
    max_tokens=100_000, max_cost_usd="5.00", max_steps=50, max_wall_seconds=600,
)
```

The envelope is charged by the same `meter()` middleware that charges the
run `Budget` — one model call is one step, plus its tokens and cost. Its
cost axis is an exact `Decimal` ledger like `Budget`'s; `used_cost()` /
`remaining_cost()` are the exact reads and the `*_usd` attributes are float
mirrors for display.

Unlike `Budget`, there is no `on_exceeded` switch: `ActorBudget.charge`
never raises, so an in-flight call always completes, and the cognition's
next pre-flight stops the loop with `stop_reason="budget_exhausted"` naming
the axis that ran out. A token-out is a "wind down" signal; a wall-out may
warrant a hard cancel, and the reason string tells you which.

`BudgetExhausted` (distinct from `Budget`'s `MeterExceeded`) is raised when a
fan-out cannot be carved — **fail-fast, before any child runs**, on any of the
three reservation axes:

- the parent envelope is already exhausted, or
- a slice would round to nothing (N children against fewer than N tokens,
  steps, or micro-dollars).

Both cases used to produce N children that each stopped on their first check —
a fan-out that looked like it ran and did nothing. The exception names the
axis. Slices are equal (so reservation order cannot skew fairness) and the
money axis is carved in `Decimal`, so nothing is lost to float rounding; each
child is granted exactly what was reserved for it and no more.

## Gotchas

- **`meter()` is not on by default.** You wire it into
  `Invoker(chat_middleware=[..., meter(), ...])`. Put it after
  `tracing()` and before caching / retry so the meter observes every
  attempted call.
- **`Budget.max_cost_usd=None` is unlimited.** Set both `max_cost_usd`
  and `max_calls` unless you have a reason to leave one open.
- **`Quota` counts by attempted request, not by success.** A tenant
  that keeps triggering timeouts still burns RPM. This is deliberate —
  a naive "count on success" surface lets a broken client hammer the
  provider.
- **`MeterExceeded` vs `BudgetExhausted` are two different signals.**
  `MeterExceeded` (from `agentkit.runtime.meter`) is the per-run
  ceiling; `BudgetExhausted` (from `agentkit.agents.control.budget`)
  is the per-actor `ActorBudget` signal. Catch each where it fires.
- **A budget is always overrun by at most one call's cost.** `_check`
  compares `spent > ceiling` *after* the work has run; there is no
  pre-flight estimate. Set the ceiling slightly below your true limit,
  and use `on_exceeded="stop"` so the overshoot is recoverable rather
  than fatal.
- **An unpriced model costs `$0.00`.** `pricing.cost()` returns zero
  for a model it doesn't know, so a ceiling never fires and the run is
  effectively unbounded.

    Registering the model in the [model registry](provider-from-env.md)
    does **not** fix this — that table routes names to providers and
    declares capabilities, and `ModelEntry` has no price field. The
    price table is separate. Pass your own `pricing=` callable to the
    provider instead:

    ```python
    def my_pricing(model: str, usage: Usage) -> float:
        rate_in, rate_out = MY_RATES[model]          # $ per 1M tokens
        return (usage.input_tokens * rate_in + usage.output_tokens * rate_out) / 1e6

    llm = claude(api_key=..., pricing=my_pricing)
    ```

    It takes `(model, usage)` and returns dollars; `None` (the default)
    uses the bundled table. If you meter spend, assume the bundled
    table is stale for anything recent and supply your own.
- **An over-precise *ceiling* is refused; an over-precise *charge* is
  quantized.** `Budget(max_cost_usd="0.0000001")` raises
  `MoneyPrecisionError` at construction, because a ceiling is your
  stated intent and rounding it changes what you asked for. A charge
  is a measurement, so it is recorded at 6dp rather than aborting a
  run mid-flight.
- **Quantize at read, not per charge.** `budget.spent_cents()` rounds
  once, at the end. Rounding each charge to cents would round every
  sub-cent call to zero and undercount the whole run.

## Related

- [Tutorial · Step 4](../tutorial.md#step-4-cap-the-run-with-a-budget)
  — the same primitive introduced inside the walkthrough.
- [Concepts · Runtime](../concepts/runtime.md) — where `Budget` /
  `Quota` / `Meter` sit alongside `RunContext` and `Invoker`.
- [Parallel agents with cancellation](parallel-agents-with-cancellation.md)
  — the concurrency surface `Budget.semaphore()` bounds.
