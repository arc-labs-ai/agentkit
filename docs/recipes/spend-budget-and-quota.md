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
Overspend raises `MeterExceeded` — the invoker unwinds and the loop
halts cleanly.

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

`Budget._check` uses **strict greater-than** (`spent > max`) — a call
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

## Per-agent budgets

For multi-agent runs, `ActorBudget` gives each child agent its own
four-axis envelope (`tokens`, `cost_usd`, `steps`, `wall_seconds`) with
reservation accounting. When `run_agents(...)` fans children out under
a parent's `ActorBudget`, each child gets a `1/N` slice reserved before
dispatch; overspend on one child can't drain the parent's envelope
because `settle_child(...)` caps at the reservation. See
`agentkit.agents.control.budget.ActorBudget`. `ActorBudget` raises
`BudgetExhausted` (a distinct exception from `Budget`'s
`MeterExceeded`), carrying the exhausted axis so callers can react
differently to a token-out vs wall-out.

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

## Related

- [Tutorial · Step 4](../tutorial.md#step-4-cap-the-run-with-a-budget)
  — the same primitive introduced inside the walkthrough.
- [Concepts · Runtime](../concepts/runtime.md) — where `Budget` /
  `Quota` / `Meter` sit alongside `RunContext` and `Invoker`.
- [Parallel agents with cancellation](parallel-agents-with-cancellation.md)
  — the concurrency surface `Budget.semaphore()` bounds.
