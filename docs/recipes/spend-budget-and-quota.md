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

## Working code

```python
import asyncio

from agentkit import Budget, ChatRequest, Message, MeterExceeded, Quota, Scope
from agentkit.middlewares import meter
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    llm = FakeLLM("ok")

    ctx = make_test_ctx(
        llm=llm,
        scope=Scope(org_id="acme", domain_id="research"),
        budget=Budget(max_cost_usd=0.00025),      # ~2 chat calls at $0.0001 each
        meters=[Quota(max_rpm=3)],                # extra meter beyond Budget
        chat_middleware=[meter()],
    )

    req = ChatRequest(messages=[Message("user", "hello")], model="m")
    for i in range(1, 6):
        try:
            await ctx.invoker.chat(req, ctx)
            print(f"call {i}: OK    spent=${ctx.budget.spent_usd:.4f}")
        except MeterExceeded as exc:
            print(f"call {i}: STOP  {exc}")
            break


asyncio.run(main())
```

## How it works

The `meter()` middleware calls `guard(...)` on every entry in
`ctx.all_meters` before the invoker runs the call, and `charge(...)`
after. Both hit an async lock so totals are invariant under concurrent
workers.

`Budget._check` uses **strict greater-than** (`spent > max`) — a call
whose cost lands exactly on the ceiling completes; the next one trips
`MeterExceeded`. `Budget` is also the run's depth + concurrency
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

- **`meter()` is not on by default.** `make_test_ctx()` doesn't wire
  a chat chain unless you pass one. In a real wire-up, put `meter()`
  after `tracing()` and before caching / retry so the meter observes
  every attempted call.
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
