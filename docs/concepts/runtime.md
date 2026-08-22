# Runtime

!!! abstract "Where this fits in the four themes"
    This page covers primitives from the **Control** theme
    (`Budget`, `Quota`, `MeterExceeded`, `Autonomy` on `RunContext`,
    `CancellationToken`) and threads the **State** theme
    (`RunContext` is what carries per-request state through every hop
    — scope, budget, cancel, services). The `Invoker`'s middleware
    dispatch is the entry point into the **Behaviour** theme's chain.
    See the four-theme grid on the [landing](../index.md).

**What this is.** The runtime is the *per-request universe* every piece
of agent code runs in. Concretely, that's `RunContext` (which carries a
correlation id, tenant `Scope`, `Budget`, `Services` bundle, and a
`CancellationToken`), the `Invoker` that dispatches LLM and tool calls
through the middleware chain, and the `Meter` / `Quota` / `EventBus`
that make cost and lifecycle visible.

**Why it exists.** Passing loose keyword arguments through an agent
graph is how state leaks. A `RunContext` is the one object every layer
takes and every layer respects — cancel it and every hop stops; charge
it and the budget shows up in the meter; scope it and memory partitions
correctly. If the kernel is the *vocabulary*, the runtime is the *frame
of reference*.

## The pieces

### `RunContext`

The single value threaded through every call. Carries:

- `correlation_id` — the request id that ties trace, meter, and log
  events together.
- `scope: Scope` — tenant / domain axis, used to partition caches,
  quotas, and scoped memory.
- `budget: Budget` — cost and call ceilings for the whole run.
- `services: Services` — the wired-in `Invoker`, `Observer`, `Trace`,
  and (optionally) `Store`.
- `cancel: CancellationToken` — cooperative cancellation surface.
- `autonomy` — the run's autonomy tier (`auto` / `gated` / `manual`),
  read by tools and cognitions that gate on human approval.

### `Invoker`

The one entrypoint for LLM and tool calls. It wraps each call in a
`Call(...)` envelope and drives it through the composed middleware
chain. Because everything goes through the invoker, the whole system
gets tracing / metering / retry / caching by editing one list.

### `Budget`, `Quota`, `Meter`

`Budget` is the run-scoped ceiling. `Quota` is the multi-run,
scope-partitioned ceiling (per tenant, per org). `Meter` is the
middleware that accrues `Usage` and enforces both.

Three properties worth knowing:

- **Money is `Decimal`.** Binary floating point cannot represent
  `0.01`, so a float ledger cannot be reconciled to the cent. Read
  `budget.spent()` (exact) or `budget.spent_cents()` (invoicing);
  `budget.spent_usd` is a float mirror kept in sync for display.
- **The whole `Usage` accumulates**, not just a cost scalar.
  `budget.usage` carries input / output / cache-read / cache-write
  tokens for the entire agent tree, because `Budget` is shared by
  reference across `ctx.child()`.
- **`charge()` returns a `Charge` verdict.** Raising is still the
  default, but `on_exceeded="stop"` lets the caller act on exhaustion
  instead — which is what allows a tool-loop run to write a checkpoint
  *before* it stops. See
  [the recipe](../recipes/spend-budget-and-quota.md#making-exhaustion-recoverable).

### Concurrency: the permit pool is per LEVEL

`ctx.semaphore()` returns the pool for **this context's depth**, not one pool
for the whole tree. A single tree-wide semaphore deadlocks nested fan-out: a
parent's fan-out holds its permits for the entire duration of each child run,
so an inner fan-out draws from a pool its own ancestors have already drained.
At `max_concurrency=2`, an agent dispatching two `as_tool` sub-agents that
each dispatch their own tools hung forever.

Every nesting boundary goes through `ctx.child()` (`as_tool`, `run_agents`,
the coordinator policies), so keying on depth breaks the cycle structurally —
an ancestor at depth *d* can only hold permits from pool *d*, and its children
draw from pool *d+1*.

The trade is honest: the bound is `max_concurrency` **per level**, so
worst-case in-flight work is `max_concurrency * (max_depth + 1)`. Set
`max_concurrency` with that in mind. A single tree-wide cap cannot be both
deadlock-free and respected by nested acquisition.

### `Services.asker`

The human-in-the-loop transport (`agents.control.elicitation.Asker`). When
set, a cognition **parks** on a gated decision — it awaits the person
from inside its own coroutine, so live unserialisable state survives.
When unset, the classic checkpoint-and-resume path runs unchanged. The
runtime never branches on transport; implementing `async def ask` is
the whole integration.

### `EventBus`

Fan-out for lifecycle and observation events, kept in-process by
default. Adapters can bridge it to Redis Streams, OTel, or a
downstream message broker.

### `NullCtx`

A minimal context used in unit tests and single-shot scripts, so you
can call the middleware chain without wiring a full services bundle.

## The invariants it enforces

1. **One context per request.** Never fabricate a fresh `RunContext`
   inside a middleware or tool; forward the one you were given.
2. **Cancellation is cooperative.** Every `await` point in an agent
   loop must be reachable to `cancel.raise_if_set()`; blocking
   primitives are banned on the hot path.
3. **Budgets are enforced, not advisory.** `Budget` overspend raises
   `MeterExceeded` (from `agentkit.runtime.meter`) and unwinds the
   call — it doesn't just warn. (Distinct from `BudgetExhausted` in
   `agentkit.agents`, which is a per-actor `ActorBudget` signal.)
   Under `on_exceeded="stop"` the enforcement is a returned verdict
   rather than an exception — still enforced, but recoverable.
4. **Enforcement is post-hoc by one call.** `spent > ceiling` is
   evaluated after the work runs; there is no pre-flight estimate, so
   a budget is always overrun by at most one call's cost. Set the
   ceiling below your true limit.

## API

Full generated reference lives at
[API › runtime](../api-reference/runtime.md).
