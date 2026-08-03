# Runtime

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
- `autonomy` — the run's autonomy tier (`suggest` / `confirm` / `auto`),
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

### `EventBus`

Fan-out for lifecycle and observation events, kept in-process by
default. Adapters can bridge it to Redis Streams, OTel, or the
research·io Broker.

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
   `BudgetExhausted` and unwinds the call — it doesn't just warn.

## API

Full generated reference lives at
[API › runtime](../api-reference/runtime.md).
