# Runtime

The runtime is the per-request universe your agent code runs inside. One
object — `RunContext` — carries who the request belongs to, what it is
allowed to spend, how to stop it, and which collaborators it can reach.
Everything else in this package exists to make that object useful:
the `Invoker` that runs a call, the meters that bound it, and the event
bus that reports on it.

If the [kernel](kernel.md) is the vocabulary, the runtime is the frame of
reference.

## The problem it solves

Passing loose keyword arguments through an agent graph is how state
leaks. Two hops in, someone forgets to forward `tenant_id`, and a cache
lookup crosses a customer boundary. Three hops in, a sub-agent starts its
own budget, and the run costs four times what the ceiling said. A cancel
button stops the outer coroutine and leaves six tool calls in flight.

A `RunContext` is the single object every layer takes and every layer
respects. Cancel it and every hop stops. Charge it and the spend shows up
in the meter for the whole tree. Scope it and memory, caches and quotas
partition correctly — automatically, because none of them derive the key
themselves.

## The smallest example

```python
import asyncio

from agentkit import Budget, ChatRequest, Message, MeterExceeded, Usage
from agentkit.middlewares import meter
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    ctx = make_test_ctx(
        invoker=Invoker(llm=FakeLLM("ok", usage=Usage(100, 50, 0.02)), chat_middleware=[meter()]),
        budget=Budget(max_cost_usd="0.03"),
    )
    request = ChatRequest(messages=[Message("user", "hello")], model="fake-model")

    await ctx.invoker.chat(request, ctx)
    print(ctx.budget.spent(), ctx.budget.usage.total_tokens)   # 0.020000 150

    await ctx.invoker.chat(request, ctx)                       # 0.04 > 0.03
    print("never reached")


try:
    asyncio.run(main())
except MeterExceeded as exc:
    print("stopped:", exc)                 # stopped: cost $0.04 > $0.03
```

Nothing in that example told the second call about the first. The
`Budget` on the context is what remembers, and the `meter()` middleware
is what reads it.

## How it works — the mental model

A `RunContext` holds three kinds of thing, and it is worth keeping them
separate in your head:

- **Identity** — `correlation_id`, `scope`, `depth`. Answers *whose
  request is this, and how deep in the tree am I?*
- **Meters** — `budget` (always present) plus any extra meters such as a
  tenant `Quota`. Answers *what is this allowed to consume?*
- **Services** — the process-shared collaborators: `invoker`, `store`,
  `checkpointer`, `vector`, `trace`, `observer`, `metrics`, `asker`.
  Answers *who can I reach?*

Cross-cutting behaviour is deliberately **not** a field here. Tracing,
retry, metering and caching live in the invoker's middleware chain, so
changing them is editing a list rather than threading a new argument.

A nested unit of work gets `ctx.child()`: depth + 1, with budget,
services, meters and the cancel token **shared by reference**. That
sharing is the whole design — a child cannot start a fresh budget, and
cancelling a parent cancels its subtree.

```python
import asyncio

from agentkit import CancellationToken, Cancelled
from agentkit.testing import make_test_ctx


async def main() -> None:
    cancel = CancellationToken()
    ctx = make_test_ctx(cancel=cancel)
    child = ctx.child()          # depth+1, same budget / cancel / services

    print(ctx.depth, child.depth, child.cancel is ctx.cancel)   # 0 1 True
    print(ctx.semaphore() is child.semaphore())                 # False — one pool per level

    cancel.cancel()              # a parent cancels its whole subtree
    try:
        child.check_cancelled()
    except Cancelled as exc:
        print("child stopped:", exc)


asyncio.run(main())
```

## `RunContext`

| Field | What it is for |
|---|---|
| `correlation_id` | The run id that ties trace, meter, checkpoint and log events together. |
| `scope` | Tenant / domain axis. Partitions caches, quotas and scoped memory. |
| `budget` | The run's always-on meter, and the tree's depth / concurrency authority. |
| `services` | The wired-in collaborators (see below). |
| `meters` | Extra meters beyond the budget — typically a per-tenant `Quota`. |
| `depth` | How deep in the agent tree this context sits. |
| `cancel` | The cooperative `CancellationToken`, shared across the tree. |
| `autonomy` | `"auto"` / `"gated"` / `"manual"` — the run-wide human-in-the-loop tier. |
| `actor_budget` | Opt-in per-agent slice of the run budget (see [Agents](agents.md)). |
| `signal_channel` | Opt-in coordinator inbox children fan signals up to. |

Convenience properties read straight through to `services`, so patterns
write `ctx.invoker`, `ctx.trace`, `ctx.store`, `ctx.checkpointer`,
`ctx.vector`, `ctx.observer`, `ctx.asker`, `ctx.metrics`.

`ctx.all_meters` is `[budget, *meters]` — the list the `meter()`
middleware guards and charges. `ctx.emit(kind, render, payload=...)`
publishes a product-facing observation and is documented never to raise
into the run: a Redis hiccup or a slow subscriber in an observer adapter
cannot break an agent loop.

### `Services`

`Services()` with no arguments is fully usable. `trace`, `observer`,
`metrics` and `replay` default to no-op implementations and `sampler`
defaults to "record every span", so observability is never a required
dependency and code that calls `ctx.trace.span(...)` works with nothing
wired. Production overrides each with a real adapter.

Everything else defaults to `None` and is genuinely optional:
`invoker`, `store`, `checkpointer`, `vector`, `asker`.

!!! warning "Passing `None` explicitly is not the same as omitting"
    `Services(trace=None)` clobbers the no-op default, and then
    `ctx.trace.span(...)` raises. That is why `make_test_ctx` omits the
    keyword rather than forwarding a `None`.

## `Invoker`

The one entry point for model and tool calls. It wraps each request in a
`Call` envelope and drives it through the composed chain to a terminal
that calls the actual seam.

```python
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM

invoker = Invoker(
    llm=FakeLLM("ok"),
    chat_middleware=[tracing(), meter(), retry()],
    tool_middleware=[tracing(), meter()],
)
print(len(invoker.chat_middleware), len(invoker.tool_middleware))   # 3 2
```

- `stream(request, ctx)` is the primitive — an async iterator of `Delta`s.
- `chat(request, ctx)` is `collect(stream(...))`.
- `invoke_tool(request, ctx)` collects the one-item tool stream.

The app builds the two chains **once**. Because every call goes through
here, tracing, metering, retry, fallback and caching are properties of a
list, not of the loop. See [Middlewares](middlewares.md).

Two details worth knowing:

- The composed lists stay addressable as `invoker.chat_middleware` and
  `invoker.tool_middleware` (as tuples, so introspection cannot mutate a
  chain that was already composed). Without that, the wiring is
  write-only, and a missing middleware becomes a silent behavioural hole
  instead of a diagnosable one — it is how `Agent` can warn you that you
  declared `output=` but left `output_coerce()` out of the chain.
- Per-call data travels in `meta`, not on the context. Smuggling an
  output adapter via `ctx._output_adapter` would be concurrency-unsafe:
  two agents sharing a context would stomp each other.

## Budget, Quota, Meter, Charge

`Budget` is the **run-scoped** ceiling. `Quota` is the **multi-run,
scope-partitioned** ceiling — per tenant, per org. `Meter` is the
protocol both satisfy: `guard(call)` before the work, `charge(call,
usage)` after it. `meter()` is the middleware that drives them.

A `Budget` also carries the tree's structural limits: `max_depth`
(default 4) and `max_concurrency` (default 8).

The two enforce at different moments, and that is deliberate:
`Quota` refuses on `guard`, before the work, because a noisy neighbour
should not get to spend first. `Budget` enforces on `charge`, after —
see the overrun note below.

### Money is `Decimal`, and the mirror is not

Binary floating point cannot represent `0.01`, so a float ledger cannot
be reconciled to the cent.

- `budget.spent()` — exact `Decimal`. **This is the authority.**
- `budget.spent_cents()` — exact, quantized for invoicing.
- `budget.remaining()` — exact headroom, clamped at zero; `None` when no
  ceiling is set.
- `budget.spent_usd` — a float **mirror**, re-derived from the exact
  ledger after every charge so the two cannot drift. For display.

`max_cost_usd` accepts a `Decimal`, `float`, `int` or `str` and is
normalised to an exact `Decimal`. A ceiling with more decimal places than
the money scale raises `MoneyPrecisionError` **at construction** rather
than being silently rounded — a ceiling is the operator's stated intent,
so quietly rounding it is wrong. A *charge* is a measurement, so it is
quantized rather than refused: a custom pricing callable returning full
float precision must not abort a run mid-flight.

The whole `Usage` accumulates, not just a cost scalar. `budget.usage`
carries input, output, cache-read and cache-write tokens for the entire
agent tree, because the budget is shared by reference across
`ctx.child()`.

### Exhaustion can be a verdict instead of an exception

`charge()` returns a `Charge` — a value, not an exception — carrying
`ok`, `reason`, exact `spent`, `remaining`, `calls`, and the cumulative
`usage`. What happens on a crossed ceiling is chosen by `on_exceeded`:

```python
import asyncio

from agentkit import Budget, Usage


async def main() -> None:
    budget = Budget(max_cost_usd="0.03", on_exceeded="stop")
    verdict = await budget.charge(None, Usage(100, 50, 0.05))
    print(verdict.ok, verdict.reason)              # False cost $0.05 > $0.03
    print(verdict.remaining, verdict.usage.total_tokens)   # 0 150
    print(budget.exhausted())                      # True


asyncio.run(main())
```

`"raise"` stays the default deliberately. Flipping it would silently
change the control flow of every existing wiring: a run that used to
abort would continue past its ceiling in any caller that ignores the
return value, which is worse than the problem being fixed. Callers opt
into recoverability.

`"stop"` is what lets a tool-loop run write a checkpoint **before** it
stops, ending with `stop_reason="budget_exhausted"` and a resumable run
rather than a dead one. `(await budget.charge(...)).raise_if_exceeded()`
converts the verdict back to an exception at one specific call site, and
`budget.exhausted()` is the cheap synchronous read for a loop that wants
to check between units of work.

!!! warning "A budget is overrun by at most one call"
    `spent > ceiling` is evaluated **after** the work runs; there is no
    pre-flight estimate. Set the ceiling below your true limit, and use
    `on_exceeded="stop"` to make the overshoot recoverable rather than
    fatal.

Two exception types are easy to confuse:

- `MeterExceeded` (`agentkit.runtime`) — a `Budget` or `Quota` ceiling.
  Also raised by `ctx.child()` when the next depth would exceed
  `budget.max_depth`.
- `BudgetExhausted` (`agentkit.agents`) — a per-actor `ActorBudget`
  slice. Different scope, different type. See [Agents](agents.md).

Worked wiring: [Cap spend with Budget and Quota](../recipes/spend-budget-and-quota.md).

## Concurrency: the permit pool is per LEVEL

`ctx.semaphore()` returns the pool for **this context's depth**, not one
pool for the whole tree.

A single tree-wide semaphore deadlocks nested fan-out. A parent's fan-out
holds its permits for the entire duration of each child run, so an inner
fan-out draws from a pool its own ancestors have already drained. At
`max_concurrency=2`, an agent dispatching two `as_tool` sub-agents that
each dispatch their own tools hung forever: the two outer permits are
held until the inner runs finish, and the inner runs cannot start without
a permit. Deeper trees hit the same wall at any cap.

Keying on depth breaks the cycle structurally. An ancestor at depth *d*
can only ever hold permits from pool *d*, and its children draw from pool
*d+1*, so no acquisition can wait on a permit held by its own ancestor.
Every nesting boundary in the framework goes through `ctx.child()` —
`as_tool`, `run_agents`, the coordinator policies — so depth genuinely
increments at each level.

The trade is honest: the bound is `max_concurrency` **per level**, so
worst-case in-flight work is `max_concurrency * (max_depth + 1)`. Set
`max_concurrency` with that in mind. A single tree-wide cap cannot be
both deadlock-free and respected by nested acquisition — a re-entrant
permit would let one level's fan-out multiply without limit, which is a
weaker bound, not a stronger one.

## `Services.asker` — the seam that turns a suspend into a park

`Asker` (from `agentkit.agents.control.elicitation`) is the
human-in-the-loop transport. When it is set, a cognition hitting a gated
decision **parks**: it awaits the person from inside its own coroutine,
so live, unserialisable state survives. When it is unset, the classic
checkpoint-and-resume path runs unchanged.

The runtime never branches on transport. Terminal, HTTP, queue and Slack
are all the same to it — implementing `async def ask` is the whole
integration. See
[Elicit a value from a human](../recipes/elicit-a-value-from-a-human.md).

## `EventBus`

In-process pub/sub for stream-scoped events, generic over the event type
(`EventBus[MyEvent]`). The mechanism is framework-level; what counts as
an event is yours.

Subscribers receive a `VersionedEvent[E]` — your event plus a monotonic
`version` the bus stamps at publish time. The version is bus bookkeeping,
not part of your payload: it exists so a subscriber can dedupe, and so a
reconnecting one can ask for `from_version=K` and replay the ring from
there. Unwrap `.event` before sending anything onward; the version does
not cross a wire on its own.

```python
import asyncio

from agentkit import EventBus


async def main() -> None:
    bus: EventBus[str] = EventBus()
    await bus.publish("run-1", "started")
    await bus.publish("run-1", "tool: search")

    async def read() -> None:
        async for ev in bus.subscribe("run-1", name="ui", from_version=1):
            print(ev.version, ev.event)
            if ev.event == "done":
                break

    reader = asyncio.create_task(read())
    await asyncio.sleep(0)
    await bus.publish("run-1", "done")
    await reader
    await bus.shutdown()


asyncio.run(main())
```

Three properties shape how you use it:

- **Backpressure is per subscriber.** Queues are bounded
  (`queue_size`, default 256). When one overflows, *that subscriber's*
  event is dropped and logged with its `name`; other subscribers and the
  publisher continue. A slow consumer never stalls the run.
- **Replay is a ring, not an event store.** Each stream keeps the most
  recent `ring_size` events (default 1024), and a subscriber may ask for
  `from_version=K`. Beyond the ring's window, reconstruct from durable
  storage — this is pragmatic decoupling, not event sourcing.
- **Subscribe and replay do not race.** Version stamping, log append,
  live delivery and the replay snapshot happen under one per-channel
  lock, so a new subscriber can neither miss nor duplicate an event
  published while it was attaching. Replay yields happen outside the
  lock, so a slow subscriber does not back-pressure publishers.

`publish` to a closed stream re-opens it — a write is an explicit
statement that the id is live again. `subscribe` never does, so a late
subscriber terminates cleanly instead of parking forever on a sentinel
nobody will send.

The public surface is the upgrade path: swap the bus for Redis Streams,
NATS or Kafka behind the same `publish` / `subscribe` / `close_stream`
interface and no call site changes.

## `NullCtx`

A context that satisfies the structural `Ctx` protocol and does nothing:
spans record nothing, `store` and `checkpointer` are `None`,
`check_cancelled()` is a no-op, `child()` returns itself, and `invoker`
raises if you touch it.

It is for callers that use a *subset* of the context surface — a
single-shot planner that only needs `RequestBuilder.build`'s prompt
assembly, for instance. It is explicitly **not** for production use
inside the invoker pipeline: there is no budgeting, no tracing, no
cancellation. For tests that want a real context with fakes behind it,
use `make_test_ctx()` from `agentkit.testing` instead.

## What bites people

- **Never fabricate a fresh `RunContext` inside a middleware or tool.**
  Forward the one you were given, or derive one with `ctx.child()`.
  A new context means a new budget and a detached cancel token.
- **Cancellation is cooperative.** It only works at `await` points that
  can reach `cancel.raise_if_cancelled()`. Blocking primitives on the hot
  path defeat it, and so does swallowing `CancelledError`.
- **`ctx.child()` can raise.** Exceeding `budget.max_depth` (default 4)
  raises `MeterExceeded` at the moment the child is created.
- **`budget.max_cost_usd` can be raised mid-run.** The ceiling is
  re-derived when the field is assigned, which is what lets an operator
  lift a ceiling and resume. The re-derivation is non-strict, because
  raising `MoneyPrecisionError` from inside `charge()` would be exactly
  the unrecoverable abort this design removes.
- **Don't sum `usage.cost_usd` for accounting.** It re-rounds on every
  addition. Read `budget.spent()`. See
  [Kernel › value types](kernel.md#usagecost_usd-is-an-approximation-budget-keeps-the-ledger).

!!! abstract "Where this fits in the four themes"
    This page covers the **Control** theme's primitives (`Budget`,
    `Quota`, `MeterExceeded`, `Autonomy`, `CancellationToken`) and
    threads the **State** theme — `RunContext` is what carries
    per-request state through every hop. The `Invoker`'s dispatch is the
    entry point into the **Behaviour** theme's chain. See the four-theme
    grid on the [landing page](../index.md).

## Related

- [Kernel](kernel.md) — the value types and ports this frame carries.
- [Middlewares](middlewares.md) — what the `Invoker` actually runs.
- [Agents](agents.md) — the loops that consume a `RunContext`.
- [Observability](observability.md) — what the trace, observer and metrics seams do with what they are handed.
- [Testing](testing.md) — building a real `RunContext` in a unit test.
- [API › runtime](../api-reference/runtime.md) — the generated reference.
