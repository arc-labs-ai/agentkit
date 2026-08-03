# Middlewares

**What this is.** A middleware is a thin async wrapper around a `Call`
that adds one cross-cutting concern — tracing, cost metering, retry,
memoization, output coercion, compaction, or a guardrail — and then
forwards to the next middleware in the chain. `agentkit.middlewares`
ships the standard set; you can add your own by conforming to the
kernel's `Middleware` protocol.

**Why it exists.** In a naive agent loop, every concern grows into the
loop body until it's untestable. The middleware chain moves each
concern to a single well-named file, orders them by contract, and lets
you swap or reorder them by editing a list. It's the same idea as
ASGI / Rack / Express middleware, applied to LLM and tool calls.

## The stack

The default chain the batteries-included `Chat` builds:

```
tracing  →  meter  →  retry (with breaker)
```

The full chain the framework's `Agent` typically composes for chat
calls:

```
tracing  →  meter  →  retry  →  memoize  →  compaction  →  security  →  output_coerce
```

Middlewares run **outer-to-inner** on the way in, and **inner-to-outer**
on the way back — the standard onion. `tracing` outermost so every
event is captured; `output_coerce` innermost so it sees the raw model
result.

## What ships in `agentkit.middlewares`

| Middleware         | Responsibility                                                        |
|--------------------|-----------------------------------------------------------------------|
| `tracing`          | Emit start/end spans with correlation, prompt version, tool name.     |
| `meter`            | Accumulate `Usage` and enforce `Budget` / `Quota`.                    |
| `retry`            | Provider-aware retry with jitter and optional circuit breaker.        |
| `fallback`         | Try a next LLM when the primary raises a terminal error.              |
| `memoize`          | Cache identical requests, keyed by scope-aware content hash.          |
| `idempotent`       | Deduplicate calls that carry an idempotency key.                      |
| `semantic_memoize` | Vector-similarity cache for near-duplicate requests.                  |
| `output_coerce`    | Coerce free-form output into a typed shape (`SchemaAdapter`).         |
| `compaction`       | Route through a `Compactor` if the projected input exceeds a budget.  |
| `egress`           | Gate what leaves the process (allowlists, size caps).                 |
| `audit`            | Structured audit records for every call.                              |
| `security`         | Redact / block known-secret patterns before they leave the process.   |

## Writing your own

A middleware is any callable that takes `(Call, next)` and returns
whatever `next(Call)` returns — usually an `LLMResult` for chat calls
or the tool's return value for tool calls:

```python
from agentkit.kernel import Call, LLMResult

async def latency_log(call: Call, next):
    import time
    t = time.perf_counter()
    result: LLMResult = await next(call)
    dt = time.perf_counter() - t
    call.ctx.services.trace.event("latency", {"ms": dt * 1000})
    return result
```

Add it to the chain when you build your `Invoker` or `Chat`.

## The invariants it enforces

1. **Middlewares don't mutate calls.** They forward, or they construct
   a new `Call` and forward that.
2. **Order is deterministic.** The chain is a list, not a set; there
   is no discovery step.
3. **Never swallow cancellation.** A middleware that catches broad
   exceptions must re-raise `CancelledError`.

## API

Full generated reference lives at
[API › middlewares](../api-reference/middlewares.md).
