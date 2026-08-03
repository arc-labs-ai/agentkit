# How do I write my own middleware?

## When you'd want this

Every cross-cutting concern in agentkit is a middleware — tracing,
metering, retry, caching, egress checks, audit. When you have your
own — redact a secret before it goes on the wire, tag every request
with a header, cache on a custom key, add a stopwatch — write one too.

There are two shapes, and picking the right one keeps the code
straight.

## Working code

```python
import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from agentkit import (
    BaseMiddleware,
    Call,
    ChatRequest,
    Handler,
    Message,
    MiddlewareContext,
)
from agentkit.middlewares import tracing
from agentkit.testing import FakeLLM, make_test_ctx


# ── Style 1: BaseMiddleware — for transform / guard / observe.
class Redact(BaseMiddleware):
    """Rewrite outgoing user messages to redact a secret string."""

    async def on_request(self, ctx: MiddlewareContext) -> None:
        req = ctx.request
        redacted = [
            Message(role=m.role, content=(m.content or "").replace("SECRET", "[REDACTED]"))
            for m in req.messages
        ]
        # `ctx.request = ...` is the writable seam — mutating the list
        # in place would not change the unit of work.
        ctx.request = ChatRequest(
            messages=redacted,
            model=req.model,
            tools=req.tools,
            response_format=req.response_format,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )


# ── Style 2: raw (call, next) — for resilience / caching / instrumentation.
async def stopwatch(call: Call, nxt: Handler) -> AsyncIterator[Any]:
    """Time the whole call. Only raw middleware can wrap `next` in a context
    manager or re-invoke it — retry/fallback/memoize all live at this layer."""
    started = time.perf_counter()
    async for item in nxt(call):
        yield item
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"[stopwatch] {call.kind} took {elapsed_ms:.2f}ms")


async def main() -> None:
    ctx = make_test_ctx(
        llm=FakeLLM("ok"),
        chat_middleware=[tracing(), stopwatch, Redact()],
    )
    req = ChatRequest(
        messages=[Message("user", "the code is SECRET, please echo")],
        model="gpt-4o-mini",
    )
    result = await ctx.invoker.chat(req, ctx)
    print(f"[result] {result.content!r}")


asyncio.run(main())
```

## How it works

Every call — a chat turn, a tool execution — is wrapped in a `Call`
envelope (`kind`, `request`, `ctx`, `meta`). The `Invoker` folds a
list of middlewares over a terminal handler with `chain(...)`: the
first entry is outermost, the last sits closest to the LLM. Each
middleware is an async generator over a single streaming contract —
chat calls yield `Delta`s, tool calls yield one item.

### Style 1 — `BaseMiddleware`

Override only the phases you need. Each is `async` and can be a plain
coroutine OR an async generator:

- `on_request(ctx)` — before: mutate `ctx.request`; raise `Blocked` to
  refuse.
- `on_response(ctx, result)` — after success: return / yield a
  transformed result; default passes through.
- `on_error(ctx, exc)` — on failure: return / yield a recovered value,
  else raise.

`buffers = False` (the default) streams `Delta`s through and lets you
observe the assembled result; `on_response` return is ignored.
`buffers = True` collects the stream first so `on_response` can
**transform** the result — pick this only when you have to; buffering
loses incremental streaming.

Use `BaseMiddleware` for anything that *transforms*, *guards*, or
*observes*. It cannot re-invoke or wrap `next` in a context manager.

### Style 2 — raw `(call, next)`

A plain async generator taking the `Call` and the `next` handler.
Because you drive `next(call)` yourself, you can:

- **Re-invoke** it (retry, fallback with a rewritten request).
- **Skip** it (memoize on a cache hit).
- **Wrap** it in a context manager (`tracing` holds a span open across
  the whole call).

This is where every resilience, caching, and instrumentation
middleware in agentkit lives.

### Ordering

`chain(middlewares, terminal)` folds right, so `middlewares[0]` is
outermost. A canonical chat chain:

```python
chat_middleware = [
    tracing(),        # outermost — one span covers everything below
    compaction(...),  # transform: shrink the prompt before meter sees tokens
    meter(),          # guard/charge every attempt
    fallback([...]),  # rewrite + re-invoke on hard failures
    retry(...),       # re-invoke on transient failures
]
```

Reorder or swap by editing the list. There is no hidden default chain
you have to override — the app owns the list.

## Gotchas

- **Mutating `ctx.messages` in place does nothing.** `MiddlewareContext.messages`
  returns a copy. To rewrite the transcript, assign a new `ChatRequest`
  to `ctx.request`.
- **`buffers=True` disables incremental streaming.** The whole stream
  is collected, `on_response` runs, then the transformed result is
  re-emitted as one terminal delta. If your user watches token-by-token
  output, don't buffer.
- **The two styles compose freely.** `chain([tracing(), stopwatch,
  Redact()])` mixes raw and `BaseMiddleware` in one list — the fold
  adapts `BaseMiddleware` via `.as_middleware()` internally.
- **Tool chain vs chat chain.** `Invoker` takes both
  (`chat_middleware=`, `tool_middleware=`). Retry on the chat chain
  recovers from provider blips; on the tool chain it recovers from
  tool crashes — pick where each concern belongs.

## Related

- [Concepts · Middlewares](../concepts/middlewares.md) — the mental
  model of the chain, and the shipped middlewares.
- [Example 03](https://github.com/arc-labs-ai/agentkit/blob/main/examples/03_composed_middlewares.py)
  — the canonical `tracing → retry → output_coerce` composition in a
  runnable script.
- `agentkit.kernel.middleware` module docstring — the contract, in
  code.
