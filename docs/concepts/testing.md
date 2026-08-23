# Testing

`agentkit.testing` is how you run an agent in a unit test — no API key,
no network, no wall-clock waiting, deterministic every time.

## The problem it solves

Agent code is mostly *wiring*: which middleware sits where, what the
cognition does when a tool returns an error, whether the budget stops
the run before it stops your invoice. None of that needs a real model to
verify, and all of it breaks silently when it is only exercised in
staging.

The trap is that a test which *looks* like it covers the wiring often
covers nothing at all. A caching test with no cache, a budget assertion
with no meter, a scripted tool loop that ran out of script and quietly
repeated its last turn — all three used to pass. The third one is now an
error rather than a trap (see *A script that runs out raises*), because
it was the harness hiding the very thing it was built to catch. This page
shows the doubles and then the traps, because the traps are the part that
has actually cost this repo time.

## The smallest thing that works

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM("The cert expired on 2026-08-01."))
    result = await Agent("analyst", model="fake-model").run(
        "Why did the deploy fail?", ctx
    )
    print(result.output)                     # The cert expired on 2026-08-01.
    print(result.usage.input_tokens)         # 10


asyncio.run(main())
```

Two pieces, and the distinction between them matters:

- **`make_test_ctx(...)` is not a fake.** It builds a *real*
  `RunContext` with real `Services` and a real `Invoker`. Your code
  under test runs the production path.
- **`FakeLLM` and friends are the doubles.** They sit at the port
  boundary — the seam where agentkit meets something it does not own.

That is the whole reason an agentkit app is testable without a network:
every external system is behind a [Port](adapters.md), and every port
has an offline implementation.

## `make_test_ctx` — the one context factory

It replaces the per-file `_ctx(llm)` helpers that otherwise breed across
a test suite. Every knob is a keyword argument and everything you leave
out gets a defensive default, so the no-argument form is usable for
trivial tests.

```python
import asyncio
from decimal import Decimal

from agentkit import Agent
from agentkit.kernel.types import Scope
from agentkit.runtime import Budget
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    ctx = make_test_ctx(
        llm=FakeLLM("ok"),
        scope=Scope(org_id=7, domain_id=2),
        budget=Budget(max_cost_usd=Decimal("0.10"), max_calls=3),
        correlation_id="run-42",
        autonomy="manual",
    )
    print(ctx.correlation_id, ctx.autonomy, ctx.scope.key())
    # run-42 manual org7:dom2

    await Agent("a", model="fake-model").run("go", ctx)


asyncio.run(main())
```

| Knob | What it wires |
| --- | --- |
| `llm=` | wrapped into `Invoker(llm=..., chat_middleware=..., tool_middleware=...)` |
| `invoker=` | a fully-built `Invoker`, when you need to control the chain yourself |
| `scope=` | tenant isolation. `Scope()` (zero-tenant) is the default |
| `budget=` | a `Budget`; see the meter trap below |
| `store=` / `vector=` / `checkpointer=` | the durable seams |
| `asker=` | the human-in-the-loop transport |
| `observer=` / `trace=` | observation and tracing sinks |
| `cancel=` | a `CancellationToken` |
| `chat_middleware=` / `tool_middleware=` | the two chains |
| `meters=` | extra meters beyond the budget (e.g. a tenant `Quota`) |
| `correlation_id=` / `autonomy=` | run identity and the autonomy level |

Pass `llm=` **or** `invoker=`, not both. Leave both unset and you get a
`RunContext` with `services.invoker = None`, which is exactly right for
capability tests (`RequestBuilder`, `Compactor`) that never invoke a
model.

!!! note "Why `observer` / `trace` default to `None` here"

    `make_test_ctx` *omits* them from the `Services(...)` call when
    they are unset rather than forwarding `None`. Passing `None`
    explicitly would clobber `Services`' own `NoopObserver` /
    `NoopTrace` factories and break any code that calls
    `ctx.trace.span(...)` or `ctx.observer.emit(...)`.

## `FakeLLM` — three forms

### A string, or a dict of substring matches

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    llm = FakeLLM({"weather": "It is raining.", "price": "$15/Mtok"})
    ctx = make_test_ctx(llm=llm)
    agent = Agent("a", model="fake-model")

    print(repr((await agent.run("what is the weather", ctx)).output))
    # 'It is raining.'
    print(repr((await agent.run("something else entirely", ctx)).output))
    # '{}'   <- the trap: an unmatched key is not an error


asyncio.run(main())
```

Keys are matched as **substrings of the user text**, first match wins.
An unmatched query returns the literal `"{}"` — chosen so a
structured-output test gets parseable JSON rather than a crash, but it
means a typo in your key produces a passing test that asserted on
nothing.

### A callable

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    seen: list[str] = []

    def responder(*, system: str, user: str, model: str) -> str:
        seen.append(user)
        return f"echo: {user}"

    ctx = make_test_ctx(llm=FakeLLM(responder))
    print((await Agent("a", model="fake-model").run("ping", ctx)).output)
    # echo: ping
    print(seen)     # ['ping']


asyncio.run(main())
```

The callable is invoked with keyword arguments `system`, `user` and
`model`, so it doubles as an assertion point on what the agent actually
sent.

### `FakeLLM.script([...])` — a multi-step tool loop

```python
import asyncio

from agentkit import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import ToolCall
from agentkit.testing import FakeLLM, FakeTool, Turn, make_test_ctx


async def main() -> None:
    tool = FakeTool(name="lookup_cert", responder={"notAfter": "2026-08-01"})
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "lookup_cert", {"host": "example.com"}),)),
            Turn(content="The certificate expired on 2026-08-01."),
        ]
    )
    ctx = make_test_ctx(llm=llm)
    agent = Agent("sre", model="fake-model", cognition=ReActCognition(tools=[tool]))

    result = await agent.run("Why did the deploy fail?", ctx)

    print(result.output)     # The certificate expired on 2026-08-01.
    print(tool.calls)        # [{'host': 'example.com'}]
    print(llm.calls)         # 2


asyncio.run(main())
```

A `Turn` is `content`, `tool_calls` and an optional `usage`. A turn
carrying tool calls sets `finish_reason="tool_calls"`; a turn with only
content sets `"stop"`. That is what drives a `ReActCognition` through a
real loop iteration.

A script is a finite claim about how many turns the run takes. Asking for
one more raises `ScriptExhausted`; pass `repeat_last=True` when a test
deliberately drives an unbounded loop. See *A script that runs out
raises* below.

`FakeLLM.calls` counts invocations, and every double in this package
records something — that is the assertion surface.

### Failure injection

```python
import asyncio

from agentkit import Agent
from agentkit.middlewares import retry
from agentkit.testing import FakeClock, FakeLLM, make_test_ctx


async def main() -> None:
    clock = FakeClock()
    llm = FakeLLM("recovered", fail_times=2, fail_exc=TimeoutError("upstream timeout"))
    ctx = make_test_ctx(
        llm=llm, chat_middleware=[retry(max_attempts=3, sleep=clock.sleep)]
    )

    result = await Agent("a", model="fake-model").run("go", ctx)

    print(result.output)                    # recovered
    print("provider calls:", llm.calls)     # 3
    print("backoffs waited:", len(clock.sleeps))   # 2
    print("wall clock advanced:", clock.now() > 0)  # True


asyncio.run(main())
```

`fail_times=n` raises `fail_exc` on the first `n` calls. It raises
**before the first delta**, which makes it a pre-stream failure — the
only kind `retry()` can re-invoke, because you cannot un-send tokens
once streaming has committed.

`FakeClock` makes the backoff free: `sleep(s)` records the duration and
advances virtual time instead of blocking. `clock.sleeps` is the list of
waits; `clock.advance(s)` moves time without recording a sleep, for
simulating external time passing between steps.

`delay=` adds a real `asyncio.sleep` before each call — use it for
cancellation and timeout tests, where the wait is the thing under test.

## The rest of the doubles

| Double | Port | Records |
| --- | --- | --- |
| `FakeLLM` | `LLMPort` | `.calls` |
| `FakeTool` | `ToolPort` | `.calls` — a copy of each `args` dict |
| `FakeClock` | `ClockPort` | `.sleeps` |
| `FakeSearch` | `SearchPort` | `.calls` |
| `FakeFetch` | `FetchPort` | `.calls` |
| `FakeMemory` | `MemorySource` | `.queries`, `.writes` |
| `FakeGrounder` | `RequestBuilder` grounder | `.calls` — the `(ctx, task)` tuples |
| `FakeCompactor` | `Compactor` | `.called` |
| `RecordingTracer` | `TracePort` | `.spans` |
| `FakeCtx` | `Ctx` | `.spans` |

```python
import asyncio

from agentkit.kernel.ports import FetchResponse, SearchHit
from agentkit.memory.base import MemoryItem
from agentkit.testing import FakeCtx, FakeFetch, FakeMemory, FakeSearch


async def main() -> None:
    search = FakeSearch(
        {"cert": [SearchHit("https://ex/1", "Cert rotation", "rotate every 90d")]}
    )
    print(len(await search.search("how do I rotate a cert", k=5)))   # 1
    print(await search.search("unrelated"), search.calls)            # [] 2

    fetch = FakeFetch(
        {
            "https://ex/1": FetchResponse(
                "https://ex/1", 200, {"content-type": "text/html"},
                "<p>hi</p>", "text/html", 1.7e9,
            )
        }
    )
    print((await fetch.fetch("https://ex/1")).body)    # <p>hi</p>
    try:
        await fetch.fetch("https://ex/2")
    except KeyError as exc:
        print("KeyError:", exc)      # no silent 404s in tests

    mem = FakeMemory(responses={"cert": [MemoryItem("rotate every 90d", "runbook")]})
    print(len(await mem.query("cert", k=2, ctx=FakeCtx())))   # 1
    print(mem.queries)                                        # [('cert', 2, None)]


asyncio.run(main())
```

`FakeSearch` and `FakeMemory` return an empty list for an unmatched
query; `FakeFetch` deliberately **raises** `KeyError` instead, so a test
fails loudly rather than silently falling back to a stub page.

`FakeMemory` also has a fixture mode: pass `items=[...]` and `query`
ignores the query string entirely, returning the first `k` items — for
tests that only care that the agent received *some* items.

### Assert on spans

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, RecordingTracer, make_test_ctx


async def main() -> None:
    tracer = RecordingTracer()
    ctx = make_test_ctx(llm=FakeLLM("ok"), trace=tracer)
    await Agent("analyst", model="fake-model").run("go", ctx)
    print([(s.name, s.kind) for s in tracer.spans])
    # [('invoke_agent', 'client'), ('compose', 'compose')]


asyncio.run(main())
```

`RecordingTracer(exporter=spans.append)` fires a callable on span close
if you want a flat list. `RecordingSpan` records `.attrs` and `.events`,
so you can assert on attribute stamping (for example, that
`RequestBuilder` set `agentkit.prompt.version`).

### Assert on observations

Observations do not come from `Agent.run` itself — they are emitted by
cognitions and by your own code through `ctx.emit`. Wire an observer
adapter and read it back:

```python
import asyncio

from agentkit.adapters.observer import CollectingObserver
from agentkit.testing import make_test_ctx


async def main() -> None:
    sink = CollectingObserver()
    ctx = make_test_ctx(observer=sink)
    await ctx.emit("step", "planning", agent="analyst")
    await ctx.emit("result", "done", payload={"ok": True}, agent="analyst")
    await sink.close()
    print([o.kind for o in sink.items])     # ['step', 'result']


asyncio.run(main())
```

## Naming: `Fake*` vs `Null*` vs `Noop*`

The convention is load-bearing, because two of these are production
code:

- **`Fake*`** — test doubles. They live in `agentkit.testing` and are
  kept off the production import path (pinned by
  `tests/meta/test_public_api_surface.py`).
- **`Null*` / `Noop*`** — production-grade null objects. `NullCtx`,
  `NoopObserver`, `NoopTrace`, `NoopMetrics`, `NoopReplayStore` live in
  `runtime/` and `kernel/`. They are what makes `Services()` with no
  arguments fully usable, and they are the right thing to ship.

`FakeCtx` and `NullCtx` differ in exactly one way: `NullCtx` absorbs
operations and records nothing; `FakeCtx` records spans so a test can
assert on them. Both are valid — pick based on whether you need the
recording.

## What bites people

!!! danger "A `memoize()` test without a store proves nothing"

    This one has cost this repo time **twice**. The first thing
    `memoize()` does on every call is check for a store — `store or
    call.ctx.store` — and, finding neither, it yields straight from
    `nxt(call)` and returns. No warning, no error. The middleware is
    in the chain, the test is green, and the caching it claims to
    exercise never ran:

    ```python
    import asyncio

    from agentkit import Agent
    from agentkit.adapters.store import InMemoryStore
    from agentkit.middlewares import memoize
    from agentkit.testing import FakeLLM, make_test_ctx


    async def run_twice(ctx, llm) -> int:
        agent = Agent("analyst", model="fake-model")
        await agent.run("same question", ctx)
        await agent.run("same question", ctx)
        return llm.calls


    async def main() -> None:
        # WRONG — no store. memoize() is a no-op.
        llm = FakeLLM("cached?")
        ctx = make_test_ctx(llm=llm, chat_middleware=[memoize()])
        print("no store:  ", await run_twice(ctx, llm), "provider calls")
        # no store:   2 provider calls

        # RIGHT — a store, so the cache actually exists.
        llm = FakeLLM("cached?")
        ctx = make_test_ctx(llm=llm, store=InMemoryStore(), chat_middleware=[memoize()])
        print("with store:", await run_twice(ctx, llm), "provider calls")
        # with store: 1 provider calls


    asyncio.run(main())
    ```

    Always pass `store=InMemoryStore()`, and always assert on
    `llm.calls` rather than on the output — the output is identical
    either way, which is precisely why the broken version looks fine.

    The same applies to `idempotent()` and `semantic_memoize()`; the
    latter needs `vector=InMemoryVector()`.

!!! danger "A `Budget` is not charged unless `meter()` is in the chain"

    `make_test_ctx(budget=...)` installs the budget, but nothing
    charges it. The `meter()` middleware is what reads `usage` off a
    result and charges every meter in `ctx.all_meters`:

    ```python
    import asyncio
    from decimal import Decimal

    from agentkit import Agent
    from agentkit.middlewares import meter
    from agentkit.runtime import Budget
    from agentkit.testing import FakeLLM, make_test_ctx


    async def main() -> None:
        ctx = make_test_ctx(llm=FakeLLM("ok"), budget=Budget(max_cost_usd=Decimal("0.10")))
        await Agent("a", model="fake-model").run("go", ctx)
        print("unmetered:", ctx.budget.spent(), ctx.budget.calls)
        # unmetered: 0.000000 0

        ctx = make_test_ctx(
            llm=FakeLLM("ok"),
            budget=Budget(max_cost_usd=Decimal("0.10")),
            chat_middleware=[meter()],
        )
        await Agent("a", model="fake-model").run("go", ctx)
        print("metered:  ", ctx.budget.spent(), ctx.budget.calls)
        # metered:   0.000100 1


    asyncio.run(main())
    ```

    A test asserting "the budget stopped the run" without `meter()` in
    the chain is asserting that nothing happened.

!!! warning "A script that runs out raises"

    `FakeLLM.script` used to clamp its index, so once the turns were
    exhausted every subsequent call replayed the final turn — a 2-turn
    script asked for 4 turns answered `['one', 'two', 'two', 'two']`. A
    loop that should have terminated therefore did not fail; it settled
    into a stable, plausible, wrong answer and the test went green. The
    double built to catch non-termination was the thing hiding it.

    It now raises `ScriptExhausted`, naming how many turns the script had
    and which turn was asked for:

    ```python
    import asyncio

    from agentkit import Agent
    from agentkit.testing import FakeLLM, ScriptExhausted, Turn, make_test_ctx


    async def main() -> None:
        llm = FakeLLM.script([Turn(content="one"), Turn(content="two")])
        ctx = make_test_ctx(llm=llm)
        agent = Agent("a", model="fake-model")
        try:
            for _ in range(3):
                print((await agent.run("go", ctx)).output)
        except ScriptExhausted as exc:
            print(exc)  # ... has 2 turn(s), but the agent asked for turn 3 ...


    asyncio.run(main())
    ```

    `ScriptExhausted` is a `BaseException`, not an `Exception`, for the
    same reason `asyncio.CancelledError` is: the handlers between the
    fake and your test body — react reflecting bad output back to the
    model, tool failures becoming tool messages, `resilience` classifying
    a pre-stream fault as retryable — are all correct for a real provider
    fault, and all of them would swallow this one.

    When the unbounded loop is the point of the test, say so:

    ```python
    from agentkit.kernel.types import ToolCall
    from agentkit.testing import FakeLLM, Turn

    # one tool-call turn, replayed forever; max_iterations ends the run
    llm = FakeLLM.script(
        [Turn(tool_calls=(ToolCall("c1", "fetch", {}),))],
        repeat_last=True,
    )
    ```

    Only the script form has anything to exhaust. `FakeLLM("x")` and the
    dict/callable forms are rules for answering, not finite scripts, so
    they answer every call and always will.

!!! warning "One `FakeLLM` shared across runs keeps counting"

    `calls`, `_turn_idx` and the `fail_times` latch all live on the
    instance. Reusing one `FakeLLM` across two `agent.run(...)` calls
    is often what you want (that is how the memoize example above
    works), but it means `fail_times=2` fails twice *in total*, not
    twice per run. Build a fresh one per scenario when in doubt.

!!! warning "`FakeLLM` ignores `tools=` entirely"

    It replies from its script or its response map regardless of what
    tool schemas were passed. That is what makes it deterministic, and
    it also means a test cannot use `FakeLLM` to check that the right
    tools reached the provider. Assert on that with the callable form,
    or with a `RecordingTracer` and the span attributes.

!!! warning "`make_test_ctx()` gives you `invoker=None` by default"

    With neither `llm=` nor `invoker=`, `services.invoker` is `None`.
    That is correct for capability tests, and it is an
    `AttributeError` on `None` the moment something tries to run an
    agent. If a test fails there, you forgot the `llm=`.

## Related

- [Adapters](adapters.md) — the ports these doubles stand in for, and
  the in-process reference implementations (`InMemoryStore`,
  `InMemoryVector`, `InMemoryCheckpointStore`) that are production code
  rather than doubles.
- [Runtime](runtime.md) — `RunContext`, `Services`, `Budget`, `Invoker`:
  what `make_test_ctx` is actually building.
- [Middlewares](middlewares.md) — `memoize()`, `retry()`, `meter()` and
  the rest of the chain the traps above are about.
- [Context](context.md) — `WorkingContext`, which you can pass to
  `agent.run(task, ctx, context=...)` to seed a transcript.
- [API › testing](../api-reference/testing.md) — the generated
  reference.
