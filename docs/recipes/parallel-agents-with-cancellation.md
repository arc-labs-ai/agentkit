# How do I run agents in parallel and cancel all of them if one fails?

## When you'd want this

Fan out a Planner's sub-questions across N Researchers. Race two
implementations of the same skill and keep the first. Kick off a batch
of independent enrichment tasks. In each case you want two properties
the raw `asyncio.gather` doesn't give you:

- **Sibling cancellation on failure.** If one Researcher blows up, the
  others should stop; you don't want to keep paying to finish work
  you're about to throw away.
- **A shared budget and cancel token.** Cost accrues on the parent's
  `Budget`; an external cancel (operator hit stop) propagates to every
  child at every safe point.

`run_agents(...)` is the primitive. It builds one `ctx.child()` per
pair (depth+1, sharing `budget` / `services` / `cancel` by reference)
and calls `gather_bounded(...)` under the parent's semaphore.

## Working code

```python
import asyncio

from agentkit import Agent, CancellationToken, run_agents
from agentkit.testing import FakeLLM, make_test_ctx


async def demo_sibling_cancel() -> None:
    """One failing agent cancels the rest — TaskGroup semantics."""
    llm = FakeLLM("ok", fail_times=1, fail_exc=RuntimeError("scripted failure"))
    ctx = make_test_ctx(llm=llm)
    a = Agent(name="a", model="m", prompt="brief")
    b = Agent(name="b", model="m", prompt="brief")
    try:
        await run_agents([(a, "topic-a"), (b, "topic-b")], ctx)
    except* RuntimeError as eg:
        for exc in eg.exceptions:
            print(f"[sibling-cancel] caught: {exc}")


async def demo_best_effort() -> None:
    """`best_effort=True` isolates failures into Failure objects."""
    llm = FakeLLM("ok", fail_times=1, fail_exc=RuntimeError("scripted failure"))
    ctx = make_test_ctx(llm=llm)
    a = Agent(name="a", model="m", prompt="brief")
    b = Agent(name="b", model="m", prompt="brief")
    results = await run_agents([(a, "topic-a"), (b, "topic-b")], ctx, best_effort=True)
    for r in results:
        print(f"[best-effort] {type(r).__name__}: {r}")


async def demo_external_cancel() -> None:
    """External signal trips the token; the agent unwinds cooperatively."""
    from agentkit.kernel.concurrency import Cancelled

    ctx = make_test_ctx(llm=FakeLLM("ok"), cancel=CancellationToken())

    async def cancel_after(delay: float) -> None:
        await asyncio.sleep(delay)
        assert ctx.cancel is not None
        ctx.cancel.cancel()

    async def long_running() -> None:
        for _ in range(20):
            await asyncio.sleep(0.02)
            ctx.check_cancelled()

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(cancel_after(0.05))
            tg.create_task(long_running())
    except* Cancelled as eg:
        print(f"[external-cancel] observed: {[type(e).__name__ for e in eg.exceptions]}")


async def main() -> None:
    await demo_sibling_cancel()
    await demo_best_effort()
    await demo_external_cancel()


asyncio.run(main())
```

## How it works

**`run_agents(pairs, ctx)`** — the default mode. Builds per-child
contexts, runs each `agent.run(task, child_ctx)` inside an
`asyncio.TaskGroup` bounded by `ctx.semaphore()`. TaskGroup semantics
give sibling-cancellation for free: the first failure raises,
outstanding tasks are cancelled, and the raised exceptions are grouped
into an `ExceptionGroup` you catch with `except*`.

**`run_agents(pairs, ctx, best_effort=True)`** — for batches where one
failure must NOT sink the others. Each slot in the returned list is
either an `AgentResult` or a `Failure` (from
`agentkit.kernel.errors`). `Failure` carries the exception on
`.cause`, the classified category on `.category`, and a `source` string
naming the slot — so a caller can retry, route around, or escalate
uniformly instead of guessing what a raw exception meant.

**`CancellationToken`** is cooperative — a well-behaved async loop
checks `ctx.check_cancelled()` at every safe point (loop top, between
steps, between tool calls). Cancelling the token doesn't kill running
coroutines; it makes the next check raise `Cancelled`. `run_agents`
shares the token by reference across all children, so cancelling the
parent cancels the subtree.

## Gotchas

- **`ExceptionGroup` isn't a plain exception.** Use `except*` (Python
  3.11+, agentkit requires 3.12) or iterate `eg.exceptions` inside a
  plain `except ExceptionGroup`.
- **Cooperative cancel means points matter.** A synchronous CPU-bound
  loop inside a tool never checks the token. Yield an `await
  asyncio.sleep(0)` or check `ctx.check_cancelled()` on your own
  iteration boundaries. Blocking I/O belongs on a `ProcessPoolExecutor`
  or a threadpool so the event loop stays reactive.
- **`Budget.max_concurrency` bounds fan-out, not `run_agents`.** The
  semaphore lives on `Budget` and defaults to 8; if you fan out 100
  children they'll still run 8 at a time. Increase
  `Budget(max_concurrency=...)` if you need more.
- **`best_effort=True` still charges the shared budget.** A
  `Failure`-in-a-slot doesn't refund the cost of the LLM calls that
  raced to error.

## Related

- [Concepts · Agents](../concepts/agents.md) — where `run_agents`
  fits alongside `Workflow` and the coordinator cognitions.
- [Cap spend with Budget and Quota](spend-budget-and-quota.md) — the
  ceiling that stops a runaway fan-out.
- `agentkit.kernel.concurrency` module docstring — the full concurrency
  surface (`gather_bounded`, `gather_best_effort`, `run_sync`).
