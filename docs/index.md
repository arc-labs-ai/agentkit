# agentkit

A low-level, composable framework for **agentic-AI features** in Python.
Every cross-cutting concern (streaming, retries, cost, cache, guardrails,
memory, tools) is a typed `Protocol` you can inject at wire-up — not a
hidden loop that decides for you.

## The 60-second why

Most LLM libraries fuse two things: the *loop* that talks to a model, and
every *concern* that hangs off it. Change one, break the other. agentkit
splits them apart:

- **kernel** — value types + ports + middleware contract, opinion-free.
- **runtime** — `RunContext`, `Invoker`, `Budget`, `Quota`, `EventBus`.
- **middlewares** — `tracing → meter → retry → memoize → guard` on a plain chain.
- **agents** — `Agent`, `Workflow`, and a `Cognition` you can swap
  (`SingleCall`, `ReAct`, `Coordinator`).
- **capabilities** — optional collaborators: `Compactor`, `Guardrail`,
  `Checkpointer`, `Evaluator`, output-schema adapters.
- **adapters** — the concrete LLM / vector / store / observer glue.

You start with a batteries-included `Chat` and drop down as far as you
need. Nothing is stuck behind a class hierarchy.

## Quickstart

```python
import asyncio
from agentkit import claude

async def main() -> None:
    async with claude(api_key="sk-...", model="claude-sonnet-4-6") as chat:
        result = await chat(
            "Summarize the theory of general relativity in three bullets.",
            system="You are a careful physics tutor.",
        )
        print(result.content)
        print(f"cost: ${result.usage.cost_usd:.4f}")

asyncio.run(main())
```

That call already runs through the standard `tracing → meter → retry`
chain on a real `RunContext`, so you get cost accounting, structured
tracing, and provider retries for free. When you need more —
tools, an agent loop, checkpoints, guardrails — you keep the same
context and add another Protocol.

## Where to go next

- **[Getting started](getting-started.md)** — install, first agent,
  what the moving parts are.
- **Concepts** — start with the [Kernel](concepts/kernel.md) and
  [Runtime](concepts/runtime.md), then [Agents](concepts/agents.md).
- **[API reference](api-reference/agents.md)** — auto-generated from
  docstrings, one page per subpackage.
- **[Examples](examples.md)** — end-to-end demos in `examples/`.

## Design bets

- **Composition over inheritance.** No agent subclassing; new shapes are
  new Protocol implementations.
- **Async-first end to end.** Streaming is a first-class shape, not a
  callback afterthought.
- **Zero core runtime deps.** Extras (`http`, `postgres`, `redis`,
  `observability`) are opt-in seams.
- **Testable by construction.** `agentkit.testing` ships fakes for every
  Protocol; the standard test rig is `make_test_ctx()`.
