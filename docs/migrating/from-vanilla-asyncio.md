# From vanilla asyncio + provider SDK

You've written a `while` loop over a provider's chat API. The loop
appends messages, checks for tool calls, dispatches them, and either
returns or iterates. It's ~150 lines of Python and it works. This
page is for you.

## What you have now

Roughly this shape — swap your SDK, but the skeleton doesn't move:

```python
"""The hand-rolled loop most agent scripts start as."""

import asyncio
import json
import os

from openai import AsyncOpenAI  # pip install openai


client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


async def search(query: str) -> str:
    # Real code would hit an API. Return a canned result for the shape.
    return "Distributed cognition in cephalopods (science.org, 2023)"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for `query`.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


async def run(task: str) -> str:
    messages = [
        {"role": "system", "content": "You are a terse briefer. Cite every claim."},
        {"role": "user", "content": task},
    ]
    for _ in range(8):
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append({"role": "assistant",
                         "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = await search(**args)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result})
    return "iteration ceiling hit"


print(asyncio.run(run("Brief me on octopus cognition.")))
```

This works. It's small. It's yours. Then production happens.

## The same thing in agentkit

```python
"""Same behavior. Framework holds the loop, tools, cancel, budget."""

import asyncio

from agentkit import Agent, RunContext, Scope, Services, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Invoker


# Your LLMPort of choice. Batteries-included:
# from agentkit.adapters.llm import providers
# llm = providers.openai(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
# Or roll your own — LLMPort is three methods: stream (returns an
# async iterator), plus async def chat and async def complete.
llm = ...  # LLMPort


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the web for `query`. Returns bulleted hits with source and year."""
    return "Distributed cognition in cephalopods (science.org, 2023)"


async def main() -> None:
    services = Services(
        invoker=Invoker(
            llm=llm,
            chat_middleware=[tracing(), meter(), retry()],
        ),
    )
    ctx = RunContext(correlation_id="run-1", scope=Scope(), services=services)

    agent = Agent(
        name="briefer",
        model="gpt-4o-mini",
        prompt="You are a terse briefer. Cite every claim.",
        cognition=ReActCognition(tools=[search], max_iterations=8),
    )

    result = await agent.run("Brief me on octopus cognition.", ctx)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
```

Same code, minus the loop. The tool declaration is `@tool` with a
required `side_effecting=` flag; the loop is `ReActCognition`; the
middleware chain gives you tracing + cost accounting + retry for free.

## The things you were about to write next

Here's the punch list your hand-rolled loop is missing — each one is
zero or one lines to add on top of the agentkit version above.

### Cancel one run cleanly from outside

Hand the run a token; flip it from anywhere.

```python
from agentkit import CancellationToken

token = CancellationToken()
ctx = RunContext(correlation_id="run-1", scope=Scope(), services=svc, cancel=token)

# Later, from another task:
token.cancel()   # every check_cancelled() in the tree now raises Cancelled
```

Your hand-rolled loop had no cancel. The `for` loop over
`resp.choices` was uninterruptable.

### Cap the run's spend

`Budget(max_cost_usd=...)` charged under an async lock by `meter()`.
Overspend raises `MeterExceeded` and the loop unwinds cleanly.

```python
from agentkit import Budget, MeterExceeded

ctx = RunContext(
    correlation_id="run-1",
    scope=Scope(),
    services=svc,
    budget=Budget(max_cost_usd=0.10, max_calls=8),
)

try:
    await agent.run(task, ctx)
except MeterExceeded as exc:
    print(f"halted at ceiling: {exc}")
```

Your hand-rolled loop counted iterations (`for _ in range(8)`).
That's a step cap, not a spend cap — a single long tool result can
blow you past a dollar limit inside one iteration. `Budget` catches
that on the way to the next `meter()` call.

### Retry provider blips without a broken `try/except`

`retry()` middleware retries on transient errors with jitter and an
optional circuit breaker. It sits inside `meter()` so every attempt
still counts against the budget.

```python
from agentkit.middlewares import retry
from agentkit.kernel.resilience import CircuitBreaker

chat_middleware = [
    tracing(),
    meter(),
    retry(breaker=CircuitBreaker("provider.chat")),
]
```

Your hand-rolled loop had none. A 500 from the provider crashed the
whole run.

### Human approval before a side-effecting tool

`autonomy="gated"` + `side_effecting=True` on the tool + a
`Checkpointer` = the loop suspends with a `Suspended`. Your driver
approves or denies per pending tool call. Fresh process can resume.

```python
from agentkit import Suspended
from agentkit.capabilities import Checkpointer
from agentkit.adapters.checkpoint import InMemoryCheckpointStore

@tool(side_effecting=True)
async def publish(title: str) -> str:
    """Publish `title` to the team wiki. Not idempotent."""
    return f"published {title}"

ctx = RunContext(
    correlation_id="run-1",
    scope=Scope(),
    services=Services(invoker=..., checkpointer=Checkpointer(port=InMemoryCheckpointStore())),
    autonomy="gated",
)

result = await agent.run("Publish the brief.", ctx)
susp = result.evals.get("suspended")
if isinstance(susp, Suspended):
    decisions = {tc.id: "approve" for tc in susp.pending}
    result = await agent.resume(susp.run_id, decisions, ctx)
```

Your hand-rolled loop had no hook. If the model called `publish`, it
ran.

### Fan out several agents in parallel with shared budget and cancel

`run_agents(...)` runs each child under `ctx.child()`, sharing budget
and cancel by reference. Structured concurrency: the first failure
cancels the siblings.

```python
from agentkit import run_agents

results = await run_agents(
    [(researcher_agent, "sub-question-1"),
     (researcher_agent, "sub-question-2")],
    ctx,   # children share ctx.budget and ctx.cancel by reference
)
```

Your hand-rolled loop was one agent, one task, one direction. Adding
parallel meant another 40 lines and no sibling-cancel.

### Cache identical requests

`memoize()` middleware skips the LLM on a scope-partitioned content-hash
hit. Free RUM for a chat call your test suite repeats a hundred
times.

```python
from agentkit.middlewares import memoize

chat_middleware = [tracing(), meter(), retry(), memoize()]
```

### Observe every LLM call in a real trace backend

`arc-agentkit[observability]` bridges `TracePort` / `MetricsPort` to
OpenTelemetry. Every chat and tool call gets a span; every attempt
gets a histogram observation. No callback plumbing.

```python
# pip install "arc-agentkit[observability]"
from agentkit.adapters.observability import (
    otel_tracer, otel_meter, otel_exporter_otlp_http, otel_metrics_exporter_otlp_http,
)

otel_exporter_otlp_http()
otel_metrics_exporter_otlp_http(interval_ms=15_000)

services = Services(
    invoker=my_invoker,
    trace=otel_tracer(),
    metrics=otel_meter(),
)
```

### Structured output that survives model drift

`Agent(output=MyPydanticModel)` builds a `SchemaAdapter`; the schema
lands in the prompt; `output_coerce` middleware coerces the output;
parse failures repair up to `max_repairs` times.

```python
from pydantic import BaseModel
from agentkit import adapt

class Brief(BaseModel):
    summary: str
    citations: list[str]

agent = Agent(
    name="briefer",
    model="gpt-4o-mini",
    prompt="Return a brief with citations.",
    cognition=ReActCognition(tools=[search]),
    output=Brief,
    max_repairs=1,
)
result = await agent.run(task, ctx)
result.parsed  # -> Brief(summary=..., citations=[...])
```

## When to stay with vanilla asyncio

Ceremony has a cost. agentkit's benefits show up when the code has to
survive real production concerns — budget, cancel, retry, HITL,
resume, observability, multi-agent. If your script is:

- One file, one provider, one tool, one user, one run at a time.
- Not paid work — a demo or a personal automation.
- Never going to grow beyond ~200 lines.

...then agentkit is more machinery than payoff. Keep your `while`
loop. Come back when the script does something you can't lose or
overspend.

## Migration order that works

1. **Wrap the LLM.** Point an `LLMPort` at whatever provider client
   you already have. The batteries-included presets under
   `agentkit.adapters.llm.providers` are the easiest starting point.
2. **Rewrite each tool.** `@tool(side_effecting=...)` + a docstring;
   test each in isolation. Existing `async def` functions are already
   the right shape.
3. **Replace the loop.** `Agent(cognition=ReActCognition(tools=...))`
   subsumes your `for` loop, the JSON parsing, the message accumulation,
   and the max-iteration cap.
4. **Add the middleware chain.** `[tracing(), meter(), retry()]` on
   the `Invoker`. This alone gives you cost accounting and provider
   retries.
5. **Add `Budget`.** `Budget(max_cost_usd=..., max_calls=...)` on the
   `RunContext`. `meter()` enforces it.
6. **Turn on HITL when needed.** `autonomy="gated"` + a
   `Checkpointer` + handle `Suspended` in the driver.

Steps 1–3 replace your loop with equivalent behavior in fewer lines.
Steps 4–6 buy the production concerns you were about to write from
scratch.

## Related

- [Cheatsheet](../cheatsheet.md) — every primitive, tight code.
- [Tutorial](../tutorial.md) — the tool-loop walked step by step.
- [Concepts › Agents](../concepts/agents.md) — why the `Agent` /
  `Cognition` split is load-bearing.
