# agentkit

A low-level, async-first framework for building AI agents in Python.

<p>
  <code>Python 3.12+</code> ·
  <code>Apache-2.0</code> ·
  <code>zero runtime dependencies in the core</code> ·
  <code>mypy --strict</code>
</p>

## What is agentkit

agentkit is a **framework**, not an agent library. It gives you the
primitives to build your own agent — an `Agent` class, a `Cognition`
strategy, a `RunContext` that threads services + budgets + cancel
through every call, a middleware chain you compose yourself, and typed
value types the whole stack shares. What it does with an LLM, what
tools it exposes, what it stores, when it stops — every one of those
decisions is yours.

It is not batteries-included (no built-in personas, no bundled prompts,
no opinionated stack pushing you toward one vendor). It is not a
chain-builder or a graph DSL either. If you want scaffolding that hides
the loop, use one of those. agentkit hands you the loop and gets out
of the way. Reach for it when you want an agent runtime whose seams are
visible, whose types survive `mypy --strict`, whose core has **zero
runtime dependencies**, and whose control plane (cancel, budget, retry,
human-in-the-loop suspend/resume) is first-class instead of tacked on.

## The five layers

<div class="grid cards" markdown>

-   __Kernel__

    ---

    Opinion-free primitives: value types (`Message`, `ToolCall`,
    `LLMResult`, `Usage`), the `Protocol` for every seam
    (`LLMPort`, `StorePort`, `VectorPort`, …), the middleware contract,
    and the small library of concurrency + resilience helpers.

    [:octicons-arrow-right-24: Learn more](concepts/kernel.md)

-   __Runtime__

    ---

    The per-request universe: `RunContext` (identity + services +
    cancel + budget), `Invoker` (walks the middleware chain to a
    terminal LLM/tool call), `Budget` and `Quota` (spend meters that
    halt runs cleanly on overrun).

    [:octicons-arrow-right-24: Learn more](concepts/runtime.md)

-   __Capabilities__

    ---

    Optional cross-cutting collaborators: `RequestBuilder` (prompt +
    grounding), `Compactor` (four strategies), `Guardrail`,
    `Checkpointer` (durable suspend/resume), `Evaluator`, and
    `SchemaAdapter` for structured outputs.

    [:octicons-arrow-right-24: Learn more](concepts/capabilities.md)

</div>

Above these sit **Agents** (`Agent`, `Cognition`, `Workflow`, plus the
control primitives that make multi-agent flows safe) and
**Middlewares** (`tracing`, `retry`, `fallback`, `memoize`,
`output_coerce`, `meter`, `compaction`, `egress`, `audit`, `security`).
Adapters (`LLM`, `Store`, `Vector`, `Checkpoint`, `Observer`) sit
behind opt-in extras.

## When to reach for agentkit

Frameworks make different bets. Here is an honest one — pick the tool
whose bet matches yours.

| Tool                        | Right when                                                              | Not this                                                              |
|-----------------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **agentkit**                | You want typed seams, an explicit loop, cancel/budget/HITL first-class. | You want the framework to decide what your agent does.                |
| **LangGraph**               | You want a graph DSL that draws the flow for you.                       | You want to keep the loop as plain Python.                            |
| **LangChain**               | You want a large catalog of prebuilt integrations glued together.       | You care about a minimal, typed core with no vendor gravity.          |
| **Instructor**              | You only need structured outputs on top of a single provider SDK.       | You need tools, multi-turn loops, cancel, budget, resume.             |
| **Vanilla asyncio + SDK**   | The whole thing is one file and one provider.                           | The code has to survive real production concerns.                     |

If you're prototyping in a notebook and never leaving, anything works.
agentkit's benefits — typed seams, cancel/budget, checkpointing,
`Suspend`/`resume` — only pay off when the code has to keep running.

## Quickstart

Zero-dep, no API keys. `FakeLLM` and `make_test_ctx` from
`agentkit.testing` build a full `RunContext` with only the LLM faked:

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM("42"))
    agent = Agent(
        name="answerer",
        model="gpt-4o-mini",
        prompt="Answer the question in as few words as possible.",
    )

    result = await agent.run("what is 6 * 7?", ctx)
    print(result.output)                              # -> "42"
    print(f"cost: ${result.usage.cost_usd:.4f}")      # -> cost: $0.0001


asyncio.run(main())
```

Pointed at a real provider, the same code loses two lines and gains an
API key:

```python
from agentkit import claude

async with claude(api_key="sk-...", model="claude-sonnet-4-6") as chat:
    result = await chat("what is 6 * 7?", system="Answer briefly.")
    print(result.content)
```

For a step-by-step walkthrough that builds a research-briefing agent
with tools, streaming, a budget, and human-in-the-loop approval, work
through the [15-minute tutorial](tutorial.md).

## Where to go next

<div class="grid cards" markdown>

-   __Tutorial__

    ---

    Fifteen minutes, five steps, one runnable script per step —
    from a single chat call to a gated tool the human has to approve.

    [:octicons-arrow-right-24: Start the tutorial](tutorial.md)

-   __Concepts__

    ---

    The mental model of each primitive: `Kernel`, `Runtime`,
    `Capabilities`, `Agents`, `Middlewares`, `Prompts`. Read in order
    or jump to what you need.

    [:octicons-arrow-right-24: Read the concepts](concepts/kernel.md)

-   __Recipes__

    ---

    Focused answers to "how do I X?": human-in-the-loop, resume after
    a crash, cap spend, parallel agents with cancel, custom middleware,
    OpenTelemetry.

    [:octicons-arrow-right-24: Browse the recipes](recipes/index.md)

-   __Examples__

    ---

    Three self-contained scripts in the repo — single agent, streaming
    with tools, a hand-composed middleware chain. All run against
    `FakeLLM`, no API keys.

    [:octicons-arrow-right-24: See the examples](examples.md)

</div>
