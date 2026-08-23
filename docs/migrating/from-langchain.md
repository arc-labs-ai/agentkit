# From LangChain

If your existing code is built on `langchain` — `LLMChain`,
`AgentExecutor`, `Tool`, `ConversationBufferMemory`, `Callbacks` —
this page is the concept-by-concept mapping to agentkit primitives,
followed by a side-by-side rewrite of a canonical tool-using agent,
and an honest read on the tradeoffs.

## Concept mapping

Each row is a straight substitution — the agentkit primitive fills
the same slot in your composition.

| LangChain                              | agentkit                                                                            | Notes                                                                                          |
|----------------------------------------|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `LLM` / `ChatModel` (base class)       | `LLMPort` (Protocol)                                                                | Structural — implement four methods, no inheritance.                                           |
| `LLMChain` / `RunnableSequence`        | `SingleCallCognition` (or your own)                                                 | The single-call regime lives in a Cognition, not a class hierarchy.                            |
| `AgentExecutor(agent, tools, ...)`     | `Agent(cognition=ReActCognition(tools=...))`                                        | Loop control moves onto `Cognition`; `Agent` holds identity + chat config.                     |
| `AgentExecutor.invoke(input)`          | `await agent.run(task, ctx)`                                                        | `task` first, `ctx` second — argument order bites everyone once.                               |
| `Tool` / `StructuredTool.from_function`| `@tool(side_effecting=..., ...)`                                                    | `side_effecting=` is REQUIRED; forgetting it raises at decoration time.                        |
| `BaseCallbackHandler`                  | `Middleware` + `ObserverPort`                                                       | Middlewares intercept calls; `ObserverPort` receives lifecycle events.                         |
| `ConversationBufferMemory`             | `WorkingContext` (transcript) + `MemorySource` (RAG)                                | Two split concerns: in-flight state vs external-reach retrieval.                               |
| `VectorStoreRetriever`                 | `VectorMemory(vector=port)`                                                         | The `VectorPort` is the swappable seam.                                                        |
| `ChatMessageHistory` / `save_context`  | `Checkpointer(port=CheckpointPort)`                                                 | Durable snapshot; powers HITL + crash-resume.                                                  |
| `RunnableWithFallbacks`                | `fallback([alt_llm])` middleware                                                    | Composition is a list of middlewares, not a `Runnable` operator.                               |
| `retry(...)` on a Runnable             | `retry(breaker=CircuitBreaker(...))` middleware                                     | Same idea, wired the same way.                                                                 |
| `RateLimiter` on a Runnable            | `Quota(max_rpm=..., max_tpm=..., max_usd=...)`                                      | Per-tenant windows, keyed by `Scope`.                                                          |
| Callbacks for cost                     | `Budget(max_cost_usd=..., max_calls=...)` + `meter()`                               | Enforced ceiling, not a warning — overspend raises `MeterExceeded`.                            |
| Human-in-the-loop via `interrupt`      | `autonomy="gated"` + `@tool(side_effecting=True)` + `agent.resume(...)`             | Real pause: snapshot + return `Suspended`; a fresh process can resume.                         |
| LCEL streaming (`.astream_events`)     | `agent.stream(task, ctx)` yielding `StreamEvent`                                    | Closed union of event types (`message_delta`, `tool_call`, `tool_result`, `step`, `final`).    |
| `RunnableParallel`                     | `run_agents([(a, task_a), (b, task_b)], ctx)`                                       | Structured concurrency: `TaskGroup` bounded by `Budget.semaphore()`.                           |
| `RunnableBranch`                       | Custom `Cognition` OR `CoordinatorCognition(policy=SelectorPolicy(...))`            | If the branching is loop-shaped, write a cognition; if it's routing, use a coordinator policy. |

## Side-by-side rewrite

The canonical tool-using agent — a system prompt, one search tool,
and a ReAct-style loop. Below is the LangChain shape most codebases
have; then the same thing in agentkit.

### Before — LangChain

```python
"""LangChain version, roughly equivalent to what many code-bases have."""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as lc_tool


@lc_tool
def search(query: str) -> str:
    """Search the web for `query`. Returns bulleted hits with source and year."""
    return "- 'Distributed cognition in cephalopods' (science.org, 2023)"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a terse briefer. Cite every claim."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search], max_iterations=6)

result = executor.invoke({"input": "Brief me on octopus cognition."})
print(result["output"])
```

### After — agentkit

```python
"""agentkit version. Same behavior; swap the LLM plug-in for whichever
provider fits your deploy (OpenAI shown; ClaudeCliCognition, Anthropic,
DeepSeek, OpenRouter, or any LLMPort works the same way)."""

import asyncio

from agentkit import Agent, RunContext, Scope, Services, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Invoker

# Your LLMPort of choice — batteries-included providers or your own.
# from agentkit.adapters.llm import providers
# llm = providers.openai(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
llm = ...  # LLMPort


@tool(side_effecting=False)                       # required keyword
async def search(query: str) -> str:
    """Search the web for `query`. Returns bulleted hits with source and year."""
    return "- 'Distributed cognition in cephalopods' (science.org, 2023)"


async def main() -> None:
    services = Services(
        invoker=Invoker(
            llm=llm,
            chat_middleware=[tracing(), meter(), retry()],
        ),
    )
    ctx = RunContext(
        correlation_id="run-1",
        scope=Scope(),
        services=services,
    )

    agent = Agent(
        name="briefer",
        model="gpt-4o-mini",
        prompt="You are a terse briefer. Cite every claim.",
        cognition=ReActCognition(tools=[search], max_iterations=6),
    )

    result = await agent.run("Brief me on octopus cognition.", ctx)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
```

Everything you had before is still here; nothing is hidden inside the
framework. The chain (`tracing → meter → retry`) is a list you can
edit; adding `memoize()` for scope-aware caching or `compaction(...)`
for context shrinking is one entry. Turning on HITL is
`autonomy="gated"` and one `side_effecting=True` on any tool that
mutates the world.

## What changes in the mental model

**Loop control moves to `Cognition`.** In LangChain the `AgentExecutor`
IS the loop. In agentkit the loop lives inside `ReActCognition` (or
whatever `Cognition` you handed the `Agent`). Change loops by handing
in a different cognition — no subclass, no re-wiring.

**Cross-cutting concerns move to middleware.** Callbacks in LangChain
observe; in agentkit `Middleware`s can observe, transform, guard, and
re-invoke. `retry` is a middleware. `meter` (Budget + Quota) is a
middleware. `memoize` is a middleware. All swap by editing a list.

**Cost and cancel become first-class.** `Budget(max_cost_usd=...)`
halts the run when spent overshoots — enforced, not warned.
`CancellationToken` on `RunContext` propagates through
`ctx.child()` — cancel a parent, every child stops at its next
`check_cancelled()`.

**HITL is a real pause.** `autonomy="gated"` on the run + a
`side_effecting=True` tool + a `Checkpointer(port=...)` snapshots
state and returns a `Suspended`. Your driver reads the pending tool
calls, collects human decisions, then calls
`agent.resume(run_id, decisions, ctx)` — potentially from a fresh
process, weeks later. The state lives in the checkpoint store, not in
your process.

## What LangChain still has that we don't

**A larger integration catalog.** Community `langchain_*` packages
cover a wide range of vector stores, document loaders, and provider
SDKs out of the box. agentkit ships batteries for
`claude` / `openai` / `deepseek` / `openrouter`, MCP consumption,
Postgres/Redis stores, and OpenTelemetry — but not a
`WhateverProvider` integration bundle. If you were reaching for a
LangChain community package specifically because it exists, you'll
write a small `LLMPort` or `VectorPort` implementation instead — a
few dozen lines of code, and you own it.

**RAG-shaped abstractions.** LangChain has a well-worn RAG pipeline
grammar (`Retriever` + `DocumentTransformer` + reranker + parent-doc
retrieval). agentkit models memory as a `MemorySource` Protocol you
compose from `VectorMemory`, `JournalMemory`, `FileMemory`,
`CachedMemory`, `ScopedMemory` — flexible, but the RAG-specific
grammar isn't in the box.

**A hosted LangSmith trace UI.** agentkit's `TracePort` /
`ObserverPort` are Protocol seams; the `arc-agentkit[observability]`
extra bridges them to OpenTelemetry (any OTLP backend — Tempo,
Jaeger, Datadog, Honeycomb, Grafana Cloud). LangSmith is not a first-
party target. If you want it, wire it as an `ObserverPort` adapter.

**A CLI-first developer experience.** LangChain has `langchain-cli`
for scaffolding. agentkit doesn't; a fresh project is a `pip install`
plus the [Tutorial](../tutorial.md).

## What agentkit has that LangChain doesn't (as of today)

- Enforced budgets — `Budget(max_cost_usd=..., max_calls=..., max_depth=...)`
  charging under an async lock, invariant under concurrent workers.
- HITL suspend/resume as a real pause with durable state — not
  `interrupt` on a graph you have to snapshot yourself.
- `mypy --strict` clean public surface with `py.typed`.
- Zero runtime dependencies in the core. `pip install arc-agentkit`
  gives you a working framework with no `pydantic`, no `httpx`, no
  vendor SDK.
- Structured concurrency (`asyncio.TaskGroup`) with a shared
  cancellation token that stops the subtree on the first sibling
  failure.
- `ActorBudget` — per-child slice of the parent's envelope, so
  fanning out ten children over a $5 budget gives each $0.50 with
  fail-fast on reservation exhaustion.

## Migration order that works

1. **Wrap the LLM.** Point an `LLMPort` at whatever provider client
   you already have. The batteries-included presets under
   `agentkit.adapters.llm.providers` (`claude` / `openai` /
   `deepseek` / `openrouter`) are the easiest place to start.
2. **Move one tool at a time.** Rewrite each `@tool` decorator (state
   `side_effecting=`), test in isolation with `make_test_ctx`.
3. **Move the loop.** Replace `AgentExecutor` with `Agent(...,
   cognition=ReActCognition(tools=...))`. Everything downstream still
   works.
4. **Add a budget.** `Budget(max_cost_usd=..., max_calls=...)` on the
   `RunContext`. Wire `meter()` into the middleware chain.
5. **Turn on gating.** `autonomy="gated"` and `side_effecting=True`
   on the tools that mutate the world. Wire a `Checkpointer`. Handle
   `Suspended` in your driver.

Steps 1–3 are semantically equivalent to what you had. Steps 4–5
buy the "your agent overspent" and "your agent rm -rf'd in prod"
protections you came here for.

## Idiom-by-idiom cheat sheet

Common LangChain phrases and their agentkit spelling.

### "I want streaming events I can render live"

```python
# LangChain (LCEL):
async for ev in chain.astream_events(input, version="v2"):
    if ev["event"] == "on_chat_model_stream":
        print(ev["data"]["chunk"].content, end="")

# agentkit:
async for ev in agent.stream(task, ctx):
    if ev.type == "message_delta":
        print(ev.text, end="")
    elif ev.type == "tool_call":
        print(f"\n[tool] {ev.tool_call.name}")
    elif ev.type == "final":
        print(f"\n[usage] {ev.result.usage}")
```

`StreamEvent.type` is a closed literal union. Missing cases surface
under `mypy --strict`. There is no untyped `event["data"]["chunk"]`
lookup.

### "I want a per-request configuration override"

```python
# LangChain:
chain.invoke(input, config={"tags": ["debug"], "callbacks": [my_handler]})

# agentkit: build a fresh RunContext with the overrides you want:
ctx_debug = RunContext(
    correlation_id="req-debug-1",
    scope=scope,
    services=Services(invoker=my_invoker, observer=my_observer),
)
await agent.run(input, ctx_debug)
```

`RunContext` IS the config surface. No parallel `RunnableConfig` type
to remember.

### "I want per-tenant isolation"

```python
# agentkit — Scope threads through everything scope-aware:
from agentkit import Scope

for tenant in tenants:
    ctx = RunContext(
        correlation_id=str(uuid.uuid4()),
        scope=Scope(org_id=tenant.org, domain_id=tenant.domain),
        services=shared_services,          # SAFE — Services is app-shared
        budget=Budget(max_cost_usd=tenant.per_run_budget),
        meters=[shared_quotas[tenant.org]],  # per-tenant Quota, keyed by Scope
    )
    await agent.run(task, ctx)
```

`Scope` propagates into: cache keys (via `memoize`), quota buckets
(via `Quota._reqs[scope_key]`), and `ScopedMemory` boundaries.

### "I want to structure my outputs with Pydantic"

```python
# LangChain:
from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Brief)
chain = prompt | llm | parser

# agentkit:
from pydantic import BaseModel
from agentkit import Agent

class Brief(BaseModel):
    summary: str
    citations: list[str]

agent = Agent(name="briefer", model="...", prompt="...", output=Brief, max_repairs=1)
result = await agent.run(task, ctx)
result.parsed  # -> Brief(summary=..., citations=[...])
```

`output_coerce` middleware validates on the way out; a parse failure
reflects the error to the model as a repair message.

### "I want to store chat history"

```python
# LangChain:
from langchain_core.chat_history import BaseChatMessageHistory
# ... configure a store, wrap the chain, deal with session ids ...

# agentkit — history goes through the same Checkpointer that powers HITL:
checkpointer = Checkpointer(port=PostgresCheckpointStore(pool=DSN))
# Continue the same session:
ctx = RunContext(correlation_id=session_id, scope=scope,
                 services=Services(..., checkpointer=checkpointer))
```

The `run_id` (= `correlation_id`) is the session key. Same interface
whether it's the same worker or a fresh process weeks later.

### "I want to call multiple LLMs and pick the first to succeed"

```python
# LangChain:
llm.with_fallbacks([alt_llm_1, alt_llm_2])

# agentkit:
from agentkit.middlewares import fallback
chat_middleware = [tracing(), meter(), fallback(models=["gpt-4o", "gpt-4o-mini"]), retry()]
```

The `fallback` middleware rewrites `request.model` on transient
failure and re-invokes `next`. Combined with `retry` inside, you get
transient-retry-per-model + fallback-to-next-model in the right
order.

### "I want a caching layer over identical LLM calls"

```python
# LangChain:
from langchain_core.caches import InMemoryCache
llm.cache = InMemoryCache()

# agentkit:
from agentkit.middlewares import memoize
chat_middleware = [tracing(), meter(), retry(), memoize()]
```

`memoize` is scope-aware — cache keys are hashed with the run's
`Scope` so a tenant's cache doesn't leak into another's. For
near-duplicate hits, use `semantic_memoize(vector=...)` with a
similarity threshold.

### "I want an evaluation harness"

```python
# LangChain (LangSmith): hosted, external.
# agentkit: Evaluator is a CONCRETE class you configure, not a Protocol.
# It runs `code_checks` (cheap, deterministic) and an optional LLM judge.
from agentkit import Evaluator

evaluator = Evaluator(
    code_checks={
        "mentions_source": lambda out: "http" in str(out),
        "is_concise": lambda out: len(str(out)) < 1200,
    },
    judge_model="claude-sonnet-4-6",
    rubric='Score 0..1 for faithfulness. Return JSON {"score": float, "reason": str}.',
)

scores = evaluator.code_evals(result.output)          # dict[str, bool]
verdict = await evaluator.judge(ctx, goal=task, output=result.output)
```

Wire it into your test loop; agentkit doesn't ship a runner because
the shape of "a test loop" is your business — pytest + parametrize is
the canonical answer.

## Related

- [Cheatsheet](../cheatsheet.md) — every primitive, tight code.
- [Tutorial](../tutorial.md) — the same shape as this rewrite,
  walked step-by-step.
- [Anti-patterns](../anti-patterns.md) — the fifteen mistakes you'll
  make in your first week; read them once now.
