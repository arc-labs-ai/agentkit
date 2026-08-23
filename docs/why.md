# Why agentkit

Every agent framework makes a bet. This page states agentkit's bet plainly,
lists the twelve concrete guarantees you get if you take it, and compares
the bet against the alternatives so you can pick the tool whose bet
matches the code you're actually about to write.

The bet: **an agent is composed, not scripted.** Four orthogonal concerns —
**cognition**, **control**, **state**, **behaviour** — plug together
through typed Protocols. Every seam is a swap. Nothing is hidden inside a
loop you can't rewrite.

## What you get

Twelve concrete guarantees, each a capability the framework hands you and
competitors mostly don't.

### 1. A run has an explicit ceiling that halts it, not a dashboard that warns you

`Budget(max_cost_usd=..., max_calls=..., max_depth=...)` on the
`RunContext` is enforced under an async lock by the `meter()` middleware
on every chat call. Overspend raises `MeterExceeded`; the cognition
propagates it; the run stops. Totals are invariant under concurrent
workers — a coordinator fanning ten researchers out on the same
`Budget` cannot race past its ceiling. Same primitive scoped by tenant:
`Quota(max_rpm=..., max_tpm=..., max_usd=...)` on the
same `meter()` chain.

### 2. Every side-effecting tool goes through an approval gate you actually control

`@tool(side_effecting=True)` marks a tool as world-mutating at
decoration time; forgetting the flag is a `ToolDefinitionError`
(the error is the fix). `RunContext.autonomy` is one of `"auto"`,
`"gated"`, or `"manual"`. Under `"gated"`, every `side_effecting=True`
tool call suspends the loop, snapshots state through the `Checkpointer`,
returns a `Suspended` — and waits for your driver to call
`agent.resume(run_id, decisions, ctx)` with `"approve"` /
`"reject"` / a JSON args-override per pending `ToolCall.id`. The gate
is a real pause, not an animation.

### 3. A crash is not the end of a run

The same `Checkpointer` that powers HITL powers durable resume.
`ReActCognition` snapshots after every successful tool iteration; a
fresh process running the same `run_id` against the same
`CheckpointPort` hydrates the transcript, `Usage`, and the next
iteration index. `InMemoryCheckpointStore` for tests;
`PostgresCheckpointStore` (from `arc-agentkit[postgres]`) for
production. One producer per `run_id` — enforced by monotonic
versioning at the port.

### 4. Cooperative cancel across the whole subtree

`CancellationToken` on `RunContext` is shared by reference through
`ctx.child()`. Cancelling the parent flips every child's next
`ctx.check_cancelled()` into a `Cancelled` exception. `run_agents(...)`
runs children inside `asyncio.TaskGroup` bounded by
`Budget.semaphore()` — the first failure cancels the siblings and
raises an `ExceptionGroup` you catch with `except*`. Fan out ten
researchers, one blows up, the other nine stop *before* they burn the
same budget on work you'll throw away.

### 5. The middleware chain is a list you own, not a decorator soup

Two chains — `chat_middleware`, `tool_middleware` — are handed to the
`Invoker`. `chain([tracing(), meter(), retry(), memoize(),
compaction(), security(), output_coerce()], terminal)` is a plain
Python list: reorder, swap, or drop by editing the file. No plugin
discovery, no registration decorator, no priority number to argue
about. Ordering is deterministic because a list is deterministic.

### 6. Every seam is a typed Protocol, `mypy --strict` clean

`LLMPort`, `StorePort`, `VectorPort`, `SearchPort`, `FetchPort`,
`ClockPort`, `CheckpointPort`, `TracePort`, `ObserverPort`,
`MetricsPort`, `SamplerPort`, `ToolPort` — every I/O seam is a
`typing.Protocol`. Swap OpenAI for DeepSeek by rewriting one line of
wire-up. Swap the checkpoint store for your own by implementing four
async methods. `py.typed` ships in the wheel.

### 7. Zero runtime dependencies in the core

`pip install arc-agentkit` gives you a working framework. No
`pydantic`, no `httpx`, no vendor SDK, no `openai`, no `anthropic`.
The batteries-included providers (`arc-agentkit[http]`), Postgres
adapters (`arc-agentkit[postgres]`), Redis store (`arc-agentkit[redis]`),
OpenTelemetry bridge (`arc-agentkit[observability]`), and MCP client
(`arc-agentkit[mcp]`) are opt-in extras. Nothing you don't use is
imported.

### 8. Structured output that survives model drift

`Agent(output=MyPydanticModel)` (or `attrs` / `dataclass` / raw JSON
Schema dict) builds a `SchemaAdapter` at construction time. The
adapter's schema lands in the cache-stable prompt prefix; the
`output_coerce` middleware coerces raw text on the way out. Parse
failure surfaces to the model as a repair message with a bounded
`max_repairs` retry budget. Same primitive for tool outputs:
`@tool(..., output_schema=MyModel)` validates every tool return.

### 9. Compose agents without subclassing anything

`Agent` is a `@dataclass`. Its `cognition=` field takes any
`Cognition` Protocol impl. Change the loop by handing in a different
cognition. `SingleCallCognition` for one-shot, `ReActCognition` for a
tool loop, `CoordinatorCognition(children=..., policy=...)` for
multi-agent, `ClaudeCliCognition` to delegate to the local `claude`
CLI, or a 60-line `Cognition` subclass of your own. `Skill` composes
prompt + cognition + tools + memory into a single wirable unit;
`skill.as_agent()` and `skill.as_tool()` let it live as either.

### 10. Provider-neutral by design

The core doesn't know what an LLM is beyond `LLMPort`. Batteries-included
adapters — `claude(...)`, `openai(...)`, `deepseek(...)`,
`openrouter(...)` — each return a `Chat` wired with the standard
`tracing → meter → retry` chain. `ClaudeCliCognition` delegates to a
locally-installed `claude` CLI (no API key needed on your server; the
CLI's own auth is used). Any HTTP endpoint speaking a chat-style API
is one `LLMPort` implementation away.

### 11. MCP is a first-class citizen, not a translation layer

`arc-agentkit[mcp]` gives you `MCPClient(StdioServer(...))` and three
adapters: `mcp_tools(client)` returns objects satisfying agentkit's
`Tool` Protocol, drop-in for `ReActCognition(tools=...)`.
`mcp_resources(client)` returns a `MemorySource` the agent can query
for grounding. `mcp_prompts(client)` returns
`dict[str, Prompt]` for server-authored versioned prompts. One
network to consume every MCP server ever written.

### 12. A first-class testing kit that avoids double-implementation

`FakeLLM`, `FakeFetch`, `FakeSearch`, `FakeMemory`, `FakeTool`,
`FakeCompactor`, `FakeGrounder`, `FakeClock`, `RecordingTracer`,
`FakeCtx`, and `make_test_ctx(...)` — the same doubles the framework's
own suite uses. Zero API keys required to unit-test your agents end
to end. Test doubles live under `agentkit.testing.*` and are
**deliberately** not re-exported from the top-level package: a
`from agentkit import FakeLLM` shape would let production code accidentally
pin a test double. The boundary is enforced.

## When agentkit is the fit

If you tick most of these, agentkit is likely the right pick:

- The code is going into production; a demo notebook is not the target.
- You need cost caps, cancellation, and HITL as first-class concerns —
  not the framework's `v0.3` addition.
- You want typed seams and `mypy --strict` on your own code, and the
  framework had better not fight you.
- You want the loop to be visible and rewritable, not hidden behind a
  graph DSL.
- You're wiring the LLM yourself — no vendor is going to lock you in.

## When it's probably not

- You want a visual flow graph you can drag and drop. Reach for
  LangGraph or a hosted equivalent.
- You want a hosted, one-click agent with built-in personas, memory,
  and deploy. Reach for a hosted framework.
- You're building a personal script that never grows beyond one file
  and one provider. Vanilla `asyncio` plus the provider SDK is fine
  and agentkit's typed seams and control plane will just be ceremony.
- You need a giant catalog of pre-glued integrations. LangChain has
  more of them today. agentkit ships fewer, on purpose.

## How the alternatives stack up

Each cell states the tool's actual bet, not marketing.

| Tool                        | The bet                                                                                       | Where it wins                                                     | Where it hurts                                                       |
|-----------------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------|
| **agentkit**                | Typed, composable seams; explicit loop; cancel/budget/HITL as core, not add-ons.              | Production code that must survive. Multi-agent with hard limits.  | Prototyping in a notebook where ceremony outweighs benefit.          |
| **LangGraph**               | Agents as a directed graph; the graph DSL is the mental model.                                | Flow you can draw; branching your ops team can review visually.   | Loop dynamics you can't model as a fixed graph. Type gymnastics.     |
| **LangChain**               | A large catalog of prebuilt integrations glued together via `Runnable`s.                      | You need a specific integration and don't want to write it.       | Minimal, typed core. Vendor gravity comes with the catalog.          |
| **LlamaIndex**              | RAG-shaped abstractions with agents added on top.                                             | Document-heavy retrieval with sensible defaults.                  | Non-RAG shapes push against the framework's grain.                   |
| **Instructor**              | Structured outputs on top of a single provider SDK.                                           | You already have a script; you just need typed outputs.           | No tool loop, no cancel, no budget, no resume — only outputs.        |
| **Pydantic-AI**             | Typed agents backed by Pydantic; friendly for one-agent, one-provider setups.                 | Solo agent with structured I/O; readable and typed.               | Multi-agent coordination, HITL suspend/resume, durability.           |
| **Vanilla asyncio + SDK**   | The whole thing is one file and one provider; nothing is worth abstracting.                   | Truly small scripts. The 30-line loop that never grows.           | Retries, cancel, budget, HITL, resume — all you-write-it.            |
| **OpenAI Assistants API**   | The vendor owns the loop, the state, the tools, the memory.                                   | You want to hand off *everything*; you don't care about lock-in.  | The vendor changes the deal. State lives outside your process.       |
| **Claude Agent SDK**        | The vendor owns the loop, the tools, the sandbox; you write handlers.                         | You want the CLI's tool surface (`Read`/`Bash`/`Edit`) in code.   | You can't rewrite the loop. Middleware is theirs, not yours.         |

## Sanity check: what agentkit is *not*

- **Not an agent product.** No built-in personas. No "researcher"
  agent that comes with a prompt. You author agents; the framework
  runs them.
- **Not a chain-builder.** No graph DSL, no fluent `.pipe(...).and()`.
  Composition is Python: a list of middlewares, a cognition object,
  an `Agent` you construct.
- **Not opinionated about LLMs.** The core doesn't import a provider.
  The `claude()`, `openai()`, `deepseek()`, `openrouter()` presets are
  convenience wrappers over `LLMPort` and are opt-in. Bring your own
  by implementing `LLMPort` — it's four methods.
- **Not a hosted service.** Runs where your Python runs. State lives
  where you point the `CheckpointPort`.

## What "typed seams" buys you concretely

Every abstract argument for typed seams sounds fine and buys nothing
in practice unless you can point at code. Here's what the
`typing.Protocol`-driven design lets you do without editing the loop.

### Swap OpenAI for DeepSeek for OpenRouter without touching the agent

```python
# Wire once — swap the provider by changing the preset call.
from agentkit import claude, openai, deepseek, openrouter
# One line different across all four:
async with openai(api_key=..., model="gpt-4o-mini") as chat: ...
async with deepseek(api_key=..., model="deepseek-chat") as chat: ...
async with openrouter(api_key=..., model="anthropic/claude-3.5-sonnet") as chat: ...
```

Or drop to `LLMPort` and write your own vLLM / Ollama / in-house
wrapper. Four async methods. The `Agent` doesn't care.

### Swap the checkpoint store from in-memory to Postgres

```python
# Test:
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
port = InMemoryCheckpointStore()

# Production (arc-agentkit[postgres]):
from agentkit.adapters.checkpoint import PostgresCheckpointStore
port = PostgresCheckpointStore(pool="postgres://...")

checkpointer = Checkpointer(port=port)   # same object, different backing
```

`ReActCognition`'s durable-resume path doesn't move.

### Point tracing at any OTLP backend

```python
# Test:
services = Services(trace=NoopTrace())

# Production:
from agentkit.adapters.observability import otel_tracer, otel_exporter_otlp_http
otel_exporter_otlp_http()   # reads OTEL_EXPORTER_OTLP_ENDPOINT etc.
services = Services(trace=otel_tracer())
```

Same `tracing()` middleware in the chain. Same spans. Different
backend.

## The failure modes we've seen — and their one-line fixes

| Failure                                                            | Root cause                                          | Fix                                                              |
|--------------------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------|
| Agent looped 40 times, spent $217                                  | No enforced budget                                  | `Budget(max_cost_usd=5.0)` on the `RunContext`                   |
| Worker OOM'd, transcript lost                                      | No durable state                                    | `Checkpointer(port=PostgresCheckpointStore(...))`                |
| Tool called `rm -rf` in prod                                       | No approval gate                                    | `autonomy="gated"` + `@tool(side_effecting=True)`                |
| Provider blipped, whole run crashed                                | No retry                                            | Add `retry(breaker=CircuitBreaker(...))` to the chain            |
| Tenant burned the shared limit                                     | No per-scope quota                                  | `Quota(max_rpm=..., max_tpm=..., max_usd=...)` on `ctx.meters`   |
| Cancel from the UI didn't stop anything                            | No cooperative cancel                               | `CancellationToken` on `RunContext`, checked at safe points      |
| Fan-out kept running after one child raised                        | Naive `asyncio.gather`                              | `run_agents(...)` — `TaskGroup` cancels siblings on first fail   |
| Test fixtures pulled in real API keys                              | No test doubles                                     | `agentkit.testing.*` — `FakeLLM` + `make_test_ctx()`             |
| Cost from cache hits was double-counted                            | Meter after cache                                   | Middleware order: `meter` above `memoize` in the chain           |
| Structured output silently drifted from the schema                 | No coercion in the loop                             | `Agent(output=MyPydanticModel)` + `output_coerce` middleware     |

## The bet, restated

Three sentences: **cognition, control, state, behaviour** are
orthogonal. Each is a typed Protocol. You compose an agent by picking
one implementation of each; you change the agent by picking a
different one. The framework's only opinion is the shape of that
composition.

Everything else — how the agent decides, what it can do, what it
knows, how it's watched — is a decision you make in Python you can
read.

## Next

- **[Cheatsheet](cheatsheet.md)** — every primitive, tight code,
  skimmable in 90 seconds.
- **[Tutorial](tutorial.md)** — a 15-minute end-to-end walk from one
  chat call to a gated tool the human has to approve.
- **[Recipes](recipes/index.md)** — problem-first answers to "how do
  I X?".
- **[Anti-patterns](anti-patterns.md)** — the fifteen traps every
  first-time user falls into.
- **[Migrating › From LangChain](migrating/from-langchain.md)** —
  concept-by-concept mapping with a full before/after.
- **[Migrating › From vanilla asyncio](migrating/from-vanilla-asyncio.md)** —
  for the reader who's hand-rolled a `messages.create` loop.
