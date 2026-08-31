# Why agentkit

Every agent framework makes a bet. This page states agentkit's bet plainly,
lists the sixteen concrete guarantees you get if you take it, and compares
the bet against the alternatives so you can pick the tool whose bet
matches the code you're actually about to write.

It is the long version, for a reader who is already interested. If you
are still deciding whether this framework is for you at all, the
[landing page](index.md) is the 30-second version; if you just want
something running, go to [Getting started](getting-started.md).

The bet: **an agent is composed, not scripted.** Four orthogonal concerns —
**cognition**, **control**, **state**, **behaviour** — plug together
through typed Protocols. Every seam is a swap. Nothing is hidden inside a
loop you can't rewrite.

## The four themes

Every concept in agentkit falls into one of those four buckets. Learn the
four and the rest is which class fills which slot.

<div class="grid cards" markdown>

-   __Cognition__

    ---

    How the agent decides the next step. `SingleCallCognition`,
    `ReActCognition`, `CoordinatorCognition`, `ClaudeCliCognition`,
    or your own `Cognition` Protocol impl.

    [:octicons-arrow-right-24: Concepts › Agents](concepts/agents.md)

-   __Control__

    ---

    What limits the agent's authority. `Autonomy`, `Budget`, `Quota`,
    `CancellationToken`, `RunPolicy`, `Suspended` + `resume()` for
    HITL.

    [:octicons-arrow-right-24: Concepts › Runtime](concepts/runtime.md)

-   __State__

    ---

    What the agent knows. `WorkingContext`, `MemorySource`
    (`VectorMemory` / `JournalMemory` / `FileMemory` /
    `ScratchpadMemory`), `Prompt`, `Checkpointer`.

    [:octicons-arrow-right-24: Concepts › Capabilities](concepts/capabilities.md)

-   __Behaviour__

    ---

    How every call is intercepted. The middleware chain (`tracing`,
    `retry`, `meter`, `compaction`, `security`, `output_coerce`, …)
    and capabilities (`RequestBuilder`, `Compactor`, `Guardrail`).

    [:octicons-arrow-right-24: Concepts › Middlewares](concepts/middlewares.md)

</div>

**Adapters** — the `claude()` / `openai()` / `deepseek()` /
`openrouter()` presets, `ClaudeCliCognition`, `MCPClient`, and any
`LLMPort` you write — are **plug-ins**, not the point. They fill the LLM
slot in a composition; the composition is what agentkit is about.

## What you get

Sixteen concrete guarantees, each a capability the framework hands you and
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
wire-up. Swap the checkpoint store for your own by implementing
`CheckpointPort`'s five methods (`save`, `latest`, `at_version`,
`list_versions`, `delete`). `py.typed` ships in the wheel.

### 7. Zero runtime dependencies in the core

`pip install arc-agentkit` gives you a working framework. No
`pydantic`, no `httpx`, no vendor SDK, no `openai`, no `anthropic`.
The batteries-included providers (`arc-agentkit[http]`), Postgres
adapters (`arc-agentkit[postgres]`), Redis store (`arc-agentkit[redis]`),
OpenTelemetry bridge (`arc-agentkit[observability]`), MCP client
(`arc-agentkit[mcp]`) and the faster JSON codec (`arc-agentkit[fast]`)
are opt-in extras. Nothing you don't use is imported — and nothing you
do use changes behaviour based on which extras happen to be present.

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

The wire runs both ways. `serve_registry(registry, name=..., ctx=...)`
publishes a `ToolRegistry` you already have as an MCP server, and
`ToolSchema.parameters` goes out as `inputSchema` **unchanged** — no
translation step, because a translation step is a second description of
one thing and second descriptions drift into a model being shown a
schema the tool does not validate against.

### 12. A first-class testing kit that avoids double-implementation

`FakeLLM`, `FakeFetch`, `FakeSearch`, `FakeMemory`, `FakeTool`,
`FakeCompactor`, `FakeGrounder`, `FakeClock`, `FakeClaudeCli`,
`RecordingTracer`,
`FakeCtx`, and `make_test_ctx(...)` — the same doubles the framework's
own suite uses. Zero API keys required to unit-test your agents end
to end. Test doubles live under `agentkit.testing.*` and are
**deliberately** not re-exported from the top-level package: a
`from agentkit import FakeLLM` shape would let production code accidentally
pin a test double. The boundary is enforced.

### 13. Delegating the loop to the Claude CLI no longer delegates the controls

`ClaudeCliCognition` gives the whole loop to a subprocess, and the
subprocess does not walk your `Invoker`. So a service that wired
`egress`, `audit` and `guard`, then reached for the CLI, got a session
where **none of them applied** and nothing said so. `WebFetch` reached
anywhere. There were no audit records.

Three things close that now, and they are guarantees rather than
features because each one changes what you can promise about a run.
The cognition **warns and names the middlewares that stopped applying**,
once per instance — not a vague caution that trains you to add a
`filterwarnings` line. `hook_settings(middleware=..., ctx=...,
tools=...)` regenerates the same chain as a `PreToolUse` hook, so it
reaches `Write`, `Edit` and `Bash`; it **fails closed** on a payload it
cannot parse and anchors its matcher (`^(Write|Edit)$`, so an `Edit`
rule does not also fire for `NotebookEdit`). And `RunPolicy` can now see
the session at all: `ClaudeCliCognition` declares capability tags
derived from `tools=`, so the lethal-trifecta check refuses
`("Read", "WebFetch")` — `WebFetch` supplies two legs by itself — and
refuses the default `tools=None`, which means every built-in tool and
is therefore the whole trifecta.

Two honest costs. The hook runs in a *separate process*, so it cannot
hold your live `ctx`; middlewares that need one are refused at
generation time rather than accepted and silently skipped, and the
refusal is the point. And the CLI's own hook timeout is a **fail-open**
— a hook it times out prints an error and then runs the tool — so the
inner deadlines have to fire first.

### 14. An approval leaves a record, not a count

`ApprovalServer(asker=...)` answers the CLI's `--permission-prompt-tool`
with your existing `Asker`, so the CLI's permissions stop being a choice
between `bypassPermissions` (anything, unattended) and `dontAsk` (the
run just fails). What it hands back afterwards is the guarantee:
`.decisions` is a list of `ApprovalDecision` — `tool`, `arguments`,
`allowed`, `reason`, `source`, `at`, `asked`.

A counter cannot answer the questions an incident asks. *Allowed because
a person said so* and *allowed because the run's autonomy tier does not
gate this call* are the two facts an auditor most needs kept apart, and
a boolean collapses them; `source` is a closed set (`asker`,
`auto_allow`, `autonomy`, `timeout`, `error`) that keeps them apart. A
`timeout` deny must never read afterwards as a human who said no.
`asked` is tracked separately from `source == "asker"` because a prompt
can reach a person and then expire — whether somebody was interrupted is
a fact about what the run cost a human, and it is not recoverable from
the verdict.

The default is deny, and `timeout_s` is enforced by the server rather
than trusted to the `Asker`, because the `Asker` protocol permits
waiting forever and a queue worker holding a CLI subprocess open
indefinitely is a resource leak with a model attached.

### 15. The CLI path is testable offline, and the test exercises the real code

Testing anything that went through `ClaudeCliCognition` used to need a
real `claude` binary, real auth and real money — so in practice it was
tested by hand, or not at all, which is how a path that spawns processes
and parses a stream ends up the least-covered code in a service.

`FakeClaudeCli.script([...])` handed to `ClaudeCliCognition(spawn=...)`
sits at the **spawn seam**, not in front of the cognition. That
placement is the guarantee: the test still runs the real stream-json
parsing, the real budget charging and the real `StreamEvent` mapping,
which is where every bug on this path has actually lived. It earned that
claim on its first hostile input — a line that is not valid UTF-8 raised
`UnicodeDecodeError`, which is not a `JSONDecodeError`, so it escaped
the parser's handler; the reader stopped before the `result` payload and
a completed run was charged $0.00.

### 16. The servers agentkit stands up are authenticated by default

`serve_registry` and `ApprovalServer` both bind loopback and both
**require a generated bearer token**. `cli_kwargs()` carries it in the
config document's headers; `auth_headers` is the same credential for a
non-CLI client. Nothing accepts a caller-supplied token, because a token
a caller can set is a token that ends up a constant in somebody's config
file.

Loopback alone used to be the containment, and that argument holds
exactly as long as nothing untrusted shares the host — which is the
wrong assumption for the case these servers exist to serve. The point of
`ClaudeCliCognition` is that the CLI runs `Bash`: a build, a package
install, a script out of a repository the agent was pointed at, inside
the same network namespace as the server holding your tools and your
approval gate.

The honest version is that a token is a weaker fence than a `0700`
directory: it turns *anything on the host* into *anything that can read
this process's temp directory*. That is why the config file is `0600`
and why `stop()` removes it along with the listener — the file is a live
credential now, so a stale one outliving its server would be worse than
untidy. `auth="none"` restores the old behaviour, as something you ask
for by name rather than something you get by default.

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
  by implementing `LLMPort` — three methods (`stream`, `chat`,
  `complete`), and `chat` can just collect `stream`.
- **Not a hosted service.** Runs where your Python runs. State lives
  where you point the `CheckpointPort`.

## What "typed seams" buys you concretely

Every abstract argument for typed seams sounds fine and buys nothing
in practice unless you can point at code. Here's what the
`typing.Protocol`-driven design lets you do without editing the loop.

### Swap OpenAI for DeepSeek for OpenRouter without touching the agent

Constructing a provider does no network I/O, so the whole block below
runs offline on `arc-agentkit[http]`; only an actual call needs the
matching key.

```python
from agentkit.adapters.llm import providers
from agentkit.runtime import Invoker, Services

# Pick one. The only line that differs is the preset and the model name.
llm = providers.claude(api_key="sk-placeholder", model="claude-sonnet-4-6")
llm = providers.openai(api_key="sk-placeholder", model="gpt-4o-mini")
llm = providers.deepseek(api_key="sk-placeholder", model="deepseek-chat")
llm = providers.openrouter(api_key="sk-placeholder", model="anthropic/claude-sonnet-4-6")

# Whichever you picked lands in the same slot. Nothing downstream moves.
services = Services(invoker=Invoker(llm=llm))
print(type(llm).__name__)
```

```text
OpenAICompatibleLLM
```

`providers.*` return an `LLMPort` for you to wire yourself.

If you want the batteries-included version instead, reach for
`agentkit.client`: `claude()` / `openai()` / `deepseek()` /
`openrouter()` return a `Chat` that has already wrapped the same port in
an `Invoker` with `tracing → meter → retry`. It is also an async context
manager, so the HTTP pool closes on exit.

Or drop to `LLMPort` and write your own vLLM / Ollama / in-house
wrapper. The `Agent` doesn't care.

### Swap the checkpoint store from in-memory to Postgres

```python
from agentkit import Checkpointer
from agentkit.adapters.checkpoint import InMemoryCheckpointStore

port = InMemoryCheckpointStore()

# Production (arc-agentkit[postgres]) — same object, different backing:
#
#   import asyncpg
#   from agentkit.adapters.checkpoint import PostgresCheckpointStore
#
#   pool = await asyncpg.create_pool("postgresql://…")
#   port = PostgresCheckpointStore(pool)

checkpointer = Checkpointer(port=port)
print(type(checkpointer.port).__name__)
```

```text
InMemoryCheckpointStore
```

!!! warning "`PostgresCheckpointStore` takes a live pool, not a DSN"
    Its one argument is an `asyncpg.Pool` — you create the pool
    yourself with `await asyncpg.create_pool(dsn)` and hand it over.
    That is deliberate: the pool's lifecycle, sizing and shutdown belong
    to your application, not to a checkpoint store.

`ReActCognition`'s durable-resume path doesn't move either way.

### Point tracing at any OTLP backend

`Services()` installs a `NoopTrace` and a `NoopObserver` by default, so
tests and local runs need no wiring at all:

```python
from agentkit.runtime import Services

services = Services()
print(type(services.trace).__name__)
```

```text
NoopTrace
```

In production, hand it a real tracer (needs
`arc-agentkit[observability]`):

```python
from agentkit.adapters.observability import otel_exporter_otlp_http, otel_tracer
from agentkit.runtime import Services

otel_exporter_otlp_http()   # reads OTEL_EXPORTER_OTLP_ENDPOINT etc.
services = Services(trace=otel_tracer())
print(type(services.trace).__name__)
```

```text
OtelTracePort
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
| CLI session's `WebFetch` ignored the egress allowlist              | Native CLI tools never reach the `Invoker`          | `hook_settings(...)` → `ClaudeCliCognition(settings=...)`        |
| "Who approved that?" answered with a number                        | A counter, not a record                             | `ApprovalServer(asker=...)` → `.decisions`                       |
| The CLI path was only ever tested by hand                          | No double at the spawn seam                         | `FakeClaudeCli.script([...])` + `ClaudeCliCognition(spawn=...)`  |
| One fact took two of the top-`k` slots the model reads             | Fan-out merged its sources blindly                  | `CompositeMemory(sources, dedupe="id")`                          |
| Retry re-ran an attempt that kept failing the same way             | A count cannot tell progress from a loop            | `attempt_until_stuck(fn, fingerprint=...)` → `Stuck`             |

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

- **[Getting started](getting-started.md)** — install, extras, and a
  first agent that runs offline.
- **[Tutorial](tutorial.md)** — an end-to-end walk from one chat call
  to a gated tool a human has to approve. Only the last step needs an
  API key.
- **[Cheatsheet](cheatsheet.md)** — every primitive, tight code,
  skimmable in 90 seconds.
- **[Recipes](recipes/index.md)** — problem-first answers to "how do
  I X?".
- **[Anti-patterns](anti-patterns.md)** — the fifteen traps every
  first-time user falls into.
- **[Concepts › The Claude CLI](concepts/claude-cli.md)** — guarantees
  13 to 16 in full: the one loop you don't own, what stops applying
  when you take it, and the seams that reach back in.
- **[Mental models](mental-models/README.md)** — four worked product
  scenarios that show these guarantees composing under load, and what
  breaks when one of them slips.
- **[Migrating › From LangChain](migrating/from-langchain.md)** —
  concept-by-concept mapping with a full before/after.
- **[Migrating › From vanilla asyncio](migrating/from-vanilla-asyncio.md)** —
  for the reader who's hand-rolled a `messages.create` loop.
