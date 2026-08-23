# Getting started

By the end of this page you will have an agent that runs on your machine
with no API key, and the same agent pointed at a real provider.

It takes about five minutes. If you want the longer walk — a tool, a
stream, a spend ceiling, a human approval gate — that is the
[Tutorial](tutorial.md), which starts from the same script.

## Install

```bash
pip install arc-agentkit
```

!!! note "Distribution name vs import name"
    On PyPI the distribution is **`arc-agentkit`** (the `agentkit` name
    on PyPI is owned by an unrelated project). The **import name stays
    `agentkit`**, so all documentation and examples use `import
    agentkit` as-is. Same pattern as `pip install Django` → `import
    django`.

    Before the first release, install directly from Git:

    ```bash
    pip install "arc-agentkit @ git+https://github.com/arc-labs-ai/agentkit"
    ```

The core declares **no runtime dependencies** — `pip install
arc-agentkit` pulls nothing else into your environment. Everything that
needs a third-party library lives behind an opt-in extra, so you install
only the seams you actually wire.

| Extra | Pulls in | Install it when you want |
|---|---|---|
| `http` | `httpx[http2]` | The bundled provider adapters — `claude()`, `openai()`, `deepseek()`, `openrouter()`. This is the one most people need. |
| `postgres` | `asyncpg`, `pgvector` | `PostgresCheckpointStore` (durable resume), `PostgresStore`, `PgVectorStore`. |
| `redis` | `redis` | `RedisStore` — a Redis-backed `StorePort`. |
| `observability` | `opentelemetry-api` / `-sdk` / OTLP HTTP exporter | `otel_tracer()`, `otel_meter()` and the OTLP exporters, so `tracing()` spans leave the process. |
| `mcp` | `mcp` | `MCPClient` plus `mcp_tools()` / `mcp_resources()` / `mcp_prompts()` for consuming Model Context Protocol servers. |
| `fast` | `orjson` | A faster JSON codec. Pure speed — nothing changes in the API, and everything works without it. |

```bash
pip install "arc-agentkit[http]"            # the usual starting point
pip install "arc-agentkit[http,postgres]"   # extras compose
```

## Requirements

- **Python 3.12 or newer.** The codebase uses 3.12 generics and is
  typed under `mypy --strict`.
- **No API key** for the first example below — it runs against the
  `FakeLLM` test double that ships in the package.
- For a real run, either an API key for a provider (Anthropic, OpenAI,
  DeepSeek and OpenRouter ship as presets), or the local `claude` CLI on
  your `PATH`, which `ClaudeCliCognition` drives using the CLI's own
  login — no server-side key.

## Verify the install

```bash
python -c "import agentkit; print(agentkit.__version__)"
```

Prints the installed version (`0.1.0` today). If it errors, the wheel
isn't on your Python path — check which interpreter your shell resolved.

## Your first agent — no API key

This runs as-is, offline, on a bare `pip install arc-agentkit`.

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    # The agent: identity, which model it is for, what it is told to do.
    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt="Answer in one short sentence.",
    )

    # The run context: everything the run is allowed to touch, including
    # which LLM it talks to. Here that is a deterministic fake.
    ctx = make_test_ctx(llm=FakeLLM("Octopuses solve mazes and use tools."))

    result = await agent.run("What do we know about octopus cognition?", ctx)
    print(result.output)
    print(f"cost: ${result.usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

```text
Octopuses solve mazes and use tools.
cost: $0.0001
```

Three things to take from this, because they are the shape of every
later example:

- **`Agent` is a plain dataclass.** Name, model, prompt. It holds no
  connection and no state. You never subclass it.
- **The LLM is not on the agent — it is on the context.** `agent.run`
  reaches the model through `ctx`, so the same `Agent` value runs
  against a fake in a test and a real provider in production with no
  edit.
- **`FakeLLM` is not a stub you have to write.** It ships in
  `agentkit.testing`, it is the same double the framework's own test
  suite uses, and it charges a scripted `Usage` on every call — so cost,
  budget and metering code paths are exercised offline. It lives under
  `agentkit.testing` rather than the top-level package on purpose: a
  `from agentkit import FakeLLM` shape would let production code pin a
  test double by accident.

`make_test_ctx(...)` is the one-line context factory — it builds a real
`RunContext` with a real `Invoker`, and no-op tracing/observation
defaults. In application code you build `RunContext` and `Services`
yourself, which is what the next section shows.

## The same agent against a real provider

Same `Agent`. Only the wiring of `ctx` changes.

!!! warning "These snippets bill you"
    Each tab below makes a real request. The first two need an API key
    in your environment; the third needs the `claude` CLI installed and
    logged in. Nothing on this page before this point costs money.

=== "Anthropic"

    Needs `pip install "arc-agentkit[http]"` and `ANTHROPIC_API_KEY`.

    ```python
    import asyncio
    import os

    from agentkit import Agent, Scope
    from agentkit.adapters.llm import providers
    from agentkit.middlewares import meter, retry, tracing
    from agentkit.runtime import Invoker, RunContext, Services


    async def main() -> None:
        llm = providers.claude(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-sonnet-4-6",
        )
        services = Services(
            invoker=Invoker(llm=llm, chat_middleware=[tracing(), meter(), retry()]),
        )
        ctx = RunContext(correlation_id="run-1", scope=Scope(), services=services)

        agent = Agent(
            name="briefer",
            model="claude-sonnet-4-6",
            prompt="Answer in one short sentence.",
        )
        result = await agent.run("What do we know about octopus cognition?", ctx)
        print(result.output)
        print(f"cost: ${result.usage.cost_usd:.4f}")


    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "OpenAI"

    Needs `pip install "arc-agentkit[http]"` and `OPENAI_API_KEY`. The
    only lines that differ are the preset and the model name.

    ```python
    import asyncio
    import os

    from agentkit import Agent, Scope
    from agentkit.adapters.llm import providers
    from agentkit.middlewares import meter, retry, tracing
    from agentkit.runtime import Invoker, RunContext, Services


    async def main() -> None:
        llm = providers.openai(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
        )
        services = Services(
            invoker=Invoker(llm=llm, chat_middleware=[tracing(), meter(), retry()]),
        )
        ctx = RunContext(correlation_id="run-1", scope=Scope(), services=services)

        agent = Agent(
            name="briefer",
            model="gpt-4o-mini",
            prompt="Answer in one short sentence.",
        )
        result = await agent.run("What do we know about octopus cognition?", ctx)
        print(result.output)
        print(f"cost: ${result.usage.cost_usd:.4f}")


    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "Local claude CLI (no key)"

    Needs the [`claude` CLI](https://docs.claude.com/en/docs/claude-code)
    on `PATH` and one prior `claude login`. agentkit holds no key; the
    CLI's own auth is used.

    ```python
    import asyncio

    from agentkit import Agent, Scope
    from agentkit.agents.cognition import ClaudeCliCognition
    from agentkit.runtime import RunContext, Services


    async def main() -> None:
        agent = Agent(
            name="briefer",
            prompt="Answer in one short sentence.",
            cognition=ClaudeCliCognition(model="claude-sonnet-4-6"),
        )
        ctx = RunContext(correlation_id="run-1", scope=Scope(), services=Services())

        result = await agent.run("What do we know about octopus cognition?", ctx)
        print(result.output)
        print(f"cost estimate: ${result.usage.cost_usd:.4f}")


    if __name__ == "__main__":
        asyncio.run(main())
    ```

    `ClaudeCliCognition` subprocesses the CLI once per `agent.run(...)`
    and maps its stream-JSON output onto agentkit `StreamEvent`s. The
    cost is the CLI's own estimate from published per-token prices, not
    a billed figure — hence "cost estimate".

Two details worth noticing in the first two tabs:

- **`providers.claude(...)` returns an `LLMPort`, not a client you
  call.** It goes into an `Invoker`, and the `Invoker` is what walks the
  middleware chain on the way to it. `tracing()`, `meter()` and
  `retry()` are a plain Python list — reorder it by editing the line.
- **`meter()` is what makes a `Budget` real.** A `Budget` on the
  `RunContext` with no `meter()` in the chain is never charged and never
  halts anything. `make_test_ctx` and the presets in `agentkit.client`
  wire it for you; hand-built `Invoker`s do not.

## Where to go next

Pick by what you are trying to do.

- **"Show me the whole thing, one step at a time."** →
  [Tutorial](tutorial.md). Six steps, each a complete script, each
  motivated by a problem the previous step left open. The first five
  need no API key.
- **"I know what I want, just give me the invocation."** →
  [Cheatsheet](cheatsheet.md).
- **"How do I cap spend / pause for a human / survive a crash / wire
  OpenTelemetry / consume MCP?"** → [Recipes](recipes/index.md).
- **"What are the pieces actually called?"** →
  [Concepts](concepts/kernel.md), starting with
  [Runtime](concepts/runtime.md) for `RunContext` / `Budget` and
  [Agents](concepts/agents.md) for `Cognition` / `Autonomy`.
- **"Why is it built like this?"** → [Why agentkit](why.md).
- **"I have working code and I am about to deploy it."** →
  [Anti-patterns](anti-patterns.md). Read it once; it is the list of
  traps that cost other people money.
- **"I am coming from something else."** →
  [From LangChain](migrating/from-langchain.md) or
  [From vanilla asyncio](migrating/from-vanilla-asyncio.md).
