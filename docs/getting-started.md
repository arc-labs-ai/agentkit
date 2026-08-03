# Getting started

## Install

```bash
pip install arc-agentkit
```

!!! note "Distribution name vs import name"
    On PyPI the distribution is **`arc-agentkit`** (the `agentkit` name
    on PyPI is owned by an unrelated project). The **import name stays
    `agentkit`**, so all documentation and examples use `import
    agentkit` as-is. Follows the same pattern as `pip install Django`
    → `import django`.

    Before the first release, install directly from Git:

    ```bash
    pip install "arc-agentkit @ git+https://github.com/arc-labs-ai/agentkit"
    ```

The core has **zero runtime dependencies**. Provider integrations live
behind opt-in extras — pull only what you need:

```bash
pip install "arc-agentkit[http]"            # bundled httpx-based LLM providers
pip install "arc-agentkit[postgres]"        # asyncpg + pgvector adapters
pip install "arc-agentkit[redis]"           # redis-backed store
pip install "arc-agentkit[observability]"   # OpenTelemetry exporter
```

## Requirements

- Python 3.12 or newer.
- An API key for at least one LLM provider (Anthropic, OpenAI, DeepSeek,
  or OpenRouter are shipped as batteries-included presets).

## Your first agent

The shortest useful program is a batteries-included `Chat` call. It
already runs on a real `RunContext` with the standard
`tracing → meter → retry` middleware chain, so you get cost accounting
and structured retries for free:

```python
import asyncio
from agentkit import claude

async def main() -> None:
    async with claude(api_key="sk-...", model="claude-sonnet-4-6") as chat:
        result = await chat(
            "Give me three surprising facts about octopuses.",
            system="You reply in tight, cited bullets.",
        )
        print(result.content)
        print(f"tokens: {result.usage.total_tokens}   cost: ${result.usage.cost_usd:.4f}")

asyncio.run(main())
```

## Adding a tool

The moment you want the model to *do* something, drop from `Chat` to an
`Agent`. Register a plain async function as a tool and let a `ReAct`
cognition drive the loop:

```python
from agentkit import Agent, ReActCognition, tool

@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"{city}: 72F, clear"

agent = Agent(
    name="weather-bot",
    cognition=ReActCognition(),
    tools=[get_weather],
)
```

You then call `await agent.run(ctx, "What's the weather in Paris?")` on
a `RunContext` you build (or borrow from a `Chat`).

## Where to go next

- **[Concepts › Kernel](concepts/kernel.md)** — the value types every
  layer speaks.
- **[Concepts › Runtime](concepts/runtime.md)** — `RunContext`, `Invoker`,
  `Budget`, `Quota`.
- **[Concepts › Agents](concepts/agents.md)** — `Agent` / `Workflow` /
  `Cognition` and the control primitives.
- **[Examples](examples.md)** — end-to-end demos in the `examples/`
  folder.
