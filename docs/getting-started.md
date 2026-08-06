# Getting started

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

The core has **zero runtime dependencies**. Provider integrations live
behind opt-in extras — pull only what you need:

```bash
pip install "arc-agentkit[http]"            # bundled httpx-based LLM providers
pip install "arc-agentkit[postgres]"        # asyncpg + pgvector adapters
pip install "arc-agentkit[redis]"           # redis-backed store
pip install "arc-agentkit[observability]"   # OpenTelemetry exporter
pip install "arc-agentkit[mcp]"             # Model Context Protocol client
```

## Requirements

- Python 3.12 or newer.
- One of: an API key for an LLM provider (Anthropic, OpenAI, DeepSeek,
  OpenRouter are shipped as batteries-included presets), or the local
  `claude` CLI on your PATH (the `ClaudeCliCognition` plug-in delegates
  to it and needs no server-side key).

## Verify the install

```bash
python -c "import agentkit; print(agentkit.__version__)"
```

Prints the installed version. If it errors, the wheel isn't on your
Python path — check the interpreter your shell resolved.

## Where to go next

- **[Tutorial](tutorial.md)** — fifteen minutes, five steps, one
  runnable script per step. Builds a real research-briefing agent with
  a tool, a stream, a budget, and a human-approval gate.
- **[Landing](index.md)** — the four-theme grid (cognition · control ·
  state · behaviour) that names every primitive. Start here for the
  mental model.
- **[Why agentkit](why.md)** — twelve concrete guarantees and a
  side-by-side comparison with LangChain / LangGraph / Instructor /
  Pydantic-AI / hosted alternatives.
- **[Cheatsheet](cheatsheet.md)** — every primitive, tight code,
  skimmable in 90 seconds. Reach for it when you know what you want and
  just need the invocation.
