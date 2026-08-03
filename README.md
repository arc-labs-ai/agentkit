<div align="center">

# agentkit

**A low-level, async-first framework for building AI agents in Python.**

[![CI](https://github.com/arc-labs-ai/agentkit/actions/workflows/ci.yml/badge.svg)](https://github.com/arc-labs-ai/agentkit/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/arc-agentkit)](https://pypi.org/project/arc-agentkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/arc-agentkit)](https://pypi.org/project/arc-agentkit/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

[Docs](https://arc-labs-ai.github.io/agentkit/) · [Quickstart](#quickstart) · [Examples](./examples) · [Changelog](./CHANGELOG.md)

</div>

---

## What is agentkit

agentkit is a **framework**, not an agent library. It gives you the primitives
to build your own agent — an `Agent` class, a `Cognition` strategy, a `Ctx`
that threads services + budgets + cancel through every call, a middleware
chain you compose yourself, and typed value types the whole stack shares.
What it does with an LLM, what tools it exposes, what it stores, when it
stops — every one of those decisions is yours.

It is not a batteries-included agent (no built-in personas, no bundled
prompts, no opinionated stack pushing you toward one vendor). It is not a
chain-builder or a graph DSL either. If you want scaffolding that hides the
loop, use one of those; agentkit hands you the loop and gets out of the way.

Use agentkit when you want an agent runtime whose seams are visible, whose
types survive `mypy --strict`, whose core has **zero runtime dependencies**,
and whose control plane (cancel, budget, retry, HITL suspend/resume) is
first-class instead of tacked on. Used in production at [Arc Labs](https://arc-labs.ai).

## Install

```bash
pip install arc-agentkit
```

Requires Python 3.12+. The core has no runtime dependencies. Concrete
adapters live behind optional extras — install only what you use:

```bash
pip install "arc-agentkit[http]"           # httpx-based LLM providers (OpenAI, Anthropic, DeepSeek, ...)
pip install "arc-agentkit[postgres]"       # Postgres checkpoint store + pgvector memory
pip install "arc-agentkit[redis]"          # Redis-backed store
pip install "arc-agentkit[observability]"  # OpenTelemetry metrics + tracing exporters
pip install "arc-agentkit[fast]"           # orjson for hot-path JSON
```

## Quickstart

Define an agent, invoke it via a `Ctx` bound to a fake LLM, get a typed
`AgentResult`:

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
    print(result.output)         # -> "42"
    print(result.usage)          # -> Usage(input_tokens=10, output_tokens=5, cost_usd=0.0001, cache_read_tokens=0, cache_write_tokens=0)


if __name__ == "__main__":
    asyncio.run(main())
```

Point the same code at a real provider by using the batteries-included
`claude` chat directly:

```python
from agentkit import claude

async with claude(api_key="sk-...", model="claude-sonnet-4-6") as chat:
    result = await chat("what is 6 * 7?", system="Answer in as few words as possible.")
    print(result.content)
```

More runnable examples — tool loops, streaming, composed middlewares —
under [`examples/`](./examples).

## Concepts

Five layers, one direction of dependency. Each is a link into the docs where
depth lives:

- **[Kernel](https://arc-labs-ai.github.io/agentkit/kernel/)** — opinion-free
  primitives: immutable value types (`Message`, `ToolCall`, `LLMResult`,
  `Usage`), the port Protocols every seam (`LLMPort`, `StorePort`,
  `VectorPort`, …) implements, the middleware contract, and concurrency +
  resilience helpers. Zero third-party imports.
- **[Runtime](https://arc-labs-ai.github.io/agentkit/runtime/)** —
  `RunContext` (identity + services + cancel + budget), `Invoker` (the one
  runner that walks the middleware chain to a terminal LLM/tool call),
  `Budget` and `Quota` (spend meters that halt runs cleanly on overrun).
- **[Agent + Cognition](https://arc-labs-ai.github.io/agentkit/agents/)** —
  `Agent` is identity + chat-call config; `Cognition` is the pluggable
  turn-taking strategy. Ships with `SingleCallCognition`, `ReActCognition`
  (tool loop + HITL suspend + durable resume), and `CoordinatorCognition`
  (multi-agent orchestration). Adding a new regime is one Protocol impl —
  no `Agent` subclassing.
- **[Middleware](https://arc-labs-ai.github.io/agentkit/middlewares/)** —
  cross-cutting concerns as composable functions: `tracing`, `retry`,
  `fallback`, `memoize`, `output_coerce`, `meter`, `compaction`, `egress`,
  `audit`, `security`. Two ordered lists (chat + tool) handed to `Invoker`;
  reorder or swap by editing the list.
- **[Capabilities](https://arc-labs-ai.github.io/agentkit/capabilities/)** —
  optional cross-cutting collaborators: `RequestBuilder` (prompt +
  grounding), `Compactor` (four strategies), `Guardrail`, `Evaluator`,
  `Checkpointer`, `SchemaAdapter` (structured outputs across Pydantic,
  attrs, dataclass, or raw JSON Schema).

The bet: every seam is a typed `Protocol` injected at wire-up, so the loop
stays legible and unit-testable, and swapping a backend never edits the loop.

## What makes agentkit distinct

- **Zero runtime dependencies in the core.** Install `arc-agentkit` alone
  and you have a working framework — no `pydantic`, no `httpx`, no vendor
  SDK. Extras (`http`, `postgres`, `redis`, `observability`) are opt-in.
- **Ports and adapters, all the way down.** Every I/O seam is a `Protocol`.
  Swap the LLM, the store, the vector DB, the checkpointer, the tracer,
  the metrics port — the loop doesn't move.
- **Typed public surface, `mypy --strict` clean.** Every re-exported name
  in `agentkit` carries full type information. `py.typed` ships in the
  wheel. Frozen dataclasses for the value layer.
- **Human-in-the-loop is a real pause.** A gated agent suspends to a
  checkpoint; a fresh worker (potentially in a new process, weeks later)
  can `resume(run_id, decisions)` and continue. No in-memory state, no
  polling.
- **Cost you can trust.** `Budget` charges every LLM call under a lock;
  totals are invariant under concurrent workers. Hit the ceiling and the
  run stops with a `MeterExceeded`, not silently over-spends.
- **A first-class testing kit.** `FakeLLM`, `FakeFetch`, `FakeSearch`,
  `FakeMemory`, `FakeTool`, plus `make_test_ctx()` — the same doubles
  the framework's own suite uses. Zero API keys required to unit-test
  your agents end to end.

## When agentkit is (probably) not the fit

- You want a graph DSL that draws the flow for you — reach for LangGraph.
- You want a batteries-included agent product (built-in personas, hosted
  memory, one-click deploy) — reach for a hosted framework.
- You want to prototype in a notebook and never leave — anything works;
  agentkit's benefits (typed seams, cancel/budget, checkpointing) only
  pay off when the code has to survive production.

## Documentation

- **Site**: [arc-labs-ai.github.io/agentkit](https://arc-labs-ai.github.io/agentkit/)
- **Examples**: [`examples/`](./examples) — runnable, no API keys required.
- **Mental models**: [`docs/mental-models/`](./docs/mental-models/) —
  four end-to-end use cases with narrative walkthroughs of internal state
  at each step (multi-tenant chat with docs, autonomous DevOps investigator,
  long-running data enrichment, coordinated multi-agent research).
- **Changelog**: [`CHANGELOG.md`](./CHANGELOG.md).

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the
development setup, coding conventions (`ruff` + `mypy --strict` + `pytest`
must all pass), and the PR checklist.

Quick local loop:

```bash
git clone https://github.com/arc-labs-ai/agentkit
cd agentkit
uv sync                                 # install deps + dev group
uv run pytest                           # full test suite
uv run mypy agentkit                    # strict typecheck
uv run ruff check .                     # lint
```

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

Copyright (c) Arc Labs.
