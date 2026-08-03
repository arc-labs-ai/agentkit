"""Example 01 — a single agent, driven by a fake LLM.

This is the README quickstart, expanded into a runnable script. It shows the
minimum viable wire-up:

- Build a `RunContext` bound to a `FakeLLM` (via `make_test_ctx`).
- Declare an `Agent` — name, model, prompt.
- Call `agent.run(task, ctx)` and read the typed `AgentResult`.

`FakeLLM` is a deterministic, offline-only LLM double from `agentkit.testing`.
It always returns the string it was constructed with, and it charges a small
scripted `Usage` on each call — enough to exercise the whole runtime without
an API key.

Run:
    uv run python examples/01_single_agent.py
"""

from __future__ import annotations

import asyncio

from agentkit.testing import FakeLLM, make_test_ctx

from agentkit import Agent


async def main() -> None:
    # `make_test_ctx` builds a REAL RunContext. Only the LLM is faked; the
    # invoker, budget, meters, and services are the same objects a real run
    # would use — no separate "test mode" branch in the framework.
    ctx = make_test_ctx(llm=FakeLLM("42"))

    agent = Agent(
        name="answerer",
        model="gpt-4o-mini",
        prompt="Answer the question in as few words as possible.",
    )

    result = await agent.run("what is 6 * 7?", ctx)

    print(f"output:        {result.output!r}")
    print(f"usage.tokens:  {result.usage.total_tokens}")
    print(f"usage.cost_usd: {result.usage.cost_usd}")


# Expected output:
#   output:        '42'
#   usage.tokens:  15
#   usage.cost_usd: 0.0001


if __name__ == "__main__":
    asyncio.run(main())
