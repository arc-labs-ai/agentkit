"""A one-agent, offline harness using the canonical AgentRunner and CallRuntime."""

from __future__ import annotations

import asyncio

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet
from agentkit.core.testing import scripted_runtime
from agentkit.models import Binding, Mode, PriceTable
from agentkit.models.contracts.types import Turn

PRICES = PriceTable(
    "example",
    "2026-10-13",
    {"test-model": {"input_uncached": "1", "input_cache_read": "0.1", "input_cache_write_5m": "1.25",
                    "input_cache_write_1h": "2", "output": "5"}},
)


async def main() -> None:
    async with scripted_runtime(
        [Turn(messages=("42",), raw_usage={"input_tokens": 10, "output_tokens": 5})], PRICES
    ) as runtime:
        runner = AgentRunner(runtime)
        answer = await runner.run_agent(
            AgentDefinition("answerer", "1", "Answer briefly."),
            InstructionSet("What is 6 times 7?"),
            {},
            binding=Binding(Mode.OPENAI_API, "test-model"),
            call_id="single-agent-example",
        )
        print(f"output:         {answer.text!r}")
        input_tokens = answer.usage.input_uncached + answer.usage.input_cache_read + answer.usage.input_cache_write_5m + answer.usage.input_cache_write_1h
        print(f"usage.tokens:   {input_tokens + answer.usage.output}")
        print(f"usage.cost_usd: {answer.cost_usd}")


if __name__ == "__main__":
    asyncio.run(main())
