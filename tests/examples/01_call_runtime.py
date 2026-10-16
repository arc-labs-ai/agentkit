"""A model-neutral one-shot call through the canonical CallRuntime."""

import asyncio

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
    async with scripted_runtime([Turn(messages=("42",), raw_usage={"input_tokens": 10, "output_tokens": 5})], PRICES) as runtime:
        answer = await runtime.ask(Binding(Mode.OPENAI_API, "test-model"), "What is 6 times 7?", call_id="quickstart")
        print(answer.text)
        print(answer.usage)
        print(answer.cost_usd)


if __name__ == "__main__":
    asyncio.run(main())
