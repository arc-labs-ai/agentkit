from __future__ import annotations

from decimal import Decimal

import pytest

from agentkit.core.testing import ScriptedAdapterExhausted, scripted_runtime
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.types import Binding, Turn

PRICES = PriceTable(
    "scripted",
    "2026-10-13",
    {"scripted-model": {"input_uncached": "1", "output": "2"}},
)


@pytest.mark.asyncio
async def test_scripted_runtime_exposes_adapter_requests_and_answers():
    runtime = scripted_runtime([Turn(messages=("done",), raw_usage={"input_tokens": 1, "output_tokens": 1})], PRICES)
    async with runtime:
        answer = await runtime.ask(Binding("openai_api", "scripted-model"), "task", call_id="scripted")
    assert answer.text == "done"
    assert answer.cost_usd == Decimal("0.000003")


@pytest.mark.asyncio
async def test_scripted_adapter_exhaustion_is_not_converted_into_a_fake_answer():
    runtime = scripted_runtime([], PRICES)
    with pytest.raises(ScriptedAdapterExhausted):
        async with runtime:
            await runtime.ask(Binding("openai_api", "scripted-model"), "task", call_id="exhausted")
