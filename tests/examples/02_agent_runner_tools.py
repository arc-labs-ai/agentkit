"""A ReAct harness with host-authorized tools and a scripted provider."""

import asyncio

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet, ToolGrant
from agentkit.core.runtime import CallRuntime
from agentkit.core.testing import ScriptedAdapter
from agentkit.models import Binding, Mode, PriceTable
from agentkit.models.contracts.types import Turn
from agentkit.tools import FunctionTool, ToolRegistry

PRICES = PriceTable(
    "example",
    "2026-10-13",
    {"test-model": {"input_uncached": "1", "input_cache_read": "0.1", "input_cache_write_5m": "1.25",
                    "input_cache_write_1h": "2", "output": "5"}},
)


async def main() -> None:
    async def lookup_order(order_id: str) -> str:
        """Look up an order in the host's authorized view."""
        return {"order-42": "Captured twice; no refund issued."}[order_id]

    tools = ToolRegistry.from_tools([FunctionTool.from_callable(lookup_order, side_effecting=False)])
    adapter = ScriptedAdapter([
        Turn(tool_calls=({"id": "lookup-1", "name": "lookup_order", "arguments": {"order_id": "order-42"}},),
             raw_usage={"input_tokens": 12, "output_tokens": 3}),
        Turn(("The order was captured twice, and no refund has been issued yet.",),
             raw_usage={"input_tokens": 20, "output_tokens": 15}),
    ])
    async with CallRuntime(adapters={Mode.OPENAI_API: adapter}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, tools=tools)
        agent = AgentDefinition(
            id="support", version="1", description="Answer order questions.",
            tools=ToolGrant(allow=("lookup_order",)),
        )
        answer = await runner.run_agent(
            agent, InstructionSet("Check order-42 and state what happened."), {},
            binding=Binding(Mode.OPENAI_API, "test-model"), call_id="support-42",
        )
        print(answer.text)
        print(f"outcome={answer.outcome}, turns={len(adapter.requests)}")


if __name__ == "__main__":
    asyncio.run(main())
