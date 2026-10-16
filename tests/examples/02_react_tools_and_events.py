"""A ReAct agent with a host tool, using the public AgentRunner event contract.

Run with::

    uv run python tests/examples/02_react_tools_and_events.py
"""

from __future__ import annotations

import asyncio

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet, ToolGrant
from agentkit.core.testing import scripted_runtime
from agentkit.models import Binding, Mode, PriceTable
from agentkit.models.contracts.constants import EventKind, MindKind
from agentkit.models.contracts.types import CallEvent, Turn
from agentkit.tools import FunctionTool, ToolRegistry

PRICES = PriceTable(
    "example",
    "2026-10-13",
    {"test-model": {"input_uncached": "1", "input_cache_read": "0.1", "input_cache_write_5m": "1.25",
                    "input_cache_write_1h": "2", "output": "5"}},
)


async def get_weather(city: str) -> str:
    """Return the current weather for a city from the host's read-only weather source."""
    return f"weather in {city}: 22C, clear skies, wind 6 km/h"


async def main() -> None:
    tools = ToolRegistry.from_tools([FunctionTool.from_callable(get_weather, side_effecting=False)])
    adapter_turns = [
        Turn(
            tool_calls=({"id": "weather-1", "name": "get_weather", "arguments": {"city": "Paris"}},),
            raw_usage={"input_tokens": 12, "output_tokens": 3},
        ),
        Turn(
            messages=("Paris is 22C and clear right now.",),
            raw_usage={"input_tokens": 20, "output_tokens": 15},
        ),
    ]
    events: list[CallEvent] = []
    async with scripted_runtime(adapter_turns, PRICES) as runtime:
        runner = AgentRunner(runtime, tools=tools)
        answer = await runner.run_agent(
            AgentDefinition(
                "weatherbot",
                "1",
                "Answer weather questions using the authorized weather lookup tool.",
                mind=MindKind.REACT,
                tools=ToolGrant(allow=("get_weather",)),
            ),
            InstructionSet("How is the weather in Paris?"),
            {},
            binding=Binding(Mode.OPENAI_API, "test-model"),
            call_id="weather-example",
            events=events.append,
        )

    for event in events:
        if event.kind == EventKind.TOOL_CALL:
            print(f"[tool_call] {event.fields.get('name')} (arguments masked by default)")
        elif event.kind == EventKind.TOOL_RESULT:
            print(
                f"[tool_result] size={event.fields.get('result_size')} "
                f"digest={event.fields.get('result_digest')}"
            )
        elif event.kind == EventKind.MESSAGE_DELTA:
            print(f"[message_delta] {event.fields.get('length')} characters")
    print(f"[final] {answer.text}")


if __name__ == "__main__":
    asyncio.run(main())
