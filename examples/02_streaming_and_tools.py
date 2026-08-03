"""Example 02 — a ReAct agent with a tool + streaming output.

Demonstrates:

- `@tool(side_effecting=...)` turns a plain async function into a `FunctionTool`.
  `side_effecting` is required — the loop's HITL gating and idempotency logic
  depend on knowing whether a tool mutates the world.
- `ReActCognition(tools=[...])` swaps `Agent`'s default `SingleCallCognition`
  for the tool-use loop. A plain list of tools is wrapped into a
  `ToolRegistry` automatically.
- `FakeLLM.script([Turn(...), ...])` replays a scripted multi-turn tool loop
  across successive chat calls: turn 1 requests the tool, turn 2 returns the
  final answer.
- `agent.stream(task, ctx)` yields typed `StreamEvent`s as the run unfolds
  (`message_delta` per token, `tool_call` / `tool_result` per tool invocation,
  terminal `final`). `agent.run(...)` is just this stream collected.

Run:
    uv run python examples/02_streaming_and_tools.py
"""

from __future__ import annotations

import asyncio

from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx

from agentkit import Agent, ToolCall, tool


@tool(side_effecting=False)
async def get_weather(city: str) -> str:
    """Return the current weather for `city`. Read-only stub for the example."""
    return f"weather in {city}: 22C, clear skies, wind 6 km/h"


async def main() -> None:
    # Scripted LLM: turn 1 asks for the tool, turn 2 speaks the final answer.
    # The framework routes the tool call, injects the tool result into the
    # transcript, and re-invokes the LLM — the fake replays turn 2.
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "get_weather", {"city": "Paris"}),)),
            Turn(content="Paris is 22C and clear right now."),
        ]
    )
    ctx = make_test_ctx(llm=llm)

    agent = Agent(
        name="weatherbot",
        model="gpt-4o-mini",
        prompt="Answer weather questions. Use the get_weather tool when needed.",
        cognition=ReActCognition(tools=[get_weather]),
    )

    print("streaming events from agent.stream(...):")
    async for ev in agent.stream("How is the weather in Paris?", ctx):
        if ev.type == "message_delta":
            print(ev.text, end="", flush=True)
        elif ev.type == "tool_call":
            args = dict(ev.tool_call.arguments)
            print(f"\n  [tool_call] {ev.tool_call.name}({args})")
        elif ev.type == "tool_result":
            print(f"  [tool_result] {ev.tool_result}")
        elif ev.type == "step":
            print(f"  [step] {ev.text}")
        elif ev.type == "final":
            print(f"\n  [final] output={ev.result.output!r} usage={ev.result.usage}")


# Expected output (line ordering may vary slightly for the interleaved deltas):
#   streaming events from agent.stream(...):
#     [tool_call] get_weather({'city': 'Paris'})
#     [tool_result] weather in Paris: 22C, clear skies, wind 6 km/h
#     [step] iteration:1
#   Paris is 22C and clear right now.
#     [final] output='Paris is 22C and clear right now.' usage=Usage(...)


if __name__ == "__main__":
    asyncio.run(main())
