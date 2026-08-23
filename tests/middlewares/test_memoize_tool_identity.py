"""`memoize()`'s default key must include the TOOL's identity.

`memoize.py`'s own module docstring advertises the middleware for "read-only
tool reuse", but `default_key` hashed only chat fields — model / messages /
tools / temperature / max_tokens / response_format. A `ToolRequest` has none of
them, so every tool call in a scope hashed to one key.

Measured before the fix: `weather(SF)` ran once, then `stock(AAPL)` returned
`{'city': 'SF'}` and `weather(NYC)` returned `{'city': 'SF'}` too; the stock
tool's execution list was `[]`. A cache that answers one tool with another
tool's result is strictly worse than no cache.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentkit.adapters.store.memory import InMemoryStore
from agentkit.kernel.types import ChatRequest, Message, Scope, ToolRequest
from agentkit.middlewares import memoize
from agentkit.middlewares.memoize import default_key
from agentkit.runtime import Budget, Invoker, RunContext, Services


class _EchoTool:
    """Echoes its arguments back, tagged with the tool name — so a collision is
    unmistakable in the returned value AND in the execution log."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.executions: list[dict[str, Any]] = []

    async def run(self, arguments: dict[str, Any], ctx: Any) -> dict[str, Any]:
        self.executions.append(dict(arguments))
        return {"tool": self.name, **arguments}


def _wire() -> tuple[Invoker, RunContext]:
    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=[memoize(store=store)])
    ctx = RunContext("run", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
    return inv, ctx


def _call(inv: Invoker, ctx: RunContext, tool: _EchoTool, **arguments: Any) -> Any:
    return asyncio.run(inv.invoke_tool(ToolRequest(tool.name, arguments, tool), ctx))


def test_two_different_tools_do_not_share_a_cache_entry() -> None:
    """The load-bearing case: `weather` ran first, so `stock` was served
    `weather`'s answer and never executed at all."""
    inv, ctx = _wire()
    weather, stock = _EchoTool("weather"), _EchoTool("stock")

    async def go() -> tuple[Any, Any]:
        a = await inv.invoke_tool(ToolRequest("weather", {"city": "SF"}, weather), ctx)
        b = await inv.invoke_tool(ToolRequest("stock", {"ticker": "AAPL"}, stock), ctx)
        return a, b

    a, b = asyncio.run(go())

    assert a == {"tool": "weather", "city": "SF"}
    assert b == {"tool": "stock", "ticker": "AAPL"}, "one tool was served another tool's cached result"
    assert stock.executions == [{"ticker": "AAPL"}], "the second tool never ran"


def test_two_different_tools_with_IDENTICAL_arguments_do_not_collide() -> None:
    """The nastiest shape of the same bug: identical argument dicts make the
    arguments alone insufficient — the tool NAME has to be in the key."""
    inv, ctx = _wire()
    convert, translate = _EchoTool("convert"), _EchoTool("translate")

    async def go() -> tuple[Any, Any]:
        a = await inv.invoke_tool(ToolRequest("convert", {"value": "hello"}, convert), ctx)
        b = await inv.invoke_tool(ToolRequest("translate", {"value": "hello"}, translate), ctx)
        return a, b

    a, b = asyncio.run(go())

    assert a["tool"] == "convert"
    assert b["tool"] == "translate", "identical arguments collapsed two distinct tools into one entry"
    assert translate.executions == [{"value": "hello"}]


def test_the_same_tool_with_different_arguments_is_not_a_hit() -> None:
    """`weather(NYC)` came back as `{'city': 'SF'}` before the fix."""
    inv, ctx = _wire()
    weather = _EchoTool("weather")

    async def go() -> tuple[Any, Any]:
        a = await inv.invoke_tool(ToolRequest("weather", {"city": "SF"}, weather), ctx)
        b = await inv.invoke_tool(ToolRequest("weather", {"city": "NYC"}, weather), ctx)
        return a, b

    a, b = asyncio.run(go())

    assert a == {"tool": "weather", "city": "SF"}
    assert b == {"tool": "weather", "city": "NYC"}
    assert weather.executions == [{"city": "SF"}, {"city": "NYC"}]


def test_the_same_tool_with_the_same_arguments_IS_reused() -> None:
    """POSITIVE CONTROL. Partitioning by tool identity must not amount to
    switching the cache off: a repeat of the exact same call still executes the
    tool ONCE. A "fix" that simply skips tool calls fails right here."""
    inv, ctx = _wire()
    weather = _EchoTool("weather")

    async def go() -> tuple[Any, Any]:
        a = await inv.invoke_tool(ToolRequest("weather", {"city": "SF"}, weather), ctx)
        b = await inv.invoke_tool(ToolRequest("weather", {"city": "SF"}, weather), ctx)
        return a, b

    a, b = asyncio.run(go())

    assert a == b == {"tool": "weather", "city": "SF"}
    assert weather.executions == [{"city": "SF"}], "an identical read-only tool call was not memoized"


def test_a_tool_key_can_never_alias_a_chat_key() -> None:
    """Different key SPACES, not just different inputs — a tool key and a chat
    key are namespaced apart so no hash arrangement can bring them together."""
    tool_call = type("C", (), {"kind": "tool", "request": ToolRequest("x", {}, None)})()
    chat_call = type(
        "C", (), {"kind": "chat", "request": ChatRequest([Message("user", "hi")], "m")}
    )()

    assert default_key(tool_call).startswith("memo:tool:")
    assert not default_key(chat_call).startswith("memo:tool:")
    assert default_key(tool_call) != default_key(chat_call)


def test_the_default_key_is_stable_for_the_same_tool_call() -> None:
    """A key that isn't stable never hits — the other half of the contract that
    `test_the_same_tool_with_the_same_arguments_IS_reused` exercises end to end."""

    def key_for(name: str, arguments: dict[str, Any]) -> str:
        return default_key(type("C", (), {"kind": "tool", "request": ToolRequest(name, arguments, None)})())

    assert key_for("weather", {"city": "SF"}) == key_for("weather", {"city": "SF"})
    # ...and argument ORDER is not identity (the hash sorts keys).
    assert key_for("f", {"a": 1, "b": 2}) == key_for("f", {"b": 2, "a": 1})
    assert key_for("weather", {"city": "SF"}) != key_for("weather", {"city": "NYC"})
    assert key_for("weather", {"city": "SF"}) != key_for("stock", {"city": "SF"})


def test_tool_entries_are_still_partitioned_by_tenant() -> None:
    """The tool key space inherits `_scoped()` like every other key — a tool
    result must not cross an org boundary."""
    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=[memoize(store=store)])
    tool = _EchoTool("lookup")

    async def go() -> tuple[Any, Any]:
        a_ctx = RunContext("run", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
        b_ctx = RunContext("run", Scope(org_id=999), Budget(), Services(invoker=inv, store=store))
        a = await inv.invoke_tool(ToolRequest("lookup", {"q": "revenue"}, tool), a_ctx)
        b = await inv.invoke_tool(ToolRequest("lookup", {"q": "revenue"}, tool), b_ctx)
        return a, b

    asyncio.run(go())
    assert len(tool.executions) == 2, "a tool result crossed a tenant boundary"
