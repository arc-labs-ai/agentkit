"""Unit tests for the FakeMemory and FakeTool test doubles.

These cover the user-facing contract: static / callable responders for
FakeTool, query lookup + items-override + write recording for FakeMemory,
and — most importantly — that both fakes satisfy the runtime-checkable
Protocols they're meant to stub (so consumers can drop them in anywhere a
``Tool`` or ``MemorySource`` is required).
"""

from __future__ import annotations

import pytest

from agentkit.memory.base import MemoryItem, MemorySource
from agentkit.testing import FakeCtx, FakeMemory, FakeTool
from agentkit.tools.base import Tool

# ----------------------------- FakeTool ----------------------------------


@pytest.mark.asyncio
async def test_fake_tool_responder_static_value() -> None:
    tool = FakeTool(responder={"result": 42})
    ctx = FakeCtx()

    out = await tool.run({"x": 1}, ctx)

    assert out == {"result": 42}


@pytest.mark.asyncio
async def test_fake_tool_responder_callable() -> None:
    tool = FakeTool(responder=lambda args: args["x"] * 2)
    ctx = FakeCtx()

    out = await tool.run({"x": 21}, ctx)

    assert out == 42


@pytest.mark.asyncio
async def test_fake_tool_responder_async_callable() -> None:
    async def responder(args: dict) -> int:
        return args["x"] + 1

    tool = FakeTool(responder=responder)
    ctx = FakeCtx()

    out = await tool.run({"x": 5}, ctx)

    assert out == 6


@pytest.mark.asyncio
async def test_fake_tool_responder_callable_raises() -> None:
    def boom(args: dict) -> None:
        raise RuntimeError("nope")

    tool = FakeTool(responder=boom)
    ctx = FakeCtx()

    with pytest.raises(RuntimeError, match="nope"):
        await tool.run({}, ctx)


@pytest.mark.asyncio
async def test_fake_tool_records_calls() -> None:
    tool = FakeTool(responder="ok")
    ctx = FakeCtx()

    await tool.run({"x": 1}, ctx)
    await tool.run({"x": 2}, ctx)
    await tool.run({"x": 3}, ctx)

    assert tool.calls == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_fake_tool_conforms_to_protocol() -> None:
    tool = FakeTool()
    assert isinstance(tool, Tool)


# ----------------------------- FakeMemory --------------------------------


@pytest.mark.asyncio
async def test_fake_memory_query_lookup() -> None:
    item = MemoryItem(content="cats are great", source="fake_memory")
    mem = FakeMemory(responses={"cats": [item]})
    ctx = FakeCtx()

    out = await mem.query("cats", k=3, ctx=ctx)

    assert out == [item]


@pytest.mark.asyncio
async def test_fake_memory_default_fallback() -> None:
    default_item = MemoryItem(content="default", source="fake_memory")
    mem = FakeMemory(default=[default_item])
    ctx = FakeCtx()

    out = await mem.query("unknown-query", k=3, ctx=ctx)

    assert out == [default_item]


@pytest.mark.asyncio
async def test_fake_memory_default_empty_when_miss() -> None:
    mem = FakeMemory()
    ctx = FakeCtx()

    out = await mem.query("anything", k=3, ctx=ctx)

    assert out == []


@pytest.mark.asyncio
async def test_fake_memory_items_override() -> None:
    items = [
        MemoryItem(content="a", source="fake_memory"),
        MemoryItem(content="b", source="fake_memory"),
        MemoryItem(content="c", source="fake_memory"),
    ]
    mem = FakeMemory(items=items)
    ctx = FakeCtx()

    # ``items`` overrides — even an unknown query returns the static fixture.
    out = await mem.query("anything-at-all", k=2, ctx=ctx)

    assert out == items[:2]


@pytest.mark.asyncio
async def test_fake_memory_records_queries_and_writes() -> None:
    mem = FakeMemory()
    ctx = FakeCtx()

    await mem.query("foo", k=5, ctx=ctx)
    await mem.query("bar", k=3, ctx=ctx, where={"author": "neo"})

    item = MemoryItem(content="written", source="fake_memory")
    await mem.write([item], ctx=ctx)
    await mem.write([item, item], ctx=ctx)

    assert mem.queries == [
        ("foo", 5, None),
        ("bar", 3, {"author": "neo"}),
    ]
    assert mem.writes == [[item], [item, item]]


def test_fake_memory_conforms_to_protocol() -> None:
    mem = FakeMemory()
    assert isinstance(mem, MemorySource)
