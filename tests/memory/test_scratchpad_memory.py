"""``ScratchpadMemory`` — ``MemorySource`` over ``WorkingContext.scratchpad``.

Pins the key-or-value lexical match, the write-through to the
underlying scratchpad, and the metadata-key addressing convention.
"""

from __future__ import annotations

import asyncio

from agentkit.context.context import WorkingContext
from agentkit.memory import MemoryItem, ScratchpadMemory
from agentkit.testing import make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def test_query_matches_keys():
    wc = WorkingContext()
    wc.scratchpad["sub_question_count"] = 3
    wc.scratchpad["topic"] = "elephants"

    mem = ScratchpadMemory(context=wc)
    ctx = make_test_ctx()
    items = _run(mem.query("question", k=10, ctx=ctx))

    keys = sorted(i.metadata["key"] for i in items)
    assert keys == ["sub_question_count"]
    assert items[0].content == "3"
    assert items[0].source == "scratchpad"
    assert items[0].score is None


def test_query_matches_values():
    wc = WorkingContext()
    wc.scratchpad["a"] = "elephants migrate"
    wc.scratchpad["b"] = "cats sleep"
    wc.scratchpad["c"] = "elephants again"

    mem = ScratchpadMemory(context=wc)
    ctx = make_test_ctx()
    items = _run(mem.query("elephants", k=10, ctx=ctx))

    keys = sorted(i.metadata["key"] for i in items)
    assert keys == ["a", "c"]


def test_query_caps_results_at_k():
    wc = WorkingContext()
    for i in range(5):
        wc.scratchpad[f"key{i}"] = "needle"

    mem = ScratchpadMemory(context=wc)
    ctx = make_test_ctx()
    items = _run(mem.query("needle", k=2, ctx=ctx))
    assert len(items) == 2


def test_write_uses_metadata_key():
    wc = WorkingContext()
    mem = ScratchpadMemory(context=wc)
    ctx = make_test_ctx()

    _run(
        mem.write(
            [
                MemoryItem(
                    content="planner output",
                    source="x",
                    metadata={"key": "plan"},
                )
            ],
            ctx=ctx,
        )
    )

    assert wc.scratchpad["plan"] == "planner output"


def test_write_falls_back_to_truncated_content_key():
    wc = WorkingContext()
    mem = ScratchpadMemory(context=wc)
    ctx = make_test_ctx()
    long_content = "x" * 100

    _run(mem.write([MemoryItem(content=long_content, source="x")], ctx=ctx))

    # Key is the first 32 chars of content.
    assert wc.scratchpad[long_content[:32]] == long_content
