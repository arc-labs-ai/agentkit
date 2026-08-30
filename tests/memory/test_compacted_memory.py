"""``CompactedMemory`` — shrink results from any source via a ``Compactor``.

These tests pin:

1. Each returned item's ``content`` is reduced by the wrapped compactor.
2. ``max_items`` caps the post-compaction list.
3. ``write`` delegates to the inner source unchanged.
4. Score / metadata / source attribution are preserved across the wrap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message
from agentkit.memory import CompactedMemory, MemoryItem
from agentkit.testing import make_test_ctx


def _run(coro):
    return asyncio.run(coro)


# ── fakes ─────────────────────────────────────────────────────────────────


@dataclass
class _FixedMemory:
    """A MemorySource that returns a fixed list of items, recording writes."""

    items: list[MemoryItem]
    writes: list[list[MemoryItem]] = field(default_factory=list)
    name: str = "fixed"

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del query, ctx, where
        return list(self.items[:k])

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        del ctx
        self.writes.append(list(items))


@dataclass
class _ShorteningCompactor:
    """Compactor that deterministically shortens each input message's content."""

    keep: int = 5  # number of characters to keep

    async def compact(self, messages: list[Message], ctx: Any) -> list[Message]:
        del ctx
        return [Message(m.role, (m.content or "")[: self.keep] + "…") for m in messages]


# ── tests ─────────────────────────────────────────────────────────────────


def test_compactor_shortens_each_item_content():
    """The compactor runs over each item; resulting ``content`` is shorter."""
    inner = _FixedMemory(
        items=[
            MemoryItem(content="x" * 100, source="inner"),
            MemoryItem(content="y" * 100, source="inner"),
        ]
    )
    mem = CompactedMemory(inner=inner, compactor=_ShorteningCompactor(keep=5))
    ctx = make_test_ctx()
    out = _run(mem.query("q", k=5, ctx=ctx))

    assert len(out) == 2
    # 5 kept chars + ellipsis = 6 chars, vs. the 100-char original
    assert all(len(item.content) < 10 for item in out)
    assert out[0].content == "xxxxx…"
    assert out[1].content == "yyyyy…"


def test_max_items_caps_the_result():
    """``max_items`` caps the post-compaction list."""
    inner = _FixedMemory(
        items=[MemoryItem(content=f"item-{i}" * 10, source="inner") for i in range(5)]
    )
    mem = CompactedMemory(
        inner=inner,
        compactor=_ShorteningCompactor(keep=3),
        max_items=2,
    )
    ctx = make_test_ctx()
    out = _run(mem.query("q", k=10, ctx=ctx))
    assert len(out) == 2


def test_write_delegates_to_inner():
    """``CompactedMemory.write`` passes items through to the inner source."""
    inner = _FixedMemory(items=[])
    mem = CompactedMemory(inner=inner, compactor=_ShorteningCompactor())
    ctx = make_test_ctx()
    payload = [MemoryItem(content="abc", source="x")]
    _run(mem.write(payload, ctx=ctx))
    assert inner.writes == [payload]


def test_score_and_metadata_preserved():
    """Compaction touches ``content`` only — score, metadata, source kept."""
    inner = _FixedMemory(
        items=[
            MemoryItem(
                content="lengthy text",
                source="inner-src",
                score=0.42,
                metadata={"url": "u"},
            )
        ]
    )
    mem = CompactedMemory(inner=inner, compactor=_ShorteningCompactor(keep=4))
    ctx = make_test_ctx()
    out = _run(mem.query("q", k=1, ctx=ctx))
    assert out[0].score == 0.42
    assert out[0].metadata == {"url": "u"}
    assert out[0].source == "inner-src"


def test_compaction_carries_the_record_id_through():
    """``CompactedMemory`` rebuilds every item field by field, so a field it
    forgets is a field that silently disappears behind the wrapper. ``id`` is
    the one that costs an answer: a compacted source inside a
    ``CompositeMemory`` would fall back to a content digest, and compaction is
    exactly what makes two copies of one fact stop matching on content."""
    inner = _FixedMemory(items=[MemoryItem(content="a long chunk", source="fixed", id="r7", score=0.4)])
    mem = CompactedMemory(inner=inner, compactor=_ShorteningCompactor(keep=4))
    out = _run(mem.query("q", k=5, ctx=make_test_ctx()))
    assert [i.id for i in out] == ["r7"]
