"""``CompositeMemory.write`` partial-failure semantics.

If one backend raises after another has already committed, the caller
must see a signal that some backends succeeded. ``CompositeMemory.write``
runs every write to completion and raises a structured
:class:`CompositeWriteError` naming which backends accepted and which
failed, so callers can compensate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentkit.memory.base import MemoryItem
from agentkit.memory.composite import CompositeMemory, CompositeWriteError


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _MockSource:
    """Test double — records every write it accepts; can be configured
    to raise."""

    name: str
    raise_on_write: BaseException | None = None
    received: list[MemoryItem] = field(default_factory=list)

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Any,
        where: dict | None = None,
    ) -> list[MemoryItem]:  # noqa: ARG002 — read-side not under test
        return []

    async def write(self, items: Iterable[MemoryItem], *, ctx: Any) -> None:  # noqa: ARG002
        if self.raise_on_write is not None:
            raise self.raise_on_write
        self.received.extend(items)


def _items() -> list[MemoryItem]:
    return [MemoryItem(content="a", source="t"), MemoryItem(content="b", source="t")]


def test_composite_write_happy_path_unchanged():
    """Every source succeeds → no exception, every backend got the items."""
    a = _MockSource("a")
    b = _MockSource("b")
    cm = CompositeMemory(sources=[a, b])

    _run(cm.write(_items(), ctx=None))

    assert [m.content for m in a.received] == ["a", "b"]
    assert [m.content for m in b.received] == ["a", "b"]


def test_composite_write_raises_structured_error_on_one_failure():
    """Source ``b`` raises; source ``a`` commits. The caller sees a
    ``CompositeWriteError`` naming both — they know to retry only ``b``
    and that ``a`` is already written."""
    a = _MockSource("a")
    b = _MockSource("b", raise_on_write=RuntimeError("backend offline"))
    c = _MockSource("c")
    cm = CompositeMemory(sources=[a, b, c])

    with pytest.raises(CompositeWriteError) as ei:
        _run(cm.write(_items(), ctx=None))

    err = ei.value
    assert sorted(err.accepted) == ["a", "c"]
    assert list(err.failed) == ["b"]
    assert isinstance(err.failed["b"], RuntimeError)
    assert "backend offline" in str(err.failed["b"])
    # The error message itself enumerates the split for operator logs.
    assert "a" in str(err) and "c" in str(err) and "b" in str(err)

    # The accepted sources DID receive the write — partial commit
    # confirmed.
    assert [m.content for m in a.received] == ["a", "b"]
    assert [m.content for m in c.received] == ["a", "b"]
    # The failed source did NOT receive (its write raised first).
    assert b.received == []


def test_composite_write_runs_every_backend_even_when_one_fails():
    """A slow backend isn't blamed for an early one's crash — both
    sides of the split get to attempt the write before the error
    surfaces. The write must not short-circuit on first failure."""
    fast_fail = _MockSource("fast_fail", raise_on_write=RuntimeError("quick"))
    slow = _MockSource("slow")

    # Wrap the slow source's write so we can assert it ran.
    original_write = slow.write
    ran = {"slow": False}

    async def slow_write(items, *, ctx):
        ran["slow"] = True
        await original_write(items, ctx=ctx)

    slow.write = slow_write  # type: ignore[method-assign]

    cm = CompositeMemory(sources=[fast_fail, slow])
    with pytest.raises(CompositeWriteError):
        _run(cm.write(_items(), ctx=None))

    assert ran["slow"] is True
    assert [m.content for m in slow.received] == ["a", "b"]


def test_composite_write_all_failures_surfaces_all_in_split():
    """Defensive — every backend failing produces a fully-failed split
    with an empty ``accepted`` list and named errors per source."""
    a = _MockSource("a", raise_on_write=ValueError("bad shape"))
    b = _MockSource("b", raise_on_write=RuntimeError("redis down"))
    cm = CompositeMemory(sources=[a, b])

    with pytest.raises(CompositeWriteError) as ei:
        _run(cm.write(_items(), ctx=None))

    err = ei.value
    assert err.accepted == []
    assert sorted(err.failed) == ["a", "b"]
    assert isinstance(err.failed["a"], ValueError)
    assert isinstance(err.failed["b"], RuntimeError)


def test_composite_write_empty_items_is_a_noop():
    """Regression check — no items → no backend touched, no error."""
    a = _MockSource("a", raise_on_write=RuntimeError("should not fire"))
    cm = CompositeMemory(sources=[a])
    _run(cm.write([], ctx=None))
    assert a.received == []
