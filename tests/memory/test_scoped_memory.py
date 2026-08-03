"""``ScopedMemory`` — the fail-loud multi-tenant guard.

These tests pin the contract:

1. Default policy refuses when ``ctx.scope`` is empty (no ``org_id`` and/or
   no ``domain_id``) — raises ``PermissionError``.
2. A custom ``enforce`` callable is called with ``ctx`` and a falsy return
   triggers ``PermissionError``; a raise inside ``enforce`` is also
   translated into ``PermissionError``.
3. On success the wrapper transparently delegates to the inner source's
   ``query`` and ``write`` — same args, same return.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Scope
from agentkit.memory import MemoryItem, ScopedMemory
from agentkit.memory.decorators import _default_enforce
from agentkit.testing import make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def test_default_enforce_accepts_org_id_zero():
    """Regression: ``org_id=0`` is a legitimate tenant identifier (e.g.
    the ``root`` org in tests / fixtures). The previous ``bool(...)``
    check misclassified it as unscoped; the fix uses an explicit
    ``is not None`` predicate."""
    scope = Scope(org_id=0, domain_id=1)
    assert _default_enforce(scope) is True

    # And still refuses genuinely unpopulated scopes.
    assert _default_enforce(Scope()) is False
    assert _default_enforce(None) is False


# ── recording fake MemorySource ────────────────────────────────────────────


@dataclass
class _RecordingMemory:
    """Records every query / write so the test can assert the inner source
    received what the wrapper forwarded — and was NOT called when the
    guard refused."""

    return_items: list[MemoryItem] = field(default_factory=list)
    queries: list[tuple[str, int, Ctx, dict | None]] = field(default_factory=list)
    writes: list[tuple[list[MemoryItem], Ctx]] = field(default_factory=list)
    name: str = "inner"

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        self.queries.append((query, k, ctx, where))
        return list(self.return_items)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        self.writes.append((list(items), ctx))


# ── default enforce: both scope axes required ─────────────────────────────


def test_default_enforce_raises_when_scope_is_empty():
    """A run with the zero-tenant default ``Scope()`` (both fields ``None``)
    cannot reach a ScopedMemory. The whole point is to fail loud when a
    caller forgot to set the tenant."""
    inner = _RecordingMemory()
    guarded = ScopedMemory(inner=inner)
    ctx = make_test_ctx()  # Scope() — both axes None

    with pytest.raises(PermissionError):
        _run(guarded.query("q", k=3, ctx=ctx))
    with pytest.raises(PermissionError):
        _run(guarded.write([MemoryItem(content="x", source="t")], ctx=ctx))

    # inner source was NOT touched
    assert inner.queries == []
    assert inner.writes == []


def test_default_enforce_raises_when_only_one_axis_set():
    """Both ``org_id`` AND ``domain_id`` are required by the default
    policy — half-scoped is still a leak risk."""
    inner = _RecordingMemory()
    guarded = ScopedMemory(inner=inner)

    with pytest.raises(PermissionError):
        _run(guarded.query("q", k=1, ctx=make_test_ctx(scope=Scope(org_id=1))))
    with pytest.raises(PermissionError):
        _run(guarded.query("q", k=1, ctx=make_test_ctx(scope=Scope(domain_id=2))))


def test_default_enforce_allows_when_both_axes_set():
    """Happy path: both ``org_id`` and ``domain_id`` populated ⇒ the
    inner source sees the full args."""
    inner = _RecordingMemory(return_items=[MemoryItem(content="hit", source="inner")])
    guarded = ScopedMemory(inner=inner)
    ctx = make_test_ctx(scope=Scope(1, 2))

    out = _run(guarded.query("q", k=3, ctx=ctx, where={"k": "v"}))
    assert [i.content for i in out] == ["hit"]
    assert inner.queries == [("q", 3, ctx, {"k": "v"})]


# ── custom enforce: caller's policy wins ──────────────────────────────────


def test_custom_enforce_is_called_with_ctx_and_falsy_return_blocks():
    """When ``enforce`` is set, it REPLACES the default check. A falsy
    return is treated as 'denied' and raises ``PermissionError``."""
    inner = _RecordingMemory()
    seen: list[Ctx] = []

    def enforce(ctx: Ctx) -> bool:
        seen.append(ctx)
        return False

    guarded = ScopedMemory(inner=inner, enforce=enforce)
    ctx = make_test_ctx(scope=Scope(1, 2))  # would pass default — but custom denies

    with pytest.raises(PermissionError):
        _run(guarded.query("q", k=1, ctx=ctx))

    assert seen == [ctx]
    assert inner.queries == []


def test_custom_enforce_truthy_return_delegates():
    """A truthy ``enforce`` return means 'allowed' — the wrapper
    forwards transparently to the inner source."""
    inner = _RecordingMemory(return_items=[MemoryItem(content="ok", source="inner")])

    def always_allow(_ctx: Ctx) -> bool:
        return True

    guarded = ScopedMemory(inner=inner, enforce=always_allow)
    ctx = make_test_ctx()  # no scope — but custom enforce overrides

    out = _run(guarded.query("q", k=1, ctx=ctx))
    assert [i.content for i in out] == ["ok"]


def test_custom_enforce_raising_is_translated_to_permission_error():
    """An exception inside ``enforce`` is translated to a
    ``PermissionError`` so callers have ONE exception type to catch at
    the boundary. The original exception is chained as ``__cause__``."""
    inner = _RecordingMemory()

    class _BoomError(RuntimeError):
        pass

    def enforce(_ctx: Ctx) -> Any:
        raise _BoomError("policy explosion")

    guarded = ScopedMemory(inner=inner, enforce=enforce)
    ctx = make_test_ctx(scope=Scope(1, 2))

    with pytest.raises(PermissionError) as excinfo:
        _run(guarded.query("q", k=1, ctx=ctx))
    assert isinstance(excinfo.value.__cause__, _BoomError)
    assert inner.queries == []


# ── write goes through the same gate ──────────────────────────────────────


def test_write_runs_through_the_enforce_gate_too():
    """Writes are AS sensitive as reads — a bypass on write would let a
    caller stamp another tenant's bucket. The gate fires on both
    paths."""
    inner = _RecordingMemory()
    guarded = ScopedMemory(inner=inner)

    with pytest.raises(PermissionError):
        _run(guarded.write([MemoryItem(content="x", source="t")], ctx=make_test_ctx()))
    assert inner.writes == []

    # ... and delegates when scoped properly
    ctx = make_test_ctx(scope=Scope(1, 2))
    items = [MemoryItem(content="ok", source="t")]
    _run(guarded.write(items, ctx=ctx))
    assert len(inner.writes) == 1
    assert [i.content for i in inner.writes[0][0]] == ["ok"]
    assert inner.writes[0][1] is ctx


# ── diagnostic name mirrors the inner source ──────────────────────────────


def test_name_mirrors_inner_source_by_default():
    """Downstream attribution shouldn't know the guard is there — the
    name defaults to the wrapped source's name."""
    inner = _RecordingMemory(name="vector")
    guarded = ScopedMemory(inner=inner)
    assert guarded.name == "vector"


def test_explicit_name_overrides_default():
    inner = _RecordingMemory(name="vector")
    guarded = ScopedMemory(inner=inner, name="vector-scoped")
    assert guarded.name == "vector-scoped"
