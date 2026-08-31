"""``ReadOnlyMemory`` — a source that must never be written says so.

The gap this closes: a curated knowledge base, an operator-maintained
registry, a corpus of recorded facts. Every one of them is read-only *by
policy* rather than by backend, and until this decorator the only
protection was that nothing happened to call ``write`` — a property of
the code as it currently stands, not a rule. The first cognition that
learned to cache its findings would have written into the registry and
nothing would have complained.

Two policies, because a composite forces the distinction:

- ``on_write="refuse"`` (default) — raise ``MemoryWriteRefused``. The
  right answer when the caller has no business writing here at all.
- ``on_write="ignore"`` — let the fan-out succeed. ``CompositeMemory.write``
  broadcasts to every source, and one read-only member must not make the
  whole composite unwritable.

The failure mode these tests exist to prevent is the second policy
degenerating into a silent no-op that reports success. ``ignore`` must
still be *accounted for*: a counter on the decorator, a
``memory.write_refused`` observation on the run timeline, and a third
``refused`` bucket on ``CompositeWriteError`` so ``accepted`` never
claims a source took a write it dropped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentkit.adapters.observer.sinks import CollectingObserver
from agentkit.kernel.errors import AgentkitError
from agentkit.kernel.types import Scope
from agentkit.memory import (
    CachedMemory,
    CompactedMemory,
    CompositeMemory,
    CompositeWriteError,
    MemoryItem,
    MemoryWriteRefused,
    ReadOnlyMemory,
    ScopedMemory,
    SequentialMemory,
    ToolMemory,
)
from agentkit.testing import make_test_ctx


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _ctx(observer: Any = None) -> Any:
    """A fully scoped ctx — ``ScopedMemory``'s default enforce needs both
    tenant axes, and ``CachedMemory`` warns without a scope key."""
    return make_test_ctx(scope=Scope(org_id=1, domain_id=2), observer=observer)


@dataclass
class _Spy:
    """Records every call. The load-bearing assertion in most of these
    tests is that ``writes`` stays EMPTY — "provably not called" is the
    whole point of a read-only guard, and a decorator that forwarded the
    write and swallowed the result would pass a weaker test."""

    name: str = "registry"
    return_items: list[MemoryItem] = field(default_factory=list)
    queries: list[tuple[str, int, dict[str, Any] | None]] = field(default_factory=list)
    writes: list[list[MemoryItem]] = field(default_factory=list)
    raise_on_write: BaseException | None = None

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Any,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del ctx
        self.queries.append((query, k, where))
        return list(self.return_items)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Any) -> None:
        del ctx
        if self.raise_on_write is not None:
            raise self.raise_on_write
        self.writes.append(list(items))


def _items() -> list[MemoryItem]:
    return [MemoryItem(content="a", source="registry"), MemoryItem(content="b", source="registry")]


# ── refusal ────────────────────────────────────────────────────────────────


def test_write_refuses_and_names_the_wrapped_source() -> None:
    """The default policy raises, the message names the source, and the
    inner backend is never touched. An error that didn't name the source
    would be useless inside a fan-out over six backends."""
    spy = _Spy(name="curated-kb")
    ro = ReadOnlyMemory(spy)

    with pytest.raises(MemoryWriteRefused) as ei:
        _run(ro.write(_items(), ctx=_ctx()))

    assert ei.value.source == "curated-kb"
    assert ei.value.n == 2
    assert "curated-kb" in str(ei.value)
    assert spy.writes == []


def test_refusal_is_in_the_agentkit_error_taxonomy() -> None:
    """A defensive ``except AgentkitError`` at a run boundary must catch
    it — that is the promise ``kernel/errors.py`` makes for every
    framework-raised exception."""
    assert issubclass(MemoryWriteRefused, AgentkitError)


def test_empty_write_still_refuses() -> None:
    """``write([])`` performed no write, and it still raises.

    The decision is deliberate: the policy is about the ATTEMPT, not
    about bytes moved. A caller that reaches ``write`` on a read-only
    source is running code that will carry items the moment its input is
    non-empty, and letting the empty case through means the violation is
    discovered in production rather than on the first test run — exactly
    the "true because nothing happens to call it" failure this decorator
    exists to convert into a rule. ``CompositeMemory.write`` short-circuits
    on an empty list before reaching any source, so no fan-out pays for
    this."""
    spy = _Spy()
    with pytest.raises(MemoryWriteRefused) as ei:
        _run(ReadOnlyMemory(spy).write([], ctx=_ctx()))
    assert ei.value.n == 0
    assert spy.writes == []


# ── ignore ─────────────────────────────────────────────────────────────────


def test_ignore_does_not_raise_and_never_calls_the_inner_source() -> None:
    spy = _Spy()
    ro = ReadOnlyMemory(spy, on_write="ignore")

    _run(ro.write(_items(), ctx=_ctx()))

    assert spy.writes == []
    assert ro.refused_writes == 1


def test_ignore_short_circuits_before_a_raising_inner_source() -> None:
    """Proves the "unreachable" claim rather than asserting it: the inner
    source raises on any write, and the ignore path completes anyway —
    the only way that holds is if ``inner.write`` was never awaited."""
    spy = _Spy(raise_on_write=RuntimeError("backend would have exploded"))
    _run(ReadOnlyMemory(spy, on_write="ignore").write(_items(), ctx=_ctx()))
    assert spy.writes == []


def test_ignore_emits_an_observation_so_the_drop_is_not_silent() -> None:
    """A silent no-op that reports success is the exact failure mode this
    repo keeps writing tests against. The ignore path lands on the run
    timeline with the source and the item count."""
    obs = CollectingObserver()
    ro = ReadOnlyMemory(_Spy(name="facts"), on_write="ignore")

    _run(ro.write(_items(), ctx=_ctx(observer=obs)))

    refused = [o for o in obs.items if o.kind == "memory.write_refused"]
    assert len(refused) == 1
    assert refused[0].payload == {"n": 2, "source": "facts", "policy": "ignore"}


def test_refuse_does_not_emit_because_the_raise_is_the_record() -> None:
    """Only the ignore path needs an observation. A refusal already
    surfaces as an exception (and as a ``failed`` entry inside a
    composite), so emitting there would double-report the same event."""
    obs = CollectingObserver()
    with pytest.raises(MemoryWriteRefused):
        _run(ReadOnlyMemory(_Spy()).write(_items(), ctx=_ctx(observer=obs)))
    assert [o for o in obs.items if o.kind == "memory.write_refused"] == []


def test_ignore_survives_a_ctx_without_emit() -> None:
    """The observation is best-effort. A bare ctx stub (the shape half
    this repo's memory tests use) must not turn a policy no-op into a
    crash."""
    spy = _Spy()
    ro = ReadOnlyMemory(spy, on_write="ignore")
    _run(ro.write(_items(), ctx=None))
    assert ro.refused_writes == 1


# ── construction ───────────────────────────────────────────────────────────


def test_invalid_policy_is_refused_at_construction() -> None:
    """Not at first write. A typo'd policy that only surfaces the first
    time something tries to write would sit dormant for exactly as long
    as the bug this decorator replaces."""
    with pytest.raises(ValueError, match="on_write"):
        ReadOnlyMemory(_Spy(), on_write="silent")  # type: ignore[arg-type]


def test_name_mirrors_the_wrapped_source() -> None:
    """``MemoryItem.source`` is stamped from the backend's ``name`` and
    composite dedupe keys on that identity, so the guard must be
    invisible in attribution — same label wrapped or not."""
    spy = _Spy(name="registry")
    assert ReadOnlyMemory(spy).name == "registry"
    assert ReadOnlyMemory(spy, on_write="ignore").name == "registry"
    # An explicit override still wins for callers who want the guard visible.
    assert ReadOnlyMemory(spy, name="pinned").name == "pinned"


def test_double_wrapping_is_idempotent_in_behaviour() -> None:
    """``ReadOnlyMemory(ReadOnlyMemory(x))`` — the outer refuses first, so
    the inner never runs, and the name survives both hops."""
    spy = _Spy(name="registry")
    ro = ReadOnlyMemory(ReadOnlyMemory(spy))
    assert ro.name == "registry"
    with pytest.raises(MemoryWriteRefused) as ei:
        _run(ro.write(_items(), ctx=_ctx()))
    assert ei.value.source == "registry"
    assert spy.writes == []


def test_accepts_writes_marker_is_false() -> None:
    """The structural marker ``CompositeMemory`` reads to bucket a source
    as refused rather than accepted."""
    assert ReadOnlyMemory(_Spy()).accepts_writes is False
    assert ReadOnlyMemory(_Spy(), on_write="ignore").accepts_writes is False


# ── query is untouched ─────────────────────────────────────────────────────


def test_query_passes_through_completely_untouched() -> None:
    """This decorator constrains ONE verb. ``k`` and ``where`` arrive at
    the backend byte-for-byte and the items come back unmodified —
    including ``source``, which composite dedupe keys on."""
    got = [MemoryItem(content="x", source="registry", score=0.5, metadata={"chunk": 3})]
    spy = _Spy(return_items=got)
    ro = ReadOnlyMemory(spy, on_write="ignore")

    out = _run(ro.query("who owns billing?", k=7, ctx=_ctx(), where={"tags": ["ops", "sec"]}))

    assert spy.queries == [("who owns billing?", 7, {"tags": ["ops", "sec"]})]
    assert out == got
    assert out[0].source == "registry"


def test_query_still_works_after_a_refused_write() -> None:
    """A refusal must not poison the read path."""
    spy = _Spy(return_items=[MemoryItem(content="x", source="registry")])
    ro = ReadOnlyMemory(spy)
    with pytest.raises(MemoryWriteRefused):
        _run(ro.write(_items(), ctx=_ctx()))
    assert len(_run(ro.query("q", k=1, ctx=_ctx()))) == 1


# ── nesting, both orders ───────────────────────────────────────────────────


def test_scoped_outside_read_only() -> None:
    """``ScopedMemory(ReadOnlyMemory(x))`` — the tenant guard passes, the
    read-only guard still refuses, and the refusal propagates through."""
    spy = _Spy()
    stack = ScopedMemory(ReadOnlyMemory(spy))
    assert stack.name == "registry"
    with pytest.raises(MemoryWriteRefused):
        _run(stack.write(_items(), ctx=_ctx()))
    assert spy.writes == []


def test_scoped_outside_read_only_ignore_does_not_report_a_write() -> None:
    """``ScopedMemory.write`` emits ``memory.written`` for the run
    timeline. Wrapping a read-only source, that emission would be a lie —
    the timeline would show a write to a source that dropped it. The
    guard reads the inner marker and stays quiet."""
    obs = CollectingObserver()
    stack = ScopedMemory(ReadOnlyMemory(_Spy(), on_write="ignore"))

    _run(stack.write(_items(), ctx=_ctx(observer=obs)))

    kinds = [o.kind for o in obs.items]
    assert "memory.written" not in kinds
    assert "memory.write_refused" in kinds


def test_read_only_outside_scoped() -> None:
    """``ReadOnlyMemory(ScopedMemory(x))`` — refused before the tenant
    guard even runs. Both orders are legal; this one is cheaper."""
    spy = _Spy()
    stack = ReadOnlyMemory(ScopedMemory(spy))
    with pytest.raises(MemoryWriteRefused):
        _run(stack.write(_items(), ctx=_ctx()))
    assert spy.writes == []


def test_read_only_outside_cached_leaves_the_cache_intact() -> None:
    """``CachedMemory.write`` clears its cache before delegating. Wrapped
    on the OUTSIDE, the read-only guard refuses first, so a rejected
    write cannot evict a warm cache."""
    spy = _Spy(return_items=[MemoryItem(content="x", source="registry")])
    stack = ReadOnlyMemory(CachedMemory(spy))
    ctx = _ctx()

    _run(stack.query("q", k=1, ctx=ctx))
    with pytest.raises(MemoryWriteRefused):
        _run(stack.write(_items(), ctx=ctx))
    _run(stack.query("q", k=1, ctx=ctx))

    # One backend hit — the second query was served from the still-warm cache.
    assert len(spy.queries) == 1


def test_cached_outside_read_only_passes_queries_through() -> None:
    """The reverse order. The cache sits above the guard, the guard's
    query passthrough keeps it correct, and the write still refuses."""
    spy = _Spy(return_items=[MemoryItem(content="x", source="registry")])
    stack = CachedMemory(ReadOnlyMemory(spy))
    ctx = _ctx()
    assert stack.name == "registry"

    assert len(_run(stack.query("q", k=1, ctx=ctx))) == 1
    assert len(_run(stack.query("q", k=1, ctx=ctx))) == 1
    assert len(spy.queries) == 1
    with pytest.raises(MemoryWriteRefused):
        _run(stack.write(_items(), ctx=ctx))
    assert spy.writes == []


def test_accepts_writes_propagates_through_the_other_decorators() -> None:
    """The marker has to survive a stack, otherwise a
    ``ScopedMemory(ReadOnlyMemory(x))`` member of a composite gets
    bucketed as ``accepted`` and the split lies again."""
    ro = ReadOnlyMemory(_Spy(), on_write="ignore")
    assert ScopedMemory(ro).accepts_writes is False
    assert CachedMemory(ro).accepts_writes is False
    # A plain writable source keeps the permissive default.
    assert ScopedMemory(_Spy()).accepts_writes is True


# ── inside a composite ─────────────────────────────────────────────────────


def test_composite_write_succeeds_with_one_read_only_member() -> None:
    """The headline case. One read-only member must not make the whole
    composite unwritable — the writable sources commit and no error is
    raised."""
    writable = _Spy(name="vectors")
    ro_inner = _Spy(name="registry")
    cm = CompositeMemory(sources=[writable, ReadOnlyMemory(ro_inner, on_write="ignore")])

    _run(cm.write(_items(), ctx=_ctx()))

    assert [m.content for m in writable.writes[0]] == ["a", "b"]
    assert ro_inner.writes == []


def test_composite_refused_member_is_accounted_for_not_counted_as_accepted() -> None:
    """The third bucket. When another source fails and the split
    surfaces, a source that DROPPED the write must not appear in
    ``accepted`` — that word means "committed", and an operator reading
    the postmortem would decide not to replay a write that never
    landed."""
    ok = _Spy(name="vectors")
    bad = _Spy(name="redis", raise_on_write=RuntimeError("redis down"))
    ro = ReadOnlyMemory(_Spy(name="registry"), on_write="ignore")
    cm = CompositeMemory(sources=[ok, bad, ro])

    with pytest.raises(CompositeWriteError) as ei:
        _run(cm.write(_items(), ctx=_ctx()))

    err = ei.value
    assert err.accepted == ["vectors"]
    assert err.refused == ["registry"]
    assert list(err.failed) == ["redis"]
    assert "registry" in str(err)


def test_composite_of_only_read_only_ignore_members_succeeds() -> None:
    """Degenerate but legal: every member read-only under ``ignore``.
    The write succeeds (nothing failed) and every member is counted."""
    a, b = _Spy(name="kb-a"), _Spy(name="kb-b")
    cm = CompositeMemory(
        sources=[ReadOnlyMemory(a, on_write="ignore"), ReadOnlyMemory(b, on_write="ignore")]
    )
    _run(cm.write(_items(), ctx=_ctx()))
    assert a.writes == [] and b.writes == []


def test_composite_of_only_read_only_refuse_members_fails_wholesale() -> None:
    """Under the default policy the same topology is a hard error with an
    empty ``accepted`` — the honest report for a composite that could not
    write anywhere."""
    cm = CompositeMemory(sources=[ReadOnlyMemory(_Spy(name="kb-a")), ReadOnlyMemory(_Spy("kb-b"))])
    with pytest.raises(CompositeWriteError) as ei:
        _run(cm.write(_items(), ctx=_ctx()))
    err = ei.value
    assert err.accepted == []
    assert sorted(err.failed) == ["kb-a", "kb-b"]
    assert all(isinstance(e, MemoryWriteRefused) for e in err.failed.values())


def test_composite_query_treats_a_read_only_member_like_any_other() -> None:
    """The read path is where a curated registry earns its keep. Its
    items must reach the merged pool with their ``source`` stamp intact —
    that stamp is the identity a composite's merge/dedupe keys on."""
    ro = ReadOnlyMemory(_Spy(name="registry", return_items=[MemoryItem("r", "registry", 0.9)]))
    other = _Spy(name="vectors", return_items=[MemoryItem("v", "vectors", 0.5)])
    cm = CompositeMemory(sources=[ro, other])

    out = _run(cm.query("q", k=5, ctx=_ctx()))

    assert [(i.content, i.source) for i in out] == [("r", "registry"), ("v", "vectors")]


# ── gaps found in review ───────────────────────────────────────────────────


def test_refused_writes_counts_every_drop_not_just_the_first() -> None:
    """The counter is the mechanism a caller uses to PROVE a source
    turned writes away, so it has to be a count and not a flag. Asserting
    only ``== 1`` after a single write left ``self.refused_writes = 1``
    (an assignment, not an increment) indistinguishable from the real
    thing — a decorator that reported "1" forever would still look
    correct."""
    ro = ReadOnlyMemory(_Spy(), on_write="ignore")
    for _ in range(3):
        _run(ro.write(_items(), ctx=_ctx()))
    assert ro.refused_writes == 3


def test_refused_writes_starts_at_zero_and_refuse_policy_never_increments() -> None:
    """Under ``refuse`` the raise is the record; the counter belongs to
    the ignore path alone."""
    ro = ReadOnlyMemory(_Spy())
    assert ro.refused_writes == 0
    with pytest.raises(MemoryWriteRefused):
        _run(ro.write(_items(), ctx=_ctx()))
    assert ro.refused_writes == 0


def test_compacted_memory_mirrors_the_marker() -> None:
    """``CompactedMemory`` was the one decorator whose mirroring nothing
    asserted. Its ``write`` is a straight pass-through, so a
    ``CompactedMemory(ReadOnlyMemory(...))`` in a fan-out must bucket as
    refused like the other two stacks."""

    async def _compactor(items: list[MemoryItem], *, ctx: Any) -> list[MemoryItem]:
        del ctx
        return items

    ro = ReadOnlyMemory(_Spy(), on_write="ignore")
    assert CompactedMemory(ro, compactor=_compactor).accepts_writes is False
    assert CompactedMemory(_Spy(), compactor=_compactor).accepts_writes is True


def test_tool_memory_declares_it_does_not_accept_writes() -> None:
    """``ToolMemory.write`` has always been a silent no-op. The marker is
    what stops ``CompositeMemory`` reporting that no-op as a commit, and
    nothing asserted it — flipping it back to ``True`` left the whole
    suite green."""
    assert ToolMemory.accepts_writes is False


def test_tool_memory_in_a_fan_out_is_refused_not_accepted() -> None:
    """The marker's reason for existing, exercised end to end on the
    adapter that motivated it."""
    ok = _Spy(name="vectors")
    bad = _Spy(name="redis", raise_on_write=RuntimeError("redis down"))
    tools = ToolMemory(tool=object(), name="tools")
    cm = CompositeMemory(sources=[ok, bad, tools])

    with pytest.raises(CompositeWriteError) as ei:
        _run(cm.write(_items(), ctx=_ctx()))

    assert ei.value.accepted == ["vectors"]
    assert ei.value.refused == ["tools"]


# ── the marker has to survive a composite boundary ─────────────────────────


def test_composite_of_only_read_only_members_does_not_claim_to_accept_writes() -> None:
    """A ``CompositeMemory`` is itself a ``MemorySource`` and composites
    nest. Without a marker of its own the propagation dies at the first
    composite boundary."""
    inner = CompositeMemory(sources=[ReadOnlyMemory(_Spy(name="kb"), on_write="ignore")])
    assert inner.accepts_writes is False
    # One writable member is enough — the composite really can commit.
    mixed = CompositeMemory(
        sources=[_Spy(name="vectors"), ReadOnlyMemory(_Spy(name="kb"), on_write="ignore")]
    )
    assert mixed.accepts_writes is True
    # An empty composite commits nowhere.
    assert CompositeMemory(sources=[]).accepts_writes is False


def test_nested_all_read_only_composite_is_not_reported_as_accepted() -> None:
    """The regression this closes, observed before the fix: the outer
    split reported ``accepted=['inner-ro']`` for a nested composite whose
    only member DROPPED the write. An operator reading that postmortem
    would skip a replay for a write that never landed — the precise
    failure the ``refused`` bucket exists to prevent, surviving one level
    of nesting."""
    inner = CompositeMemory(
        sources=[ReadOnlyMemory(_Spy(name="kb"), on_write="ignore")], name="inner-ro"
    )
    bad = _Spy(name="redis", raise_on_write=RuntimeError("redis down"))
    ok = _Spy(name="vectors")
    outer = CompositeMemory(sources=[inner, bad, ok])

    with pytest.raises(CompositeWriteError) as ei:
        _run(outer.write(_items(), ctx=_ctx()))

    assert ei.value.accepted == ["vectors"]
    assert ei.value.refused == ["inner-ro"]


def test_sequential_memory_mirrors_its_write_target() -> None:
    """``SequentialMemory.write`` touches ``sources[0]`` only, so that is
    the source its marker has to mirror — a read-only cache tier means
    the chain stores nothing on write."""
    ro = ReadOnlyMemory(_Spy(name="kb"), on_write="ignore")
    assert SequentialMemory(sources=[ro, _Spy(name="vectors")]).accepts_writes is False
    # Writable first tier: the chain does commit, regardless of tier two.
    assert SequentialMemory(sources=[_Spy(name="vectors"), ro]).accepts_writes is True
    assert SequentialMemory(sources=[]).accepts_writes is False


def test_sequential_memory_with_a_read_only_head_is_refused_in_a_fan_out() -> None:
    """End to end: the chain drops the write and the enclosing fan-out
    says so instead of claiming a commit."""
    chain = SequentialMemory(
        sources=[ReadOnlyMemory(_Spy(name="kb"), on_write="ignore")], name="cache-chain"
    )
    bad = _Spy(name="redis", raise_on_write=RuntimeError("redis down"))
    cm = CompositeMemory(sources=[chain, bad])

    with pytest.raises(CompositeWriteError) as ei:
        _run(cm.write(_items(), ctx=_ctx()))

    assert ei.value.accepted == []
    assert ei.value.refused == ["cache-chain"]
