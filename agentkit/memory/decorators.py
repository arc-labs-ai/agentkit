"""Memory decorators — composable wrappers over any ``MemorySource``.

Each decorator IS a ``MemorySource`` (structurally), so they nest
arbitrarily. The discipline is single-responsibility: ``ScopedMemory``
enforces tenant boundaries, ``CompactedMemory`` shrinks results,
``CachedMemory`` short-circuits identical queries, ``ReadOnlyMemory``
refuses writes. Stack them in the order that matches your concerns.

``accepts_writes`` is the one attribute these decorators agree on beyond
the Protocol. It is a structural marker, read with
``getattr(source, "accepts_writes", True)`` so every existing backend
keeps the permissive default, and it exists because
``CompositeMemory.write`` otherwise has no way to tell "this source
committed" from "this source dropped the write and returned normally" —
it would report both as ``accepted``. Each pass-through decorator
mirrors the marker from the source it wraps, so a
``ScopedMemory(ReadOnlyMemory(kb))`` inside a fan-out is still bucketed
as refused rather than accepted. It is deliberately NOT on the
``MemorySource`` Protocol: ``MemorySource`` is ``runtime_checkable``, so
adding a non-method member would make ``isinstance(x, MemorySource)``
False for every backend that predates it.
"""

from __future__ import annotations

import contextlib
import json
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, get_args

from agentkit.kernel.errors import AgentkitError
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message
from agentkit.memory.base import MemoryItem, MemorySource


class MemoryWriteRefused(AgentkitError):
    """A ``ReadOnlyMemory`` turned away a write.

    Subclasses ``AgentkitError`` rather than ``ValueError`` or a bare
    ``RuntimeError`` because ``kernel/errors.py`` makes one promise about
    this surface — "every framework-raised exception is a subclass of
    ``AgentkitError`` so a defensive ``except AgentkitError:`` catches
    the whole surface" — and a run boundary that catches the taxonomy
    should not need a second clause for this one.

    Named rather than anonymous for the reason ``CompositeWriteError``
    is: it lands inside ``CompositeWriteError.failed`` when a read-only
    member sits in a fan-out under the default policy, and an operator
    reading that split needs to tell "the registry is deliberately
    read-only" apart from "the registry is down". ``PermissionError``
    was the other candidate and was rejected — ``ScopedMemory`` already
    raises it for a tenant-boundary violation, and collapsing "wrong
    tenant" into "right tenant, immutable source" would make the two
    indistinguishable at the catch site.

    Attributes:
        source: the wrapped source's ``name`` — the label the write was
            aimed at, which is what identifies it inside a fan-out.
        n: how many items the refused call carried. ``0`` is a real
            value; see ``ReadOnlyMemory.write`` for why an empty write
            still refuses.
    """

    def __init__(self, *, source: str, n: int) -> None:
        self.source = source
        self.n = n
        super().__init__(
            f"ReadOnlyMemory: refused a write of {n} item(s) to read-only source {source!r}. "
            f"Pass on_write='ignore' if this source is one read-only member of a composite."
        )


def _default_enforce(scope: Any) -> bool:
    """Default policy: both tenant axes (``org_id`` AND ``domain_id``)
    must be populated. A query against an unscoped run is a programmer
    error — refuse loudly rather than silently leak across tenants.

    Explicit ``is not None`` checks — a valid tenant ID may be ``0``
    (e.g. the ``root`` org or the ``default`` domain), and ``bool(0)``
    would misclassify it as unscoped.
    """
    if scope is None:
        return False
    return (
        getattr(scope, "org_id", None) is not None and getattr(scope, "domain_id", None) is not None
    )


@dataclass(slots=True)
class ScopedMemory:
    """Fail-loud multi-tenant guard around any ``MemorySource``.

    Every ``query`` and every ``write`` runs ``enforce(ctx)`` first.
    If ``enforce`` is ``None``, the default check fires — both
    ``ctx.scope.org_id`` and ``ctx.scope.domain_id`` must be set.

    A custom ``enforce`` callable lets callers pin a stricter policy
    (e.g. "must match this specific tenant", "must be inside an
    approved checkpoint"). Returning a falsy value OR raising any
    exception inside ``enforce`` results in a ``PermissionError`` — the
    inner source is NOT touched.

    The decorator's ``name`` mirrors the wrapped source so downstream
    consumers attributing ``MemoryItem.source`` see the same label
    they'd see without the wrapper — the guard is invisible on the
    happy path.
    """

    inner: MemorySource
    enforce: Callable[[Ctx], Any] | None = None
    name: str = field(default="")

    def __post_init__(self) -> None:
        # Mirror the inner source's diagnostic label so attribution
        # stays stable when callers wrap/unwrap the guard.
        if not self.name:
            self.name = getattr(self.inner, "name", "scoped")

    @property
    def accepts_writes(self) -> bool:
        """Mirror the wrapped source. A tenant guard around a read-only
        source is still read-only, and a composite that read ``True``
        here would file the member under ``accepted``."""
        return bool(getattr(self.inner, "accepts_writes", True))

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        self._check(ctx)
        return await self.inner.query(query, k=k, ctx=ctx, where=where)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        self._check(ctx)
        # Materialize once so we can (a) forward a list to the inner
        # source and (b) report ``n`` on the observation without
        # re-iterating an exhausted generator.
        materialized = list(items)
        await self.inner.write(materialized, ctx=ctx)
        # Best-effort observability — a memory write is the kind of
        # side-effect operators want to see on the run's timeline
        # (e.g. "the Researcher just cached 12 items to the vector
        # store"). ``RunContext.emit`` is already defensive; the guard
        # here is only for ctx stubs that don't expose ``emit``.
        #
        # ``accepts_writes`` gates it because the inner source may be a
        # ``ReadOnlyMemory(..., on_write="ignore")``, whose write returns
        # normally having stored nothing. Emitting ``memory.written``
        # there would put a write on the operator's timeline that never
        # happened — and the read-only source emits its own
        # ``memory.write_refused`` for the same event, so the timeline
        # stays complete either way.
        emit = getattr(ctx, "emit", None)
        if emit is not None and self.accepts_writes:
            with contextlib.suppress(Exception):
                await emit(
                    "memory.written",
                    payload={"n": len(materialized), "source": self.name},
                )

    def _check(self, ctx: Ctx) -> None:
        # Custom ``enforce`` receives the whole ``ctx`` (its typed contract);
        # the default policy only reads the scope, so we hand it the scope
        # directly — that lets tests exercise ``_default_enforce(scope)`` in
        # isolation and keeps the default's responsibility crisp.
        try:
            if self.enforce is not None:
                ok = self.enforce(ctx)
            else:
                ok = _default_enforce(getattr(ctx, "scope", None))
        except Exception as exc:  # enforce raised → translate to PermissionError
            raise PermissionError(
                f"ScopedMemory: enforce check raised on {self.inner!r}: {exc}"
            ) from exc
        if not ok:
            raise PermissionError(
                f"ScopedMemory: enforce check failed on {self.inner!r} "
                f"(scope={getattr(ctx, 'scope', None)!r})"
            )


@dataclass(slots=True)
class CompactedMemory:
    """Decorator: shrink each ``MemoryItem.content`` via a ``Compactor``.

    Used when raw chunks would blow the prefix budget. Adapts the
    framework's ``Compactor`` capability (which operates on
    ``list[Message]``) to operate on individual items: each item's
    content is wrapped as a single message, compacted, and the result
    becomes the new content.

    ``max_items`` caps the post-compaction list — useful when the
    inner source returns more items than the cognition wants to fold
    into the prompt.
    """

    inner: MemorySource
    compactor: Any  # Compactor capability
    max_items: int | None = None
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.inner, "name", "compacted")

    @property
    def accepts_writes(self) -> bool:
        """Compaction is a read-side concern; ``write`` is a straight
        pass-through, so the marker is the wrapped source's."""
        return bool(getattr(self.inner, "accepts_writes", True))

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        items = await self.inner.query(query, k=k, ctx=ctx, where=where)
        compacted: list[MemoryItem] = []
        for item in items:
            messages = [Message("user", item.content)]
            shrunk = await self.compactor.compact(messages, ctx)
            text = "\n".join(m.content for m in shrunk if m.content)
            compacted.append(
                MemoryItem(
                    content=text or item.content,
                    source=item.source,
                    # Carried through even though ``content`` changed: ``id``
                    # names the RECORD, and a summary of a chunk is still that
                    # chunk. Dropping it here would be worst exactly where it
                    # matters — compaction is what makes two copies of one fact
                    # stop matching on text, so a compacted source inside a
                    # ``CompositeMemory`` would lose its only remaining way to
                    # be recognised as a duplicate.
                    id=item.id,
                    score=item.score,
                    metadata=item.metadata,
                )
            )
        if self.max_items is not None:
            return compacted[: self.max_items]
        return compacted

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        await self.inner.write(items, ctx=ctx)


@dataclass(slots=True)
class CachedMemory:
    """Decorator: cache query → results with a TTL.

    The cache key is ``(query, k, frozenset(where or {}))``. Hits
    served without touching the inner source. Writes invalidate the
    cache (best-effort — a write may surface stale results to an
    in-flight query, which is the typical "read-after-write
    eventual consistency" trade-off).

    ``ttl_seconds`` is wall-clock monotonic — the cache never serves
    an entry older than this. ``max_entries`` caps the cache size;
    eviction is LRU-ish (oldest insertion drops when full).

    Not thread-safe; agents are single-flow per loop.
    """

    inner: MemorySource
    ttl_seconds: float = 60.0
    max_entries: int = 256
    name: str = field(default="")
    strict_scope: bool = False
    _cache: dict[tuple[Any, ...], tuple[float, list[MemoryItem]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.inner, "name", "cached")

    @property
    def accepts_writes(self) -> bool:
        """The cache never stores anything durably itself — it forwards
        writes — so whether this stack accepts writes is the inner
        source's answer."""
        return bool(getattr(self.inner, "accepts_writes", True))

    @staticmethod
    def _key(query: str, k: int, where: dict[str, Any] | None, scope_key: str) -> tuple[Any, ...]:
        # Two invariants on the cache key:
        #
        # 1. Include ``scope_key`` — the tenant axis. Without it,
        #    stacking ``ScopedMemory(CachedMemory(Inner))`` would serve
        #    tenant A's items to tenant B on a cache hit (their
        #    ``enforce`` pass runs, but the underlying items come from
        #    A's cached fetch).
        # 2. Serialize ``where`` via canonical JSON, not
        #    ``frozenset(...items())``. A ``where={"tags":
        #    ["security","finance"]}`` breaks the set fallback because
        #    lists aren't hashable. JSON dump handles nested collections
        #    and stays deterministic across processes.
        where_key = json.dumps(where or {}, sort_keys=True, default=str)
        return (scope_key, query, k, where_key)

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        scope_key = getattr(getattr(ctx, "scope", None), "key", lambda: "")()
        if not scope_key:
            # No scope on ctx (or an empty ``key()``) → the cache would
            # bucket every tenant under the same empty key, silently
            # crossing tenants on hits. In strict mode this is a hard
            # error; otherwise warn once so the drift is at least visible.
            msg = (
                "CachedMemory requires a scoped RunContext — "
                "wrap with ScopedMemory or provide ctx.scope"
            )
            if self.strict_scope:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
        key = self._key(query, k, where, scope_key)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None:
            ts, items = cached
            if now - ts < self.ttl_seconds:
                # Return a defensive copy. Without it, a caller that
                # extends / sorts / dedupes the returned list would
                # corrupt every subsequent hit against the same key for
                # the TTL window.
                return list(items)
            # expired — drop and miss-through
            del self._cache[key]
        items = await self.inner.query(query, k=k, ctx=ctx, where=where)
        if len(self._cache) >= self.max_entries:
            # Drop the oldest entry (insertion-order dicts give us FIFO).
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        # Store a defensive copy so post-fetch mutation by an outer
        # decorator (rerank/dedup) doesn't leak into the cache either.
        self._cache[key] = (now, list(items))
        return items

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        self._cache.clear()
        await self.inner.write(items, ctx=ctx)


OnWritePolicy = Literal["refuse", "ignore"]
"""What a ``ReadOnlyMemory`` does with a write. Closed ``Literal`` so a
typo is a type error at the call site, not a runtime surprise; the tuple
below re-derives the members with ``get_args`` so the runtime check and
the type can never drift apart."""

_ON_WRITE_POLICIES: tuple[str, ...] = get_args(OnWritePolicy)


@dataclass(slots=True)
class ReadOnlyMemory:
    """Decorator: a source that is read-only *by policy* says so.

    Every application has at least one source that must never be written
    — a curated knowledge base, an operator-maintained registry, a corpus
    of recorded facts the agent may consult and must not extend. Before
    this decorator the only protection was that nothing happened to call
    ``write``, which is a property of the code as it currently stands
    rather than a rule: the first cognition taught to persist what it
    learned would have written into the registry and nothing would have
    complained. Wrapping the source turns the convention into an
    enforced boundary that survives the next refactor.

    Single-responsibility like its neighbours: it constrains ONE verb.
    ``query`` is a byte-for-byte pass-through — same ``k``, same
    ``where``, same items back with their ``source`` stamp intact, which
    matters because that stamp is the identity a composite's merge and
    dedupe key on. A read-only member participates in recall exactly like
    any other source; that is the whole point of having it.

    Two policies, because a fan-out forces the distinction:

    * ``on_write="refuse"`` (default) — raise ``MemoryWriteRefused``. The
      right answer for a source nothing has any business writing to; the
      exception names the source so a stack trace points at the policy
      rather than at the plumbing.
    * ``on_write="ignore"`` — return normally, having written nothing.
      ``CompositeMemory.write`` broadcasts to every source, and one
      read-only member must not make the whole composite unwritable.

    ``ignore`` is the dangerous one, because "silently succeeds" is the
    failure mode this package keeps writing tests against. It is
    accounted for three ways rather than none: ``refused_writes`` counts
    the turned-away calls on the instance, a ``memory.write_refused``
    observation lands on the run's timeline with the source and the item
    count, and ``accepts_writes = False`` lets ``CompositeMemory`` file
    the member under a ``refused`` bucket instead of claiming in
    ``CompositeWriteError.accepted`` that it committed.

    Nests either way round. ``ReadOnlyMemory(ScopedMemory(x))`` refuses
    before the tenant guard runs (cheaper, and no ``PermissionError``
    masking the real reason); ``ScopedMemory(ReadOnlyMemory(x))`` checks
    the tenant first. Double wrapping is harmless — the outer refuses and
    the inner never runs.

    ``name`` mirrors the wrapped source, like every other decorator here,
    so the guard is invisible in attribution.
    """

    inner: MemorySource
    on_write: OnWritePolicy = "refuse"
    name: str = field(default="")
    refused_writes: int = field(default=0, init=False)

    # Structural marker, read by ``CompositeMemory.write``. ``ClassVar``
    # rather than a field: it is a fact about the type, not a knob, and a
    # field would land in the generated ``__init__`` where a caller could
    # set ``accepts_writes=True`` on a read-only source.
    accepts_writes: ClassVar[bool] = False

    def __post_init__(self) -> None:
        # Validate at CONSTRUCTION, not at first write. A typo'd policy
        # that only surfaced the first time something tried to write
        # would lie dormant for exactly as long as the "nothing happens
        # to call write" bug this class exists to replace — and it would
        # surface as a refusal in production rather than a ValueError at
        # wiring time.
        if self.on_write not in _ON_WRITE_POLICIES:
            raise ValueError(
                f"ReadOnlyMemory: on_write must be one of {_ON_WRITE_POLICIES!r}, "
                f"got {self.on_write!r}"
            )
        if not self.name:
            self.name = getattr(self.inner, "name", "read-only")

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Untouched. No filtering, no re-stamping, no defensive copy —
        this decorator has an opinion about ``write`` and none at all
        about recall."""
        return await self.inner.query(query, k=k, ctx=ctx, where=where)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        """Refuse or drop, per ``on_write``. ``self.inner.write`` is never
        awaited on either path — that is what makes "read-only" a fact
        about the wrapper rather than a hope about the backend, and it is
        why a wrapped source that raises on write is unreachable here.

        ``items`` is materialised before the branch so the refusal can
        report ``n``. It is the only reason to touch the iterable at all,
        and it costs one list build on a path that is about to raise or
        drop.

        An EMPTY write still refuses. The policy is about the attempt,
        not about bytes moved: a caller that reaches ``write`` on a
        read-only source is running code that will carry items the moment
        its input is non-empty, and letting ``write([])`` through means
        the violation is found in production instead of on the first test
        run. Nothing in the package pays for this — both
        ``CompositeMemory.write`` and ``SequentialMemory.write``
        short-circuit on an empty list before reaching any source.
        """
        materialised = list(items)
        if self.on_write == "refuse":
            raise MemoryWriteRefused(source=self.name, n=len(materialised))
        # ``ignore`` — count the drop on the instance so a caller (or a
        # test) can prove the source turned writes away rather than
        # inferring it from the absence of an error.
        self.refused_writes += 1
        # ...and put it on the run's timeline. Best-effort, mirroring
        # ``ScopedMemory``'s ``memory.written``: the guard is for ctx
        # stubs without ``emit``, and ``RunContext.emit`` already swallows
        # observer faults. Only the ``ignore`` path emits — a refusal
        # already surfaces as an exception (and as a ``failed`` entry in
        # ``CompositeWriteError``), so emitting there would double-report
        # one event.
        emit = getattr(ctx, "emit", None)
        if emit is not None:
            with contextlib.suppress(Exception):
                await emit(
                    "memory.write_refused",
                    payload={
                        "n": len(materialised),
                        "source": self.name,
                        "policy": "ignore",
                    },
                )


__all__ = [
    "CachedMemory",
    "CompactedMemory",
    "MemoryWriteRefused",
    "OnWritePolicy",
    "ReadOnlyMemory",
    "ScopedMemory",
]
