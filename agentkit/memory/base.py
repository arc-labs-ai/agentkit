"""MemorySource Protocol — the single seam for "what can this agent reach for?"

One Protocol, many backends (vector, key-value, files, journal, tool-wrapped,
composite). Adding a new memory backend is implementing this Protocol; no
Agent subclassing, no parallel class hierarchy.

The shape is deliberately narrow:

- ``query(query, k, ctx)`` — ask for items relevant to ``query``.
- ``write(items, ctx)`` — index items for later recall (no-op when the
  backend is read-only, e.g. a wrapped search tool).

Every backend returns the same ``MemoryItem`` shape regardless of native
representation — the cognition that consumes results never sees backend
specifics. This is the Adapter discipline applied at the Protocol boundary.

The Protocol coexists with ``Tool`` (LLM-decided side effects) — they
overlap (a web-search tool is also a knowledge probe) but answer different
questions: tools are what the *model* decides to call mid-loop; memory is
what the *cognition* fetches around an LLM call to ground reasoning. The
``ToolMemory`` adapter bridges the two when a single backend serves both.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentkit.kernel._frozen import deep_freeze
from agentkit.kernel.protocols import Ctx


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """The uniform result shape every ``MemorySource`` returns.

    ``content`` is what the agent reads (already-formatted text).
    ``source`` names the backend that produced it (the ``MemorySource.name``)
    so consumers can attribute results and cognitions can format citations.
    ``id`` is the backend's own identifier for the RECORD — a vector store's
    chunk id, a journal row key. Stable within a source; not assumed
    comparable across sources unless the caller knows they share a keyspace
    (which is the usual case here, because the journal is normally what the
    vector store was built from). It exists so ``CompositeMemory`` can tell
    that two sources returned the SAME fact rather than two facts: without it
    the merge was pure concatenation, a reranker scored both copies, and the
    top-k the model saw was one fact occupying two slots. ``None`` is the
    honest answer for a backend that has no stable key (a tool-wrapped search,
    a scratchpad), and dedupe falls back to a content digest for those.
    ``score`` is the backend's relevance signal (cosine similarity for
    vector search, recency for journal, etc.); ``None`` when the backend
    doesn't rank.
    ``metadata`` carries backend-specific extras (chunk id, file path,
    timestamp) that the cognition or the rendering layer may consult. It is
    FROZEN at construction (see ``__post_init__``) — still a ``dict`` for
    every consumer, but a backend that wants to annotate an item builds a new
    one: ``dataclasses.replace(item, metadata={**item.metadata, "path": p})``.

    ``id`` is KEYWORD-ONLY, and that is load-bearing rather than stylistic.
    This type is constructed positionally in tens of places
    (``MemoryItem("c", "vector", 0.91, meta)``), so declaring ``id`` third —
    where it reads most naturally, next to ``source`` — would have rebound
    ``0.91`` to ``id`` and ``meta`` to ``score`` at every one of them: no
    error, wrong data, silently. Appending it after ``metadata`` would have
    fixed today and left the same trap for the next field; ``kw_only`` closes
    it permanently and lets the declaration keep the order a reader wants.
    """

    content: str
    source: str
    id: str | None = field(default=None, kw_only=True)
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze ``metadata`` — deeply — so an item cannot be re-annotated
        after the backend produced it.

        ``frozen=True`` stopped at the field reference: ``item.metadata = {}``
        raised while ``item.metadata["path"] = ...`` went through. The habit
        that made this worth closing is the one the fan-out invites — a
        ``CompositeMemory`` merges items from several sources and a reranker
        stamps its own scores onto them, and because backends pass ``metadata``
        THROUGH (``decorators.py`` and ``tool.py`` both hand the same object to
        a new ``MemoryItem``), one stamp could land on an item another source
        still holds. ``deep_freeze`` copies, so a passed-through payload is
        un-aliased at each hop, and ``deep_freeze`` is idempotent, so the hop
        that re-freezes an already-frozen payload costs one isinstance check
        rather than a second walk.

        Deep rather than shallow because ``metadata`` is documented as
        "backend-specific extras" and holds whatever the store did — nested
        JSON, chunk-offset lists, provider blobs.

        Cost is O(payload) and paid once per item. A recall metadata is small
        (a chunk id, a path, a timestamp): 0.49 µs empty, ~7.6 µs at 344 B of
        JSON, against a backend call that is a network round trip. A k=20
        recall therefore pays tens of microseconds against tens of
        milliseconds.

        This is a BREAKING change for a backend that annotated after
        construction. The migration is one line::

            item = dataclasses.replace(item, metadata={**item.metadata, "path": p})

        ``dataclasses.replace`` re-runs ``__post_init__``, so the rebuilt item
        is frozen too.
        """
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))

    def __hash__(self) -> int:
        """Hash the RECALL identity — ``(content, source, score)`` — never
        ``metadata``.

        ``metadata`` is ``field(default_factory=dict)``, so the
        dataclass-generated all-fields hash reached a dict on every item any
        backend has ever returned. Measured before this fix::

            hash(MemoryItem(content="c", source="s"))                TypeError: unhashable type: 'dict'
            hash(MemoryItem(content="c", source="s", metadata={"a": 1}))
            TypeError: unhashable type: 'dict'

        Unhashable by type rather than by value, which is why it never showed
        up as an intermittent failure — nothing could hash a recall result at
        all. The obvious caller is the one this shape invites: a composite
        memory pulls the same document out of two sources and wants
        ``set(items)`` (or a ``seen`` set inside a reranker) to collapse the
        duplicate before the top-k cut. That was a ``TypeError``, so callers
        wrote list scans instead.

        ``metadata`` is documented as "backend-specific extras (chunk id, file
        path, timestamp)" and is exactly the kind of thing that cannot be
        hashed: a vector backend passes through whatever the store held —
        nested JSON, lists of chunk offsets, provider blobs. It is also the
        field most likely to DIFFER between two records of the same content
        (two chunk ids for the same passage), so keeping it out of the hash is
        what makes the dedup above collapse anything at all.

        ``content`` / ``source`` / ``score`` are the item as a consumer reads
        it: the text, who produced it, how relevant they said it was. All
        three are hashable by construction (``str`` / ``str`` /
        ``float | None``), so the hash is TOTAL — it cannot be broken by a
        backend's payload.

        Cost is O(1) in the payload — measured at 0.22 µs whether ``metadata``
        holds one key or 100_000, because it is never read. ``content`` is a
        ``str`` of unbounded length but CPython caches a string's hash on the
        object, so a long passage is walked at most once.

        ``id`` is deliberately absent too, for a different reason than
        ``metadata``: it is hashable, it is just not what this hash is FOR.
        Collapsing two records that share an ``id`` is not a set operation —
        the survivor has to keep the higher ``score`` and record which sources
        agreed, and a ``set`` can express neither. That merge belongs to
        ``CompositeMemory`` (see its ``dedupe`` mode), and putting ``id`` in
        the hash would only make two copies of one fact land in different
        buckets, which is the opposite of what a caller reaching for
        ``set(items)`` wants.

        Sound rather than a workaround: ``__eq__`` still compares ``metadata``
        and ``id``, and the hash invariant only requires EQUAL objects to hash equally,
        never that unequal ones differ. Two items with identical text from the
        same source but different chunk ids collide into one bucket and
        ``__eq__`` separates them there — so a ``set`` still keeps both, which
        is the honest answer for records that genuinely differ.
        """
        return hash((self.content, self.source, self.score))


@runtime_checkable
class MemorySource(Protocol):
    """A queryable source of information the agent can reach for.

    ``name`` is the stable diagnostic label — stamped on returned
    ``MemoryItem.source``, also used for trace attribution. Implementations
    SHOULD set a sensible class-level default.

    Implementations MUST honour ``ctx.scope`` if the underlying backend
    crosses tenants. Multi-tenant enforcement is the implementation's job
    OR a ``ScopedMemory`` decorator's job; the Protocol doesn't enforce it
    itself.
    """

    name: str

    async def query(
        self, query: str, *, k: int, ctx: Ctx, where: dict[str, Any] | None = None
    ) -> list[MemoryItem]: ...

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """Optional second pass over a set of merged ``MemoryItem``s.

    A composite memory may pull from many sources, each returning items
    scored on its own scale (vector cosine vs lexical match vs recency).
    A Reranker scores them on a common axis so the top-k cut is sensible.

    The default reranker is ``score_sort_rerank`` — sort by ``score`` desc
    with ``None`` last. Real applications wire a cross-encoder or an LLM
    judge here.
    """

    async def rerank(self, query: str, items: list[MemoryItem], *, k: int) -> list[MemoryItem]: ...


async def score_sort_rerank(query: str, items: list[MemoryItem], *, k: int) -> list[MemoryItem]:
    """Default: stable sort by score descending. ``None`` scores sink to
    the bottom but aren't dropped — a tool-wrapped source that doesn't
    rank still gets to surface in the top-k."""
    del query
    scored = [
        (i.score if i.score is not None else float("-inf"), idx, i) for idx, i in enumerate(items)
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _s, _idx, i in scored[:k]]


__all__ = [
    "MemoryItem",
    "MemorySource",
    "Reranker",
    "score_sort_rerank",
]
