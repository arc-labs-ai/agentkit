"""Composite memory shapes — combine many ``MemorySource``s as one.

Two compositions cover the practical cases:

- ``CompositeMemory`` — parallel fan-out + merge + optional rerank. The
  default when "ask everywhere and rank the union" is right. The merge is a
  UNION, not a concatenation: see ``dedupe``.
- ``SequentialMemory`` — try sources in order, stop at first hit (or
  short-circuit when k items are collected). The right shape when a
  cheap cache should answer before an expensive vector lookup.

Both implement the ``MemorySource`` Protocol, so they nest arbitrarily —
a ``CompositeMemory`` of two ``SequentialMemory`` chains is a valid
"two-arm rerank" topology.

Write semantics: ``CompositeMemory.write`` broadcasts to every source;
``SequentialMemory.write`` writes to the FIRST source (writes happen at
the cache tier in the typical setup). Backends that are read-only
(``ToolMemory``, or any source wrapped in ``ReadOnlyMemory`` with
``on_write="ignore"``) ignore writes via a no-op and declare it by
setting ``accepts_writes = False``, which is what keeps the fan-out's
report honest — see the ``refused`` bucket below.

Partial-write failures surface a :class:`CompositeWriteError` naming
which backends accepted the write and which failed. A naive
``gather(return_exceptions=False)`` would propagate the first raising
source verbatim — the caller would have no signal that another source's
write committed. ``CompositeMemory.write`` instead collects per-source
outcomes and raises only when at least one failed, with the full split
attached.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem, MemorySource, Reranker, score_sort_rerank

DedupeMode = Literal["id", "content"]
"""How ``CompositeMemory`` decides two items are the same fact.

``"id"`` merges items sharing a non-``None`` ``MemoryItem.id`` OR identical
content; ``"content"`` merges on content alone. ``None`` — the third value of
the parameter, not of this alias — is the old pure concatenation."""

#: Metadata keys stamped on a survivor. Only ever written when a collision
#: actually happened, so an item that never merged comes back identical to what
#: its backend returned — otherwise ``dedupe`` would show up in the metadata of
#: every result and break equality against the backend's own item.
DEDUPE_SOURCES_KEY = "dedupe_sources"
DEDUPE_COUNT_KEY = "dedupe_count"


def _content_digest(content: str) -> str | None:
    """The content half of the identity two items must share to be merged.

    The digest is taken of the STRIPPED text. Stripping is not cosmetic: a
    journal line ends in a newline and the chunk indexed from it does not,
    which is the normal case rather than a corner one, and raw equality would
    leave that pair un-merged. Interior whitespace is left alone — indentation
    inside a code chunk is content, and collapsing it would merge two genuinely
    different snippets.

    sha256 rather than ``hash()`` because the key has to be stable across
    processes: ``CachedMemory`` pickles items and PYTHONHASHSEED randomises
    ``str.__hash__`` per interpreter, so a hash-based key would group
    differently in the run that read a cache than in the run that wrote it.
    Cost is ~1 µs for a 1 KB passage, against the network round trip that
    fetched it.

    Returns ``None`` for text that is empty once stripped. Absence of content
    is not evidence of sameness, and because stripping maps ``""``, ``"   "``
    and ``"\\n\\t"`` onto ONE digest, treating it as a key chained every blank
    record in the pool into a single group and deleted all but one. Blank
    content is reachable without malice — a chunk that is whitespace once
    boilerplate is stripped, a tool hit with an empty snippet, a journal row
    with a blank body — and the content relation fires in BOTH modes, so this
    was the widest way to lose a fact here. Blank items still merge on a shared
    ``id``; they just no longer merge on having nothing to say.
    """
    stripped = content.strip()
    if not stripped:
        return None
    return hashlib.sha256(stripped.encode("utf-8")).hexdigest()


def _rank(score: float | None) -> float:
    """``None`` sorts BELOW every real score, matching ``score_sort_rerank``.

    ``score=None`` means "this backend does not rank", not "this is bad", and
    the default reranker already sinks it to the bottom. If an unranked copy
    could win a collision, the merged item would inherit ``None`` and get
    pushed out of the top-k — re-introducing, inside the fix, the exact
    dropped-fact failure the dedupe exists to prevent."""
    return score if score is not None else float("-inf")


def _merge_duplicates(items: list[MemoryItem], mode: DedupeMode) -> list[MemoryItem]:
    """Collapse items that are the same fact, keeping the best-scored copy.

    "Same fact" is a UNION of two relations, not a single key, and that shape
    is the whole design decision here.

    The obvious implementation is one key per item: use ``id`` when the item
    has one, fall back to a content digest when it does not. It is wrong in a
    way that only shows up later. Under it, giving a backend an ``id`` makes it
    stop matching backends that have none — so the moment ``VectorMemory``
    started reporting its chunk ids, the headline case REGRESSED: a vector hit
    (``id="r7"``) and the journal row it was built from (no id) fall under
    different keys and the duplicate survives. A field whose population makes
    the answer worse is a field nobody can safely populate.

    So in ``"id"`` mode two items merge if they share a non-``None`` ``id`` OR
    the same content digest, and the grouping is the transitive closure of
    both. ``"content"`` mode drops the id relation entirely — its whole purpose
    is stores whose ids are not comparable (two stores can both call their
    first row ``"1"``), where trusting a shared id merges two unrelated facts
    and DELETES one. That single difference is what separates the modes: a
    shared id with different text collapses under ``"id"`` and does not under
    ``"content"``.

    Union-find rather than a dict of keys because the two relations chain: an
    item with ``id="r7"`` links to another ``id="r7"`` that renders the fact
    differently, which links by text to a third that carries no id at all.
    Pools here are ``len(sources) * k`` — tens of items — so the near-linear
    cost is noise against the network calls that produced them.

    Order is FIRST-SEEN, not best-score: a group takes the slot its earliest
    member occupied. That keeps the merge a stable transformation of the
    concatenation, so ``score_sort_rerank`` (a stable sort with the index as
    tiebreak) still breaks ties on source declaration order and the same
    sources returning the same batches produce an identical list every run —
    which is what lets a prompt built from these items stay cache-hittable and
    an eval stay reproducible. Score ties go to the first-seen copy for the
    same reason: two unranked backends agreeing is the common tie, and
    "whichever arrived last" would make the output depend on nothing a caller
    can see.
    """
    parent = list(range(len(items)))

    def find(x: int) -> int:
        # Iterative with path compression — a recursive find would be bounded
        # by the recursion limit on a pool this code has no reason to bound.
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        # Union toward the LOWER index so a group's root is always its
        # first-seen member. Union-by-rank would be marginally faster and would
        # make the representative arbitrary, which is exactly the determinism
        # the ordering contract above depends on.
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_by_id: dict[str, int] = {}
    first_by_content: dict[str, int] = {}
    for i, item in enumerate(items):
        # ``item.id`` is tested for TRUTH, not for ``is not None``: an id of
        # ``""`` is a backend saying "I have no key", and admitting it as a
        # real identity put every such record into one group and dropped all
        # but the best-scored. ``ToolMemory._coerce_id`` already normalises
        # ``""`` to ``None``, so the strict check also made the same store
        # dedupe differently depending on whether its rows arrived through the
        # tool adapter or through ``VectorMemory``, which passes ``Chunk.id``
        # straight through.
        if mode == "id" and item.id:
            union(first_by_id.setdefault(item.id, i), i)
        digest = _content_digest(item.content)
        if digest is not None:
            union(first_by_content.setdefault(digest, i), i)

    # Group members in first-seen order, keyed by root (which is itself the
    # first-seen index, so iterating roots in encounter order is stable).
    groups: dict[int, list[MemoryItem]] = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)

    out: list[MemoryItem] = []
    for members in groups.values():
        winner = members[0]
        for candidate in members[1:]:
            if _rank(candidate.score) > _rank(winner.score):
                winner = candidate
        if len(members) == 1:
            out.append(winner)
            continue
        # Order-preserving unique source names. Additive stamp: the winner's
        # own metadata (chunk id, path) survives, or the dedupe would trade one
        # lost signal for another.
        #
        # Two keys rather than one because they answer different questions.
        # ``dedupe_sources`` is INDEPENDENT corroboration — that a vector store
        # and a journal both returned this fact is real evidence a reranker
        # should use, and it is exactly what concatenation destroyed. A single
        # backend returning one record twice (an over-eager chunker) is not
        # corroboration, so its name appears once; ``dedupe_count`` is what
        # still reports that two copies arrived.
        #
        # A member that ALREADY carries a stamp has its stamp ABSORBED rather
        # than overwritten, because this class is documented to nest ("they
        # nest arbitrarily") and rebuilding the stamp from each member's
        # ``source`` field corrupted it there. An inner composite collapsing
        # vector+journal hands the outer one a survivor whose ``source`` is
        # just ``"vector"``, so ``journal`` vanished from the list and the
        # count said 2 where three copies had collapsed — the corroboration
        # signal, which is the whole output of the feature, reported smaller
        # than the truth and with the wrong names.
        agreed: list[str] = []
        collapsed = 0
        for m in members:
            agreed.extend(_prior_sources(m))
            collapsed += _prior_count(m)
        agreed = list(dict.fromkeys(agreed))
        out.append(
            dataclasses.replace(
                winner,
                metadata={
                    **winner.metadata,
                    DEDUPE_SOURCES_KEY: agreed,
                    DEDUPE_COUNT_KEY: collapsed,
                },
            )
        )
    return out


def _prior_sources(item: MemoryItem) -> list[str]:
    """The source names an item already represents — its own, or the ones a
    previous merge recorded on it.

    Defensive about the stored value because the stamp travels through
    storage: most backends round-trip ``metadata`` verbatim, so a persisted or
    hand-written ``dedupe_sources`` can be any JSON shape. A malformed
    annotation degrades to "this item counts as its own source" and never
    raises — a bad value in a passenger field must not take down the query path
    that merely carried it."""
    prior = item.metadata.get(DEDUPE_SOURCES_KEY)
    if isinstance(prior, list):
        names = [s for s in prior if isinstance(s, str)]
        if names:
            return names
    return [item.source]


def _prior_count(item: MemoryItem) -> int:
    """How many copies this item already represents. See ``_prior_sources``
    for why a stored value is not trusted to be an ``int``; ``bool`` is
    excluded because ``isinstance(True, int)`` would score a flag as a count."""
    prior = item.metadata.get(DEDUPE_COUNT_KEY)
    if isinstance(prior, int) and not isinstance(prior, bool) and prior >= 1:
        return prior
    return 1


class CompositeWriteError(RuntimeError):
    """At least one source in a ``CompositeMemory.write`` fan-out failed.

    Carries a structured per-source split so callers can compensate
    (retry the failed sources, or surface the partial-commit state to
    the operator). Raising the first child exception verbatim would
    discard everything else and postmortems would lose the information
    about which backends DID commit.

    ``refused`` is a THIRD bucket, not a slice of ``accepted``. A source
    that declares ``accepts_writes = False`` (``ReadOnlyMemory`` with
    ``on_write="ignore"``) returns from ``write`` without raising and
    without storing anything, so the two-bucket split had exactly one
    place to put it and that place was a lie: ``accepted`` is what an
    operator reads to decide which backends NOT to replay after a partial
    commit, and replaying is skipped for a source that never committed.
    Keyword-only with a default so the constructor stays
    backward-compatible for callers that build the error themselves.

    Attributes:
        accepted: list of source names whose ``write`` succeeded.
        refused: list of source names that declared themselves read-only
            and dropped the write. Not an error — the fan-out still
            succeeds when nothing failed — but never conflated with
            ``accepted``.
        failed: dict mapping source name → BaseException raised by that
            source.
    """

    def __init__(
        self,
        *,
        accepted: list[str],
        failed: dict[str, BaseException],
        refused: list[str] | None = None,
    ) -> None:
        self.accepted = accepted
        self.refused = list(refused or [])
        self.failed = failed
        failed_summary = ", ".join(
            f"{name}: {type(exc).__name__}: {exc}" for name, exc in failed.items()
        )
        # ``refused`` only reaches the message when non-empty — the
        # common split is two buckets and an operator should not have to
        # read past "refused: []" to find the failures.
        refused_summary = f"; refused (read-only): {self.refused}" if self.refused else ""
        super().__init__(
            f"CompositeMemory.write partial failure — "
            f"accepted: {accepted or '(none)'}{refused_summary}; failed: {failed_summary}"
        )


@dataclass(slots=True)
class CompositeMemory:
    """Parallel fan-out across many sources.

    Each query runs against every source concurrently. Results are
    merged, then optionally reranked. Top-k is taken from the reranked
    list. The default reranker is ``score_sort_rerank`` (sort by score
    desc); pass a cross-encoder or LLM judge for richer semantics.

    Backpressure: ``asyncio.gather`` is bounded by Python's default
    semaphore behaviour — for very wide trees, prefer nesting
    ``CompositeMemory(CompositeMemory(...), ...)`` to keep fanout
    moderate at each level.

    ``dedupe`` decides what happens when two sources hold the SAME fact,
    which in this composition is the normal case rather than an accident:
    ask a vector store and a journal the same question and they overlap,
    because the journal is usually what the store was built from. The merge
    used to be ``[item for batch in batches for item in batch]`` — pure
    concatenation — so a reranker scored both copies and the top-k the model
    saw was one fact occupying two slots. That is a WRONG answer, not a
    missing feature, which is why ``"id"`` is the DEFAULT and ``None`` is the
    opt-out a caller chooses rather than inherits:

    - ``"id"``      — merge on a shared ``MemoryItem.id`` OR identical
      content. A union rather than "id, else content" because a per-item
      fallback would mean that populating ``id`` on a backend stops it
      matching backends that have none — the headline case would regress the
      moment ``VectorMemory`` started reporting its chunk ids. See
      ``_merge_duplicates``.
    - ``"content"`` — content only, for stores whose ids are not comparable
      (two stores can both call their first row ``"1"``, and trusting that
      merges two unrelated facts, which deletes one). The single behavioural
      difference from ``"id"``: a shared id with different text.
    - ``None``      — the old concatenation, kept for callers who genuinely
      want every copy (a diversity reranker, a provenance audit).

    A survivor of an actual collision is stamped with ``dedupe_sources`` (the
    distinct source names that produced a copy) and ``dedupe_count``. Items
    that never collided come back exactly as their backend returned them.

    Dedupe runs BEFORE rerank, which is the point of it: a cross-encoder
    scoring the same passage twice pays double and then still spends two
    top-k slots on it. Cost is one sha256 per id-less item (~1 µs for a 1 KB
    passage) against the network round trips that produced the pool.
    """

    sources: list[MemorySource]
    reranker: Reranker | None = None
    name: str = field(default="composite")
    dedupe: DedupeMode | None = "id"

    @property
    def accepts_writes(self) -> bool:
        """True when at least one member can actually commit.

        A composite is itself a ``MemorySource`` and composites nest, so
        without this the marker dies at the first composite boundary: an
        all-read-only ``CompositeMemory`` nested in an outer fan-out
        returns normally from ``write`` having stored nothing, and the
        outer split files it under ``accepted`` — the exact lie the
        ``refused`` bucket exists to prevent, one level up.

        ``any`` rather than ``all`` because the bucket asks "did anything
        land here?", and a composite with one writable member did commit.
        An empty composite commits nothing, and ``any(())`` is ``False``,
        which is the honest answer."""
        return any(bool(getattr(s, "accepts_writes", True)) for s in self.sources)

    async def query(
        self, query: str, *, k: int, ctx: Ctx, where: dict[str, Any] | None = None
    ) -> list[MemoryItem]:
        if not self.sources:
            return []
        # Fan out concurrently; each source returns at most k items, so
        # the merged pool is at most ``len(sources) * k``. The reranker
        # narrows to k.
        batches = await asyncio.gather(
            *(s.query(query, k=k, ctx=ctx, where=where) for s in self.sources)
        )
        merged: list[MemoryItem] = [item for batch in batches for item in batch]
        if self.dedupe is not None:
            merged = _merge_duplicates(merged, self.dedupe)
        rerank = self.reranker.rerank if self.reranker is not None else score_sort_rerank
        return await rerank(query, merged, k=k)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        """Broadcast writes to every source. A backend that can't accept
        writes implements ``write`` as a no-op (see ``ToolMemory``) and
        sets ``accepts_writes = False`` so the split can tell it apart
        from a backend that committed.

        Partial failures surface as :class:`CompositeWriteError` with a
        per-source accepted/refused/failed split. Without
        ``return_exceptions=True`` the first child exception would
        propagate and the rest would be discarded, so a caller seeing a
        failure would have no way to know which backends had already
        committed."""
        materialised = list(items)
        if not materialised:
            return
        # Run every write to completion. Each source's outcome lands in
        # the results list (either ``None`` for success or the raised
        # exception); the wave never short-circuits on the first failure
        # so a slow backend isn't blamed for an early one's crash.
        outcomes = await asyncio.gather(
            *(s.write(materialised, ctx=ctx) for s in self.sources),
            return_exceptions=True,
        )
        accepted: list[str] = []
        refused: list[str] = []
        failed: dict[str, BaseException] = {}
        for source, outcome in zip(self.sources, outcomes, strict=True):
            name = getattr(source, "name", source.__class__.__name__)
            if isinstance(outcome, BaseException):
                failed[name] = outcome
            elif not getattr(source, "accepts_writes", True):
                # A source that returned normally having stored nothing —
                # ``ReadOnlyMemory(..., on_write="ignore")``, or any
                # backend that sets the marker. ``getattr`` with a
                # permissive default so every source written before the
                # marker existed keeps behaving exactly as it did.
                refused.append(name)
            else:
                accepted.append(name)
        if failed:
            raise CompositeWriteError(accepted=accepted, failed=failed, refused=refused)


@dataclass(slots=True)
class SequentialMemory:
    """Try sources in order; stop when ``k`` items are collected.

    Use for cache-then-fallback patterns: a fast in-memory source first,
    an expensive vector source second. The second source isn't hit if
    the first answered. Items returned in the order sources were
    declared (preserves the "cache hit beats fresh fetch" intent).

    NO ``dedupe`` here, deliberately, and it is not because the exposure is
    absent: a cache that returns 3 of 5 wanted items is followed by a vector
    lookup that can return copies of those same 3, so duplicates reach the
    caller exactly as they did in ``CompositeMemory``. The difference is that
    the fix cannot be a merge pass. This class asks each source for
    ``k - len(collected)`` items, so collapsing a duplicate afterwards leaves a
    short list with nothing left to backfill from — the source that could have
    supplied the replacement has already been asked and answered. Doing it
    honestly means re-querying with a widened ``k`` (or over-fetching every
    tier by the expected overlap), which changes the round-trip count and the
    cost profile of the shape whose entire purpose is not making the second
    call. That is a design decision with a latency budget attached, not a
    merge, so it is left as its own item rather than smuggled in here. Wrap
    the chain in ``CompositeMemory(sources=[chain], dedupe="id")`` for a
    single-arm dedupe in the meantime.
    """

    sources: list[MemorySource]
    name: str = field(default="sequential")

    @property
    def accepts_writes(self) -> bool:
        """Mirrors the FIRST source, because that is the only one
        ``write`` ever touches — downstream sources are read-only from
        this composite's point of view. A ``SequentialMemory`` whose
        cache tier is a ``ReadOnlyMemory`` stores nothing on write, and
        an enclosing fan-out must not report that as a commit. No
        sources means no write target, hence ``False``."""
        if not self.sources:
            return False
        return bool(getattr(self.sources[0], "accepts_writes", True))

    async def query(
        self, query: str, *, k: int, ctx: Ctx, where: dict[str, Any] | None = None
    ) -> list[MemoryItem]:
        collected: list[MemoryItem] = []
        for s in self.sources:
            if len(collected) >= k:
                break
            remaining = k - len(collected)
            collected.extend(await s.query(query, k=remaining, ctx=ctx, where=where))
        return collected[:k]

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        """Write to the FIRST source only — typical cache-tier pattern.
        Downstream sources are read-only from this composite's POV."""
        materialised = list(items)
        if not materialised or not self.sources:
            return
        await self.sources[0].write(materialised, ctx=ctx)


__all__ = [
    "DEDUPE_COUNT_KEY",
    "DEDUPE_SOURCES_KEY",
    "CompositeMemory",
    "CompositeWriteError",
    "DedupeMode",
    "SequentialMemory",
]
