"""``CompositeMemory`` fan-out dedupe — the one memory bug that returns a WRONG
answer rather than a missing capability.

Before this, the merge was ``[item for batch in batches for item in batch]``:
pure concatenation. Two sources holding the same fact returned it twice, the
reranker scored both copies, and the top-k the model saw was one fact occupying
two slots. That is worst in exactly the composition the class exists for —
asking a vector store and a journal the same question, where the overlap is not
incidental but the normal case, because the journal is usually what the vector
store was built from.

The tests below pin the four things the fix has to get right: the collapse
itself, WHICH copy survives (the higher-scored one), what the survivor
remembers about the copies it absorbed (that two independent sources agreed is
signal a reranker should use, and concatenation threw it away), and that
``dedupe=None`` still reproduces the old output item for item for anyone who
wants concatenation on purpose.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem, score_sort_rerank
from agentkit.memory.composite import CompositeMemory
from agentkit.testing import make_test_ctx


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _Source:
    """Test double returning a fixed batch. ``k`` is ignored on purpose — the
    unit under test is the merge, not a backend's own top-k."""

    name: str
    items: list[MemoryItem] = field(default_factory=list)

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del query, k, ctx, where
        return list(self.items)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        del items, ctx


# ── the four the spec demands ───────────────────────────────────────────────


def test_two_sources_holding_one_fact_with_one_id_yield_one_item():
    """The headline case. A journal row that was also indexed into the vector
    store carries the same record id in both, so the fan-out sees it twice."""
    vector = _Source(
        "vector", [MemoryItem(content="certs rotate every 90d", source="vector", id="r7", score=0.9)]
    )
    journal = _Source(
        "journal", [MemoryItem(content="certs rotate every 90d", source="journal", id="r7", score=0.4)]
    )
    out = _run(CompositeMemory(sources=[vector, journal]).query("certs", k=5, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].content == "certs rotate every 90d"


def test_the_survivor_keeps_the_higher_score_and_names_both_sources():
    """Keeping the higher score is what makes the collapse safe in front of a
    reranker: the fact must not be demoted because one of its two holders
    ranked it badly. Naming both sources preserves the agreement signal that
    concatenation only expressed as "it appeared twice"."""
    a = _Source("vector", [MemoryItem(content="f", source="vector", id="r7", score=0.4)])
    b = _Source("journal", [MemoryItem(content="f", source="journal", id="r7", score=0.9)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert [i.score for i in out] == [0.9]
    assert out[0].source == "journal"
    assert out[0].metadata["dedupe_sources"] == ["vector", "journal"]
    assert out[0].metadata["dedupe_count"] == 2


def test_dedupe_none_reproduces_the_old_concatenation_exactly():
    """``dedupe=None`` is today's behaviour kept available as a CHOICE. It has
    to be the old list item-for-item — same duplicates, same order, same
    untouched metadata — or "opt out" would not mean opt out."""
    left = [
        MemoryItem(content="f", source="a", id="r7", score=0.4),
        MemoryItem(content="g", source="a", score=0.1),
    ]
    right = [MemoryItem(content="f", source="b", id="r7", score=0.9)]
    ctx = make_test_ctx()
    cm = CompositeMemory(sources=[_Source("a", left), _Source("b", right)], dedupe=None)
    out = _run(cm.query("q", k=5, ctx=ctx))
    expected = _run(score_sort_rerank("q", [*left, *right], k=5))
    assert out == expected
    assert len(out) == 3
    assert all(i.metadata == {} for i in out)


def test_ids_absent_everywhere_falls_back_to_a_content_digest():
    """A backend with no stable record id (a tool-wrapped search, a scratchpad)
    must still dedupe. Falling through to "no ids, so no dedupe" would leave
    the default silently doing nothing for the backends most likely to overlap
    on text."""
    a = _Source("tool", [MemoryItem(content="the same passage", source="tool", score=0.2)])
    b = _Source("files", [MemoryItem(content="the same passage", source="files", score=0.7)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].score == 0.7
    assert out[0].metadata["dedupe_sources"] == ["tool", "files"]


# ── edges that decide whether the rule is actually total ────────────────────


def test_a_scored_copy_beats_an_unscored_one():
    """``score=None`` is "this backend does not rank", not "this is bad". The
    default reranker sinks ``None`` to the bottom, so letting an unranked copy
    win the collision would push a well-scored fact out of the top-k — the
    exact failure the dedupe exists to prevent, re-introduced by the merge."""
    a = _Source("journal", [MemoryItem(content="f", source="journal", id="r7", score=None)])
    b = _Source("vector", [MemoryItem(content="f", source="vector", id="r7", score=0.3)])
    first = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert [i.score for i in first] == [0.3]
    # ...and the same when the unranked copy arrives SECOND, so the rule is not
    # accidentally "whoever came last".
    c = _Source("vector", [MemoryItem(content="f", source="vector", id="r7", score=0.3)])
    d = _Source("journal", [MemoryItem(content="f", source="journal", id="r7", score=None)])
    second = _run(CompositeMemory(sources=[c, d]).query("q", k=5, ctx=make_test_ctx()))
    assert [i.score for i in second] == [0.3]


def test_both_copies_unscored_keeps_the_first_seen_one():
    """Two unranked backends agreeing is the tie the fan-out actually produces.
    First-seen wins, which makes source declaration order the tiebreak and the
    result reproducible run to run."""
    a = _Source("journal", [MemoryItem(content="f", source="journal", id="r7")])
    b = _Source("scratch", [MemoryItem(content="f", source="scratch", id="r7")])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert [(i.source, i.score) for i in out] == [("journal", None)]


def test_three_way_collision_names_all_three_in_declaration_order():
    """The merge has to be an accumulate, not a pairwise collapse that forgets
    the middle source."""
    srcs = [
        _Source("a", [MemoryItem(content="f", source="a", id="r7", score=0.1)]),
        _Source("b", [MemoryItem(content="f", source="b", id="r7", score=0.8)]),
        _Source("c", [MemoryItem(content="f", source="c", id="r7", score=0.5)]),
    ]
    out = _run(CompositeMemory(sources=srcs).query("q", k=5, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].score == 0.8
    assert out[0].metadata["dedupe_sources"] == ["a", "b", "c"]
    assert out[0].metadata["dedupe_count"] == 3


def test_the_same_id_twice_from_one_source_collapses_but_claims_no_agreement():
    """A backend that returns the same record twice (an over-eager chunker) is
    still a duplicate in the top-k and still collapses. It is NOT agreement,
    though: ``dedupe_sources`` stays one name so a reranker reading it as
    "independent corroboration" is not lied to, while ``dedupe_count`` still
    says two copies arrived."""
    a = _Source(
        "vector",
        [
            MemoryItem(content="f", source="vector", id="r7", score=0.2),
            MemoryItem(content="f", source="vector", id="r7", score=0.6),
        ],
    )
    out = _run(CompositeMemory(sources=[a]).query("q", k=5, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].score == 0.6
    assert out[0].metadata["dedupe_sources"] == ["vector"]
    assert out[0].metadata["dedupe_count"] == 2


def test_same_id_different_content_keeps_the_higher_scored_rendering():
    """The documented call: the id is the declared identity, so two different
    texts under one id are two RENDERINGS of one fact (a raw chunk and a
    summary of it, a stale copy and a fresh one), not two facts. The
    higher-scored rendering survives; the collapse is recorded in metadata so
    it is not silent."""
    a = _Source("vector", [MemoryItem(content="raw chunk", source="vector", id="r7", score=0.3)])
    b = _Source("journal", [MemoryItem(content="fresher summary", source="journal", id="r7", score=0.9)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert [i.content for i in out] == ["fresher summary"]
    assert out[0].metadata["dedupe_count"] == 2


def test_carrying_an_id_never_makes_dedupe_worse_than_not_carrying_one():
    """The trap this design had to avoid, and the reason ``"id"`` mode is a
    UNION of the two relations rather than a per-item fallback.

    Under a fallback ("keyed by id if it has one, else by content"), giving a
    backend an id makes it stop matching backends that have none — so the
    headline scenario REGRESSES the moment ``VectorMemory`` starts reporting
    its chunk ids: a vector hit (id=r7) and the journal row it was built from
    (no id) land under different keys and the duplicate survives. Populating an
    id would have made the answer worse, which is not a property any field
    should have. So ``"id"`` mode merges on same-id OR same-content."""
    vector = _Source("vector", [MemoryItem(content="certs rotate every 90d", source="vector", id="r7", score=0.9)])
    journal = _Source("journal", [MemoryItem(content="certs rotate every 90d", source="journal", score=0.4)])
    out = _run(CompositeMemory(sources=[vector, journal]).query("q", k=9, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].score == 0.9
    assert out[0].metadata["dedupe_sources"] == ["vector", "journal"]


def test_mixed_ids_merge_on_either_relation_and_the_grouping_is_transitive():
    """Mixed is the case the spec's "when every item carries one" does not
    name. Three copies of one fact — one with an id, one with the SAME id but
    a different rendering, one with neither but matching text — must land in
    one group, because the relations chain: the id links the first two, the
    text links the first and third."""
    a = _Source(
        "vector",
        [
            MemoryItem(content="rotate certs", source="vector", id="r7", score=0.5),
            MemoryItem(content="unrelated", source="vector", score=0.6),
        ],
    )
    b = _Source("journal", [MemoryItem(content="rotate the certs quarterly", source="journal", id="r7", score=0.2)])
    c = _Source("tool", [MemoryItem(content="rotate certs", source="tool", score=0.7)])
    out = _run(CompositeMemory(sources=[a, b, c]).query("q", k=9, ctx=make_test_ctx()))
    assert sorted(i.content for i in out) == ["rotate certs", "unrelated"]
    merged = next(i for i in out if i.content == "rotate certs")
    assert merged.score == 0.7
    assert merged.metadata["dedupe_sources"] == ["vector", "journal", "tool"]
    assert merged.metadata["dedupe_count"] == 3


def test_content_mode_refuses_to_merge_on_an_id_alone():
    """POSITIVE CONTROL for the mode split — it held under the per-item
    fallback this design replaced, and must survive the change to a union.

    It is the ONLY thing that separates the two modes, so it is the only thing
    worth testing them apart on. ``"content"`` is for stores whose ids
    are not comparable — two stores can both call their first row ``"1"`` —
    and trusting an id there merges two unrelated facts, which deletes one. So
    ``"content"`` compares text and nothing else; the same pair collapses under
    ``"id"``, where the shared key is taken at its word."""
    a = _Source("store-a", [MemoryItem(content="refunds take 5 days", source="store-a", id="1", score=0.2)])
    b = _Source("store-b", [MemoryItem(content="store credit never expires", source="store-b", id="1", score=0.7)])
    pair = [a, b]
    assert len(_run(CompositeMemory(sources=pair, dedupe="content").query("q", k=5, ctx=make_test_ctx()))) == 2
    assert len(_run(CompositeMemory(sources=pair, dedupe="id").query("q", k=5, ctx=make_test_ctx()))) == 1


def test_content_dedupe_ignores_leading_and_trailing_whitespace():
    """A journal line ends in a newline and the chunk indexed from it does not.
    That is the normal case, not a corner one, so the digest is taken of the
    stripped text. Interior whitespace is left alone — indentation inside a
    code chunk is content."""
    a = _Source("journal", [MemoryItem(content="rotate certs\n", source="journal", score=0.1)])
    b = _Source("vector", [MemoryItem(content="  rotate certs", source="vector", score=0.6)])
    out = _run(CompositeMemory(sources=[a, b], dedupe="content").query("q", k=5, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].score == 0.6
    # ...but interior whitespace still separates two items.
    c = _Source("a", [MemoryItem(content="a  b", source="a")])
    d = _Source("b", [MemoryItem(content="a b", source="b")])
    both = _run(CompositeMemory(sources=[c, d], dedupe="content").query("q", k=5, ctx=make_test_ctx()))
    assert len(both) == 2


def test_no_sources_and_empty_batches_stay_empty():
    """POSITIVE CONTROL — passed before the dedupe existed and must keep
    passing. The merge must not divide by the number of sources or index [0];
    a fold written as ``reduce`` over batches would fail the first line here."""
    assert _run(CompositeMemory(sources=[]).query("q", k=5, ctx=make_test_ctx())) == []
    empty = [_Source("a"), _Source("b", [MemoryItem(content="only", source="b")])]
    out = _run(CompositeMemory(sources=empty).query("q", k=5, ctx=make_test_ctx()))
    assert [i.content for i in out] == ["only"]


def test_an_item_that_never_collided_is_returned_untouched():
    """POSITIVE CONTROL — the pass-through case, which concatenation also got
    right and the dedupe must not break. Stamping every item would make
    ``dedupe`` visible in the metadata of every result and break equality
    against the item the backend returned; only a survivor of an actual
    collision is annotated."""
    a = _Source("a", [MemoryItem(content="x", source="a", score=0.5, metadata={"path": "/p"})])
    out = _run(CompositeMemory(sources=[a]).query("q", k=5, ctx=make_test_ctx()))
    assert out == [MemoryItem(content="x", source="a", score=0.5, metadata={"path": "/p"})]
    assert "dedupe_sources" not in out[0].metadata


def test_the_stamp_preserves_the_backends_own_metadata():
    """The annotation is additive. Dropping the winner's chunk id or path to
    make room for it would trade one lost signal for another."""
    a = _Source("a", [MemoryItem(content="x", source="a", id="r7", score=0.5, metadata={"path": "/p"})])
    b = _Source("b", [MemoryItem(content="x", source="b", id="r7", score=0.1)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert out[0].metadata["path"] == "/p"
    assert out[0].metadata["dedupe_count"] == 2


def test_dedupe_frees_top_k_slots_for_distinct_facts():
    """The whole point, stated as the user-visible symptom: with k=2 and one
    fact held by both sources, concatenation handed the model that fact twice
    and dropped the second-best distinct fact entirely."""
    a = _Source(
        "vector",
        [
            MemoryItem(content="dup", source="vector", id="r7", score=0.90),
            MemoryItem(content="other", source="vector", id="r8", score=0.50),
        ],
    )
    b = _Source("journal", [MemoryItem(content="dup", source="journal", id="r7", score=0.89)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=2, ctx=make_test_ctx()))
    assert [i.content for i in out] == ["dup", "other"]


def test_merged_order_is_stable_across_repeated_runs():
    """Determinism is a contract: the same sources returning the same batches
    must produce identical output, or a cached prompt stops being cached and an
    eval stops being reproducible. First-seen order feeds a stable sort, so
    ties break on source declaration order every time."""
    srcs = [
        _Source(
            "a",
            [
                MemoryItem(content="p", source="a", score=0.5),
                MemoryItem(content="q", source="a", score=0.5),
            ],
        ),
        _Source(
            "b",
            [
                MemoryItem(content="r", source="b", score=0.5),
                MemoryItem(content="p", source="b", score=0.5),
            ],
        ),
    ]
    cm = CompositeMemory(sources=srcs)
    runs = [[i.content for i in _run(cm.query("q", k=9, ctx=make_test_ctx()))] for _ in range(5)]
    assert runs == [["p", "q", "r"]] * 5


def test_the_reranker_sees_the_deduped_pool_not_the_raw_one():
    """Dedupe has to happen BEFORE rerank. A cross-encoder scoring the same
    passage twice pays double and then still spends two top-k slots on it."""
    seen: list[list[MemoryItem]] = []

    @dataclass
    class _R:
        async def rerank(self, query: str, items: list[MemoryItem], *, k: int) -> list[MemoryItem]:
            del query
            seen.append(list(items))
            return items[:k]

    a = _Source("a", [MemoryItem(content="f", source="a", id="r7", score=0.1)])
    b = _Source("b", [MemoryItem(content="f", source="b", id="r7", score=0.2)])
    _run(CompositeMemory(sources=[a, b], reranker=_R()).query("q", k=5, ctx=make_test_ctx()))
    assert len(seen[0]) == 1


def test_dedupe_defaults_to_id_mode():
    """The default is the fix, not the bug: a caller who never heard of this
    parameter stops getting duplicates. ``dedupe=None`` is available for anyone
    who wants concatenation, but they have to say so."""
    assert CompositeMemory(sources=[]).dedupe == "id"


# ── degenerate identities: an absent key must not read as a shared one ──────


def test_an_empty_string_id_is_treated_as_no_id_at_all():
    """An id of ``""`` is a backend saying "I have no key", not "my key is the
    empty string" — and the union rule turns that difference into deleted facts.

    ``item.id is not None`` let ``""`` through as a real identity, so every
    such record in the pool landed in ONE group and all but the best-scored
    were dropped. ``ToolMemory._coerce_id`` already normalises ``""`` to
    ``None`` (``return raw or None``), so before this the same store deduped
    differently depending on whether its rows reached the pool through the tool
    adapter or through ``VectorMemory``, which passes ``Chunk.id`` straight
    through."""
    a = _Source("vector", [MemoryItem(content="fact one", source="vector", id="", score=0.9)])
    b = _Source("journal", [MemoryItem(content="fact two", source="journal", id="", score=0.8)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert [i.content for i in out] == ["fact one", "fact two"]


def test_blank_content_is_not_evidence_that_two_records_are_the_same_fact():
    """The content relation fires in BOTH modes and needs no id, so a blank
    digest was the widest way to lose data here.

    ``_content_digest`` hashes ``content.strip()``, which maps ``""``,
    ``"   "`` and ``"\\n\\t"`` onto one digest. Three records with three
    DISTINCT ids therefore chained into a single group and two facts were
    deleted — and empty-after-strip content is reachable without malice: a
    chunk that is whitespace once boilerplate is stripped, a tool hit with an
    empty snippet, a journal row with a blank body. Absence of text is not
    evidence of sameness."""
    a = _Source("vector", [MemoryItem(content="   ", source="vector", id="a", score=0.9)])
    b = _Source("journal", [MemoryItem(content="\n\t", source="journal", id="b", score=0.8)])
    c = _Source("tool", [MemoryItem(content="", source="tool", id="c", score=0.7)])
    out = _run(CompositeMemory(sources=[a, b, c]).query("q", k=5, ctx=make_test_ctx()))
    assert [i.id for i in out] == ["a", "b", "c"]


def test_blank_items_sharing_an_id_still_collapse():
    """The other half of the rule above: blank content removes the CONTENT
    relation, it does not make a record un-mergeable. Two backends returning
    the same empty row still share a declared identity."""
    a = _Source("vector", [MemoryItem(content="", source="vector", id="r7", score=0.2)])
    b = _Source("journal", [MemoryItem(content="   ", source="journal", id="r7", score=0.9)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].score == 0.9


# ── nesting: the composite is documented to compose with itself ─────────────


def test_a_nested_composite_reports_every_source_that_agreed_not_just_two():
    """``dedupe_sources`` is the feature's OUTPUT signal — "how many
    independent backends returned this?" — and nesting silently corrupted it.

    An inner composite collapses vector+journal and stamps the survivor; the
    outer composite then merges that survivor with a third copy. The outer
    stamp was rebuilt from each member's ``source`` FIELD, which on the inner
    survivor is just ``"vector"`` — so ``journal`` vanished from the list and
    ``dedupe_count`` said 2 when three copies had collapsed. A reranker
    branching on corroboration got a smaller number and the wrong names, in the
    exact topology the class docstring advertises ("they nest arbitrarily")."""
    inner = CompositeMemory(
        sources=[
            _Source(
                "vector",
                [MemoryItem(content="certs rotate every 90d", source="vector", id="r7", score=0.9)],
            ),
            _Source(
                "journal",
                [MemoryItem(content="certs rotate every 90d", source="journal", id="r7", score=0.5)],
            ),
        ],
        name="inner",
    )
    tool = _Source(
        "tool", [MemoryItem(content="certs rotate every 90d", source="tool", id="r7", score=0.3)]
    )
    out = _run(CompositeMemory(sources=[inner, tool]).query("q", k=5, ctx=make_test_ctx()))
    assert len(out) == 1
    assert out[0].metadata["dedupe_count"] == 3
    assert out[0].metadata["dedupe_sources"] == ["vector", "journal", "tool"]


def test_an_existing_stamp_is_absorbed_rather_than_overwritten():
    """A member arriving with the reserved keys already set is not a hostile
    input, it is the nested case above seen from one level down — and any
    backend that round-trips metadata hands one back too. Absorbing keeps the
    count honest; overwriting silently replaced an upstream corroboration
    record with this level's."""
    a = _Source(
        "s1",
        [
            MemoryItem(
                content="t",
                source="s1",
                id="z",
                score=0.9,
                metadata={"dedupe_sources": ["upstream_a", "upstream_b"], "dedupe_count": 2},
            )
        ],
    )
    b = _Source("s2", [MemoryItem(content="t", source="s2", id="z", score=0.5)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert out[0].metadata["dedupe_count"] == 3
    assert out[0].metadata["dedupe_sources"] == ["upstream_a", "upstream_b", "s2"]


def test_a_corrupt_stamp_from_a_backend_does_not_crash_the_merge():
    """The stamp travels through storage — any backend that round-trips
    metadata can hand back a ``dedupe_count`` that is a string or a
    ``dedupe_sources`` that is not a list. Absorbing has to degrade to counting
    the member, never raise: a malformed annotation must not take down the
    query path that merely passed through it."""
    a = _Source(
        "s1",
        [
            MemoryItem(
                content="t",
                source="s1",
                id="z",
                score=0.9,
                metadata={"dedupe_sources": "not-a-list", "dedupe_count": "seven"},
            )
        ],
    )
    b = _Source("s2", [MemoryItem(content="t", source="s2", id="z", score=0.5)])
    out = _run(CompositeMemory(sources=[a, b]).query("q", k=5, ctx=make_test_ctx()))
    assert out[0].metadata["dedupe_count"] == 2
    assert out[0].metadata["dedupe_sources"] == ["s1", "s2"]
