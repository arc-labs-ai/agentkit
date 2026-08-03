"""Failure as first-class data (ch24 / R13): classify-backed construction + composition.
Assertions are classify-agnostic (we test the retriable invariant + aggregation, not classify's verdicts)."""

from agentkit.kernel.errors import Failure, compose_failures
from agentkit.kernel.resilience import ErrorClass


def test_failure_of_sets_fields_and_retriable_invariant():
    f = Failure.of(TimeoutError("upstream timeout"), source="fetch", partial_output={"x": 1})
    assert f.source == "fetch"
    assert isinstance(f.cause, TimeoutError)
    assert f.partial_output == {"x": 1}
    assert f.message
    # The retriable invariant matches ``run_with_resilience``: retriable
    # for any non-PERMANENT category (TRANSIENT + UNKNOWN).
    assert f.retriable == (f.category != ErrorClass.PERMANENT)


def test_compose_none_returns_none():
    assert compose_failures([None, None]) is None


def test_compose_single_passthrough():
    one = Failure(ErrorClass.PERMANENT, "a", "boom")
    assert compose_failures([None, one]) is one


def test_compose_permanent_dominates():
    a = Failure(ErrorClass.TRANSIENT, "x", "t")
    b = Failure(ErrorClass.PERMANENT, "y", "p")
    agg = compose_failures([a, b], source="parent")
    assert agg.category == ErrorClass.PERMANENT and agg.retriable is False
    assert agg.children == (a, b) and agg.source == "parent" and "2 failures" in agg.message


def test_compose_all_transient_is_retriable():
    agg = compose_failures(
        [Failure(ErrorClass.TRANSIENT, "x", "t1"), Failure(ErrorClass.TRANSIENT, "y", "t2")]
    )
    assert agg.category == ErrorClass.TRANSIENT and agg.retriable is True


def test_compose_mixed_is_unknown_and_retriable():
    """A mixed TRANSIENT+UNKNOWN aggregate is UNKNOWN and retriable —
    matches ``run_with_resilience``, where being conservative on UNKNOWN
    means trying again."""
    agg = compose_failures(
        [Failure(ErrorClass.TRANSIENT, "x", "t"), Failure(ErrorClass.UNKNOWN, "y", "u")]
    )
    assert agg.category == ErrorClass.UNKNOWN and agg.retriable is True


def test_compose_unknown_only_is_retriable():
    """All-UNKNOWN aggregate also reads retriable, for the same reason."""
    agg = compose_failures(
        [Failure(ErrorClass.UNKNOWN, "x", "u1"), Failure(ErrorClass.UNKNOWN, "y", "u2")]
    )
    assert agg.category == ErrorClass.UNKNOWN and agg.retriable is True


def test_failure_of_unknown_is_retriable():
    """``Failure.of`` on an UNKNOWN-classified exception reads retriable
    so callers reading ``.retriable`` make the same call the kernel
    would have."""
    f = Failure.of(RuntimeError("opaque error nobody categorized"), source="x")
    # The classify call returns UNKNOWN for this message (no transient/
    # permanent substring match), so this pins both the classification
    # AND the retriable invariant in one place.
    assert f.category == ErrorClass.UNKNOWN
    assert f.retriable is True


# ── Nested composites: deep aggregation semantics ───────────────────────────
#
# Real production graphs produce depth-3 Failure trees: a Workflow
# aggregates node failures, each node's Failure may itself aggregate
# sub-agent failures. The composite function preserves the tree
# rather than flattening — the audit trail keeps every hop's source
# attribution. But the category / retriable roll-up must still be
# correct at every depth.


def test_compose_of_composites_preserves_nesting_not_flattening():
    """A composite of composites keeps the inner composites AS
    children — no flattening. The audit trail can walk the tree and
    see which sub-run produced which cluster of failures."""
    from agentkit.kernel.errors import compose_failures

    leaves_a = [
        Failure(category=ErrorClass.TRANSIENT, source="researcher-a", message="429"),
        Failure(category=ErrorClass.TRANSIENT, source="researcher-b", message="timeout"),
    ]
    leaves_b = [
        Failure(category=ErrorClass.TRANSIENT, source="researcher-c", message="429"),
    ]
    inner_a = compose_failures(leaves_a, source="researcher-group-a")
    inner_b = compose_failures(leaves_b, source="researcher-group-b")

    outer = compose_failures([inner_a, inner_b], source="run")

    assert outer is not None
    # Two children: the two inner composites, not the four leaves.
    assert len(outer.children) == 2
    assert outer.children[0] is inner_a
    assert outer.children[1] is inner_b
    # Depth-2 grandchildren are still walkable — one composite has
    # two leaves, the other passes-through the single leaf.
    assert len(outer.children[0].children) == 2  # inner_a is a composite
    assert outer.children[1] is leaves_b[0]  # inner_b was single → passthrough


def test_compose_permanent_dominates_through_nesting():
    """PERMANENT is contagious upward. A depth-3 tree where a single
    leaf is PERMANENT rolls up PERMANENT at the top — critically,
    ``retriable`` at the top is ``False`` even though most of the
    tree is TRANSIENT."""
    from agentkit.kernel.errors import compose_failures

    perm_leaf = Failure(category=ErrorClass.PERMANENT, source="deep", message="403 forbidden")
    trans_leaves = [
        Failure(category=ErrorClass.TRANSIENT, source="a", message="429"),
        Failure(category=ErrorClass.TRANSIENT, source="b", message="timeout"),
    ]
    inner = compose_failures([*trans_leaves, perm_leaf], source="mid")
    outer = compose_failures([inner], source="top")

    # Depth-2 inner: PERMANENT dominates the leaf set.
    assert inner is not None and inner.category == ErrorClass.PERMANENT
    assert inner.retriable is False
    # Depth-1 outer: single-child passthrough — inner surfaces as-is,
    # so the top is also PERMANENT / non-retriable.
    assert outer is inner


def test_compose_all_transient_still_transient_at_any_depth():
    """Symmetric to PERMANENT-dominates: if every leaf is TRANSIENT,
    every level of the tree stays TRANSIENT (and retriable). This is
    the "all children are retryable, so the aggregate is retryable"
    story that ``run_with_resilience`` can act on at each layer."""
    from agentkit.kernel.errors import compose_failures

    def _t(name: str) -> Failure:
        return Failure(category=ErrorClass.TRANSIENT, source=name, message="429")

    inner_a = compose_failures([_t("a1"), _t("a2")], source="inner-a")
    inner_b = compose_failures([_t("b1"), _t("b2")], source="inner-b")
    top = compose_failures([inner_a, inner_b], source="top")

    assert top is not None
    assert top.category == ErrorClass.TRANSIENT
    assert top.retriable is True
    # Every level down the tree agrees.
    assert all(child.category == ErrorClass.TRANSIENT for child in top.children)


def test_compose_mixed_transient_and_unknown_rolls_up_unknown():
    """The category rule: TRANSIENT-only aggregates → TRANSIENT;
    anything else non-PERMANENT → UNKNOWN. Verify a mix of
    TRANSIENT + UNKNOWN at depth-1 rolls up to UNKNOWN (retriable)."""
    from agentkit.kernel.errors import compose_failures

    inner = compose_failures(
        [
            Failure(category=ErrorClass.TRANSIENT, source="a", message="429"),
            Failure(category=ErrorClass.UNKNOWN, source="b", message="opaque"),
        ],
        source="inner",
    )
    assert inner is not None and inner.category == ErrorClass.UNKNOWN
    # Retriable — UNKNOWN is conservative-retry (matches ``run_with_resilience``).
    assert inner.retriable is True


def test_compose_all_none_or_empty_returns_none():
    """An aggregate with no real failures is None — the caller uses
    ``is None`` as the "no failure" signal, without needing to
    unpack a Failure with an empty children tuple. Keeps the value
    shape simple: a Failure ALWAYS represents at least one real
    failure."""
    from agentkit.kernel.errors import compose_failures

    assert compose_failures([]) is None
    assert compose_failures([None, None, None]) is None
