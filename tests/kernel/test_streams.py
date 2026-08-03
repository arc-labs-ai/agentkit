"""Reactive stream operators (ch29 / R23): the dependency-free Rx-style algebra over async iterables —
map/filter/take/scan/distinct/merge/buffer. Stdlib-asyncio only, deterministic."""

import asyncio

from agentkit.kernel import streams


def _run(coro):
    return asyncio.run(coro)


async def _arange(n):
    for i in range(n):
        yield i


async def _from(items):
    for x in items:
        yield x


def test_map_and_filter_compose_with_sync_and_async_fns():
    async def go():
        async def double(x):  # async mapper
            return x * 2

        s = streams.filter_stream(streams.map_stream(_arange(5), double), lambda x: x % 4 == 0)
        return await streams.collect(s)

    assert _run(go()) == [0, 4, 8]  # 0,2,4,6,8 → keep multiples of 4


def test_take_take_while_take_until():
    async def go():
        a = await streams.collect(streams.take(_arange(10), 3))
        b = await streams.collect(streams.take_while(_from([1, 2, 9, 3]), lambda x: x < 5))
        c = await streams.collect(streams.take_until(_from([1, 2, 9, 3]), lambda x: x == 9))
        return a, b, c

    a, b, c = _run(go())
    assert a == [0, 1, 2]
    assert b == [1, 2]  # stops at 9 (exclusive)
    assert c == [1, 2, 9]  # includes 9 (inclusive)


def test_distinct_until_changed_coalesces_consecutive():
    async def go():
        return await streams.collect(
            streams.distinct_until_changed(_from([1, 1, 2, 2, 2, 1, 3, 3]))
        )

    assert _run(go()) == [1, 2, 1, 3]  # only consecutive dups dropped


def test_scan_is_a_rolling_accumulator():
    async def go():
        return await streams.collect(streams.scan(_arange(4), lambda acc, x: acc + x, 0))

    assert _run(go()) == [0, 1, 3, 6]  # running sum


def test_buffer_batches_and_flushes_tail():
    async def go():
        return await streams.collect(streams.buffer(_arange(5), 2))

    assert _run(go()) == [[0, 1], [2, 3], [4]]  # partial tail flushed


def test_merge_interleaves_all_sources_without_loss():
    async def go():
        merged = streams.merge(_from(["a1", "a2", "a3"]), _from(["b1", "b2"]))
        return await streams.collect(merged)

    got = _run(go())
    assert sorted(got) == ["a1", "a2", "a3", "b1", "b2"]  # order interleaves; nothing dropped


def test_streams_exposed_as_a_namespace():
    import agentkit

    assert agentkit.streams is streams and "streams" in agentkit.__all__


# ── merge cleanup must not deadlock on consumer early-break ──────────────────
#
# When a consumer breaks early with a full queue, the pump's cleanup path must
# not block on ``queue.put``. The sentinel path splits accordingly: natural
# completion awaits, cancelled pumps use ``put_nowait`` and tolerate
# :class:`asyncio.QueueFull`.


def test_merge_early_break_with_full_buffer_does_not_deadlock():
    """Consumer breaks after one item; the buffer is intentionally tiny so
    the pumps fill it and block on ``put``. Merge must complete promptly
    rather than deadlock on the cancelled pump's sentinel put."""

    async def _fast(items):
        for x in items:
            yield x

    async def go():
        merged = streams.merge(
            _fast(["a1", "a2", "a3", "a4"]),
            _fast(["b1", "b2", "b3", "b4"]),
            buffer=1,  # tiny buffer forces pumps to block on put
        )
        first = None
        async for item in merged:
            first = item
            break  # consumer early-break — must not hang
        return first

    # Wrap in a wall-clock timeout — if a deadlock regresses this fails
    # loudly instead of stalling the test runner.
    async def with_timeout():
        return await asyncio.wait_for(go(), timeout=2.0)

    result = _run(with_timeout())
    assert result in {"a1", "b1"}  # whichever pump scheduled first


def test_merge_natural_completion_still_drains_all_sources():
    """The happy path: multiple slow + fast sources are fully drained
    when merge completes naturally."""

    async def _slow():
        for x in ("s1", "s2"):
            await asyncio.sleep(0)  # yield control so other pumps interleave
            yield x

    async def _fast():
        for x in ("f1", "f2", "f3"):
            yield x

    async def go():
        return await streams.collect(streams.merge(_slow(), _fast(), buffer=2))

    got = _run(go())
    assert sorted(got) == ["f1", "f2", "f3", "s1", "s2"]


# ── Operator algebra: property tests ─────────────────────────────────────────
#
# The operators form a small algebra callers compose freely
# (``map(map(s, f), g) == map(s, g∘f)``, ``filter(filter(s, p), q)``
# is equivalent to ``filter(s, p∧q)``, etc.). These laws hold in
# any Rx-shaped algebra; verifying them here means callers can
# reorder / fuse operators locally without breaking downstream
# consumers.

import operator

from hypothesis import given, settings
from hypothesis import strategies as st


def _collect(gen):
    async def _run_it():
        return [x async for x in gen]

    return asyncio.run(_run_it())


def _stream_from(items):
    async def gen():
        for x in items:
            yield x

    return gen()


@given(items=st.lists(st.integers(min_value=-1000, max_value=1000), max_size=30))
@settings(max_examples=50, deadline=None)
def test_map_stream_preserves_length(items) -> None:
    """map_stream(s, f) yields exactly one output per input — a
    transformation, never a filter."""
    out = _collect(streams.map_stream(_stream_from(items), lambda x: x * 2))
    assert len(out) == len(items)


@given(items=st.lists(st.integers(min_value=-100, max_value=100), max_size=20))
@settings(max_examples=50, deadline=None)
def test_map_stream_composition_law(items) -> None:
    """The functor law: ``map(map(s, f), g)`` is observationally
    equal to ``map(s, lambda x: g(f(x)))``. Consumers can fuse two
    map stages into one without changing behaviour."""

    def f(x: int) -> int:
        return x + 1

    def g(x: int) -> int:
        return x * 3

    two_stages = _collect(streams.map_stream(streams.map_stream(_stream_from(items), f), g))
    fused = _collect(streams.map_stream(_stream_from(items), lambda x: g(f(x))))
    assert two_stages == fused


@given(items=st.lists(st.integers(min_value=-100, max_value=100), max_size=30))
@settings(max_examples=50, deadline=None)
def test_filter_stream_composition_law(items) -> None:
    """``filter(filter(s, p), q)`` == ``filter(s, p and q)``. Two
    filter stages combine into a single predicate; the visible
    result is byte-identical."""
    p = lambda x: x >= 0
    q = lambda x: x % 2 == 0

    two_stages = _collect(streams.filter_stream(streams.filter_stream(_stream_from(items), p), q))
    fused = _collect(streams.filter_stream(_stream_from(items), lambda x: p(x) and q(x)))
    assert two_stages == fused


@given(
    items=st.lists(st.integers(), max_size=30),
    n=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=40, deadline=None)
def test_take_bounded_by_input_and_n(items, n) -> None:
    """``take(s, n)`` yields at most min(n, len(items)) — never more."""
    out = _collect(streams.take(_stream_from(items), n))
    assert len(out) == min(max(0, n), len(items))
    assert out == items[: max(0, n)]


@given(items=st.lists(st.integers(), min_size=1, max_size=30))
@settings(max_examples=40, deadline=None)
def test_take_zero_yields_empty(items) -> None:
    """A zero-``n`` take short-circuits immediately."""
    out = _collect(streams.take(_stream_from(items), 0))
    assert out == []


@given(items=st.lists(st.integers(min_value=0, max_value=1000), max_size=30))
@settings(max_examples=40, deadline=None)
def test_scan_final_matches_reduce(items) -> None:
    """``scan(s, +, 0)`` emits every intermediate sum, so its last
    element equals ``reduce(+, items, 0)``. Callers that only want
    the final total can consume the last scan output."""
    from functools import reduce

    scanned = _collect(streams.scan(_stream_from(items), operator.add, 0))
    expected_final = reduce(operator.add, items, 0)
    if items:
        assert scanned[-1] == expected_final


@given(items=st.lists(st.integers(min_value=0, max_value=100), max_size=30))
@settings(max_examples=40, deadline=None)
def test_distinct_until_changed_keeps_first_of_each_run(items) -> None:
    """The output equals the input with consecutive duplicates
    collapsed — pattern-preserving, order-preserving."""
    out = _collect(streams.distinct_until_changed(_stream_from(items)))
    # Recompute the expected in a way that mirrors the operator's
    # semantics: keep an item only when it differs from the previous
    # emitted item.
    expected: list = []
    prev: object = object()  # sentinel
    for x in items:
        if x != prev:
            expected.append(x)
            prev = x
    assert out == expected


@given(
    items=st.lists(st.integers(), max_size=30),
    batch_size=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=40, deadline=None)
def test_buffer_reassembles_items_exactly(items, batch_size) -> None:
    """``sum(buffer(s, n), [])`` reconstructs the original stream —
    the operator only shapes items into batches, never drops or
    reorders."""
    batches = _collect(streams.buffer(_stream_from(items), batch_size))
    reconstructed = [x for batch in batches for x in batch]
    assert reconstructed == items
    # All but the tail batch must be exactly ``batch_size``.
    for batch in batches[:-1]:
        assert len(batch) == batch_size


def test_merge_early_break_from_infinite_sources_completes() -> None:
    """Adversarial cancellation: two infinite pump sources with a
    tight buffer, consumer breaks after N items. The merge generator
    must return cleanly — no dangling tasks that outlive the enclosing
    ``asyncio.run``. The stricter no-leak invariant is verified
    implicitly: ``asyncio.run`` shuts down cleanly only when every
    task reaches a terminal state; a stray pending pump would surface
    as a warning or hang."""

    async def _infinite():
        i = 0
        while True:
            yield i
            i += 1
            await asyncio.sleep(0)

    async def go():
        merged = streams.merge(_infinite(), _infinite(), buffer=2)
        count = 0
        async for _ in merged:
            count += 1
            if count == 3:
                break
        return count

    # Wall-clock ceiling: a leaked pump would either deadlock or
    # keep the loop alive; either surfaces here.
    async def with_timeout():
        return await asyncio.wait_for(go(), timeout=2.0)

    assert asyncio.run(with_timeout()) == 3


def test_streams_operators_preserve_element_type_signature() -> None:
    """Regression on the TypeVar parameterization: chaining
    ``distinct_until_changed(map_stream(events, lambda e: e.kind))``
    must still type-check as ``AsyncIterator[str]`` — the map's
    output type flows through the ``distinct_until_changed``
    signature. Verified structurally by running the pipeline; a
    regression that lost the TypeVars still passes here (mypy is
    the real gate) but a runtime bug in the chain would surface."""
    from dataclasses import dataclass

    @dataclass
    class Ev:
        kind: str

    events = [Ev("a"), Ev("a"), Ev("b"), Ev("a")]

    async def go():
        kinds = streams.map_stream(_stream_from(events), lambda e: e.kind)
        return await streams.collect(streams.distinct_until_changed(kinds))

    assert asyncio.run(go()) == ["a", "b", "a"]
