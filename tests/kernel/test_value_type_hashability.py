"""The four kernel value types that were frozen in name only: `Checkpoint`,
`SearchHit`, `FetchResponse` (ports.py) and `Observation` (observation.py).

Each is a `frozen=True` dataclass carrying a mutable payload, so the generated
all-fields `__hash__` inherited a `dict` and raised. Measured before the fix::

    hash(Checkpoint("r", 1, {"turn": 3}, 1.7e9, "running"))  TypeError: unhashable type: 'dict'
    hash(SearchHit("https://x", "t", "s", 0.5, {"d": "x"}))  TypeError: unhashable type: 'dict'
    hash(FetchResponse("https://x", 200, {...}, "hi", ...))  TypeError: unhashable type: 'dict'
    hash(Observation(kind="result", payload={"k": "v"}))     TypeError: unhashable type: 'dict'

`deepcopy` and `pickle` already worked on all four, so `__hash__` was the only
thing missing — which is why the fix adds a method and changes nothing else.

`Observation` is the one worth staring at: `payload: Any` hashes fine when a
test puts a string there and stops hashing the moment it holds the dict that
`ctx.emit(..., payload={...})` actually passes. It was found by the value-type
ratchet rather than by a caller, because a caller only finds it at the one call
site that happens to hash one.

Every fix hashes an identity SUBSET and leaves `__eq__` untouched. The tests
below check BOTH halves of what that means: the hash exists (bug fixed), and
two records that share the subset but differ in payload still compare unequal
and both survive in a `set` (equality unchanged). The `_hash_is_o1_*` tests
prove payload-independence STRUCTURALLY — `hash(small) == hash(huge)` — rather
than by timing, so they cannot go flaky on a loaded CI box.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import pickle
from typing import Any

import pytest

from agentkit.adapters.store import InMemoryStore
from agentkit.capabilities.checkpointer.persistence import StoreBackedCheckpointStore
from agentkit.kernel.observation import Observation, TraceContext
from agentkit.kernel.ports import Checkpoint, CheckpointStatus, FetchResponse, SearchHit


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# A payload with every shape a real one has: nesting, lists, None, mixed types.
# Flat scalars would let a value-inclusive hash pass and hide the bug — the same
# vacuous-pass trap the ratchet's representative-instance rule exists to defeat.
DEEP: dict[str, Any] = {
    "turn": 3,
    "transcript": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "args": {"q": ["a", "b"]}}]},
    ],
    "scratchpad": {"plan": {"steps": ["a", {"b": [1, 2, {"c": None}]}], "done": False}},
}


def _cp(state: Any = None, *, run_id: str = "run-1", version: int = 1, **kw: Any) -> Checkpoint:
    return Checkpoint(
        run_id=run_id,
        version=version,
        state={"turn": 3} if state is None else state,
        created_at=1_700_000_000.0,
        status=CheckpointStatus.RUNNING,
        **kw,
    )


def _hit(**kw: Any) -> SearchHit:
    base: dict[str, Any] = {
        "url": "https://example.com/a",
        "title": "A",
        "snippet": "sa",
        "score": 0.9,
        "metadata": {"domain": "example.com"},
    }
    return SearchHit(**{**base, **kw})


def _resp(**kw: Any) -> FetchResponse:
    base: dict[str, Any] = {
        "url": "https://example.com",
        "status": 200,
        "headers": {"content-type": "text/html"},
        "body": "<html>hi</html>",
        "content_type": "text/html",
        "fetched_at": 1_700_000_000.0,
    }
    return FetchResponse(**{**base, **kw})


def _obs(**kw: Any) -> Observation:
    base: dict[str, Any] = {
        "kind": "result",
        "seq": 7,
        "ts": 1_700_000_000.0,
        "agent": "writer",
        "render": "wrote intro",
        "run_id": "run-1",
        "payload": {"words": 120},
    }
    return Observation(**{**base, **kw})


# ── the bug: a representative instance must be hashable ────────────

# `FetchResponse` is absent from this sweep on purpose: its mutable field is
# `headers: dict[str, str]`, a flat string map by contract, so feeding it deep
# JSON would test a shape it never holds. It gets its own cases below, where the
# interesting axis is the unbounded `body` rather than the payload's depth.
ALL = pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda p: _cp(p), id="Checkpoint.state"),
        pytest.param(lambda p: _cp(metadata=p), id="Checkpoint.metadata"),
        pytest.param(lambda p: _hit(metadata=p), id="SearchHit.metadata"),
        pytest.param(lambda p: _obs(payload=p), id="Observation.payload"),
    ],
)


@ALL
def test_a_representative_payload_does_not_cost_the_type_its_hash(factory: Any) -> None:
    """The bug, verbatim. A dict payload is the NORMAL case for all four types —
    `Checkpoint.state` is application JSON, `SearchHit.metadata` is provider
    JSON, `FetchResponse.headers` is HTTP, and `Observation.payload` is what
    `ctx.emit` documents — so before the fix every real instance raised
    `TypeError: unhashable type: 'dict'` here."""
    assert isinstance(hash(factory({"k": "v"})), int)


@ALL
def test_deeply_nested_json_payloads_are_hashable(factory: Any) -> None:
    """Nesting is where a value-inclusive hash would fail even if the top level
    happened to be flat: a list or dict anywhere below the surface is
    unhashable. The hash must not read the payload at ANY depth."""
    assert isinstance(hash(factory(DEEP)), int)


@ALL
def test_empty_and_none_payloads_are_hashable(factory: Any) -> None:
    """The degenerate ends. `{}` is the default-constructed `metadata` on both
    `Checkpoint` and `SearchHit`, and `None` is the declared default of
    `Observation.payload` — an empty dict is still a dict, so `{}` raised
    before the fix exactly like a full one did."""
    assert isinstance(hash(factory({})), int)
    assert isinstance(hash(factory(None)), int)


@ALL
def test_a_payload_holding_an_unhashable_object_is_still_hashable(factory: Any) -> None:
    """A payload is arbitrary application data, and durable state has held
    sets and lists since before it was durable. A type that is hashable only
    when the caller stored scalars is not hashable, it is a trap."""

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    assert isinstance(hash(factory({"list": [1, 2], "set": {1, 2}, "obj": Unhashable()})), int)


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="empty"),
        pytest.param({"content-type": "text/html"}, id="one"),
        pytest.param({f"x-h{i}": str(i) for i in range(200)}, id="two-hundred"),
    ],
)
def test_fetch_response_is_hashable_whatever_the_headers(headers: dict[str, str]) -> None:
    """`headers` is a required field with no default, so EVERY `FetchResponse`
    ever constructed carried a dict and every one of them raised. An empty map
    raised exactly like a 200-header one — the bug is the type, not the size."""
    assert isinstance(hash(_resp(headers=headers)), int)


def test_all_four_survive_a_set_together() -> None:
    """The end-to-end shape of the complaint: putting these value types in a
    `set` is what a caller does with a value type, and every one of them raised
    on the way in."""
    assert len({_cp(DEEP), _hit(metadata=DEEP), _resp(), _obs(payload=DEEP)}) == 4


# ── payload-independence, proven structurally (never by timing) ────


def test_checkpoint_hash_is_o1_in_state_size() -> None:
    """Two checkpoints differing ONLY in state size hash identically, which is
    the structural proof that state is never read. Stated as an equality rather
    than a stopwatch so the test cannot go flaky: a timing assertion on a
    shared CI box measures the box, not the code.

    (For the record, the timing agrees: 0.22 µs for the 1-key state and 0.20 µs
    for the 100_000-key one.)"""
    huge = {f"k{i}": i for i in range(100_000)}
    assert hash(_cp({"turn": 3})) == hash(_cp(huge))
    assert _cp({"turn": 3}) != _cp(huge)  # …and they are still different records


def test_observation_hash_is_o1_in_payload_size() -> None:
    """Same proof for the hot path: an observation is fanned out to every
    attached observer on every emit, and a `result` payload is a whole agent
    output."""
    huge = {f"k{i}": i for i in range(100_000)}
    assert hash(_obs(payload={"words": 120})) == hash(_obs(payload=huge))


def test_fetch_response_hash_is_o1_in_body_size() -> None:
    """`body` is a `str`, so it COULD have gone in the hash — and that is the
    trap. `hash(str)` is O(len) on first hash, so folding a 4 MiB page in would
    cost 2.3 ms per set insertion against 0.23 µs here. Proven structurally:
    a 12-byte body and a 4 MiB body hash the same."""
    assert hash(_resp(body="hi")) == hash(_resp(body="x" * 4 * 1024 * 1024))


def test_search_hit_hash_ignores_query_dependent_fields() -> None:
    """`score` and `snippet` are properties of the QUERY, not of the result:
    one document ranked 1st here and 4th there, with a different extracted
    snippet each time, is one document and belongs in one bucket — which is
    what makes cross-provider dedup work."""
    assert hash(_hit(score=0.9, snippet="sa")) == hash(_hit(score=0.1, snippet="different"))


# ── __eq__ is untouched: collide in the bucket, stay unequal ───────
#
# POSITIVE CONTROLS — these pass BOTH before and after the fix. Hashing a
# subset is sound only because equality still compares everything; if a fix had
# reached for `eq=False` or narrowed `__eq__` to match the hash, these are the
# tests that would catch it.


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(_cp({"turn": 1}), _cp({"turn": 2}), id="Checkpoint"),
        pytest.param(_hit(metadata={"d": "a"}), _hit(metadata={"d": "b"}), id="SearchHit"),
        pytest.param(_resp(body="one"), _resp(body="two"), id="FetchResponse"),
        pytest.param(_obs(payload={"n": 1}), _obs(payload={"n": 2}), id="Observation"),
    ],
)
def test_two_records_differing_only_in_payload_are_still_unequal(a: Any, b: Any) -> None:
    """Two instances that differ ONLY in the excluded payload must NOT become
    equal. The hash invariant requires equal objects to hash equally; it never
    requires unequal objects to hash differently, so this is the half that has
    to keep holding."""
    assert a != b
    assert a == copy.deepcopy(a)  # …and equality still works in the normal direction


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(_cp({"turn": 1}), _cp({"turn": 2}), id="Checkpoint"),
        pytest.param(_hit(metadata={"d": "a"}), _hit(metadata={"d": "b"}), id="SearchHit"),
        pytest.param(_resp(body="one"), _resp(body="two"), id="FetchResponse"),
        pytest.param(_obs(payload={"n": 1}), _obs(payload={"n": 2}), id="Observation"),
    ],
)
def test_payload_only_differences_collide_in_one_bucket_and_both_survive(a: Any, b: Any) -> None:
    """The soundness argument, executed. These share the hashed subset, so they
    land in the SAME bucket — and `__eq__` separates them there, so a `set`
    keeps both. That is what a bucket is for; a `set` consults `__eq__`, the
    hash only chooses where to look."""
    assert hash(a) == hash(b) and a != b
    assert len({a, b}) == 2
    # A dict keyed on either one must not read back the other's value.
    d = {a: "a", b: "b"}
    assert d[a] == "a" and d[b] == "b"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(_cp(version=1), _cp(version=2), id="Checkpoint.version"),
        pytest.param(_cp(run_id="r1"), _cp(run_id="r2"), id="Checkpoint.run_id"),
        pytest.param(_hit(url="https://a"), _hit(url="https://b"), id="SearchHit.url"),
        pytest.param(_resp(url="https://a"), _resp(url="https://b"), id="FetchResponse.url"),
        pytest.param(_obs(seq=1), _obs(seq=2), id="Observation.seq"),
        pytest.param(_obs(run_id="r1"), _obs(run_id="r2"), id="Observation.run_id"),
    ],
)
def test_the_hashed_subset_still_discriminates(a: Any, b: Any) -> None:
    """A hash that ignored everything would satisfy the invariant and be
    useless. Records differing in the IDENTITY fields must land in different
    buckets, which is what makes these usable as dict keys at all."""
    assert hash(a) != hash(b)
    assert len({a, b}) == 2


# ── construction / field access / copy semantics are unchanged ─────
#
# POSITIVE CONTROLS — pass before and after. The payloads are deliberately NOT
# frozen into `MappingProxyType`: `Checkpoint.state` is `json.dumps`-ed into a
# JSONB column by the Postgres adapter and passed through verbatim by
# `StoreBackedCheckpointStore`, and a mappingproxy is neither JSON-serialisable
# nor picklable. This commit is about hashability only.


def test_payload_fields_are_still_dicts_the_persistence_layer_accepts() -> None:
    """The shape every existing reader depends on. If a future fix reaches for
    `MappingProxyType` to make these hashable, this is the test that says the
    persistence layer noticed.

    Loosened from ``type(payload) is dict`` to ``isinstance`` when the payloads
    became `FrozenDict` (see `test_frozen_value_payloads.py`). The exact-type
    spelling was over-tight for what this guard is actually protecting: nothing
    downstream branches on the concrete class, they branch on `isinstance` and
    then serialise. `FrozenDict` is a `dict` SUBCLASS chosen precisely so both
    keep working — which is what the added `json.dumps` line asserts directly,
    rather than by proxy through a type check. `MappingProxyType` still fails
    both lines, so the guard this test exists for is intact."""
    for payload in (_cp(DEEP).state, _cp(DEEP).metadata, _hit().metadata, _resp().headers):
        assert isinstance(payload, dict)
        assert json.loads(json.dumps(payload)) == payload
    assert _obs(payload=DEEP).payload == DEEP


def test_field_access_and_defaults_are_unchanged() -> None:
    cp = _cp(DEEP)
    assert cp.run_id == "run-1" and cp.version == 1 and cp.state["turn"] == 3
    assert cp.status is CheckpointStatus.RUNNING and cp.metadata == {}
    hit = _hit()
    assert hit.url == "https://example.com/a" and hit.score == 0.9
    resp = _resp()
    assert resp.status == 200 and resp.headers["content-type"] == "text/html"
    obs = _obs(trace_context=TraceContext(trace_id="t" * 32, span_id="s" * 16))
    assert obs.kind == "result" and obs.render == "wrote intro"
    assert obs.trace_context is not None and obs.trace_context.span_id == "s" * 16
    assert Observation(kind="progress").payload is None  # declared default survives


def test_the_frozen_shell_still_rejects_field_assignment() -> None:
    """`frozen=True` is untouched — defining `__hash__` does not re-open the
    shell. (The payload BEHIND the field is still rewritable; that is the other
    half of this bug shape and is deliberately out of scope here.)"""
    cp = _cp()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cp.version = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "inst",
    [
        pytest.param(_cp(DEEP), id="Checkpoint"),
        pytest.param(_hit(metadata=DEEP), id="SearchHit"),
        pytest.param(_resp(), id="FetchResponse"),
        pytest.param(_obs(payload=DEEP), id="Observation"),
    ],
)
def test_deepcopy_and_pickle_still_round_trip(inst: Any) -> None:
    """POSITIVE CONTROL, passing before and after: these already worked —
    `__hash__` was the ONLY thing missing — so this is a regression guard, not
    a fix. It is what rules out the `MappingProxyType` route, which buys
    hashability by breaking exactly this. Not academic either:
    `Checkpointer.snapshot` deep-copies state at the durable seam, and these
    types cross process boundaries, which means pickle."""
    assert copy.deepcopy(inst) == inst
    assert pickle.loads(pickle.dumps(inst)) == inst


@pytest.mark.parametrize(
    "inst",
    [
        pytest.param(_cp(DEEP), id="Checkpoint"),
        pytest.param(_hit(metadata=DEEP), id="SearchHit"),
        pytest.param(_resp(), id="FetchResponse"),
        pytest.param(_obs(payload=DEEP), id="Observation"),
    ],
)
def test_a_copy_lands_in_the_same_bucket_as_its_original(inst: Any) -> None:
    """The invariant itself: equal objects hash equally. A record that has been
    round-tripped through a queue or a store is `==` to the original, so it MUST
    hash to the same bucket or every dict built from a restored record silently
    misses."""
    for clone in (copy.deepcopy(inst), pickle.loads(pickle.dumps(inst))):
        assert clone == inst and hash(clone) == hash(inst)


# ── the durable seam: persistence is byte-for-byte unchanged ───────


def test_checkpoint_survives_the_store_backed_round_trip_unchanged() -> None:
    """POSITIVE CONTROL on the path that made freezing the payload a non-option:
    `StoreBackedCheckpointStore` writes `cp.state` through verbatim and rebuilds
    a `Checkpoint` from it. Same object graph out as in — this passes before and
    after the fix, which is the whole claim: persistence is untouched."""
    cp = _cp(DEEP, metadata={"who": "writer"})
    port = StoreBackedCheckpointStore(InMemoryStore())
    _run(port.save(cp))
    back = _run(port.latest("run-1"))
    assert back == cp
    # ``isinstance``, not ``type(...) is dict``: the state comes back as a
    # `FrozenDict` since the payloads were frozen, and a checkpoint that came
    # back MUTABLE would be the actual bug (see `test_frozen_value_payloads.py`
    # — a record frozen on the way in and loose on the way out is half fixed).
    # What this control cares about is that persistence is untouched, so it
    # asserts that directly: same value out as in, still JSON-serialisable.
    assert back is not None and back.state == DEEP and isinstance(back.state, dict)
    assert json.dumps(back.state) == json.dumps(DEEP)
    assert back.metadata == {"who": "writer"} and back.status == CheckpointStatus.RUNNING
    assert _run(port.list_versions("run-1")) == [1]
    assert _run(port.at_version("run-1", 1)) == cp


def test_a_restored_checkpoint_hashes_to_the_durable_key_it_was_stored_under() -> None:
    """The other half, and the reason `(run_id, version)` is the right subset:
    it is the `PRIMARY KEY (run_id, version)` of the `agentkit_checkpoints`
    table, so a checkpoint read back out of a store lands in the same bucket as
    the one that was written — no matter what happened to `state` in between."""
    cp = _cp(DEEP, metadata={"who": "writer"})
    port = StoreBackedCheckpointStore(InMemoryStore())
    _run(port.save(cp))
    back = _run(port.latest("run-1"))
    assert back is not None and hash(back) == hash(cp)


def test_checkpoint_state_is_still_json_serialisable_byte_for_byte() -> None:
    """The Postgres adapter does `json.dumps(cp.state)` into a JSONB column and
    `json.loads` on the way back. A `MappingProxyType` would have raised
    `TypeError: Object of type mappingproxy is not JSON serializable` right
    here — this test is why the fix is a `__hash__` and nothing else."""
    cp = _cp(DEEP, metadata={"who": "writer"})
    assert json.dumps(cp.state) == json.dumps(DEEP)
    assert json.loads(json.dumps(cp.state)) == DEEP
    assert json.dumps(cp.metadata) == json.dumps({"who": "writer"})


def test_dataclasses_asdict_still_works_on_all_four() -> None:
    """`asdict` recurses into the payload and would trip over any proxy or
    custom mapping put there. Callers serialise these types this way; the
    output must stay plain, JSON-safe dicts."""
    for inst in (_cp(DEEP), _hit(metadata=DEEP), _resp(), _obs(payload=DEEP)):
        d = dataclasses.asdict(inst)
        assert type(d) is dict
        json.dumps(d, default=str)  # no un-serialisable leaf snuck in
