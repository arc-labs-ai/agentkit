"""The four public value types whose "frozen" payloads are now actually frozen.

`tests/kernel/test_frozen_payloads.py` covers the primitive (`FrozenDict` /
`FrozenList` / `deep_freeze`). This file covers WIRING it into the types that
carry mutable payloads across a trust boundary:

    Checkpoint.state / .metadata   a DURABLE record — written to a JSONB column
    SearchHit.metadata             fanned out across providers and rerankers
    FetchResponse.headers          cached and served to later callers
    Observation.payload            fanned out to several retaining sinks

`Checkpoint` is the headline. `frozen=True` stopped at the field reference, so
``cp.state = {}`` raised while ``cp.state["turn"] = 99`` silently rewrote a
snapshot that had already been committed. The bar for the fix was never "refuse
mutation" — it was "refuse mutation without breaking the durable path", which
is why half of this file is a persistence round trip rather than a mutation
assertion.
"""

import copy
import dataclasses
import json
import pickle

import pytest

from agentkit.adapters.checkpoint.in_memory import InMemoryCheckpointStore
from agentkit.adapters.checkpoint.postgres import _row_to_checkpoint
from agentkit.adapters.store import InMemoryStore
from agentkit.capabilities.checkpointer import StoreBackedCheckpointStore
from agentkit.kernel._frozen import FrozenDict, FrozenList
from agentkit.kernel.observation import Observation
from agentkit.kernel.ports import Checkpoint, CheckpointStatus, FetchResponse, SearchHit

# A state shaped like the real thing: nested dicts, a list of messages, and a
# list-inside-dict-inside-list, because every one of those is a separate way to
# reach past a shallow freeze.
STATE = {
    "turn": 3,
    "transcript": [{"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": [{"id": "c1"}]}],
    "scratchpad": {"notes": ["a", "b"]},
}


def _cp(state=None, metadata=None) -> Checkpoint:
    return Checkpoint(
        run_id="r1",
        version=1,
        state=STATE if state is None else state,
        created_at=1_700_000_000.0,
        status=CheckpointStatus.RUNNING,
        metadata={"producer": "react"} if metadata is None else metadata,
    )


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# ── the bug itself ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda cp: cp.state.__setitem__("turn", 99), id="state-setitem"),
        pytest.param(lambda cp: cp.state.pop("turn"), id="state-pop"),
        pytest.param(lambda cp: cp.state.update({"turn": 99}), id="state-update"),
        pytest.param(lambda cp: cp.state.setdefault("new", 1), id="state-setdefault"),
        pytest.param(lambda cp: cp.state.clear(), id="state-clear"),
        pytest.param(lambda cp: cp.metadata.__setitem__("producer", "evil"), id="metadata-setitem"),
        pytest.param(lambda cp: cp.metadata.update({"x": 1}), id="metadata-update"),
    ],
)
def test_a_persisted_checkpoint_cannot_be_rewritten_in_place(mutate) -> None:
    """THE bug. A Checkpoint is a durable record — the row is already in
    Postgres by the time a caller holds one — and every route below rewrote it
    in memory while ``cp.state = {}`` correctly raised. The in-memory record
    and the durable row then disagree with nothing to say so."""
    with pytest.raises(TypeError, match="frozen value"):
        mutate(_cp())


def test_the_field_reference_is_still_frozen_too() -> None:
    """The half that always worked — asserted so a regression that traded one
    for the other is visible."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        _cp().state = {}  # type: ignore[misc]


def test_nested_checkpoint_state_is_frozen_all_the_way_down() -> None:
    """A shallow freeze is the same bug one level lower and harder to see: a
    resumer reaching into ``cp.state["scratchpad"]["notes"]`` is doing exactly
    what a resumer does."""
    cp = _cp()
    with pytest.raises(TypeError):
        cp.state["scratchpad"]["notes"].append("c")
    with pytest.raises(TypeError):
        cp.state["transcript"][0]["role"] = "system"
    with pytest.raises(TypeError):
        cp.state["transcript"][1]["tool_calls"][0]["id"] = "c2"


def test_deeply_nested_dict_in_list_in_dict_is_frozen() -> None:
    """``cp.state["a"]["b"][0]["c"] = 1`` — the exact four-level shape, spelled
    out, because each container type has its own freeze branch."""
    cp = _cp(state={"a": {"b": [{"c": 0}]}})
    with pytest.raises(TypeError):
        cp.state["a"]["b"][0]["c"] = 1


@pytest.mark.parametrize(
    "build,mutate",
    [
        pytest.param(
            lambda: SearchHit("https://x", "t", "s", 0.5, {"domain": "x.com", "cites": [{"n": 1}]}),
            lambda h: h.metadata.__setitem__("domain", "evil.com"),
            id="searchhit-metadata",
        ),
        pytest.param(
            lambda: SearchHit("https://x", "t", "s", 0.5, {"domain": "x.com", "cites": [{"n": 1}]}),
            lambda h: h.metadata["cites"].append({"n": 2}),
            id="searchhit-metadata-nested",
        ),
        pytest.param(
            lambda: FetchResponse("https://x", 200, {"etag": "abc"}, "<html/>", "text/html", 1.0),
            lambda r: r.headers.__setitem__("etag", "def"),
            id="fetchresponse-headers",
        ),
        pytest.param(
            lambda: Observation(kind="result", payload={"status": "ok", "items": [1]}),
            lambda o: o.payload.__setitem__("status", "failed"),
            id="observation-payload",
        ),
        pytest.param(
            lambda: Observation(kind="result", payload={"status": "ok", "items": [1]}),
            lambda o: o.payload["items"].append(2),
            id="observation-payload-nested",
        ),
    ],
)
def test_the_other_three_types_refuse_payload_mutation(build, mutate) -> None:
    """Same shape, three more types. A SearchHit is unioned across providers and
    reranked; a FetchResponse is CACHED and served to later callers; an
    Observation is fanned out to an audit sink, a socket forwarder and a rollup
    buffer at once. In each case one consumer's in-place edit rewrote the record
    every other consumer was holding."""
    with pytest.raises(TypeError, match="frozen value"):
        mutate(build())


# ── the durable path: this is the constraint the fix had to clear ──────────


def test_checkpoint_state_json_dumps_byte_identically_to_a_plain_dict() -> None:
    """`PostgresCheckpointStore.save` does ``json.dumps(cp.state)`` into a
    ``$3::jsonb`` parameter. `FrozenDict` is a `dict` SUBCLASS precisely so
    this stays byte-for-byte what it was — a `MappingProxyType` raises
    ``TypeError: not JSON serializable`` right here, which is why it was
    rejected. Byte-identity (not just round-trip equality) is the assertion
    because JSONB ingestion, and any checksum over the wire payload, sees
    bytes."""
    cp = _cp()
    assert json.dumps(cp.state) == json.dumps(STATE)
    assert json.dumps(cp.state, sort_keys=True) == json.dumps(STATE, sort_keys=True)
    assert json.dumps(cp.metadata) == json.dumps({"producer": "react"})


def test_a_checkpoint_survives_the_store_backed_round_trip() -> None:
    """`StoreBackedCheckpointStore` passes ``cp.state`` through verbatim in
    ``_to_dict`` and rebuilds via ``_from_dict``. Exercised end to end —
    save → latest → at_version → list_versions — because a mutability fix that
    breaks a durable write is a bad trade."""
    cp = _cp()
    store = StoreBackedCheckpointStore(InMemoryStore())

    async def go():
        await store.save(cp)
        return await store.latest("r1"), await store.at_version("r1", 1), await store.list_versions("r1")

    latest, at_version, versions = _run(go())
    assert latest == cp
    assert at_version == cp
    assert versions == [1]
    assert latest.state == STATE
    assert json.dumps(latest.state) == json.dumps(STATE)


def test_a_checkpoint_survives_the_in_memory_checkpoint_store() -> None:
    """The store the offline tests actually run on. It holds the object by
    reference, so it is also the one where an in-place rewrite of ``cp.state``
    would have been invisible: the mutated object IS what ``latest()``
    returns."""
    cp = _cp()
    store = InMemoryCheckpointStore()

    async def go():
        await store.save(cp)
        return await store.latest("r1"), await store.at_version("r1", 1), await store.list_versions("r1")

    latest, at_version, versions = _run(go())
    assert latest == cp and at_version == cp and versions == [1]


def test_a_checkpoint_restored_from_storage_comes_back_frozen() -> None:
    """The other half of the guarantee, and the easier one to miss. A record
    that is frozen on the way IN and mutable on the way OUT is only half fixed
    — resume is precisely where a caller reaches for
    ``cp.state.pop("pending")``.

    Both real decoders rebuild through the ``Checkpoint`` constructor, so
    ``__post_init__`` re-freezes what ``json.loads`` handed back as a plain
    dict. Asserted on both rather than assumed from one."""
    cp = _cp()

    store = StoreBackedCheckpointStore(InMemoryStore())

    async def go():
        await store.save(cp)
        return await store.latest("r1")

    restored = _run(go())
    assert isinstance(restored.state, FrozenDict)
    with pytest.raises(TypeError):
        restored.state["turn"] = 99
    with pytest.raises(TypeError):
        restored.state["scratchpad"]["notes"].append("c")

    # The Postgres path, at the row decoder — asyncpg hands JSONB back as a
    # `str`, so `_row_to_checkpoint` `json.loads`es it into a PLAIN dict and
    # the freeze has to be re-applied by the constructor, not carried over.
    row = {
        "run_id": "r1",
        "version": 1,
        "state": json.dumps(cp.state),
        "created_at": 1_700_000_000.0,
        "status": "running",
        "metadata": json.dumps(cp.metadata),
    }
    decoded = _row_to_checkpoint(row)
    assert decoded == cp
    assert isinstance(decoded.state, FrozenDict) and isinstance(decoded.metadata, FrozenDict)
    with pytest.raises(TypeError):
        decoded.state["turn"] = 99


# ── positive controls: every consumer keeps working ────────────────────────


def test_a_frozen_payload_is_still_the_type_consumers_branch_on() -> None:
    """`isinstance(x, dict)` / `isinstance(x, list)` guards are everywhere in
    the serialisers. A `MappingProxyType` fails both and silently takes the
    else-branch."""
    cp = _cp()
    assert isinstance(cp.state, dict)
    assert isinstance(cp.state["transcript"], list)
    assert isinstance(cp.metadata, dict)
    assert isinstance(SearchHit("u", "t", "s", None, {"a": 1}).metadata, dict)
    assert isinstance(FetchResponse("u", 200, {"a": "b"}, "", "text/html", 1.0).headers, dict)
    assert isinstance(Observation(kind="result", payload={"a": 1}).payload, dict)
    assert isinstance(Observation(kind="result", payload=[1, 2]).payload, list)


def test_a_frozen_payload_still_reads_like_a_dict() -> None:
    """Indexing, iteration, len, membership, `.get`, `.items`, and `dict(...)`
    — the read surface a consumer already depends on."""
    cp = _cp()
    assert cp.state["turn"] == 3
    assert cp.state.get("missing") is None
    assert len(cp.state) == 3
    assert "scratchpad" in cp.state
    assert sorted(cp.state) == ["scratchpad", "transcript", "turn"]
    assert dict(cp.state) == STATE
    assert dict(cp.state.items())["turn"] == 3
    assert len(cp.state["transcript"]) == 2
    assert cp.state["transcript"][0]["role"] == "user"
    assert [m["role"] for m in cp.state["transcript"]] == ["user", "assistant"]


def test_a_frozen_payload_still_compares_equal_to_a_plain_one() -> None:
    """Equality against plain dicts is load-bearing — the store round-trip
    assertions above, and most of the existing suite, compare a restored
    checkpoint against a literal."""
    cp = _cp()
    assert cp.state == STATE
    assert STATE == cp.state
    assert cp == Checkpoint("r1", 1, dict(STATE), 1_700_000_000.0, CheckpointStatus.RUNNING, {"producer": "react"})


def test_a_frozen_payload_still_survives_asdict_deepcopy_and_pickle() -> None:
    """`Checkpointer.snapshot` deep-copies state on every save, `AgentResult`
    round-trips through `asdict`, and pickle is how a checkpoint crosses a
    process boundary. `MappingProxyType` fails all three."""
    cp = _cp()
    assert json.loads(json.dumps(dataclasses.asdict(cp)))["state"] == STATE
    assert copy.deepcopy(cp) == cp
    assert pickle.loads(pickle.dumps(cp)) == cp
    assert copy.copy(cp) == cp
    # ...and the copies are still frozen, or the guarantee leaks through any copy.
    with pytest.raises(TypeError):
        copy.deepcopy(cp).state["turn"] = 99
    with pytest.raises(TypeError):
        pickle.loads(pickle.dumps(cp)).state["turn"] = 99
    obs = Observation(kind="result", payload={"a": [1]})
    assert pickle.loads(pickle.dumps(obs)) == obs
    with pytest.raises(TypeError):
        copy.deepcopy(obs).payload["a"].append(2)


def test_equality_and_hash_are_untouched() -> None:
    """REGRESSION GUARD. `__eq__`/`__hash__` were fixed in 7e0d4cd to exclude
    payloads; freezing must not have moved either. Two checkpoints for one
    ``(run_id, version)`` that differ in state still hash equal and still
    compare unequal — a collision `__eq__` resolves, which is the whole design."""
    a = _cp(state={"turn": 1})
    b = _cp(state={"turn": 2})
    assert hash(a) == hash(b) == hash(("r1", 1))
    assert a != b
    assert len({a, b}) == 2
    assert hash(SearchHit("u", "t", "s", 0.5, {"a": 1})) == hash(("u", "t"))
    # The stream key was ``(run_id, agent, seq, ts, kind)`` when this line was
    # written. `seq`/`ts` are gone — nothing in the package ever set them, so
    # the two slots this literal spelled as ``0, 0.0`` were constants on every
    # record a real run produced. The key is now ``(run_id, agent, kind)``.
    assert hash(Observation(kind="progress", payload={"a": 1})) == hash(("", "", "progress"))
    assert hash(FetchResponse("u", 200, {"a": "b"}, "x", "text/html", 1.0)) == hash(("u", 200, "text/html", 1.0))


# ── edge cases ─────────────────────────────────────────────────────────────


def test_empty_payloads_freeze_cleanly() -> None:
    """The default-constructed case — ``metadata`` defaults to ``{}`` on two of
    these types, so this is the MOST common instance, not an edge one."""
    cp = _cp(state={}, metadata={})
    assert cp.state == {} and cp.metadata == {}
    assert isinstance(cp.state, FrozenDict)
    with pytest.raises(TypeError):
        cp.state["sneak"] = 1
    hit = SearchHit("https://x", "t", "s")  # metadata via default_factory
    assert isinstance(hit.metadata, FrozenDict)
    with pytest.raises(TypeError):
        hit.metadata["sneak"] = 1


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(None, id="none"),
        pytest.param("tick", id="str"),
        pytest.param(42, id="int"),
        pytest.param(3.5, id="float"),
        pytest.param(True, id="bool"),
        pytest.param((1, 2), id="tuple"),
    ],
)
def test_a_non_container_observation_payload_passes_straight_through(payload) -> None:
    """``Observation.payload`` is annotated ``Any`` and genuinely is: `ctx.emit`
    is documented with a dict but callers pass bare strings and ints, and the
    default is `None`. `deep_freeze` returns non-containers untouched, so these
    must come back as the IDENTICAL object — not a copy, not a wrapper.

    This is the direction a freeze most easily breaks by accident: wrapping or
    copying a scalar would change ``obs.payload is what_i_passed`` and put a
    per-emit copy on the hot path for payloads that never needed one."""
    obs = Observation(kind="progress", payload=payload)
    assert obs.payload is payload
    assert type(obs.payload) is type(payload)


def test_a_container_observation_payload_is_frozen() -> None:
    """The other direction of the same behaviour, asserted next to it so the
    pair reads as one decision rather than two."""
    d = Observation(kind="result", payload={"a": 1})
    assert isinstance(d.payload, FrozenDict)
    with pytest.raises(TypeError):
        d.payload["a"] = 2
    lst = Observation(kind="result", payload=[{"a": 1}])
    assert isinstance(lst.payload, FrozenList)
    with pytest.raises(TypeError):
        lst.payload.append(2)
    with pytest.raises(TypeError):
        lst.payload[0]["a"] = 2


def test_a_caller_cannot_edit_a_payload_after_handing_it_over() -> None:
    """`deep_freeze` COPIES, which is what un-aliases the caller's object.
    Freezing in place would leave the producer holding a live handle to the
    durable record — the checkpointer builds state in a scratch dict and keeps
    using it, so this is the realistic sequence, not a contrived one."""
    scratch = {"turn": 1, "notes": ["a"]}
    cp = _cp(state=scratch)
    scratch["turn"] = 99
    scratch["notes"].append("b")
    assert cp.state == {"turn": 1, "notes": ["a"]}, "the checkpoint must not track the producer's scratch dict"

    hdrs = {"etag": "abc"}
    resp = FetchResponse("https://x", 200, hdrs, "<html/>", "text/html", 1.0)
    hdrs["etag"] = "def"
    assert resp.headers == {"etag": "abc"}


def test_fetchresponse_body_is_left_alone() -> None:
    """``body`` is a `str` — already immutable, and unbounded (this is the type
    a crawler holds thousands of). It is deliberately not walked; asserting
    identity keeps a future "freeze everything" edit from putting an O(page)
    copy on every fetch."""
    body = "<html>" + "x" * 10_000 + "</html>"
    resp = FetchResponse("https://x", 200, {"a": "b"}, body, "text/html", 1.0)
    assert resp.body is body


def test_freezing_is_idempotent_across_types() -> None:
    """A payload handed from one frozen value to another must not pay a second
    O(payload) walk — a resumed checkpoint's state going straight into an
    Observation is exactly that hand-off."""
    cp = _cp()
    obs = Observation(kind="result", payload=cp.state)
    assert obs.payload is cp.state
    assert Checkpoint("r2", 1, cp.state, 1.0, CheckpointStatus.DONE).state is cp.state
