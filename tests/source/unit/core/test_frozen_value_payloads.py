"""`Checkpoint` — the DURABLE value type whose "frozen" payload is actually frozen.

`test_frozen_payloads.py` covers the mechanism (`FrozenDict` / `FrozenList` /
`deep_freeze`) and `test_frozen_request_payloads.py` covers the request-path
types the kernel carries through a turn. This file covers the one value that has
already been WRITTEN DOWN when a caller reaches for it.

The bug being pinned, as it actually read::

    cp = Checkpoint(...)        # a durable record, committed to a JSONB column
    cp.state = {}               # FrozenInstanceError, as intended
    cp.state["turn"] = 99       # ...but this rewrote the snapshot, silently

After that line the in-memory record and the durable row disagree and nothing
says so: a real backend's ``latest()`` returns the row, an in-memory store's
returns the mutated object, and a resumed run replays a state that was never
persisted. ``frozen=True`` stopped at the field reference, so ``__post_init__``
has to close it.

The freeze had a constraint to clear before it could ship, and that is the
second half of this file: a checkpoint is ``json.dumps``'d into a JSONB column,
walked by ``dataclasses.asdict``, deep-copied by a checkpoint store on every
snapshot and compared against plain-dict records. ``MappingProxyType`` — the
obvious mechanism — breaks all four, which is why the payload is a ``dict``
SUBCLASS instead. A test that only proved "mutation refused" would have been
satisfied by the mechanism that breaks the durable path.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from typing import Any

import pytest

from agentkit.core.contracts.ports import Checkpoint, CheckpointStatus
from agentkit.core.serialization._immutable import FrozenDict, FrozenList, thaw

_STATE: dict[str, Any] = {
    "turn": 3,
    "history": [{"role": "user", "content": "hi"}],
    "scratchpad": {"notes": ["first"], "flags": {"gated": True}},
}


def _checkpoint(state: dict[str, Any] | None = None, **overrides: Any) -> Checkpoint:
    kwargs: dict[str, Any] = {
        "run_id": "run-1",
        "version": 1,
        "state": copy.deepcopy(_STATE) if state is None else state,
        "created_at": 1_700_000_000.0,
        "status": CheckpointStatus.RUNNING,
        "metadata": {"producer": "react", "labels": ["resumable"]},
    }
    kwargs.update(overrides)
    return Checkpoint(**kwargs)


# ── the mutability the record was losing ───────────────────────────────────


def test_a_committed_snapshot_cannot_be_rewritten_through_its_state_field() -> None:
    """The headline. ``cp.state = {}`` always raised; ``cp.state["turn"] = 99``
    did not, and that is the one a caller actually writes — the resume path is
    precisely where someone reaches for ``cp.state.pop("pending")`` after the
    row has been committed. Every mutation ROUTE has to be closed, not just
    ``[k] = v``: ``update``, ``setdefault`` and ``|=`` are what a caller tries
    next, and one open route is not a frozen record."""
    cp = _checkpoint()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cp.state = {}  # type: ignore[misc]
    for mutate in (
        lambda s: s.__setitem__("turn", 99),
        lambda s: s.update({"turn": 99}),
        lambda s: s.setdefault("pending", []),
        lambda s: s.pop("turn"),
        lambda s: s.__delitem__("turn"),
        lambda s: s.clear(),
    ):
        with pytest.raises(TypeError, match="frozen value"):
            mutate(cp.state)
    assert cp.state["turn"] == 3


def test_the_freeze_reaches_the_bottom_of_a_run_state() -> None:
    """A shallow freeze is the same bug one level down and harder to see: run
    state nests by nature — a transcript of messages, each with tool calls, plus
    a scratchpad of lists — so ``cp.state["scratchpad"]["notes"].append(...)``
    is the realistic write, not a top-level assignment. Freezing only the outer
    dict would make the record LOOK protected in every assertion anyone writes
    while leaving the actual rewrite path open."""
    cp = _checkpoint()
    assert isinstance(cp.state, FrozenDict)
    assert isinstance(cp.state["history"], FrozenList)
    assert isinstance(cp.state["scratchpad"]["flags"], FrozenDict)
    with pytest.raises(TypeError, match="frozen value"):
        cp.state["scratchpad"]["notes"].append("second")
    with pytest.raises(TypeError, match="frozen value"):
        cp.state["history"][0]["content"] = "rewritten"
    with pytest.raises(TypeError, match="frozen value"):
        cp.state["scratchpad"]["flags"]["gated"] = False


def test_a_snapshot_does_not_track_the_producers_own_dict() -> None:
    """POSITIVE CONTROL for copy-on-freeze, and the reason freezing in place
    would not have been enough. The producer holds the live working state and
    keeps editing it after the snapshot is taken — that is what a working state
    is for. If the checkpoint aliased it, the "snapshot" would keep changing
    after the row was written, which is the same disagreement between record and
    row as the in-place rewrite, arriving from the other side."""
    live = {"turn": 3, "notes": ["first"]}
    cp = _checkpoint(state=live)
    live["turn"] = 99
    live["notes"].append("second")
    assert cp.state == {"turn": 3, "notes": ["first"]}, "the snapshot followed the producer's edits"


def test_the_records_metadata_is_frozen_on_the_same_terms_as_its_state() -> None:
    """``metadata`` is the other payload on the record and is read by the same
    consumers — auto-resume filters on it, audit renders it. It is smaller, which
    is exactly why it gets forgotten; a record that is half frozen offers no
    guarantee a caller can rely on, because the caller cannot see which half."""
    cp = _checkpoint()
    assert isinstance(cp.metadata, FrozenDict)
    with pytest.raises(TypeError, match="frozen value"):
        cp.metadata["producer"] = "rewritten"
    with pytest.raises(TypeError, match="frozen value"):
        cp.metadata["labels"].append("terminal")


def test_a_default_constructed_record_is_frozen_too() -> None:
    """``metadata`` defaults to an empty dict via ``default_factory``, so the
    default-constructed record is the one most instances actually take — and a
    freeze applied only to explicitly-passed payloads would protect the rare
    case and leave the common one open."""
    cp = Checkpoint("run-1", 1, {}, 1_700_000_000.0, CheckpointStatus.DONE)
    assert isinstance(cp.state, FrozenDict) and isinstance(cp.metadata, FrozenDict)
    with pytest.raises(TypeError, match="frozen value"):
        cp.metadata["sneaked"] = 1


def test_a_record_rebuilt_from_a_stored_row_comes_back_frozen() -> None:
    """The half that matters most, and the half a constructor-side freeze is the
    only way to get. Every decoder — the Postgres row decoder, a store-backed
    ``_from_dict`` — rebuilds through this same constructor with a plain
    ``json.loads`` dict, so the record is frozen on the way OUT of storage as
    well as in. A record that is frozen on save and mutable on resume is only
    half fixed: resume is where the mutation is attempted."""
    row = json.dumps({"state": _STATE, "metadata": {"producer": "react"}})
    decoded = json.loads(row)
    restored = _checkpoint(state=decoded["state"], metadata=decoded["metadata"])
    assert isinstance(restored.state, FrozenDict)
    with pytest.raises(TypeError, match="frozen value"):
        restored.state["history"].append({"role": "assistant"})


def test_replacing_a_field_re_freezes_rather_than_leaking_a_writable_payload() -> None:
    """``dataclasses.replace`` is the sanctioned way to change a frozen value —
    it is what the error message tells callers to use — and it re-enters
    ``__post_init__``. If the freeze lived anywhere else (a factory, a decoder),
    the one path the framework recommends would hand back an unprotected record,
    which is the worst possible place for the guarantee to lapse."""
    evolved = dataclasses.replace(_checkpoint(), version=2, state={"turn": 4, "notes": ["new"]})
    assert evolved.version == 2
    with pytest.raises(TypeError, match="frozen value"):
        evolved.state["notes"].append("x")


# ── the four things the durable path still has to do ───────────────────────


def test_a_frozen_state_serialises_byte_identically_to_the_plain_dict() -> None:
    """The constraint that ruled ``MappingProxyType`` out. ``Checkpoint.state``
    is ``json.dumps``'d straight into a JSONB column, and a proxy raises
    ``TypeError: Object of type mappingproxy is not JSON serializable`` there —
    trading a mutability bug for a durable-write outage. BYTE-identical, not
    merely equivalent, because the encoded row is what gets compared, digested
    and diffed downstream."""
    cp = _checkpoint()
    assert json.dumps(cp.state) == json.dumps(_STATE)
    assert json.loads(json.dumps(cp.state)) == _STATE


def test_a_frozen_record_still_walks_through_dataclasses_asdict() -> None:
    """``asdict`` is the other serialisation route these records travel, and it
    takes its mapping branch on ``isinstance(x, dict)`` — which a proxy fails,
    silently falling through to deep-copying the object instead. A ``dict``
    subclass keeps the branch."""
    cp = _checkpoint()
    walked = dataclasses.asdict(cp)
    assert walked["state"] == _STATE
    assert json.dumps(walked["state"]) == json.dumps(_STATE)


def test_a_checkpoint_store_can_still_deepcopy_and_pickle_a_frozen_record() -> None:
    """``CheckpointPort.snapshot`` deep-copies state on every save, so a payload
    that cannot be copied surfaces as a BROKEN SAVE, not a broken copy. Copy and
    pickle rebuild a dict subclass by creating an empty one and repopulating it
    through ``__setitem__`` — the method being blocked — which is what
    ``__reduce__`` exists to prevent. Each clone is checked for frozen-ness
    separately: only pickle reaches ``__reduce__``, so asserting the deepcopy
    alone leaves a clone that silently comes back as a plain, writable dict."""
    cp = _checkpoint()
    for clone in (copy.deepcopy(cp), copy.copy(cp), pickle.loads(pickle.dumps(cp))):
        assert clone == cp
        with pytest.raises(TypeError, match="frozen value"):
            clone.state["turn"] = 99
        with pytest.raises(TypeError, match="frozen value"):
            clone.state["scratchpad"]["notes"].append("second")


def test_a_frozen_record_still_compares_equal_to_a_plain_dict_one() -> None:
    """Freezing must not change what equality means, or every consumer that
    compares a decoded record against an expected literal changes behaviour —
    including the adapters' own round-trip assertions. ``FrozenDict`` compares
    equal to the ``dict`` it subclasses in both directions, which is what makes
    this a transparent change at the storage boundary."""
    assert _checkpoint().state == _STATE
    assert _STATE == _checkpoint().state
    assert _checkpoint() == _checkpoint()
    assert isinstance(_checkpoint().state, dict)


def test_resuming_a_run_thaws_the_snapshot_back_into_writable_state() -> None:
    """Freezing is a property of a SNAPSHOT, not of the data. A durable record is
    rightly frozen; a context RESUMED from it is a live working context and has
    to be writable again. Measured before ``thaw`` existed: a resumed run raised
    ``TypeError: this payload belongs to a frozen value...`` on its FIRST
    ``ctx.note(...)``. A top-level ``dict(...)`` — the obvious fix — only moves
    that failure one level down, to the first nested write."""
    resumed = thaw(_checkpoint().state)
    resumed["turn"] = 4
    resumed["scratchpad"]["notes"].append("second")
    resumed["history"][0]["content"] = "edited"
    assert type(resumed) is dict and type(resumed["history"]) is list
    assert resumed["turn"] == 4
