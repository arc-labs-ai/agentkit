"""Hashability of the L0 port value types — and the exact identity each one hashes on.

These three values are the ones callers put in a ``set`` or use as a ``dict``
key: search hits are unioned across providers, fetch responses are cached, and
checkpoints are held per durable row. All three are ``@dataclass(frozen=True)``
carrying a plain ``dict`` field, so the dataclass-generated all-fields
``__hash__`` was unhashable from the first instance — each type therefore
hand-writes ``__hash__`` over an identity SUBSET and leaves ``__eq__`` alone.

Two distinct failures are pinned here, and they fail in opposite directions:

* The hash goes away again — a field that cannot be hashed creeps back into the
  key — and a durable record raises ``TypeError: unhashable type: 'dict'`` at
  the seam that caches it.
* The hash grows a field ``__eq__`` treats as incidental — a query-dependent
  score, a fetched page body — and the value silently lands in a DIFFERENT
  bucket than the logically-same value. Nothing raises: a cache or an
  idempotency key simply misses (or, with the asymmetry reversed, hits another
  call's entry), and the wrong answer is returned as if it were computed.

`tests/source/conformance/test_value_invariants.py` sweeps the public value
types for frozen-ness and credential-free ``repr``; `test_frozen_payloads.py`
covers the ``deep_freeze`` primitive. Neither touches hashing, which is what
this file is for.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from agentkit.core.contracts import ports
from agentkit.core.contracts.ports import Checkpoint, CheckpointStatus, FetchResponse, SearchHit


class _HashCountingStr(str):
    """A ``str`` that records how often something hashed it.

    Used as a probe rather than a timer: the invariant "``FetchResponse``'s hash
    never reads the body" is a statement about WHICH fields enter the key, and a
    wall-clock measurement of that is both slower and flakier than asking the
    body itself whether it was touched.
    """

    def __new__(cls, value: str) -> _HashCountingStr:
        obj = super().__new__(cls, value)
        obj.hash_count = 0
        return obj

    def __hash__(self) -> int:
        self.hash_count += 1
        return super().__hash__()


def _checkpoint(**overrides: Any) -> Checkpoint:
    kwargs: dict[str, Any] = {
        "run_id": "run-1",
        "version": 1,
        "state": {"turn": 3, "history": [{"role": "user"}]},
        "created_at": 1_700_000_000.0,
        "status": CheckpointStatus.RUNNING,
        "metadata": {"producer": "react"},
    }
    kwargs.update(overrides)
    return Checkpoint(**kwargs)


def _search_hit(**overrides: Any) -> SearchHit:
    kwargs: dict[str, Any] = {
        "url": "https://example.test/doc",
        "title": "A document",
        "snippet": "the first query's extracted snippet",
        "score": 0.91,
        "metadata": {"provider": "alpha", "citations": [{"snippet": "x"}]},
    }
    kwargs.update(overrides)
    return SearchHit(**kwargs)


def _fetch_response(**overrides: Any) -> FetchResponse:
    kwargs: dict[str, Any] = {
        "url": "https://example.test/page",
        "status": 200,
        "headers": {"content-type": "text/html", "etag": "W/'1'"},
        "body": "<html>hi</html>",
        "content_type": "text/html",
        "fetched_at": 1_700_000_000.0,
    }
    kwargs.update(overrides)
    return FetchResponse(**kwargs)


# Every frozen port value that carries a mutable container must appear here —
# the sweep at the bottom of this file fails if a new one ships without one.
_FACTORIES = {Checkpoint: _checkpoint, SearchHit: _search_hit, FetchResponse: _fetch_response}


# ── hashable at all ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", list(_FACTORIES.values()), ids=[t.__name__ for t in _FACTORIES])
def test_a_port_value_carrying_opaque_json_is_still_usable_as_a_set_member(factory) -> None:
    """The generated all-fields hash reads every compared field, and on each of
    these types at least one of those fields is a plain ``dict`` — so before the
    hand-written ``__hash__`` existed, ``hash(value)`` raised
    ``TypeError: unhashable type: 'dict'`` on the DEFAULT construction, not on
    some exotic payload. That matters because set membership is the whole point
    of normalising every adapter onto one shape: fanning a query across two
    providers and unioning the results, holding fetched pages in a crawler's
    seen-set, keying checkpoints by durable row. Without this the union raises
    at runtime on the first result that comes back."""
    value = factory()
    assert isinstance(hash(value), int)
    assert value in {value}
    assert {value: "payload"}[value] == "payload"


@pytest.mark.parametrize("factory", list(_FACTORIES.values()), ids=[t.__name__ for t in _FACTORIES])
def test_a_port_value_rebuilt_from_storage_finds_its_own_dict_entry(factory) -> None:
    """Hash and ``__eq__`` have to agree, or a value that compares equal to a
    cached one still misses it. These types are reconstructed constantly — a
    checkpoint comes back through the same constructor on the way OUT of
    storage, a hit is re-normalised from a provider's JSON — and the
    reconstruction is only interchangeable with the original if it lands in the
    same bucket AND compares equal there. A ``__hash__`` over a subset is sound,
    but only while every field in that subset is also a compared field."""
    cache = {factory(): "cached"}
    rebuilt = factory()
    assert hash(rebuilt) == hash(next(iter(cache)))
    assert cache[rebuilt] == "cached"


# ── Checkpoint: the hash is the durable primary key ─────────────────────────


def test_two_snapshots_of_one_durable_row_share_a_bucket_whatever_their_state() -> None:
    """``(run_id, version)`` is not a convenient subset — it is the
    ``PRIMARY KEY (run_id, version)`` of the checkpoints table. Two checkpoints
    hash equal exactly when they name the same durable row, which is the
    discrimination a caller holding checkpoints in a ``set``/``dict`` wants.

    Folding ``state`` back into the key breaks that in two ways at once. State
    is opaque application JSON — nested lists and dicts, unhashable by
    construction — so a value-inclusive hash fails by VALUE rather than by type,
    the worst failure mode for a value type: it works for flat states and raises
    for the real ones. And even where it does not raise, reading the state makes
    the hash O(transcript) at the one seam that hashes on every durable save.

    Both state pairs below matter. Values-differ catches a hash that reads the
    payload; keys-differ catches one that reads only the shape of it — a
    ``tuple(self.state)`` over the key set, which stays hashable and so raises
    nothing while still scattering one row across buckets."""
    row = {"run_id": "run-1", "version": 7}
    same_keys_other_values = _checkpoint(**row, state={"turn": 3}, metadata={"a": 1})
    other_values = _checkpoint(**row, state={"turn": 99}, metadata={"a": 2})
    other_keys = _checkpoint(**row, state={"step": 99, "pending": ["call-1"]}, metadata={"b": 3})

    assert hash(same_keys_other_values) == hash(other_values)
    assert hash(same_keys_other_values) == hash(other_keys)

    # ...and the collision is deliberate, not a loss of discrimination:
    # ``__eq__`` still compares every field, which is what separates them
    # inside the shared bucket.
    assert same_keys_other_values != other_values
    assert len({same_keys_other_values, other_values, other_keys}) == 3


def test_a_checkpoints_hash_still_separates_different_durable_rows() -> None:
    """POSITIVE CONTROL for the test above. "Hash on a subset" degenerates into
    "hash on nothing" if the subset shrinks to empty or to a constant: every
    checkpoint in the framework would then land in one bucket and every
    ``set``/``dict`` operation over them would fall back to a linear scan of
    ``__eq__`` calls. The two fields that name the row must stay in the key."""
    base = _checkpoint(run_id="run-1", version=1)
    assert hash(base) != hash(_checkpoint(run_id="run-2", version=1))
    assert hash(base) != hash(_checkpoint(run_id="run-1", version=2))


# ── SearchHit: one document, whatever a query ranked it ─────────────────────


def test_one_document_hashes_to_one_bucket_however_two_queries_ranked_it() -> None:
    """A web result IS its URL — that is what provider-crossing dedup keys on —
    and the title travels with it as the normalised record's stable half.

    ``score`` and ``snippet`` are excluded because they are QUERY-dependent, not
    result-dependent: the same document ranked 1st for one query and 4th for
    another, with a different snippet extracted each time, is ONE document. Fold
    the score into the key and the union that fanning a query across providers
    exists to perform stops deduplicating — the same page arrives twice under
    two provider scores, lands in two buckets, and is re-ranked, re-fetched and
    cited twice. Nothing raises; the caller just gets a duplicate."""
    first_query = _search_hit(score=0.91, snippet="ranked first")
    second_query = _search_hit(score=0.12, snippet="ranked fourth")
    unscored = _search_hit(score=None, snippet="no score reported")

    assert hash(first_query) == hash(second_query) == hash(unscored)
    # The metadata is arbitrary provider JSON — nested, unhashable by
    # construction, and O(payload) to read — so it is out of the key too.
    assert hash(first_query) == hash(_search_hit(metadata={"provider": "beta", "rank": [4]}))
    assert first_query != second_query


def test_a_search_hits_hash_still_separates_different_documents() -> None:
    """POSITIVE CONTROL: the identity half of the key has to keep discriminating,
    or dedup-by-bucket becomes dedup-by-nothing and every hit collides."""
    assert hash(_search_hit(url="https://example.test/a")) != hash(_search_hit(url="https://example.test/b"))
    assert hash(_search_hit(title="A")) != hash(_search_hit(title="B"))


# ── FetchResponse: the key never reads the page ─────────────────────────────


def test_hashing_a_fetch_response_never_reads_the_page_body() -> None:
    """``body`` is excluded even though ``str`` IS hashable, and that is the
    load-bearing exclusion. A fetched page is unbounded — this is the type a
    crawler holds thousands of — and ``hash(str)`` is O(len) with no digest
    shared across instances, so a body-inclusive hash makes every ``set``
    insertion cost a full scan of the document. CPython caches a string's hash
    ON the string object, so the cost falls on the FIRST hash of each instance,
    which for a crawler inserting each response exactly once is every one of
    them.

    Asserted by instrumenting the body rather than by timing it: the invariant
    is which fields enter the key, and the body being hashed zero times is the
    precise statement of it. ``fetched_at`` is IN the key on purpose — two
    fetches of one URL at different times are different responses, and the
    timestamp separates them without reading the payload."""
    body = _HashCountingStr("<html>" + "x" * 10_000 + "</html>")
    response = _fetch_response(body=body)

    assert hash(response) == hash(response)
    assert body.hash_count == 0, "the response's hash read its body"


def test_two_fetches_of_one_url_at_one_instant_share_a_bucket_whatever_the_body() -> None:
    """The companion to the probe above, stated as the user-visible consequence:
    a page re-fetched at the same timestamp after an edit collides into one
    bucket, where ``__eq__`` tells the two apart. Excluding the body is sound
    rather than a workaround — the hash invariant only requires that EQUAL
    objects hash equally, never that unequal ones differ."""
    original = _fetch_response(body="<html>before</html>")
    edited = _fetch_response(body="<html>after the edit</html>")

    assert hash(original) == hash(edited)
    # ...and headers, which a caching middleware normalises in place, are out of
    # the key too — they are a plain dict at the type level.
    assert hash(original) == hash(_fetch_response(headers={"content-type": "text/html", "etag": "W/'2'"}))
    assert original != edited


def test_a_fetch_responses_hash_still_separates_responses_a_cache_must_not_merge() -> None:
    """POSITIVE CONTROL. ``url``, ``status``, ``content_type`` and ``fetched_at``
    are the response's identity; a cache that merged a 200 with a 404, or two
    fetches of one URL minutes apart, would serve a stale or wrong page."""
    base = _fetch_response()
    assert hash(base) != hash(_fetch_response(url="https://example.test/other"))
    assert hash(base) != hash(_fetch_response(status=404))
    assert hash(base) != hash(_fetch_response(content_type="application/json"))
    assert hash(base) != hash(_fetch_response(fetched_at=1_700_000_060.0))


# ── the ratchet: a NEW port value cannot ship unhashable ────────────────────


def _frozen_port_values_with_mutable_payloads() -> list[type]:
    found = []
    for obj in vars(ports).values():
        if not isinstance(obj, type) or not dataclasses.is_dataclass(obj):
            continue
        if getattr(obj, "__module__", None) != ports.__name__:
            continue
        if not obj.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            continue
        # Field types are STRINGS here (`from __future__ import annotations`),
        # which is enough: the question is what the declaration promises.
        if any(("dict" in str(f.type)) or ("list" in str(f.type)) for f in dataclasses.fields(obj)):
            found.append(obj)
    return found


def test_every_frozen_port_value_with_a_mutable_payload_hand_writes_its_hash() -> None:
    """The failure mode these three types each hit is a property of the SHAPE —
    a frozen dataclass declaring a ``dict`` or ``list`` field — so the next type
    with that shape will hit it too, and per-type tests only protect the types
    someone remembered. This is the ratchet: a new frozen port value carrying a
    container must either define its own ``__hash__`` over an identity subset or
    be explicitly opted out, and it must be exercised above rather than merely
    discovered here."""
    discovered = _frozen_port_values_with_mutable_payloads()
    assert discovered, "the sweep found nothing — it has stopped discovering anything"
    for cls in discovered:
        assert "__hash__" in cls.__dict__ and cls.__hash__ is not None, (
            f"{cls.__name__} declares a container field and inherits the dataclass all-fields "
            "hash, which cannot hash it — give it a __hash__ over its identity subset"
        )
        assert cls in _FACTORIES, (
            f"{cls.__name__} is a new frozen port value with a container payload; add a factory "
            "and assert what its hash's identity subset is, with the reason"
        )
