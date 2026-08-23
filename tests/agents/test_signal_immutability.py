"""Signal payloads are values, not buffers.

``@dataclass(frozen=True)`` only ever blocked REASSIGNMENT. The six
collection payloads in the protocol stayed mutable and aliased, so on
the code before this module::

    frozen blocks reassignment : FrozenInstanceError
    ...but the payload mutates : final_delta=['x', 'FORGED'] metrics={'used_cost': 999.0}
    caller's list is ALIASED   : signal sees ['x', 'FORGED', 'changed after emit']
    hashable (audit dedupe)    : NO — unhashable type: 'list'
    hashable w/o payload       : yes

Three separate defects: a receiver could rewrite a ``DoneSignal``'s
reported spend after the parent absorbed it; a sender kept a live
handle into every signal it had already emitted; and payload-carrying
signals were unhashable while payload-free ones were not, so "put
signals in a set to dedupe" worked or raised depending on which signal
class you happened to hold.

Every test below the ``POSITIVE CONTROLS`` banner passes both before
and after the fix — they guard the ergonomics the fix had to preserve.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, Generic, TypeVar

import pytest

from agentkit.agents.control.channel import SignalChannel
from agentkit.agents.control.signals import (
    FROZEN_PAYLOAD,
    BlockedSignal,
    BudgetReducedSignal,
    CancelSignal,
    ContextUpdateSignal,
    DataSignal,
    DoneSignal,
    EscalateSignal,
    MergeWithPeerSignal,
    ProgressSignal,
    RedirectSignal,
    SignalEnvelope,
)

MutT = TypeVar("MutT")


@dataclass(slots=True, frozen=True)
class SampleMutation:
    agent_id: str
    body: str


@dataclass(slots=True)
class UnhashableMutation:
    """A mutation type a project might plausibly write: mutable, and
    therefore unhashable. Nothing in the protocol requires ``MutationT``
    to be hashable, so a signal carrying one must still hash."""

    agent_id: str
    tags: list[str] = field(default_factory=list)


def _payload_signals() -> list[tuple[str, SignalEnvelope]]:
    """One instance of every signal that carries a collection payload."""
    return [
        ("BudgetReducedSignal", BudgetReducedSignal(constraints={"cost": 1.0})),
        ("ContextUpdateSignal", ContextUpdateSignal(mutations=["m"])),
        ("ProgressSignal", ProgressSignal(mutations=["m"])),
        ("DoneSignal", DoneSignal(final_delta=["m"], metrics={"used_cost": 1.0})),
        ("EscalateSignal", EscalateSignal(options=["retry"])),
    ]


def _all_signals() -> list[tuple[str, SignalEnvelope]]:
    return [
        *_payload_signals(),
        ("SignalEnvelope", SignalEnvelope()),
        ("CancelSignal", CancelSignal(reason="stop")),
        ("RedirectSignal", RedirectSignal[dict[str, Any]](new_state={"task": "x"})),
        ("MergeWithPeerSignal", MergeWithPeerSignal(survivor_agent_id="b")),
        ("BlockedSignal", BlockedSignal(reason="stuck")),
    ]


# ── The bug: a receiver could edit an already-emitted signal ────────


def test_done_signal_metrics_cannot_be_rewritten_after_the_fact() -> None:
    """The exact forgery the audit trail has to rule out: ``DoneSignal``
    is what the budget ledger reconciles reported spend from, and the
    receiver could write a new ``used_cost`` into a signal the sender
    already shipped."""
    done = DoneSignal[SampleMutation](
        final_delta=[SampleMutation(agent_id="a", body="m1")],
        metrics={"used_cost": 1.0},
    )

    with pytest.raises(TypeError):
        done.metrics["used_cost"] = 999.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        done.final_delta.append(  # type: ignore[attr-defined]
            SampleMutation(agent_id="forger", body="FORGED")
        )

    assert done.metrics["used_cost"] == 1.0
    assert len(done.final_delta) == 1


@pytest.mark.parametrize(
    ("name", "signal", "attr"),
    [
        ("BudgetReducedSignal", BudgetReducedSignal(constraints={"cost": 1.0}), "constraints"),
        ("ContextUpdateSignal", ContextUpdateSignal(mutations=["m"]), "mutations"),
        ("ProgressSignal", ProgressSignal(mutations=["m"]), "mutations"),
        ("DoneSignal", DoneSignal(final_delta=["m"]), "final_delta"),
        ("DoneSignal", DoneSignal(metrics={"used_cost": 1.0}), "metrics"),
        ("EscalateSignal", EscalateSignal(options=["retry"]), "options"),
    ],
)
def test_every_payload_field_is_read_only(
    name: str, signal: SignalEnvelope, attr: str
) -> None:
    """All six payloads, not just the one that motivated the fix — a
    per-field fix would leave the others as the same latent bug.

    This asserts the GUARANTEE (mutation is refused), not the mechanism. It
    used to assert `not isinstance(payload, (list, dict))`, which was really a
    test that the payload was a `MappingProxyType`/`tuple`. When mapping
    payloads moved to `FrozenDict` — a `dict` SUBCLASS, so that `json.dumps`
    and `dataclasses.asdict` keep working on an audit record — that assertion
    failed while the actual behaviour was strictly better. A test pinned to a
    mechanism blocks improving the mechanism.
    """
    payload = getattr(signal, attr)
    with pytest.raises(TypeError):
        if isinstance(payload, dict):
            payload["injected"] = "forged"
        else:
            payload[0] = "forged"  # type: ignore[index]
    if isinstance(payload, Mapping):
        with pytest.raises(TypeError):
            payload["forged"] = 1.0
    else:
        assert not hasattr(payload, "append")
        with pytest.raises(TypeError):
            payload[0] = "forged"


def test_sender_cannot_retroactively_change_an_emitted_signal() -> None:
    """The aliasing half, and the nastier one: the child emits a slice
    of a journal list it goes on appending to. Aliased, every later
    append silently rewrote signals the parent had already absorbed,
    with no moment in the stream where that shows up."""
    journal = [SampleMutation(agent_id="a", body="m1")]
    progress = ProgressSignal[SampleMutation](mutations=journal)

    journal.append(SampleMutation(agent_id="a", body="written after emit"))

    assert len(progress.mutations) == 1
    assert progress.mutations[0].body == "m1"


def test_sender_cannot_retroactively_change_an_emitted_mapping() -> None:
    budget = {"cost": 10.0}
    signal = BudgetReducedSignal(constraints=budget)
    budget["cost"] = 0.0
    budget["tokens"] = 5.0
    assert dict(signal.constraints) == {"cost": 10.0}


@pytest.mark.asyncio
async def test_payload_stays_frozen_through_the_channel() -> None:
    """The channel re-stamps via ``dataclasses.replace``, which re-runs
    ``__init__``. The freeze has to survive that seam, since the stamped
    copy is the one the parent actually receives."""
    channel = SignalChannel[Any, Any](agent_id="agent_a")
    await channel.emit(DoneSignal[SampleMutation](metrics={"used_cost": 1.0}))

    out = channel.outbox.get_nowait()
    assert out.sender_id == "agent_a"
    with pytest.raises(TypeError):
        out.metrics["used_cost"] = 999.0


# ── The bug: hashability depended on which signal you held ─────────


@pytest.mark.parametrize(("name", "signal"), _all_signals())
def test_every_signal_is_hashable(name: str, signal: SignalEnvelope) -> None:
    """Payload-free signals hashed and payload-carrying ones raised
    ``TypeError: unhashable type: 'list'``. Consistency is the point:
    a caller putting signals in a set for dedupe should not have to know
    which arm of the protocol it is holding."""
    assert isinstance(hash(signal), int)
    assert len({signal, signal}) == 1


def test_hashing_does_not_require_hashable_mutations() -> None:
    """``MutationT`` is a user type with no hashability contract, so the
    hash covers the envelope and the scalars, never the payload. A
    mutable mutation dataclass is the normal case, not an exotic one."""
    signal = ProgressSignal[UnhashableMutation](
        mutations=[UnhashableMutation(agent_id="a", tags=["t"])]
    )
    with pytest.raises(TypeError):
        hash(signal.mutations[0])
    assert isinstance(hash(signal), int)


def test_redirect_hash_does_not_require_hashable_state() -> None:
    """Same reasoning for ``StateT`` — a dict snapshot is the common
    case, and it must not be the one signal class that fails to hash."""
    assert isinstance(hash(RedirectSignal[dict[str, Any]](new_state={"task": "x"})), int)


def test_equal_signals_hash_equally() -> None:
    """The invariant the hash must not break. Excluding the payload
    weakens the hash (more collisions); it must never split two signals
    that compare equal."""
    a = DoneSignal[str](final_delta=["m"], metrics={"c": 1.0}, sender_id="child")
    b = DoneSignal[str](final_delta=["m"], metrics={"c": 1.0}, sender_id="child")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_signals_dedupe_in_a_set() -> None:
    """The stated use case end to end: distinct signals stay distinct
    once the channel has stamped them, duplicates collapse."""
    made = [
        DoneSignal[str](final_delta=["m"], sender_id="child", timestamp_us=t)
        for t in (1, 2, 2)
    ]
    assert len(set(made)) == 2


# ── Survives the replay / audit boundary ───────────────────────────


@pytest.mark.parametrize(("name", "signal"), _all_signals())
def test_signal_survives_deepcopy(name: str, signal: SignalEnvelope) -> None:
    """``Checkpointer.snapshot`` deep-copies state at the durable seam.
    Freezing a mapping with ``MappingProxyType`` and stopping there
    breaks this with ``TypeError: cannot pickle 'mappingproxy' object``
    — the regression the ``Prompt`` fix already paid for once."""
    clone = copy.deepcopy(signal)
    assert clone == signal
    assert clone is not signal


@pytest.mark.parametrize(("name", "signal"), _all_signals())
def test_signal_survives_pickle(name: str, signal: SignalEnvelope) -> None:
    assert pickle.loads(pickle.dumps(signal)) == signal


def test_deepcopy_does_not_alias_payload_contents() -> None:
    """A deep copy has to be deep THROUGH the frozen container, or the
    checkpoint and the live signal share mutation objects and the
    snapshot is not a snapshot."""
    original = ProgressSignal[UnhashableMutation](
        mutations=[UnhashableMutation(agent_id="a", tags=["t"])]
    )
    clone = copy.deepcopy(original)
    clone.mutations[0].tags.append("added-to-the-copy")
    assert original.mutations[0].tags == ["t"]


def test_round_trip_payload_is_still_frozen() -> None:
    """Rebuilding through the constructor means ``__post_init__`` runs
    again — a round trip that handed back a plain dict would launder the
    guarantee away at exactly the replay boundary it exists for."""
    revived = pickle.loads(pickle.dumps(DoneSignal[str](metrics={"used_cost": 1.0})))
    with pytest.raises(TypeError):
        revived.metrics["used_cost"] = 999.0
    revived2 = copy.deepcopy(ProgressSignal[str](mutations=["m"]))
    with pytest.raises(AttributeError):
        revived2.mutations.append("m2")


# ── Subclassing: the documented extension point ────────────────────


@dataclass(slots=True, frozen=True)
class FindingSignal(DataSignal, Generic[MutT]):
    """What a project is told to write: a domain signal with its own
    collection payload, opted in via ``FROZEN_PAYLOAD``."""

    findings: list[MutT] = field(
        default_factory=list, hash=False, metadata=FROZEN_PAYLOAD
    )
    url: str = ""


def test_subclass_payload_is_frozen_and_unaliased() -> None:
    caller = ["f1"]
    signal = FindingSignal[str](findings=caller, url="https://x")
    caller.append("after emit")
    assert len(signal.findings) == 1
    with pytest.raises(AttributeError):
        signal.findings.append("f2")


def test_subclass_round_trips_and_hashes() -> None:
    """``__reduce__`` is built from ``fields()`` rather than a fixed
    argument list precisely so a user subclass with extra fields
    (``url`` here) round-trips instead of losing them."""
    signal = FindingSignal[str](findings=["f1"], url="https://x")
    assert pickle.loads(pickle.dumps(signal)) == signal
    assert copy.deepcopy(signal) == signal
    assert isinstance(hash(signal), int)
    assert pickle.loads(pickle.dumps(signal)).url == "https://x"


@dataclass(slots=True, frozen=True)
class InheritedPayloadSignal(DoneSignal[MutT], Generic[MutT]):
    """Subclassing a payload-carrying signal, not just the base: the
    inherited ``final_delta`` / ``metrics`` must stay frozen even though
    the subclass never mentions them."""

    verdict: str = ""


def test_inherited_payload_stays_frozen_in_a_subclass() -> None:
    signal = InheritedPayloadSignal[str](
        final_delta=["m"], metrics={"used_cost": 1.0}, verdict="ok"
    )
    with pytest.raises(AttributeError):
        signal.final_delta.append("m2")
    with pytest.raises(TypeError):
        signal.metrics["used_cost"] = 999.0
    assert isinstance(hash(signal), int)
    assert pickle.loads(pickle.dumps(signal)).verdict == "ok"


@dataclass(slots=True, frozen=True)
class NormalisingSignal(DataSignal):
    """A subclass that needs its own ``__post_init__`` REPLACES the
    envelope's, so it has to chain — by NAMING the base, because a bare
    ``super()`` does not work under ``slots=True``."""

    options: list[str] = field(
        default_factory=list, hash=False, metadata=FROZEN_PAYLOAD
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", [o.lower() for o in self.options])
        SignalEnvelope.__post_init__(self)


def test_subclass_post_init_can_extend_the_freeze() -> None:
    """The escape hatch is asserted rather than assumed, in the exact
    spelling the docstring recommends."""
    signal = NormalisingSignal(options=["RETRY"])
    assert list(signal.options) == ["retry"]
    with pytest.raises(AttributeError):
        signal.options.append("retire")


def test_a_subclass_post_init_must_chain_and_the_prescribed_spelling_always_works() -> None:
    """A subclass that overrides ``__post_init__`` has to chain, or its payload
    is never frozen. The docstring prescribes the EXPLICIT spelling —
    ``SignalEnvelope.__post_init__(self)`` — and that is what this pins,
    because it is the part that is true on every interpreter.

    Zero-argument ``super()`` is deliberately NOT asserted either way. It used
    to be a reliable footgun: ``@dataclass(slots=True)`` cannot add slots in
    place, so it returns a REPLACEMENT class while the zero-arg ``super()``
    closes over the original, which the instance is not an instance of. CPython
    has since fixed that, and the fix landed mid-3.13:

        3.12      TypeError
        3.13.2    TypeError  (reworded)
        3.13.14   works

    An earlier version of this test asserted the raise, so it passed locally on
    3.13.2 and failed CI on 3.13.14 — pinning a CPython BUG rather than this
    module's contract. What actually matters is invariant across all three: the
    payload ends up frozen. So that is what is asserted, whichever way the
    interpreter goes.
    """

    @dataclass(slots=True, frozen=True)
    class Prescribed(DataSignal):
        items: list[str] = field(
            default_factory=list, hash=False, metadata=FROZEN_PAYLOAD
        )

        def __post_init__(self) -> None:
            SignalEnvelope.__post_init__(self)

    # The prescribed form: works everywhere, and freezes.
    sig = Prescribed(items=["a"])
    assert sig.items == ("a",)
    with pytest.raises(AttributeError):
        sig.items.append("b")  # type: ignore[attr-defined]

    @dataclass(slots=True, frozen=True)
    class BareSuperSlots(DataSignal):
        items: list[str] = field(
            default_factory=list, hash=False, metadata=FROZEN_PAYLOAD
        )

        def __post_init__(self) -> None:
            super().__post_init__()

    # Version-tolerant: either the interpreter refuses the zero-arg super, or
    # it chains and the payload comes back frozen. The one outcome that would
    # be a real defect — constructing successfully with an UNFROZEN payload —
    # is excluded either way.
    try:
        built = BareSuperSlots(items=["a"])
    except TypeError as exc:
        assert "instance or subtype" in str(exc)
    else:
        assert built.items == ("a",), "chained but did not freeze"

    # Without slots there is no replacement class, so bare super has always
    # worked — that half is version-independent.
    @dataclass(frozen=True)
    class BareSuperNoSlots(DataSignal):
        items: list[str] = field(
            default_factory=list, hash=False, metadata=FROZEN_PAYLOAD
        )

        def __post_init__(self) -> None:
            super().__post_init__()

    assert BareSuperNoSlots(items=["a"]).items == ("a",)


# ══ POSITIVE CONTROLS — pass before AND after the fix ══════════════


def test_construction_with_plain_list_and_dict_still_works() -> None:
    """The documented ergonomic. Callers pass builtins; the freeze is
    the framework's business, not theirs."""
    done = DoneSignal[SampleMutation](
        final_delta=[SampleMutation(agent_id="a", body="m1")],
        confidence=0.9,
        metrics={"used_cost": 1.5, "used_tokens": 100.0},
    )
    assert done.final_delta[0].agent_id == "a"
    assert done.metrics["used_cost"] == 1.5
    assert done.confidence == 0.9
    assert len(done.final_delta) == 1
    assert list(done.metrics) == ["used_cost", "used_tokens"]
    assert "used_tokens" in done.metrics
    assert [m.body for m in done.final_delta] == ["m1"]


def test_frozen_mapping_still_compares_equal_to_a_plain_dict() -> None:
    """A read-only ``Mapping`` is not a ``dict``, but it compares equal
    to one in both directions — so existing assertions and consumers
    reading ``signal.metrics == {...}`` are unaffected."""
    metrics = BudgetReducedSignal(constraints={"cost": 1.0}).constraints
    assert metrics == {"cost": 1.0}
    assert {"cost": 1.0} == metrics
    assert dict(metrics) == {"cost": 1.0}


def test_defaults_are_empty_and_usable() -> None:
    """Empty payloads — the overwhelmingly common construction, and the
    one an over-eager freeze would be most likely to break."""
    assert len(ProgressSignal[str]().mutations) == 0
    assert list(DoneSignal[str]().final_delta) == []
    assert dict(DoneSignal[str]().metrics) == {}
    assert not EscalateSignal().options
    assert list(ContextUpdateSignal[str](mutations=[]).mutations) == []
    assert BudgetReducedSignal(constraints={}).constraints == {}


def test_equality_between_signals_with_equal_payloads() -> None:
    a = ProgressSignal[SampleMutation](
        mutations=[SampleMutation(agent_id="a", body="m")], confidence=0.5
    )
    b = ProgressSignal[SampleMutation](
        mutations=[SampleMutation(agent_id="a", body="m")], confidence=0.5
    )
    c = ProgressSignal[SampleMutation](mutations=[], confidence=0.5)
    assert a == b
    assert a != c


def test_payload_is_still_compared_by_equality() -> None:
    """Excluding the payload from the HASH must not exclude it from
    ``__eq__`` — that would make two different deltas the same signal."""
    assert DoneSignal[str](final_delta=["a"]) != DoneSignal[str](final_delta=["b"])
    assert DoneSignal[str](metrics={"c": 1.0}) != DoneSignal[str](metrics={"c": 2.0})


def test_dataclasses_replace_preserves_the_payload() -> None:
    """``SignalChannel.emit`` stamps via ``replace``, so this is the
    framework's own hot path, not just a stdlib nicety."""
    done = DoneSignal[str](final_delta=["m"], metrics={"used_cost": 1.0})
    stamped = replace(done, sender_id="child", timestamp_us=42)
    assert list(stamped.final_delta) == ["m"]
    assert stamped.metrics == {"used_cost": 1.0}
    assert stamped.sender_id == "child"
    assert replace(done, final_delta=["n"]).final_delta[0] == "n"


def test_generics_and_direction_typing_still_hold() -> None:
    """The parameterised forms are the documented usage and the
    ``isinstance`` dispatch is how the framework routes."""
    progress = ProgressSignal[SampleMutation](mutations=[])
    assert isinstance(progress, DataSignal)
    assert not isinstance(progress, CancelSignal)
    assert isinstance(ContextUpdateSignal[SampleMutation](mutations=[]), SignalEnvelope)
    assert RedirectSignal[str](new_state="s").new_state == "s"


def test_envelope_fields_are_untouched() -> None:
    signal = ProgressSignal[str](
        mutations=["m"],
        correlation_id="c1",
        causation_id="c0",
        sender_id="child",
        timestamp_us=7,
    )
    assert (signal.correlation_id, signal.causation_id) == ("c1", "c0")
    assert (signal.sender_id, signal.timestamp_us) == ("child", 7)


def test_reassignment_is_still_refused() -> None:
    with pytest.raises(Exception, match="cannot assign to field"):
        DoneSignal[str]().confidence = 0.9  # type: ignore[misc]


# ── Documented edges ───────────────────────────────────────────────


def test_freezing_reaches_nested_containers_but_not_user_objects() -> None:
    """The freeze now reaches CONTAINERS all the way down, and still stops at
    the project's own objects.

    This test previously pinned the opposite — `..._is_shallow_by_design` —
    and asserted that a `list[dict]` payload still handed out mutable dicts.
    That was the right call for its stated reason: going deeper must not mean
    recursively rewriting `MutationT` objects the framework cannot
    reconstruct. But it over-corrected. `deep_freeze` only ever replaces dicts
    and lists; a `MutationT` instance is returned untouched, by identity. So
    the rationale is preserved while the hole it left — a signal's nested
    `{"used_cost": ...}` still being rewritable after the fact, on a record
    whose whole purpose is audit — is closed.
    """

    class Mutation:
        """A project's own mutation type. Must survive by IDENTITY."""

        def __init__(self) -> None:
            self.editable = True

    mine = Mutation()
    payload: list = [{"kind": "note"}, mine]
    signal = ContextUpdateSignal(mutations=payload)

    # The container is frozen...
    with pytest.raises((AttributeError, TypeError)):
        signal.mutations.append({"kind": "forged"})  # type: ignore[attr-defined]
    # ...and so is the nested dict, which is the part that used to be open.
    with pytest.raises(TypeError):
        signal.mutations[0]["kind"] = "edited"

    # The caller's list is un-aliased.
    payload.append({"kind": "not in the signal"})
    assert len(signal.mutations) == 2

    # But the project's own object is the SAME object, untouched — the line
    # this module deliberately refuses to cross.
    assert signal.mutations[1] is mine
    signal.mutations[1].editable = False  # its own semantics, not ours to police
    assert mine.editable is False


def test_a_bare_string_payload_is_refused() -> None:
    """``Sequence[str]`` accepts a bare ``str`` where the old
    ``list[str]`` did not, and ``tuple("retry")`` would silently become
    five one-character options. Refusing keeps the type-checker
    protection the widened annotation would otherwise have cost."""
    with pytest.raises(TypeError, match="not a bare str"):
        EscalateSignal(options="retry")  # type: ignore[arg-type]


def test_generator_payloads_are_materialised() -> None:
    """A consequence of copy-then-freeze worth having: a generator is
    consumed once at construction instead of being stored as a
    one-shot iterator that the second reader finds empty."""
    signal = ProgressSignal[str](mutations=(f"m{i}" for i in range(3)))
    assert list(signal.mutations) == ["m0", "m1", "m2"]
    assert list(signal.mutations) == ["m0", "m1", "m2"]


def test_payload_fields_are_excluded_from_the_hash_not_from_eq() -> None:
    """Pins the mechanism, since it is the part a future edit is most
    likely to drop: ``hash=False`` on the field, ``compare`` left on.
    Naming the six expected fields keeps this from passing vacuously if
    the marker is ever renamed or dropped."""
    marked: set[str] = set()
    for signal in (s for _, s in _payload_signals()):
        for f in fields(signal):
            if not f.metadata.get("agentkit.frozen_payload"):
                continue
            marked.add(f"{type(signal).__name__}.{f.name}")
            assert f.hash is False, f"{type(signal).__name__}.{f.name}"
            assert f.compare is True, f"{type(signal).__name__}.{f.name}"

    assert marked == {
        "BudgetReducedSignal.constraints",
        "ContextUpdateSignal.mutations",
        "ProgressSignal.mutations",
        "DoneSignal.final_delta",
        "DoneSignal.metrics",
        "EscalateSignal.options",
    }
