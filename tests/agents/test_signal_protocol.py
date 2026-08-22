"""Tests for the agent coordination primitives — signals, channel, ActorBudget."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.agents.control.budget import ActorBudget, BudgetExhausted
from agentkit.agents.control.channel import MergeInbox, SignalChannel
from agentkit.agents.control.signals import (
    CancelSignal,
    DataSignal,
    DoneSignal,
    ProgressSignal,
)


@dataclass(slots=True)
class SampleMutation:
    agent_id: str
    body: str


@pytest.mark.asyncio
async def test_channel_emit_stamps_sender_and_timestamp() -> None:
    channel = SignalChannel[Any, Any](agent_id="agent_a")
    signal = ProgressSignal[SampleMutation](mutations=[])
    assert signal.sender_id is None
    assert signal.timestamp_us == 0

    await channel.emit(signal)

    out = channel.outbox.get_nowait()
    assert out.sender_id == "agent_a"
    assert out.timestamp_us > 0


@pytest.mark.asyncio
async def test_channel_emit_preserves_caller_set_sender_and_timestamp() -> None:
    channel = SignalChannel[Any, Any](agent_id="agent_a")
    signal = ProgressSignal[SampleMutation](
        mutations=[],
        sender_id="explicit_sender",
        timestamp_us=42,
    )
    await channel.emit(signal)
    out = channel.outbox.get_nowait()
    assert out.sender_id == "explicit_sender"
    assert out.timestamp_us == 42


@pytest.mark.asyncio
async def test_emit_fans_to_outbox_and_parent_merge_inbox() -> None:
    """Emit dual-writes outbox AND parent's merge inbox — what makes
    child progress stream upward without polling each descendant.

    The envelope is frozen, so ``emit`` publishes a stamped copy rather
    than the caller's original object. Both queues must see the SAME
    stamped copy so the audit trail and merge-loop agree on envelope
    identity (correlation_id / causation_id etc).
    """
    parent_merge: MergeInbox[DataSignal] = MergeInbox()
    channel = SignalChannel[Any, Any](agent_id="child")
    channel.attach_parent(parent_merge)

    signal = ProgressSignal[SampleMutation](mutations=[])
    await channel.emit(signal)

    out = channel.outbox.get_nowait()
    assert out.sender_id == "child"
    assert out.timestamp_us != 0
    assert out.mutations == signal.mutations

    sender, fwd = parent_merge.get_nowait()
    assert sender == "child"
    assert fwd is out  # outbox and merge_inbox see the SAME stamped envelope


@pytest.mark.asyncio
async def test_detached_channel_emits_only_to_outbox() -> None:
    channel = SignalChannel[Any, Any](agent_id="orphan")
    await channel.emit(ProgressSignal[SampleMutation]())
    assert not channel.outbox.empty()


@pytest.mark.asyncio
async def test_merge_inbox_preserves_arrival_order() -> None:
    inbox: MergeInbox[DataSignal] = MergeInbox()
    s1 = ProgressSignal[SampleMutation]()
    s2 = DoneSignal[SampleMutation](confidence=0.7)
    s3 = ProgressSignal[SampleMutation]()
    await inbox.put(("child_a", s1))
    await inbox.put(("child_b", s2))
    await inbox.put(("child_a", s3))

    assert inbox.get_nowait() == ("child_a", s1)
    assert inbox.get_nowait() == ("child_b", s2)
    assert inbox.get_nowait() == ("child_a", s3)
    assert inbox.empty()


def test_cancel_signal_carries_reason() -> None:
    cancel = CancelSignal(reason="user requested")
    assert cancel.reason == "user requested"


def test_progress_signal_carries_mutations() -> None:
    progress = ProgressSignal[SampleMutation](
        mutations=[SampleMutation(agent_id="a", body="m1")],
        confidence=0.85,
    )
    assert progress.mutations[0].agent_id == "a"
    assert progress.confidence == 0.85


def test_actor_budget_reserve_blocks_further_spawn_above_cap() -> None:
    b = ActorBudget(max_tokens=1000, max_cost_usd=1.0, max_steps=10, max_wall_seconds=60.0)
    b.reserve_for_child(tokens=600, cost_usd=0.5, steps=5)
    assert b.can_spawn_child(request_tokens=400, request_cost_usd=0.4, request_steps=4)
    assert not b.can_spawn_child(request_tokens=500, request_cost_usd=0.1, request_steps=1)


def test_actor_budget_reserve_raises_when_overcommitted() -> None:
    b = ActorBudget(max_tokens=100, max_cost_usd=1.0, max_steps=10, max_wall_seconds=60.0)
    with pytest.raises(BudgetExhausted) as ei:
        b.reserve_for_child(tokens=200, cost_usd=0.1, steps=1)
    assert ei.value.axis == "tokens"


def test_actor_budget_settle_capped_at_reservation() -> None:
    """A child overspending locally never silently pushes the parent's
    books past the reservation — the cap is the framework's enforcement
    of the slice."""
    b = ActorBudget(max_tokens=1000, max_cost_usd=1.0, max_steps=10, max_wall_seconds=60.0)
    b.reserve_for_child(tokens=400, cost_usd=0.3, steps=4)
    b.settle_child(
        reserved_tokens=400,
        reserved_cost_usd=0.3,
        reserved_steps=4,
        used_tokens=500,
        used_cost_usd=0.4,
        used_steps=5,
    )
    assert b.reserved_tokens == 0
    assert b.used_tokens == 400
    assert b.used_cost_usd == pytest.approx(0.3)
    assert b.used_steps == 4


def test_actor_budget_tighten_never_goes_below_already_used() -> None:
    """Live-tightening from a parent's BudgetReducedSignal clamps to
    used+reserved — otherwise it would retroactively exhaust the agent."""
    b = ActorBudget(max_tokens=1000, max_cost_usd=1.0, max_steps=10, max_wall_seconds=60.0)
    b.charge(tokens=300)
    b.tighten(new_max_tokens=100)
    assert b.max_tokens == 300


def test_actor_budget_distinct_from_run_budget() -> None:
    from agentkit.runtime import Budget as RunBudget

    actor = ActorBudget(max_tokens=100, max_cost_usd=1.0, max_steps=10, max_wall_seconds=10.0)
    assert ActorBudget is not RunBudget
    assert actor.reserved_tokens == 0


# ── Envelope immutability ────────────────────────────────────────────────────
#
# Signals fan out to multiple readers (outbox for audit, parent's
# merge_inbox for live dispatch). A mutable envelope shared across
# readers would let one consumer observe a stamped ``sender_id``
# while another observed the pre-stamp ``None`` — a race with no
# stable observable state.


def test_signal_envelope_is_frozen() -> None:
    """The envelope refuses in-place mutation of any field. Callers
    that need to update a stamp build a new instance via
    ``dataclasses.replace``."""
    from dataclasses import FrozenInstanceError

    signal = ProgressSignal[SampleMutation](mutations=[])
    with pytest.raises(FrozenInstanceError):
        signal.sender_id = "clobber"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        signal.timestamp_us = 42  # type: ignore[misc]


@pytest.mark.asyncio
async def test_channel_emit_does_not_mutate_caller_signal() -> None:
    """The caller's original signal instance stays byte-identical
    after emit — the channel publishes a stamped copy. A caller that
    holds a reference to what they emitted for logging/audit sees the
    unstamped original, and the queue sees the stamped copy."""
    channel = SignalChannel[Any, Any](agent_id="agent_a")
    signal = ProgressSignal[SampleMutation](mutations=[])
    signal_id = id(signal)

    await channel.emit(signal)

    # Caller's instance: still None sender / 0 timestamp.
    assert signal.sender_id is None
    assert signal.timestamp_us == 0
    # The queue got a distinct instance (with the stamps applied).
    out = channel.outbox.get_nowait()
    assert id(out) != signal_id
    assert out.sender_id == "agent_a"
    assert out.timestamp_us > 0


@pytest.mark.asyncio
async def test_channel_emit_produces_same_stamped_copy_for_outbox_and_parent() -> None:
    """Both the outbox and the parent's merge_inbox see the SAME
    stamped envelope — so audit and merge-loop dispatch agree on the
    envelope's identity (correlation_id / causation_id / stamp)."""
    parent_merge: MergeInbox[DataSignal] = MergeInbox()
    channel = SignalChannel[Any, Any](agent_id="child")
    channel.attach_parent(parent_merge)

    await channel.emit(ProgressSignal[SampleMutation](mutations=[]))

    out = channel.outbox.get_nowait()
    sender, fwd = parent_merge.get_nowait()
    assert sender == "child"
    assert fwd is out  # exact reference equality across the two queues


@pytest.mark.asyncio
async def test_channel_emit_stamps_are_monotonic_across_signals() -> None:
    """The channel's clock stamps timestamps in monotonic order across
    a burst of emits from the same agent — replay tools can order a
    cascade by ``timestamp_us`` without needing correlation ids."""
    channel = SignalChannel[Any, Any](agent_id="agent_a")

    for _ in range(5):
        await channel.emit(ProgressSignal[SampleMutation](mutations=[]))

    stamps = []
    while not channel.outbox.empty():
        stamps.append(channel.outbox.get_nowait().timestamp_us)

    assert stamps == sorted(stamps)


@pytest.mark.asyncio
async def test_channel_emit_stamps_child_id_when_forwarded_to_parent() -> None:
    """Sender attribution: the merge_inbox always sees the child's
    agent_id as the sender key (not the raw envelope's — a child that
    forwards a signal from its OWN child would otherwise mask the
    intermediate hop)."""
    grandparent = MergeInbox[DataSignal]()
    channel = SignalChannel[Any, Any](agent_id="child")
    channel.attach_parent(grandparent)

    # The signal comes in already stamped with a grandchild's id.
    pre_stamped = ProgressSignal[SampleMutation](
        mutations=[], sender_id="grandchild", timestamp_us=999
    )
    await channel.emit(pre_stamped)

    sender, fwd = grandparent.get_nowait()
    # The MergeInbox key is the emitting channel's agent_id — the
    # forwarder — regardless of who first stamped the envelope.
    assert sender == "child"
    # The envelope itself keeps its original stamp — no forwarder rewrite.
    assert fwd.sender_id == "grandchild"
    assert fwd.timestamp_us == 999


@pytest.mark.asyncio
async def test_channel_send_to_stamps_control_signal_with_parent_marker() -> None:
    """``send_to`` (parent → child) stamps ``sender_id="parent"`` if
    unset. The runner may override with a real id; this is the fallback
    that surfaces the parent-child direction in the audit trail."""
    channel = SignalChannel[Any, Any](agent_id="child")
    await channel.send_to(CancelSignal(reason="shutdown"))

    signal = channel.inbox.get_nowait()
    assert signal.sender_id == "parent"
    assert signal.timestamp_us > 0
    assert signal.reason == "shutdown"


@pytest.mark.asyncio
async def test_channel_try_send_to_full_inbox_drops_without_raising() -> None:
    """``try_send_to`` is best-effort: a full inbox is a legitimate
    signal to drop rather than block, and the return value tells the
    caller whether the send landed."""
    channel = SignalChannel[Any, Any](agent_id="child", buffer_size=1)
    assert channel.try_send_to(CancelSignal(reason="first")) is True
    # inbox is now full — try_send_to returns False, doesn't raise.
    assert channel.try_send_to(CancelSignal(reason="second")) is False


@pytest.mark.asyncio
async def test_two_channels_emit_independently() -> None:
    """Concurrent emit from two independent channels writes to
    independent outboxes — no cross-talk on stamps or ordering."""
    import asyncio

    ch_a = SignalChannel[Any, Any](agent_id="agent_a")
    ch_b = SignalChannel[Any, Any](agent_id="agent_b")

    async def emit_burst(ch, count):
        for _ in range(count):
            await ch.emit(ProgressSignal[SampleMutation](mutations=[]))

    await asyncio.gather(emit_burst(ch_a, 10), emit_burst(ch_b, 10))

    a_ids = []
    while not ch_a.outbox.empty():
        a_ids.append(ch_a.outbox.get_nowait().sender_id)
    b_ids = []
    while not ch_b.outbox.empty():
        b_ids.append(ch_b.outbox.get_nowait().sender_id)

    assert a_ids == ["agent_a"] * 10
    assert b_ids == ["agent_b"] * 10


@pytest.mark.asyncio
async def test_signal_channel_emit_stamps_observation() -> None:
    """A ``SignalChannel`` wired with an observer stamps a
    ``signal.emitted`` observation for every ``emit``, carrying the
    envelope's ``sender_id`` / ``topic`` / ``timestamp_us``. This is
    the audit seam that lets a runner surface the coordination cascade
    alongside LLM / tool / memory observations on the same stream.

    The observer only sees writes that landed on the outbox — a
    misbehaving observer never masks the coordination cascade because
    ``emit`` finishes the queue writes BEFORE calling the observer.
    """
    from agentkit.adapters.observer import CollectingObserver

    observer = CollectingObserver()
    channel = SignalChannel[Any, Any](agent_id="agent_a", observer=observer)

    await channel.emit(ProgressSignal[SampleMutation](mutations=[]))

    # Queue write still lands (observability doesn't gate the cascade).
    stamped = channel.outbox.get_nowait()
    assert stamped.sender_id == "agent_a"

    # Exactly one observation, carrying the coordinated shape.
    emitted = [o for o in observer.items if o.kind == "signal.emitted"]
    assert len(emitted) == 1
    obs = emitted[0]
    assert obs.payload["sender_id"] == "agent_a"
    assert obs.payload["topic"] == "ProgressSignal"
    assert obs.payload["timestamp_us"] == stamped.timestamp_us


@pytest.mark.asyncio
async def test_signal_channel_emit_survives_broken_observer() -> None:
    """A misbehaving observer must never break the coordination
    cascade — the outbox and merge-inbox writes above the observer
    call are the load-bearing invariant; the observer is best-effort."""

    class _BoomObserver:
        async def emit(self, obs: Any) -> None:  # noqa: ARG002
            raise RuntimeError("observer down")

        async def close(self) -> None:
            return None

    parent_merge: MergeInbox[DataSignal] = MergeInbox()
    channel = SignalChannel[Any, Any](agent_id="child", observer=_BoomObserver())
    channel.attach_parent(parent_merge)

    # Must not raise.
    await channel.emit(ProgressSignal[SampleMutation](mutations=[]))

    # Both writes still landed.
    assert not channel.outbox.empty()
    sender, _fwd = parent_merge.get_nowait()
    assert sender == "child"


def test_all_signal_types_are_frozen() -> None:
    """Every signal subclass carries the freeze — an app that
    subclasses one and stamps it in place would silently regress the
    concurrency invariant."""
    from dataclasses import FrozenInstanceError

    for signal in (
        CancelSignal(reason="r"),
        ProgressSignal[SampleMutation](mutations=[]),
        DoneSignal[SampleMutation](final_delta=[]),
    ):
        with pytest.raises(FrozenInstanceError):
            signal.sender_id = "clobber"  # type: ignore[misc]


# ── The audit tap must never wedge the run it audits ────────────────────────
#
# The documented concurrency model has the owning agent reading ``inbox`` and
# ``merge_inbox``; NOTHING reads ``outbox`` — it is the audit / replay tap.
# ``emit`` awaited it anyway, so a healthy run with a perfectly draining parent
# stopped dead once the tap filled: measured, the 9th emit on a
# ``buffer_size=8`` channel blocked forever, and at the default size an agent
# went silent after its 256th signal. That is a deadlock wearing backpressure's
# clothes.


@pytest.mark.asyncio
async def test_a_full_outbox_does_not_block_the_emitter() -> None:
    """THE regression. The parent drains everything; only the unread audit tap
    is full, and emitting must still make progress."""
    parent: MergeInbox[ProgressSignal] = MergeInbox(buffer_size=1024)
    channel: SignalChannel[CancelSignal, ProgressSignal] = SignalChannel(
        agent_id="worker", buffer_size=4
    )
    channel.attach_parent(parent)

    for i in range(12):
        await asyncio.wait_for(channel.emit(ProgressSignal(mutations=[i])), timeout=1.0)

    # Every signal reached the real consumer — dropping is the TAP's behaviour,
    # never the delivery path's.
    assert parent.qsize() == 12


@pytest.mark.asyncio
async def test_the_tap_keeps_the_most_recent_window() -> None:
    """Dropping the oldest keeps a diagnostic tap useful under load; dropping
    the newest would make it go blind exactly when a run gets busy."""
    channel: SignalChannel[CancelSignal, ProgressSignal] = SignalChannel(
        agent_id="worker", buffer_size=3
    )
    for i in range(10):
        await channel.emit(ProgressSignal(mutations=[i]))

    seen = []
    while not channel.outbox.empty():
        seen.append(channel.outbox.get_nowait().mutations[0])
    assert seen == [7, 8, 9]


@pytest.mark.asyncio
async def test_dropping_is_counted_not_silent() -> None:
    """A consumer reading the tap has to be able to tell a complete transcript
    from a window."""
    channel: SignalChannel[CancelSignal, ProgressSignal] = SignalChannel(
        agent_id="worker", buffer_size=3
    )
    for i in range(3):
        await channel.emit(ProgressSignal(mutations=[i]))
    assert channel.dropped == 0  # still complete

    for i in range(4):
        await channel.emit(ProgressSignal(mutations=[i]))
    assert channel.dropped == 4


@pytest.mark.asyncio
async def test_the_parent_still_applies_backpressure() -> None:
    """The negative control: making the tap non-blocking must NOT make the
    delivery path non-blocking. A slow parent still makes the child wait —
    that is the one place backpressure belongs."""
    parent: MergeInbox[ProgressSignal] = MergeInbox(buffer_size=2)
    channel: SignalChannel[CancelSignal, ProgressSignal] = SignalChannel(
        agent_id="worker", buffer_size=1024
    )
    channel.attach_parent(parent)

    await channel.emit(ProgressSignal(mutations=[0]))
    await channel.emit(ProgressSignal(mutations=[1]))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(channel.emit(ProgressSignal(mutations=[2])), timeout=0.05)
