"""Observation channel (ch23 / R12): emit via RunContext, collecting + bounded-queue observers,
the never-drop-results backpressure rule. Offline & deterministic."""

import asyncio
import dataclasses
import re

import pytest

from agentkit.adapters.observer import (
    CollectingObserver,
    PolicyObserver,
    QueueObserver,
    RollupObserver,
)
from agentkit.kernel.observation import NoopObserver, Observation, TraceContext
from agentkit.kernel.types import Scope
from agentkit.runtime.context import RunContext, Services


def run(coro):
    return asyncio.run(coro)


def test_default_observer_is_noop_and_emit_never_raises():
    async def go():
        ctx = RunContext("cid", Scope(1, 2))                 # no observer injected → NoopObserver
        assert isinstance(ctx.observer, NoopObserver)
        await ctx.emit("progress", "tick", payload={"i": 1})  # must not raise
    run(go())


def test_emit_through_collecting_observer_fills_run_id():
    async def go():
        obs = CollectingObserver()
        ctx = RunContext("run-7", Scope(1, 2), services=Services(observer=obs))
        await ctx.emit("summary", "wrote intro", payload={"words": 120}, agent="writer")
        assert len(obs.items) == 1
        o = obs.items[0]
        assert o.kind == "summary" and o.render == "wrote intro" and o.payload == {"words": 120}
        assert o.run_id == "run-7" and o.agent == "writer"
    run(go())


def test_observer_is_shared_to_children():
    async def go():
        obs = CollectingObserver()
        ctx = RunContext("cid", Scope(1, 2), services=Services(observer=obs))
        await ctx.child().emit("progress", "child tick")
        assert obs.items and obs.items[0].render == "child tick"
    run(go())


def test_queue_observer_streams_in_order_until_close():
    async def go():
        q = QueueObserver(maxsize=8)
        await q.emit(Observation(kind="progress", render="a"))
        await q.emit(Observation(kind="result", render="done"))
        await q.close()
        got = [o.render async for o in q.stream()]
        assert got == ["a", "done"]
    run(go())


def test_queue_observer_coalesces_progress_but_never_drops_results():
    async def go():
        q = QueueObserver(maxsize=2)
        # fill with progress, then overflow with more progress (coalesced — oldest dropped)
        for i in range(6):
            await q.emit(Observation(kind="progress", render=f"p{i}"))
        # a result must still get in (never dropped), even though the buffer was full
        await q.emit(Observation(kind="result", render="RESULT"))
        await q.close()
        got = [o.render async for o in q.stream()]
        assert "RESULT" in got                       # result preserved
        assert len([r for r in got if r.startswith("p")]) <= 2   # progress was bounded/coalesced
    run(go())


# ---- PolicyObserver: the emission-cadence knob (ch23 §23.2) ------------------------------------

def _emit_all(observer):
    async def go():
        for kind in ("progress", "step", "summary", "interrupt", "result"):
            await observer.emit(Observation(kind=kind, render=kind))
    run(go())


def test_policy_everything_forwards_all_kinds():
    sink = CollectingObserver()
    _emit_all(PolicyObserver.everything(sink))
    assert [o.kind for o in sink.items] == ["progress", "step", "summary", "interrupt", "result"]


def test_policy_summaries_keeps_summary_and_always_forwarded_drops_finer():
    sink = CollectingObserver()
    _emit_all(PolicyObserver.summaries(sink))
    assert [o.kind for o in sink.items] == ["progress", "summary", "interrupt", "result"]  # step dropped


def test_policy_result_only_keeps_critical_and_interrupt():
    sink = CollectingObserver()
    _emit_all(PolicyObserver.result_only(sink))
    assert [o.kind for o in sink.items] == ["interrupt", "result"]   # result/error/interrupt always pass


def test_policy_never_silences_a_result_even_when_not_allowed():
    sink = CollectingObserver()
    policy = PolicyObserver(sink, allow={"summary"})       # results not explicitly allowed
    run(policy.emit(Observation(kind="result", render="done")))
    assert [o.kind for o in sink.items] == ["result"]      # forwarded regardless of allow-set


# ---- RollupObserver: the rolled-up-summary cadence (ch23 §23.2) --------------------------------

def test_rollup_emits_one_summary_every_n():
    sink = CollectingObserver()
    roll = RollupObserver(sink, every=4)

    async def go():
        for i in range(8):
            await roll.emit(Observation(kind="progress", render=f"p{i}", run_id="r"))
    run(go())

    assert [o.kind for o in sink.items] == ["summary", "summary"]     # 8 progress → 2 roll-ups
    assert all(o.payload["rolled"] == 4 for o in sink.items)
    assert sink.items[0].run_id == "r"                                # correlation carried through


def test_rollup_flushes_before_a_critical_then_forwards_it():
    sink = CollectingObserver()
    roll = RollupObserver(sink, every=8)                              # threshold not reached by 3 items

    async def go():
        for i in range(3):
            await roll.emit(Observation(kind="progress", render=f"p{i}"))
        await roll.emit(Observation(kind="result", render="done"))    # flush the 3 buffered, then forward
    run(go())

    assert [o.kind for o in sink.items] == ["summary", "result"]
    assert sink.items[0].payload["rolled"] == 3


def test_rollup_close_flushes_tail_and_uses_custom_async_summarizer():
    sink = CollectingObserver()

    async def summarize(buf):
        return f"rolled {len(buf)}: " + ",".join(o.render for o in buf)

    roll = RollupObserver(sink, every=10, summarize=summarize)

    async def go():
        await roll.emit(Observation(kind="progress", render="a"))
        await roll.emit(Observation(kind="progress", render="b"))
        await roll.close()                                            # flush the tail of 2
    run(go())

    assert len(sink.items) == 1 and sink.items[0].render == "rolled 2: a,b"


# ---- the fields the record actually carries (dead-field ratchet) -------------------------------
#
# `Observation` used to declare `seq: int = 0` and `ts: float = 0.0`. Measured
# on this tree before the fix: `Observation(kind="tool_result", payload={}).seq`
# was 0 and `.ts` was 0.0, and NOTHING in the package ever assigned either one —
# not `RunContext.emit`, not `SignalChannel`, not `RollupObserver`, not any
# adapter — while `__hash__` folded both into the stream key and called `seq`
# "that emitter's monotonic counter". They were removed rather than populated
# (see the class docstring for why); these tests are the ratchet that stops a
# field with no producer from being added back.


class _FakeSpanTracer:
    """Minimal `TracePort`: enough for `emit` to attach a `trace_context` and
    mirror CRITICAL kinds as span events. Not a tracing test — this exists so
    the sweep below can exercise EVERY field on the record in one emit."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def span(self, name: str, kind: str, **attrs):  # pragma: no cover — unused here
        raise NotImplementedError

    def current_span_id(self) -> TraceContext:
        return TraceContext(trace_id="0" * 31 + "1", span_id="0" * 15 + "1")

    def add_event_to_current_span(self, name: str, **fields) -> None:
        self.events.append((name, fields))


def _one_fully_populated_emit() -> Observation:
    """Drive the real emit seam with every knob it exposes turned on."""
    obs = CollectingObserver()

    async def go():
        ctx = RunContext(
            "run-9",
            Scope(1, 2),
            services=Services(observer=obs, trace=_FakeSpanTracer()),
        )
        await ctx.emit(
            "result",
            "wrote intro",
            payload={"words": 120},
            agent="writer",
            parent_id="parent-1",
        )

    run(go())
    assert len(obs.items) == 1
    return obs.items[0]


def test_every_observation_field_is_populated_by_the_emit_seam():
    """THE BUG, as a ratchet. One `ctx.emit` with every argument supplied and a
    tracer attached must leave no field sitting at its class default — a field
    that survives this sweep is one no caller can ever see a real value in.

    Before the fix this failed with ``{'seq', 'ts'}``: a full-fat emit through
    the framework's own seam produced ``seq=0, ts=0.0``, the exact defaults, on
    a record whose ``__hash__`` claimed ``seq`` was a monotonic counter.

    Deliberately no exemption set. If a future field genuinely cannot be filled
    at emit time, that is the argument for not putting it on this record."""
    emitted = _one_fully_populated_emit()
    blank = {f.name: f.default for f in dataclasses.fields(Observation)}

    dead = {
        name
        for name, default in blank.items()
        if default is not dataclasses.MISSING and getattr(emitted, name) == default
    }
    assert dead == set(), f"fields the framework never populates: {sorted(dead)}"


def test_the_removed_fields_are_gone_rather_than_defaulted():
    """Removed, not quietly defaulted to something else: constructing with them
    is a `TypeError`, so a caller carrying `seq=`/`ts=` forward gets told,
    instead of setting an attribute nothing reads."""
    o = Observation(kind="progress")
    assert not hasattr(o, "seq") and not hasattr(o, "ts")
    for dead in ("seq", "ts"):
        with pytest.raises(TypeError):
            Observation(kind="progress", **{dead: 1})


def test_the_hash_docstring_names_exactly_the_fields_the_hash_reads():
    """The docstring and the tuple must agree — the original defect was a
    docstring describing a counter the code never incremented. Probe which
    fields actually move the hash, then hold the prose to it."""
    documented = re.search(r"STREAM key — ``\(([^)]+)\)``", Observation.__hash__.__doc__)
    assert documented is not None, "the hash docstring must state its stream key"
    named = {f.strip() for f in documented.group(1).split(",")}

    base = Observation(kind="progress", agent="a", render="r", run_id="r1", payload={"k": 1})
    probes = {
        "kind": "summary",
        "agent": "b",
        "render": "other",
        "run_id": "r2",
        "payload": {"k": 2},
        "parent_id": "p",
        "trace_context": TraceContext(trace_id="f" * 32, span_id="f" * 16),
    }
    reads = {
        name: hash(dataclasses.replace(base, **{name: value})) != hash(base)
        for name, value in probes.items()
    }
    assert {n for n, moved in reads.items() if moved} == named


def test_a_real_runs_stream_keys_are_no_coarser_than_before_the_removal():
    """POSITIVE CONTROL for the removal's headline claim, passing both ways.

    `seq`/`ts` were constants on every record a real run produced, so folding
    them into the key bought exactly zero discrimination. 1000 progress
    observations from one agent in one run collapse to a single bucket with the
    fields and without them; `__eq__` is what separates them, which is what
    keeps a `set`-based dedup of a replayed stream exact."""
    stream = [
        Observation(kind="progress", render=f"step {i}", run_id="run-1",
                    agent="writer", payload={"i": i})
        for i in range(1000)
    ]
    assert len({hash(o) for o in stream}) == 1          # one bucket, as before the removal
    assert len(set(stream)) == 1000                     # …and dedup is still exact
    assert len({(o.run_id, o.agent, o.kind) for o in stream}) == 1
