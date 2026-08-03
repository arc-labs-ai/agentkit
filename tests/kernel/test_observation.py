"""Observation channel (ch23 / R12): emit via RunContext, collecting + bounded-queue observers,
the never-drop-results backpressure rule. Offline & deterministic."""

import asyncio

from agentkit.adapters.observer import (
    CollectingObserver,
    PolicyObserver,
    QueueObserver,
    RollupObserver,
)
from agentkit.kernel.observation import NoopObserver, Observation
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
