"""``Workflow.map`` — a node whose fan-out width is decided at RUNTIME.

Every other ``Workflow`` builder authors a node, so the graph is a fact about the source.
``map`` authors ONE node that expands into N element runs when it executes, where N comes
from data a previous node just produced. The tests here pin the three properties that make
that safe rather than merely convenient:

* it reuses the spine — ``gather_bounded`` + ``ctx.semaphore()`` for concurrency (per DEPTH,
  so a nested fan-out cannot deadlock), ``ctx.check_cancelled()`` between elements;
* the expansion is RECORDED in ``done`` (element keys + an identity list), not just its
  results, so a resume can tell which elements finished;
* the expansion is deterministic given the same inputs, and a drift is a loud error rather
  than a silently mis-threaded output.

Offline & deterministic: FakeLLM + asyncio.run, no sleeps used as synchronisation.
"""

import asyncio
import json

import pytest

from agentkit.agents import Agent, Workflow
from agentkit.agents.workflow import MapExpansionChanged
from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.errors import Failure
from agentkit.kernel.types import Scope
from agentkit.runtime.meter import Budget
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def _ctx(**kw):
    kw.setdefault("llm", FakeLLM("X"))
    kw.setdefault("scope", Scope(1, 2))
    kw.setdefault("correlation_id", "wf-map")
    return make_test_ctx(**kw)


def _first_cause(eg):
    """A node failure surfaces through the wave's TaskGroup as an ExceptionGroup."""
    while isinstance(eg, BaseExceptionGroup):
        eg = eg.exceptions[0]
    return eg


class _JsonStore:
    """StorePort that round-trips through JSON — proves the expansion record survives a
    *real* (serializing) store, not just ``InMemoryStore``'s live objects."""

    def __init__(self) -> None:
        self._d: dict = {}

    async def get(self, key):
        raw = self._d.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key, value):
        self._d[key] = json.dumps(value)

    async def delete(self, key):
        self._d.pop(key, None)


# ── the core contract ────────────────────────────────────────────────────────


def test_map_over_three_elements_runs_three_and_threads_them_to_the_dependent():
    wf = Workflow("plan-then-execute")
    wf.fn("plan", lambda inp: ["a", "b", "c"])
    wf.map("implement", over=lambda inp: inp["plan"], each=lambda item: item.upper(), after="plan")
    wf.fn("ship", lambda inp: "+".join(inp["implement"]), after="implement")

    res = _run(wf.run("g", _ctx()))

    assert res.stop_reason == "complete"
    assert res.outputs["implement"] == ["A", "B", "C"]
    assert res.outputs["ship"] == "A+B+C"
    # The expansion is RECORDED, not merely its aggregate result.
    assert res.outputs["implement[0]"] == "A"
    assert res.outputs["implement[2]"] == "C"
    assert list(res.outputs["implement#expansion"]) == ["a", "b", "c"]
    # A map is ONE node, so it costs ONE step — a 500-element map must not
    # consume the max_steps backstop.
    assert res.steps == 3


def test_map_counts_as_one_step_regardless_of_width():
    wf = Workflow(max_steps=2)
    wf.fn("plan", lambda inp: list(range(50)))
    wf.map("work", over=lambda inp: inp["plan"], each=lambda item: item * 2, after="plan")
    res = _run(wf.run("g", _ctx()))
    assert res.stop_reason == "complete" and res.steps == 2


def test_map_element_can_be_an_agent_and_usage_merges():
    wf = Workflow()
    wf.fn("plan", lambda inp: ["one", "two"])
    wf.map(
        "impl",
        over=lambda inp: inp["plan"],
        each=lambda item: Agent(f"dev-{item}", "m"),
        after="plan",
    )
    res = _run(wf.run("build", _ctx(llm=FakeLLM("DONE"))))
    assert res.outputs["impl"] == ["DONE", "DONE"]
    assert res.usage.input_tokens == 20  # both element agents merged into the node's usage


def test_map_each_may_take_the_index():
    wf = Workflow()
    wf.fn("plan", lambda inp: ["a", "b"])
    wf.map("m", over=lambda inp: inp["plan"], each=lambda item, i: f"{i}:{item}", after="plan")
    res = _run(wf.run("g", _ctx()))
    assert res.outputs["m"] == ["0:a", "1:b"]


def test_map_over_may_take_the_goal():
    wf = Workflow()
    wf.fn("plan", lambda inp: 2)
    wf.map("m", over=lambda inp, goal: [goal] * inp["plan"], each=lambda item: item, after="plan")
    res = _run(wf.run("GOAL", _ctx()))
    assert res.outputs["m"] == ["GOAL", "GOAL"]


# ── concurrency: bounded by the semaphore, and provably overlapping ──────────


def test_map_wave_provably_overlaps_rather_than_serialising():
    """Three elements must be in flight AT THE SAME TIME.

    Proved by construction rather than by timing: every element blocks until the
    in-flight count reaches three. A serialising implementation never reaches three,
    so the ``wait_for`` expires and the test fails instead of passing on a sleep.
    """
    state = {"inflight": 0, "peak": 0}
    all_three = asyncio.Event()

    async def work(item):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        if state["inflight"] >= 3:
            all_three.set()
        await asyncio.wait_for(all_three.wait(), 5.0)
        state["inflight"] -= 1
        return item

    wf = Workflow()
    wf.fn("plan", lambda inp: [1, 2, 3])
    wf.map("m", over=lambda inp: inp["plan"], each=lambda item: work(item), after="plan")
    res = _run(wf.run("g", _ctx()))

    assert res.outputs["m"] == [1, 2, 3]
    assert state["peak"] == 3


def test_map_bounded_by_caps_the_in_flight_width():
    """``bounded_by=2`` over four elements: exactly two overlap, never three.

    Two measurements, because either alone is a test that passes for the wrong
    reason. The event is the FLOOR — a serialising map never reaches two in flight,
    so ``wait_for`` expires. The yield loop is the CEILING — every element stays
    parked for ten scheduler turns after the barrier clears, which is every chance
    an unbounded implementation needs to start the other two (measured: ``peak``
    reaches 4 when the width bound is dropped).
    """
    state = {"inflight": 0, "peak": 0}
    both = asyncio.Event()

    async def work(item):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        if state["inflight"] >= 2:
            both.set()
        await asyncio.wait_for(both.wait(), 5.0)
        for _ in range(10):
            await asyncio.sleep(0)
        state["inflight"] -= 1
        return item

    wf = Workflow()
    wf.fn("plan", lambda inp: [1, 2, 3, 4])
    wf.map("m", over=lambda inp: inp["plan"], each=lambda i: work(i), after="plan", bounded_by=2)
    res = _run(wf.run("g", _ctx()))

    assert res.outputs["m"] == [1, 2, 3, 4]
    assert state["peak"] == 2


def test_map_respects_the_level_semaphore_even_without_bounded_by():
    """No ``bounded_by`` still means bounded: the tree semaphore for this level."""
    state = {"inflight": 0, "peak": 0}

    async def work(item):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        for _ in range(10):
            await asyncio.sleep(0)
        state["inflight"] -= 1
        return item

    wf = Workflow()
    wf.fn("plan", lambda inp: list(range(10)))
    wf.map("m", over=lambda inp: inp["plan"], each=lambda i: work(i), after="plan")
    res = _run(wf.run("g", _ctx(budget=Budget(max_concurrency=3))))

    assert res.outputs["m"] == list(range(10))
    assert state["peak"] == 3  # the tree semaphore, not an unbounded gather


def test_nested_map_does_not_deadlock_on_a_single_permit_pool():
    """A map inside a map with ``max_concurrency=1``.

    ``ctx.semaphore()`` is keyed on depth; a single tree-wide pool would wedge here,
    because the outer expansion holds every permit while the inner one waits for it.
    """
    inner = Workflow("inner")
    inner.map("leaf", over=lambda inp: [1, 2], each=lambda x: x * 10)

    wf = Workflow("outer")
    wf.fn("plan", lambda inp: ["p", "q"])
    wf.map(
        "outer",
        over=lambda inp: inp["plan"],
        each=lambda item: inner,
        after="plan",
    )
    res = _run(wf.run("g", _ctx(budget=Budget(max_concurrency=1))))
    assert res.stop_reason == "complete"
    assert res.outputs["outer"][0]["leaf"] == [10, 20]


def test_map_emits_its_runtime_width():
    """The expansion is the one thing about the run a reader cannot get from the source."""
    from agentkit.adapters.observer import CollectingObserver

    obs = CollectingObserver()
    wf = Workflow()
    wf.fn("plan", lambda inp: ["a", "b", "c"])
    wf.map("m", over=lambda inp: inp["plan"], each=lambda i: i, after="plan")
    _run(wf.run("g", _ctx(observer=obs)))

    widths = [o.payload for o in obs.items if o.payload and o.payload.get("node") == "m"]
    assert widths == [{"node": "m", "elements": 3}]


# ── failure semantics ────────────────────────────────────────────────────────


def test_map_element_failure_fails_the_node_like_any_other():
    def work(item):
        if item == "b":
            raise ValueError("element b exploded")
        return item

    wf = Workflow()
    wf.fn("plan", lambda inp: ["a", "b"])
    wf.map("m", over=lambda inp: inp["plan"], each=lambda i: work(i), after="plan")
    wf.fn("never", lambda inp: "ran", after="m")

    with pytest.raises(BaseException) as ei:
        _run(wf.run("g", _ctx()))
    assert isinstance(_first_cause(ei.value), ValueError)


def test_map_best_effort_isolates_a_failed_element():
    def work(item):
        if item == "b":
            raise ValueError("element b exploded")
        return item.upper()

    wf = Workflow()
    wf.fn("plan", lambda inp: ["a", "b", "c"])
    wf.map(
        "m",
        over=lambda inp: inp["plan"],
        each=lambda i: work(i),
        after="plan",
        best_effort=True,
    )
    res = _run(wf.run("g", _ctx()))

    out = res.outputs["m"]
    assert out[0] == "A" and out[2] == "C"
    assert isinstance(out[1], Failure) and isinstance(out[1].cause, ValueError)
    # A failed element is NOT recorded as finished — a resume must retry it.
    assert "m[1]" not in res.outputs
    assert res.outputs["m[0]"] == "A"


def test_map_over_raising_surfaces_the_error():
    def boom(inp):
        raise KeyError("no such input")

    wf = Workflow()
    wf.fn("plan", lambda inp: [1])
    wf.map("m", over=boom, each=lambda i: i, after="plan")
    with pytest.raises(BaseException) as ei:
        _run(wf.run("g", _ctx()))
    assert isinstance(_first_cause(ei.value), KeyError)


# ── resume across a dynamically sized node ───────────────────────────────────


def _resumable_wf(work):
    wf = Workflow("plan-then-execute")
    wf.fn("plan", lambda inp: ["a", "b", "c"])
    wf.map(
        "impl",
        over=lambda inp: inp["plan"],
        each=lambda item: work(item),
        after="plan",
        bounded_by=1,  # sequential, so "the first two finished" is deterministic
    )
    wf.fn("ship", lambda inp: "+".join(inp["impl"]), after="impl")
    return wf


def test_map_resume_after_two_of_three_runs_only_the_third():
    seen: list[str] = []
    fail = {"on": True}

    async def work(item):
        seen.append(item)
        if item == "c" and fail["on"]:
            raise RuntimeError("c is flaky")
        return item.upper()

    store = _JsonStore()
    with pytest.raises(BaseException) as ei:
        _run(_resumable_wf(work).run("g", _ctx(store=store)))
    assert isinstance(_first_cause(ei.value), RuntimeError)
    assert seen == ["a", "b", "c"]

    fail["on"] = False
    seen.clear()
    res = _run(_resumable_wf(work).resume("wf-map", {}, _ctx(store=store)))

    assert res.stop_reason == "complete"
    assert seen == ["c"]  # ONLY the third element re-ran
    assert res.outputs["impl"] == ["A", "B", "C"]
    assert res.outputs["ship"] == "A+B+C"


def test_map_resume_raises_when_the_expansion_drifted():
    """A reordered ``over`` cannot be resumed against positional element keys —
    reusing ``impl[0]`` would thread element *a*'s output into element *c*'s slot."""
    order = {"n": 0}
    fail = {"on": True}

    async def work(item):
        if item == "c" and fail["on"]:
            raise RuntimeError("c is flaky")
        return item.upper()

    def _wf():
        wf = Workflow()
        wf.fn("plan", lambda inp: None)

        def over(inp):
            order["n"] += 1
            return ["a", "b", "c"] if order["n"] == 1 else ["c", "b", "a"]

        wf.map("impl", over=over, each=lambda i: work(i), after="plan", bounded_by=1)
        return wf

    store = _JsonStore()
    with pytest.raises(BaseException):
        _run(_wf().run("g", _ctx(store=store)))

    fail["on"] = False
    with pytest.raises(BaseException) as ei:
        _run(_wf().resume("wf-map", {}, _ctx(store=store)))
    cause = _first_cause(ei.value)
    assert isinstance(cause, MapExpansionChanged)
    assert "impl" in str(cause)


def test_map_key_hook_pins_identity_to_a_stable_field():
    """``key=`` is the escape hatch for an ``over`` that cannot be made byte-identical.

    Here every expansion stamps a fresh nonce onto each item, so the default
    ``str(item)`` identity would read as drift and refuse the resume (the negative
    control is ``test_map_resume_raises_when_the_expansion_drifted``). Keying on the
    stable ``id`` field makes the same run resumable.
    """
    nonce = {"n": 0}
    fail = {"on": True}
    seen: list[str] = []

    def over(inp):
        nonce["n"] += 1
        return [{"id": i, "nonce": nonce["n"]} for i in ("a", "b", "c")]

    async def work(item):
        seen.append(item["id"])
        if item["id"] == "c" and fail["on"]:
            raise RuntimeError("flaky")
        return item["id"].upper()

    def _wf():
        wf = Workflow()
        wf.map(
            "impl",
            over=over,
            each=lambda i: work(i),
            bounded_by=1,
            key=lambda item: item["id"],
        )
        return wf

    store = _JsonStore()
    with pytest.raises(BaseException):
        _run(_wf().run("g", _ctx(store=store)))
    assert seen == ["a", "b", "c"]

    fail["on"] = False
    seen.clear()
    res = _run(_wf().resume("wf-map", {}, _ctx(store=store)))
    assert res.stop_reason == "complete" and seen == ["c"]
    assert res.outputs["impl"] == ["A", "B", "C"]
    assert list(res.outputs["impl#expansion"]) == ["a", "b", "c"]


def test_map_output_survives_a_gate_through_a_serializing_store():
    def _wf():
        wf = Workflow()
        wf.fn("plan", lambda inp: ["a", "b"])
        wf.map("impl", over=lambda inp: inp["plan"], each=lambda i: i.upper(), after="plan")
        wf.human_gate("review", after="impl")
        wf.fn("ship", lambda inp: "|".join(inp["impl"]), after=["impl", "review"])
        return wf

    store = _JsonStore()
    res = _run(_wf().run("g", _ctx(store=store)))
    assert res.stop_reason == "suspended"

    res2 = _run(_wf().resume(res.suspended.run_id, {"review": "ok"}, _ctx(store=store)))
    assert res2.stop_reason == "complete" and res2.outputs["ship"] == "A|B"


# ── cancellation ─────────────────────────────────────────────────────────────


def test_map_cancellation_mid_expansion_keeps_partial_results_and_runs_finally():
    token = CancellationToken()
    finallys: list[str] = []

    async def work(item):
        try:
            if item == "b":
                token.cancel()  # abort lands on element "c"
            return item.upper()
        finally:
            finallys.append(item)

    wf = Workflow()
    wf.fn("plan", lambda inp: ["a", "b", "c"])
    wf.map(
        "impl",
        over=lambda inp: inp["plan"],
        each=lambda i: work(i),
        after="plan",
        bounded_by=1,
    )
    store = _JsonStore()
    with pytest.raises(BaseException) as ei:
        _run(wf.run("g", _ctx(cancel=token, store=store)))
    assert isinstance(_first_cause(ei.value), Cancelled)
    assert finallys == ["a", "b"]  # cleanup ran for the elements that started

    # The partial expansion was checkpointed — a resume can pick up at "c".
    saved = _run(store.get("checkpoint:wf-map"))
    assert saved is not None


# ── the shapes that must not hang or silently misbehave ──────────────────────


def test_map_over_an_empty_collection_completes():
    wf = Workflow()
    wf.fn("plan", lambda inp: [])
    wf.map("m", over=lambda inp: inp["plan"], each=lambda i: i, after="plan")
    wf.fn("after", lambda inp: f"saw {len(inp['m'])}", after="m")
    res = _run(wf.run("g", _ctx()))
    assert res.stop_reason == "complete"
    assert res.outputs["m"] == [] and res.outputs["after"] == "saw 0"
    assert list(res.outputs["m#expansion"]) == []


def test_map_with_one_element():
    wf = Workflow()
    wf.map("m", over=lambda inp: ["solo"], each=lambda i: i.upper())
    res = _run(wf.run("g", _ctx()))
    assert res.outputs["m"] == ["SOLO"]


def test_map_with_five_hundred_elements():
    wf = Workflow()
    wf.fn("plan", lambda inp: list(range(500)))
    wf.map("m", over=lambda inp: inp["plan"], each=lambda i: i + 1, after="plan")
    res = _run(wf.run("g", _ctx()))
    assert res.outputs["m"][:3] == [1, 2, 3] and len(res.outputs["m"]) == 500


def test_map_over_a_generator_is_consumed_exactly_once():
    yielded = {"n": 0}

    def gen():
        for x in ["a", "b"]:
            yielded["n"] += 1
            yield x

    wf = Workflow()
    wf.map("m", over=lambda inp: gen(), each=lambda i: i.upper())
    res = _run(wf.run("g", _ctx()))
    assert res.outputs["m"] == ["A", "B"]
    assert yielded["n"] == 2  # materialised once, not re-iterated into emptiness


def test_map_over_a_map_expands_the_second_from_the_first():
    wf = Workflow()
    wf.map("first", over=lambda inp: [1, 2], each=lambda i: [i, i * 10])
    wf.map(
        "second",
        over=lambda inp: [x for pair in inp["first"] for x in pair],
        each=lambda i: i + 1,
        after="first",
    )
    res = _run(wf.run("g", _ctx()))
    assert res.outputs["first"] == [[1, 10], [2, 20]]
    assert res.outputs["second"] == [2, 11, 3, 21]


# ── construction-time guards ─────────────────────────────────────────────────


@pytest.mark.parametrize("width", [0, -1])
def test_map_bounded_by_must_be_positive(width):
    wf = Workflow()
    with pytest.raises(ValueError, match="bounded_by"):
        wf.map("m", over=lambda inp: [], each=lambda i: i, bounded_by=width)


def test_map_reserved_element_name_collision_is_refused_in_both_orders():
    wf = Workflow()
    wf.map("impl", over=lambda inp: [], each=lambda i: i)
    with pytest.raises(ValueError, match="impl"):
        wf.fn("impl[0]", lambda inp: 1)  # would collide with an element key
    with pytest.raises(ValueError, match="impl"):
        wf.fn("impl#expansion", lambda inp: 1)  # would collide with the record

    wf2 = Workflow()
    wf2.fn("impl[0]", lambda inp: 1)
    with pytest.raises(ValueError, match="impl"):
        wf2.map("impl", over=lambda inp: [], each=lambda i: i)

    # A map can be the VICTIM as well as the aggressor.
    wf3 = Workflow()
    wf3.map("impl", over=lambda inp: [], each=lambda i: i)
    with pytest.raises(ValueError, match="impl"):
        wf3.map("impl[0]", over=lambda inp: [], each=lambda i: i)


def test_bracketed_node_name_is_fine_when_no_map_owns_it():
    """POSITIVE CONTROL: the guard keys on an actual map node, not on the spelling.
    A graph with a node literally called ``a[0]`` and no map ``a`` is untouched."""
    wf = Workflow()
    wf.fn("a[0]", lambda inp: "ok")
    wf.fn("b#expansion", lambda inp: "also ok", after="a[0]")
    res = _run(wf.run("g", _ctx()))
    assert res.outputs["a[0]"] == "ok" and res.outputs["b#expansion"] == "also ok"


# ── routes ───────────────────────────────────────────────────────────────────


def test_route_can_target_a_map_node_and_the_loop_back_re_expands():
    """A loop-back onto a map must clear its element records AND its expansion record.

    The collection SHRINKS on each pass, so a leftover record is not a cosmetic
    smudge: the re-expansion no longer matches what was recorded and the run dies on
    ``MapExpansionChanged`` — or, for an unchanged collection, silently reuses every
    element result and loops without recomputing anything.
    """
    rounds = {"n": 0}

    def plan(inp):
        rounds["n"] += 1
        return ["x"] * (4 - rounds["n"])  # 3, then 2, then 1

    wf = Workflow(max_steps=20)
    wf.fn("plan", plan)
    wf.map("work", over=lambda inp: inp["plan"], each=lambda i: i, after="plan")
    wf.fn("check", lambda inp: len(inp["work"]) > 1, after="work")
    wf.route("check", when=lambda o: o, to="plan")

    res = _run(wf.run("g", _ctx()))
    assert res.stop_reason == "complete"
    assert res.outputs["work"] == ["x"]
    assert list(res.outputs["work#expansion"]) == ["x"]
    # The wider expansions left nothing behind.
    assert "work[1]" not in res.outputs and "work[2]" not in res.outputs


def test_route_from_a_map_reads_the_expanded_list():
    wf = Workflow(max_steps=10)
    wf.map("m", over=lambda inp: [1, 2], each=lambda i: i)
    wf.fn("tail", lambda inp: "done", after="m")
    wf.route("m", when=lambda out: sum(out) > 100, to="tail")  # never fires
    res = _run(wf.run("g", _ctx()))
    assert res.stop_reason == "complete" and res.outputs["tail"] == "done"


# ── review additions: gaps the first pass left open ──────────────────────────


def test_map_bounded_by_does_not_escape_the_level_semaphore():
    """``bounded_by`` is an EXTRA bound, never a replacement for the tree's.

    The other ``bounded_by`` test uses a width BELOW ``max_concurrency``, so it passes
    identically whether or not each element also takes a level permit — meaning a map
    could quietly opt out of the run's concurrency ceiling and nothing would notice.
    Here ``bounded_by`` (10) is WIDER than ``max_concurrency`` (3), so the effective
    width is the level pool's; dropping the inner ``async with level_sem`` drives
    ``peak`` to 10.
    """
    state = {"inflight": 0, "peak": 0}

    async def work(item):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        for _ in range(10):  # every chance an unbounded map needs to start the rest
            await asyncio.sleep(0)
        state["inflight"] -= 1
        return item

    wf = Workflow()
    wf.map("m", over=lambda inp: list(range(10)), each=lambda i: work(i), bounded_by=10)
    res = _run(wf.run("g", _ctx(budget=Budget(max_concurrency=3))))

    assert res.outputs["m"] == list(range(10))
    assert state["peak"] == 3  # min(bounded_by, max_concurrency), not bounded_by


class _Recorder:
    """Minimal runnable — ``run(task, ctx)`` — so the task string is directly assertable."""

    def __init__(self, seen: list[str]) -> None:
        self.seen = seen

    async def run(self, task, ctx):
        self.seen.append(task)

        class _R:
            output = task
            usage = None

        return _R()


def test_map_prompt_builds_the_task_for_a_runnable_element():
    seen: list[str] = []
    wf = Workflow()
    wf.map(
        "m",
        over=lambda inp: ["a", "b"],
        each=lambda item: _Recorder(seen),
        prompt=lambda item, goal: f"{goal}::{item}",
    )
    res = _run(wf.run("GOAL", _ctx()))
    assert seen == ["GOAL::a", "GOAL::b"]
    assert res.outputs["m"] == ["GOAL::a", "GOAL::b"]


def test_map_without_prompt_falls_back_to_the_default_task():
    """POSITIVE CONTROL for the test above — the default still threads goal + item."""
    seen: list[str] = []
    wf = Workflow()
    wf.map("m", over=lambda inp: ["a"], each=lambda item: _Recorder(seen))
    _run(wf.run("GOAL", _ctx()))
    assert seen == ["GOAL\n\n[item] a"]


def test_map_prompt_with_a_non_runnable_element_is_refused_not_dropped():
    """A ``prompt=`` the framework cannot deliver must not be silently discarded.

    Only the runnable shape has anywhere to put a task string; the other three are
    handed the item. Dropping the prompt ran the element with a task its author never
    wrote and reported SUCCESS — the failure mode this raise converts into a loud one.
    """
    wf = Workflow()
    wf.map(
        "m",
        over=lambda inp: ["a"],
        each=lambda item: item.upper(),  # plain data, not a runnable
        prompt=lambda item, goal: "never delivered",
    )
    with pytest.raises(BaseException) as ei:
        _run(wf.run("g", _ctx()))
    cause = _first_cause(ei.value)
    assert isinstance(cause, ValueError) and "prompt=" in str(cause)


def test_map_long_identities_are_capped_but_stay_distinguishable():
    """The identity record is capped so it cannot bloat every checkpoint — and the
    digest tail is what stops the cap from WEAKENING the drift guard.

    Two items sharing a 120-character prefix must not collapse onto one identity: if
    they did, a re-expansion that swapped them would sail through the guard and thread
    one element's output into the other's slot. Truncating without the digest passes
    every other test in this file.
    """
    prefix = "p" * 200
    a, b = prefix + "AAA", prefix + "BBB"

    wf = Workflow()
    wf.map("m", over=lambda inp: [a, b], each=lambda i: i[-3:])
    res = _run(wf.run("g", _ctx()))

    ids = list(res.outputs["m#expansion"])
    assert ids[0] != ids[1]  # the whole point of the digest
    assert all(len(i) < len(a) for i in ids)  # and it is genuinely capped

    # …and the drift guard still fires when two such items are swapped.
    order = {"n": 0}
    fail = {"on": True}

    async def work(item):
        if item.endswith("BBB") and fail["on"]:
            raise RuntimeError("flaky")
        return item[-3:]

    def _wf():
        w = Workflow()

        def over(inp):
            order["n"] += 1
            return [a, b] if order["n"] == 1 else [b, a]

        w.map("m", over=over, each=lambda i: work(i), bounded_by=1)
        return w

    store = _JsonStore()
    with pytest.raises(BaseException):
        _run(_wf().run("g", _ctx(store=store)))
    fail["on"] = False
    with pytest.raises(BaseException) as ei:
        _run(_wf().resume("wf-map", {}, _ctx(store=store)))
    assert isinstance(_first_cause(ei.value), MapExpansionChanged)


def test_map_checkpointer_failure_does_not_replace_the_element_failure():
    """Recording partial progress is best-effort; the caller's error is not.

    A failing map snapshots ``{goal, done, steps}`` on its way out. If that write itself
    raises — full disk, wedged Redis — the run must still surface the exception the
    ELEMENT raised. Letting the storage error win replaces a diagnosable
    ``RuntimeError('c is flaky')`` with an unrelated one, which is how an incident gets
    misrouted.
    """

    class _BrokenStore(_JsonStore):
        async def set(self, key, value):
            raise OSError("disk full")

    async def work(item):
        if item == "c":
            raise RuntimeError("c is flaky")
        return item.upper()

    with pytest.raises(BaseException) as ei:
        _run(_resumable_wf(work).run("g", _ctx(store=_BrokenStore())))
    assert isinstance(_first_cause(ei.value), RuntimeError)


def test_two_maps_in_one_wave_resume_skips_every_element_that_finished():
    """The partial record is the SHARED ``done``, so a failing map rescues its sibling's
    elements too — including the sibling the wave's TaskGroup cancelled on the way out.

    Without it, the well-behaved map in a mixed wave re-runs all N elements on resume
    even though every one of them had already completed and been recorded.
    """
    ran: dict[str, list[int]] = {"a": [], "b": []}
    fail = {"on": True}

    def _wf(ma_finished):
        wf = Workflow()
        wf.fn("plan", lambda inp: [1, 2, 3])

        async def wa(x):
            ran["a"].append(x)
            if x == 3:
                ma_finished.set()
            return x

        async def wb(x):
            ran["b"].append(x)
            if x == 3 and fail["on"]:
                # Ordered by construction, not by scheduler luck: ``mb`` cannot fail
                # until ``ma`` has finished every element, so "the sibling re-ran
                # nothing" is a property of the recording, not of interleaving.
                await asyncio.wait_for(ma_finished.wait(), 5.0)
                raise RuntimeError("b3 boom")
            return x

        wf.map("ma", over=lambda i: i["plan"], each=lambda x: wa(x), after="plan", bounded_by=1)
        wf.map("mb", over=lambda i: i["plan"], each=lambda x: wb(x), after="plan", bounded_by=1)
        wf.fn("tail", lambda i: (i["ma"], i["mb"]), after=["ma", "mb"])
        return wf

    store = _JsonStore()

    async def _first():
        return await _wf(asyncio.Event()).run("g", _ctx(store=store))

    with pytest.raises(BaseException) as ei:
        _run(_first())
    assert isinstance(_first_cause(ei.value), RuntimeError)
    assert ran["a"] == [1, 2, 3] and ran["b"] == [1, 2, 3]

    fail["on"] = False
    ran["a"].clear()
    ran["b"].clear()

    async def _second():
        return await _wf(asyncio.Event()).resume("wf-map", {}, _ctx(store=store))

    res = _run(_second())

    assert res.stop_reason == "complete"
    assert ran["a"] == []  # the sibling map re-ran NOTHING
    assert ran["b"] == [3]  # only the element that never finished
    assert res.outputs["tail"] == ([1, 2, 3], [1, 2, 3])
