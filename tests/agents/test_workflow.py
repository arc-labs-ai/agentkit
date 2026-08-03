"""Workflow (ch16 / R9): the explicit-control graph engine — DAG scheduling, typed data on edges,
conditional loop-back (bounded by max_steps), human-gate suspend/resume, and the shared spine
(cancellation, usage). Offline & deterministic via FakeLLM + asyncio.run."""

import asyncio
import json

import pytest

from agentkit.adapters.store import FileStore, InMemoryStore
from agentkit.agents import Agent, Workflow
from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def test_linear_dag_threads_typed_outputs():
    wf = Workflow()
    wf.agent("design", Agent("architect", "m"))
    wf.fn("doc", lambda inp: f"doc of {inp['design']}", after="design")
    res = _run(
        wf.run(
            "build", make_test_ctx(llm=FakeLLM("PLAN"), scope=Scope(1, 2), correlation_id="wf-1")
        )
    )
    assert res.stop_reason == "complete"
    assert res.outputs["design"] == "PLAN"
    assert res.outputs["doc"] == "doc of PLAN"  # design's output fed into doc


def test_independent_nodes_run_in_one_wave_and_merge_usage():
    wf = Workflow()
    wf.agent("a", Agent("a", "m"))
    wf.agent("b", Agent("b", "m"))
    wf.fn("join", lambda inp: sorted(inp.values()), after=["a", "b"])
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("X"), scope=Scope(1, 2), correlation_id="wf-1"))
    )
    assert res.outputs["join"] == ["X", "X"]  # join saw both upstreams
    assert res.usage.input_tokens == 20  # both agent calls merged (10 + 10)


def test_conditional_loop_back_reruns_until_condition_clears():
    attempts = {"n": 0}

    def attempt(inp):
        attempts["n"] += 1
        return attempts["n"]

    wf = Workflow(max_steps=20)
    wf.fn("attempt", attempt)
    wf.fn("check", lambda inp: {"failed": inp["attempt"] < 3}, after="attempt")
    wf.route("check", when=lambda o: o["failed"], to="attempt")  # bounded loop-back
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-1"))
    )
    assert res.stop_reason == "complete"
    assert attempts["n"] == 3 and res.outputs["attempt"] == 3


def test_max_steps_guards_an_endless_cycle():
    wf = Workflow(max_steps=4)
    wf.fn("attempt", lambda inp: 1)
    wf.fn("check", lambda inp: True, after="attempt")
    wf.route("check", when=lambda o: o, to="attempt")  # always loops
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-1"))
    )
    assert res.stop_reason == "max_steps"


def test_human_gate_suspends_then_resume_completes():
    def _wf():
        wf = Workflow()
        wf.agent("design", Agent("d", "m"))
        wf.human_gate("review", after="design")
        wf.fn("ship", lambda inp: f"shipping with {inp['review']}", after="review")
        return wf

    store = InMemoryStore()
    res = _run(
        _wf().run(
            "build",
            make_test_ctx(
                llm=FakeLLM("PLAN"), store=store, scope=Scope(1, 2), correlation_id="wf-1"
            ),
        )
    )
    assert res.stop_reason == "suspended" and res.suspended.pending == ("review",)
    assert "ship" not in res.outputs  # downstream did not run

    res2 = _run(
        _wf().resume(
            res.suspended.run_id,
            {"review": "approved"},
            make_test_ctx(
                llm=FakeLLM("PLAN"), store=store, scope=Scope(1, 2), correlation_id="wf-1"
            ),
        )
    )
    assert res2.stop_reason == "complete"
    assert res2.outputs["review"] == "approved" and res2.outputs["ship"] == "shipping with approved"


def test_cancellation_aborts_before_first_wave():
    token = CancellationToken()
    token.cancel()
    wf = Workflow()
    wf.fn("x", lambda inp: 1)
    with pytest.raises(Cancelled):
        _run(
            wf.run(
                "g",
                make_test_ctx(
                    llm=FakeLLM("x"), cancel=token, scope=Scope(1, 2), correlation_id="wf-1"
                ),
            )
        )


def test_dependency_cycle_is_reported_as_deadlock():
    wf = Workflow()
    wf.fn("a", lambda inp: 1, after="b")
    wf.fn("b", lambda inp: 2, after="a")  # mutual deps → nothing ready
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-1"))
    )
    assert res.stop_reason == "deadlock"


class _JsonStore:
    """A StorePort that round-trips values through JSON — proves a checkpoint survives a *real*
    (serializing) store, unlike InMemoryStore which keeps live Python objects."""

    def __init__(self) -> None:
        self._d: dict = {}

    async def get(self, key):
        raw = self._d.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key, value):
        self._d[key] = json.dumps(value)  # raises TypeError if `value` isn't serializable

    async def delete(self, key):
        self._d.pop(key, None)


def _ctx_store(store):
    return make_test_ctx(llm=FakeLLM("x"), store=store, scope=Scope(1, 2), correlation_id="wf-1")


def test_durable_resume_survives_a_serializing_store():
    # a sub-workflow before the gate — its output must be JSON-serializable to checkpoint (the bug fix)
    child = Workflow("child")
    child.fn("compute", lambda inp: {"value": 42})

    def _wf():
        wf = Workflow("parent")
        wf.subworkflow("sub", child)
        wf.human_gate("review", after="sub")
        wf.fn(
            "ship", lambda inp: f"shipped {inp['sub']['compute']['value']}", after=["sub", "review"]
        )
        return wf

    store = _JsonStore()
    res = _run(_wf().run("go", _ctx_store(store)))  # checkpoint must JSON-encode cleanly
    assert res.stop_reason == "suspended"

    res2 = _run(_wf().resume(res.suspended.run_id, {"review": "ok"}, _ctx_store(store)))
    assert res2.stop_reason == "complete"
    assert res2.outputs["ship"] == "shipped 42"  # sub output survived the round-trip


def test_durable_resume_across_a_process_restart_with_filestore(tmp_path):
    # the real production scenario: suspend writes the checkpoint to disk; a BRAND-NEW FileStore instance
    # (simulating a fresh process after a crash/restart) resumes it from disk and completes correctly.
    def _wf():
        wf = Workflow("deploy")
        wf.fn("plan", lambda inp: {"steps": 3})
        wf.human_gate("approve", after="plan")
        wf.fn(
            "ship", lambda inp: f"shipped {inp['plan']['steps']} steps", after=["plan", "approve"]
        )
        return wf

    res = _run(
        _wf().run("go", _ctx_store(FileStore(str(tmp_path))))
    )  # process #1 → suspends to disk
    assert res.stop_reason == "suspended"

    res2 = _run(
        _wf().resume(
            res.suspended.run_id, {"approve": "approved"}, _ctx_store(FileStore(str(tmp_path)))
        )
    )  # process #2 → resumes from disk
    assert res2.stop_reason == "complete"
    assert res2.outputs["approve"] == "approved" and res2.outputs["ship"] == "shipped 3 steps"


# ── wave commit must catch arity / size mismatches loudly ────────────────────
#
# The wave loop uses ``zip(ready, outs, strict=True)`` for size matching
# plus a tuple-arity check per result, so a node whose ``run`` returned
# the wrong shape (or whose result was missing from ``outs``) surfaces
# a structured error naming the offending node instead of silently
# truncating.


def test_workflow_wave_raises_when_node_run_returns_wrong_arity():
    """A node's ``run`` returning a single value (forgot the ``Usage``)
    surfaces as a structured ``RuntimeError`` naming the offending node,
    not as a confusing ``ValueError: not enough values to unpack`` deep
    inside the wave loop."""
    from agentkit.agents.workflow import _Node

    wf = Workflow()

    # A valid first node.
    wf.fn("good", lambda inp: "ok")

    # Patch in a malformed node directly: its ``run`` returns a bare
    # string instead of ``(output, Usage)``. Public ``wf.fn`` always
    # wraps the user's callable to return a proper tuple, but a custom
    # extension (or a future node kind) could ship a wrong ``run``;
    # the guard makes that failure mode visible.
    async def _bad_run(inputs, goal, ctx):
        return "no usage tuple"  # ← wrong shape: not a (out, Usage) pair

    wf._nodes["bad"] = _Node(name="bad", kind="fn", after=("good",), run=_bad_run)
    wf._order.append("bad")

    with pytest.raises(RuntimeError, match=r"Workflow node 'bad' returned str"):
        _run(
            wf.run(
                "g",
                make_test_ctx(llm=FakeLLM("X"), scope=Scope(1, 2), correlation_id="wf-arity"),
            )
        )


def test_workflow_wave_happy_path_unaffected_by_arity_guard():
    """Regression check — every wave node returns the proper ``(out,
    Usage)`` tuple via ``wf.fn``/``wf.agent``/``wf.tool``, so the guard
    is invisible on the happy path."""
    wf = Workflow()
    wf.fn("a", lambda inp: "alpha")
    wf.fn("b", lambda inp: "beta", after="a")
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("Y"), scope=Scope(1, 2), correlation_id="wf-happy"))
    )
    assert res.stop_reason == "complete"
    assert res.outputs["a"] == "alpha"
    assert res.outputs["b"] == "beta"


# ── Concurrent runs of the same Workflow are isolated ────────────────────────
#
# A Workflow's builder state (``_nodes``, ``_order``, ``_routes``)
# is read-only at run time; only the ``_execute`` closure mutates
# the ``done``/``pending``/``steps`` locals it owns. Two runs of
# the same Workflow instance must therefore be observationally
# independent — one's node execution order cannot leak into the
# other's ``done`` map.


def test_two_concurrent_workflow_runs_stay_isolated() -> None:
    """Two runs of the same Workflow instance produce independent
    outputs — each observes only its own goal + its own execution
    of the fn nodes."""

    wf = Workflow(name="isolation")
    wf.fn("head", lambda inp, goal: f"head:{goal}")
    wf.fn("tail", lambda inp, goal: f"tail:{inp['head']}", after="head")

    async def go():
        return await asyncio.gather(
            wf.run(
                "goal-a",
                make_test_ctx(llm=FakeLLM("Y"), scope=Scope(1, 2), correlation_id="wf-a"),
            ),
            wf.run(
                "goal-b",
                make_test_ctx(llm=FakeLLM("Y"), scope=Scope(1, 2), correlation_id="wf-b"),
            ),
        )

    a, b = asyncio.run(go())

    assert a.outputs["head"] == "head:goal-a"
    assert a.outputs["tail"] == "tail:head:goal-a"
    assert b.outputs["head"] == "head:goal-b"
    assert b.outputs["tail"] == "tail:head:goal-b"


def test_workflow_run_does_not_mutate_builder_state() -> None:
    """A run's execution is a pure function over the builder state.
    ``_nodes`` / ``_order`` / ``_routes`` observable after a run
    match what they were before — so the Workflow instance stays
    reusable across runs without accumulating cruft."""
    wf = Workflow()
    wf.fn("a", lambda inp: "alpha")
    wf.fn("b", lambda inp: "beta", after="a")

    nodes_before = dict(wf._nodes)
    order_before = list(wf._order)
    routes_before = dict(wf._routes)

    _run(wf.run("g", make_test_ctx(llm=FakeLLM("Y"), scope=Scope(1, 2))))

    assert wf._nodes == nodes_before
    assert wf._order == order_before
    assert wf._routes == routes_before


def test_workflow_resume_deep_copies_done() -> None:
    """A caller mutating a node output post-resume must NOT corrupt the persisted
    checkpoint on ``InMemoryStore`` (which stores by reference — a shallow ``dict(...)``
    would leave the inner mutable values aliased between the store's copy and the
    live workflow state exposed as ``WorkflowResult.outputs``)."""
    from agentkit.agents.workflow import _ckpt_key

    def _wf() -> Workflow:
        # ``step_a`` is pre-satisfied via the checkpoint; ``review`` is a human_gate
        # that isn't decided on the first resume — so resume suspends again (the
        # checkpoint gets re-written), giving us a clean second-read to compare against.
        wf = Workflow("deepcopy")
        wf.fn("step_a", lambda inp: [{"mutable": [1, 2]}])
        wf.human_gate("review", after="step_a")
        return wf

    store = InMemoryStore()
    original_value = [{"mutable": [1, 2]}]
    checkpoint = {
        "goal": "g",
        "done": {"step_a": original_value},
        "steps": 1,
    }

    async def go() -> None:
        await store.set(_ckpt_key("run-1"), checkpoint)

        # First resume — no decision provided, so it re-suspends at ``review``.
        res1 = await _wf().resume(
            "run-1",
            {},
            make_test_ctx(llm=FakeLLM("x"), store=store, scope=Scope(1, 2), correlation_id="run-1"),
        )
        assert res1.stop_reason == "suspended"

        # Mutate the inner list returned from the workflow. Under the bug this would
        # corrupt both ``original_value`` (aliased into the seed dict) AND the just-
        # re-written store checkpoint (which reused the same references via
        # ``dict(done)``). The deep-copy at both store boundaries breaks the alias.
        res1.outputs["step_a"][0]["mutable"].append(999)
        res1.outputs["step_a"].append({"injected": True})

        # Second resume — reads the store fresh; step_a must match its original shape.
        res2 = await _wf().resume(
            "run-1",
            {},
            make_test_ctx(llm=FakeLLM("x"), store=store, scope=Scope(1, 2), correlation_id="run-1"),
        )
        assert res2.stop_reason == "suspended"
        assert res2.outputs["step_a"] == [{"mutable": [1, 2]}]

        # And the caller's original seed dict is also unharmed — deep-copy on READ
        # means resume never returned a reference into ``checkpoint``.
        assert original_value == [{"mutable": [1, 2]}]

    _run(go())


def test_workflow_with_async_planner_via_planpolicy_awaits_before_iterating() -> None:
    """``PlanPolicy`` accepts an async planner and awaits its result
    before iterating. A sync-only iteration would raise
    ``TypeError: 'coroutine' object is not iterable``."""
    import inspect as _inspect

    from agentkit.agents.policies.plan import Planner, PlanPolicy, Step

    plan_calls: list[str] = []

    class AsyncPlanner:
        async def plan(self, goal: str, ctx) -> list[Step]:
            plan_calls.append(goal)
            await asyncio.sleep(0)
            return [Step(agent="does_not_run", input=goal, group=0)]

    policy = PlanPolicy(planner=AsyncPlanner())
    assert isinstance(policy.planner, Planner)
    assert _inspect.iscoroutinefunction(policy.planner.plan)  # sanity — planner IS async

    # The policy's ``execute`` awaits the coroutine before iterating.
    # We invoke ``planner.plan`` directly here to prove the coroutine
    # nature exists — the actual awaitable-detection code is exercised
    # in the coordinator integration tests below.
    coro = policy.planner.plan("g", make_test_ctx(llm=FakeLLM("Y"), scope=Scope(1, 2)))
    assert _inspect.iscoroutine(coro)
    steps = _run(coro)
    assert steps[0].agent == "does_not_run"
    assert plan_calls == ["g"]


# ── on_existing: idempotency guard for Workflow.run ──────────────────────────
#
# ``Workflow.run(run_id=...)`` was silently overwriting a previous run's
# state under the same correlation_id. The ``on_existing`` kwarg lets a
# caller opt into idempotency semantics:
#   * ``"start_fresh"`` (default) — historical behaviour; ignore any prior state.
#   * ``"resume"``      — replay from the persisted checkpoint if one exists.
#   * ``"fail"``        — raise if any checkpoint (terminal or not) exists.


def test_workflow_on_existing_fail_raises_if_snapshot_exists():
    """``on_existing="fail"`` consults the checkpointer for ANY snapshot
    (terminal or not) and raises ``CheckpointerError`` — protecting an
    at-least-once pipeline from silently re-running a completed job."""
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer
    from agentkit.kernel.errors import CheckpointerError
    from agentkit.kernel.ports import CheckpointStatus

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    # Pre-populate a snapshot for the run_id — even a DONE (terminal)
    # snapshot must trip the guard, because ``fail`` semantics are
    # "any persisted state", not "only resumable state".
    _run(
        cpt.snapshot("wf-fail", {"goal": "g", "done": {}, "steps": 0}, status=CheckpointStatus.DONE)
    )

    wf = Workflow()
    wf.fn("a", lambda inp: "alpha")

    with pytest.raises(CheckpointerError, match="wf-fail"):
        _run(
            wf.run(
                "g",
                make_test_ctx(
                    llm=FakeLLM("x"),
                    checkpointer=cpt,
                    scope=Scope(1, 2),
                    correlation_id="wf-fail",
                ),
                on_existing="fail",
            )
        )


def test_workflow_on_existing_fail_no_snapshot_proceeds():
    """When there's no prior state for the run_id, ``on_existing="fail"``
    behaves like a normal run — the guard fires only on a real
    collision."""
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    wf = Workflow()
    wf.fn("a", lambda inp: "alpha")

    res = _run(
        wf.run(
            "g",
            make_test_ctx(
                llm=FakeLLM("x"),
                checkpointer=cpt,
                scope=Scope(1, 2),
                correlation_id="wf-clean",
            ),
            on_existing="fail",
        )
    )
    assert res.stop_reason == "complete"
    assert res.outputs["a"] == "alpha"


def test_workflow_on_existing_resume_picks_up_from_checkpoint():
    """``on_existing="resume"`` replays from the persisted ``done``
    map — nodes already recorded in the checkpoint are NOT re-run."""
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    # Pre-populate a resumable snapshot where step_a is already done.
    _run(
        cpt.snapshot(
            "wf-resume",
            {"goal": "orig-goal", "done": {"step_a": "alpha-from-checkpoint"}, "steps": 1},
        )
    )

    ran: list[str] = []

    def step_a(inp):
        ran.append("step_a")
        return "alpha-fresh"

    def step_b(inp):
        ran.append("step_b")
        return f"beta({inp['step_a']})"

    wf = Workflow()
    wf.fn("step_a", step_a)
    wf.fn("step_b", step_b, after="step_a")

    res = _run(
        wf.run(
            "new-goal",  # ignored — resumed run uses the checkpoint's goal
            make_test_ctx(
                llm=FakeLLM("x"),
                checkpointer=cpt,
                scope=Scope(1, 2),
                correlation_id="wf-resume",
            ),
            on_existing="resume",
        )
    )
    assert res.stop_reason == "complete"
    # step_a came from the checkpoint — the user's step_a callable
    # never ran, and step_b's input threaded the persisted value.
    assert ran == ["step_b"]
    assert res.outputs["step_a"] == "alpha-from-checkpoint"
    assert res.outputs["step_b"] == "beta(alpha-from-checkpoint)"


def test_workflow_on_existing_resume_falls_through_when_no_checkpoint():
    """``on_existing="resume"`` with no persisted state = fresh run."""
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    wf = Workflow()
    wf.fn("a", lambda inp: "alpha")

    res = _run(
        wf.run(
            "g",
            make_test_ctx(
                llm=FakeLLM("x"),
                checkpointer=cpt,
                scope=Scope(1, 2),
                correlation_id="wf-empty",
            ),
            on_existing="resume",
        )
    )
    assert res.stop_reason == "complete"
    assert res.outputs["a"] == "alpha"


def test_workflow_on_existing_resume_ignores_terminal_snapshot():
    """A terminal (``DONE``) checkpoint is NOT resumable — the
    checkpointer's default filter hides it, so ``on_existing="resume"``
    starts fresh. This is the safety net against replaying a finished
    job when a caller pipes an old run_id back through the resume path."""
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer
    from agentkit.kernel.ports import CheckpointStatus

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(
        cpt.snapshot(
            "wf-done",
            {"goal": "old", "done": {"a": "stale"}, "steps": 1},
            status=CheckpointStatus.DONE,
        )
    )

    ran: list[str] = []

    def a(inp):
        ran.append("a")
        return "fresh"

    wf = Workflow()
    wf.fn("a", a)

    res = _run(
        wf.run(
            "new-goal",
            make_test_ctx(
                llm=FakeLLM("x"),
                checkpointer=cpt,
                scope=Scope(1, 2),
                correlation_id="wf-done",
            ),
            on_existing="resume",
        )
    )
    assert res.stop_reason == "complete"
    # The DONE snapshot was skipped by the default terminal filter —
    # the user's node re-ran and produced the fresh output.
    assert ran == ["a"]
    assert res.outputs["a"] == "fresh"


def test_workflow_on_existing_start_fresh_ignores_existing():
    """Default ``on_existing="start_fresh"`` preserves backward compat —
    a pre-populated snapshot is IGNORED and every node re-runs from
    step 1. This is the pre-fix behaviour, now explicit."""
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(
        cpt.snapshot(
            "wf-fresh",
            {"goal": "old", "done": {"a": "stale-from-checkpoint"}, "steps": 1},
        )
    )

    ran: list[str] = []

    def a(inp):
        ran.append("a")
        return "fresh"

    wf = Workflow()
    wf.fn("a", a)

    # No ``on_existing=`` kwarg — the default is ``"start_fresh"``.
    res = _run(
        wf.run(
            "g",
            make_test_ctx(
                llm=FakeLLM("x"),
                checkpointer=cpt,
                scope=Scope(1, 2),
                correlation_id="wf-fresh",
            ),
        )
    )
    assert res.stop_reason == "complete"
    assert ran == ["a"]  # the node re-ran despite the pre-populated snapshot
    assert res.outputs["a"] == "fresh"


def test_workflow_on_existing_without_checkpointer_degrades_cleanly():
    """When ``ctx.checkpointer`` is ``None``, the guard has nowhere to
    look — ``"fail"`` cannot fire, ``"resume"`` has nothing to resume
    from. Both degrade to a fresh run instead of raising an
    unrelated ``AttributeError``."""
    wf = Workflow()
    wf.fn("a", lambda inp: "alpha")

    for mode in ("fail", "resume"):
        res = _run(
            wf.run(
                "g",
                make_test_ctx(
                    llm=FakeLLM("x"),
                    scope=Scope(1, 2),
                    correlation_id="wf-nocpt",
                ),
                on_existing=mode,
            )
        )
        assert res.stop_reason == "complete"
        assert res.outputs["a"] == "alpha"
