"""Workflow (ch16 / R9): the explicit-control graph engine — DAG scheduling, typed data on edges,
conditional loop-back (bounded by max_steps), human-gate suspend/resume, and the shared spine
(cancellation, usage). Offline & deterministic via FakeLLM + asyncio.run."""

import asyncio
import copy
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
    """A caller must NOT be able to corrupt the persisted checkpoint through the
    node outputs resume hands back, on ``InMemoryStore`` (which stores by
    reference — a shallow ``dict(...)`` would leave the inner mutable values
    aliased between the store's copy and the live workflow state exposed as
    ``WorkflowResult.outputs``).

    This test used to prove that by MUTATING through the returned outputs
    (``res1.outputs["step_a"][0]["mutable"].append(999)``) and asserting the
    store was unharmed. ``WorkflowResult.outputs`` is now frozen at
    construction, so that mutation raises instead — a strictly stronger
    guarantee, and the reason the assertion below is a ``pytest.raises``.

    The un-aliasing it was really testing is still pinned, structurally
    (``is not``) rather than by consequence, and from the other end: the
    caller's seed value is mutated instead, and the second resume must still
    read the original shape out of the store."""
    from agentkit.capabilities.checkpointer import ckpt_key

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
    # Seeded as the RAW record ``StoreBackedCheckpointStore`` reads, not via
    # ``Checkpointer.snapshot`` — snapshot deep-copies at its own seam, which
    # would un-alias ``original_value`` before the workflow ever sees it and
    # leave this test proving the checkpointer's copy rather than the
    # workflow's. Writing the KV directly is what a real serializing backend
    # hands back, minus the serialization: ``InMemoryStore`` returns the very
    # object it was given.
    checkpoint = {
        "run_id": "run-1",
        "version": 1,
        "state": {"goal": "g", "done": {"step_a": original_value}, "steps": 1},
        "created_at": 0.0,
        "status": "suspended",  # non-terminal, or ``cp.resume`` filters it out
        "metadata": {},
    }

    async def go() -> None:
        await store.set(ckpt_key("run-1"), checkpoint)

        # First resume — no decision provided, so it re-suspends at ``review``.
        res1 = await _wf().resume(
            "run-1",
            {},
            make_test_ctx(llm=FakeLLM("x"), store=store, scope=Scope(1, 2), correlation_id="run-1"),
        )
        assert res1.stop_reason == "suspended"

        # The mutation this test used to perform is now refused outright — the
        # returned outputs are a frozen COPY, so the corruption route is closed
        # at the value rather than defended at the store boundary.
        with pytest.raises(TypeError, match="frozen value"):
            res1.outputs["step_a"][0]["mutable"].append(999)
        with pytest.raises(TypeError, match="frozen value"):
            res1.outputs["step_a"].append({"injected": True})

        # The un-aliasing itself, stated directly: nothing the caller can reach
        # through ``outputs`` is the object the seed dict (and therefore the
        # store) holds.
        assert res1.outputs["step_a"][0]["mutable"] is not original_value[0]["mutable"]
        assert res1.outputs["step_a"] is not original_value

        # Now corrupt from the OTHER end — the caller's own seed value, which is
        # still an ordinary list. The deep-copy at the store boundary means the
        # persisted record never held this reference either.
        original_value[0]["mutable"].append(999)

        # Second resume — reads the store fresh; step_a must match its original shape.
        res2 = await _wf().resume(
            "run-1",
            {},
            make_test_ctx(llm=FakeLLM("x"), store=store, scope=Scope(1, 2), correlation_id="run-1"),
        )
        assert res2.stop_reason == "suspended"
        assert res2.outputs["step_a"] == [{"mutable": [1, 2]}]

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


def test_a_workflow_gate_warns_when_the_suspend_cannot_be_persisted() -> None:
    """A suspend with no store wired returned `stop_reason="suspended"` and a
    `Suspended` object while persisting NOTHING.

    The truth surfaced later — usually in a different process — as
    "no suspended workflow <id> to resume", with nothing pointing back at the
    missing store. That is the silent, well-formed failure class: the return
    value says resumable, the system is not.

    Workflow now resolves its durable seam through the shared
    `resolve_checkpointer`, so this fires only when NEITHER a checkpointer nor
    a store is wired.
    """
    import asyncio
    import warnings

    import pytest as _pytest

    from agentkit import Scope, Workflow
    from agentkit.runtime import RunContext, Services

    wf = Workflow(max_steps=10)
    wf.fn("prep", lambda inputs: "ready")
    wf.human_gate("approve", after="prep")
    wf.fn("act", lambda inputs: "done", after="approve")

    ctx = RunContext("wf-unpersisted", Scope(), services=Services())  # no store

    with _pytest.warns(UserWarning, match="no durable seam is wired"):
        result = asyncio.run(wf.run("go", ctx))
    assert result.stop_reason == "suspended"

    # And the downstream failure the warning predicts really does happen.
    with _pytest.raises(ValueError, match="no suspended workflow"):
        asyncio.run(wf.resume("wf-unpersisted", {"approve": "yes"}, ctx))

    # With a store wired, no warning and the resume works.
    from agentkit.adapters.store import InMemoryStore

    ctx2 = RunContext("wf-ok", Scope(), services=Services(store=InMemoryStore()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert asyncio.run(wf.run("go", ctx2)).stop_reason == "suspended"
    assert not [w for w in caught if "no durable seam" in str(w.message)]
    assert asyncio.run(wf.resume("wf-ok", {"approve": "yes"}, ctx2)).stop_reason == "complete"


def test_a_workflow_gate_persists_through_a_checkpointer_alone() -> None:
    """The trap this fix removes.

    Workflow used to persist ONLY through `ctx.store`, while the ReAct
    cognition preferred `ctx.checkpointer`. So wiring the documented durable
    seam — a `Checkpointer` — left workflow human-gates silently unpersisted,
    and the failure surfaced later as "no suspended workflow <id> to resume".
    Both producers now share one resolution order.
    """
    import asyncio
    import warnings

    from agentkit import Scope, Workflow
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer
    from agentkit.runtime import RunContext, Services

    wf = Workflow(max_steps=10)
    wf.fn("prep", lambda inputs: "ready")
    wf.human_gate("approve", after="prep")
    wf.fn("act", lambda inputs: "done", after="approve")

    cp = Checkpointer(port=InMemoryCheckpointStore())
    ctx = RunContext("wf-cp", Scope(), services=Services(checkpointer=cp))  # NO store

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = asyncio.run(wf.run("go", ctx))
    assert first.stop_reason == "suspended"
    assert not [w for w in caught if "no durable seam" in str(w.message)]

    # It really was written, and marked SUSPENDED rather than RUNNING — the
    # status a bare KV write could not express.
    saved = asyncio.run(cp.resume("wf-cp"))
    assert saved is not None and saved.status == "suspended"
    assert saved.state["goal"] == "go" and "prep" in saved.state["done"]

    resumed = asyncio.run(wf.resume("wf-cp", {"approve": "yes"}, ctx))
    assert resumed.stop_reason == "complete"
    assert resumed.outputs["act"] == "done"
    # Terminal → the checkpoint is reclaimed, so a naive re-resume cannot replay.
    assert asyncio.run(cp.resume("wf-cp")) is None


def test_a_terminal_resume_reclaims_the_checkpoint_so_a_naive_rerun_starts_fresh() -> None:
    """The hazard the reclaim in ``Workflow.resume`` exists for, stated end to end.

    The gate writes its checkpoint with status SUSPENDED and nothing downgrades
    it on the way out, so a finished run that KEPT its record would still read
    as resumable — which is exactly what ``on_existing="resume"`` and any
    "resume if anything exists" supervisor act on.

    Measured by disabling the reclaim branch on this graph: after the run
    completed, a second ``resume(run_id, {"approve": "yes"})`` returned
    ``complete`` again with ``act`` and ``ship`` each having executed TWICE
    (2 instead of 1), and ``run(..., decisions=..., on_existing="resume")``
    re-executed both the same way. There is one seam to clear now — Workflow
    never writes the old private ``workflow:<run_id>`` KV slot — so this is the
    whole of the protection.
    """
    import asyncio

    from agentkit import Scope, Workflow
    from agentkit.adapters.checkpoint import InMemoryCheckpointStore
    from agentkit.capabilities import Checkpointer
    from agentkit.runtime import RunContext, Services

    ran = {"prep": 0, "act": 0, "ship": 0}

    def _bump(key, out):
        def _fn(inputs):
            ran[key] += 1
            return out
        return _fn

    def _wf() -> Workflow:
        wf = Workflow(max_steps=10)
        wf.fn("prep", _bump("prep", "ready"))
        wf.human_gate("approve", after="prep")
        wf.fn("act", _bump("act", "act"), after="approve")
        wf.fn("ship", _bump("ship", "ship"), after="act")
        return wf

    cp = Checkpointer(port=InMemoryCheckpointStore())

    def _ctx():
        return RunContext("wf-reclaim", Scope(), services=Services(checkpointer=cp))

    assert asyncio.run(_wf().run("go", _ctx())).stop_reason == "suspended"
    # A suspended run still resumes — the reclaim must not fire mid-flight.
    mid = asyncio.run(_wf().run("go", _ctx(), on_existing="resume"))
    assert mid.stop_reason == "suspended" and mid.outputs["prep"] == "ready"

    assert mid.steps == 1 and ran["prep"] == 1  # replayed the record; prep NOT re-run

    assert asyncio.run(_wf().resume("wf-reclaim", {"approve": "yes"}, _ctx())).stop_reason == (
        "complete"
    )
    assert ran == {"prep": 1, "act": 1, "ship": 1}

    # Nothing is left to replay, at either level of the seam.
    assert asyncio.run(cp.resume("wf-reclaim")) is None
    assert asyncio.run(cp.resume("wf-reclaim", include_terminal=True)) is None

    # So a naive "resume if anything exists" supervisor gets a FRESH run rather
    # than a replay. ``prep`` is the discriminator: a replay would restore it
    # from the finished run's ``done`` map and skip it (as the mid-flight call
    # above did, ``ran["prep"] == 1``), while a fresh run must execute it again.
    again = asyncio.run(_wf().run("go", _ctx(), decisions={"approve": "yes"}, on_existing="resume"))
    assert again.stop_reason == "complete"
    assert ran == {"prep": 2, "act": 2, "ship": 2}

    # And a second ``resume`` of the reclaimed run refuses outright.
    with pytest.raises(ValueError, match="no suspended workflow"):
        asyncio.run(_wf().resume("wf-reclaim", {"approve": "yes"}, _ctx()))


# --------------------------------------------------------------------------------------------
# Regression: the step budget must bound EVERY path, and `max_steps=0` must mean zero.
#
# The backstop used to run after the wave, guarded by `and pending`. A self-route
# (`route("a", …, to="a")`) re-arms its own node in the route pass a few lines LATER, so at
# check time `pending` was empty and the guard never fired: measured 39,782 executions in 3s
# with `max_steps=20`, never terminating. A route to a genuine ancestor (`b → a`) only looked
# bounded because its wave happened to leave a sibling pending. `max_steps=0` likewise ran one
# node (measured `steps=1`) because nothing was checked before the first wave.
# --------------------------------------------------------------------------------------------


def _counter():
    runs = {"n": 0}

    def bump(inp):
        runs["n"] += 1
        return runs["n"]

    return runs, bump


def test_self_route_is_bounded_by_max_steps():
    """A node routing to ITSELF looped forever (39,782 runs in 3s, no stop_reason)."""
    runs, bump = _counter()
    wf = Workflow(max_steps=6)
    wf.fn("a", bump)
    wf.route("a", when=lambda o: True, to="a")  # always self-loops
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-self"))
    )
    assert res.stop_reason == "max_steps"
    assert res.steps == 6 and runs["n"] == 6  # bounded EXACTLY, not "eventually"


def test_two_node_cycle_is_bounded_by_max_steps():
    """The ancestor loop-back (`b → a`) stays bounded — and now stops ON the budget, not past it."""
    wf = Workflow(max_steps=5)
    wf.fn("a", lambda inp: 1)
    wf.fn("b", lambda inp: True, after="a")
    wf.route("b", when=lambda o: o, to="a")
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-cyc"))
    )
    assert res.stop_reason == "max_steps" and res.steps == 5


def test_self_route_stops_on_its_own_when_the_condition_clears():
    """POSITIVE CONTROL: the bound must not disable self-routing. A self-route whose
    condition goes false still loops, terminates by itself, and feeds its final output
    downstream — a fix that simply refused self-routes would fail here."""
    runs, bump = _counter()
    wf = Workflow(max_steps=20)
    wf.fn("a", bump)
    wf.fn("b", lambda inp: f"saw {inp['a']}", after="a")
    wf.route("a", when=lambda o: o < 3, to="a")  # self-loop until the 3rd attempt
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-clear"))
    )
    assert res.stop_reason == "complete"
    assert runs["n"] == 3  # looped, then stopped on its own condition
    assert res.outputs["b"] == "saw 3"  # downstream re-ran against the final value


def test_self_route_combined_with_another_route_still_reaches_the_other_branch():
    """A self-route sharing a node with a second route: both are evaluated, and the
    workflow still completes well inside the budget."""
    runs, bump = _counter()
    wf = Workflow(max_steps=20)
    wf.fn("a", bump)
    wf.fn("retry_marker", lambda inp: "retried")
    wf.route("a", when=lambda o: o < 2, to="a")  # self-loop on the first pass
    wf.route("a", when=lambda o: o >= 2, to="retry_marker")  # then fan out elsewhere
    # Before the fix this pair raised ``KeyError: 'a'``: the self-route popped a's own
    # entry from ``done`` and the second route re-read it.
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-both"))
    )
    assert res.stop_reason == "complete"
    assert runs["n"] == 2 and res.outputs["retry_marker"] == "retried"


def test_max_steps_zero_runs_nothing():
    """`max_steps=0` used to run one node (measured `steps=1`, `nodes_ran=[0]`)."""
    ran = {"n": 0}
    wf = Workflow(max_steps=0)
    wf.fn("a", lambda inp: ran.__setitem__("n", ran["n"] + 1) or "ran")
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-zero"))
    )
    assert res.stop_reason == "max_steps"
    assert res.steps == 0 and res.outputs == {} and ran["n"] == 0  # zero means zero


def test_max_steps_one_runs_exactly_one_wave():
    """The tightest non-zero budget: one wave, then stop with the rest still pending."""
    wf = Workflow(max_steps=1)
    wf.fn("a", lambda inp: "alpha")
    wf.fn("b", lambda inp: "beta", after="a")
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-one"))
    )
    assert res.stop_reason == "max_steps" and res.steps == 1
    assert res.outputs == {"a": "alpha"}  # b never ran


def test_a_budget_that_exactly_covers_the_graph_still_completes():
    """POSITIVE CONTROL for the pre-wave check: an acyclic graph whose node count equals
    `max_steps` must finish `complete`, not be strangled one step short."""
    wf = Workflow(max_steps=3)
    wf.fn("a", lambda inp: 1)
    wf.fn("b", lambda inp: inp["a"] + 1, after="a")
    wf.fn("c", lambda inp: inp["b"] + 1, after="b")
    res = _run(
        wf.run("g", make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="wf-exact"))
    )
    assert res.stop_reason == "complete" and res.steps == 3 and res.outputs["c"] == 3


# --------------------------------------------------------------------------------------------
# Regression: a workflow `tool` node must carry the tool's OWN `url_arg` / `side_effecting`
# into the ToolRequest — the react cognition already does (`_tool_request`). It didn't, so a
# tool declared `url_arg="url", side_effecting=True` reached the invoker as `None` / `False`:
# measured, an egress()-guarded workflow fetched https://evil.com/x against an allowlist of
# example.com (Egress only checks when `url_arg` is set), and idempotent() ran the same charge
# twice (its `when` reads `request.side_effecting`).
# --------------------------------------------------------------------------------------------


def _fetcher(hits, *, url_arg="url", side_effecting=True):
    from agentkit.tools.function import FunctionTool

    return FunctionTool(
        "fetch",
        lambda a, c: hits.__setitem__("n", hits["n"] + 1) or "body",
        description="fetch a URL and return its body; egress-gated in these tests",
        side_effecting=side_effecting,
        url_arg=url_arg,
    )


def _egress_ctx():
    from agentkit.capabilities import Guardrail
    from agentkit.middlewares.egress_audit import egress

    return make_test_ctx(
        llm=FakeLLM("x"),
        tool_middleware=[egress(Guardrail(egress_allow=("example.com",)))],
        scope=Scope(1, 2),
        correlation_id="wf-egress",
    )


def _first_cause(eg):
    """`gather_bounded` runs the wave in a TaskGroup, so a node failure surfaces as an
    ExceptionGroup — unwrap to the middleware's real exception."""
    while isinstance(eg, BaseExceptionGroup):
        eg = eg.exceptions[0]
    return eg


def test_workflow_tool_node_inherits_url_arg_so_egress_can_block():
    """The SSRF-guard bypass: the blocked host went through, and the tool executed."""
    hits = {"n": 0}
    wf = Workflow(max_steps=5)
    wf.tool("f", _fetcher(hits), args=lambda inp: {"url": "https://evil.com/x"})
    with pytest.raises(BaseException) as ei:  # noqa: B017 — ExceptionGroup or PermissionError
        _run(wf.run("g", _egress_ctx()))
    assert isinstance(_first_cause(ei.value), PermissionError)
    assert hits["n"] == 0  # blocked BEFORE the side effect


def test_workflow_tool_node_still_allows_a_permitted_url():
    """POSITIVE CONTROL: inheriting `url_arg` must not turn egress into deny-everything —
    an allowlisted host still runs."""
    hits = {"n": 0}
    wf = Workflow(max_steps=5)
    wf.tool("f", _fetcher(hits), args=lambda inp: {"url": "https://api.example.com/x"})
    res = _run(wf.run("g", _egress_ctx()))
    assert res.stop_reason == "complete" and res.outputs["f"] == "body" and hits["n"] == 1


def test_workflow_tool_node_without_url_arg_is_not_egress_gated():
    """A tool that declares no `url_arg` stays ungated even when an argument happens to be
    called `url` — the inheritance must read the TOOL, not the argument names."""
    hits = {"n": 0}
    wf = Workflow(max_steps=5)
    wf.tool("f", _fetcher(hits, url_arg=None), args=lambda inp: {"url": "https://evil.com/x"})
    res = _run(wf.run("g", _egress_ctx()))
    assert res.stop_reason == "complete" and hits["n"] == 1


def test_workflow_tool_node_kwargs_still_override_the_tool_declaration():
    """POSITIVE CONTROL for the explicit kwargs: a graph author can still mark a
    self-declaring-nothing tool as URL-bearing, and that gating takes effect."""
    hits = {"n": 0}
    wf = Workflow(max_steps=5)
    wf.tool(
        "f",
        _fetcher(hits, url_arg=None, side_effecting=False),
        args=lambda inp: {"url": "https://evil.com/x"},
        url_arg="url",
    )
    with pytest.raises(BaseException) as ei:  # noqa: B017
        _run(wf.run("g", _egress_ctx()))
    assert isinstance(_first_cause(ei.value), PermissionError)
    assert hits["n"] == 0


def test_workflow_tool_node_cannot_downgrade_a_side_effecting_tool():
    """The inheritance is ESCALATE-ONLY. A graph author must not be able to pass
    ``side_effecting=False`` and quietly opt a tool out of the guards its own
    author asked for — the tool knows what it does, the node does not.

    Written because the fix reads ``side_effecting or getattr(...)``, which has
    this property by construction rather than by intent. Without this test,
    "clarifying" it into ``side_effecting if side_effecting is not None else …``
    would silently allow the downgrade.
    """
    from agentkit.middlewares.memoize import idempotent
    from agentkit.tools.function import FunctionTool

    hits = {"n": 0}

    async def _charge(args, ctx):
        hits["n"] += 1
        return "charged"

    tool = FunctionTool(
        name="charge",
        fn=_charge,
        description="Charge a card. Side-effecting, and says so on the tool itself.",
        side_effecting=True,
    )
    wf = Workflow(max_steps=5)
    wf.tool("a", tool, args=lambda inp: {"id": "x"}, side_effecting=False)
    wf.tool("b", tool, args=lambda inp: {"id": "x"}, side_effecting=False, after="a")

    # ``idempotent()`` keeps its dedupe record in the store, so it needs one
    # plus a stable scope/run id. (Omitting the store made this test fail
    # against a correct fix — the harness was wrong, not the code.)
    ctx = make_test_ctx(
        llm=FakeLLM("ok"),
        tool_middleware=[idempotent()],
        store=InMemoryStore(),
        scope=Scope(1, 2),
        correlation_id="wf-downgrade",
    )
    _run(wf.run("g", ctx))

    # Deduped, because the TOOL's declaration survived the node's False.
    assert hits["n"] == 1


def test_workflow_tool_node_stamps_the_escalated_flag_onto_the_request():
    """The SAME escalate-only rule, asserted on the propagated flag itself
    rather than through ``idempotent()``'s behaviour.

    The test above reads the rule through a dedupe count, and that stopped
    being a faithful probe: ``memoize``/``idempotent`` now consult the TOOL as
    well as the request (a ``ToolRequest`` built positionally never carries the
    flag), so they dedupe correctly even when the node's downgrade DOES reach
    the request. The mutant that reintroduces the downgrade therefore survived
    that test — masked by defence-in-depth downstream.

    The workflow-level invariant is still real, because other consumers read
    ``request.side_effecting`` and have no tool object to fall back on — most
    importantly the approval gate, where a downgrade means a mutation executing
    without ever being offered for approval. So assert the flag where it is
    actually set.
    """
    from agentkit.tools.function import FunctionTool

    seen: list[bool] = []

    async def _spy(call, nxt):
        seen.append(bool(getattr(call.request, "side_effecting", False)))
        async for x in nxt(call):
            yield x

    async def _charge(args, ctx):
        return "charged"

    tool = FunctionTool(
        name="charge",
        fn=_charge,
        description="Charge a card. Side-effecting, and says so on the tool itself.",
        side_effecting=True,
    )
    wf = Workflow(max_steps=5)
    wf.tool("a", tool, args=lambda inp: {"id": "x"}, side_effecting=False)

    _run(wf.run("g", make_test_ctx(llm=FakeLLM("ok"), tool_middleware=[_spy])))

    assert seen == [True], (
        "the node's side_effecting=False must not reach the request — a consumer "
        "with no tool object to fall back on would treat a mutation as read-only"
    )


def test_a_read_only_tool_is_not_escalated_by_the_node_default():
    """POSITIVE CONTROL for the test above. If the propagation were 'always
    True' rather than 'escalate-only', the assertion above would pass for the
    wrong reason and every read-only tool would start demanding approval."""
    from agentkit.tools.function import FunctionTool

    seen: list[bool] = []

    async def _spy(call, nxt):
        seen.append(bool(getattr(call.request, "side_effecting", False)))
        async for x in nxt(call):
            yield x

    async def _read(args, ctx):
        return "value"

    tool = FunctionTool(
        name="lookup",
        fn=_read,
        description="Read-only lookup that declares itself free of side effects.",
        side_effecting=False,
    )
    wf = Workflow(max_steps=5)
    wf.tool("a", tool, args=lambda inp: {"id": "x"})

    _run(wf.run("g", make_test_ctx(llm=FakeLLM("ok"), tool_middleware=[_spy])))

    assert seen == [False]


def test_workflow_tool_node_cannot_suppress_an_egress_check():
    """Same rule for ``url_arg``: passing ``None`` inherits the tool's, it does
    not disable the check."""
    hits = {"n": 0}
    wf = Workflow(max_steps=5)
    wf.tool(
        "f",
        _fetcher(hits, url_arg="url", side_effecting=False),
        args=lambda inp: {"url": "https://evil.com/x"},
        url_arg=None,
    )
    with pytest.raises(BaseException) as ei:  # noqa: B017
        _run(wf.run("g", _egress_ctx()))
    assert isinstance(_first_cause(ei.value), PermissionError)
    assert hits["n"] == 0


def test_workflow_tool_node_inherits_side_effecting_so_idempotent_dedupes():
    """`idempotent()` keys on (run, scope, tool, args) but only fires when the REQUEST says
    side-effecting — measured: the same charge executed twice from a workflow."""
    from agentkit.middlewares.memoize import idempotent
    from agentkit.tools.function import FunctionTool

    calls = {"n": 0}
    charge = FunctionTool(
        "charge",
        lambda a, c: calls.__setitem__("n", calls["n"] + 1) or {"ok": 1},
        description="charge an account a given amount; mutates the ledger",
        side_effecting=True,
    )
    ctx = make_test_ctx(
        llm=FakeLLM("x"),
        tool_middleware=[idempotent()],
        store=InMemoryStore(),
        scope=Scope(1, 2),
        correlation_id="wf-idem",
    )
    wf = Workflow(max_steps=5)
    wf.tool("c1", charge, args=lambda inp: {"amt": 5})
    wf.tool("c2", charge, args=lambda inp: {"amt": 5}, after="c1")
    res = _run(wf.run("g", ctx))
    assert res.stop_reason == "complete"
    assert calls["n"] == 1  # at-least-once retry cannot re-fire the charge
    assert res.outputs["c1"] == res.outputs["c2"] == {"ok": 1}  # both nodes still got a result


def test_workflow_tool_node_read_only_tool_is_not_deduped():
    """POSITIVE CONTROL: inheritance must not mark everything side-effecting — a read-only
    tool is still invoked once per node."""
    from agentkit.middlewares.memoize import idempotent
    from agentkit.tools.function import FunctionTool

    calls = {"n": 0}
    read = FunctionTool(
        "lookup",
        lambda a, c: calls.__setitem__("n", calls["n"] + 1) or "row",
        description="look up a row by id; read-only, safe to call repeatedly",
        side_effecting=False,
    )
    ctx = make_test_ctx(
        llm=FakeLLM("x"),
        tool_middleware=[idempotent()],
        store=InMemoryStore(),
        scope=Scope(1, 2),
        correlation_id="wf-ro",
    )
    wf = Workflow(max_steps=5)
    wf.tool("r1", read, args=lambda inp: {"id": 1})
    wf.tool("r2", read, args=lambda inp: {"id": 1}, after="r1")
    res = _run(wf.run("g", ctx))
    assert res.stop_reason == "complete" and calls["n"] == 2


def test_max_steps_overshoot_is_bounded_by_the_widest_wave():
    """`max_steps` is checked BETWEEN waves, so a wave runs whole once started
    and the count can overshoot by up to (widest wave - 1). Measured:
    `max_steps=2` with three independent roots runs all three and reports
    `steps=3`.

    Deliberate — stopping mid-wave drops siblings that are already running —
    and bounded, because the graph's width is static. This test exists so the
    overshoot stays a known property with a known limit, rather than something
    a future change quietly widens.
    """
    ran = []
    wf = Workflow(max_steps=2)
    for name in "abc":
        wf.fn(name, (lambda n: lambda goal, ctx=None, **kw: (ran.append(n), n)[1])(name))

    res = _run(wf.run("g", make_test_ctx(llm=FakeLLM("ok"))))

    assert sorted(ran) == ["a", "b", "c"], "a started wave must not drop siblings"
    assert res.steps == 3
    assert res.steps - 2 <= 3 - 1, "the overshoot must not exceed (widest wave - 1)"
    assert res.stop_reason == "complete"


def test_a_narrow_graph_never_overshoots():
    """POSITIVE CONTROL: with one node per wave the ceiling is exact, which is
    what makes the overshoot above attributable to WIDTH and nothing else."""
    ran = []
    wf = Workflow(max_steps=2)
    wf.fn("a", lambda goal, ctx=None, **kw: (ran.append("a"), "a")[1])
    wf.fn("b", lambda goal, ctx=None, **kw: (ran.append("b"), "b")[1], after="a")
    wf.fn("c", lambda goal, ctx=None, **kw: (ran.append("c"), "c")[1], after="b")

    res = _run(wf.run("g", make_test_ctx(llm=FakeLLM("ok"))))

    assert res.steps == 2 and res.stop_reason == "max_steps"
    assert ran == ["a", "b"], "the third wave must not start"


def test_resume_can_commit_into_a_restored_frozen_checkpoint_state():
    """A resumed workflow must be able to record new node outputs.

    `Checkpoint.state` is deep-frozen, so the map restored from it is a
    `FrozenDict`. Both resume paths deep-copy it intending a MUTABLE working
    copy — but `FrozenDict.__deepcopy__` faithfully returns another FrozenDict,
    so the copy was frozen too and the `done[node.name] = out` commit raised
    `TypeError` on every resume, against a real store as well as a fake one.

    That is this fix's own bug class firing in a consumer: workflow resume was
    rewriting a restored durable record in memory, and freezing the record is
    what made it visible. Fixed by unwrapping the top level; the node outputs
    stay frozen, which is correct.
    """
    from agentkit.kernel._frozen import FrozenDict, deep_freeze

    frozen_state = deep_freeze({"done": {"a": {"nested": [1]}}, "usage": None, "steps": 1})
    assert isinstance(frozen_state["done"], FrozenDict), "premise: the state is frozen"

    # The exact expression both resume paths use, then the commit they perform.
    working = dict(copy.deepcopy(frozen_state["done"]))
    working["b"] = "new output"  # must NOT raise

    assert working["b"] == "new output"
    assert working["a"] == {"nested": [1]}, "restored outputs survive unchanged"
    # The restored VALUES stay frozen — they came from a durable record.
    with pytest.raises(TypeError):
        working["a"]["nested"].append(2)
    # ...and the original checkpoint state was not touched by any of this.
    assert "b" not in frozen_state["done"]
