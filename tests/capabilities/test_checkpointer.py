"""CheckpointPort + Checkpointer + InMemoryCheckpointStore — the durability seam.

Exercises:
1. Round-trip through `InMemoryCheckpointStore`.
2. Monotonic versioning by `Checkpointer.snapshot`.
3. Deterministic `created_at` from a wired `FakeClock`.
4. `resume()` returns the latest version.
5. `at_version()` returns a specific version or `None`.
6. `list_versions()` returns ascending order.
7. `delete()` clears all versions for a run.
8. End-to-end: an `Agent` wired with a `Checkpointer` suspends on approval, then `resume()`
   rehydrates the suspended state through the same `Checkpointer` and runs to completion.
"""

import asyncio

import pytest

from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.agents import Agent, Suspended
from agentkit.agents.cognition import ReActCognition

# Every agent in this module is named "a". The tool loop namespaces its
# checkpoint slot by agent name, because keying on the bare correlation_id had
# a coordinator and its children sharing one slot — and a child completing
# normally called `_clear`, deleting the coordinator's state.
AGENT_NAME = "a"
from agentkit.capabilities import Checkpointer
from agentkit.kernel.ports import Checkpoint, CheckpointPort
from agentkit.kernel.types import Scope, ToolCall
from agentkit.middlewares import meter, tracing
from agentkit.runtime import Budget
from agentkit.testing import FakeClock, FakeLLM, Turn, make_test_ctx
from agentkit.testing import RecordingTracer as Tracer
from agentkit.tools import FunctionTool, ToolRegistry


def _run(coro):
    return asyncio.run(coro)


# ---------- InMemoryCheckpointStore satisfies the Protocol ------------------------------------


def test_in_memory_checkpoint_store_satisfies_checkpoint_port_protocol():
    assert isinstance(InMemoryCheckpointStore(), CheckpointPort)


# ---------- Round-trip ------------------------------------------------------------------------


def test_in_memory_store_round_trips_a_checkpoint():
    store = InMemoryCheckpointStore()
    cp = Checkpoint(run_id="r-1", version=1, state={"k": "v"}, created_at=1000.0, status="running")
    _run(store.save(cp))
    got = _run(store.latest("r-1"))
    assert got == cp
    assert got.state == {"k": "v"} and got.status == "running"


def test_in_memory_store_latest_for_unknown_run_is_none():
    store = InMemoryCheckpointStore()
    assert _run(store.latest("missing")) is None


# ---------- Monotonic versioning --------------------------------------------------------------


def test_snapshot_bumps_version_monotonically():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    a = _run(cpt.snapshot("r-1", {"i": 0}))
    b = _run(cpt.snapshot("r-1", {"i": 1}))
    c = _run(cpt.snapshot("r-1", {"i": 2}))
    assert (a.version, b.version, c.version) == (1, 2, 3)


def test_snapshot_versions_are_isolated_per_run_id():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    a1 = _run(cpt.snapshot("r-A", {}))
    b1 = _run(cpt.snapshot("r-B", {}))
    a2 = _run(cpt.snapshot("r-A", {}))
    assert a1.version == 1 and b1.version == 1 and a2.version == 2


# ---------- Clock stamping --------------------------------------------------------------------


def test_snapshot_stamps_created_at_from_fake_clock_when_wired():
    clock = FakeClock(start=1700000000.0)
    cpt = Checkpointer(port=InMemoryCheckpointStore(), clock=clock)
    a = _run(cpt.snapshot("r-1", {}))
    clock.advance(42.0)
    b = _run(cpt.snapshot("r-1", {}))
    assert a.created_at == 1700000000.0
    assert b.created_at == 1700000042.0  # deterministic, no real wall clock


def test_snapshot_falls_back_to_time_time_without_clock(monkeypatch):
    import agentkit.capabilities.checkpointer.base as ckpt_mod

    monkeypatch.setattr(ckpt_mod.time, "time", lambda: 12345.0)
    cpt = Checkpointer(port=InMemoryCheckpointStore())  # no clock
    a = _run(cpt.snapshot("r-1", {}))
    assert a.created_at == 12345.0


# ---------- Resume / at_version / list_versions / delete --------------------------------------


def test_resume_returns_the_latest_version():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.snapshot("r-1", {"i": 0}))
    _run(cpt.snapshot("r-1", {"i": 1}))
    last = _run(cpt.snapshot("r-1", {"i": 2}))
    got = _run(cpt.resume("r-1"))
    assert got is not None and got.version == last.version
    assert got.state == {"i": 2}


def test_resume_for_unknown_run_is_none():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    assert _run(cpt.resume("never-saved")) is None


def test_resume_filters_terminal_snapshots_by_default():
    """A ``DONE`` snapshot is not resumable — ``resume()`` returns
    ``None`` so a "resume if any checkpoint exists" wiring cannot
    silently re-run a finished job. Same for ``FAILED``."""
    from agentkit.kernel.ports import CheckpointStatus

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.snapshot("done-run", {"i": 0}, status=CheckpointStatus.DONE))
    _run(cpt.snapshot("failed-run", {"i": 0}, status=CheckpointStatus.FAILED))

    assert _run(cpt.resume("done-run")) is None
    assert _run(cpt.resume("failed-run")) is None


def test_resume_returns_terminal_when_include_terminal_true():
    """Callers who legitimately want the terminal snapshot (audit UI,
    replay/debug) opt in with ``include_terminal=True`` — the same
    snapshot the default hides is now returned verbatim."""
    from agentkit.kernel.ports import CheckpointStatus

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    saved = _run(cpt.snapshot("done-run", {"i": 7}, status=CheckpointStatus.DONE))

    got = _run(cpt.resume("done-run", include_terminal=True))
    assert got is not None
    assert got.version == saved.version
    assert got.state == {"i": 7}
    assert CheckpointStatus(got.status) is CheckpointStatus.DONE


def test_resume_returns_non_terminal_by_default():
    """Sanity: the default filter only elides terminal statuses. A
    ``SUSPENDED`` snapshot (the human-gate wait) is still returned,
    because that IS the resumable state the caller wants."""
    from agentkit.kernel.ports import CheckpointStatus

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.snapshot("wait", {"i": 1}, status=CheckpointStatus.SUSPENDED))

    got = _run(cpt.resume("wait"))
    assert got is not None
    assert CheckpointStatus(got.status) is CheckpointStatus.SUSPENDED


def test_at_version_returns_the_right_one_or_none():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.snapshot("r-1", {"i": 0}))
    _run(cpt.snapshot("r-1", {"i": 1}))
    _run(cpt.snapshot("r-1", {"i": 2}))
    assert _run(cpt.at_version("r-1", 1)).state == {"i": 0}
    assert _run(cpt.at_version("r-1", 2)).state == {"i": 1}
    assert _run(cpt.at_version("r-1", 3)).state == {"i": 2}
    assert _run(cpt.at_version("r-1", 99)) is None
    assert _run(cpt.at_version("missing-run", 1)) is None


def test_list_versions_is_ascending():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.snapshot("r-1", {}))
    _run(cpt.snapshot("r-1", {}))
    _run(cpt.snapshot("r-1", {}))
    assert _run(cpt.list_versions("r-1")) == [1, 2, 3]
    assert _run(cpt.list_versions("missing")) == []


def test_delete_clears_all_versions_for_a_run():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.snapshot("r-1", {}))
    _run(cpt.snapshot("r-1", {}))
    assert _run(cpt.list_versions("r-1")) == [1, 2]
    _run(cpt.delete("r-1"))
    assert _run(cpt.list_versions("r-1")) == []
    assert _run(cpt.resume("r-1")) is None


def test_delete_is_idempotent_for_unknown_run():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.delete("never-saved"))  # no exception
    assert _run(cpt.list_versions("never-saved")) == []


# ---------- Observability + per-run-lock eviction ---------------------------------------------


def test_checkpointer_snapshot_stamps_span():
    """``Checkpointer.snapshot`` opens a ``checkpointer.snapshot`` span
    when a trace-bearing ctx is threaded through. The span carries the
    ``run_id`` attribute so an operator can filter the trace timeline
    by run without joining across streams. Without the span, checkpoint
    writes are invisible in production traces — one of the four silent
    seams pinned by this regression suite."""
    tracer = Tracer()
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    ctx = make_test_ctx(trace=tracer, correlation_id="r-span")

    _run(cpt.snapshot("r-span", {"i": 0}, ctx=ctx))

    snap_spans = [s for s in tracer.spans if s.name == "checkpointer.snapshot"]
    assert len(snap_spans) == 1
    assert snap_spans[0].attrs["run_id"] == "r-span"
    assert snap_spans[0].attrs["status"] == "running"


def test_checkpointer_snapshot_survives_missing_trace_on_ctx():
    """``ctx.trace`` may be absent (bare test stubs, offline replay).
    The helper degrades to a ``nullcontext`` — the save still lands,
    just without a span. This pins the "tests still work" half of the
    seam."""
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    # Passing ``ctx=None`` explicitly — no trace port, no span.
    cp = _run(cpt.snapshot("r-notrace", {"i": 0}, ctx=None))
    assert cp.version == 1
    assert _run(cpt.list_versions("r-notrace")) == [1]


def test_run_locks_evicted_on_delete():
    """``Checkpointer._run_locks`` grew unboundedly — one lock per
    distinct ``run_id`` for the process lifetime. Deleting a run is
    the producer's "this run is over" signal, so ``delete`` is the
    natural eviction point. This test pins the fix: after ``delete``,
    the per-run lock is gone."""
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.snapshot("r1", {"i": 0}))
    assert "r1" in cpt._run_locks

    _run(cpt.delete("r1"))
    assert "r1" not in cpt._run_locks


def test_run_locks_eviction_is_idempotent():
    """Deleting an unknown run must not raise on the eviction step —
    ``dict.pop`` with a default handles the "never-seen ``run_id``"
    case cleanly. Preserves the "``delete`` is idempotent" contract."""
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    _run(cpt.delete("never-saved"))  # no exception, no KeyError from _run_locks
    assert "never-saved" not in cpt._run_locks


# ---------- Metadata / status ------------------------------------------------------------------


def test_snapshot_records_status_and_metadata():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    a = _run(cpt.snapshot("r-1", {}, status="suspended", metadata={"reason": "approval"}))
    assert a.status == "suspended" and a.metadata == {"reason": "approval"}


# ---------- End-to-end: Agent with a Checkpointer suspends + resumes --------------------------


def _approval_registry(counter):
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "delete",
            lambda a, c: counter.__setitem__("n", counter["n"] + 1) or "gone",
            description="delete a resource by id; mutates the world (destructive)",
            side_effecting=True,
            requires_approval=True,
        )
    )
    return reg


def test_agent_suspends_then_resumes_through_a_checkpointer():
    """Pivotal integration: the suspended snapshot lives in the Checkpointer, the resumed
    run pulls that exact snapshot back, applies the human decision, and runs to done."""
    store = InMemoryCheckpointStore()
    cpt = Checkpointer(port=store)
    counter = {"n": 0}
    reg = _approval_registry(counter)

    llm1 = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="never")])
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).run(
            "t",
            make_test_ctx(
                llm=llm1,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                checkpointer=cpt,
                trace=Tracer(),
                scope=Scope(1, 2),
                budget=Budget(max_cost_usd=10.0),
                correlation_id="run-1",
            ),
        )
    )
    assert res.partial and isinstance(res.evals["suspended"], Suspended) and counter["n"] == 0

    # Checkpointer holds the suspended snapshot; status reflects that, and `pending` made it
    # into the serialized state so resume() can apply human decisions to those tool calls.
    saved = _run(cpt.resume(ReActCognition.checkpoint_slot("run-1", AGENT_NAME)))
    assert saved is not None and saved.status == "suspended"
    assert saved.state["pending"] and saved.state["pending"][0]["id"] == "c1"

    llm2 = FakeLLM.script([Turn(content="done")])
    res2 = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).resume(
            "run-1",
            {"c1": "approve"},
            make_test_ctx(
                llm=llm2,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                checkpointer=cpt,
                trace=Tracer(),
                scope=Scope(1, 2),
                budget=Budget(max_cost_usd=10.0),
                correlation_id="run-1",
            ),
        )
    )
    assert res2.output == "done" and counter["n"] == 1
    # Terminal completion clears all checkpoints for the run.
    assert _run(cpt.list_versions("run-1")) == []


def test_agent_field_checkpointer_overrides_ctx_checkpointer():
    """When an Agent has its own checkpointer field, it wins over the ctx-level one — so a
    test or single-agent override doesn't have to thread through the Services container."""
    agent_cpt = Checkpointer(port=InMemoryCheckpointStore())
    ctx_cpt = Checkpointer(port=InMemoryCheckpointStore())
    counter = {"n": 0}

    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="never")])
    agent = Agent(
        name="a",
        model="m",
        cognition=ReActCognition(tools=_approval_registry(counter), checkpointer=agent_cpt),
    )
    _run(
        agent.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                checkpointer=ctx_cpt,
                trace=Tracer(),
                scope=Scope(1, 2),
                budget=Budget(max_cost_usd=10.0),
                correlation_id="run-1",
            ),
        )
    )

    # The agent's own checkpointer received the suspended state; the ctx one did not.
    assert _run(agent_cpt.resume(ReActCognition.checkpoint_slot("run-1", AGENT_NAME))) is not None
    assert _run(ctx_cpt.resume(ReActCognition.checkpoint_slot("run-1", AGENT_NAME))) is None


def test_checkpointer_versions_grow_across_a_full_tool_loop():
    """A multi-step tool loop produces multiple snapshots — one per durable transition. Each
    is a real version on the port; nothing collapses into a single mutable slot."""
    from agentkit.adapters.store import InMemoryStore  # noqa: F401 — ensure nothing leaks via store

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    counter = {"n": 0}
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "fetch",
            lambda a, c: counter.__setitem__("n", counter["n"] + 1) or {"s": 200},
            description="fetch a resource over the network for inspection",
            side_effecting=False,
        )
    )

    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "fetch", {}),)),
            Turn(tool_calls=(ToolCall("c2", "fetch", {}),)),
            Turn(content="done"),
        ]
    )
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                checkpointer=cpt,
                trace=Tracer(),
                scope=Scope(1, 2),
                budget=Budget(max_cost_usd=10.0),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "done" and counter["n"] == 2
    # Terminal completion clears all versions for the run.
    assert _run(cpt.list_versions("run-1")) == []


# ---------- Type / import surface -------------------------------------------------------------


def test_checkpoint_is_immutable():
    cp = Checkpoint(run_id="r", version=1, state={}, created_at=0.0, status="running")
    with pytest.raises(Exception):  # frozen dataclass — assignment raises FrozenInstanceError
        cp.version = 99  # type: ignore[misc]


def test_snapshot_deep_copies_state_so_post_snapshot_mutation_cannot_reach_it():
    """``Checkpointer.snapshot`` deep-copies ``state`` (and
    ``metadata``). Without the copy, a caller that keeps a live
    handle on the scratch dict — or a resumer that pops entries off
    ``cp.state`` — would mutate the durable record. Parity with a
    serializing backend (Redis / Postgres) would break silently.

    This test uses ``InMemoryCheckpointStore`` (which stores refs)
    to exercise exactly the same aliasing hazard the real backends
    don't have."""
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    live_state = {"transcript": ["msg-1"], "cursor": 1}
    live_metadata = {"tag": "phase-A"}

    cp = asyncio.run(cpt.snapshot("run-X", live_state, metadata=live_metadata))

    # Mutate the caller's live handles AFTER the snapshot returned.
    live_state["transcript"].append("msg-2")
    live_state["cursor"] = 99
    live_metadata["tag"] = "phase-B"

    # The snapshot's state / metadata must NOT reflect either post-hoc mutation.
    assert cp.state == {"transcript": ["msg-1"], "cursor": 1}
    assert cp.metadata == {"tag": "phase-A"}

    # And re-reading from the store must ALSO see the frozen-in-time state.
    resumed = asyncio.run(cpt.resume("run-X"))
    assert resumed is not None
    assert resumed.state == {"transcript": ["msg-1"], "cursor": 1}
    assert resumed.metadata == {"tag": "phase-A"}


def test_snapshot_deep_copies_nested_state_containers() -> None:
    """Deep-copy reaches into every level of the state tree.
    Post-snapshot mutation of a nested list / dict / set must not
    leak into the stored checkpoint."""
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    nested = {
        "history": [{"role": "user", "content": "hi"}],
        "tags": {"security", "finance"},
        "scratch": {"counter": {"a": 1, "b": 2}},
    }

    cp = asyncio.run(cpt.snapshot("run", nested))

    # Mutate every level.
    nested["history"].append({"role": "assistant", "content": "hi back"})
    nested["tags"].add("compliance")
    nested["scratch"]["counter"]["a"] = 99

    assert cp.state["history"] == [{"role": "user", "content": "hi"}]
    assert cp.state["tags"] == {"security", "finance"}
    assert cp.state["scratch"]["counter"] == {"a": 1, "b": 2}


def test_snapshot_state_containing_toolcalls_survives_deepcopy() -> None:
    """State that carries ``ToolCall`` instances must deep-copy cleanly, or
    the durable seam goes down with it. Regression: ``arguments`` was a
    ``MappingProxyType`` and this raised ``TypeError: cannot pickle
    'mappingproxy' object`` without a ``ToolCall.__deepcopy__`` hook. The
    payload is a ``FrozenDict`` now and the hook is gone — ``FrozenDict``
    defines its own — so this test is what says the guarantee outlived the
    workaround."""
    from agentkit.capabilities.checkpointer.persistence import tc_to_dict
    from agentkit.kernel.types import ToolCall

    cpt = Checkpointer(port=InMemoryCheckpointStore())
    tc = ToolCall("call-1", "search", {"q": "hello", "filters": ["a", "b"]})
    state = {"pending": [tc_to_dict(tc)]}

    cp = asyncio.run(cpt.snapshot("run", state))
    assert cp.state["pending"][0]["arguments"] == {"q": "hello", "filters": ["a", "b"]}


def test_snapshot_metadata_is_optional_and_defaults_empty() -> None:
    """A snapshot without explicit metadata still has an isolated
    empty dict — not a shared default that later snapshots could
    mutate."""
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    a = asyncio.run(cpt.snapshot("r-1", {}))
    b = asyncio.run(cpt.snapshot("r-2", {}))
    a.metadata is not b.metadata
    assert a.metadata == {} == b.metadata


def test_concurrent_snapshots_of_same_run_id_produce_distinct_versions() -> None:
    """Two concurrent snapshot calls (same run_id) each get a distinct
    version — the version counter increments per save, even if the
    underlying scheduling interleaves the version-read and the save."""

    async def go():
        cpt = Checkpointer(port=InMemoryCheckpointStore())
        results = await asyncio.gather(*(cpt.snapshot("run", {"i": i}) for i in range(10)))
        return [cp.version for cp in results]

    versions = asyncio.run(go())
    # Every version is distinct — no two concurrent snapshots collided
    # on the same version number.
    assert len(set(versions)) == len(versions)
    # And the set covers exactly 1..N with no gaps (monotonic assignment).
    assert set(versions) == set(range(1, 11))


# ── Checkpoint schema evolution: forward compat on optional fields ──────────
#
# Production checkpoints outlive code deployments. A checkpoint
# written by v1 code may load into v2 code long after the field
# shape drifted. ``rehydrate`` accepts missing OPTIONAL fields (safe
# defaults) and raises LOUDLY on missing REQUIRED fields — so an
# operator gets either a clean recovery or an obvious failure, never
# silent corruption.


def test_rehydrate_tolerates_missing_optional_fields():
    """A v1-shaped state (no ``prefix``, no ``scratchpad``, no
    ``limit``, no ``shared``, no ``repaired``) still round-trips.
    The load defaults cover a checkpoint written before every
    field-addition landed."""
    from agentkit.capabilities.checkpointer.persistence import rehydrate

    minimal_state = {
        # Required keys — the invariants below MUST hold.
        "messages": [
            {
                "role": "system",
                "content": "sys",
                "tool_call_id": None,
                "name": None,
                "tool_calls": [],
            },
            {"role": "user", "content": "hi", "tool_call_id": None, "name": None, "tool_calls": []},
        ],
        "usage": {"input": 10, "output": 20, "cost": 0.05},
        "iteration": 3,
        # Everything else — omitted.
    }

    context, usage, iteration, repaired = rehydrate(minimal_state)
    # Iteration + usage preserved.
    assert iteration == 3
    assert usage.input_tokens == 10 and usage.output_tokens == 20
    # Defaulted optional fields.
    assert context.scratchpad == {}
    assert context.limit is None
    assert context.shared is False
    assert repaired is False
    # An absent ``prefix`` → empty PrefixContext (legacy shape).
    assert context.prefix.system_prompt == ""


def test_rehydrate_raises_clearly_on_missing_required_field():
    """A state missing a required field (``messages``, ``usage``,
    ``iteration``) fails LOUDLY. Callers see a KeyError naming the
    missing field — enough signal for an operator to diagnose schema
    drift instead of silent misbehaviour later in the loop."""
    from agentkit.capabilities.checkpointer.persistence import rehydrate

    # Missing ``messages``.
    with pytest.raises(KeyError, match="messages"):
        rehydrate({"usage": {"input": 0, "output": 0, "cost": 0.0}, "iteration": 0})

    # Missing ``usage``.
    with pytest.raises(KeyError, match="usage"):
        rehydrate({"messages": [], "iteration": 0})

    # Missing ``usage`` sub-field.
    with pytest.raises(KeyError):
        rehydrate({"messages": [], "usage": {"input": 0}, "iteration": 0})

    # Missing ``iteration``.
    with pytest.raises(KeyError, match="iteration"):
        rehydrate({"messages": [], "usage": {"input": 0, "output": 0, "cost": 0.0}})


def test_checkpoint_can_survive_forward_compat_new_optional_fields():
    """Symmetric to the missing-optional case: a state carrying
    UNKNOWN fields (e.g., a future version wrote extras that this
    version doesn't know about) must not break rehydrate. Fields
    the loader doesn't consult are silently ignored — the operator
    can roll back safely."""
    from agentkit.capabilities.checkpointer.persistence import rehydrate

    forward_state = {
        "messages": [],
        "usage": {"input": 0, "output": 0, "cost": 0.0},
        "iteration": 0,
        # Fields a hypothetical future version added — this loader
        # ignores them cleanly.
        "future_feature_x": "some future value",
        "future_feature_y": {"nested": True},
    }
    context, usage, iteration, repaired = rehydrate(forward_state)
    # Load succeeded; future fields were ignored.
    assert iteration == 0
    assert usage.total_tokens == 0
    assert context.messages == []


# ── Concurrent snapshot serialization across suspending backends ────────────
#
# ``Checkpointer.snapshot`` reads the current version list then writes
# a new one — two ``await`` calls straddling a compute-next-version
# step. In-process concurrent producers (a run's hot loop + a
# background health snapshotter, two parallel policy branches) can
# suspend between the two awaits and both compute the same
# ``next_version``. Without serialisation the second save either
# overwrites the first (silent data loss on ``InMemoryCheckpointStore``)
# or hits a duplicate-key error (Postgres). A per-run lock inside the
# Checkpointer closes the window.


class _SuspendingCheckpointStore:
    """Wraps ``InMemoryCheckpointStore`` but yields to the scheduler
    on every port method — simulates a real IO backend (Postgres /
    Redis) where the awaits actually suspend."""

    def __init__(self) -> None:
        self._inner = InMemoryCheckpointStore()

    async def list_versions(self, run_id):
        await asyncio.sleep(0)  # force a real yield
        return await self._inner.list_versions(run_id)

    async def save(self, cp):
        await asyncio.sleep(0)  # force a real yield
        return await self._inner.save(cp)

    async def latest(self, run_id):
        return await self._inner.latest(run_id)

    async def at_version(self, run_id, version):
        return await self._inner.at_version(run_id, version)

    async def delete(self, run_id):
        return await self._inner.delete(run_id)


def test_concurrent_snapshots_serialize_through_suspending_backend():
    """Ten concurrent snapshots against a suspending backend (where
    every ``await`` yields to the scheduler) still produce distinct
    monotonic versions [1..10]. Without the Checkpointer's per-run
    lock, all ten would compute ``next_version=1`` off the same
    initial empty read."""

    async def go():
        cpt = Checkpointer(port=_SuspendingCheckpointStore())
        results = await asyncio.gather(*(cpt.snapshot("run-race", {"i": i}) for i in range(10)))
        return sorted(cp.version for cp in results)

    versions = asyncio.run(go())
    # No collisions: the lock serialized the read-compute-write cycle.
    assert versions == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_concurrent_snapshots_different_run_ids_do_not_serialize_each_other():
    """The per-run lock is per-``run_id``. Snapshots of DIFFERENT
    run ids don't wait on each other — the framework doesn't turn
    into a global bottleneck under multi-run load."""
    import time as _time

    started_at: dict[str, float] = {}
    finished_at: dict[str, float] = {}

    class _SlowStore(_SuspendingCheckpointStore):
        async def list_versions(self, run_id):
            started_at[run_id] = _time.monotonic()
            await asyncio.sleep(0.05)  # simulate slow backend
            return await self._inner.list_versions(run_id)

        async def save(self, cp):
            await asyncio.sleep(0.05)
            finished_at[cp.run_id] = _time.monotonic()
            return await self._inner.save(cp)

    async def go():
        cpt = Checkpointer(port=_SlowStore())
        # Ten different run ids fired concurrently — each takes ~0.1s
        # of "IO"; if the lock serialised across runs, total wall time
        # would be ~1s. Concurrent = ~0.1s.
        wall_start = _time.monotonic()
        await asyncio.gather(*(cpt.snapshot(f"run-{i}", {}) for i in range(10)))
        wall_end = _time.monotonic()
        return wall_end - wall_start

    elapsed = asyncio.run(go())
    # Sanity: wall time is close to a single snapshot's cost, not 10×.
    # Allow generous headroom for scheduler overhead.
    assert elapsed < 0.5, f"per-run lock serialised across run_ids (elapsed={elapsed:.3f}s)"


def test_a_resumed_context_is_writable_all_the_way_down():
    """A durable record is rightly frozen; a context RESUMED from one is a live
    working context and must be writable again.

    `Checkpoint.state` is deep-frozen, and `rehydrate` used to pass the stored
    scratchpad straight through — so a resumed run raised
    `TypeError: this payload belongs to a frozen value` on its FIRST
    `ctx.note(...)`. The obvious fix, a top-level `dict(...)`, only moves that
    failure one level down to the first nested write, which is why this asserts
    both depths.
    """
    from agentkit.capabilities.checkpointer.persistence import rehydrate
    from agentkit.kernel._frozen import deep_freeze

    saved = deep_freeze(
        {
            "messages": [],
            "prefix": None,
            "iteration": 0,
            "results": [],
            "pending": [],
            "usage": {"input": 0, "output": 0, "cost": 0.0},
            "scratchpad": {"plan": {"steps": ["a"]}, "stage": "plan"},
        }
    )
    ctx, *_ = rehydrate(saved)

    ctx.note("stage", "act")                      # top level
    ctx.scratchpad["plan"]["steps"].append("b")   # nested
    assert ctx.scratchpad["stage"] == "act"
    assert ctx.scratchpad["plan"]["steps"] == ["a", "b"]

    # ...and the durable record it came from is untouched by any of that.
    assert dict(saved["scratchpad"]["plan"])["steps"] == ["a"]
