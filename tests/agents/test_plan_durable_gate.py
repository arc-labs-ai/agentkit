"""A ``PlanPolicy`` human gate is durable on the persistence people deploy.

Three defects, all in the same seam:

1. The checkpoint was written by putting LIVE dataclasses into ``ctx.store.set``
   — ``Step``, ``Usage``, ``AgentResult``. ``InMemoryStore`` holds objects, so
   the whole feature tested green while any serializing store raised
   ``TypeError: Object of type Step is not JSON serializable``. The gate did not
   work on a ``FileStore``, and would not have worked on Redis or Postgres.
2. ``resume`` required ``ctx.store``, so a ``Services(checkpointer=...)``
   wiring — the documented durable seam — could suspend but never resume.
3. Reaching a gate with NO seam wired returned a ``Suspended`` in silence: the
   caller got a run id whose every ``resume`` raises "no suspended plan".

The fix routes the plan through the one shared ``Checkpointer`` resolution
order, at its own namespaced slot, with a JSON-safe encoding applied
unconditionally so an in-memory test cannot pass where a real store fails.
"""

from __future__ import annotations

import asyncio
import json
import tempfile

import pytest

from agentkit import Agent
from agentkit.adapters.checkpoint.in_memory import InMemoryCheckpointStore
from agentkit.adapters.store import InMemoryStore
from agentkit.adapters.store.file import FileStore
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.plan import (
    PlanPolicy,
    StaticPlanner,
    Step,
    _ckpt_key,
    _encode_plan_state,
    checkpoint_slot,
)
from agentkit.agents.result import AgentResult
from agentkit.capabilities.checkpointer import Checkpointer
from agentkit.kernel.errors import Failure
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.resilience import ErrorClass
from agentkit.kernel.types import Usage
from agentkit.testing import FakeLLM, make_test_ctx


class _Rec:
    def __init__(self, name: str, out: str = "OUT") -> None:
        self.name, self.output, self.calls = name, out, []

    async def run(self, task, ctx, *, context=None):  # noqa: ANN001, ANN202
        self.calls.append(task)
        return AgentResult(output=self.output, usage=Usage(3, 4, 0.25))


def _plan() -> list[Step]:
    return [Step("r1", "q", group=0), Step.gate("review", group=1), Step("synth", "go", group=2)]


def _coord(policy: PlanPolicy, children: dict) -> Agent:
    return Agent(name="c", cognition=CoordinatorCognition(children=children, policy=policy))


# ── 1. a real, serializing store ────────────────────────────────────────────


def test_a_gate_survives_a_serializing_store() -> None:
    """THE regression: suspend + resume against a ``FileStore``, which is JSON
    on disk. This raised ``TypeError: Object of type Step is not JSON
    serializable`` before the encoding existed."""
    with tempfile.TemporaryDirectory() as d:
        synth = _Rec("synth")
        policy = PlanPolicy(planner=StaticPlanner(_plan()))
        coord = _coord(policy, {"r1": _Rec("r1"), "synth": synth})
        ctx = make_test_ctx(llm=FakeLLM("ok"), store=FileStore(d), correlation_id="fs1")

        res = asyncio.run(coord.run("goal", ctx))
        assert res.is_suspended and synth.calls == []

        resumed = asyncio.run(policy.resume(coord, {"review": "approve"}, ctx))
        assert resumed.stop_reason == "complete"
        assert synth.calls == ["go"]
        # Usage accumulated ACROSS the suspend — the pre-gate spend survived the
        # wire, which is the whole point of encoding it rather than dropping it.
        assert resumed.usage.total_tokens == 14  # (3+4) per child, two children


def test_the_persisted_payload_is_plain_json() -> None:
    """Directly assert the wire shape rather than only that a round trip works:
    a payload that happens to survive one store must survive every store."""
    payload = _encode_plan_state(
        task="goal",
        steps=_plan(),
        results=[AgentResult(output="a", usage=Usage(1, 2, 0.5))],
        errors=[("ghost", Failure(category=ErrorClass.PERMANENT, source="PlanPolicy", message="m"))],
        usage=Usage(1, 2, 0.5),
        gate_group=1,
    )
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["steps"][1]["gate_name"] == "review"
    assert round_tripped["errors"][0] == {
        "agent": "ghost",
        "category": "permanent",
        "source": "PlanPolicy",
        "message": "m",
        "retriable": False,
    }
    assert round_tripped["usage"] == {"input": 1, "output": 2, "cost": 0.5}
    assert round_tripped["v"] == 1


def test_an_in_memory_store_uses_the_same_encoding() -> None:
    """The encoding is unconditional. If it were applied only for serializing
    backends, every in-memory test would keep passing over a broken wire format
    — which is exactly how this bug survived."""
    policy = PlanPolicy(planner=StaticPlanner(_plan()))
    store = InMemoryStore()
    coord = _coord(policy, {"r1": _Rec("r1"), "synth": _Rec("synth")})
    ctx = make_test_ctx(llm=FakeLLM("ok"), store=store, correlation_id="mem1")
    asyncio.run(coord.run("goal", ctx))

    record = asyncio.run(store.get(f"checkpoint:{checkpoint_slot('mem1')}"))
    json.dumps(record)  # would raise on a live-object payload
    assert isinstance(record["state"]["steps"][0], dict)


# ── 2. a checkpointer-only wiring ───────────────────────────────────────────


def test_a_checkpointer_only_wiring_can_suspend_and_resume() -> None:
    """``Services(checkpointer=...)`` is the documented durable seam. Requiring
    ``ctx.store`` here was the same asymmetry already fixed for ``Workflow``:
    suspend worked, resume raised."""
    cp = Checkpointer(port=InMemoryCheckpointStore())
    synth = _Rec("synth")
    policy = PlanPolicy(planner=StaticPlanner(_plan()))
    coord = _coord(policy, {"r1": _Rec("r1"), "synth": synth})
    ctx = make_test_ctx(llm=FakeLLM("ok"), checkpointer=cp, correlation_id="cp1")

    assert asyncio.run(coord.run("goal", ctx)).is_suspended
    assert asyncio.run(policy.resume(coord, {"review": "approve"}, ctx)).stop_reason == "complete"
    assert synth.calls == ["go"]


def test_the_slot_is_suspended_not_running() -> None:
    """``Checkpointer.resume`` returns ``None`` for terminal statuses, and an
    auto-resume supervisor needs "waiting on a human" to be distinguishable from
    "engine in motion"."""
    cp = Checkpointer(port=InMemoryCheckpointStore())
    policy = PlanPolicy(planner=StaticPlanner(_plan()))
    coord = _coord(policy, {"r1": _Rec("r1"), "synth": _Rec("synth")})
    ctx = make_test_ctx(llm=FakeLLM("ok"), checkpointer=cp, correlation_id="cp2")
    asyncio.run(coord.run("goal", ctx))

    saved = asyncio.run(cp.resume(checkpoint_slot("cp2")))
    assert saved is not None
    assert CheckpointStatus(saved.status) is CheckpointStatus.SUSPENDED


# ── 3. no seam at all ───────────────────────────────────────────────────────


def test_a_gate_with_no_durable_seam_warns() -> None:
    """The suspend is real but unrecoverable. Saying so beats handing back a run
    id whose every resume raises."""
    policy = PlanPolicy(planner=StaticPlanner(_plan()))
    coord = _coord(policy, {"r1": _Rec("r1"), "synth": _Rec("synth")})
    ctx = make_test_ctx(llm=FakeLLM("ok"), correlation_id="bare1")  # no store, no checkpointer

    with pytest.warns(UserWarning, match="no durable seam"):
        res = asyncio.run(coord.run("goal", ctx))
    assert res.is_suspended  # still a truthful result — only durability is lost

    with pytest.raises(ValueError, match="requires a durable seam"):
        asyncio.run(policy.resume(coord, {"review": "approve"}, ctx))


# ── 4. the slot is namespaced ───────────────────────────────────────────────


def test_the_plan_slot_cannot_collide_with_a_child_or_a_coordinator() -> None:
    """Namespacing is per producer, exactly as for the tool loop's
    ``{run_id}:agent:{name}``. A bare run id would put a plan in the same slot a
    nested coordinator writes to — and a child completing deletes its slot."""
    assert checkpoint_slot("r") == "r:plan"
    from agentkit.agents.cognition.react import ReActCognition

    assert checkpoint_slot("r") != ReActCognition.checkpoint_slot("r", "worker")
    assert checkpoint_slot("r") != "r"


# ── 5. a record written before the upgrade ──────────────────────────────────


def test_a_legacy_raw_store_record_still_resumes() -> None:
    """A plan that suspended before this change has live objects under the old
    ``plan_policy:<run_id>`` key. It must finish, not crash on a dict lookup
    into a ``Step`` — a suspended run is someone waiting for an answer."""
    synth = _Rec("synth")
    policy = PlanPolicy(planner=StaticPlanner(_plan()))
    coord = _coord(policy, {"r1": _Rec("r1"), "synth": synth})
    store = InMemoryStore()
    ctx = make_test_ctx(llm=FakeLLM("ok"), store=store, correlation_id="old1")

    # Exactly what the old code wrote: no "v", live dataclasses.
    asyncio.run(
        store.set(
            _ckpt_key("old1"),
            {
                "task": "goal",
                "steps": _plan(),
                "results": [AgentResult(output="pre", usage=Usage(1, 1, 0.1))],
                "errors": [],
                "usage": Usage(1, 1, 0.1),
                "gate_group": 1,
            },
        )
    )

    resumed = asyncio.run(policy.resume(coord, {"review": "approve"}, ctx))
    assert resumed.stop_reason == "complete" and synth.calls == ["go"]
    # And the consumed legacy key is cleared, so it cannot be replayed.
    assert asyncio.run(store.get(_ckpt_key("old1"))) is None


# ── 6. an unserialisable child payload degrades, it does not explode ────────


def test_unserialisable_evals_are_dropped_with_a_warning() -> None:
    """A child's ``evals``/``parsed`` can hold anything — a Pydantic model, a
    ``Message`` list from a nested coordinator. Letting the SUSPEND raise would
    lose the entire run at the gate, which is the worst available outcome; the
    plan degrades those two fields instead and says so."""

    class _Opaque:
        pass

    with pytest.warns(UserWarning, match="could not be serialized"):
        payload = _encode_plan_state(
            task="goal",
            steps=_plan(),
            results=[AgentResult(output="a", usage=Usage(), parsed=_Opaque())],
            errors=[],
            usage=Usage(),
            gate_group=1,
        )
    json.dumps(payload)  # the checkpoint is writable
    assert payload["results"][0]["output"] == "a"  # the answer itself survives
    assert "parsed" not in payload["results"][0]


def test_a_serialisable_result_keeps_its_evals() -> None:
    """The positive control: the degradation is conditional, not a blanket
    strip."""
    payload = _encode_plan_state(
        task="goal",
        steps=_plan(),
        results=[AgentResult(output="a", usage=Usage(), evals={"note": "keep me"})],
        errors=[],
        usage=Usage(),
        gate_group=1,
    )
    assert payload["results"][0]["evals"] == {"note": "keep me"}
