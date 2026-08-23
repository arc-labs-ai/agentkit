"""Two producers in one run must not share a checkpoint slot.

Every producer keyed durable state on `ctx.correlation_id`, and `ctx.child()`
propagates that unchanged — so a coordinator and each of its child agents wrote
to the SAME slot. That is not merely an overwrite: a child finishing normally
calls `_clear`, which is `Checkpointer.delete(run_id)`, and that removes ALL
versions for the id.

Measured before the fix: a coordinator wrote its in-progress turn state, one
child completed successfully, and the coordinator's checkpoint was gone — so a
crash then lost a run that HAD checkpointed. Version numbering restarted too,
breaking the monotonic-version guarantee the Checkpointer documents.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit import Agent
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.types import Delta, ToolCall, Usage
from agentkit.testing import make_test_ctx
from agentkit.tools import tool

RUN = "shared-run"


@tool(side_effecting=False)
def probe(q: str) -> str:
    """A read-only tool, so the loop takes a second turn and then clears."""
    return "ok"


class _ToolThenAnswer:
    """One tool turn, then a final answer — which is the path that calls
    `_clear`, and therefore the path that used to delete a sibling's state."""

    async def stream(self, *, messages, tools=None, **_kw):
        ran = any(getattr(m, "role", "") == "tool" for m in messages)
        if not ran:
            yield Delta(text="tooling", model="m", provider="f")
            yield Delta(
                tool_calls=(ToolCall("t1", "probe", {"q": "x"}),),
                usage=Usage(1, 1, 0.0),
                finish_reason="tool_calls",
                model="m",
                provider="f",
            )
        else:
            yield Delta(text="done", model="m", provider="f")
            yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")


def _cp() -> Checkpointer:
    return Checkpointer(port=InMemoryCheckpointStore())


def test_a_child_completing_does_not_delete_the_coordinators_checkpoint() -> None:
    """The load-bearing regression."""
    cp = _cp()
    asyncio.run(
        cp.snapshot(RUN, {"turn": 1, "transcript": []}, status=CheckpointStatus.RUNNING)
    )

    child = Agent("worker", "m", cognition=ReActCognition(tools=[probe], checkpointer=cp))
    ctx = make_test_ctx(llm=_ToolThenAnswer(), checkpointer=cp, correlation_id=RUN)
    assert asyncio.run(child.run("go", ctx)).stop_reason == "complete"

    coordinator_state = asyncio.run(cp.resume(RUN))
    assert coordinator_state is not None, "the child's completion deleted the coordinator's state"
    assert coordinator_state.state["turn"] == 1

    # ...and the child DID clean up its own slot, which is what `_clear` is for.
    slot = ReActCognition.checkpoint_slot(RUN, "worker")
    assert asyncio.run(cp.resume(slot)) is None


def test_two_siblings_do_not_share_a_slot() -> None:
    """Parallel children get their own `ctx.child()` at the same depth and the
    same correlation_id, so keying on the run id alone had them clobbering each
    other as well as their parent."""
    cp = _cp()
    a = ReActCognition.checkpoint_slot(RUN, "alpha")
    b = ReActCognition.checkpoint_slot(RUN, "beta")
    assert a != b

    async def go():
        for name in ("alpha", "beta"):
            agent = Agent(name, "m", cognition=ReActCognition(tools=[probe], checkpointer=cp))
            ctx = make_test_ctx(
                llm=_ToolThenAnswer(), checkpointer=cp, correlation_id=RUN
            )
            # Suspend rather than complete, so each leaves state behind.
            ctx.autonomy = "manual"
            await agent.run("go", ctx)

    asyncio.run(go())
    assert asyncio.run(cp.resume(a)) is not None
    assert asyncio.run(cp.resume(b)) is not None


def test_the_slot_derivation_is_public_and_stable() -> None:
    """An operator tool that lists or clears durable state needs the same
    derivation the loop uses; hardcoding the format in two places is how they
    drift."""
    assert ReActCognition.checkpoint_slot("r", "a") == "r:agent:a"
    assert ReActCognition.checkpoint_slot("r", "a") != ReActCognition.checkpoint_slot("r", "b")


def test_the_tool_loop_never_reads_the_bare_run_id() -> None:
    """The bare correlation_id is the slot the COORDINATOR policies own, so the
    child loop must not read it — not as a fallback, not ever.

    `_load` used to try the bare id when its own slot came up empty, to rescue
    suspends written before the namespacing landed. The rescue and the hazard
    were the same line of code: a child handed its coordinator's
    `{turn, transcript, results, ...}` dies inside `rehydrate` on the first
    leaf-shaped key it reaches (`KeyError: 'prefix'` today, `KeyError:
    'messages'` before `prefix` joined the blob). A shape check kept the two
    apart; deleting the read deletes the hazard outright, and this pins that
    the fallback stays gone.
    """
    cp = _cp()
    # A coordinator-shaped payload sitting at the bare run id.
    asyncio.run(
        cp.snapshot(RUN, {"turn": 3, "transcript": [], "results": {}},
                    status=CheckpointStatus.RUNNING)
    )

    agent = Agent("worker", "m", cognition=ReActCognition(tools=[probe], checkpointer=cp))
    ctx = make_test_ctx(llm=_ToolThenAnswer(), checkpointer=cp, correlation_id=RUN)
    # A fresh drive starts fresh rather than rehydrating someone else's state.
    result = asyncio.run(agent.run("go", ctx))
    assert result.stop_reason == "complete"
    # And the coordinator's payload is untouched — not read, not overwritten,
    # and not deleted by the child's terminal `_clear`.
    still = asyncio.run(cp.resume(RUN))
    assert still is not None and still.state["turn"] == 3


def test_resume_does_not_fall_back_to_the_bare_run_id_either() -> None:
    """The other door into `_load`. Even a genuinely SUSPENDED record at the
    bare id must not satisfy `Agent.resume` for a child agent — its own slot is
    the only slot, so an empty one is "nothing to resume" and says so."""
    cp = _cp()
    llm = _ToolThenAnswer()
    agent = Agent("worker", "m", cognition=ReActCognition(tools=[probe], checkpointer=cp))

    # Produce a real leaf-shaped suspend, then MOVE it to the bare id.
    ctx = make_test_ctx(llm=llm, checkpointer=cp, correlation_id=RUN, autonomy="manual")
    assert asyncio.run(agent.run("go", ctx)).stop_reason == "suspended"
    slot = ReActCognition.checkpoint_slot(RUN, "worker")
    saved = asyncio.run(cp.resume(slot))
    assert saved is not None
    asyncio.run(cp.delete(slot))
    asyncio.run(cp.snapshot(RUN, saved.state, status=CheckpointStatus.SUSPENDED))

    with pytest.raises(ValueError, match="no suspended run"):
        asyncio.run(
            agent.resume(RUN, {"t1": "approve"},
                         make_test_ctx(llm=llm, checkpointer=cp, correlation_id=RUN))
        )

    # ...while the same suspend in its OWN slot resumes, which is the path that
    # actually has to work.
    asyncio.run(cp.delete(RUN))
    asyncio.run(cp.snapshot(slot, saved.state, status=CheckpointStatus.SUSPENDED))
    resumed = asyncio.run(
        agent.resume(RUN, {"t1": "approve"},
                     make_test_ctx(llm=llm, checkpointer=cp, correlation_id=RUN))
    )
    assert resumed.stop_reason == "complete"


# ── one resolution order, shared by every producer ──────────────────────────


def _coordinator_over(child: Agent, *, turns: int = 1) -> Agent:
    from agentkit.agents.cognition import CoordinatorCognition
    from agentkit.agents.policies import RoundRobinPolicy

    return Agent(
        "boss",
        "m",
        cognition=CoordinatorCognition(
            children={child.name: child}, policy=RoundRobinPolicy(max_turns=turns)
        ),
    )


class _PlainAnswer:
    async def stream(self, **_kw):
        yield Delta(text="answer", model="m", provider="f")
        yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")


def test_a_coordinator_persists_through_a_store_only_wiring() -> None:
    """The coordinator policies had a THIRD checkpointer resolution order,
    stopping at `ctx.checkpointer` and deliberately excluding the store bridge
    because "coordinator runs require a real Checkpointer for durability".

    The bridge is exactly as durable as the store behind it; its only
    documented limitation is a single slot per run with no version history, and
    no policy reads history — they call `resume` for the latest and nothing
    else. Meanwhile the cost was silent: a `Services(store=...)` wiring gave
    durable ReAct runs, durable Workflow gates, and coordinator runs that
    persisted NOTHING. Measured: zero keys in the store after a completed run.
    """
    from agentkit.adapters.store import InMemoryStore
    from agentkit.kernel.types import Scope
    from agentkit.runtime import Invoker, RunContext, Services

    store = InMemoryStore()
    coord = _coordinator_over(Agent("worker", "m"))
    ctx = RunContext(
        "coord-run",
        Scope(),
        services=Services(invoker=Invoker(llm=_PlainAnswer()), store=store),
    )
    asyncio.run(coord.run("go", ctx))

    keys = [k for k in store._kv if "coord-run" in k]
    assert keys, "a store-only wiring left the coordinator run with no durable state"


def test_a_coordinator_and_its_tool_looping_child_hold_distinct_slots() -> None:
    """What makes ONE shared resolution order viable across producers: the tool
    loop namespaces its slot per agent, so a coordinator writing at the run id
    and a child writing at `{run_id}:agent:{name}` cannot collide — and the
    child's `_clear` cannot delete the coordinator's state."""
    from agentkit.adapters.store import InMemoryStore
    from agentkit.kernel.types import Scope
    from agentkit.runtime import Invoker, RunContext, Services

    @tool(side_effecting=True)
    def risky(q: str) -> str:
        """Side-effecting, so the child gates and leaves a suspended slot."""
        return "ok"

    class _ToolThenGate:
        async def stream(self, *, messages, tools=None, **_kw):
            offered = {t.name for t in (tools or ())}
            ran = any(getattr(m, "role", "") == "tool" for m in messages)
            if "risky" in offered and not ran:
                yield Delta(text="tooling", model="m", provider="f")
                yield Delta(
                    tool_calls=(ToolCall("t1", "risky", {"q": "x"}),),
                    usage=Usage(1, 1, 0.0),
                    finish_reason="tool_calls",
                    model="m",
                    provider="f",
                )
            else:
                yield Delta(text="answer", model="m", provider="f")
                yield Delta(
                    usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f"
                )

    store = InMemoryStore()
    child = Agent("worker", "m", cognition=ReActCognition(tools=[risky]))
    coord = _coordinator_over(child, turns=2)
    ctx = RunContext(
        "run-x",
        Scope(),
        services=Services(invoker=Invoker(llm=_ToolThenGate()), store=store),
        autonomy="manual",  # gate the side-effecting tool so the child suspends
    )
    asyncio.run(coord.run("go", ctx))

    keys = sorted(k for k in store._kv if "run-x" in k)
    assert "checkpoint:run-x" in keys, "the coordinator's slot is missing"
    assert "checkpoint:run-x:agent:worker" in keys, "the child's slot is missing"


def test_every_producer_shares_one_resolution_order() -> None:
    """Three orders was how the asymmetry appeared in the first place. Asserted
    structurally so a fourth producer cannot quietly invent another."""
    from agentkit.agents.policies.roundrobin import _resolve_checkpointer
    from agentkit.capabilities.checkpointer import Checkpointer, resolve_checkpointer

    class _Ctx:
        checkpointer = None
        store = None

    explicit = Checkpointer(port=InMemoryCheckpointStore())
    coordinator = Agent("boss", "m", cognition=ReActCognition(tools=[], checkpointer=explicit))

    # An explicit per-cognition checkpointer wins, in both.
    assert _resolve_checkpointer(coordinator, _Ctx()) is explicit
    assert resolve_checkpointer(_Ctx(), explicit) is explicit

    # And with nothing wired, both agree there is no durable seam.
    bare = Agent("boss", "m")
    assert _resolve_checkpointer(bare, _Ctx()) is None
    assert resolve_checkpointer(_Ctx()) is None
