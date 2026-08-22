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


def test_a_suspend_written_by_the_legacy_slot_still_resumes() -> None:
    """Switching slots must not orphan an IN-FLIGHT suspend across an upgrade —
    a suspended run is waiting on a person and cannot be reconstructed."""
    cp = _cp()
    llm = _ToolThenAnswer()
    agent = Agent("worker", "m", cognition=ReActCognition(tools=[probe], checkpointer=cp))

    # Produce a real suspend, then MOVE it to the legacy slot to stand in for a
    # checkpoint written by an older build.
    ctx = make_test_ctx(llm=llm, checkpointer=cp, correlation_id=RUN, autonomy="manual")
    assert asyncio.run(agent.run("go", ctx)).stop_reason == "suspended"
    slot = ReActCognition.checkpoint_slot(RUN, "worker")
    saved = asyncio.run(cp.resume(slot))
    assert saved is not None
    asyncio.run(cp.delete(slot))
    asyncio.run(cp.snapshot(RUN, saved.state, status=CheckpointStatus.SUSPENDED))

    resumed = asyncio.run(
        agent.resume(RUN, {"t1": "approve"},
                     make_test_ctx(llm=llm, checkpointer=cp, correlation_id=RUN))
    )
    assert resumed.stop_reason == "complete"


def test_the_legacy_read_refuses_another_producers_state() -> None:
    """The fallback reads the bare correlation_id — which is exactly the slot
    OTHER producers still use. Reading it unconditionally re-introduced the
    collision from the other direction: a child loop picked up its
    coordinator's `{turn, transcript, ...}` and died in `rehydrate` with
    `KeyError: 'messages'`. The shape check is load-bearing.
    """
    cp = _cp()
    # A coordinator-shaped payload sitting in the legacy slot.
    asyncio.run(
        cp.snapshot(RUN, {"turn": 3, "transcript": [], "results": {}},
                    status=CheckpointStatus.RUNNING)
    )

    agent = Agent("worker", "m", cognition=ReActCognition(tools=[probe], checkpointer=cp))
    ctx = make_test_ctx(llm=_ToolThenAnswer(), checkpointer=cp, correlation_id=RUN)
    # Must start fresh rather than trying to rehydrate someone else's state.
    result = asyncio.run(agent.run("go", ctx))
    assert result.stop_reason == "complete"
    # And the coordinator's payload is untouched.
    still = asyncio.run(cp.resume(RUN))
    assert still is not None and still.state["turn"] == 3
