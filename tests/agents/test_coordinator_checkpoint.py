"""Coordinator `Agent` durability via `Checkpointer` (Wave 4c).

Mirrors leaf `Agent`'s Wave-4b durability: a coordinator Agent with `RoundRobinPolicy` or
`SelectorPolicy` accepts a `Checkpointer` and snapshots per turn; a restart with the same
`correlation_id` rehydrates and resumes from where the earlier run left off.

Coverage:
1. Per-turn snapshots — `list_versions(run_id)` returns N versions for N completed turns.
2. Cancellation mid-run — the latest snapshot stays at status="running" so resume works.
3. Resume picks up at the right turn — transcript + results are preserved across processes.
4. Blackboard mode — caller's `WorkingContext.messages` repopulated on resume.
5. Terminal `status="done"` snapshot when a coordinator finishes normally.
6. `SelectorPolicy` parity — same resume semantics as `RoundRobinPolicy`.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit import Agent, AgentResult
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.control.termination import MaxMessages
from agentkit.agents.policies.roundrobin import RoundRobinPolicy
from agentkit.agents.policies.selector_policy import SelectorPolicy
from agentkit.capabilities import Checkpointer
from agentkit.context import WorkingContext
from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.types import Scope, Usage
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


# Standard kwargs every call site passes — bundled here so we don't repeat
# `correlation_id="team-run-1", scope=Scope(1, 2)` on every invocation.
_RUN_ID = "team-run-1"
_DEFAULTS = {"correlation_id": _RUN_ID, "scope": Scope(1, 2)}


def _children() -> dict[str, Agent]:
    return {"alice": Agent("alice", "m"), "bob": Agent("bob", "m")}


def _coord(
    *,
    termination,
    checkpointer=None,
    context_arg=None,
    children=None,
    policy_kwargs: dict | None = None,
) -> Agent:
    """Build a coordinator Agent with RoundRobinPolicy + a checkpointer."""
    return Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=children if children is not None else _children(),
            policy=RoundRobinPolicy(**(policy_kwargs or {})),
            termination=termination,
            checkpointer=checkpointer,
        ),
    )


# ---- 1. per-turn snapshots ---------------------------------------------------------------------


def test_coordinator_snapshots_after_each_turn():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    coord = _coord(termination=MaxMessages(3), checkpointer=cpt)
    res = _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), **_DEFAULTS)))
    assert res.evals["stop_reason"] == "max_messages" and len(res.evals["results"]) == 3
    # 2 running snapshots (after turns 0, 1) + 1 done snapshot (after turn 2 → termination)
    versions = _run(cpt.list_versions("team-run-1"))
    assert versions == [1, 2, 3]
    # Inspecting the terminal DONE snapshot — opt into ``include_terminal``;
    # the default filter would hide it (a resume-if-any-checkpoint wiring
    # must not silently re-run a finished job).
    final = _run(cpt.resume("team-run-1", include_terminal=True))
    assert final is not None and final.status == "done"
    assert final.state["turn"] == 3  # next turn would be 3
    assert final.state["stop_reason"] == "max_messages"


# ---- 2 + 3. cancellation mid-run, then resume from same run_id ---------------------------------


class _CountingAgent:
    """An Agent-like that records how many times its `run` was called. Useful to verify
    resume doesn't re-execute already-completed turns."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def run(self, task, ctx):
        self.calls += 1
        return AgentResult(output=f"{self.name}-{self.calls}", usage=Usage(1, 1, 0.0))


class _CancelOnSecondReply(_CountingAgent):
    def __init__(self, name: str, token: CancellationToken) -> None:
        super().__init__(name)
        self._token = token

    async def run(self, task, ctx):
        res = await super().run(task, ctx)
        if self.calls == 1:  # bob's first turn (= turn 1 overall) → cancel afterwards
            self._token.cancel()
        return res


def test_cancellation_after_two_turns_leaves_a_running_snapshot_at_turn_two():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    token = CancellationToken()

    # Cancel after the second turn completes — the next turn's check_cancelled() raises.
    alice = _CountingAgent("alice")
    bob_cancel = _CancelOnSecondReply("bob", token)
    coord = _coord(
        termination=MaxMessages(10),
        checkpointer=cpt,
        children={"alice": alice, "bob": bob_cancel},
        policy_kwargs={"max_turns": 10},
    )
    with pytest.raises(Cancelled):
        _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), cancel=token, **_DEFAULTS)))

    # Snapshots: one per completed turn — turn 0 (alice), turn 1 (bob). Turn 2 cancelled
    # before the child ran. Latest snapshot status="running", turn=2 (next index to run).
    versions = _run(cpt.list_versions("team-run-1"))
    assert versions == [1, 2]
    latest = _run(cpt.resume("team-run-1"))
    assert latest is not None
    assert latest.status == "running"
    assert latest.state["turn"] == 2
    assert len(latest.state["results"]) == 2  # only the two completed turns


def test_resume_picks_up_after_the_last_completed_turn():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    token = CancellationToken()
    alice = _CountingAgent("alice")
    bob = _CancelOnSecondReply("bob", token)
    coord = _coord(
        termination=MaxMessages(10),
        checkpointer=cpt,
        children={"alice": alice, "bob": bob},
        policy_kwargs={"max_turns": 10},
    )
    with pytest.raises(Cancelled):
        _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), cancel=token, **_DEFAULTS)))
    assert alice.calls == 1 and bob.calls == 1

    # Rebuild the coordinator — same checkpointer, same correlation_id — and resume. The
    # children are fresh (their call counts start at zero) so any re-execution of turn 0 /
    # turn 1 would surface as alice.calls or bob.calls > 1.
    alice2, bob2 = _CountingAgent("alice"), _CountingAgent("bob")
    resumed = _coord(
        termination=MaxMessages(4),
        checkpointer=cpt,
        children={"alice": alice2, "bob": bob2},
        policy_kwargs={"max_turns": 10},
    )
    res = _run(resumed.run("start", make_test_ctx(llm=FakeLLM("ok"), **_DEFAULTS)))

    # Original turns NOT re-run — alice2 picks up at turn 2, bob2 at turn 3.
    assert alice2.calls == 1 and bob2.calls == 1
    # The result aggregates the original two turns + the two new turns.
    assert len(res.evals["results"]) == 4
    outputs = [r.output for r in res.evals["results"]]
    assert outputs == ["alice-1", "bob-1", "alice-1", "bob-1"]
    # Transcript still opens with the original task and contains 1 + 4 = 5 messages.
    transcript = res.evals["messages"]
    assert transcript[0].content == "start"
    assert len(transcript) == 5
    # Terminal snapshot was written → latest is now done. Inspecting it
    # requires ``include_terminal=True`` (the default filter treats
    # DONE/FAILED as "no resumable state").
    latest = _run(cpt.resume("team-run-1", include_terminal=True))
    assert latest is not None and latest.status == "done"


# ---- 4. blackboard mode -----------------------------------------------------------------------


def test_blackboard_messages_repopulated_on_resume():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    token = CancellationToken()
    alice = _CountingAgent("alice")
    bob = _CancelOnSecondReply("bob", token)

    wc1 = WorkingContext().note("topic", "rehydrate")
    coord = _coord(
        termination=MaxMessages(10),
        checkpointer=cpt,
        children={"alice": alice, "bob": bob},
        policy_kwargs={"max_turns": 10},
    )
    with pytest.raises(Cancelled):
        _run(
            coord.run(
                "start",
                make_test_ctx(llm=FakeLLM("ok"), cancel=token, **_DEFAULTS),
                context=wc1,
            )
        )
    # The original blackboard has the two completed turns + the task.
    assert [m.content for m in wc1.messages] == ["start", "alice-1", "bob-1"]
    assert wc1.get("topic") == "rehydrate"

    # The caller passes a FRESH WorkingContext on resume. Its `.messages` is the source of
    # truth, so on resume the coordinator repopulates it from the checkpoint (and likewise
    # restores the scratchpad).
    wc2 = WorkingContext()
    resumed = _coord(
        termination=MaxMessages(3),  # stop after one more turn so the test is tight
        checkpointer=cpt,
        children={"alice": _CountingAgent("alice"), "bob": _CountingAgent("bob")},
        policy_kwargs={"max_turns": 10},
    )
    res = _run(resumed.run("start", make_test_ctx(llm=FakeLLM("ok"), **_DEFAULTS), context=wc2))
    # Original turns survived; one new turn appended.
    assert [m.content for m in wc2.messages] == ["start", "alice-1", "bob-1", "alice-1"]
    assert wc2.get("topic") == "rehydrate"  # scratchpad rehydrated too
    # the returned transcript mirrors the live blackboard's contents
    assert [m.content for m in res.evals["messages"]] == [m.content for m in wc2.messages]


# ---- 5. terminal snapshot -----------------------------------------------------------------------


def test_normal_completion_writes_a_done_snapshot():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    coord = _coord(termination=MaxMessages(2), checkpointer=cpt)
    res = _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), **_DEFAULTS)))
    assert res.evals["stop_reason"] == "max_messages"
    # A terminal DONE snapshot IS on the port — the default filter hides
    # it from ``resume()`` (that's the whole point) but the audit-path
    # opt-in returns it verbatim.
    assert _run(cpt.resume("team-run-1")) is None
    latest = _run(cpt.resume("team-run-1", include_terminal=True))
    assert latest is not None
    assert latest.status == "done"
    assert latest.state["stop_reason"] == "max_messages"
    # And a follow-up `run()` with the same run_id sees `status="done"` and starts fresh
    # rather than rehydrating a finished run.
    res2 = _run(coord.run("again", make_test_ctx(llm=FakeLLM("ok"), **_DEFAULTS)))
    assert res2.evals["messages"][0].content == "again"  # not rehydrated from prior transcript


# ---- 6. SelectorPolicy parity -------------------------------------------------------------------


def test_selector_policy_resumes_with_same_semantics():
    cpt = Checkpointer(port=InMemoryCheckpointStore())
    token = CancellationToken()
    alice = _CountingAgent("alice")
    bob = _CancelOnSecondReply("bob", token)

    # A simple round-robin-ish selector — stateless, derives the next speaker from the
    # transcript length. This is safe across resume because the transcript is rehydrated.
    def select(transcript, agents):
        assistant_count = sum(1 for m in transcript if m.role == "assistant")
        return agents[assistant_count % len(agents)].name

    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children={"alice": alice, "bob": bob},
            policy=SelectorPolicy(selector=select, max_turns=10),
            termination=MaxMessages(10),
            checkpointer=cpt,
        ),
    )
    with pytest.raises(Cancelled):
        _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), cancel=token, **_DEFAULTS)))
    versions = _run(cpt.list_versions("team-run-1"))
    assert versions == [1, 2]

    # Fresh children + same selector + same checkpointer → resume.
    alice2, bob2 = _CountingAgent("alice"), _CountingAgent("bob")
    resumed = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children={"alice": alice2, "bob": bob2},
            policy=SelectorPolicy(selector=select, max_turns=10),
            termination=MaxMessages(4),
            checkpointer=cpt,
        ),
    )
    res = _run(resumed.run("start", make_test_ctx(llm=FakeLLM("ok"), **_DEFAULTS)))
    assert alice2.calls == 1 and bob2.calls == 1  # no re-execution of completed turns
    assert [m.name for m in res.evals["messages"][1:]] == ["alice", "bob", "alice", "bob"]


# ---- field wins over ctx --------------------------------------------------------------------


def test_coordinator_checkpointer_field_overrides_ctx_checkpointer():
    field_cpt = Checkpointer(port=InMemoryCheckpointStore())
    ctx_cpt = Checkpointer(port=InMemoryCheckpointStore())
    coord = _coord(termination=MaxMessages(2), checkpointer=field_cpt)
    _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), checkpointer=ctx_cpt, **_DEFAULTS)))
    # Only the field-bound checkpointer recorded the run. The final
    # snapshot is DONE (terminal), so opting in with ``include_terminal``
    # is what surfaces it — the default filter hides terminal snapshots.
    assert _run(field_cpt.resume("team-run-1", include_terminal=True)) is not None
    assert _run(ctx_cpt.resume("team-run-1", include_terminal=True)) is None


def test_coordinator_falls_back_to_ctx_checkpointer():
    ctx_cpt = Checkpointer(port=InMemoryCheckpointStore())
    coord = _coord(termination=MaxMessages(2))  # no field
    _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), checkpointer=ctx_cpt, **_DEFAULTS)))
    # DONE terminal snapshot — needs ``include_terminal`` to inspect.
    assert _run(ctx_cpt.resume("team-run-1", include_terminal=True)) is not None


def test_no_checkpointer_means_no_snapshots_and_no_behaviour_change():
    # Sanity: an unconfigured coordinator behaves exactly as it did pre-Wave-4c.
    coord = _coord(termination=MaxMessages(2))
    res = _run(coord.run("start", make_test_ctx(llm=FakeLLM("ok"), **_DEFAULTS)))
    assert res.evals["stop_reason"] == "max_messages" and len(res.evals["results"]) == 2
