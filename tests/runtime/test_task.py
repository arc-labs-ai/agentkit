"""Task management (ch21 / R15): the Task value, the heuristic progress assessor, and the
stall-aware coordinator `Agent` + `LedgerPolicy` (route on progress, re-plan on stall,
bounded by max_replans + max_rounds). Offline & deterministic — control flow driven by a
scripted assessor."""

import asyncio

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.ledger import (
    LedgerPolicy,
    ProgressLedger,
    Task,
    TaskLedger,
    heuristic_assessor,
)
from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.types import Message, Scope
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def _scripted(*snapshots):
    it = iter(snapshots)

    def assess(ledger, transcript, names):
        return next(it)

    return assess


def _coord(policy: LedgerPolicy) -> Agent:
    return Agent(
        name="ledger",
        cognition=CoordinatorCognition(children={"a": Agent("a", "m")}, policy=policy),
    )


# ---- Task value -------------------------------------------------------------------------------


def test_task_subtasks_status_and_done_count():
    t = Task("build")
    t.subtask("design").mark("done")
    t.subtask("ship")  # still pending
    assert t.done_count == 1
    assert not t.is_terminal
    t.mark("abandoned")
    assert t.is_terminal


# ---- heuristic_assessor -----------------------------------------------------------------------


def test_heuristic_assessor_round_robins_and_reads_the_plan():
    assess = heuristic_assessor()
    transcript = [Message("user", "g"), Message("assistant", "working", name="a")]
    pl = assess(TaskLedger("g", plan=["step1"]), transcript, ["a", "b"])
    assert (
        not pl.satisfied and pl.next_speaker == "b" and pl.instruction == "step1"
    )  # 1 reply → names[1]


def test_heuristic_assessor_detects_done_and_loops():
    assess = heuristic_assessor()
    done = [Message("assistant", "all DONE", name="a")]
    assert assess(TaskLedger("g"), done, ["a"]).satisfied
    loop = [Message("assistant", "x", name="a"), Message("assistant", "x", name="b")]
    assert assess(TaskLedger("g"), loop, ["a"]).in_a_loop


# ---- LedgerPolicy control flow ----------------------------------------------------------------


def test_routes_then_stops_when_satisfied():
    coord = _coord(
        LedgerPolicy(
            assessor=_scripted(ProgressLedger(next_speaker="a"), ProgressLedger(satisfied=True))
        )
    )
    res = _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))
    assert res.evals["stop_reason"] == "satisfied" and len(res.evals["results"]) == 1


def test_replans_on_stall_then_continues():
    coord = _coord(
        LedgerPolicy(
            assessor=_scripted(
                ProgressLedger(in_a_loop=True),  # stall → re-plan
                ProgressLedger(next_speaker="a"),  # then route
                ProgressLedger(satisfied=True),  # then done
            )
        )
    )
    res = _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))
    assert res.evals["stop_reason"] == "satisfied"
    assert res.evals["replans"] == 1 and len(res.evals["results"]) == 1
    assert "replan #1" in res.evals["ledger"].facts


def test_gives_up_after_max_replans():
    coord = _coord(
        LedgerPolicy(
            max_replans=2,
            assessor=lambda ld, t, n: ProgressLedger(in_a_loop=True),  # always stalled
        )
    )
    res = _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))
    assert res.evals["stop_reason"] == "stalled"
    assert res.evals["replans"] == 2 and len(res.evals["results"]) == 0


def test_max_rounds_is_the_hard_backstop():
    coord = _coord(
        LedgerPolicy(
            max_rounds=3,
            assessor=lambda ld, t, n: ProgressLedger(progress_being_made=True, next_speaker="a"),
        )
    )
    res = _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))
    assert res.evals["stop_reason"] == "max_rounds" and len(res.evals["results"]) == 3


def test_honors_cancellation():
    token = CancellationToken()
    token.cancel()
    coord = _coord(
        LedgerPolicy(assessor=lambda ld, t, n: ProgressLedger(next_speaker="a")),
    )
    with pytest.raises(Cancelled):
        _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2), cancel=token)))
