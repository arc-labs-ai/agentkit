"""LLM-driven control (ch15 / ch18): `llm_selector` (a model picks the next speaker) and
`judge_termination` (a model decides the task is done). Both route through an ordinary Agent on the
run's Invoker — so they're offline-testable with `FakeLLM` — and both fail safe. Deterministic.

Also covers ``RunPolicy`` regressions: the ``deny`` mode name (formerly drifted as
``"block"``) and ``Agent.stream`` invoking ``policy.check`` before the first cognition
drive so a deny-mode policy short-circuits the run before any LLM call."""

import asyncio

import pytest

from agentkit.agents import Agent, RoundRobinPolicy, SelectorPolicy
from agentkit.agents.cognition import CoordinatorCognition, ReActCognition
from agentkit.agents.control import MaxMessages
from agentkit.agents.control.safety import PolicyVerdict, RunPolicy
from agentkit.agents.control.termination import judge_termination
from agentkit.agents.policies.selector_policy import llm_selector
from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import FunctionTool


def _run(coro):
    return asyncio.run(coro)


# ---- RunPolicy (Rule of Two) regressions -------------------------------------------------------
#
# The mode field drifted historically ("block") vs. what the docs called out ("deny") — the
# guard silently no-op'd. These tests pin the mode name AND assert the wiring point in
# ``Agent.stream`` so ``policy.check`` runs BEFORE the first cognition drive.


def _tool(name, caps):
    """Test stub tool carrying a ``caps`` tuple for the RunPolicy trifecta check."""
    return FunctionTool(
        name,
        lambda a, c: None,
        description="test stub tool used for RunPolicy capability checks",
        side_effecting=False,
        caps=caps,
    )


def _trifecta_tools():
    return [
        _tool("read_email", ("private_data",)),
        _tool("read_web", ("untrusted_content",)),
        _tool("http_post", ("egress",)),
    ]


def test_runpolicy_deny_mode_raises_on_trifecta():
    """``mode="deny"`` + trifecta tools → ``PermissionError`` raised out of ``check``.

    Regression: the field previously accepted ``"block"`` as the raise-mode name; the
    docs called it ``"deny"``. The mismatched literal silently degraded ``deny`` to
    flag mode. Pin the name here so drift can't come back.
    """
    with pytest.raises(PermissionError, match="lethal trifecta"):
        RunPolicy(mode="deny").check(_trifecta_tools())


def test_runpolicy_flag_mode_returns_flagged_verdict():
    """``mode="flag"`` + trifecta tools → ``PolicyVerdict(allowed=False, ...)`` — the run
    isn't hard-stopped but a caller can gate/split on the verdict."""
    verdict = RunPolicy(mode="flag").check(_trifecta_tools())
    assert isinstance(verdict, PolicyVerdict)
    assert verdict.allowed is False
    assert set(verdict.capabilities) == {"private_data", "untrusted_content", "egress"}
    assert "lethal trifecta" in verdict.reason


def test_agent_run_invokes_policy_check_before_first_turn():
    """``Agent(policy=RunPolicy(mode="deny"), cognition=ReActCognition(tools=<trifecta>))``
    → ``agent.run(...)`` raises ``PermissionError`` BEFORE any LLM call.

    Wired by ``Agent.stream``: the policy check fires inside the ``invoke_agent`` span
    and BEFORE the cognition drive is iterated, so a deny verdict prevents the very
    first chat call. Proven by giving the run a ``FakeLLM`` and asserting its call
    counter stays at zero across the raise.
    """
    llm = FakeLLM("should never be called")
    ctx = make_test_ctx(llm=llm, scope=Scope(1, 2), correlation_id="r")
    agent = Agent(
        name="risky",
        model="m",
        cognition=ReActCognition(tools=_trifecta_tools()),
        policy=RunPolicy(mode="deny"),
    )
    with pytest.raises(PermissionError, match="lethal trifecta"):
        _run(agent.run("go", ctx))
    assert llm.calls == 0, "LLM was invoked before the policy gate fired"


def _children():
    return {"alice": Agent("alice", "m"), "bob": Agent("bob", "m")}


def _selector_coord(selector, termination=MaxMessages(2)):
    return Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=_children(),
            policy=SelectorPolicy(selector=selector),
            termination=termination,
        ),
    )


def _roundrobin_coord(termination, *, max_turns=50):
    return Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=_children(),
            policy=RoundRobinPolicy(max_turns=max_turns),
            termination=termination,
        ),
    )


# ---- llm_selector: a model chooses who speaks next ---------------------------------------------


def test_llm_selector_picks_the_agent_the_model_names():
    def fake(**kw):  # the selector prompt ends with "...speak next."
        return "bob" if "speak next" in kw["user"] else "working"

    coord = _selector_coord(llm_selector(Agent("router", "m")))
    res = _run(coord.run("start", make_test_ctx(llm=FakeLLM(fake), scope=Scope(1, 2))))
    assert [m.name for m in res.evals["messages"][1:]] == ["bob", "bob"]


def test_llm_selector_falls_back_to_round_robin_on_unparseable_reply():
    def fake(**kw):
        return "hmm not sure" if "speak next" in kw["user"] else "working"  # names no agent

    coord = _selector_coord(llm_selector(Agent("router", "m")))
    res = _run(coord.run("start", make_test_ctx(llm=FakeLLM(fake), scope=Scope(1, 2))))
    # never stalls — falls through to round-robin
    assert [m.name for m in res.evals["messages"][1:]] == ["alice", "bob"]


# ---- judge_termination: a model decides the task is done ---------------------------------------


def test_judge_termination_stops_on_yes():
    def fake(**kw):
        return "YES" if "YES or NO" in kw["user"] else "doing work"

    ctx = make_test_ctx(llm=FakeLLM(fake), scope=Scope(1, 2))
    coord = _roundrobin_coord(judge_termination(Agent("judge", "m"), ctx), max_turns=5)
    res = _run(coord.run("start", ctx))
    assert res.evals["stop_reason"] == "judged_complete"
    assert len(res.evals["results"]) == 1


def test_judge_termination_continues_until_ceiling_on_no():
    def fake(**kw):
        return "NO" if "YES or NO" in kw["user"] else "doing work"

    ctx = make_test_ctx(llm=FakeLLM(fake), scope=Scope(1, 2))
    coord = _roundrobin_coord(judge_termination(Agent("judge", "m"), ctx), max_turns=3)
    res = _run(coord.run("start", ctx))
    assert res.evals["stop_reason"] == "max_turns"
    assert len(res.evals["results"]) == 3


def test_judge_termination_uses_the_live_run_ctx_when_none_is_captured():
    """Regression: judge_termination need not capture a ctx at build time — the coordinator
    threads the LIVE run ctx into the condition each turn, so the judge runs on the actual
    run's budget/cancel/trace."""

    def fake(**kw):
        return "YES" if "YES or NO" in kw["user"] else "doing work"

    coord = _roundrobin_coord(judge_termination(Agent("judge", "m")), max_turns=5)  # no ctx
    res = _run(coord.run("start", make_test_ctx(llm=FakeLLM(fake), scope=Scope(1, 2))))
    assert res.evals["stop_reason"] == "judged_complete"
    assert len(res.evals["results"]) == 1


def test_judge_termination_is_fail_safe_on_error():
    class BoomJudge:  # a judge that errors must NOT force a stop
        async def run(self, prompt, ctx):
            raise RuntimeError("judge down")

    ctx = make_test_ctx(llm=FakeLLM("work"), scope=Scope(1, 2))
    coord = _roundrobin_coord(judge_termination(BoomJudge(), ctx), max_turns=2)
    res = _run(coord.run("start", ctx))
    # ceiling stays the backstop when the judge fails
    assert res.evals["stop_reason"] == "max_turns"
    assert len(res.evals["results"]) == 2
