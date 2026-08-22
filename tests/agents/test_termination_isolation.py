"""A termination condition is run-local state, and a judge's answer is read literally.

Three defects, all in the "smart stop" layer:

1. **The coordinator shared one condition across concurrent runs.** A
   ``TerminationCondition`` is stateful (``MaxTurns.turn``,
   ``MaxMessages.count``, ``Timeout._start``, the latched ``_stop``), it lives
   on the cognition, and the cognition lives on a long-lived ``Agent`` object a
   server reuses. ``ReActCognition`` deep-copies per drive for exactly this
   reason; the coordinator policies did not. Two concurrent runs with
   ``MaxTurns(4)`` got 3 turns and 2 turns.

2. **Cloning broke ``ExternalTermination``.** The caller holds a handle, the run
   holds a copy, so ``set()`` never reached a running loop — an "externally
   triggered stop" that only worked if triggered before the run started.

3. **``judge_termination`` matched ``YES`` as a substring.** "Not yet —
   yesterday's draft is still open." stopped the run. So did "There is no
   simple yes/no answer here." Negation and hedging are exactly what a judge
   produces, and the docstring promised the opposite: "stops only on an
   explicit affirmative".
"""

from __future__ import annotations

import asyncio
import copy

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.control.termination import (
    ExternalTermination,
    MaxMessages,
    MaxTurns,
    Stop,
    Timeout,
    judge_termination,
)
from agentkit.agents.policies.roundrobin import RoundRobinPolicy
from agentkit.agents.policies.selector_policy import SelectorPolicy
from agentkit.agents.result import AgentResult
from agentkit.kernel.types import Message, Usage
from agentkit.testing import FakeLLM, make_test_ctx


class _Slow:
    """A child that yields to the event loop, so two concurrent coordinator runs
    genuinely interleave rather than running to completion one at a time."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def run(self, task, ctx, *, context=None):  # noqa: ANN001, ANN202
        await asyncio.sleep(0.001)
        return AgentResult(output=f"{self.name} spoke", usage=Usage())


def _turns(res: AgentResult) -> int:
    return len([m for m in res.evals["messages"] if m.role == "assistant"])


# ── 1. concurrent runs of one coordinator ───────────────────────────────────


@pytest.mark.parametrize("policy_name", ["roundrobin", "selector"])
def test_concurrent_runs_do_not_share_a_turn_counter(policy_name: str) -> None:
    """THE regression, for both coordinator policies. Each run must see its own
    ``MaxTurns(4)``; before the fix they counted into one counter and reset each
    other mid-flight."""
    children = {"a": _Slow("a"), "b": _Slow("b")}
    policy = (
        RoundRobinPolicy(max_turns=20)
        if policy_name == "roundrobin"
        else SelectorPolicy(selector=lambda transcript, agents: None, max_turns=20)
    )
    coord = Agent(
        name="c",
        cognition=CoordinatorCognition(
            children=children, policy=policy, termination=MaxTurns(4)
        ),
    )

    async def both():
        return await asyncio.gather(
            coord.run("A", make_test_ctx(llm=FakeLLM("x"), correlation_id="cA")),
            coord.run("B", make_test_ctx(llm=FakeLLM("x"), correlation_id="cB")),
        )

    ra, rb = asyncio.run(both())
    assert (_turns(ra), _turns(rb)) == (4, 4)


def test_the_shared_condition_is_left_untouched() -> None:
    """The clone is what the run advances. The instance the CALLER holds must
    still read zero afterwards — otherwise "reuse this coordinator" silently
    means "each run gets fewer turns than the last"."""
    shared = MaxTurns(3)
    coord = Agent(
        name="c",
        cognition=CoordinatorCognition(
            children={"a": _Slow("a")}, policy=RoundRobinPolicy(max_turns=20), termination=shared
        ),
    )
    asyncio.run(coord.run("go", make_test_ctx(llm=FakeLLM("x"), correlation_id="u1")))
    assert shared.turn == 0 and not shared.terminated


def test_sequential_runs_each_get_the_full_allowance() -> None:
    """The positive control the old code already passed (via ``reset()``), kept
    so the clone cannot regress it."""
    coord = Agent(
        name="c",
        cognition=CoordinatorCognition(
            children={"a": _Slow("a")}, policy=RoundRobinPolicy(max_turns=20),
            termination=MaxTurns(3),
        ),
    )
    for i in range(3):
        res = asyncio.run(
            coord.run(f"s{i}", make_test_ctx(llm=FakeLLM("x"), correlation_id=f"s{i}"))
        )
        assert _turns(res) == 3


# ── 2. the external switch survives cloning ─────────────────────────────────


def test_an_external_stop_reaches_a_cloned_condition() -> None:
    """``__deepcopy__`` returns ``self``: an external stop switch is not per-run
    state. Copying it is what made ``set()`` unable to stop a running loop."""
    ext = ExternalTermination()
    clone = copy.deepcopy(ext)
    assert clone is ext  # the identity IS the contract here

    assert asyncio.run(clone([])) is None
    ext.set()
    stop = asyncio.run(clone([]))
    assert stop is not None and stop.reason == "external"


def test_an_external_stop_inside_a_composite_still_shares() -> None:
    """A composite clones normally; only the external leaf opts out, so
    ``MaxTurns | ExternalTermination`` keeps a run-local counter AND a live
    switch."""
    ext = ExternalTermination()
    combo = MaxTurns(99) | ext
    clone = copy.deepcopy(combo)

    assert clone is not combo
    assert clone.conditions[0] is not combo.conditions[0]  # the counter is copied
    assert clone.conditions[1] is ext  # the switch is not

    ext.set()
    assert asyncio.run(clone([])) is not None


def test_other_conditions_are_genuinely_copied() -> None:
    """The negative control for the ``__deepcopy__`` carve-out: it must apply to
    the external switch ONLY, or the concurrency fix above is undone."""
    for cond in (MaxTurns(2), MaxMessages(2), Timeout(1.0)):
        assert copy.deepcopy(cond) is not cond


# ── 3. the judge answers, and we read the answer ────────────────────────────


class _Judge:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def run(self, prompt, ctx):  # noqa: ANN001, ANN202
        return AgentResult(output=self.reply, usage=Usage())


def _judged(reply: str) -> bool:
    cond = judge_termination(_Judge(reply))
    got = asyncio.run(cond([Message("assistant", "…")], make_test_ctx(llm=FakeLLM("x"))))
    return got is not None


@pytest.mark.parametrize(
    ("reply", "stops"),
    [
        ("YES", True),
        ("yes", True),  # case-insensitive
        ("**YES**", True),  # a model that formats
        ("  YES ", True),
        ("YES, the brief is complete", True),
        ("NO", False),
        ("", False),
        # THE regressions — each of these used to stop the run:
        ("Not yet — yesterday's draft is still open.", False),
        ("There is no simple yes/no answer here.", False),
        ("NO — yes it needs another pass", False),
        # Deliberately conservative: the affirmative must LEAD. One extra turn
        # is bounded by the hard ceiling; a false stop truncates the work and
        # reports it complete, and nothing catches that.
        ("Answer: YES", False),
    ],
)
def test_only_a_leading_affirmative_stops(reply: str, stops: bool) -> None:
    assert _judged(reply) is stops


def test_a_judge_that_raises_never_forces_a_stop() -> None:
    """Pre-existing contract, pinned here because the matching rewrite sits
    right beside it: an infrastructure failure must not look like completion."""

    class _Broken:
        async def run(self, prompt, ctx):  # noqa: ANN001, ANN202
            raise RuntimeError("judge is down")

    cond = judge_termination(_Broken())
    assert asyncio.run(cond([Message("assistant", "…")], make_test_ctx(llm=FakeLLM("x")))) is None


def test_a_custom_affirmative_is_matched_literally() -> None:
    """``yes=`` is caller-supplied, so it is regex-escaped and may be a phrase."""
    cond = judge_termination(_Judge("task complete — ship it"), yes="task complete")
    assert asyncio.run(cond([Message("assistant", "…")], make_test_ctx(llm=FakeLLM("x")))) is not None
    cond2 = judge_termination(_Judge("task completeness is unclear"), yes="task complete")
    assert asyncio.run(cond2([Message("assistant", "…")], make_test_ctx(llm=FakeLLM("x")))) is None


# ── 4. a latched Stop cannot be rewritten by a consumer ─────────────────────


def test_a_latched_stop_cannot_be_rewritten() -> None:
    """A condition hands the SAME ``Stop`` to every caller on every later turn.
    A consumer assigning to ``.reason`` was rewriting the condition's own record
    — and the policy, the trace and ``evals`` all read the rewrite."""
    cond = MaxMessages(1)
    first = asyncio.run(cond([Message("assistant", "x")]))
    assert first is not None

    with pytest.raises(Exception):  # FrozenInstanceError (a dataclasses subclass of AttributeError)
        first.reason = "hijacked"  # type: ignore[misc]

    again = asyncio.run(cond([Message("assistant", "y")]))
    assert again is not None and again.reason == "max_messages"


def test_stop_is_still_constructible_with_detail() -> None:
    """Frozen shell, ordinary construction — ``detail`` stays a plain dict
    because a ``MappingProxyType`` cannot be deep-copied, and conditions ARE
    deep-copied per run."""
    s = Stop("custom", {"k": 1})
    assert s.reason == "custom" and s.detail["k"] == 1
    assert copy.deepcopy(s) == s
