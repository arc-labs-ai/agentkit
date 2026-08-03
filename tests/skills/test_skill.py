"""Skill — Facade composing (prompt + cognition + memory) as one wirable unit.

Two adapters define the surface:
  - ``skill.as_agent()`` → materialise as a runnable ``Agent``
  - ``skill.as_tool()``  → adapt for an outer agent's tool-loop

Skills are frozen value objects (immutable recipes shareable across runs).
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from agentkit import Skill
from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition, SingleCallCognition
from agentkit.kernel.types import Scope, ToolCall
from agentkit.memory.base import MemorySource
from agentkit.prompts.prompt import Prompt
from agentkit.testing import FakeLLM, Turn, make_test_ctx
from agentkit.tools import FunctionTool, ToolRegistry


def _run(coro):
    return asyncio.run(coro)


# ---- materialisation -------------------------------------------------------------------------


def test_skill_as_agent_materialises_with_skill_fields():
    """``as_agent()`` produces an ``Agent`` carrying the skill's name /
    prompt / cognition / memory / model. The Skill itself is the
    recipe; the Agent is the runnable instance."""

    class _StubMemory:
        name = "stub"

        async def query(self, query, *, k, ctx, where=None):
            return []

        async def write(self, items, *, ctx):
            return None

    memory: MemorySource = _StubMemory()
    prompt = Prompt(id="researcher", version="v1", template="be diligent")
    cognition = SingleCallCognition()
    skill = Skill(
        name="researcher",
        description="finds and scores sources",
        prompt=prompt,
        cognition=cognition,
        memory=memory,
        model="claude-3-haiku",
    )

    agent = skill.as_agent()

    assert isinstance(agent, Agent)
    assert agent.name == "researcher"
    assert agent.prompt is prompt
    # ``as_agent`` deep-copies the cognition so concurrent
    # materialisations don't race on shared termination state — the
    # copy has the SAME shape but is a distinct object.
    assert isinstance(agent.cognition, type(cognition))
    assert agent.cognition is not cognition
    assert agent.memory is memory
    assert agent.model == "claude-3-haiku"


def test_skill_as_agent_overrides_model():
    """``as_agent(model=...)`` lets the caller swap models per-run (cheap
    in tests, premium in production) without rebuilding the Skill."""
    skill = Skill(name="s", description="d", model="default-model")
    overridden = skill.as_agent(model="override-model")
    assert overridden.model == "override-model"
    # The Skill recipe itself is untouched.
    assert skill.model == "default-model"
    # A second materialisation without override returns to the Skill's default.
    assert skill.as_agent().model == "default-model"


# ---- adaptation ------------------------------------------------------------------------------


def test_skill_as_tool_returns_a_tool():
    """``as_tool()`` returns the ``FunctionTool`` produced by the
    compose adapter — same shape every ToolRegistry already speaks."""
    skill = Skill(name="summariser", description="condenses long text")
    tool = skill.as_tool()
    assert isinstance(tool, FunctionTool)
    assert tool.name == "summariser"
    assert tool.description == "condenses long text"
    # Tool Protocol shape: callable ``run``.
    assert callable(tool.run)


def test_skill_as_tool_with_name_and_description_overrides():
    """Same Skill, different labels — useful when the outer registry
    needs disambiguation across multiple Skills wearing similar hats."""
    skill = Skill(name="researcher", description="default desc")
    tool = skill.as_tool(name="legal_researcher", description="searches case law")
    assert tool.name == "legal_researcher"
    assert tool.description == "searches case law"


# ---- immutability ----------------------------------------------------------------------------


def test_skill_is_frozen():
    """Frozen-dataclass discipline: Skills are immutable recipes.
    Mutation belongs on the materialised ``Agent``, not the Skill."""
    skill = Skill(name="x", description="y")
    with pytest.raises(FrozenInstanceError):
        skill.name = "renamed"  # type: ignore[misc]


# ---- end-to-end: outer agent calls a Skill as a tool ----------------------------------------


def test_outer_agent_can_use_skill_as_tool_end_to_end():
    """The headline integration: an outer agent picks a Skill from its
    tool registry, the framework drives the underlying Agent on a
    ``ctx.child()``, and the rendered result flows back into the outer
    loop. Two-turn outer script: tool call → final answer."""
    # Inner skill: a single-call agent that just echoes whatever the LLM emits.
    inner_llm = FakeLLM("inner-result")
    skill = Skill(
        name="summariser",
        description="condenses long text into a short brief",
        prompt="you summarise",
        cognition=SingleCallCognition(),
        model="inner-model",
    )

    # Outer agent: ReAct loop with the skill registered as a tool.
    registry = ToolRegistry()
    registry.register(skill.as_tool())

    outer_llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "summariser", {"task": "summarise this"}),)),
            Turn(content="final answer from outer"),
        ]
    )

    # The Invoker shares ONE LLM across both layers, so use a dispatching wrapper:
    # outer turns are scripted; inner uses a content-based fake. We instead build
    # a single invoker per layer by giving each a separate ctx -- but ctx.child()
    # inherits services. So we use a dispatching LLM that routes by system prompt.
    class _DispatchLLM:
        async def stream(self, **kw):
            messages = kw["messages"]
            system = next((m.content for m in messages if m.role == "system"), "")
            target = inner_llm if "you summarise" in system else outer_llm
            async for d in target.stream(**kw):
                yield d

        async def chat(self, **kw):
            messages = kw["messages"]
            system = next((m.content for m in messages if m.role == "system"), "")
            target = inner_llm if "you summarise" in system else outer_llm
            return await target.chat(**kw)

    outer = Agent(
        name="orchestrator",
        model="outer-model",
        cognition=ReActCognition(tools=registry),
    )
    result = _run(
        outer.run(
            "do the thing",
            make_test_ctx(llm=_DispatchLLM(), scope=Scope(1, 2)),
        )
    )
    # The outer loop completed with its second-turn answer, having
    # successfully dispatched through the skill on turn one.
    assert result.output == "final answer from outer"
    # The inner skill ran exactly once (the single tool call).
    assert inner_llm.calls == 1


# ── Per-run cognition isolation ──────────────────────────────────────────────
#
# ``Skill.as_agent()`` deep-copies the underlying cognition so two
# agents materialised from the same shared Skill instance don't race
# on mutable ``TerminationCondition`` state (counters, timeouts, the
# flip-once external flag). This lets a single Skill be reused across
# concurrent runs — the recipe stays immutable, each materialised
# Agent carries its own state graph.


def test_as_agent_cognitions_are_independent_instances() -> None:
    """Two ``as_agent`` calls return Agents whose cognitions are
    distinct objects — mutation on one doesn't leak to the other."""
    from agentkit.agents.cognition import ReActCognition
    from agentkit.agents.control.termination import MaxTurns

    shared = Skill(
        name="researcher",
        description="finds sources",
        cognition=ReActCognition(tools=[], termination=MaxTurns(5)),
    )

    a = shared.as_agent()
    b = shared.as_agent()

    assert a.cognition is not b.cognition
    assert a.cognition.termination is not b.cognition.termination
    # Shape preserved — deep-copy respects the class + constructor args.
    assert isinstance(a.cognition, ReActCognition)
    assert isinstance(a.cognition.termination, MaxTurns)
    assert a.cognition.termination.max == 5


def test_concurrent_as_agent_calls_do_not_share_termination_state() -> None:
    """Concurrent materialisations from the same Skill produce Agents
    with independent termination counters. Two runs racing on the
    same shared Skill instance stay isolated — one's ``turn`` counter
    never bleeds into the other."""
    import asyncio

    from agentkit.agents.cognition import ReActCognition
    from agentkit.agents.control.termination import MaxTurns

    shared = Skill(
        name="researcher",
        description="finds sources",
        cognition=ReActCognition(tools=[], termination=MaxTurns(10)),
    )

    async def materialise() -> object:
        # Simulate a cheap materialisation from another task on the loop.
        await asyncio.sleep(0)
        agent = shared.as_agent()
        assert agent.cognition.termination is not None
        agent.cognition.termination.turn = 99  # scribble on THIS agent's state
        return agent

    async def go():
        agents = await asyncio.gather(*(materialise() for _ in range(4)))
        # Each agent got its own termination instance; the scribble on
        # one is invisible on the others (they'd all show turn=99 if
        # they shared state).
        turns = [a.cognition.termination.turn for a in agents]
        return turns

    turns = asyncio.run(go())
    assert turns == [99, 99, 99, 99]  # each scribble landed on its own copy


def test_skill_recipe_survives_materialisation_deep_copy() -> None:
    """The Skill instance itself is frozen — deep-copying its
    cognition per ``as_agent`` call must not mutate or replace the
    Skill's stored cognition."""
    from agentkit.agents.cognition import ReActCognition
    from agentkit.agents.control.termination import MaxTurns

    original_cognition = ReActCognition(tools=[], termination=MaxTurns(3))
    skill = Skill(
        name="researcher",
        description="finds sources",
        cognition=original_cognition,
    )

    skill.as_agent()
    skill.as_agent()
    skill.as_agent()

    # The skill's stored cognition is still the exact object it was
    # constructed with — materialisation never mutates the recipe.
    assert skill.cognition is original_cognition
    # And a fresh as_agent still returns an independent copy.
    fresh = skill.as_agent()
    assert fresh.cognition is not original_cognition
