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


# ---- hashability: frozen in name only -------------------------------------------------------
#
# A Skill is documented as an immutable value object "you compose at wire-time,
# register once, reuse across runs" — and it could not go in a set or be a dict
# key. What made it unhashable is invisible in the annotations, which is why it
# needed diagnosing rather than reading. Measured before the fix::
#
#     hash(Skill("researcher", "digs"))
#     TypeError: unhashable type: 'SingleCallCognition'
#
# Not ``'dict'``, and not from a payload the caller passed: EVERY cognition the
# framework ships is a mutable ``@dataclass(slots=True)``, and a mutable
# dataclass with the default ``eq=True`` gets ``__hash__ = None`` from
# ``@dataclass`` itself. They are mutable on purpose — ``ReActCognition`` holds
# live termination state that ``as_agent`` deep-copies per run — so the defect
# was in Skill's hash, not in the cognitions. And because ``cognition``
# defaults to ``field(default_factory=SingleCallCognition)``, the ergonomic
# ``Skill("x", "y")`` form the docstring recommends produced an unhashable
# Skill: every Skill, not some.
#
# The fix hashes ``(name, description, prompt, model)`` — all hashable by
# construction, so the hash is total — and leaves ``__eq__`` alone.


def test_skill_is_hashable_in_its_ergonomic_two_argument_form():
    """The exact form the class docstring recommends for tests and small
    examples. It carries the default ``SingleCallCognition``, which is what
    used to make it unhashable."""
    assert isinstance(hash(Skill("researcher", "digs")), int)


def test_skill_is_hashable_with_every_cognition_the_framework_ships():
    """Not one unhashable cognition but the whole family — each is a mutable
    dataclass, so each broke Skill the same way."""
    from agentkit.agents.cognition import CoordinatorCognition
    from agentkit.agents.policies.roundrobin import RoundRobinPolicy

    for cognition in (
        SingleCallCognition(),
        ReActCognition(tools=ToolRegistry()),
        CoordinatorCognition(children={"a": Agent("a", "m")}, policy=RoundRobinPolicy()),
    ):
        assert isinstance(hash(Skill("s", "d", cognition=cognition)), int)


def test_skill_is_hashable_with_an_arbitrary_memory_implementation():
    """``MemorySource`` is a Protocol, so ``memory`` is whatever the
    application wired in — very often a mutable dataclass or a live client.
    The framework can promise nothing about its hash, so it stays out."""
    from dataclasses import dataclass

    @dataclass  # mutable + eq=True → __hash__ is None, exactly like a cognition
    class _MutableMemory:
        name: str = "stub"

        async def query(self, query, *, k, ctx, where=None):
            return []

        async def write(self, items, *, ctx):
            return None

    memory = _MutableMemory()
    with pytest.raises(TypeError):
        hash(memory)  # the field's own type is unhashable...
    assert isinstance(hash(Skill("s", "d", memory=memory)), int)  # ...the Skill is not


def test_skill_hash_ignores_cognition_and_memory_while_eq_does_not():
    """The soundness argument, exercised. Two Skills that differ only in
    cognition collide into one bucket, stay UNEQUAL, and both survive in a
    ``set`` — the hash invariant only requires EQUAL objects to hash equally."""
    a = Skill("researcher", "digs", cognition=SingleCallCognition())
    b = Skill("researcher", "digs", cognition=ReActCognition(tools=ToolRegistry()))
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_skill_hash_is_o1_in_the_cognition_it_holds():
    """Proven STRUCTURALLY rather than by timing, so it cannot go flaky: a
    Skill wired to a ReAct cognition holding 1000 tools hashes to the same
    number as the bare one. Only possible if the cognition is never read."""
    tools = ToolRegistry()
    for i in range(1000):
        tools.register(
            FunctionTool(name=f"t{i}", description="d", fn=lambda **kw: None, side_effecting=False)
        )
    assert hash(Skill("s", "d")) == hash(Skill("s", "d", cognition=ReActCognition(tools=tools)))


def test_skill_hash_separates_the_parts_it_keeps():
    """The hashed subset earns its place — name, description, prompt and model
    are the recipe's identity, and a registry keying on it needs them apart."""
    base = Skill("researcher", "digs", prompt="be diligent", model="m")
    assert hash(base) != hash(Skill("summariser", "digs", prompt="be diligent", model="m"))
    assert hash(base) != hash(Skill("researcher", "other", prompt="be diligent", model="m"))
    assert hash(base) != hash(Skill("researcher", "digs", prompt="be terse", model="m"))
    assert hash(base) != hash(Skill("researcher", "digs", prompt="be diligent", model="m2"))


def test_skill_hash_accepts_a_versioned_prompt_as_well_as_a_string():
    """``prompt`` is ``Prompt | str`` and both halves are hashable by
    construction — ``Prompt`` has its own identity hash over
    ``(id, version, template, inputs)`` — so including it cannot reintroduce
    the bug. A bound prompt hashes too."""
    p = Prompt(id="researcher", version="v1", template="be {how}", inputs=("how",))
    assert isinstance(hash(Skill("s", "d", prompt=p)), int)
    assert isinstance(hash(Skill("s", "d", prompt=p.bind(how="diligent"))), int)
    assert hash(Skill("s", "d", prompt=p)) != hash(Skill("s", "d", prompt="be diligent"))


def test_skills_can_be_registered_in_a_set_and_keyed_in_a_dict():
    """The caller this unlocks, in the words of the class docstring: compose at
    wire-time, register once, reuse. Equal recipes collapse; lookup by an
    EQUAL-but-distinct Skill hits, which an identity hash would not do."""
    registry = {Skill("researcher", "digs"): "wired"}
    assert registry[Skill("researcher", "digs")] == "wired"
    assert Skill("summariser", "condenses") not in registry
    assert len({Skill("a", "x"), Skill("a", "x"), Skill("b", "x")}) == 2


def test_skill_construction_and_materialisation_are_unchanged():
    """POSITIVE CONTROL: adding ``__hash__`` touches nothing else — fields,
    equality (which still compares the cognition), and ``as_agent`` all behave
    as before. Passes before and after the fix."""
    cognition = SingleCallCognition()
    skill = Skill("researcher", "digs", prompt="be diligent", cognition=cognition, model="m")

    assert (skill.name, skill.description, skill.model) == ("researcher", "digs", "m")
    assert skill.cognition is cognition
    assert skill == Skill("researcher", "digs", prompt="be diligent", cognition=cognition, model="m")
    # ``__eq__`` still compares the cognition — a different one is a different
    # recipe, even though the two now share a hash bucket.
    assert skill != Skill(
        "researcher", "digs", prompt="be diligent", cognition=ReActCognition(tools=ToolRegistry())
    )
    agent = skill.as_agent()
    assert agent.name == "researcher" and agent.model == "m"
    assert agent.cognition is not cognition  # still deep-copied per materialisation


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
