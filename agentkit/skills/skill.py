"""Skill — a Facade composing (prompt + cognition + memory) as one
wirable unit.

The missing primitive between ``Tool`` and ``Agent``. A Skill is what
users mean when they say "give me a researcher" — a specialised agent
package that other agents invoke as a tool, or that a host runs as a
standalone agent.

Compositions:
  - ``skill.as_agent()`` materialises the Skill as a runnable
    ``Agent``. Hosts run skills directly when they're the top of the
    stack.
  - ``skill.as_tool()`` adapts the Skill into a ``Tool`` an outer
    agent's ``ReActCognition`` can dispatch to. The outer agent picks
    the Skill from its tool registry by name. The adapter is the
    existing ``agentkit.tools.from_agent.as_tool`` helper — Skills layer
    a Facade on top.

Skills are immutable value objects (frozen dataclasses) — you compose
them at wire-time, register them once, reuse them across runs.
Mutation belongs on the materialised ``Agent``, not on the Skill recipe.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from agentkit.agents.agent import Agent
from agentkit.agents.cognition import Cognition, SingleCallCognition
from agentkit.memory.base import MemorySource
from agentkit.prompts.prompt import Prompt


@dataclass(frozen=True, slots=True)
class Skill:
    """A specialised agent recipe — prompt + cognition (+ optional memory + model).

    Required:
        name: stable identifier. Becomes the ``Tool`` name when adapted
            via ``as_tool()`` and the ``Agent.name`` when materialised
            via ``as_agent()``.
        description: one-line description. Shown to the outer LLM when
            this Skill is adapted as a tool (so it matters for tool
            selection); also the default ``Tool.description`` unless
            overridden at adapter-time.

    Optional (sensible defaults so the ergonomic ``Skill("x", "y")``
    form works for tests and small examples):
        prompt: a ``Prompt`` (versioned) OR a plain ``str`` (wrapped
            into a one-off inline ``Prompt`` by the underlying Agent).
            Defaults to ``""`` — every real Skill wants a system prompt.
        cognition: the turn-taking strategy. Defaults to
            ``SingleCallCognition()`` — the only cognition the framework
            ships that has a no-arg constructor. Real Skills that need
            tools wire a ``ReActCognition(tools=...)`` explicitly;
            multi-agent skills wire ``CoordinatorCognition(children=...,
            policy=...)``.
        memory: a ``MemorySource`` the underlying agent's
            ``RequestBuilder`` auto-grounds against. ``None`` (default)
            disables the memory hook.
        model: default model for the skill. ``as_agent(model=...)`` /
            ``as_tool(model=...)`` may override per-run (e.g. a cheaper
            model in tests, a premium model in production).

    Design notes
    ------------
    Skill is NOT an Agent subclass — it is a Facade *recipe*. ``as_agent``
    constructs a fresh ``Agent`` each call; the Skill itself stays
    immutable. This makes Skills safely shareable across runs and
    threads: nothing per-run lives on the recipe.

    The framework ships ZERO concrete Skills (no Researcher, Synthesiser,
    Critic, …) — applications compose their own from their own prompts,
    tools, and memory, the same way the framework ships zero Tools.
    """

    name: str
    description: str
    prompt: Prompt | str = ""
    cognition: Cognition = field(default_factory=SingleCallCognition)
    memory: MemorySource | None = None
    model: str | None = None

    def as_agent(self, *, model: str | None = None) -> Agent:
        """Materialise the Skill as a runnable ``Agent``.

        The caller MAY override ``model`` per-run (e.g. cheaper model in
        tests, premium model in production). Other Skill fields are
        pinned at construction and shared by every materialisation.

        The underlying ``Cognition`` is deep-copied per call.
        ``ReActCognition`` holds mutable ``TerminationCondition`` state
        (``MaxTurns.turn`` / ``Timeout._start``) that is reset at the
        start of every run — two concurrent runs materialised from the
        same shared Skill instance would race on that counter, and one
        run's ``reset()`` would blow away the other mid-count. Copying
        the cognition ties each ``Agent`` to its own state graph while
        the immutable Skill recipe stays shared.
        """
        return Agent(
            name=self.name,
            model=model if model is not None else self.model,
            prompt=self.prompt,
            cognition=copy.deepcopy(self.cognition),
            memory=self.memory,
        )

    def as_tool(
        self,
        *,
        model: str | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Adapt the Skill into a ``Tool`` an outer agent can call via
        tool-use.

        ``name`` / ``description`` default from the Skill but can be
        overridden if the outer registry needs a different label (e.g.
        the same Skill exposed under multiple names in different
        registries). ``model`` is forwarded to ``as_agent`` so the
        underlying Agent runs on the override.

        Returns a ``FunctionTool`` produced by
        ``agentkit.tools.from_agent.as_tool``.
        """
        # Late import: ``agentkit.tools.from_agent`` pulls a chain of
        # capabilities + tool types we don't want eagerly imported on
        # ``from agentkit import Skill``.
        from agentkit.tools.from_agent import as_tool

        return as_tool(
            self.as_agent(model=model),
            name=name if name is not None else self.name,
            description=description if description is not None else self.description,
        )


__all__ = ["Skill"]
