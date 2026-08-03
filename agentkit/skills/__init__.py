"""Skills — the Facade between Tool and Agent.

A ``Skill`` is a curated bundle of (prompt + cognition + memory) packaged
as a wirable unit. It is what users mean when they say "give me a
researcher" — a specialised agent recipe other agents can either
*invoke as a tool* (``skill.as_tool()``) or that a host can *materialise
as a runnable Agent* (``skill.as_agent()``).

Skills are immutable value objects. Tools live on the cognition, prompt
lives on the skill, memory lives on the skill. Compose at wire-time,
reuse across runs.

The framework ships ZERO concrete Skills — apps build their own, the
same way they bring their own Tools.
"""

from agentkit.skills.skill import Skill

__all__ = ["Skill"]
