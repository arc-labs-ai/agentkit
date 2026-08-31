"""Claude CLI integration — join agentkit's primitives to the ``claude`` CLI's own.

``ClaudeCliCognition`` (in ``agentkit.agents.cognition``) is the runner: it
subprocesses the CLI for one agent. This package is the other half — the
adapters that let agentkit values BE the CLI's configuration instead of being
restated as it by hand.

Two of them ship, and they close opposite halves of the same gap. The CLI owns
its own loop, so anything agentkit knows has to reach it as *configuration*:

* :func:`as_cli_agents` projects a :class:`~agentkit.skills.Skill` into a CLI
  sub-agent definition, so a reviewer expressed once is not restated by hand.
* :func:`hook_settings` generates ``PreToolUse`` hooks, so the tool-middleware
  chain applies to the CLI's OWN tools — ``Write``, ``Edit``, ``Bash``,
  ``WebFetch`` — which otherwise run outside the ``Invoker`` entirely and
  therefore outside ``egress``, ``guard``, ``audit`` and ``memoize``.

    from agentkit.integrations.claude_cli import as_cli_agents, hook_settings

No extra required: both are ``asyncio`` and the standard library. Serving your
own tools to the CLI instead is :func:`~agentkit.integrations.mcp.serve_registry`,
which does need the ``mcp`` extra.
"""

from __future__ import annotations

from agentkit.integrations.claude_cli.agents import SkillNotProjectable, as_cli_agents
from agentkit.integrations.claude_cli.hooks import HookDecision, HookSettings, hook_settings

__all__ = [
    "HookDecision",
    "HookSettings",
    "SkillNotProjectable",
    "as_cli_agents",
    "hook_settings",
]
