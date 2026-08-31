"""Claude CLI integration — join agentkit's primitives to the ``claude`` CLI's own.

``ClaudeCliCognition`` (in ``agentkit.agents.cognition``) is the runner: it
subprocesses the CLI for one agent. This package is the other half — the
adapters that let agentkit values BE the CLI's configuration instead of being
restated as it by hand.

    from agentkit.integrations.claude_cli import as_cli_agents
"""

from __future__ import annotations

from agentkit.integrations.claude_cli.agents import SkillNotProjectable, as_cli_agents

__all__ = ["SkillNotProjectable", "as_cli_agents"]
