"""Cognition Strategies — turn-taking plug-ins for Agent.

The Agent delegates its run loop to a ``Cognition``. Three first-party
cognitions cover the regimes the framework ships today:

- ``SingleCallCognition``  — one chat call + parse-and-repair.
- ``ReActCognition``       — tool-loop with HITL suspend/resume + durable
                             checkpoints.
- ``CoordinatorCognition`` — multi-agent loop driven by a ``Policy``.

Adding a new cognition is implementing the ``Cognition`` Protocol;
no Agent subclassing required.
"""

from agentkit.agents.cognition.base import Cognition
from agentkit.agents.cognition.claude_cli import (
    ClaudeCliCognition,
    ClaudeCliSession,
    InterruptReceipt,
)
from agentkit.agents.cognition.coordinator import CoordinatorCognition
from agentkit.agents.cognition.react import ReActCognition
from agentkit.agents.cognition.single_call import SingleCallCognition

__all__ = [
    "ClaudeCliCognition",
    "ClaudeCliSession",
    "InterruptReceipt",
    "Cognition",
    "CoordinatorCognition",
    "ReActCognition",
    "SingleCallCognition",
]
