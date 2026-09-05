"""Cognition Strategies — turn-taking plug-ins for Agent.

The Agent delegates its run loop to a ``Cognition``. Three first-party
cognitions cover the regimes the framework ships today:

- ``SingleCallCognition``  — one chat call + parse-and-repair.
- ``ReActCognition``       — tool-loop with HITL suspend/resume + durable
                             checkpoints.
- ``CoordinatorCognition`` — multi-agent loop driven by a ``Policy``.

Two more hand the loop to a coding CLI the user installed themselves, which is
the one regime where agentkit does NOT own the turn-taking:

- ``ClaudeCliCognition``   — subprocesses Anthropic's ``claude``.
- ``CodexCliCognition``    — subprocesses OpenAI's ``codex``.

They are deliberately parallel — same ``drive`` contract, same terminal-event
guarantee, same ``spawn=`` seam — and deliberately not identical, because the
binaries are not: see each class's docstring for the four places they diverge
and why papering over any of them would be a field that silently does nothing.

Both CLI cognitions share a policy layer over the things a service has to be
able to state rather than inherit — :class:`CliTimeouts` (liveness bounds with
a typed reason per bound), ``env_policy`` (which credentials the child
resolves), ``native_tool_policy`` (refusing tools the middleware cannot govern)
and :class:`StructuredOutputFailure` (schema failures with JSON paths instead
of one English sentence). The policy is shared; the transports are not, because
the binaries genuinely differ.

Adding a new cognition is implementing the ``Cognition`` Protocol;
no Agent subclassing required.
"""

from agentkit.agents.cognition._cli_common import (
    CliLineTooLong,
    CliTimedOut,
    CliTimeouts,
    EnvPolicy,
    InvalidSchemaError,
    SchemaViolation,
    StructuredOutputFailure,
)
from agentkit.agents.cognition.base import Cognition
from agentkit.agents.cognition.claude_cli import (
    ClaudeCliCognition,
    ClaudeCliSession,
    CliSpawn,
    InterruptReceipt,
)
from agentkit.agents.cognition.codex_cli import (
    CODEX_NATIVE_TOOLS,
    CODEX_SANDBOX_CAPS,
    ApprovalMode,
    CodexCliCognition,
    CodexCliSession,
    ReasoningEffort,
    SandboxMode,
)
from agentkit.agents.cognition.coordinator import CoordinatorCognition
from agentkit.agents.cognition.react import ReActCognition
from agentkit.agents.cognition.single_call import SingleCallCognition

__all__ = [
    "CODEX_NATIVE_TOOLS",
    "CODEX_SANDBOX_CAPS",
    "ApprovalMode",
    "ClaudeCliCognition",
    "ClaudeCliSession",
    "CliSpawn",
    "CliTimedOut",
    "CliTimeouts",
    "CodexCliCognition",
    "CodexCliSession",
    "Cognition",
    "CoordinatorCognition",
    "EnvPolicy",
    "InterruptReceipt",
    "InvalidSchemaError",
    "ReActCognition",
    "ReasoningEffort",
    "SandboxMode",
    "SchemaViolation",
    "SingleCallCognition",
    "StructuredOutputFailure",
]
