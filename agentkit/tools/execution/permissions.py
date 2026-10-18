"""Host-supplied policy and lifecycle contracts, independent of plugin packages."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from agentkit.tools.execution.context import ToolInvocation


class PermissionDecision(StrEnum):
    """DENY is final; ASK needs an explicit host approval handler."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class ToolInvocationRequest:
    """Policy input from the host context, with frozen, validated arguments.

    Arguments and scope are private host data and excluded from repr. Policy
    callbacks are trusted host code and must not log them indiscriminately.
    """

    invocation: ToolInvocation
    name: str
    arguments: Mapping[str, Any] = field(repr=False)
    scope: Any = field(repr=False)


@dataclass(frozen=True)
class ToolLifecycleEvent:
    """Safe ordered metadata; no arguments, results or exception messages."""

    invocation: ToolInvocation
    name: str
    phase: Literal["requested", "finished"]
    code: str


PermissionCallback = Callable[[ToolInvocationRequest], PermissionDecision | Awaitable[PermissionDecision]]
ToolObserver = Callable[[ToolLifecycleEvent], Any]
