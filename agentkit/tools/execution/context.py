"""Tool-call association is metadata, never authorization or a tenant scope."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ToolInvocation:
    """Honest identities: a host invocation, optional provider ID and MCP request ID.

    Client-supplied identifiers are correlation data, not permission evidence.
    A missing provider identifier remains None; the host ID never substitutes it.
    """

    host_invocation_id: str
    provider_tool_call_id: str | None = None
    transport_request_id: str | int | None = None


_current: ContextVar[ToolInvocation | None] = ContextVar("agentkit_tool_invocation", default=None)


def current_tool_invocation() -> ToolInvocation | None:
    """Return this task's invocation during execution/observation, or None outside it."""
    return _current.get()


@dataclass(frozen=True)
class _InvocationContext:
    parent: Any
    tool_invocation: ToolInvocation

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parent, name)


@contextmanager
def bind_tool_invocation(
    ctx: Any,
    *,
    provider_id: str | None = None,
    transport_id: str | int | None = None,
) -> Iterator[Any]:
    """Bind fresh task-local identity while preserving the host context unchanged."""
    identity = ToolInvocation(f"host-tool-{uuid4().hex}", provider_id, transport_id)
    token = _current.set(identity)
    try:
        yield _InvocationContext(ctx, identity)
    finally:
        _current.reset(token)
