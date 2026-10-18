"""Host invocation identity and task-local binding; never authorization evidence."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ToolInvocation:
    """Distinct host, provider and transport identities within a trusted call.

    Provider/transport IDs may be absent and are untrusted correlation data.
    Neither a client ID nor its metadata supplies tenant scope or permission.
    """

    host_invocation_id: str
    provider_tool_call_id: str | None = None
    transport_request_id: str | int | None = None
    call_id: str = ""
    attempt_id: str | None = None


_current: ContextVar[ToolInvocation | None] = ContextVar("agentkit_tool_invocation", default=None)


def current_tool_invocation() -> ToolInvocation | None:
    """Return the invocation during policy, implementation and observation."""
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
    """Bind a fresh identity while retaining the trusted host context."""
    identity = ToolInvocation(
        f"host-tool-{uuid4().hex}",
        provider_id,
        transport_id,
        getattr(ctx, "correlation_id", ""),
        getattr(ctx, "attempt_id", None),
    )
    token = _current.set(identity)
    try:
        yield _InvocationContext(ctx, identity)
    finally:
        _current.reset(token)
