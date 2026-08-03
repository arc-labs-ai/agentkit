"""Tool Protocol — the Command-pattern surface every action an agent can request must satisfy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolSchema


@runtime_checkable
class Tool(Protocol):
    """An action an agent can request — Command pattern done as a Protocol.

    Any object satisfying this shape IS a Tool: it carries a stable
    ``name`` + ``description``, a JSON ``schema`` describing its input
    args, an optional ``output_schema`` for return-value validation,
    flags for ``side_effecting`` and ``requires_approval`` so the
    runtime can make dispatch decisions, and an async ``run(args, ctx)``
    that actually executes.

    ``FunctionTool`` is the canonical impl wrapping a Python callable.
    Others may implement directly — remote-procedure tools, MCP tools,
    skill-as-tool adapters. The framework dispatches tools through the
    middleware chain (audit / approval-gate / output-coerce / retry)
    regardless of which concrete impl backs them.

    ``schema`` may be ``None`` (loop-invisible tool advertising no
    schema) or a ``ToolSchema``; ``output_schema`` accepts the same
    shapes ``FunctionTool.output_schema`` does — a JSON-Schema dict, a
    Pydantic / dataclass / attrs class, or ``None`` (no check).
    ``run.args`` is ``Mapping[str, Any]`` — a plain dict OR a
    ``MappingProxyType`` handed down from a ``ToolCall`` both satisfy it.
    """

    name: str
    description: str
    schema: ToolSchema | None
    output_schema: type | dict[str, Any] | None
    side_effecting: bool
    requires_approval: bool

    async def run(self, args: Mapping[str, Any], ctx: Ctx) -> Any: ...
