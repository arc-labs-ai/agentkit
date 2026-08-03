"""``ToolRegistry`` — Composite over the Tool Protocol, doing pure lookup.

Now PURE LOOKUP. Execution (span/retry/egress/idempotency/audit) moved to the Invoker's
tool-middleware chain, so the registry just holds tools and answers name→tool, schemas,
and approval questions. A `FunctionTool` declares its data (schema + safety flags); the loop
turns a `ToolCall` into a `ToolRequest` and hands it to `ctx.invoker.invoke_tool`.

Name collisions raise instead of silently overwriting: an unqualified
``self._tools[tool.name] = tool`` swap would let a typo or import-order accident
replace the active tool — the model still sees the same advertised name, but
the implementation has changed. Use ``register(tool, replace=True)`` for
deliberate hot-swap.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agentkit.tools.base import Tool
from agentkit.tools.function import FunctionTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Any, *, replace: bool = False) -> Tool:
        """Register a `FunctionTool` — or a **plain callable**, auto-converted via `from_callable`.
        A plain callable goes through the same tool-writing contract as `@tool` (docstring floor,
        explicit `side_effecting` if you want to deviate from the read-only default); a callable
        whose docstring is too thin surfaces a `ToolDefinitionError` here, not at runtime.

        Raises :class:`ValueError` on name collision unless ``replace=True``.
        The model advertises tools by name; a silent overwrite would change
        the implementation under the agent without any signal."""
        # ``Tool`` is a ``@runtime_checkable`` Protocol; ``isinstance`` refuses
        # a partially-shaped object at the seam (e.g. one with ``name``/``run``
        # but no ``schema``) rather than letting it blow up later in
        # ``schemas()`` or ``requires_approval()``.
        if not isinstance(tool, Tool):
            tool = FunctionTool.from_callable(tool)
        if tool.name in self._tools and not replace:
            raise ValueError(
                f"ToolRegistry: tool {tool.name!r} already registered. "
                "Pass replace=True to swap deliberately."
            )
        self._tools[tool.name] = tool
        result: Tool = tool
        return result

    @classmethod
    def from_tools(cls, items: Iterable[Any]) -> ToolRegistry:
        """Build a registry from a list mixing `FunctionTool`s and plain functions (`[get_weather, …]`).
        Each plain callable is validated by `FunctionTool.from_callable`; a thin docstring or other
        contract violation raises `ToolDefinitionError` here instead of silently wrapping a bad tool.

        Name collisions across the list propagate the same
        :class:`ValueError` as ``register``."""
        reg = cls()
        for it in items:
            reg.register(it)
        return reg

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"no tool {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[Any]:
        """Advertised tools, stable order (so a cacheable system+tools prefix stays byte-identical)."""
        return [t.schema for n in sorted(self._tools) if (t := self._tools[n]).schema is not None]

    def requires_approval(self, name: str) -> bool:
        return bool(getattr(self._tools.get(name), "requires_approval", False))

    def tools(self) -> list[Tool]:
        return list(self._tools.values())
