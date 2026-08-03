"""``FakeTool`` — a Tool stub for unit tests.

Accepts a deterministic ``responder`` callable (or static result) at
construction. Records every ``run`` invocation so a test can assert
on call shape + ordering.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolSchema


def _default_schema() -> ToolSchema:
    return ToolSchema(
        name="fake_tool",
        description="fake tool for unit tests",
        parameters={"type": "object"},
    )


@dataclass
class FakeTool:
    """A ``Tool``-Protocol-conforming fake.

    Construct with a static ``responder`` value (returned verbatim
    from every ``run``) OR with a callable ``responder(args) -> Any``
    that may be sync or async. Any exception the callable raises
    propagates, letting a test exercise the error path.

    Every ``run`` call is appended to ``self.calls`` as the (copied)
    ``args`` dict, so tests can do
    ``assert tool.calls == [{"x": 1}, {"x": 2}]``.

    The default ``schema`` is the permissive ``{"type": "object"}`` —
    any JSON object accepted. Override per-test if you want the
    Protocol's advertised schema to reflect a real shape.
    """

    name: str = "fake_tool"
    description: str = "fake tool for unit tests"
    responder: Any = None  # Callable[[dict], Any] | Awaitable | static value
    side_effecting: bool = False
    requires_approval: bool = False
    output_schema: dict[str, Any] | None = None
    schema: ToolSchema = field(default_factory=_default_schema)
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    async def run(self, args: dict[str, Any], ctx: Ctx) -> Any:
        del ctx
        self.calls.append(dict(args))
        responder = self.responder
        if callable(responder):
            result = responder(args)
            if inspect.isawaitable(result):
                result = await result
            return result
        return responder


__all__ = ["FakeTool"]
