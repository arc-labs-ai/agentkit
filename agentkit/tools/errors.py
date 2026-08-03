"""Tool-related error types raised at decoration/registration time and at result-validation time."""

from __future__ import annotations

from typing import Any


class ToolDefinitionError(ValueError):
    """Raised at decoration/registration time when a tool fails the framework's wiring contract:
    missing/thin docstring, missing `side_effecting=` declaration, or other static defects the
    framework can catch before the agent ever runs. Subclasses `ValueError` so existing
    `except ValueError` paths still trip — the failure mode is genuinely a bad value."""


class ToolShapeError(Exception):
    """A tool's result didn't match its declared ``output_schema``.

    Raised by :meth:`FunctionTool.run` AFTER the function executed successfully but
    the result failed schema validation. The retry middleware catches this and
    reflects the error back to the model as a tool-call failure — the model sees a
    structured "tool returned a value that doesn't match its declared output"
    message and can re-issue the call (often with different args) or pivot.

    Distinct from :class:`OutputCoercionError` (which is about MODEL response
    coercion); tool shape mismatches have a different fire site (after the tool
    function ran) and a different recovery shape on the model side (the model
    didn't author the bad value — the tool did)."""

    def __init__(
        self,
        tool_name: str,
        expected: str,
        raw: Any,
        errors: list[str] | None = None,
    ) -> None:
        super().__init__(f"tool {tool_name!r} returned value not matching schema {expected!r}")
        self.tool_name = tool_name
        self.expected = expected
        self.raw = raw
        self.errors = errors or []
