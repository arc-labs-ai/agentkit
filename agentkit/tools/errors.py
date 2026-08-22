"""Tool-related error types raised at decoration/registration time and at result-validation time."""

from __future__ import annotations

from typing import Any


class ToolDefinitionError(ValueError):
    """Raised at decoration/registration time when a tool fails the framework's wiring contract:
    missing/thin docstring, missing `side_effecting=` declaration, or other static defects the
    framework can catch before the agent ever runs. Subclasses `ValueError` so existing
    `except ValueError` paths still trip — the failure mode is genuinely a bad value."""


class ToolArgumentError(ValueError):
    """A tool CALL named arguments the tool does not accept, or omitted required ones.

    The mirror of :class:`ToolShapeError`: that one is a bad value coming OUT of a
    tool, this one is a bad call going IN. Both are raised at call time and both
    are meant to be reflected back to the model, which authored the call and is
    the only party that can fix it.

    Why this is an error rather than a shrug: unknown keys used to be dropped
    silently, and a parameter with a DEFAULT then ran with that default. A model
    calling ``notify(message="page the on-call, prod is down")`` against
    ``def notify(msg: str = "default message")`` got back ``"sent: default
    message"`` — a side-effecting tool reporting success for something it never
    did. Nothing downstream could tell.

    The message names the tool, the offending arguments and the accepted set, so
    the retry middleware can hand the model something it can act on. A tool that
    genuinely wants arbitrary keys declares ``**kwargs`` and receives them.

    Subclasses ``ValueError``: the failure mode is a bad value, and existing
    ``except ValueError`` isolation around tool calls keeps working.
    """

    def __init__(
        self,
        tool_name: str,
        *,
        unexpected: tuple[str, ...] = (),
        missing: tuple[str, ...] = (),
        accepted: tuple[str, ...] = (),
    ) -> None:
        parts = []
        if unexpected:
            parts.append(f"unexpected argument(s) {list(unexpected)}")
        if missing:
            parts.append(f"missing required argument(s) {list(missing)}")
        super().__init__(
            f"tool {tool_name!r} call rejected: "
            + "; ".join(parts)
            + f". Accepted arguments: {list(accepted) or '<none>'}"
        )
        self.tool_name = tool_name
        self.unexpected = tuple(unexpected)
        self.missing = tuple(missing)
        self.accepted = tuple(accepted)


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
