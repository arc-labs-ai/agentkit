"""Tool Protocol — explicit ``@runtime_checkable`` Protocol that ``FunctionTool``
and any custom remote/MCP/skill-as-tool adapter satisfy structurally. Brings
tools onto the same architectural footing as ``Cognition`` (turn-taking
strategy Protocol) and ``MemorySource`` (knowledge-probe Protocol).

These tests pin: (1) ``FunctionTool`` conforms, (2) a hand-rolled class with
the right shape conforms, (3) a class missing a required attribute does NOT
conform, and (4) the Protocol is declared ``@runtime_checkable`` so isinstance
works at all."""

from __future__ import annotations

from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolSchema
from agentkit.tools import FunctionTool, Tool


def _make_function_tool() -> FunctionTool:
    def lookup(query: str) -> str:
        """Look up a value by query string — demo callable for the protocol smoke."""
        return f"result: {query}"

    return FunctionTool.from_callable(lookup, side_effecting=False)


def test_function_tool_satisfies_protocol():
    """The canonical impl wraps a Python callable and MUST satisfy ``Tool``."""
    ft = _make_function_tool()
    assert isinstance(ft, Tool)


def test_custom_tool_class_satisfies_protocol():
    """A hand-rolled class with the right attributes + async ``run()`` IS a Tool —
    no inheritance needed (structural typing)."""

    class RemoteTool:
        name = "remote"
        description = "Call a remote procedure across the network."
        schema = ToolSchema(
            name="remote",
            description="Call a remote procedure across the network.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        output_schema: dict | None = None
        side_effecting = True
        requires_approval = False

        async def run(self, args: dict, ctx: Ctx) -> Any:
            return {"ok": True}

    assert isinstance(RemoteTool(), Tool)


def test_missing_field_fails_isinstance():
    """A class without ``side_effecting`` is NOT a Tool — runtime_checkable enforces
    attribute presence."""

    class HalfTool:
        name = "half"
        description = "An incomplete tool missing the side_effecting flag."
        schema = ToolSchema(
            name="half",
            description="An incomplete tool missing the side_effecting flag.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        output_schema: dict | None = None
        requires_approval = False
        # side_effecting deliberately omitted

        async def run(self, args: dict, ctx: Ctx) -> Any:
            return None

    assert not isinstance(HalfTool(), Tool)


def test_missing_run_fails_isinstance():
    """A class without ``run`` is NOT a Tool — the action verb is non-negotiable."""

    class NoRunTool:
        name = "norun"
        description = "Has the fields but no async run() entry point."
        schema = ToolSchema(
            name="norun",
            description="Has the fields but no async run() entry point.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        output_schema: dict | None = None
        side_effecting = False
        requires_approval = False

    assert not isinstance(NoRunTool(), Tool)


def test_protocol_runtime_checkable():
    """Smoke that ``Tool`` is wired with ``@runtime_checkable`` — without it,
    ``isinstance(x, Tool)`` would raise TypeError."""
    # The marker attribute set by @runtime_checkable on a Protocol class.
    assert getattr(Tool, "_is_runtime_protocol", False) is True

    # All required members surface on the Protocol's introspection set.
    members = set(getattr(Tool, "__protocol_attrs__", set()))
    for required in (
        "name",
        "description",
        "schema",
        "output_schema",
        "side_effecting",
        "requires_approval",
        "run",
    ):
        assert required in members, f"{required!r} missing from Tool protocol members"
