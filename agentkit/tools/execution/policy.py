"""One tool result policy for in-process calls and MCP serving."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from agentkit.core.contracts.types import ToolSchema
from agentkit.tools.declarations.base import Tool
from agentkit.tools.execution.cancellation import run_cancellable
from agentkit.tools.registry.registry import ToolRegistry


class DeferredToolName(StrEnum):
    SEARCH = "search_tools"
    CALL = "call_loaded_tool"


@dataclass(frozen=True)
class OverflowPolicy:
    max_bytes: int = 50_000
    directory: Path | None = None

    def __post_init__(self) -> None:
        if self.max_bytes < 256:
            raise ValueError("overflow bound must be at least 256 bytes")


@dataclass(frozen=True)
class ToolResult:
    content: tuple[Mapping[str, Any], ...] = field(hash=False)
    structured: Mapping[str, Any] | None = field(default=None, hash=False)
    is_error: bool = False
    overflowed: bool = False

    def __post_init__(self) -> None:
        from agentkit.core.serialization._immutable import deep_freeze

        object.__setattr__(self, "content", tuple(deep_freeze(dict(x)) for x in self.content))
        object.__setattr__(self, "structured", deep_freeze(self.structured))

    @property
    def text(self) -> str:
        return "\n".join(str(x.get("text", "")) for x in self.content if x.get("type") == "text")


def describe_limits(schema: dict[str, Any]) -> dict[str, Any]:
    """Add constraints in words without rewriting example/default values."""
    out: dict[str, Any] = json.loads(json.dumps(schema))
    notes = []
    for key, phrase in {
        "minimum": "At least",
        "maximum": "At most",
        "maxItems": "At most this many items",
        "maxLength": "At most this many characters",
        "pattern": "Must match pattern",
    }.items():
        if key in out:
            notes.append(f"{phrase}: {out[key]}.")
    if notes:
        out["description"] = " ".join([out.get("description", ""), *notes]).strip()
    for key in ("properties", "$defs"):
        if key in out:
            out[key] = {k: describe_limits(v) for k, v in out[key].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = describe_limits(out["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in out:
            out[key] = [describe_limits(v) for v in out[key]]
    return out


def _invalid_json_constant(value: str) -> Any:
    raise ValueError("arguments_json contains an invalid JSON numeric constant")


@dataclass
class ToolSet:
    registry: ToolRegistry
    overflow: OverflowPolicy = field(default_factory=OverflowPolicy)
    images: bool = True
    describe_limits: bool = True
    result_hook: Callable[[str, Any], Any] | None = None
    timeout_s: float = 60
    # Empty means eager loading. Deferred names are exposed by search_tools.
    deferred: tuple[str, ...] = ()
    _loaded: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.deferred and any(name in self.registry.names() for name in DeferredToolName):
            raise ValueError("deferred discovery names are reserved by ToolSet")

    def with_registry(self, registry: ToolRegistry) -> ToolSet:
        """Apply this policy to an authorized registry with fresh discovery state."""
        return replace(self, registry=registry, deferred=tuple(n for n in self.deferred if n in registry.names()))

    def schemas(self) -> list[ToolSchema]:
        schemas = []
        for s in self.registry.schemas():
            if s.name in self.deferred and s.name not in self._loaded:
                continue
            params = describe_limits(s.parameters) if self.describe_limits else s.parameters
            descriptions = [p.get("description", "") for p in params.get("properties", {}).values()]
            description = s.description + (
                " Parameter constraints: " + " ".join(descriptions) if any(descriptions) else ""
            )
            schemas.append(replace(s, parameters=params, description=description))
        if self.deferred:
            schemas.append(
                ToolSchema(
                    DeferredToolName.SEARCH,
                    "Find and load tools by name or description. Invoke each discovered tool with call_loaded_tool.",
                    {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Words to search for in tool names and descriptions.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                )
            )
            schemas.append(
                ToolSchema(
                    DeferredToolName.CALL,
                    "Execute a tool previously discovered by search_tools. "
                    "Supply its name and arguments as a JSON object string.",
                    {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Exact discovered tool name."},
                            "arguments_json": {
                                "type": "string",
                                "description": "JSON object containing arguments matching the discovered inputSchema.",
                                "maxLength": 50_000,
                            },
                        },
                        "required": ["name", "arguments_json"],
                        "additionalProperties": False,
                    },
                )
            )
        return schemas

    async def execute(self, name: str, args: Mapping[str, Any], ctx: Any) -> ToolResult:
        """Execution errors are model-visible; cancellation still propagates."""
        try:
            ctx.check_cancelled()
            if name == DeferredToolName.CALL and self.deferred:
                target = args.get("name")
                if not isinstance(target, str) or target not in self._loaded or target not in self.deferred:
                    raise PermissionError("tool must be discovered with search_tools before call_loaded_tool")
                encoded = args.get("arguments_json")
                if not isinstance(encoded, str) or len(encoded) > 50_000:
                    raise ValueError("arguments_json must be a JSON object string within 50000 characters")
                arguments = json.loads(encoded, parse_constant=_invalid_json_constant)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments_json must decode to an object")
                return await self.execute(target, arguments, ctx)
            if name == DeferredToolName.SEARCH and self.deferred:
                query = str(args.get("query", "")).lower().split()
                matched = [
                    n
                    for n in self.deferred
                    if n in self.registry.names()
                    and any(w in (n + " " + self.registry.get(n).description).lower() for w in query)
                ]
                self._loaded.update(matched[:20])
                return await self.handle(
                    name,
                    {
                        "loaded": matched[:20],
                        "invoke": {
                            "tool": DeferredToolName.CALL,
                            "instructions": "Call call_loaded_tool with name set to a loaded tool name and "
                            "arguments_json set to a JSON object string matching its inputSchema. "
                            "Use this dispatcher even if newly loaded tools are absent from your cached catalogue.",
                        },
                        "tools": [
                            {"name": s.name, "description": s.description, "inputSchema": dict(s.parameters)}
                            for s in self.schemas()
                            if s.name in matched[:20]
                        ],
                    },
                )
            if name in self.deferred and name not in self._loaded:
                raise ValueError("tool is deferred; use search_tools first")
            tool = self.registry.get(name)
            if tool.requires_approval:
                raise PermissionError("tool requires host approval before execution")
            async with asyncio.timeout(self.timeout_s):
                value = await run_cancellable(tool.run(args, ctx), ctx)
            if tool.output_schema is not None:
                from agentkit.models.output import adapt

                value = adapt(tool.output_schema).serialize(adapt(tool.output_schema).validate(value))
            return await self.handle(name, value)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            from agentkit.core.execution.concurrency import Cancelled

            if isinstance(exc, Cancelled):
                raise
            return ToolResult(({"type": "text", "text": f"{type(exc).__name__}: {exc}"},), is_error=True)

    async def handle(self, name: str, value: Any) -> ToolResult:
        if self.result_hook is not None:
            value = self.result_hook(name, value)
            if hasattr(value, "__await__"):
                value = await value
        if isinstance(value, ToolResult):
            result = value
        elif isinstance(value, dict) and isinstance(value.get("content"), (list, tuple)):
            result = ToolResult(tuple(value["content"]), value.get("structuredContent"), bool(value.get("isError")))
        elif isinstance(value, dict):
            result = ToolResult(({"type": "text", "text": json.dumps(value, ensure_ascii=False)},), value)
        else:
            result = ToolResult(({"type": "text", "text": value if isinstance(value, str) else json.dumps(value)},))
        if not self.images and any(x.get("type") == "image" for x in result.content):
            raise ValueError("image results are disabled by the host")
        blob = json.dumps(
            {"content": result.content, "structuredContent": result.structured}, ensure_ascii=False
        ).encode()
        if len(blob) > self.overflow.max_bytes:

            def spill() -> str:
                with tempfile.NamedTemporaryFile(
                    prefix="agentkit-tool-", suffix=".json", delete=False, dir=self.overflow.directory
                ) as stream:
                    stream.write(blob)
                    return stream.name

            path = await asyncio.to_thread(spill)
            result = ToolResult(
                (
                    {
                        "type": "text",
                        "text": f"Result ({len(blob)} bytes) saved to {path}. "
                        "Read that file in bounded chunks to continue.",
                    },
                ),
                is_error=result.is_error,
                overflowed=True,
            )
        return result


def serve(
    tools: ToolSet,
    *,
    ctx: Any,
    name: str = "agentkit",
    execution_observer: Callable[[str, Mapping[str, Any], ToolResult], Any] | None = None,
    timeout_s: float | None = None,
) -> Any:
    """Serve the same execution policy to CLI modes via the existing MCP transport."""
    from agentkit.integrations.mcp import serve_registry

    @dataclass
    class WrappedTool:
        original: Any
        schema: ToolSchema

        @property
        def name(self) -> str:
            return self.schema.name

        @property
        def description(self) -> str:
            return self.schema.description

        output_schema: Any = None
        side_effecting: bool = False
        requires_approval: bool = False

        async def run(self, args: Mapping[str, Any], ctx: Any) -> Any:
            from agentkit.core.serialization._immutable import deep_freeze

            observed_args = deep_freeze(dict(args))
            result = await tools.execute(self.name, args, ctx)
            if execution_observer is not None:
                observed = execution_observer(self.name, observed_args, result)
                if hasattr(observed, "__await__"):
                    await observed
            return result

    def wrap(schema: ToolSchema) -> WrappedTool:
        original = tools.registry.get(schema.name) if schema.name not in DeferredToolName else None
        return WrappedTool(
            original,
            schema,
            side_effecting=schema.name == DeferredToolName.CALL or bool(getattr(original, "side_effecting", False)),
            requires_approval=bool(getattr(original, "requires_approval", False)),
        )

    def visible() -> list[Tool]:
        return cast(list[Tool], [wrap(schema) for schema in tools.schemas()])

    # Dispatch and CLI permission configuration include every declared tool.
    # Discovery remains deferred; execution still enforces search-before-use.
    registered = {schema.name: schema for schema in tools.registry.schemas()}
    registered.update({schema.name: schema for schema in tools.schemas()})
    return serve_registry(
        cast(list[Tool], [wrap(schema) for schema in registered.values()]),
        name=name,
        ctx=ctx,
        result_hook=_mcp_result,
        json_response=not bool(tools.deferred),
        tools_provider=visible,
        timeout_s=timeout_s,
    )


def _mcp_result(name: str, value: Any) -> Any:
    from mcp import types

    if not isinstance(value, ToolResult):
        return value
    # MCP models validate each discriminated content type at the boundary.
    return types.CallToolResult(
        content=cast(Any, list(value.content)),
        structuredContent=dict(value.structured) if value.structured else None,
        isError=value.is_error,
    )
