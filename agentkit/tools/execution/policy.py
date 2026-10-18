"""One tool result policy for in-process calls and MCP serving."""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from agentkit.core.contracts.types import ToolSchema
from agentkit.core.execution.concurrency import Cancelled
from agentkit.tools.declarations.base import Tool
from agentkit.tools.declarations.errors import ToolExecutionError
from agentkit.tools.execution.cancellation import run_cancellable
from agentkit.tools.execution.context import bind_tool_invocation, current_tool_invocation
from agentkit.tools.execution.permissions import (
    PermissionCallback,
    PermissionDecision,
    ToolInvocationRequest,
    ToolLifecycleEvent,
    ToolObserver,
)
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


class _PolicyRefusal(PermissionError):
    def __init__(self, code: str, explanation: str) -> None:
        self.code = code
        super().__init__(explanation)


class _ObserverFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSet:
    """Execution policy handed across the tool seam, as a value.

    Frozen because `max_concurrency` is the one field here that is not advice:
    the fan-out site turns it into an `asyncio.Semaphore`, and a bound below one
    is a pool that admits nobody. `__post_init__` refuses such a bound, and
    while this was a mutable dataclass that refusal was advisory —
    `tools.max_concurrency = 0` after construction never reached it, so the
    value a run fanned out on was not the value anything had checked. Vary a
    policy with `dataclasses.replace` (`with_registry` already does, per
    invocation); the deferred-discovery set is still mutated in place, since
    freezing forbids rebinding a field and not mutating what one holds.
    """

    registry: ToolRegistry
    overflow: OverflowPolicy = field(default_factory=OverflowPolicy)
    images: bool = True
    describe_limits: bool = True
    result_hook: Callable[[str, Any], Any] | None = None
    timeout_s: float = 60
    max_concurrency: int = 4
    # Empty means eager loading. Deferred names are exposed by search_tools.
    deferred: tuple[str, ...] = ()
    permissions: tuple[PermissionCallback, ...] = ()
    approval: PermissionCallback | None = None
    observers: tuple[ToolObserver, ...] = ()
    max_input_bytes: int = 1_048_576
    _loaded: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("tool concurrency must be positive")
        if self.max_input_bytes < 1:
            raise ValueError("tool input bound must be positive")
        # ``object.__setattr__`` because this dataclass is frozen — the same
        # spelling ``Answer``, ``ProviderUsage`` and ``AgentDefinition`` use to
        # normalise their own fields. Frozen is load-bearing here rather than
        # decorative: the checks above are the only validation this policy gets,
        # and a plain dataclass let a caller write ``tools.max_concurrency = 0``
        # straight past them, so the bound a run fanned out on was not the one
        # anything had checked.
        object.__setattr__(self, "permissions", tuple(self.permissions))
        object.__setattr__(self, "observers", tuple(self.observers))
        if self.deferred and any(name in self.registry.names() for name in DeferredToolName):
            raise ValueError("deferred discovery names are reserved by ToolSet")

    def with_registry(self, registry: ToolRegistry) -> ToolSet:
        """Apply this policy to an authorized registry with fresh discovery state."""
        return replace(self, registry=registry, deferred=tuple(n for n in self.deferred if n in registry.names()))

    @property
    def effective_timeout_s(self) -> float | None:
        """Largest declared deadline, or ``None`` if any tool is unbounded."""
        deadlines = [getattr(tool, "timeout_s", None) or self.timeout_s for tool in self.registry.tools()]
        if any(deadline is None for deadline in deadlines):
            return None
        return max(deadlines, default=self.timeout_s)

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
        """One permission and lifecycle path for SDK and MCP dispatch.

        A registry is the explicit grant set. Host callbacks add restrictions;
        observers cannot approve effects. All awaits, including authorization,
        are cancellation-aware. Synchronous host code must cooperate itself.
        """
        if getattr(ctx, "tool_invocation", None) is None or current_tool_invocation() is None:
            with bind_tool_invocation(ctx) as invocation_ctx:
                return await self.execute(name, args, invocation_ctx)
        identity = ctx.tool_invocation
        code = "tool_error"
        try:
            await self._observe(ToolLifecycleEvent(identity, name, "requested", "pending"), ctx)
            result = await run_cancellable(self._execute(name, args, ctx), ctx)
            ctx.check_cancelled()
            code = "ok"
            if result.is_error:
                candidate = result.structured.get("code") if result.structured else None
                safe_codes = {
                    "permission_denied",
                    "permission_unattended",
                    "permission_invalid",
                    "permission_failed",
                    "tool_input_too_large",
                    "tool_arguments_invalid",
                    "tool_deferred",
                    "tool_not_discovered",
                }
                code = candidate if isinstance(candidate, str) and candidate in safe_codes else "tool_error"
            return result
        except (asyncio.CancelledError, Cancelled):
            code = "cancelled"
            raise
        except Exception as exc:
            code = "observer_failed" if isinstance(exc, _ObserverFailed) else "tool_error"
            return ToolResult(
                ({"type": "text", "text": f"{type(exc).__name__}: tool invocation failed"},), is_error=True
            )
        finally:
            # Terminal metadata contains no late result. Cancellation does not
            # skip the observer, but its own deadline still bounds cleanup.
            await self._observe(ToolLifecycleEvent(identity, name, "finished", code), None)

    async def _observe(self, event: ToolLifecycleEvent, ctx: Any) -> None:
        for observer in self.observers:
            try:
                async with asyncio.timeout(self.timeout_s):
                    observed = observer(event)
                    if inspect.isawaitable(observed):
                        if ctx is None:
                            await observed
                        else:
                            await run_cancellable(observed, ctx)
            except (asyncio.CancelledError, Cancelled):
                raise
            except Exception:
                if event.phase == "requested":
                    raise _ObserverFailed("host lifecycle observer failed") from None
                # An observer failure after an effect cannot undo or replay it.
                # Warn safely rather than replacing a settled tool result.
                import logging

                logging.getLogger(__name__).warning("Host tool terminal observer failed")

    async def _authorize(self, tool: Any, name: str, args: Mapping[str, Any], ctx: Any) -> None:
        request = ToolInvocationRequest(ctx.tool_invocation, name, args, getattr(ctx, "scope", None))
        needs_approval = bool(tool.requires_approval)
        for callback in self.permissions:
            try:
                decision = callback(request)
                if inspect.isawaitable(decision):
                    decision = await run_cancellable(decision, ctx)
            except (Cancelled, asyncio.CancelledError):
                raise
            except Exception:
                raise _PolicyRefusal("permission_failed", "Host permission callback failed; tool was not run") from None
            if not isinstance(decision, PermissionDecision):
                raise _PolicyRefusal("permission_invalid", "Host permission callback returned an invalid decision")
            if decision == PermissionDecision.DENY:
                raise _PolicyRefusal("permission_denied", "Host permission denied this tool invocation")
            needs_approval |= decision == PermissionDecision.ASK
        if needs_approval:
            if self.approval is None:
                raise _PolicyRefusal("permission_unattended", "Tool requires host approval before execution")
            try:
                decision = self.approval(request)
                if inspect.isawaitable(decision):
                    decision = await run_cancellable(decision, ctx)
            except (Cancelled, asyncio.CancelledError):
                raise
            except Exception:
                raise _PolicyRefusal("permission_failed", "Host approval failed; tool was not run") from None
            if decision is not PermissionDecision.ALLOW:
                raise _PolicyRefusal("permission_denied", "Host approval did not allow this tool invocation")
        ctx.check_cancelled()

    async def _execute(self, name: str, args: Mapping[str, Any], ctx: Any) -> ToolResult:
        try:
            ctx.check_cancelled()
            from agentkit.core.serialization._immutable import deep_freeze

            if len(json.dumps(args, allow_nan=False).encode("utf-8")) > self.max_input_bytes:
                raise _PolicyRefusal("tool_input_too_large", "Tool input exceeds the host transport bound")
            args = deep_freeze(dict(args))
            if name == DeferredToolName.CALL and self.deferred:
                target = args.get("name")
                if not isinstance(target, str) or target not in self._loaded or target not in self.deferred:
                    raise _PolicyRefusal(
                        "tool_not_discovered", "tool must be discovered with search_tools before call_loaded_tool"
                    )
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
                raise _PolicyRefusal("tool_deferred", "tool is deferred; use search_tools first")
            tool = self.registry.get(name)
            if tool.schema is not None:
                from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

                if not Draft202012Validator(tool.schema.parameters).is_valid(json.loads(json.dumps(args))):
                    raise _PolicyRefusal("tool_arguments_invalid", "Tool arguments do not match the granted schema")
            async with asyncio.timeout(self.timeout_s):
                await self._authorize(tool, name, args, ctx)
            timeout = getattr(tool, "timeout_s", None)
            if timeout is None:
                timeout = self.timeout_s
            async with asyncio.timeout(timeout):
                value = await run_cancellable(tool.run(args, ctx), ctx)
            if tool.output_schema is not None:
                from agentkit.models.output import adapt

                value = adapt(tool.output_schema).serialize(adapt(tool.output_schema).validate(value))
            return await self.handle(name, value)
        except _PolicyRefusal as exc:
            return ToolResult(({"type": "text", "text": str(exc)},), {"code": exc.code}, is_error=True)
        except ToolExecutionError as exc:
            return ToolResult(({"type": "text", "text": str(exc)},), {"code": exc.code}, is_error=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            from agentkit.core.execution.concurrency import Cancelled

            if isinstance(exc, Cancelled):
                raise
            explanation = (
                "image results are disabled by the host"
                if isinstance(exc, ValueError) and str(exc) == "image results are disabled by the host"
                else "tool requires current host permission or approval"
                if isinstance(exc, PermissionError)
                else f"tool {name!r} failed"
            )
            return ToolResult(({"type": "text", "text": f"{type(exc).__name__}: {explanation}"},), is_error=True)

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
    activity_observer: Callable[[bool], Any] | None = None,
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
        timeout_s: float | None = None

        async def run(self, args: Mapping[str, Any], ctx: Any) -> Any:
            from agentkit.core.serialization._immutable import deep_freeze

            observed_args = deep_freeze(dict(args))
            if activity_observer is not None:
                activity_observer(True)
            try:
                result = await tools.execute(self.name, args, ctx)
            finally:
                if activity_observer is not None:
                    activity_observer(False)
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
            requires_approval=bool(getattr(original, "requires_approval", False)) and tools.approval is None,
            timeout_s=getattr(original, "timeout_s", None),
        )

    def visible() -> list[Tool]:
        return cast(list[Tool], [wrap(schema) for schema in tools.schemas()])

    # Dispatch and CLI permission configuration include every declared tool.
    # Discovery remains deferred; execution still enforces search-before-use.
    registered = {schema.name: schema for schema in tools.registry.schemas()}
    registered.update({schema.name: schema for schema in tools.schemas()})
    # MCP's advertised client timeout must cover the longest permitted tool.
    # Each individual call is still bounded by ToolSet.execute's own deadline.
    timeout_s = (
        tools.effective_timeout_s
        if timeout_s is None
        else max((timeout_s, tools.effective_timeout_s) if tools.effective_timeout_s is not None else (timeout_s,))
    )
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
