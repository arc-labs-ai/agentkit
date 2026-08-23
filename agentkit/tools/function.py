"""``FunctionTool`` dataclass + ``@tool`` decorator — the canonical Tool impl backed by a Python callable.

Tool-writing discipline (baked into registration, per Anthropic's tool-design guidance):
- Description is the spec: a docstring of >=30 chars is required on every wired callable so the model
  has enough to understand the tool. A missing/thin docstring fails at decoration time, not at runtime.
- Side-effecting is declared, not guessed: `side_effecting` is a REQUIRED field on `FunctionTool`
  and a REQUIRED keyword on `@tool`. The framework's gating + idempotency rely on knowing upfront.
- Idempotency is opt-in metadata that lets safe-retry middleware do its job; read-only tools
  (`search`, `lookup`) are typically `idempotent=True, side_effecting=False`; mutations are
  `side_effecting=True, idempotent=False`. The two are independent.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentkit.capabilities.output_schema import (
    OutputCoercionError,
    SchemaAdapter,
    adapt,
)
from agentkit.kernel.protocols import Ctx
from agentkit.tools.errors import ToolArgumentError, ToolDefinitionError, ToolShapeError
from agentkit.tools.schema import (
    _build_schema,
    _infer_output_schema,
    _is_ctx_param,
    _resolved_description,
)

# Sentinel so explicit ``output_schema=None`` (opt-out of auto-inference) is
# distinguishable from "the caller didn't pass anything" (use auto-inference).
_OUTPUT_AUTO = object()


@dataclass
class FunctionTool:
    name: str
    fn: Callable[[Any, Any], Any]
    description: (
        str  # human/model-facing spec; required — a tool without it is unusable to the model
    )
    side_effecting: bool  # mutates the world → loop can gate it & idempotent middleware can dedupe
    schema: Any = None  # ToolSchema → advertised to the model; None = loop-invisible
    idempotent: bool = (
        False  # safe to retry: read-only lookups, pure functions; independent of side_effecting
    )
    requires_approval: bool = False  # the loop suspends for human approval before this runs
    caps: tuple[str, ...] = ()  # Rule-of-Two tags: "private_data" | "untrusted_content" | "egress"
    url_arg: str | None = None  # if set, this arg is a URL → egress middleware checks it
    output_schema: Any = None
    """Optional schema the tool's RESULT must match. Pydantic BaseModel /
    dataclass / attrs class / raw JSON Schema dict — same accepted shapes
    as ``Agent.output=``. When set, the tool's return value is validated
    through ``adapt(output_schema).validate(result)`` before the framework
    hands it back to the model.

    A mismatch drops a ``tool.shape_mismatch`` span event on the open
    ``execute_tool`` span and raises :class:`ToolShapeError` — catchable by
    the retry middleware so the model can see the structured error and
    recover (re-issue the call with different args or pivot).

    ``None`` (default) means NO check — fast path. Tools that already
    return well-typed Python objects get zero overhead. ``from_callable``
    auto-sets this from the function's return-type annotation when the
    annotation is a Pydantic / dataclass / attrs class; the caller can
    override (or opt out via ``output_schema=None``) on the ``@tool``
    decorator."""
    _output_adapter: SchemaAdapter[Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Build the adapter once at construction time so ``run()`` is a
        # straight ``isinstance``/dispatch on the hot path — never an
        # ``adapt()`` call per tool invocation. Failing here (unsupported
        # shape passed to ``output_schema=``) surfaces at agent-build
        # time, not at first tool call.
        if self.output_schema is not None:
            self._output_adapter = adapt(self.output_schema)
        else:
            self._output_adapter = None

    async def run(self, args: Any, ctx: Ctx) -> Any:
        """Always-async run — bridges a sync `fn` off the event loop so a blocking tool never stalls it.

        When an ``output_schema`` is wired on this tool, the result is run
        through ``adapter.validate(result)`` AFTER the function returns.
        A schema mismatch drops a ``tool.shape_mismatch`` event on the
        currently-open span (the ``execute_tool`` span the tracing
        middleware just opened) and raises :class:`ToolShapeError` so the
        retry middleware can reflect the error back to the model.

        Fast path: no adapter wired → one ``if`` branch, zero overhead."""
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(args, ctx)
        else:
            result = await asyncio.to_thread(self.fn, args, ctx)

        if self._output_adapter is None:
            return result

        try:
            return self._output_adapter.validate(result)
        except OutputCoercionError as exc:
            # Drop the narrative event on whatever span is open (the
            # ``execute_tool`` span B4b's tracing middleware opens around
            # this call). Best-effort — a misbehaving tracer must NEVER
            # break the tool path.
            with contextlib.suppress(Exception):
                trace = getattr(ctx, "trace", None)
                if trace is not None:
                    trace.add_event_to_current_span(
                        "tool.shape_mismatch",
                        tool_name=self.name,
                        expected_shape=self._output_adapter.name,
                    )
            raise ToolShapeError(
                tool_name=self.name,
                expected=self._output_adapter.name,
                raw=result,
                errors=list(exc.errors),
            ) from exc

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        side_effecting: bool = False,
        idempotent: bool = False,
        requires_approval: bool = False,
        caps: tuple[str, ...] = (),
        url_arg: str | None = None,
        output_schema: Any = _OUTPUT_AUTO,
    ) -> FunctionTool:
        """Turn a plain Python function into a tool: inspect its signature + type hints → a JSON-schema
        `ToolSchema`, and wrap execution (sync → off the loop via `to_thread`, async → awaited). The model
        calls it by name with JSON args; a parameter named `ctx`/`context` is injected with the RunContext
        (and not advertised). The function's return value is the tool result; exceptions propagate to the
        framework's per-tool isolation + typed `Failure` — no result is silently swallowed.

        The function MUST carry a docstring (or be passed an explicit `description=`) of at least
        `_MIN_DESCRIPTION_LEN` chars — otherwise raises `ToolDefinitionError` at registration time.
        `side_effecting` defaults to False here for compatibility with the `@tool` decorator which
        enforces explicit declaration; direct callers should always pass it knowingly.

        ``output_schema`` controls tool-result validation:
        - default (``_OUTPUT_AUTO`` sentinel): auto-infer from the function's
          return-type annotation. A Pydantic BaseModel / dataclass / attrs
          class return type triggers validation on every call; everything
          else (primitives, ``Any``, generics) gets no check.
        - explicit class / dict: use this as the output schema (overrides
          auto-inference).
        - explicit ``None``: OPT OUT of validation even if the return type
          looks enforceable."""
        fname_raw = name or getattr(func, "__name__", "tool")
        fname: str = str(fname_raw) if fname_raw is not None else "tool"
        desc = _resolved_description(func, description)
        schema = _build_schema(func, fname, desc)
        sig_params = [
            p
            for p in inspect.signature(func).parameters.values()
            if p.name not in ("self", "cls") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        ctx_params = [p.name for p in sig_params if _is_ctx_param(p)]
        arg_params = [p.name for p in sig_params if not _is_ctx_param(p)]
        required_params = frozenset(
            p.name
            for p in sig_params
            if not _is_ctx_param(p) and p.default is inspect.Parameter.empty
        )
        # A ``**kwargs`` in the signature is the author saying "I accept keys I
        # did not enumerate" — so extras are PASSED THROUGH rather than
        # rejected. Before, they were neither: a tool declaring ``**kwargs``
        # never saw a single extra key.
        accepts_var_kw = any(
            p.kind is p.VAR_KEYWORD for p in inspect.signature(func).parameters.values()
        )
        is_async = inspect.iscoroutinefunction(func)

        async def _invoke(args: Any, ctx: Ctx) -> Any:
            # ``args`` may be a ``MappingProxyType`` (from ``ToolCall.arguments``,
            # which is read-only). Accept any ``Mapping`` — dicts and
            # MappingProxyType alike — so kwargs are always resolvable.
            supplied = dict(args) if isinstance(args, Mapping) else {}
            kwargs = {k: supplied[k] for k in arg_params if k in supplied}

            # A model's call is CHECKED against the signature the model was
            # shown. Silently dropping an unknown key let a defaulted parameter
            # run with its default — a side-effecting tool reporting success for
            # something it was never asked to do. A ``ctx``/``context`` key from
            # the model is dropped rather than reported ONLY when the function
            # declares such a parameter: that name is not in the advertised
            # schema, and the real ctx is injected over it below. A tool with no
            # ctx parameter has no such name to shadow, so the key is reported as
            # unexpected like any other.
            unexpected = tuple(
                k for k in supplied if k not in arg_params and k not in ctx_params
            )
            missing = tuple(k for k in required_params if k not in kwargs)
            if accepts_var_kw:
                kwargs.update({k: supplied[k] for k in unexpected})
                unexpected = ()
            if unexpected or missing:
                raise ToolArgumentError(
                    fname,
                    unexpected=unexpected,
                    missing=missing,
                    accepted=tuple(arg_params),
                )

            for cp in ctx_params:
                kwargs[cp] = ctx
            if is_async:
                return await func(**kwargs)
            return await asyncio.to_thread(lambda: func(**kwargs))

        # Resolve the output_schema: explicit (including explicit None) wins;
        # otherwise auto-infer from the return-type annotation.
        resolved_output_schema = (
            _infer_output_schema(func) if output_schema is _OUTPUT_AUTO else output_schema
        )

        return cls(
            name=fname,
            fn=_invoke,
            description=desc,
            schema=schema,
            side_effecting=side_effecting,
            idempotent=idempotent,
            requires_approval=requires_approval,
            caps=tuple(caps),
            url_arg=url_arg,
            output_schema=resolved_output_schema,
        )


# Sentinel so we can detect "the user forgot to pass side_effecting=" at decoration time
# (rather than silently defaulting and shipping a poorly-declared tool).
_REQUIRED = object()


def tool(
    func: Callable[..., Any] | None = None,
    *,
    side_effecting: Any = _REQUIRED,
    idempotent: bool = False,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    caps: tuple[str, ...] = (),
    url_arg: str | None = None,
    output_schema: Any = _OUTPUT_AUTO,
) -> Any:
    """Decorator / converter: `@tool(side_effecting=False)`, `@tool(side_effecting=True, idempotent=False)`,
    or `tool(fn, side_effecting=...)` → a `FunctionTool`. `side_effecting=` is REQUIRED — the framework's
    gating + idempotency primitives rely on knowing whether a tool mutates the world. The decorated
    callable MUST carry a docstring of >=30 chars (the model needs enough to understand the tool);
    both failure modes raise `ToolDefinitionError` at decoration time (not at call time).

    ``output_schema=`` is opt-in tool-result schema validation. By default (the
    ``_OUTPUT_AUTO`` sentinel) it auto-infers from the function's return-type
    annotation — a Pydantic / dataclass / attrs class triggers validation on
    every tool result; primitives and generics are skipped. Pass an explicit
    class to override; pass ``output_schema=None`` to opt out entirely even when
    the return type looks enforceable."""
    if side_effecting is _REQUIRED:
        raise ToolDefinitionError(
            "@tool requires an explicit `side_effecting=` keyword (True if the tool mutates the "
            "world / has external effects, False if it's read-only). The framework's approval "
            "gating and idempotency depend on knowing upfront."
        )
    kwargs: dict[str, Any] = {
        "name": name,
        "description": description,
        "side_effecting": bool(side_effecting),
        "idempotent": idempotent,
        "requires_approval": requires_approval,
        "caps": caps,
        "url_arg": url_arg,
        "output_schema": output_schema,
    }
    if func is None:
        return lambda f: FunctionTool.from_callable(f, **kwargs)
    return FunctionTool.from_callable(func, **kwargs)
