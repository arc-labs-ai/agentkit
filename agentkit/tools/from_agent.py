"""Composition — making the two control models nest.

The framework's promise is that explicit (Workflow) and emergent (leaf and coordinator Agent)
control **compose over one spine**, in both directions:

  • *emergent inside explicit* — a reasoning leaf or coordinator Agent **as a graph node**
    (`Workflow.agent` / `Workflow.coordinator`). Already native.
  • *explicit/coordinator inside emergent* — a sub-workflow or coordinator Agent **as a tool**
    an `Agent` can call. That's `as_tool(...)`: wrap anything with `async run(task, ctx) ->
    result` into a `FunctionTool` whose result is rendered back to text and fed into the loop.

One adapter each way, no bridging runtime — nesting is just "everything is callable", and the budget /
cancellation / observation spine flows through the `ctx.child()` it runs on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolSchema
from agentkit.tools.function import FunctionTool


def _task_schema(name: str, description: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string", "description": "the sub-task / goal to run"}},
            "required": ["task"],
        },
    )


def render_result(res: Any) -> str:
    """Render any run result to text for re-entry into a loop: `AgentResult.output` or a
    `WorkflowResult`'s terminal outputs."""
    out = getattr(res, "output", None)
    if isinstance(out, str):  # AgentResult
        return out
    outputs = getattr(res, "outputs", None)
    if isinstance(outputs, dict):  # WorkflowResult
        return "\n".join(f"{k}: {v}" for k, v in outputs.items())
    return str(res)


def as_tool(
    runnable: Any,
    *,
    name: str,
    description: str = "",
    side_effecting: bool = False,
    requires_approval: bool = False,
    render: Callable[[Any], str] | None = None,
) -> FunctionTool:
    """Wrap any runnable (leaf `Agent` / coordinator `Agent` / `Workflow` — anything with
    `async run(task, ctx) -> result`) as a `FunctionTool` an `Agent` can call. The sub-run executes on
    a `ctx.child()` (so depth/budget/cancellation/observation all flow), and its result is rendered to text
    for the loop. This is the *explicit/coordinator-inside-emergent* composition."""
    _render = render or render_result

    async def _fn(args: Any, ctx: Ctx) -> str:
        # ``args`` is ``Any`` because two different callers reach this seam: the
        # tool chain hands down ``ToolRequest.arguments``, and a workflow node can
        # hand a bare task string straight to ``.run``.
        #
        # The abstract ``Mapping`` test stays, but NOT for the reason it used to
        # give. What comes down from a ``ToolCall`` is normally a ``FrozenDict``,
        # and that IS a ``dict`` subclass (measured: the args arrive at a
        # ``Tool.run`` as ``FrozenDict``, ``isinstance(_, dict)`` True), so a
        # narrow ``dict`` test would pass today where it once silently failed.
        # It would pass for a proxy too: ``deep_freeze`` now NORMALISES a
        # ``MappingProxyType`` into a ``FrozenDict`` — measured,
        # ``ToolCall("c", "s", MappingProxyType({...})).arguments`` is a
        # ``FrozenDict``, nested proxies included — so the "stored VERBATIM"
        # justification this comment used to give is dead.
        #
        # ``Mapping`` is kept because this ``_fn`` is ``FunctionTool.fn``, and
        # ``run(args, ctx)`` hands ``args`` down untouched: nothing on this path
        # freezes anything unless the call came through a ``ToolRequest``. Every
        # caller that skips that seam supplies its mapping raw —
        # ``test_as_tool_forwards_task_from_mappingproxy_arguments`` calls
        # ``tool.fn(MappingProxyType({"task": ...}), ctx)`` directly, where
        # ``isinstance(args, dict)`` is measurably False. ``deep_freeze`` also
        # returns every non-proxy ``Mapping`` by identity (a ``ChainMap``, a
        # project's own type — reconstructing those is the line it refuses to
        # cross), so a ``dict``-only test would drop ``task`` for them even on
        # the ``ToolRequest`` path.
        #
        # The branch that earns its keep is the ELSE: a non-mapping falls to
        # ``str(args)``, which is how a bare-string task survives. A mapping
        # falling there instead would hand the sub-runnable the dict REPR as its
        # task, which is the failure this shape exists to prevent.
        task = args.get("task", "") if isinstance(args, Mapping) else str(args)
        return _render(await runnable.run(task, ctx.child()))

    return FunctionTool(
        name=name,
        fn=_fn,
        description=description or f"run sub-task on {name}",
        schema=_task_schema(name, description),
        side_effecting=side_effecting,
        requires_approval=requires_approval,
    )


__all__ = ["as_tool", "render_result"]
