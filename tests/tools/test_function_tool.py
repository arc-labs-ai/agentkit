"""Function-as-tool (FunctionTool.from_callable / @tool): a plain Python function becomes a tool —
signature + type-hint inspection → JSON-schema ToolSchema, sync/async execution wrapper, ctx injection,
and Agent(tools=[fn]) auto-registration. Offline & deterministic."""

import asyncio

import pytest

from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import Scope, ToolCall
from agentkit.runtime import Budget, Invoker, RunContext
from agentkit.runtime.context import Services
from agentkit.testing import FakeLLM, Turn
from agentkit.tools import FunctionTool, ToolRegistry, tool


def _run(coro):
    return asyncio.run(coro)


def get_weather(location: str, days: int = 1) -> str:
    """Get weather forecast for a location."""
    return f"Weather in {location}: sunny ({days}d)"


# ---- inspection + schema generation -----------------------------------------------------------


def test_schema_is_generated_from_signature_and_hints():
    t = FunctionTool.from_callable(get_weather)
    assert t.name == "get_weather"
    assert t.schema.description == "Get weather forecast for a location."
    props = t.schema.parameters["properties"]
    assert props["location"] == {"type": "string"}
    assert props["days"] == {"type": "integer"}
    assert t.schema.parameters["required"] == ["location"]  # `days` has a default → optional


def test_type_mapping_covers_common_types():
    def f(a: str, b: int, c: float, d: bool, e: list, g: dict, h: str | None = None):
        """Exercise the common-types path of the JSON-schema inference."""
        return a

    props = FunctionTool.from_callable(f).schema.parameters["properties"]
    assert [props[k]["type"] for k in ("a", "b", "c", "d", "e", "g", "h")] == [
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
        "string",
    ]


def test_literal_becomes_enum_and_multi_union_becomes_anyof():
    """Regression: Literal → JSON-schema enum (not a lossy 'string'); a genuine multi-type union →
    anyOf (not an order-dependent single member); Optional[X] still collapses to X."""
    from typing import Literal

    def f(
        unit: Literal["c", "f"],
        n: Literal[1, 2, 3],
        mixed: int | str,
        opt: int | None = None,
        u: bool | str = "x",
    ):
        """Exercise Literal/Union/Optional schema inference paths."""
        return unit

    props = FunctionTool.from_callable(f).schema.parameters["properties"]
    assert props["unit"] == {
        "enum": ["c", "f"],
        "type": "string",
    }  # homogeneous str literals → typed enum
    assert props["n"] == {
        "enum": [1, 2, 3],
        "type": "integer",
    }  # homogeneous int literals → typed enum
    assert props["mixed"] == {
        "anyOf": [{"type": "integer"}, {"type": "string"}]
    }  # true union → anyOf
    assert props["opt"] == {"type": "integer"}  # Optional[int] → int (None dropped)
    assert props["u"] == {"anyOf": [{"type": "boolean"}, {"type": "string"}]}


# ---- execution wrapper: sync, async, ctx injection --------------------------------------------


def test_sync_function_executes_with_named_args():
    t = FunctionTool.from_callable(get_weather)
    out = _run(t.run({"location": "Paris", "days": 2}, ctx=None))
    assert out == "Weather in Paris: sunny (2d)"


def test_async_function_is_awaited():
    async def fetch(url: str) -> str:
        """Fetch a URL and return a fake body string (test stub)."""
        return f"got {url}"

    out = _run(FunctionTool.from_callable(fetch).run({"url": "x"}, ctx=None))
    assert out == "got x"


def test_ctx_param_is_injected_and_not_advertised():
    def whoami(prefix: str, ctx) -> str:
        """Return a prefixed identity string from the injected RunContext."""
        return f"{prefix}:{ctx.correlation_id}"

    t = FunctionTool.from_callable(whoami)
    assert "ctx" not in t.schema.parameters["properties"]  # ctx is injected, not a model arg
    fake_ctx = RunContext("run-9", Scope(1, 2), Budget(), Services())
    assert _run(t.run({"prefix": "id"}, fake_ctx)) == "id:run-9"


def test_annotated_context_param_is_NOT_hijacked():
    # a real data param named `context: str` must stay an advertised model arg, not the injected RunContext
    def summarize(context: str) -> str:
        """Return a fake summary of the provided context string."""
        return f"summary of {context}"

    t = FunctionTool.from_callable(summarize)
    assert t.schema.parameters["properties"]["context"] == {"type": "string"}
    assert t.schema.parameters["required"] == ["context"]
    assert _run(t.run({"context": "the doc"}, ctx=object())) == "summary of the doc"


# ---- the decorator + registry auto-conversion -------------------------------------------------


def test_tool_decorator_and_registry_autoconvert():
    @tool(side_effecting=True)
    def delete(path: str) -> str:
        """Delete the file at `path`; mutates the filesystem (destructive)."""
        return f"deleted {path}"

    assert isinstance(delete, FunctionTool) and delete.side_effecting

    reg = ToolRegistry.from_tools([get_weather])  # plain function auto-wrapped
    assert reg.names() == ["get_weather"] and reg.get("get_weather").schema is not None


# ---- Agent(tools=[fn]) end-to-end -------------------------------------------------------------


def test_agent_accepts_a_list_of_plain_functions():
    calls = {"n": 0}

    def lookup(q: str) -> str:
        """Look up an answer to query `q` (read-only stub for tests)."""
        calls["n"] += 1
        return f"answer for {q}"

    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "lookup", {"q": "redis"}),)),
            Turn(content="done"),
        ]
    )
    inv = Invoker(llm=llm)
    ctx = RunContext("r", Scope(1, 2), Budget(max_cost_usd=10.0), Services(invoker=inv))

    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=[lookup]))
    res = _run(loop.run("q", ctx))
    assert res.output == "done" and calls["n"] == 1  # the function ran as a tool


# ── ToolRegistry name-collision behavior ─────────────────────────────────────
#
# ``register()`` raises ``ValueError`` on name collision unless
# ``replace=True`` is set — a typo or import-order accident must not
# silently swap the active tool under the agent.


def _collision_tool(name: str, fn):
    return FunctionTool(
        name, fn, description="dummy tool for collision tests", side_effecting=False
    )


def test_registry_register_raises_on_name_collision():
    """Two registers of the same name → second raises, so a silent
    overwrite cannot quietly return."""
    reg = ToolRegistry()
    reg.register(_collision_tool("alpha", lambda a, c: "first"))
    with pytest.raises(ValueError, match=r"alpha.*already registered"):
        reg.register(_collision_tool("alpha", lambda a, c: "second"))


def test_registry_register_replace_true_swaps_deliberately():
    """The opt-in path: ``replace=True`` makes the swap explicit and
    succeeds. Verifies the new impl is the one returned by ``get``."""
    reg = ToolRegistry()
    reg.register(_collision_tool("alpha", lambda a, c: "first"))
    new = _collision_tool("alpha", lambda a, c: "second")
    returned = reg.register(new, replace=True)
    assert returned is new
    assert reg.get("alpha") is new


def test_registry_from_tools_propagates_collision_error():
    """``from_tools([t1, t2])`` with t1.name == t2.name uses the same
    collision guard — the constructor doesn't quietly take the last
    duplicate."""
    with pytest.raises(ValueError, match=r"alpha.*already registered"):
        ToolRegistry.from_tools(
            [
                _collision_tool("alpha", lambda a, c: "first"),
                _collision_tool("alpha", lambda a, c: "second"),
            ]
        )


# ── MappingProxyType flows correctly through tool arg handling ───────────────
#
# ``ToolCall.arguments`` is exposed as a ``MappingProxyType`` (read-only
# view). Every downstream site that interprets tool arguments must accept
# ``Mapping`` — not just ``dict`` — otherwise the args silently fall
# through to a default branch and the tool receives no arguments.


def test_as_tool_forwards_task_from_mappingproxy_arguments():
    """``as_tool`` extracts the ``task`` field from a ``MappingProxyType``
    arg dict, not just a plain ``dict``. Matching only on ``dict``
    (via ``isinstance``) would drop the ``task`` key from a ``ToolCall``
    and hand the runnable the dict's ``str`` repr instead."""
    import asyncio
    from types import MappingProxyType

    from agentkit.testing import make_test_ctx
    from agentkit.tools.from_agent import as_tool

    seen_tasks: list[str] = []

    class _EchoRunnable:
        async def run(self, task, ctx):
            seen_tasks.append(task)
            from agentkit.agents.result import AgentResult
            from agentkit.kernel.types import Usage

            return AgentResult(output=task, usage=Usage())

    tool = as_tool(_EchoRunnable(), name="echoer", description="echoes the task text")
    args = MappingProxyType({"task": "summarise this doc"})

    ctx = make_test_ctx()
    result = asyncio.run(tool.fn(args, ctx))
    assert result == "summarise this doc"
    assert seen_tasks == ["summarise this doc"]
