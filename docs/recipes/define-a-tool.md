# How do I give an agent a tool?

Tools are how an agent does something other than talk. Without one it
can only produce text; with one it can look a number up, send the
message, or change the record.

## When you'd want this

Any time the honest answer to a question is "I'd have to check". A
model asked for an order status with no way to check invents a
plausible one — the reply is fluent, correctly formatted, and wrong,
and nothing downstream can tell. A tool converts "make something up"
into "call this function and report what it said".

The other half is control. Once an agent can act, you need to say which
actions are safe to repeat, which mutate the world, and which need a
person to approve them. agentkit makes you declare that at the point
you define the tool, not at the point it fires.

## Working code

```python
"""Runs offline: FakeLLM scripts the two turns, so no API key is needed."""

import asyncio

from agentkit import Agent, ToolArgumentError, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import ToolCall
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False, idempotent=True)
async def lookup_order(order_id: str) -> str:
    """Look up an order by its id and return its status line.

    Use this whenever the user mentions an order number.
    """
    return f"order {order_id}: shipped 2026-08-14, arriving Tuesday"


@tool(side_effecting=True)
def cancel_order(order_id: str, reason: str = "customer request") -> str:
    """Cancel an order. This really cancels it — there is no undo.

    Only call it after the user has confirmed the order id.
    """
    return f"order {order_id} cancelled ({reason})"


async def main() -> None:
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="c1", name="lookup_order", arguments={"order_id": "A-771"}),)),
        Turn(content="Order A-771 shipped on 14 August and arrives Tuesday."),
    ])
    ctx = make_test_ctx(llm=llm)
    agent = Agent(
        name="support",
        model="claude-sonnet-4-6",
        prompt="Answer order questions. Use the tools; never guess a status.",
        cognition=ReActCognition(tools=[lookup_order, cancel_order]),
    )
    print((await agent.run("What happened to order A-771?", ctx)).output)

    # This is exactly what the model is shown for the tool:
    print(lookup_order.schema.name, lookup_order.schema.parameters)

    # A call the model got wrong is rejected BY NAME, not silently dropped.
    try:
        await cancel_order.run({"order": "A-771"}, ctx)
    except ToolArgumentError as exc:
        print(f"rejected: {exc}")


asyncio.run(main())
```

Output:

```text
Order A-771 shipped on 14 August and arrives Tuesday.
lookup_order {'type': 'object', 'properties': {'order_id': {'type': 'string'}}, 'required': ['order_id']}
rejected: tool 'cancel_order' call rejected: unexpected argument(s) ['order']; missing
required argument(s) ['order_id']. Accepted arguments: ['order_id', 'reason']
```

## How it works

`@tool` inspects the function and builds a `FunctionTool`:

- the **signature and type hints** become the JSON Schema the model is
  shown (`tool.schema.parameters`);
- the **docstring** becomes the tool's description — which is the
  entire spec the model gets, so it is not decoration;
- the **flags** you pass become the metadata the framework's gating,
  retry, and audit machinery reads.

At run time the model emits a `ToolCall`; `ReActCognition` turns it into
a `ToolRequest`, hands it to `ctx.invoker.invoke_tool(...)` — through
your tool middleware chain — and appends the result to the transcript
as a `tool` message. Then the loop goes round again.

A sync function is run on a worker thread (`asyncio.to_thread`), so a
blocking call inside a tool never stalls the event loop. An async
function is awaited directly.

## Declaring what a tool is

Three flags, and they are independent of one another:

| Flag | Means | Who reads it |
|---|---|---|
| `side_effecting` | mutates the world; there is no undo | the approval gate (`Autonomy`), the `idempotent()` middleware |
| `idempotent` | safe to call twice with the same args | retry middleware |
| `requires_approval` | always ask a human, whatever the autonomy tier | the approval gate |

A read-only lookup is `side_effecting=False, idempotent=True`. A
payment is `side_effecting=True, idempotent=False`. There is no default
for `side_effecting` — see the gotchas.

Two more, for the security surfaces: `caps=("private_data",
"untrusted_content", "egress")` feeds the lethal-trifecta `RunPolicy`
gate, and `url_arg="url"` tells the `egress()` middleware which argument
to check against your allowlist.

## Getting the context inside a tool

Name a parameter `ctx` (or `context`) and the `RunContext` is injected
into it. It is not advertised to the model, so it never appears in the
schema and the model cannot pass one:

```python
from agentkit import tool
from agentkit.runtime import RunContext


@tool(side_effecting=False)
async def whoami(ctx: RunContext) -> str:
    """Report which run and which tenant this call belongs to.

    The ctx parameter is injected by the framework, not by the model."""
    return f"run={ctx.correlation_id} org={ctx.scope.org_id!r}"
```

That gives a tool the tenant scope, the cancellation token, the budget,
and the trace — which is how a tool stays multi-tenant-safe without
every call site threading state through by hand.

## Validating what a tool returns

Annotate the return type with a Pydantic model, dataclass, or attrs
class and the result is validated on every call:

```python
from pydantic import BaseModel

from agentkit import ToolShapeError, tool


class Quote(BaseModel):
    symbol: str
    price: float


@tool(side_effecting=False)
async def quote(symbol: str) -> Quote:
    """Return the last traded price for a ticker symbol.

    The return type is checked on every call."""
    return Quote(symbol=symbol, price=101.5)
```

A mismatch raises `ToolShapeError` and drops a `tool.shape_mismatch`
event on the open `execute_tool` span. Because it is raised rather than
swallowed, the retry middleware can hand the structured error back to
the model, which usually re-issues the call or pivots.

Inference is automatic but overridable: `output_schema=SomeType` forces
a shape, and `output_schema=None` opts out entirely even when the
return annotation looks enforceable. Primitives (`str`, `int`, `dict`)
are never checked — nothing to check against.

## Collecting tools: `ToolRegistry`

`ReActCognition(tools=[...])` takes a plain list. When you want one
named lookup surface — a plug-in system, tools assembled from several
modules — use the registry:

```python
from agentkit import ToolRegistry, tool


@tool(side_effecting=False, idempotent=True)
async def lookup_order(order_id: str) -> str:
    """Look up an order by its id and return its status line."""
    return f"order {order_id}: shipped"


@tool(side_effecting=True)
async def cancel_order(order_id: str) -> str:
    """Cancel an order. This really cancels it — there is no undo."""
    return f"order {order_id} cancelled"


registry = ToolRegistry.from_tools([lookup_order, cancel_order])
print(registry.names())          # ['cancel_order', 'lookup_order'] — sorted, stable
print(len(registry.schemas()))   # what goes on the wire, in that same stable order
print(registry.get("lookup_order").name)
```

The sort is load-bearing: the advertised tool list is part of the
cache-stable prefix, so a stable order keeps the provider's prompt cache
warm across turns.

Registering a name twice raises rather than overwriting. A silent
overwrite would leave the model calling the same advertised name while
a different implementation answers — pass `replace=True` when you mean
it.

## An agent as a tool

`as_tool(...)` wraps anything with `async run(task, ctx)` — a leaf
`Agent`, a coordinator, a `Workflow` — into a `FunctionTool`, so a
delegating agent is just an agent with one more tool:

```python
from agentkit import Agent, as_tool

researcher = Agent(name="researcher", model="claude-sonnet-4-6", prompt="Research briefly.")
research = as_tool(
    researcher,
    name="research",
    description="Delegate a research sub-task to the researcher agent.",
)
research.schema.parameters   # {'type': 'object', 'properties': {'task': {...}}, 'required': ['task']}
```

The sub-run executes on a `ctx.child()`, so budget, cancellation, depth,
and observation all flow through it, and the sub-result is rendered back
to text for the calling loop by `render_result` (override with
`render=`).

## Gotchas

- **`side_effecting=` is required, and the failure is at decoration
  time.** `@tool` with no `side_effecting=` raises `ToolDefinitionError`
  the moment the module is imported. The gating and idempotency
  primitives cannot guess, and guessing wrong in either direction is
  bad: guess "no" and a payment runs unapproved, guess "yes" and every
  read-only lookup stops for a human.
- **A thin docstring is refused, also at decoration time.** Under 30
  characters raises `ToolDefinitionError`. The docstring IS the spec the
  model works from; a tool called `get` with the description `"gets"`
  will be called at the wrong times and nothing will explain why.
- **`ctx: Ctx` does not get injected — `ctx: RunContext` does.** The
  injection fires when the parameter is named `ctx`/`context` AND is
  either unannotated or annotated with `RunContext`. Annotate it with
  the `Ctx` Protocol and it stays an ordinary advertised parameter: it
  shows up in the schema as a string, and your call fails with
  `ToolArgumentError: missing required argument(s) ['ctx']`. Use
  `RunContext` or leave it bare.
- **An argument the model invented is an error, not a no-op.** Extra
  keys used to be dropped, which let a defaulted parameter run with its
  default — a side-effecting tool reporting success for something it was
  never asked to do. They now raise `ToolArgumentError` naming both the
  unexpected and the missing keys. Declare `**kwargs` if you genuinely
  want to accept keys you did not enumerate.
- **Arguments arrive frozen.** `ToolCall.arguments` is a `FrozenDict` —
  a `dict` subclass that reads normally but refuses mutation, nested
  values included. So `args["limit"] = 100` inside a tool raises. The
  point is that the arguments a human approved stay the arguments that
  execute. Build a new dict if you need to adjust something.
- **The tool chain is separate from the chat chain.**
  `Invoker(tool_middleware=[...])` wraps tool execution;
  `chat_middleware=[...]` wraps model calls. Retry on the tool chain
  recovers from a flaky API you're calling; retry on the chat chain
  recovers from a provider blip. They are different failures.

## Related

- [Human-in-the-loop tool approval](hitl-tool-approval.md) — making a
  `side_effecting=True` tool stop for a person.
- [Consume MCP tools from an agent](mcp-tools.md) — tools somebody else
  already wrote.
- [Split work across several agents](multi-agent-coordination.md) —
  where `as_tool` fits among the other composition shapes.
- [Test an agentkit app](test-an-agentkit-app.md) — asserting on what
  arguments a tool actually received.
