# Tools

Tools are how an agent does something other than talk.

!!! tip "Is this page for you?"

    **Reach for it when** you want the agent to *do* something
    rather than only produce text.

    **Skip it for now if** you are still deciding what the agent
    should say, not what it should touch.

## The problem it solves

A model on its own can only produce text. It cannot read your database,
call your billing API, or find out what today's price is. Left with only
its training data it will answer anyway — confidently, and sometimes
wrongly. A tool is the seam where the model stops guessing and your code
takes over: the model says *which* action it wants and with *what*
arguments, your function does the work, and the real answer goes back
into the conversation.

Two things have to be true for that to be safe. The model has to
understand what a tool does well enough to pick the right one — so the
description and the argument schema are not documentation, they are the
interface. And the framework has to know which tools change the world —
so it can pause a refund for a human, dedupe a retried email, and refuse
to serve a cached "success" for something that never happened.

agentkit makes both of those *declarations*, checked when you define the
tool rather than guessed at when it runs.

## The smallest thing that works

```python
import asyncio

from agentkit.kernel.types import ToolRequest
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import ToolRegistry, tool


@tool(side_effecting=False, idempotent=True)
def city_population(city: str) -> int:
    """Look up the most recent census population for a named city."""
    return {"lisbon": 545_000, "porto": 231_000}.get(city.lower(), 0)


registry = ToolRegistry.from_tools([city_population])


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM())
    the_tool = registry.get("city_population")
    result = await ctx.invoker.invoke_tool(
        ToolRequest(name="city_population", arguments={"city": "Lisbon"}, tool=the_tool),
        ctx,
    )
    print(result)
    print(the_tool.schema.parameters)


asyncio.run(main())
```

```text
545000
{'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
```

That is the whole shape: a plain Python function, one decorator, and a
JSON Schema the model can call. Nothing about `city_population` had to
be written twice.

Note the call path. Tools are not invoked directly by the loop — they go
through `ctx.invoker.invoke_tool`, which runs the
[tool-middleware chain](middlewares.md) (tracing, metering, egress
checks, idempotency, audit, retry) and only then calls `tool.run`. That
is why the flags you declare below actually do something.

And here is the same tool inside an agent, driven by a scripted fake
model so it runs with no API key:

```python
import asyncio

from agentkit import Agent, ToolCall
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False, idempotent=True)
def city_population(city: str) -> int:
    """Look up the most recent census population for a named city."""
    return {"lisbon": 545_000, "porto": 231_000}.get(city.lower(), 0)


llm = FakeLLM.script([
    Turn(tool_calls=(ToolCall("c1", "city_population", {"city": "Lisbon"}),)),
    Turn(content="Lisbon has about 545,000 residents."),
])

agent = Agent(
    name="atlas",
    model="fake",
    prompt="You answer questions about cities.",
    cognition=ReActCognition(tools=[city_population]),
)


async def main() -> None:
    ctx = make_test_ctx(llm=llm)
    result = await agent.run("How big is Lisbon?", ctx)
    print(result.output)


asyncio.run(main())
```

```text
Lisbon has about 545,000 residents.
```

Tools live on the **cognition**, not on the `Agent`. `ReActCognition`
accepts a plain list and wraps it in a `ToolRegistry` for you.

## How it works

### The schema comes from the signature

The model cannot see your Python. Before it can ask for a tool, it has
to be *told* the tool exists — its name, what it does, and what
arguments it takes. That description is the **schema**, and it is sent
along with every request.

Writing it by hand would mean maintaining the same information twice,
and the two copies would drift the first time someone renamed a
parameter. So agentkit reads your function signature and writes the
schema for you: `FunctionTool.from_callable` (which `@tool` calls on
your behalf) inspects the parameters and type hints and produces the
`ToolSchema` the model sees.

In practice this means **the annotations you write are a promise to the
model**, and getting one wrong misleads it rather than failing loudly —
which is why the mapping below is worth a glance.

| Python annotation | JSON Schema fragment |
|---|---|
| `str` / `int` / `float` / `bool` | `{"type": "string" \| "integer" \| "number" \| "boolean"}` |
| `list[X]`, `set[X]`, `tuple[X, ...]` | `{"type": "array", "items": <X>}` |
| `Sequence[X]`, `Iterable[X]`, `Collection[X]`, `Set[X]` | `{"type": "array", "items": <X>}` |
| bare `list` / `tuple`, or `tuple[X, Y]` | `{"type": "array"}` — no single element type to name |
| `dict[str, X]`, `Mapping[str, X]` | `{"type": "object", "additionalProperties": <X>}` |
| a `TypedDict` | its real object schema, keys and all |
| `Literal["a", "b"]` | `{"enum": ["a", "b"], "type": "string"}` |
| an `Enum` subclass | `{"enum": [...member values...]}`, typed when homogeneous |
| a Pydantic model / dataclass / attrs class | its real object schema, via the same `adapt()` the `Agent.output=` path uses |
| `Optional[X]` / `X \| None` | `{"anyOf": [<X>, {"type": "null"}]}` |
| a genuine multi-type union | `{"anyOf": [...]}` |
| anything else, or unannotated | `{"type": "string"}` |

`items` is a promise in both directions. Because the model is told the
element type, the value it sends is reconstituted to match — a
`list[Unit]` arrives as a list of `Unit` members, not of strings — and
the container is rebuilt to whatever the annotation asked for, so a
`set[X]` parameter receives a `set`.

`*args` is refused at decoration time. A tool is called with a JSON
object, so every argument arrives by keyword; a positional-only
parameter can never be filled. Use `**kwargs` if the tool genuinely
accepts arbitrary keys.

A parameter with no default lands in `required`. One with a default does
not.

The two rich cases are there because their absence actively misled the
model. A structured parameter advertised as a bare `{"type": "string"}`
made the model send a string where the function's annotation promised a
`Filter` — the schema was instructing the model to call the tool wrongly.
An `Enum` advertised as `{"type": "string"}` told the model nothing about
which strings were legal, so it invented one and the tool raised
`ValueError` on a value the schema had implied was fine.

```python
import json
from enum import Enum
from typing import Literal

from agentkit.tools import tool


class Unit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


@tool(side_effecting=False, idempotent=True)
def forecast(
    city: str,
    unit: Unit,
    days: int = 3,
    mode: Literal["hourly", "daily"] = "daily",
) -> str:
    """Return a short weather forecast for a city over the next few days."""
    return f"{city}: mild for {days} days ({unit}, {mode})"


print(json.dumps(forecast.schema.parameters))
```

```text
{"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum":
["celsius", "fahrenheit"], "type": "string"}, "days": {"type": "integer"},
"mode": {"enum": ["hourly", "daily"], "type": "string"}}, "required":
["city", "unit"]}
```

### The docstring is the description — all of it

Plain version first: **everything you write in a tool's docstring is
sent to the model.** Not the summary line, not the part above the first
blank line — the whole text, verbatim, as part of the schema attached to
every request. This is the same family of check as `*args` above: both
are settled when you *define* the tool, not when it runs.

That is worth stating outright because it used to be false, and the old
rule was this framework's most expensive silent decision. The
description used to be the docstring's **first line**. Everything below
it was read by people editing the source and by nobody else, while still
sitting in the docstring looking as though it had been delivered.

It cost a real defect. agentkit's own `ask_human` tells the model *"Use
this only when the information genuinely cannot be obtained any other
way — a one-time code, a confirmation, a preference only they know."*
That sentence is the entire difference between a tool the model reaches
for when it is genuinely stuck and one it reaches for instead of
thinking. It sits in the second paragraph. It never shipped.

The rule also cut ORDINARY docstrings mid-clause, because a single
paragraph wrapped over two source lines is two lines, not one. Measured,
two tools in this repo's own test suite were advertised to the model as:

```text
"A tool, so the ReAct loop has something to run and a reason to"
"Look up a Hit for query `q`. (Will return a malformed payload to"
```

Neither author wrote a bad docstring. The framework delivered half of a
good one and said nothing.

#### Why the fix wasn't "refuse the second paragraph"

The tidy alternative was considered and rejected on evidence. `@tool`
already refuses a missing `side_effecting=` and a thin docstring, so the
consistent move would have been to refuse a multi-paragraph docstring
too, and make the author say out loud which audience the tail was for.

The measurement killed it. Across `agentkit/`, `tests/`, `examples/` and
`docs/` there are **139 tools**. **Six** have a multi-paragraph
docstring, and in **all six** the second paragraph is model-facing
guidance — *"Use this whenever the user mentions an order number."* —
not an implementation note. Refusing them would have rejected six tools
whose authors had already written exactly the right thing, forced
`ask_human` to delete the sentence this change exists to deliver, and
still left the two truncations above in place, since those docstrings
are single paragraphs.

Taking everything costs **+7% description bytes repo-wide** (8307 → 8892
characters across all 139 tools), and **zero** of the 139 change verdict
against the 30-character floor. That is the whole trade, in numbers, and
it is why the long form is the default rather than a flag.

```python
from agentkit.tools import tool


@tool(side_effecting=False, idempotent=True)
def order_status(order_id: str) -> str:
    """Look up the current shipping status of a customer order.

    Use this whenever the user mentions an order number, even in
    passing. The status is the carrier's own wording, and it can be up
    to an hour stale.
    """
    return "in transit"


print(order_status.description)
```

```text
Look up the current shipping status of a customer order.

Use this whenever the user mentions an order number, even in
passing. The status is the carrier's own wording, and it can be up
to an hour stale.
```

#### The cost, stated plainly

The whole docstring reaching the model means the notes you wrote for a
human reach it too. `TODO: this is O(n^2)`, `see ticket #442`, `the
vendor lies about 404s` — none of that was visible before, and all of it
is now text inside the tool's instructions, competing for attention with
the sentence that actually tells the model when to call.

There is no length cap. Nothing in `agentkit/tools/schema.py` trims a
long description or warns about one; what you write is what ships, on
every request, for the whole run. Two escape hatches exist instead, and
they are deliberately different sizes.

**A `---` line ends the model-facing part.** A docstring line containing
nothing but three hyphens is the cut: everything above it is the
description, everything below it is for whoever is editing the source.

```python
from agentkit.tools import tool


@tool(side_effecting=False, idempotent=True)
def order_status(order_id: str) -> str:
    """Look up the current shipping status of a customer order.

    Use this whenever the user mentions an order number.

    ---

    Backed by the carrier's v2 endpoint, which lies about 404s.
    See ticket #442 before touching the retry count.
    """
    return "in transit"


print(order_status.description)
```

```text
Look up the current shipping status of a customer order.

Use this whenever the user mentions an order number.
```

That is an opt-**out**, and the direction was the decision. An opt-in —
`long_description=True`, say — would have made all 139 existing tools do
paperwork to keep prose their authors already meant for the model; the
opt-out is one line for the few who need it, and its absence is the
common case. It was also safe to switch on for a measured reason: across
those 139 tools, not one already contained a bare `---` line, so turning
it on changed no existing description by a single byte.

**A developer note above the cut is refused at decoration time.** This
is the other half of the whole-docstring decision. `TODO`, `FIXME`,
`XXX` and `HACK` used to be invisible to the model; the moment they
became shippable text they had to be either shipped deliberately or
moved. The framework cannot tell which the author meant, so it asks —
the same call it already makes for a missing `side_effecting=`.

```python
from agentkit.tools import ToolDefinitionError, tool

try:
    @tool(side_effecting=False)
    def catalogue_search(q: str) -> str:
        """Search the product catalogue for a free-text query.

        Notes on behaviour:
          - results are ranked by the vendor, not by us
          - TODO: this is O(n^2), rewrite before the Q3 launch
        """
        return "ok"
except ToolDefinitionError as exc:
    print(exc)
```

```text
tool 'catalogue_search' has a developer note (TODO) in the part of its docstring
that is sent to the model, where it reads as an instruction; move it below a
line containing only '---' (everything after that line is for humans) or pass an
explicit description=
```

Note *where* that note was written, because the anchoring is the part
that took a second attempt. The pattern matches at the **start of a
line**, allowing for indentation and an optional list bullet. Column
zero alone was not enough, and the gap was not hypothetical:
`inspect.getdoc` dedents by the *common* leading whitespace, so a note
written one level in keeps its extra indent all the way through — and
the most ordinary way to write one is as a bullet under a heading,
exactly as above. Both the indent and the `- ` would have stood between
`TODO` and a column-zero anchor, the check would have stayed silent, and
the note would have shipped as an instruction.

What is deliberately *not* a note is the marker used as a word inside a
sentence. A to-do-list tool may say "Append a TODO: item to the
checklist" — there the marker is preceded by prose, and the pattern
admits only whitespace and a bullet before it. A word boundary keeps
"XXXL is a size" out of it as well.

The check runs on the model-facing half only. A `TODO` *below* the `---`
line is exactly what the marker is for, and is left alone.

#### The 30-character floor, and what it now measures

`@tool` still refuses a description shorter than **30 characters**
(`_MIN_DESCRIPTION_LEN` in `agentkit/tools/schema.py`). What changed is
the thing being measured: the floor applies to the **whole chosen
text**, not to line one. The total is what the model is shown, so the
total is what has to clear the bar.

```python
from agentkit.tools import ToolDefinitionError, tool

try:
    @tool(side_effecting=False)
    def lookup(q: str) -> str:
        """Look it up."""
        return "ok"
except ToolDefinitionError as exc:
    print(exc)
```

```text
tool 'lookup' needs a docstring/description of at least 30 characters so the
model can understand it; got 11 chars: 'Look it up.'
```

Thirty characters is not a style rule. The description **is** the spec:
it is what the model reads when deciding between your twelve tools, and
`"Look it up."` does not distinguish a vector search from a DNS lookup.
A tool the model cannot tell apart from its neighbour gets picked at
random, and the failure shows up as a bad answer several turns later
rather than as an error. The floor makes that a `ToolDefinitionError` at
*decoration* time — at import, before a run starts, before any money is
spent.

Measuring the total only ever **widens** what passes: a docstring with a
thin first line and a substantial body used to be rejected and now is
not. Measured, 0 of this repo's 139 tools change verdict.

It narrows in exactly one place, and the error names that place rather
than leaving you to guess. An author who put everything below the `---`
line has a docstring that looks generous and a description that is
nearly empty:

```python
from agentkit.tools import ToolDefinitionError, tool

try:
    @tool(side_effecting=False)
    def order_status(order_id: str) -> str:
        """Look it up.

        ---

        Look up the current shipping status of a customer order. Use
        this whenever the user mentions an order number.
        """
        return "in transit"
except ToolDefinitionError as exc:
    print(exc)
```

```text
tool 'order_status' needs a docstring/description of at least 30 characters so
the model can understand it; got 11 chars: 'Look it up.' (everything below the
'---' line in its docstring is for humans and was not counted)
```

#### `description=` still wins outright

An explicit `description=` replaces the docstring entirely, and it is
passed through with nothing but a `strip()` — no marker hunt, no
developer-note check. The asymmetry is the point. A docstring is text
the framework *interprets*, addressed to two audiences at once, and
every rule above is about splitting it. `description=` is text you
handed the framework directly and meant every byte of; going looking for
markers in it would be the framework second-guessing an explicit
instruction. It still has to clear the 30-character floor.

```python
from agentkit.tools import tool


@tool(
    side_effecting=False,
    description="Search the product catalogue and return matching SKUs.",
)
def catalogue_search(q: str) -> str:
    """Search.

    TODO: this is O(n^2), rewrite before the Q3 launch.
    """
    return "ok"


print(catalogue_search.description)
```

```text
Search the product catalogue and return matching SKUs.
```

A thin docstring and a developer note, and neither is an error — because
neither is what the model was given.

!!! note "Docstrings written on Windows"
    `inspect.getdoc` dedents by finding common leading whitespace, and a
    CRLF docstring defeats it completely — measured, the four-space
    indent survives the dedent and a trailing carriage return rides
    along. Under the first-line rule that never showed, because the
    split threw the tail away. Now the whole text ships, so `schema.py`
    normalises the line endings and re-runs the dedent the `\r` defeated
    before anything reaches a provider.

### `side_effecting=` is required, and three subsystems read it

```python
from agentkit.tools import ToolDefinitionError, tool

try:
    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email to a single recipient and return the message id."""
        return "ok"
except ToolDefinitionError as exc:
    print(exc)
```

```text
@tool requires an explicit `side_effecting=` keyword (True if the tool mutates
the world / has external effects, False if it's read-only). The framework's
approval gating and idempotency depend on knowing upfront.
```

There is no default because there is no safe default. Guessing "read-only"
lets a mutation be cached; guessing "mutating" turns caching off for
everyone. Three real mechanisms consult the flag:

**1. Approval gating.** `ReActCognition` copies `side_effecting` onto the
tool request as `key_step`, and
[`should_gate`](https://github.com/arc-labs-ai/agentkit/blob/main/agentkit/agents/control/gate.py)
combines it with the run's autonomy tier: under `MANUAL` everything
pauses, under `GATED` a side-effecting step pauses too, under `AUTO`
(the default) only a tool that declared `requires_approval=True` pauses.
An author's `requires_approval=True` is honoured at every tier —
autonomy only *adds* gating.

```python
import asyncio

from agentkit import Agent, Suspended, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=True, requires_approval=True)
def refund(order_id: str, amount_usd: float) -> str:
    """Issue a refund against a customer order and return the confirmation id."""
    return f"refunded {amount_usd} on {order_id}"


agent = Agent(
    name="billing",
    model="fake",
    prompt="You handle refunds.",
    cognition=ReActCognition(tools=[refund]),
)

llm = FakeLLM.script([
    Turn(tool_calls=(ToolCall("c1", "refund", {"order_id": "A-1", "amount_usd": 20.0}),)),
    Turn(content="Refund issued."),
])


async def main() -> None:
    ctx = make_test_ctx(llm=llm)  # autonomy="auto", the default
    result = await agent.run("Refund order A-1 for $20", ctx)
    suspended = result.evals["suspended"]
    assert isinstance(suspended, Suspended)
    print(result.stop_reason, suspended.reason)
    print([(c.name, dict(c.arguments)) for c in suspended.pending])


asyncio.run(main())
```

```text
suspended awaiting_approval
[('refund', {'order_id': 'A-1', 'amount_usd': 20.0})]
```

The refund never ran. See
[Human-in-the-loop tool approval](../recipes/hitl-tool-approval.md) for
the resume half.

**2. `idempotent()` middleware.** An at-least-once retry must not re-fire
a mutation. `idempotent()` is the one caller in the framework that opts
into caching side-effecting calls, and it stays safe by pinning the key to
`(correlation_id, scope, tool name, arguments)` — replay is confined to a
single run.

```python
import asyncio

from agentkit.adapters.store.memory import InMemoryStore
from agentkit.kernel.types import Scope, ToolRequest
from agentkit.middlewares import idempotent
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import tool

sent: list[str] = []


@tool(side_effecting=True)
def send_email(to: str, body: str) -> str:
    """Send one email to a single recipient and return the provider message id."""
    sent.append(to)
    return f"queued for {to}"


async def main() -> None:
    ctx = make_test_ctx(
        llm=FakeLLM(),
        store=InMemoryStore(),
        scope=Scope(org_id=1, domain_id=1),
        tool_middleware=[idempotent()],
    )
    request = ToolRequest(
        name="send_email",
        arguments={"to": "ops@example.com", "body": "prod is down"},
        tool=send_email,
    )
    print(await ctx.invoker.invoke_tool(request, ctx))
    print(await ctx.invoker.invoke_tool(request, ctx))  # a retry of the SAME call
    print("emails actually sent:", len(sent))


asyncio.run(main())
```

```text
queued for ops@example.com
queued for ops@example.com
emails actually sent: 1
```

**3. `memoize()` refuses.** The general-purpose cache reads the same flag
and declines side-effecting calls outright. The comment in
`agentkit/middlewares/memoize.py` records the measured failure that made
this the default: `send_email(to=…)` invoked twice sent **one** email and
the second caller was handed the first call's stored `{'sent': True}`.
Caching a side effect reports success for an action that never happened;
a missed cache hit costs a re-execution. The asymmetry decides the
default.

The declaration of record is the **tool object**, not the request.
`ToolRequest.side_effecting` defaults to `False` and plenty of call sites
build the request positionally, so the middleware ORs the two — reading
the request alone fixed only half the reproduction.

!!! note "`idempotent=` is a separate axis"
    `side_effecting` asks *does this change the world?*; `idempotent`
    asks *is a retry safe?*. Read-only lookups are usually
    `side_effecting=False, idempotent=True`. Mutations are usually
    `side_effecting=True, idempotent=False`. They are independent, and
    `idempotent` defaults to `False`.

### `ctx` is injected, not advertised

A parameter named `ctx` or `context` receives the live `RunContext` and
is stripped from the schema the model sees. That is how a tool reaches
the tenant scope, the budget, the trace, or `ctx.emit`.

```python
import asyncio

from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False, idempotent=True)
def list_projects(prefix: str, ctx) -> list[str]:
    """List the calling tenant's projects whose name starts with a prefix."""
    return [f"{ctx.scope.key()}/{prefix}-alpha", f"{ctx.scope.key()}/{prefix}-beta"]


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM(), scope=Scope(org_id=7, domain_id=1))
    print(list_projects.schema.parameters)
    print(await list_projects.run({"prefix": "web"}, ctx))


asyncio.run(main())
```

```text
{'type': 'object', 'properties': {'prefix': {'type': 'string'}}, 'required': ['prefix']}
['org7:dom1/web-alpha', 'org7:dom1/web-beta']
```

The hijack is narrow on purpose: the name only counts as the injected
context when it is **unannotated or annotated as `RunContext`**. A real
data parameter like `def f(context: str)` stays advertised and is passed
normally, so a tool that genuinely wants a field called `context` is not
silently broken.

A `ctx` key arriving *from the model* is dropped rather than reported as
an error — but only when the function actually declares such a parameter.
A tool with no `ctx` parameter has no name to shadow, so the key is
rejected like any other unknown argument.

## The technical contract

### `Tool` is a Protocol

```python
from agentkit.tools import Tool

print(sorted(Tool.__annotations__))
```

```text
['description', 'name', 'output_schema', 'requires_approval', 'schema', 'side_effecting']
```

Anything with those attributes plus `async def run(args, ctx)` **is** a
tool. `Tool` is `@runtime_checkable`, and `ToolRegistry.register` uses
`isinstance` at the seam so a half-shaped object (one with `name` and
`run` but no `schema`) is refused there rather than exploding later
inside `schemas()`. `FileTool` is a duck-typed tool that satisfies the
Protocol without inheriting anything; so are MCP tools and remote-
procedure adapters.

`run`'s `args` is typed `Mapping[str, Any]` — the widest useful type. In
a normal run the invoker hands down a `FrozenDict` (a `dict` **subclass**
that refuses mutation), so `args["x"] = 1` inside a tool raises
`TypeError`. Copy with `dict(args)` if you need to edit, and note that
copy is *shallow* — nested values stay frozen all the way into the tool
body. That is the point: the arguments that were **authorised** stay the
arguments that get **executed**, and the audit record that names them
stays true.

### `ToolRegistry` is lookup, and nothing else

Execution moved out to the invoker's middleware chain, so the registry
answers three questions: name → tool, what schemas to advertise, and does
this name need approval.

- `register(tool, replace=False)` accepts a `FunctionTool` **or a plain
  callable**, which it runs through `from_callable` — same docstring
  floor, same contract, and a thin docstring fails here rather than at
  runtime.
- `from_tools([...])` builds a registry from a mixed list.
- `schemas()` returns tools in **sorted name order**, deliberately: a
  stable order keeps the cacheable system+tools prefix byte-identical
  across turns.
- Name collisions raise instead of overwriting.

```python
from agentkit.tools import ToolRegistry, tool


@tool(side_effecting=False)
def forecast(city: str) -> str:
    """Return a short weather forecast for a city over the next few days."""
    return "mild"


registry = ToolRegistry.from_tools([forecast])
try:
    registry.register(forecast)
except ValueError as exc:
    print(exc)
```

```text
ToolRegistry: tool 'forecast' already registered. Pass replace=True to swap deliberately.
```

Silently swapping would be worse than it sounds: the model still sees the
same advertised name, but the implementation behind it has changed. A
typo or an import-order accident would be indistinguishable from correct
behaviour. `replace=True` makes a deliberate hot-swap say so.

### Agents as tools

`as_tool` wraps anything with `async run(task, ctx) -> result` — a leaf
`Agent`, a coordinator `Agent`, a `Workflow` — into a `FunctionTool` an
outer agent can call. The sub-run executes on `ctx.child()`, so budget,
cancellation, depth and observation all flow through it, and the result
is rendered back to text for the outer loop.

```python
import asyncio

from agentkit import Agent, ToolCall
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx
from agentkit.tools import as_tool

researcher = Agent(
    name="researcher",
    model="fake",
    prompt="You summarise a topic in one sentence.",
)

research_tool = as_tool(
    researcher,
    name="researcher",
    description="Delegate a research sub-task and get a one-sentence summary back.",
    side_effecting=False,
)

lead = Agent(
    name="lead",
    model="fake",
    prompt="You answer by delegating research.",
    cognition=ReActCognition(tools=[research_tool]),
)

llm = FakeLLM.script([
    Turn(tool_calls=(ToolCall("c1", "researcher", {"task": "octopus tool use"}),)),
    Turn(content="Octopuses carry coconut shells as portable shelters."),
    Turn(content="They do: they carry coconut shells around as shelter."),
])


async def main() -> None:
    ctx = make_test_ctx(llm=llm)
    print(research_tool.schema.parameters)
    print((await lead.run("Do octopuses use tools?", ctx)).output)


asyncio.run(main())
```

```text
{'type': 'object', 'properties': {'task': {'type': 'string', 'description': 'the sub-task / goal to run'}}, 'required': ['task']}
They do: they carry coconut shells around as shelter.
```

The schema is fixed: one `task` string. That is the whole point — a
sub-agent takes a goal, not a parameter list, and the composition is
"everything is callable" rather than a second runtime.

`render_result` is the default renderer: it returns `AgentResult.output`
when the result has a string `output`, joins a `WorkflowResult`'s
terminal `outputs` as `key: value` lines, and falls back to `str(res)`.
Pass `render=` to override.

For the reverse direction — an `Agent` as a node **inside** a
`Workflow` — see [Agents](agents.md); it is native, no adapter needed.

### Validating what a tool returns

`output_schema=` validates the tool's **result** after the function ran.
It is auto-inferred from the return-type annotation when that annotation
is a Pydantic model, a dataclass, or an attrs class; primitives, `Any`,
generics and unions are skipped, so a tool typed `-> str` pays nothing.
Pass an explicit class to override, or `output_schema=None` to opt out
even when the return type looks enforceable.

```python
import asyncio
from dataclasses import dataclass

from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import ToolShapeError, tool


@dataclass
class Quote:
    symbol: str
    price: float


@tool(side_effecting=False, idempotent=True)
def get_quote(symbol: str) -> Quote:
    """Return the latest traded price for a stock ticker symbol."""
    return "not a Quote at all"  # the bug this catches


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM())
    print("inferred:", get_quote.output_schema)
    try:
        await get_quote.run({"symbol": "ACME"}, ctx)
    except ToolShapeError as exc:
        print(exc)
        print("raw =", repr(exc.raw), "| expected =", exc.expected)


asyncio.run(main())
```

```text
inferred: <class '__main__.Quote'>
tool 'get_quote' returned value not matching schema 'Quote'
raw = 'not a Quote at all' | expected = Quote
```

A mismatch also drops a `tool.shape_mismatch` event on the open
`execute_tool` span, best-effort — a misbehaving tracer must never break
the tool path.

### Three error types, not one

| Error | Fires | Who can fix it |
|---|---|---|
| `ToolDefinitionError` | at **decoration/registration** time | **you** — missing `side_effecting=`, thin description, `*args`, a developer note above the `---` cut |
| `ToolArgumentError` | at **call** time, before the function runs | **the model** — it authored a bad call (missing, unexpected, or uncoercible argument) |
| `ToolShapeError` | at **call** time, after the function returned | **the tool** — it produced a bad value |

The split is about *recovery*, not taxonomy.

`ToolArgumentError` carries three kinds of complaint, because a repair
turn recovers from all of them the same way — reflect and reissue:
`missing` and `unexpected` name arguments, and `invalid` names arguments
that were present but could not be coerced to the annotated type.

```python
import asyncio
import enum

from agentkit.tools import ToolArgumentError, tool


class Unit(enum.Enum):
    C = "celsius"
    F = "fahrenheit"


@tool(side_effecting=False)
async def weather(city: str, unit: Unit) -> str:
    """Report the weather for a city in the requested unit of measure."""
    return f"{type(unit).__name__}={unit!r}"


async def main() -> None:
    # A legal value arrives as the Enum member, not the wire string.
    print(await weather.run({"city": "Lisbon", "unit": "celsius"}, None))

    # An illegal one is refused before the body runs.
    try:
        await weather.run({"city": "Lisbon", "unit": "kelvin"}, None)
    except ToolArgumentError as exc:
        print(exc.invalid)


asyncio.run(main())
```

```text
Unit=<Unit.C: 'celsius'>
(('unit', "'kelvin' is not a valid Unit; expected one of ['celsius', 'fahrenheit']"),)
```

All bad arguments are collected before raising, so one repair turn learns
every mistake at once rather than discovering them one call at a time.

`ToolDefinitionError` is a bug in your wiring and there is no runtime
recovery for it, so it fires at import — before an agent exists, let
alone a bill. It subclasses `ValueError` so existing `except ValueError`
paths still trip.

`ToolArgumentError` is the model's mistake, and the model is the only
party that can fix it — so the message names the tool, the offending
arguments, and the accepted set, and the retry middleware reflects it
back as something actionable.

```python
import asyncio

from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import ToolArgumentError, tool


@tool(side_effecting=True)
def notify(msg: str = "default message") -> str:
    """Page the on-call engineer with a short incident message."""
    return f"sent: {msg}"


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM())
    try:
        await notify.run({"message": "prod is down"}, ctx)
    except ToolArgumentError as exc:
        print(exc)
        print("unexpected =", exc.unexpected, "| accepted =", exc.accepted)


asyncio.run(main())
```

```text
tool 'notify' call rejected: unexpected argument(s) ['message']. Accepted arguments: ['msg']
unexpected = ('message',) | accepted = ('msg',)
```

That example is the exact failure the error exists to stop.

Unknown keys used to be dropped silently, so the parameter ran with its
**default** instead. A model calling
`notify(message="page the on-call, prod is down")` against
`def notify(msg: str = "default message")` got back
`"sent: default message"`.

That is a side-effecting tool reporting success for something it never
did, with nothing downstream able to tell the difference.

A tool that genuinely accepts arbitrary keys declares `**kwargs`, and
then extras are passed through instead of rejected:

```python
import asyncio

from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False)
def flexible(query: str, **filters) -> str:
    """Search the catalogue with a free-form set of extra filter keys."""
    return f"{query} + {sorted(filters)}"


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM())
    print(await flexible.run({"query": "boots", "colour": "red", "size": 42}, ctx))
    print(flexible.schema.parameters)


asyncio.run(main())
```

```text
boots + ['colour', 'size']
{'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}
```

`ToolShapeError` is the mirror image: the value is bad on the way *out*,
the model did not author it, and the useful recovery is different — the
model can re-issue with different arguments or pivot entirely. It is
deliberately distinct from `OutputCoercionError`, which is about
coercing the *model's own response*: different fire site, different
recovery.

### Other things in the box

!!! note "`FileTool` advertises itself as `memory`, and that is deliberate"
    `FileTool().schema.name` is `"memory"`, which reads like a clash with
    the `agentkit.memory` package. It is not a naming slip: this is
    agentkit's implementation of Anthropic's client-side memory tool, and
    four things are fixed by that contract — the name, the six commands
    (`view`, `create`, `str_replace`, `insert`, `delete`, `rename`),
    single-call `command=` dispatch, and the `/memories` root. Renaming
    it would break the vendor integration, so the name is pinned by a
    test rather than left to judgement.


`FileTool` is a duck-typed tool that gives the model a self-managed note
tree — `view` · `create` · `str_replace` · `insert` · `delete` ·
`rename` — over a backend that defaults to `InMemoryFiles`. It is
`side_effecting=True`, and its advertised name is `memory`. The read
side of the same tree is `FileMemory`; see [Memory](memory.md).

```python
from agentkit import FileTool

ft = FileTool()
print(ft.name, ft.side_effecting, ft.requires_approval)
```

```text
memory True False
```

## What bites people

!!! tip "Rich types arrive as the type you annotated"
    An `Enum`, a `Literal` or a typed struct is coerced at the call
    boundary, driven by the same annotation the schema was derived from.
    This used to be the opposite — the body received the raw JSON value,
    so a parameter annotated `unit: Unit` got a `str` and `unit.value`
    raised `AttributeError` at runtime on a value the schema said was a
    `Unit`.

    Primitives are deliberately **not** coerced. Nothing turns `"3"` into
    `3`: a provider sending a string for an integer parameter has a bug
    the framework should surface, not launder.

    A value that cannot be coerced raises `ToolArgumentError` before the
    function runs — see [the error table](#three-error-types-not-one).

!!! warning "`Agent` has no `tools=` keyword"
    Tools go on the cognition: `ReActCognition(tools=[...])`. Passing
    `tools=` to `Agent(...)` raises `TypeError`.

!!! warning "Your whole docstring is the model's instructions"
    Everything above a `---` line goes to the model, implementation
    notes included. `TODO`/`FIXME`/`XXX`/`HACK` are caught at decoration
    time, but "see ticket #442" and "the vendor lies about 404s" are
    not, and they will ship as guidance. Put a `---` line above the
    human half, or pass `description=`.

!!! tip "Give the model less than you have"
    A `schema` of `None` makes a tool **loop-invisible**: it stays
    callable through the registry and the invoker but is never
    advertised to the model. Useful for tools a cognition drives itself.

!!! warning "Don't mutate `args`"
    It arrives frozen, deeply. `dict(args)` gives you a shallow mutable
    copy; nested values stay frozen. A tool that popped a key off its
    own arguments would make the authorised call and the executed call
    two different things, with the audit trail recording only the first.

!!! tip "`from_callable` defaults `side_effecting=False`"
    The *decorator* forces you to declare it. `FunctionTool.from_callable`
    and `ToolRegistry.register(plain_function)` default to `False` for
    compatibility. If you go through those paths — including passing bare
    functions to `ReActCognition(tools=[...])` — pass `side_effecting=`
    knowingly, or your mutation will be treated as cacheable.

## Related

- [Middlewares](middlewares.md) — the chain every tool call passes
  through: tracing, metering, egress, `idempotent()`, audit, retry.
- [Agents](agents.md) — where a `ToolRegistry` is wired, and how a tool
  loop terminates.
- [Memory](memory.md) — the other way an agent reaches outside itself,
  and `ToolMemory`, which turns a tool into a memory source.
- [Skills](skills.md) — a prompt + cognition + memory bundle that can be
  handed to another agent as a tool.
- [Human-in-the-loop tool approval](../recipes/hitl-tool-approval.md) —
  the resume half of `requires_approval=True`.
- [Concepts · Integrations](integrations.md) — MCP tools adapted into this
  same `Tool` Protocol, and the server side where agentkit exposes one.
- [Consume MCP tools from an agent](../recipes/mcp-tools.md).
- [API › tools](../api-reference/tools.md) — the generated reference.
