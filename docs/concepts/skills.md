# Skills

A Skill is a named recipe for a specialised agent, written once and
reused.

!!! tip "Is this page for you?"

    **Reach for it when** you have an agent recipe — prompt,
    cognition, tools, memory — that you want to spawn more than
    once, or hand to another agent as a tool.

    **Skip it for now if** you have one agent, constructed in one
    place. A `Skill` would be indirection with nothing behind it.

## The problem it solves

You have a "researcher": a particular system prompt, a `ReActCognition`
holding three search tools, a vector memory over your paper index, and a
model choice. You need it in four places — as the top of a CLI run, as a
tool the planner agent can call, in an eval harness, and in tests
against a cheaper model.

Build an `Agent` and pass it around and you have a shared mutable object.
Cognitions carry live per-run state — `MaxTurns.turn`, `Timeout._start` —
so two concurrent runs sharing one instance race on the counter, and one
run's `reset()` blows away the other's mid-count. Rebuild the `Agent` at
each call site instead and the prompt drifts, because four copies of a
recipe are four things to keep in sync.

A `Skill` is the recipe as an immutable value. It carries the
configuration and nothing per-run, so it is safe to share across runs and
threads. Each use *materialises* a fresh `Agent` from it.

## The smallest thing that works

```python
import asyncio

from agentkit import Skill
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False, idempotent=True)
def search_papers(topic: str) -> str:
    """Search the internal paper index and return matching titles for a topic."""
    return "Octopus tool use in the wild (2009)"


researcher = Skill(
    name="researcher",
    description="Research a topic against the internal paper index and summarise it.",
    prompt="You are a careful researcher. Cite the papers you find.",
    cognition=ReActCognition(tools=[search_papers]),
    model="fake",
)


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM("Octopuses use coconut shells as shelter."))
    agent = researcher.as_agent()
    print(type(agent).__name__, agent.name, agent.model)
    print((await agent.run("Do octopuses use tools?", ctx)).output)


asyncio.run(main())
```

```text
Agent researcher fake
Octopuses use coconut shells as shelter.
```

## Skill versus Agent

The distinction is the same one as between a **recipe** and a **meal**.

A recipe is written once, does not change, and can be cooked any number
of times. Each meal is a separate thing that gets consumed. A `Skill` is
the recipe — a frozen description of an agent. An `Agent` is what you
get when you cook it, and it accumulates state as it runs, which is
exactly why you want a fresh one per run rather than a shared one.

So they are not the same kind of thing, and `Skill` is deliberately
**not** an `Agent` subclass.

| | `Skill` | `Agent` |
|---|---|---|
| What it is | a recipe | a runnable |
| Mutability | frozen dataclass, `slots=True` | mutable |
| Per-run state | none | cognition counters, termination state |
| Lifetime | wire once, reuse forever | one materialisation per run |
| How you use it | `as_agent()` / `as_tool()` | `await agent.run(task, ctx)` |

Six fields, two required:

- `name` — the stable identifier. Becomes `Agent.name` on
  materialisation and the `Tool` name when adapted.
- `description` — one line. Shown to the **outer** LLM when the Skill is
  a tool, so it drives tool selection.
- `prompt` — a `Prompt` (versioned) or a plain `str`. Defaults to `""`.
- `cognition` — the turn-taking strategy. Defaults to
  `SingleCallCognition()`, the only shipped cognition with a no-arg
  constructor, so the ergonomic `Skill("x", "y")` form works in tests.
- `memory` — a [`MemorySource`](memory.md) the underlying agent grounds
  against. `None` disables the hook.
- `model` — the default model, overridable per materialisation.

The framework ships **zero** concrete Skills. No `Researcher`, no
`Critic`, no `Synthesiser` — the same way it ships zero Tools.
Applications compose their own from their own prompts, tools, and
memory.

## `as_agent()` deep-copies the cognition

```python
from agentkit import Skill
from agentkit.agents.cognition import ReActCognition
from agentkit.agents.control.termination import MaxTurns
from agentkit.tools import tool


@tool(side_effecting=False, idempotent=True)
def search_papers(topic: str) -> str:
    """Search the internal paper index and return matching titles for a topic."""
    return "Octopus tool use in the wild (2009)"


researcher = Skill(
    name="researcher",
    description="Research a topic against the internal paper index and summarise it.",
    cognition=ReActCognition(tools=[search_papers], termination=MaxTurns(3)),
    model="fake",
)

one = researcher.as_agent()
two = researcher.as_agent()

one.cognition.termination.turn = 2  # pretend run one is mid-count

print("run one turn:", one.cognition.termination.turn)
print("run two turn:", two.cognition.termination.turn)
print("recipe turn: ", researcher.cognition.termination.turn)
print("distinct objects:", one.cognition is not two.cognition)
```

```text
run one turn: 2
run two turn: 0
recipe turn:  0
distinct objects: True
```

This is the reason the copy exists, stated plainly: **`ReActCognition`
holds mutable `TerminationCondition` state that is reset at the start of
every run.** `MaxTurns.turn` counts up as turns pass; `Timeout._start`
records when the clock began. Two concurrent runs materialised from one
shared `Skill` would race on that counter, and one run's `reset()` would
blow away the other's count mid-flight — a loop that stops three turns
early, or never.

`copy.deepcopy(self.cognition)` ties each `Agent` to its own state graph
while the immutable recipe stays shared. It also means the tools inside
the cognition are copied per materialisation, so a tool holding
expensive or non-copyable state is worth knowing about (see Gotchas).

Everything else is pinned at construction. Only `model` can be
overridden per call:

```python
from agentkit import Skill

researcher = Skill(
    name="researcher",
    description="Research a topic against the internal paper index and summarise it.",
    model="premium-model",
)

print(researcher.as_agent().model)
print(researcher.as_agent(model="cheap-model").model)
print("recipe unchanged:", researcher.model)
```

```text
premium-model
cheap-model
recipe unchanged: premium-model
```

## `as_tool()` — a Skill another agent can call

`skill.as_tool()` runs `as_agent()` and hands the result to
[`agentkit.tools.as_tool`](tools.md#agents-as-tools). The outer agent
picks the Skill out of its tool registry by name; the sub-run executes on
a `ctx.child()`, so budget, cancellation and observation flow through.

```python
import asyncio

from agentkit import Agent, Skill, ToolCall
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx

researcher = Skill(
    name="researcher",
    description="Research a topic against the internal paper index and summarise it.",
    prompt="You are a careful researcher.",
    model="fake",
)

lead = Agent(
    name="lead",
    model="fake",
    prompt="Delegate research, then answer.",
    cognition=ReActCognition(tools=[researcher.as_tool()]),
)

llm = FakeLLM.script([
    Turn(tool_calls=(ToolCall("c1", "researcher", {"task": "octopus tool use"}),)),
    Turn(content="Octopuses carry coconut shells as portable shelters."),
    Turn(content="Yes — they carry coconut shells around as shelter."),
])


async def main() -> None:
    ctx = make_test_ctx(llm=llm)
    skill_tool = researcher.as_tool()
    print(skill_tool.name, "|", skill_tool.description)
    print("side_effecting:", skill_tool.side_effecting)
    print((await lead.run("Do octopuses use tools?", ctx)).output)


asyncio.run(main())
```

```text
researcher | Research a topic against the internal paper index and summarise it.
side_effecting: False
Yes — they carry coconut shells around as shelter.
```

`name` and `description` default from the Skill and can be overridden at
adapter time — useful when the same Skill is exposed under different
labels in different registries.

## Skills are hashable, on recipe identity

`hash(Skill)` covers `(name, description, prompt, model)` — never
`cognition`, never `memory`. So a Skill can be a dict key or live in a
`set`, which is what a registry keyed on skill identity needs.

```python
from agentkit import Skill

a = Skill("researcher", "Research a topic against the paper index and summarise it.")
b = Skill("critic", "Critique a draft against the house style guide, in detail.")

print(len({a, b, a}))
print(a in {a: "first"})
```

```text
2
True
```

The exclusions are not arbitrary. Before this, `hash(Skill("researcher",
"digs"))` raised `TypeError: unhashable type: 'SingleCallCognition'` —
not `'dict'`, the whole cognition object. Every cognition the framework
ships is a mutable `@dataclass(slots=True)`, and a mutable dataclass with
the default `eq=True` gets `__hash__ = None` from `@dataclass` itself.
They are mutable *on purpose*, for the termination-state reason above.
And since `cognition` defaults to `field(default_factory=
SingleCallCognition)`, the ergonomic `Skill("x", "y")` form produced an
unhashable Skill — every Skill, not some.

`memory` is excluded for a related reason: `MemorySource` is a
`Protocol`, so the object is whatever your application wired in — often a
mutable dataclass, or a live client holding a connection pool. The
framework cannot promise anything about its hash.

The remaining four are hashable by construction, so the hash is **total**
— no Skill you can build breaks it. `Prompt` earns its place: it is a
frozen value with an explicit `__hash__` over `(id, version, template,
inputs)`, all `str`/`tuple[str, ...]`.

This is sound rather than a workaround. `__eq__` still compares every
field, including the cognition, and the hash invariant only requires
*equal* objects to hash equally. Two Skills differing only in cognition
collide into one bucket and `__eq__` separates them there, so `set` and
dict membership stay exact. Cost is O(1) in the excluded configuration:
measured at 0.25 µs for a bare `SingleCallCognition` and 0.23 µs for a
`ReActCognition` holding 1000 tools, because neither is read.

## When to reach for a Skill

Reach for one when a specialised agent configuration has **more than one
call site**, or when the same configuration must run twice concurrently.

- The same researcher runs standalone in one entry point and as a
  sub-tool of a planner in another. One recipe, two adapters.
- A coordinator dispatches three parallel children built from one
  configuration. Materialising per child is what stops them racing on
  termination state.
- You want a registry of capabilities keyed by name — that is what the
  hash is for.
- Tests need the same wiring on a cheap model:
  `skill.as_agent(model="cheap")`.

Do **not** reach for one when you have a single agent used once. `Agent`
is not heavy, and a Skill adds a layer for nothing. Mutation belongs on
the materialised `Agent`, not on the recipe.

## What bites people

!!! warning "`prompt` and `cognition` cannot be overridden at materialisation"
    `as_agent()` and `as_tool()` take `model` only. If two call sites
    need different prompts, they need different Skills — that is the
    recipe being a recipe.

!!! warning "The Skill is frozen; the Agent it produces is not"
    `Skill` is `@dataclass(frozen=True, slots=True)`, so
    `skill.model = "x"` raises. The materialised `Agent` is a normal
    mutable dataclass, and that is where per-run adjustment belongs.

!!! warning "`as_agent()` deep-copies your tools too"
    The copy walks the whole cognition, tool registry included. A tool
    holding an open connection, a large in-memory index, or anything
    with a custom `__deepcopy__` gets copied on every materialisation.
    Keep expensive state behind a module-level handle the tool *reads*
    rather than a field it *owns*.

!!! tip "Two Skills differing only in cognition hash alike"
    They collide into one hash bucket and `__eq__` separates them, so
    `set`/dict membership is still exact — but if you build many Skills
    that share `(name, description, prompt, model)` and differ only in
    cognition, your buckets degrade. Give them distinct names; `name` is
    documented as the stable identifier for exactly this reason.

!!! note "`memory` is shared, not copied"
    Only the cognition is deep-copied. Every `Agent` materialised from
    one Skill shares the same `MemorySource` object — which is normally
    what you want (one index, many readers), but it does mean a
    stateful memory backend is shared across concurrent runs. Wrap with
    [`ScopedMemory`](memory.md#tenant-scoping) when the tenants differ.

## Related

- [Agents](agents.md) — what `as_agent()` produces, and the cognitions a
  Skill can hold.
- [Tools](tools.md) — the `as_tool` adapter underneath `skill.as_tool()`,
  and the tools a Skill's cognition wires.
- [Memory](memory.md) — the `MemorySource` a Skill can carry.
- [Prompts](prompts.md) — versioned `Prompt` objects, which a Skill
  should prefer over a bare string.
- [API › skills](../api-reference/skills.md) — the generated reference.
