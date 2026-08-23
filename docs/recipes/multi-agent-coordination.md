# How do I split work across several agents?

One agent with twelve tools and a four-paragraph prompt does every job
slightly badly. Splitting it into a researcher, a writer, and a critic
gives each one a short prompt and a small tool set — and then somebody
has to decide who goes next.

## When you'd want this

The symptom is a prompt that has grown sections. "If the question is
about billing, do X; if it's about the API, do Y; always check Z first."
Each section competes with the others for the model's attention, and
adding the thirteenth makes the first twelve worse.

Splitting the roles is easy. What's hard is the coordination, and it is
the part people rewrite three times:

- who speaks next, and on what evidence;
- how the second agent sees what the first one found;
- when to stop, given that "the model said it's done" is not by itself
  a stopping condition;
- what happens to the budget when a child loops.

A coordinator `Agent` is an `Agent` whose cognition is
`CoordinatorCognition(children=..., policy=...)`. The **policy** owns
who-speaks-next and termination; the children are ordinary agents that
do not know they are in a team.

## Working code

```python
"""Runs offline: FakeLLM stands in for the model, so no API key is needed."""

import asyncio

from agentkit import Agent
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies import PlanPolicy, StaticPlanner, Step
from agentkit.testing import FakeLLM, make_test_ctx


def team() -> dict[str, Agent]:
    return {
        "researcher": Agent(name="researcher", model="claude-sonnet-4-6", prompt="Find the facts."),
        "writer": Agent(name="writer", model="claude-sonnet-4-6", prompt="Write the brief."),
        "critic": Agent(name="critic", model="claude-sonnet-4-6", prompt="Critique the brief."),
    }


async def main() -> None:
    lead = Agent(
        name="lead",
        model="claude-sonnet-4-6",
        cognition=CoordinatorCognition(
            children=team(),
            # Group 0 runs first. Group 1's two steps run CONCURRENTLY.
            policy=PlanPolicy(planner=StaticPlanner([
                Step("researcher", "gather sources", group=0),
                Step("writer", "draft from the sources", group=1),
                Step("critic", "review the draft", group=1),
            ])),
        ),
    )

    ctx = make_test_ctx(llm=FakeLLM("done"))
    result = await lead.run("Brief me on octopus cognition.", ctx)

    print(result.stop_reason, "|", result.output)
    print("per-step results:", [r.output for r in result.evals["results"]])
    print("errors:", result.evals["errors"])
    print("total usage:", result.usage.input_tokens, "in /", result.usage.output_tokens, "out")


asyncio.run(main())
```

Output:

```text
complete | done
per-step results: ['done', 'done', 'done']
errors: []
total usage: 30 in / 15 out
```

## Picking a policy

Four ship, and the choice is really "how much does the model decide?".

| Policy | Who goes next | Reach for it when |
|---|---|---|
| `RoundRobinPolicy` | fixed rotation | a debate or a review cycle with a known cast |
| `SelectorPolicy` | a `Selector` callable you supply | routing depends on the conversation |
| `PlanPolicy` | a `Planner` produces named steps in groups | the work decomposes up front, and phases can run concurrently |
| `LedgerPolicy` | a planner plus a progress ledger | long runs that stall, and you want the stall detected |

`PlanPolicy` is the orchestrator-worker shape: `StaticPlanner` for a
fixed plan, or your own `Planner` (`plan(goal, ctx) -> list[Step]`, sync
or async) to have a model produce it. Steps in the same `group` run
concurrently under the tree semaphore; groups run in order.

`best_effort=True` isolates a failing step into
`evals["errors"]` as `(child_name, Failure)` instead of cancelling the
group — so partial progress survives.

## Routing with `Handoff`

`SelectorPolicy` takes a callable `(transcript, agents)` — optionally a
third `ctx` argument, sync or async — returning the next child's name.
The shipped one reads the typed `Handoff` verb out of the last message:

```python
import asyncio

from agentkit import Agent
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.control import MaxTurns, TextMention, route_by_handoff
from agentkit.agents.policies import SelectorPolicy
from agentkit.testing import FakeLLM, Turn, make_test_ctx


async def main() -> None:
    lead = Agent(
        name="lead",
        model="claude-sonnet-4-6",
        cognition=CoordinatorCognition(
            children={
                "triage": Agent(name="triage", model="claude-sonnet-4-6", prompt="Classify, then hand off."),
                "billing": Agent(name="billing", model="claude-sonnet-4-6", prompt="Answer billing questions."),
            },
            policy=SelectorPolicy(selector=route_by_handoff(default="triage"), max_turns=6),
            termination=TextMention("RESOLVED") | MaxTurns(4),
        ),
    )
    # Turn 1: triage (the default) emits the marker. Turn 2: the selector
    # reads it and dispatches billing, which says the stop word.
    llm = FakeLLM.script([
        Turn(content="HANDOFF:billing customer asked about an invoice"),
        Turn(content="You were charged twice; a refund is on its way. RESOLVED"),
    ])
    result = await lead.run("Why was I charged twice?", make_test_ctx(llm=llm))

    print(result.stop_reason, "/", result.evals["stop_reason"])
    print(result.output)
    print("who spoke:", [m.name for m in result.evals["messages"] if m.name])


asyncio.run(main())
```

Output:

```text
terminated / text_mention
You were charged twice; a refund is on its way. RESOLVED
who spoke: ['triage', 'billing']
```

A `Handoff(target, reason, message)` reaches the transcript one of two
ways: `handoff_tool(targets=[...])` gives the model a tool whose schema
constrains `target` to a known name, or `parse_handoff` reads a
`HANDOFF:` marker out of plain text for models without tool calling.
Either way the coordinator interprets it — the child says where it
wants control to go, the coordinator decides.

An invented target does not stall the run: `route_by_handoff` falls back
to its `default`, warning once per unknown name. Matching is exact or a
unique case-insensitive match, so `HANDOFF:Billing` still reaches
`billing`.

## Stopping

`termination=` on the cognition is a `TerminationCondition`, and they
compose with `&` / `|`:

```python
from agentkit.agents.control import MaxMessages, MaxTurns, TextMention, Timeout

stop = (TextMention("APPROVED") | MaxTurns(8)) & MaxMessages(40)
guarded = stop | Timeout(120.0)
```

Shipped: `MaxTurns`, `MaxMessages`, `TextMention`, `FunctionCall`,
`SourceMatch`, `Timeout`, `ExternalTermination` (an operator's stop
button), `FunctionalTermination` (your own predicate), and
`judge_termination` (ask a model). Always keep a countable one in the
composition — `TextMention("DONE")` alone means a team that never says
"DONE" runs until something else stops it.

## How the children see each other

Coordination is via the **blackboard**, not a return channel. A child's
reply lands on the shared transcript, and the next child is dispatched
with that transcript rendered into its task. There is no parent-side
plumbing to write: the shared `WorkingContext` scratchpad is the other
half, for structured notes a child wants to leave behind.

Children run on `ctx.child()` contexts, which share `budget`,
`services`, and `cancel` **by reference**. So one `Budget` is the
ceiling for the whole tree, `budget.usage` rolls the whole tree up, and
cancelling the parent cancels the subtree. `Budget.max_depth` bounds how
deep the tree can spawn.

For per-child envelopes rather than one shared pot, see `ActorBudget` in
[Cap spend with Budget and Quota](spend-budget-and-quota.md#per-agent-budgets).

## Three ways to nest, and when each is right

| Shape | Control | Use when |
|---|---|---|
| `CoordinatorCognition(children=..., policy=...)` | emergent | who goes next depends on what was said |
| [`Workflow`](workflow-graph.md) | explicit | the order is fixed and you want it enforced |
| [`as_tool(agent, ...)`](define-a-tool.md#an-agent-as-a-tool) | the model's | a sub-agent is one capability among the caller's tools |

They compose in both directions: a coordinator can be a `Workflow` node
(`wf.coordinator(...)`), and a `Workflow` can be a tool an agent calls.
There is no bridging runtime — everything is callable, and the budget /
cancellation / observation spine flows through the `ctx.child()` it runs
on.

## Gotchas

- **A coordinator does not stream.** `CoordinatorCognition.drive` yields
  exactly one terminal `final` event carrying the aggregated result.
  Progress comes from `policy.dispatch` observations and the child
  spans, not from tokens. If your UI needs live output, subscribe to the
  observation channel.
- **`max_turns` and `termination=` are different ceilings.** The policy's
  `max_turns` is the hard backstop (default 50 on `SelectorPolicy`); the
  termination condition is the intended stop. A run ending with
  `stop_reason="max_iterations"` means the backstop fired — your
  termination condition never did, which is usually the bug.
- **A `Failure` in a slot still cost money.** `best_effort=True` isolates
  the exception; it does not refund the calls that raced to it.
- **`PlanPolicy` validates the plan against the roster first.** A step
  naming a child that doesn't exist raises `PlanShapeError` before any
  step is dispatched. Previously it surfaced as a bare `KeyError` after
  steps 1–4 had already run and spent, with their results unreachable.
- **A plan gate is not a workflow gate.** `Step.gate("review")` suspends
  at the coordinator level and resumes with `PlanPolicy.resume`, at the
  slot `{run_id}:plan` — deliberately namespaced so a child agent's own
  checkpoint at `{run_id}:agent:{name}` cannot clobber it. Same
  primitive, different producer.

## Related

- [Run a fixed multi-step pipeline](workflow-graph.md) — when the order
  should not be up for discussion.
- [Parallel agents with cooperative cancellation](parallel-agents-with-cancellation.md)
  — `run_agents`, the fan-out primitive the policies are built on.
- [Cap spend with Budget and Quota](spend-budget-and-quota.md) — the
  ceiling a team of agents shares.
- [Give an agent a tool](define-a-tool.md) — `as_tool`, the third
  nesting shape.
