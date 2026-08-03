# Tutorial: build a research-briefing agent

Fifteen minutes, five steps. At the end you'll have an agent that takes a
question, searches for sources, drafts a brief, streams progress to the
caller, refuses to overspend, and pauses for a human before it publishes.

Every step is a complete, self-contained script — copy the whole block and
run it. Each step introduces exactly one primitive on top of the previous.

## Before you start

Install the package and drop into a Python 3.12+ shell:

```bash
pip install arc-agentkit
```

Nothing else is required. Every snippet in this tutorial uses
`agentkit.testing.FakeLLM` — no API keys, no network. To swap in a real
provider later, replace the `FakeLLM(...)` line with a `claude(...)` /
`openai(...)` client from the batteries-included preset (see
[Getting started](getting-started.md)).

## Step 1 — a single-turn agent

The smallest thing that runs. An `Agent`, a `RunContext`, and one call.

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    ctx = make_test_ctx(
        llm=FakeLLM("Octopuses appear to plan, use tools, and solve mazes."),
    )
    agent = Agent(
        name="briefer",
        model="gpt-4o-mini",
        prompt="Answer the question in one short sentence.",
    )
    result = await agent.run("What do we know about octopus cognition?", ctx)
    print(result.output)
    print(f"cost: ${result.usage.cost_usd:.4f}")


asyncio.run(main())
```

**What just happened.** `Agent` is identity + chat-call config — nothing
more. Because we didn't pass a `cognition=`, the default
`SingleCallCognition` fired one chat call and returned. `make_test_ctx`
built a real `RunContext` wired with a real `Invoker`, a real `Budget`,
and the standard `NoopTrace` / `NoopObserver` seams. The only fake is
the LLM. `AgentResult` is typed — `output` is the model text,
`usage` is the accrued `Usage` for the whole run.

## Step 2 — add a tool

To do more than one chat call, swap the default cognition for
`ReActCognition` and register a tool.

```python
import asyncio

from agentkit import Agent, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns two bulleted hits with source and year."""
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    # Scripted LLM: turn 1 requests the tool, turn 2 speaks the final brief.
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
            Turn(
                content=(
                    "Octopuses use tools and show distributed cognition "
                    "(Nature 2009, Science 2023)."
                )
            ),
        ]
    )
    ctx = make_test_ctx(llm=llm)
    agent = Agent(
        name="briefer",
        model="gpt-4o-mini",
        prompt="Research the question. Use `search` to find sources before answering.",
        cognition=ReActCognition(tools=[search]),
    )
    result = await agent.run("What do we know about octopus cognition?", ctx)
    print(result.output)


asyncio.run(main())
```

**What just happened.** `@tool(side_effecting=False)` wraps a plain async
function into a `FunctionTool`. The `side_effecting=` keyword is
required — the framework uses it to decide whether human approval is
needed (see step 5) and whether a retry is safe. `ReActCognition` owns
the tool-loop: chat → tool_call → tool_result → chat → final. A plain
`list` of tools is auto-wrapped into a `ToolRegistry`.

`FakeLLM.script([Turn(...), ...])` replays scripted turns across
successive chat calls. In production the model *decides* whether to
call the tool; here we hand-write both turns so the tutorial doesn't
depend on a live provider.

## Step 3 — stream the run

`agent.run(...)` is just `agent.stream(...)` collected. Streaming lets
you show progress, log intermediate tool calls, or forward events over
a WebSocket without waiting for the whole run.

```python
import asyncio

from agentkit import Agent, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns two bulleted hits with source and year."""
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
            Turn(
                content=(
                    "Octopuses use tools and show distributed cognition "
                    "(Nature 2009, Science 2023)."
                )
            ),
        ]
    )
    ctx = make_test_ctx(llm=llm)
    agent = Agent(
        name="briefer",
        model="gpt-4o-mini",
        prompt="Research the question. Use `search` to find sources before answering.",
        cognition=ReActCognition(tools=[search]),
    )

    async for ev in agent.stream("What do we know about octopus cognition?", ctx):
        if ev.type == "message_delta":
            print(ev.text, end="", flush=True)
        elif ev.type == "tool_call":
            print(f"\n[tool_call] {ev.tool_call.name}({dict(ev.tool_call.arguments)})")
        elif ev.type == "tool_result":
            print(f"[tool_result] {ev.tool_result!r}")
        elif ev.type == "step":
            print(f"[step] {ev.text}")
        elif ev.type == "final":
            print(f"\n[final] usage={ev.result.usage}")


asyncio.run(main())
```

**What just happened.** `StreamEvent` is a closed union. The types you'll
handle in day-to-day code:

- `message_delta` — one incremental chunk of the assistant's text
- `tool_call` — the model asked to run a tool (fires before the tool runs)
- `tool_result` — the tool returned (or an error string the model can see)
- `step` — an iteration boundary, e.g. `"iteration:1"`
- `interrupt` — the loop is about to suspend for human approval (step 5)
- `final` — exactly one terminal event carrying the `AgentResult`

The `Agent` never opens a WebSocket for you. It hands you events; you
decide where they go.

## Step 4 — cap the run with a budget

A `Budget` puts a hard ceiling on cost and call count for one run. Wire
the `meter()` middleware into the chat chain and the invoker charges the
budget under a lock — overspend raises `MeterExceeded` and the loop
halts cleanly.

```python
import asyncio

from agentkit import Agent, Budget, MeterExceeded, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.middlewares import meter
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns two bulleted hits with source and year."""
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    # FakeLLM charges Usage(10, 5, cost_usd=0.0001) per chat call.
    # A ceiling of $0.0001 lets the first call land (spent == max, not >),
    # then trips MeterExceeded on the second — the loop halts cleanly.
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
            Turn(content="Octopuses use tools and show distributed cognition."),
        ]
    )
    ctx = make_test_ctx(
        llm=llm,
        budget=Budget(max_cost_usd=0.0001, max_calls=1),
        chat_middleware=[meter()],
    )
    agent = Agent(
        name="briefer",
        model="gpt-4o-mini",
        prompt="Research the question. Use `search` before answering.",
        cognition=ReActCognition(tools=[search]),
    )

    try:
        await agent.run("What do we know about octopus cognition?", ctx)
    except MeterExceeded as exc:
        print(f"halted: {exc}")
    print(f"spent: ${ctx.budget.spent_usd:.4f}  calls: {ctx.budget.calls}")


asyncio.run(main())
```

**What just happened.** `Budget` is one instance per run; every chat call
goes through the `meter` middleware, which calls `budget.guard` before
the work and `budget.charge` after. Both take the budget's async lock —
totals are invariant under concurrent workers, so you can fan out
sub-agents in step 5 without their spends racing. The overspend raises
`MeterExceeded` (from `agentkit.runtime.meter`) inside the invoker — the
`ReActCognition` loop lets it propagate, which is the correct behavior
for a run that must not overspend.

## Step 5 — pause for human approval before publishing

Some tools should not run without a human saying yes. Declare the tool
`side_effecting=True`, set the run's `autonomy` to `"gated"`, and the
loop will `Suspend` before dispatching the tool. Your driver decides
whether to `resume(...)` and with what per-call decisions.

```python
import asyncio

from agentkit import Agent, Suspended, ToolCall, tool
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns one bulleted hit."""
    return "- 'Distributed cognition in cephalopods' (science.org, 2023)"


@tool(side_effecting=True)
async def publish_brief(title: str, body: str) -> str:
    """Publish the finished brief to the team wiki. Idempotent per `title`."""
    return f"published {title!r} ({len(body)} chars)"


async def main() -> None:
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
            Turn(
                tool_calls=(
                    ToolCall(
                        "c2",
                        "publish_brief",
                        {"title": "Octopus cognition", "body": "Short brief..."},
                    ),
                )
            ),
            Turn(content="Published. Brief is live."),
        ]
    )
    # A Checkpointer + a store is required for suspend/resume — the loop
    # snapshots its state to it, and `agent.resume(run_id, ...)` reads it back.
    checkpointer = Checkpointer(port=InMemoryCheckpointStore())
    ctx = make_test_ctx(
        llm=llm,
        checkpointer=checkpointer,
        autonomy="gated",  # GATED gates every side_effecting tool call
        correlation_id="run-42",
    )
    agent = Agent(
        name="briefer",
        model="gpt-4o-mini",
        prompt="Research, then publish.",
        cognition=ReActCognition(tools=[search, publish_brief]),
    )

    result = await agent.run("Brief the team on octopus cognition, then publish.", ctx)
    susp = result.evals.get("suspended")
    if not isinstance(susp, Suspended):
        raise SystemExit("expected the run to suspend for approval")
    print(f"[suspended] pending: {[tc.name for tc in susp.pending]}")

    # A real driver renders the pending tool calls to a human and collects
    # approve/reject decisions per ToolCall.id. Here we approve unconditionally.
    decisions = {tc.id: "approve" for tc in susp.pending}
    final = await agent.resume(susp.run_id, decisions, ctx)
    print(f"[resumed] {final.output!r}")


asyncio.run(main())
```

**What just happened.** `Autonomy.GATED` is the run-wide tier that says
"gate every key step". A key step is any tool where
`side_effecting=True`. When the loop encounters one, it snapshots its
state through the `Checkpointer`, emits an `interrupt` event per pending
tool call, and returns an `AgentResult` with `partial=True` and a
`Suspended` in `evals["suspended"]`. That `Suspended` carries the
`run_id` and a frozen tuple of `ToolCall`s awaiting decisions.

`agent.resume(run_id, decisions, ctx)` reads the snapshot back through
the same checkpointer, applies your per-call decisions (`"approve"`,
`"reject"`, `"deny"`, or a JSON string overriding the arguments),
appends the tool results to the transcript, and drives the loop to a
final answer. Because the state lives in the store, a completely fresh
process can resume weeks later — see the
[resume-after-crash recipe](recipes/resume-after-crash.md) for the
worker-restart pattern.

## Where to go next

- **[Concepts](concepts/kernel.md)** — the mental model of each
  primitive introduced here. Start with `Runtime` for `RunContext` /
  `Budget` and `Agents` for `Cognition` / `Autonomy`.
- **[Recipes](recipes/index.md)** — focused answers to "how do I X?",
  including durable resume after a crash, per-tenant `Quota`,
  parallel agents with cancellation, and writing your own middleware.
- **[Examples](examples.md)** — three fully-worked scripts in the repo:
  single agent, streaming with tools, a hand-composed middleware chain.
