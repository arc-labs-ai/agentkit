# Tutorial: build a research-briefing agent

One agent, built in six steps. It starts as a single chat call and ends
as a researcher that uses a tool, streams its progress, refuses to
overspend, and stops to ask a human before it publishes anything.

Each step is a complete script — copy the whole block and run it. Each
step exists because the previous one left a specific problem open, and
each adds exactly one primitive to close it.

## Before you start

```bash
pip install arc-agentkit
```

**You need no API key for steps 1–5.** They run against `FakeLLM`, the
deterministic test double that ships in `agentkit.testing` — the same
one the framework's own suite uses. That is on purpose: the subject of
this tutorial is the machinery *around* the model (the tool loop, the
event stream, the budget, the approval gate), and a scripted model makes
every run reproducible, free, and fast enough to iterate on.

[Step 6](#step-6-point-it-at-a-real-model) swaps in a real provider. It
is a four-line change and it is the only step that costs money.

## Step 1 — an agent that answers

The smallest thing that runs: an `Agent`, a `RunContext`, one call.

```python
import asyncio

from agentkit import Agent
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt="Answer in one short sentence.",
    )
    ctx = make_test_ctx(llm=FakeLLM("Octopuses solve mazes and use tools."))

    result = await agent.run("What do we know about octopus cognition?", ctx)
    print(result.output)
    print(f"cost: ${result.usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

```text
Octopuses solve mazes and use tools.
cost: $0.0001
```

**What just happened.** `Agent` is a dataclass holding identity and
chat-call configuration — name, model, prompt — and nothing else. It has
no connection, no session, no mutable state. Everything the run is
allowed to touch lives on the `RunContext` you pass in, including the
LLM itself.

The default cognition is `SingleCallCognition`: one chat call, then
parse-and-repair if you asked for typed output. `AgentResult` is typed —
`output` is the text, `usage` is the accrued `Usage` for the whole run,
and `stop_reason` says why it ended.

**The problem this leaves.** The agent can only say what the model
already knows. It cannot look anything up.

## Step 2 — give it a tool

To let the agent *do* something, register a tool and swap the cognition
for `ReActCognition`, which owns the tool loop: chat → tool call → tool
result → chat → final answer.

```python
import asyncio

from agentkit import Agent, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the literature for `query`. Returns bulleted hits with source and year."""
    # Real code would call your search backend. A fixed fixture keeps the
    # transcript readable and the run reproducible.
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    # The script stands in for the model's decisions: turn 1 calls the
    # tool, turn 2 answers with what came back.
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
        Turn(content="Octopuses use tools and distribute cognition across their arms."),
    ])
    ctx = make_test_ctx(llm=llm)

    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt="Research the question with `search`, then answer in one sentence.",
        cognition=ReActCognition(tools=[search]),
    )
    result = await agent.run("What do we know about octopus cognition?", ctx)
    print(result.output)
    print(f"{result.usage.input_tokens} in / {result.usage.output_tokens} out "
          f"over 2 chat calls")


if __name__ == "__main__":
    asyncio.run(main())
```

```text
Octopuses use tools and distribute cognition across their arms.
20 in / 10 out over 2 chat calls
```

**What just happened.** `@tool(...)` wraps a plain async function into a
`FunctionTool`, deriving the JSON schema the model sees from the
signature and the docstring.

Two things about that decorator bite people once each:

- **`side_effecting=` is required.** Omit it and you get a
  `ToolDefinitionError` at import time, not a surprise in production.
  The framework needs to know whether a tool changes the world in order
  to decide whether human approval applies (step 5) and whether a retry
  is safe.
- **The docstring is the model's only description of the tool**, so it
  has to be at least 30 characters. A tool called `search` documented as
  `"Search."` raises `ToolDefinitionError` too.

`ReActCognition(tools=[...])` takes a plain list and wraps it in a
`ToolRegistry`. The `Agent` itself never speaks to a provider: the
cognition goes through `ctx.invoker`, which walks the middleware chain
and lands on whatever `LLMPort` is wired underneath — which is why
swapping `FakeLLM` for a real provider in step 6 changes nothing here.

**The problem this leaves.** The run is a black box. You called `run`
and waited; if the model had taken forty seconds and six tool calls,
you would have had nothing to show anyone.

## Step 3 — watch it work

`agent.run(...)` is `agent.stream(...)` collected. Iterate the stream
instead and you see every step as it happens.

```python
import asyncio

from agentkit import Agent, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the literature for `query`. Returns bulleted hits with source and year."""
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
        Turn(content="Octopuses use tools and distribute cognition across their arms."),
    ])
    ctx = make_test_ctx(llm=llm)

    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt="Research the question with `search`, then answer in one sentence.",
        cognition=ReActCognition(tools=[search]),
    )

    async for ev in agent.stream("What do we know about octopus cognition?", ctx):
        if ev.type == "message_delta":
            print(ev.text, end="", flush=True)
        elif ev.type == "tool_call":
            print(f"[tool_call] {ev.tool_call.name}({dict(ev.tool_call.arguments)})")
        elif ev.type == "tool_result":
            print(f"[tool_result] {ev.tool_result!r}")
        elif ev.type == "step":
            print(f"[step] {ev.text}")
        elif ev.type == "final":
            print(f"\n[final] {ev.result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
```

```text
[tool_call] search({'query': 'octopus cognition'})
[tool_result] "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n- 'Distributed cognition in cephalopods' (science.org, 2023)"
[step] iteration:1
Octopuses use tools and distribute cognition across their arms.
[final] Usage(input_tokens=20, output_tokens=10, cost_usd=0.0002, cache_read_tokens=0, cache_write_tokens=0)
```

**What just happened.** `StreamEvent` is a closed union — its `type` is
a `Literal["message_delta", "tool_call", "tool_result", "step",
"interrupt", "final"]` — so `mypy --strict` will tell you when you have
missed a branch. The six kinds:

| Event | Fires when | Read |
|---|---|---|
| `message_delta` | the assistant emits a text fragment | `ev.text` |
| `tool_call` | the model asks for a tool, *before* it runs | `ev.tool_call` (a `ToolCall`) |
| `tool_result` | the tool returned, or an error string the model will see | `ev.tool_result` |
| `step` | an iteration boundary | `ev.text`, e.g. `"iteration:1"` |
| `interrupt` | the loop is about to suspend for a human (step 5) | `ev.tool_call` |
| `final` | exactly once, at the end | `ev.result` (an `AgentResult`) |

The `Agent` does not open a WebSocket, write a log line, or render
anything. It hands you events; where they go is your decision.

**The problem this leaves.** You can now watch the run — including
watching it spend your money. Nothing here stops it.

<a id="step-4-cap-the-run-with-a-budget"></a>

## Step 4 — stop it before it costs too much

A `Budget` is the run's ceiling: cost, call count, depth, concurrency.
Put one on the `RunContext` and put `meter()` in the chat chain.

!!! warning "A `Budget` with no `meter()` is decoration"
    The `Budget` object holds the ceiling; the `meter()` middleware is
    what reads usage off each call, charges the budget, and raises. Wire
    the budget and forget the middleware and you get a run that reports
    `$0.00 spent` while your provider bills you in full. This is the
    single most common wiring mistake with this framework.

```python
import asyncio

from agentkit import Agent, Budget, MeterExceeded, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.middlewares import meter
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the literature for `query`. Returns bulleted hits with source and year."""
    return "- 'Distributed cognition in cephalopods' (science.org, 2023)"


def scripted_llm() -> FakeLLM:
    """A fresh two-turn script — a FakeLLM's script is consumed as it runs."""
    return FakeLLM.script([
        Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
        Turn(content="Octopuses distribute cognition across their arms."),
    ])


AGENT = Agent(
    name="briefer",
    model="claude-sonnet-4-6",
    prompt="Research the question with `search`, then answer in one sentence.",
    cognition=ReActCognition(tools=[search]),
)
QUESTION = "What do we know about octopus cognition?"


async def main() -> None:
    # A ceiling the run fits inside: it finishes, and the bill is readable.
    ctx = make_test_ctx(
        llm=scripted_llm(),
        budget=Budget(max_cost_usd=5.0),
        chat_middleware=[meter()],
    )
    result = await AGENT.run(QUESTION, ctx)
    print(f"finished: {result.output}")
    print(f"  spent ${ctx.budget.spent_usd:.4f} over {ctx.budget.calls} calls\n")

    # A ceiling it does not fit inside: the tool loop needs two chat calls.
    tight = make_test_ctx(
        llm=scripted_llm(),
        budget=Budget(max_calls=1),
        chat_middleware=[meter()],
    )
    try:
        await AGENT.run(QUESTION, tight)
    except MeterExceeded as exc:
        print(f"halted: {exc}")
    print(f"  spent ${tight.budget.spent_usd:.4f} over {tight.budget.calls} calls")


if __name__ == "__main__":
    asyncio.run(main())
```

```text
finished: Octopuses distribute cognition across their arms.
  spent $0.0002 over 2 calls

halted: calls 2 > 1
  spent $0.0002 over 2 calls
```

**What just happened.** One `Budget` per run. `meter()` calls
`budget.guard` before each chat call and `budget.charge` after it. Both
take the budget's async lock, so totals are invariant under concurrent
workers — a coordinator fanning ten children out on the same `Budget`
cannot race past its ceiling.

Read the second output block carefully, because the semantics are
deliberate and slightly surprising:

- **The call that crosses the ceiling still ran, and is still charged.**
  `charge` updates the books *before* it evaluates the verdict, because
  the spend genuinely happened and a ledger that hides it is a ledger
  that lies. What the budget guarantees is that no *further* call
  happens.
- **Comparison is strict `>`.** A call that lands exactly on the ceiling
  is fine; the next one trips.
- **`MeterExceeded` propagates.** `ReActCognition` does not swallow it.
  For a run that must not overspend, unwinding is the correct behaviour.

Middleware order is a plain list, so it is yours to decide. In a real
chain you would write `chat_middleware=[tracing(), meter(), retry()]` —
`meter` above `retry` so every retried attempt is charged, because your
provider counts retries and your budget had better too.

**The problem this leaves.** The agent's one tool only reads. The moment
you give it a tool that changes something — publishes, deploys, emails,
deletes — a capped budget is no comfort at all.

<a id="step-5-pause-for-human-approval-before-publishing"></a>

## Step 5 — make it ask before it acts

Mark the dangerous tool `side_effecting=True`, set the run's autonomy to
`"gated"`, and give the context a `Checkpointer`. The loop now suspends
before dispatching that tool and hands the decision back to you.

```python
import asyncio

from agentkit import Agent, Suspended, ToolCall, tool
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the literature for `query`. Returns bulleted hits with source and year."""
    return "- 'Distributed cognition in cephalopods' (science.org, 2023)"


@tool(side_effecting=True)
async def publish_brief(title: str, body: str) -> str:
    """Publish the finished brief to the team wiki. Idempotent per `title`."""
    return f"published {title!r} ({len(body)} chars)"


async def main() -> None:
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus cognition"}),)),
        Turn(tool_calls=(ToolCall("c2", "publish_brief",
                                  {"title": "Octopus cognition",
                                   "body": "Distributed across the arms."}),)),
        Turn(content="Published the brief."),
    ])
    # Suspend/resume needs somewhere to put the snapshot. In-memory is
    # right for a tutorial; Postgres is right for a worker that restarts.
    ctx = make_test_ctx(
        llm=llm,
        checkpointer=Checkpointer(port=InMemoryCheckpointStore()),
        correlation_id="run-42",
        autonomy="gated",
    )

    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt=(
            "Research the topic with `search`, then call `publish_brief` "
            "with a short title and body."
        ),
        cognition=ReActCognition(tools=[search, publish_brief]),
    )

    result = await agent.run("Brief the team on octopus cognition, then publish.", ctx)
    susp = result.evals.get("suspended")
    if not isinstance(susp, Suspended):
        raise SystemExit("expected the run to suspend for approval")

    print(f"partial={result.partial}  awaiting: {[tc.name for tc in susp.pending]}")
    for tc in susp.pending:
        print(f"  {tc.id}: {tc.name}({dict(tc.arguments)})")

    # A real driver shows this to a human and collects a decision per
    # ToolCall.id. Here we approve everything.
    decisions = {tc.id: "approve" for tc in susp.pending}
    final = await agent.resume(susp.run_id, decisions, ctx)
    print(f"resumed: {final.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

```text
partial=True  awaiting: ['publish_brief']
  c2: publish_brief({'title': 'Octopus cognition', 'body': 'Distributed across the arms.'})
resumed: 'Published the brief.'
```

**What just happened.** `autonomy="gated"` means "gate every key step",
and a key step is any tool declared `side_effecting=True`. Note that
`search` ran without asking: the gate is per-tool, not per-run.

When the loop reaches a gated call it snapshots its state through the
`Checkpointer`, emits an `interrupt` event per pending call, and returns
an `AgentResult` with `partial=True` and a `Suspended` in
`evals["suspended"]`. That `Suspended` carries the `run_id` and a frozen
tuple of `ToolCall`s waiting on you.

`agent.resume(run_id, decisions, ctx)` loads the snapshot back through
the same checkpointer, applies your per-call decisions, appends the
resulting tool results to the transcript, and drives the loop to a final
answer. A decision is `"approve"`, `"reject"` (or `"deny"`), or any
other string, which is parsed as a JSON override of the tool's
arguments — that last form is how you let a human edit the call rather
than just veto it.

Because the state is in the store and not in this process's memory, the
resume does not have to happen here. A different worker, minutes later,
can resume the same `run_id` — that is the same machinery as
crash-recovery. See the
[resume-after-crash recipe](recipes/resume-after-crash.md) for the
worker-restart shape, and swap `InMemoryCheckpointStore` for
`PostgresCheckpointStore` (extra: `arc-agentkit[postgres]`) when you do.

!!! note "`resume()` is a `ReActCognition` feature"
    Calling `agent.resume(...)` on an agent whose cognition is
    `SingleCallCognition`, `CoordinatorCognition` or
    `ClaudeCliCognition` raises `RuntimeError`. Suspend/resume state
    lives in the tool-loop cognition; human-in-the-loop for the other
    shapes belongs at the workflow layer.

**The problem this leaves.** Everything so far is a fake model reading
from a script.

## Step 6 — point it at a real model

Nothing about the agent, the tools, the budget or the gate changes. Only
the construction of `ctx` does: instead of `make_test_ctx(llm=FakeLLM(…))`
you build `Services` around a real provider yourself.

!!! warning "This step bills you"
    Needs `pip install "arc-agentkit[http]"` and `ANTHROPIC_API_KEY` in
    your environment. Everything above this line is free.

```python
import asyncio
import os

from agentkit import Agent, Budget, Scope, tool
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.adapters.llm import providers
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Invoker, RunContext, Services


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the literature for `query`. Returns bulleted hits with source and year."""
    return "- 'Distributed cognition in cephalopods' (science.org, 2023)"


@tool(side_effecting=True)
async def publish_brief(title: str, body: str) -> str:
    """Publish the finished brief to the team wiki. Idempotent per `title`."""
    return f"published {title!r} ({len(body)} chars)"


async def main() -> None:
    llm = providers.claude(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )
    services = Services(
        invoker=Invoker(llm=llm, chat_middleware=[tracing(), meter(), retry()]),
        checkpointer=Checkpointer(port=InMemoryCheckpointStore()),
    )
    ctx = RunContext(
        correlation_id="run-42",
        scope=Scope(),
        budget=Budget(max_cost_usd=0.50),
        services=services,
        autonomy="gated",
    )

    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt=(
            "Research the topic with `search`, then call `publish_brief` "
            "with a short title and body."
        ),
        cognition=ReActCognition(tools=[search, publish_brief]),
    )
    async for ev in agent.stream("Brief the team on octopus cognition, then publish.", ctx):
        if ev.type == "message_delta":
            print(ev.text, end="", flush=True)
        elif ev.type == "interrupt":
            print(f"\n[approval needed] {ev.tool_call.name}")
        elif ev.type == "final":
            print(f"\n[final] partial={ev.result.partial} "
                  f"spent=${ctx.budget.spent_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

Swap `providers.claude` for `providers.openai`, `providers.deepseek` or
`providers.openrouter` (and the matching environment variable and model
name) and nothing else in the script moves. That is the whole point of
keeping the LLM on the `Invoker` rather than on the `Agent`.

To run with no key at all against your local Claude Code install, use
`ClaudeCliCognition` instead — see
[Getting started](getting-started.md#the-same-agent-against-a-real-provider)
and the [`claude` CLI recipe](recipes/claude-cli-fastapi-code-gen.md).

## Where to go next

- **[Anti-patterns](anti-patterns.md)** — the traps every first-time
  user falls into. Read this once before you deploy what you just
  built; it is the shortest path to not repeating someone else's bill.
- **[Recipes](recipes/index.md)** — durable resume after a crash,
  per-tenant `Quota`, parallel agents with cancellation, writing your
  own middleware, consuming MCP servers.
- **[Concepts](concepts/kernel.md)** — the mental model behind each
  primitive used here. [Runtime](concepts/runtime.md) for `RunContext`
  and `Budget`; [Agents](concepts/agents.md) for `Cognition` and
  `Autonomy`.
- **[Mental models](mental-models/README.md)** — four worked product
  scenarios (multi-tenant RAG, an autonomous investigator, a 10k-row
  batch job, a multi-agent research team) that put these primitives
  under load and say what breaks when each guarantee slips.
- **[Cheatsheet](cheatsheet.md)** — every primitive, tight code,
  skimmable in 90 seconds.
