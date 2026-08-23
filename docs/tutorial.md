# Tutorial: build a research-briefing agent

Fifteen minutes, five steps. At the end you'll have an agent that takes a
question, searches for sources, drafts a brief, streams progress to the
caller, refuses to overspend, and pauses for a human before it publishes.

Every step is a complete, self-contained script — copy the whole block and
run it. Each step introduces exactly one primitive on top of the previous.

## Before you start

Install the package:

```bash
pip install "arc-agentkit[http]"
```

The `[http]` extra pulls `httpx` in — that's what the bundled provider
adapters (`claude()`, `openai()`, `deepseek()`, `openrouter()`) use.

!!! note "Which LLM plug-in each step uses"
    **Step 1** delegates the loop to a locally-installed `claude` CLI via
    `ClaudeCliCognition` — zero API keys, the CLI's own auth (from a
    prior `claude login`) is used. Install the CLI from the
    [Claude Code docs](https://docs.claude.com/en/docs/claude-code)
    if you don't have it.

    **Steps 2–5** switch to `providers.claude(api_key=...)` and expect
    `ANTHROPIC_API_KEY` in the environment. The point of these steps is
    the agentkit tool loop (`ReActCognition`), the budget, and the
    approval gate — primitives that live *around* the LLM plug-in, so a
    real HTTP LLM is what you want. Swap in
    `providers.openai(api_key=...)` (and set `OPENAI_API_KEY`) if that's
    what you have — the rest of the code doesn't change.

## Step 1 — a single-turn agent

The smallest thing that runs. An `Agent`, a `RunContext`, and one call.
`ClaudeCliCognition` is the cognition plug-in; the default
`SingleCallCognition` isn't used because the CLI cognition subsumes it
(one CLI invocation per `agent.run(...)`).

```python
"""Requires the `claude` CLI on PATH and one prior `claude login`.
Zero API keys — the CLI's own auth is used."""

import asyncio

from agentkit import Agent, Scope
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.runtime import RunContext, Services


async def main() -> None:
    agent = Agent(
        name="briefer",
        prompt="Answer in one short sentence.",
        cognition=ClaudeCliCognition(model="claude-sonnet-4-6"),
    )
    ctx = RunContext(correlation_id="run-1", scope=Scope(), services=Services())

    result = await agent.run("What do we know about octopus cognition?", ctx)
    print(result.output)
    print(f"cost estimate: ${result.usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

**What just happened.** `Agent` is identity + chat-call config — nothing
more. The `cognition=` field is the Strategy plug-in that owns the loop;
`ClaudeCliCognition` subprocesses the local `claude` CLI once per
`agent.run(...)` and maps its stream-JSON output to agentkit
`StreamEvent`s. `AgentResult` is typed — `output` is the model text,
`usage` is the accrued `Usage` for the whole run.
`Usage.cost_usd` here is the CLI's own estimate (from published
per-token prices), surfaced as `cost estimate:` rather than a billed
number.

## Step 2 — add a tool

To do more than one chat call — or to teach the model to call your own
functions — swap the cognition for `ReActCognition` and register a tool.
This is also the point where we switch to a plug-in that gives agentkit
its own tool loop: `providers.claude(...)` returns an `LLMPort` we hand
to an `Invoker`. (`ClaudeCliCognition` has its own tool loop inside the
CLI — the point of this step is to teach *agentkit's* loop.)

```python
"""Requires ANTHROPIC_API_KEY in the environment.
Swap `providers.claude` for `providers.openai` + OPENAI_API_KEY if that
is what you have — nothing else in the script changes."""

import asyncio
import os

from agentkit import Agent, Scope, tool
from agentkit.adapters.llm import providers
from agentkit.agents.cognition import ReActCognition
from agentkit.runtime import Invoker, RunContext, Services


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns two bulleted hits with source and year."""
    # In real code this would call your search backend. For a self-contained
    # tutorial, we return a deterministic fixture so the tool result is
    # readable in the transcript.
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    llm = providers.claude(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )
    services = Services(invoker=Invoker(llm=llm))
    ctx = RunContext(correlation_id="run-1", scope=Scope(), services=services)

    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt="Research the question. Use `search` to find sources before answering.",
        cognition=ReActCognition(tools=[search]),
    )
    result = await agent.run("What do we know about octopus cognition?", ctx)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
```

**What just happened.** `@tool(side_effecting=False)` wraps a plain async
function into a `FunctionTool`. The `side_effecting=` keyword is
required — the framework uses it to decide whether human approval is
needed (see step 5) and whether a retry is safe. Forgetting it raises
`ToolDefinitionError` at decoration time, not at call time.

`ReActCognition` owns the tool loop: chat → tool_call → tool_result →
chat → final. A plain `list[Tool]` is auto-wrapped into a
`ToolRegistry`. The LLM plug-in lives on the `Invoker` inside `Services`
— the `Agent` never speaks to a provider directly; it goes through
`ctx.invoker`, which walks the middleware chain (empty here) and lands
on the `LLMPort`. Swap `providers.claude` for `providers.openai`,
`providers.deepseek`, or `providers.openrouter` without editing the
agent or the cognition — that's the whole point of the split.

## Step 3 — stream the run

`agent.run(...)` is `agent.stream(...)` collected. Streaming lets you
show progress, log intermediate tool calls, or forward events over a
WebSocket without waiting for the whole run.

```python
"""Requires ANTHROPIC_API_KEY in the environment."""

import asyncio
import os

from agentkit import Agent, Scope, tool
from agentkit.adapters.llm import providers
from agentkit.agents.cognition import ReActCognition
from agentkit.runtime import Invoker, RunContext, Services


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns two bulleted hits with source and year."""
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    llm = providers.claude(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )
    services = Services(invoker=Invoker(llm=llm))
    ctx = RunContext(correlation_id="run-1", scope=Scope(), services=services)

    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
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


if __name__ == "__main__":
    asyncio.run(main())
```

**What just happened.** `StreamEvent` is a closed union
(`Literal["message_delta", "tool_call", "tool_result", "step",
"interrupt", "final"]`) so `mypy --strict` exhausts the branches.
Day-to-day event shapes:

- `message_delta` — one incremental chunk of the assistant's text
  (`ev.text` is the fragment)
- `tool_call` — the model asked to run a tool (fires *before* the tool
  runs; `ev.tool_call` is a `ToolCall`)
- `tool_result` — the tool returned (or an error string the model can see)
- `step` — an iteration boundary (`ev.text` is e.g. `"iteration:1"`)
- `interrupt` — the loop is about to suspend for human approval (step 5)
- `final` — exactly one terminal event carrying the `AgentResult` on
  `ev.result`

The `Agent` never opens a WebSocket for you. It hands you events; you
decide where they go.

## Step 4 — cap the run with a budget

A `Budget` puts a hard ceiling on cost, call count, depth, and
concurrency for one run. Wire the `meter()` middleware into the chat
chain and the invoker charges the budget under an async lock — overspend
raises `MeterExceeded` (from `agentkit.runtime.meter`) and the loop
halts cleanly.

```python
"""Requires ANTHROPIC_API_KEY in the environment.

Sizing a budget in a tutorial is inherently model-dependent — we set a
generous $0.10 ceiling here so the run completes, and demonstrate the
halt path with `max_calls=1`, which reliably trips on the second chat
call (search → chat, or the parse-and-repair path)."""

import asyncio
import os

from agentkit import Agent, Budget, MeterExceeded, Scope, tool
from agentkit.adapters.llm import providers
from agentkit.agents.cognition import ReActCognition
from agentkit.middlewares import meter
from agentkit.runtime import Invoker, RunContext, Services


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns two bulleted hits with source and year."""
    return (
        "- 'Octopus vulgaris uses coconut shells as shelter' (nature.com, 2009)\n"
        "- 'Distributed cognition in cephalopods' (science.org, 2023)"
    )


async def main() -> None:
    llm = providers.claude(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )
    services = Services(invoker=Invoker(llm=llm, chat_middleware=[meter()]))
    ctx = RunContext(
        correlation_id="run-1",
        scope=Scope(),
        budget=Budget(max_cost_usd=0.10, max_calls=1),  # halts after the first chat call
        services=services,
    )
    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt="Research the question. Use `search` before answering.",
        cognition=ReActCognition(tools=[search]),
    )

    try:
        await agent.run("What do we know about octopus cognition?", ctx)
    except MeterExceeded as exc:
        print(f"halted: {exc}")
    print(f"spent: ${ctx.budget.spent_usd:.4f}  calls: {ctx.budget.calls}")


if __name__ == "__main__":
    asyncio.run(main())
```

**What just happened.** `Budget` is one instance per run; every chat
call goes through `meter()`, which calls `budget.guard` before the work
and `budget.charge` after. Both take the budget's async lock — totals
are invariant under concurrent workers, so you can fan out sub-agents
without their spends racing. `Budget._verdict` uses strict greater-than
(`spent > max`), so a call that lands exactly on the ceiling completes
and the *next* one trips `MeterExceeded`. `ReActCognition` lets the
exception propagate, which is the correct behaviour for a run that
must not overspend.

The middleware order is deterministic. In a real chain you'd write
`chat_middleware=[tracing(), meter(), retry()]` — `meter` above `retry`
so every retried attempt gets charged (your provider bill counts
retries; your budget had better too).

## Step 5 — pause for human approval before publishing

Some tools should not run without a human saying yes. Declare the tool
`side_effecting=True`, set the run's `autonomy` to `"gated"`, and the
loop will suspend before dispatching the tool. Your driver decides
whether to `resume(...)` and with what per-call decisions.

```python
"""Requires ANTHROPIC_API_KEY in the environment."""

import asyncio
import os

from agentkit import Agent, Scope, Suspended, tool
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.adapters.llm import providers
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.runtime import Invoker, RunContext, Services


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns one bulleted hit."""
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
    # A Checkpointer + a store is required for suspend/resume — the loop
    # snapshots its state to the store, and `agent.resume(run_id, ...)`
    # reads it back.
    services = Services(
        invoker=Invoker(llm=llm),
        checkpointer=Checkpointer(port=InMemoryCheckpointStore()),
    )
    ctx = RunContext(
        correlation_id="run-42",
        scope=Scope(),
        services=services,
        autonomy="gated",  # gates every @tool(side_effecting=True) call
    )
    agent = Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt=(
            "Research the topic with `search`, then call `publish_brief` "
            "with a short title and body. Do not skip the publish step."
        ),
        cognition=ReActCognition(tools=[search, publish_brief]),
    )

    result = await agent.run("Brief the team on octopus cognition, then publish.", ctx)
    susp = result.evals.get("suspended")
    if not isinstance(susp, Suspended):
        raise SystemExit("expected the run to suspend for approval")
    print(f"[suspended] pending: {[tc.name for tc in susp.pending]}")

    # A real driver renders the pending tool calls to a human and
    # collects approve/reject decisions per ToolCall.id. Here we
    # approve unconditionally.
    decisions = {tc.id: "approve" for tc in susp.pending}
    final = await agent.resume(susp.run_id, decisions, ctx)
    print(f"[resumed] {final.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

**What just happened.** `autonomy="gated"` is the run-wide tier that
says "gate every key step". A key step is any tool where
`side_effecting=True`. When the loop encounters one, it snapshots its
state through the `Checkpointer`, emits an `interrupt` event per
pending tool call, and returns an `AgentResult` with `partial=True` and
a `Suspended` in `evals["suspended"]`. That `Suspended` carries the
`run_id` and a frozen tuple of `ToolCall`s awaiting decisions.

`agent.resume(run_id, decisions, ctx)` loads the snapshot back through
the same checkpointer, applies your per-call decisions (`"approve"`,
`"reject"` / `"deny"`, or any other string parsed as a JSON args
override), appends the tool results to the transcript, and drives the
loop to a final answer. Because the state lives in the store, a
completely fresh process can resume — see the
[resume-after-crash recipe](recipes/resume-after-crash.md) for the
worker-restart pattern. In production, swap `InMemoryCheckpointStore`
for `PostgresCheckpointStore` (extra: `arc-agentkit[postgres]`).

`agent.resume(...)` only works on a `ReActCognition` agent; calling it
on `SingleCallCognition`, `CoordinatorCognition`, or
`ClaudeCliCognition` raises `RuntimeError` — suspend/resume state lives
in the tool-loop cognition. HITL for other cognitions belongs at the
workflow layer.

## Where to go next

- **[Concepts](concepts/kernel.md)** — the mental model of each
  primitive introduced here. Start with `Runtime` for `RunContext` /
  `Budget` and `Agents` for `Cognition` / `Autonomy`.
- **[Recipes](recipes/index.md)** — focused answers to "how do I X?",
  including durable resume after a crash, per-tenant `Quota`, parallel
  agents with cancellation, and writing your own middleware.
- **[Cheatsheet](cheatsheet.md)** — every primitive, tight code,
  skimmable in 90 seconds.
- **[Anti-patterns](anti-patterns.md)** — the fifteen traps every
  first-time user falls into. Read this once before you deploy the
  agent you just built.
