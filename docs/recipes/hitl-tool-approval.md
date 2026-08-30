# How do I pause a tool for human approval?

Some actions should not happen because a language model decided they
should. This is how you make an agent stop and wait for a person to say
yes — including across a process restart.

!!! tip "Need a *value*, not a yes/no?"
    This page is approve/deny with a serialisable run — the simplest
    correct thing, and unchanged. If you need a person to **supply a
    value** (a one-time code), to **park in place** because your
    coroutine holds live unserialisable state, or to put a **deadline**
    on the wait, see
    [Elicit a value from a human](elicit-a-value-from-a-human.md).
    That path is a superset and does not change this one.

## When you'd want this

Any tool that mutates the world — publishing content, spending money,
sending a message, filing a ticket, running a shell command — should
not run without a human saying yes when the run is being watched. The
run should **suspend to a checkpoint** and hand control back to your
driver code, which decides whether to resume with `"approve"`,
`"reject"`, or a modified argument list.

!!! note "Assumes `ANTHROPIC_API_KEY` in the environment"
    The snippet wires `providers.claude(...)` through an `Invoker` so
    the reader sees the real shape they'll ship. Swap
    `providers.claude` for `providers.openai` (and set
    `OPENAI_API_KEY`) if that's what you have — nothing else changes.

    To run it with **no key at all**, replace the `providers.claude(...)`
    call with a scripted `FakeLLM` from `agentkit.testing`:

    ```python
    from agentkit.kernel.types import ToolCall
    from agentkit.testing import FakeLLM, Turn

    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="c1", name="send_email", arguments={"to": "team@example.com", "subject": "Octopus brief"}),)),
        Turn(content="Sent."),
    ])
    ```

    Every other line is identical — that substitution is how this snippet
    is verified.

## Working code

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


@tool(side_effecting=True)
async def send_email(to: str, subject: str) -> str:
    """Send an email. Side-effecting — always gated under Autonomy.GATED."""
    return f"sent to {to!r}: {subject!r}"


async def main() -> None:
    llm = providers.claude(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )
    services = Services(
        invoker=Invoker(llm=llm),
        checkpointer=Checkpointer(port=InMemoryCheckpointStore()),
    )
    ctx = RunContext(
        correlation_id="run-1",
        scope=Scope(),
        services=services,
        autonomy="gated",  # gate every side_effecting tool
    )
    agent = Agent(
        name="notifier",
        model="claude-sonnet-4-6",
        prompt="Draft a one-line brief for the team, then call send_email exactly once.",
        cognition=ReActCognition(tools=[send_email]),
    )

    # First run — the loop suspends before send_email fires.
    result = await agent.run("Send the team a brief about octopus cognition.", ctx)
    susp = result.evals.get("suspended")
    assert isinstance(susp, Suspended)
    print(f"awaiting approval for: {[tc.name for tc in susp.pending]}")

    # Driver decides. Decisions are keyed by ToolCall.id:
    #   "approve"          — run with the model's args
    #   "reject" / "deny"  — inject a DENIED tool result, keep the loop going
    #   any other string   — parsed as a JSON args override, then run
    decisions = {tc.id: "approve" for tc in susp.pending}
    final = await agent.resume(susp.run_id, decisions, ctx)
    print(f"final: {final.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

## How it works

The run does not sit and wait for the human. That is the part worth
getting straight before the details.

When the agent reaches a tool it is not allowed to run unattended, it
saves its state, **ends**, and hands you back a result that says
"suspended" along with the list of calls awaiting a decision. Your
process is now free. The user might answer in ten seconds or tomorrow
morning; either way you call `resume` with their decisions and the run
continues from where it stopped — possibly in a different process
entirely.

That is why approval here costs you no open connections and no parked
coroutines, and why it survives a deploy in the middle of someone's
lunch break.

`Autonomy` is a run-wide tier carried on `RunContext.autonomy`.
`should_gate(autonomy, requires_approval, key_step)` is the shared
policy every gating surface consults:

| Autonomy   | Gates                                                        |
|------------|--------------------------------------------------------------|
| `"auto"`   | Only tools explicitly marked `requires_approval=True`        |
| `"gated"`  | Also every `side_effecting=True` tool (key step)             |
| `"manual"` | Every tool call                                              |

When the `ReActCognition` sees a gated tool call, it snapshots its
state through the `Checkpointer` (status `"suspended"`), emits an
`interrupt` event per pending tool call, and returns an `AgentResult`
with `partial=True` and a `Suspended` value in
`evals["suspended"]`. `Suspended.pending` is a **frozen tuple** of
`ToolCall`s — the operator UI reads it and the resume path threads it
back verbatim, so a mutable list can't desync the two ends of the
handshake.

`agent.resume(run_id, decisions, ctx)` loads the snapshot, applies
your per-call decisions, appends the tool results to the transcript,
and drives the loop to a final answer. You can call it from an
entirely fresh process — that's the point of the checkpointer.

## What bites people

- **`side_effecting=` is required on every `@tool`.** The framework
  fails at decoration time (not at call time) if you forget — because
  the gating and idempotency primitives cannot guess.
- **You must wire a checkpointer for `resume(...)` to work.**
  Without one, the loop still emits `Suspended`, but the state isn't
  persisted, so `resume` raises `ValueError("no suspended run … to
  resume")`. `InMemoryCheckpointStore` is fine for tests; production
  wants `PostgresCheckpointStore` (extra: `arc-agentkit[postgres]`).
- **`agent.resume(...)` only works on a `ReActCognition`.** Calling it
  on a `SingleCallCognition` or a coordinator is a contract violation
  and raises `RuntimeError` explicitly.
- **A `Suspended` result has `partial=True` and an empty
  `output`.** The final text comes from `resume(...)`, not the
  original `run(...)`.

## Related

- [Tutorial · Step 5](../tutorial.md#step-5-make-it-ask-before-it-acts)
  — the same primitive introduced inside a full walkthrough.
- [Resume after a crash](resume-after-crash.md) — the checkpointer's
  other job.
- [Concepts · Agents](../concepts/agents.md) — the mental model of
  `Cognition` and the control primitives.
