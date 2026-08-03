# How do I pause a tool for human approval?

## When you'd want this

Any tool that mutates the world — publishing content, spending money,
sending a message, filing a ticket, running a shell command — should
not run without a human saying yes when the run is being watched. The
run should **suspend to a checkpoint** and hand control back to your
driver code, which decides whether to resume with `"approve"`,
`"reject"`, or a modified argument list.

## Working code

```python
import asyncio

from agentkit import Agent, Suspended, ToolCall, tool
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=True)
async def send_email(to: str, subject: str) -> str:
    """Send an email. Side-effecting — always gated under Autonomy.GATED."""
    return f"sent to {to!r}: {subject!r}"


async def main() -> None:
    llm = FakeLLM.script(
        [
            Turn(
                tool_calls=(
                    ToolCall("c1", "send_email", {"to": "team@x", "subject": "brief"}),
                )
            ),
            Turn(content="Email sent."),
        ]
    )
    ctx = make_test_ctx(
        llm=llm,
        checkpointer=Checkpointer(port=InMemoryCheckpointStore()),
        autonomy="gated",  # gate every side_effecting tool
        correlation_id="run-1",
    )
    agent = Agent(
        name="notifier",
        model="gpt-4o-mini",
        prompt="Draft and send.",
        cognition=ReActCognition(tools=[send_email]),
    )

    # First run — the loop suspends before send_email fires.
    result = await agent.run("Send the team a brief.", ctx)
    susp = result.evals["suspended"]
    assert isinstance(susp, Suspended)
    print(f"awaiting approval for: {[tc.name for tc in susp.pending]}")

    # Driver decides. Decisions are keyed by ToolCall.id:
    #   "approve"          — run with the model's args
    #   "reject" / "deny"  — inject a DENIED tool result, keep the loop going
    #   any other string   — parsed as a JSON args override, then run
    decisions = {tc.id: "approve" for tc in susp.pending}
    final = await agent.resume(susp.run_id, decisions, ctx)
    print(f"final: {final.output!r}")


asyncio.run(main())
```

## How it works

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

## Gotchas

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

- [Tutorial · Step 5](../tutorial.md#step-5-pause-for-human-approval-before-publishing)
  — the same primitive introduced inside a full walkthrough.
- [Resume after a crash](resume-after-crash.md) — the checkpointer's
  other job.
- [Concepts · Agents](../concepts/agents.md) — the mental model of
  `Cognition` and the control primitives.
