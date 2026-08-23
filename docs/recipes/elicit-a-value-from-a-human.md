# How do I pause a run to ask a person for a value?

Sometimes the agent is not asking for permission — it is asking for
information only a person has, like the code that was just texted to
them. This is how a run stops, asks, and carries on with the answer.

## When you'd want this

Approve/deny answers one question: *may this tool run?* It cannot
express the other one: *what is the code they just texted you?*

That second shape needs four things the approval gate doesn't have:

- **elicitation** — the run names what it needs, a person supplies a
  value, and there may be no tool call anywhere nearby;
- **parking in place** — your coroutine holds live, unserialisable
  state, so unwinding to a checkpoint and re-entering via `resume()`
  is impossible;
- **a deadline** — the abandoned-tab case, so a suspended run
  degrades instead of hanging forever;
- **a typed decision** — carrying who answered and when, because
  `dict[str, str]` carries neither.

If you only need approve/deny with a serialisable run, the
[tool-approval recipe](hitl-tool-approval.md) is simpler and still
correct. This page is the superset.

!!! note "Assumes `ANTHROPIC_API_KEY` in the environment"
    To run it with **no key at all**, replace
    `resolve_llm("claude-sonnet-4-6")` with a scripted `FakeLLM` from
    `agentkit.testing`:

    ```python
    from agentkit.kernel.types import ToolCall
    from agentkit.testing import FakeLLM, Turn

    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="c1", name="ask_human",
                                  arguments={"prompt": "one-time code?"}),)),
        Turn(content="Verified."),
    ])
    ```

    Either way the script pauses for you to type at the terminal — that
    is the whole point of the page.

## Working code

```python
"""Requires ANTHROPIC_API_KEY in the environment."""

import asyncio

from agentkit import Agent, Scope
from agentkit.adapters.llm import resolve_llm
from agentkit.agents.cognition import ReActCognition
from agentkit.agents.control.elicitation import Decision, Elicitation, ask_human_tool, elicit
from agentkit.runtime import Invoker, RunContext, Services


class TerminalAsker:
    """The whole HITL integration: one method.

    The runtime never learns whether this is a terminal, an HTTP round
    trip, a queue, or a Slack bot — it only awaits `ask`. Swap this for
    your transport and nothing else changes.
    """

    async def ask(self, request: Elicitation) -> Decision:
        answer = await asyncio.to_thread(input, f"{request.prompt} > ")
        if request.kind == "value":
            return Decision(kind="value", value=answer, actor="terminal-user")
        return Decision(
            kind="approve" if answer.strip().lower() in ("y", "yes") else "deny",
            actor="terminal-user",
        )


async def main() -> None:
    services = Services(
        invoker=Invoker(llm=resolve_llm("claude-sonnet-4-6")),
        asker=TerminalAsker(),  # <- the one seam that turns a suspend into a park
    )
    ctx = RunContext("run-1", Scope(), services=services, autonomy="gated")

    # (1) The MODEL asks, when it decides it needs something only a person has.
    agent = Agent(
        name="verifier",
        model="claude-sonnet-4-6",
        prompt="Verify the user. Ask them for the one-time code, then confirm.",
        cognition=ReActCognition(
            tools=[ask_human_tool(secret=True, deadline_s=120)],
            approval_deadline_s=120,
        ),
    )
    result = await agent.run("Verify me.", ctx)
    print(result.output, result.stop_reason)

    # (2) YOUR CODE asks, from anywhere — no tool call involved, any cognition.
    decision = await elicit(
        ctx,
        Elicitation(
            id="confirm-amount",
            prompt="Confirm the transfer amount",
            kind="value",
            deadline_s=60,
        ),
    )
    if decision.kind == "expired":
        print("nobody answered; degrading")
    else:
        print(f"{decision.actor} said {decision.value} at {decision.at}")


asyncio.run(main())
```

## Park, or suspend?

One flag decides, and it's whether an `Asker` is on the context.

|  | `Services(asker=...)` set | not set |
|---|---|---|
| Gated tool call | **Parks.** The loop awaits the person from inside its own coroutine. Nothing unwinds. | **Suspends.** Checkpoint written, `AgentResult(stop_reason="suspended")` returned, `Agent.resume(...)` continues it. |
| Live unserialisable state | Survives — it's the same objects | Lost — the loop is rebuilt from a snapshot |
| Needs a checkpointer | No | Yes |
| Survives a process restart | No | Yes |

Neither is better; they trade durability against the ability to hold
live state. The suspend path is **completely unchanged** — this is
additive.

## Deadlines: expiry is an outcome, not an error

`Elicitation.deadline_s` (or `ReActCognition(approval_deadline_s=...)`)
bounds the wait. When it passes, `elicit` returns
`Decision(kind="expired")` and the run **degrades and continues** —
the model sees an `EXPIRED:` tool message, distinct from `DENIED:`,
because "nobody was there" and "someone said no" call for different
operator responses.

For a caller that genuinely cannot continue, `elicit_or_raise` raises
`ElicitationExpired` instead.

On the suspend path there is no coroutine to time out, so the deadline
is **persisted with the checkpoint** as absolute wall-clock expiry, and
surfaced on `Suspended.deadline_at` for an operator UI to render a
countdown. A `resume()` arriving after it degrades the same way: every
pending call becomes `expired`, the loop continues, and the tool does
**not** run. Approving an hour late does not move the money.

!!! danger "Your `ask` must not block the event loop"
    Scheduling is cooperative. A synchronous wait — `input()`,
    `requests.get()`, `time.sleep()` — never yields, so the timeout
    coroutine never gets to run and `deadline_s` **silently becomes an
    unbounded hang**: exactly the failure the deadline exists to
    prevent. Wrap it:

    ```python
    answer = await asyncio.to_thread(input, f"{request.prompt} > ")
    ```

## Suspended is not failed

```python
result = await agent.run(task, ctx)

if result.is_suspended:      # stop_reason == "suspended" — waiting for a person
    park_in_operator_queue(result.evals["suspended"])
elif result.is_resumable:    # e.g. "budget_exhausted" — raise the ceiling and retry
    escalate_for_more_budget()
else:
    ship(result.output)
```

`AgentResult.stop_reason` is a closed `Literal`, so a typo type-checks
as an error rather than reading as `None` at 3am. A run that actually
**failed** normally raises and produces no `AgentResult` at all — that
is the distinction. The one deliberate exception is
`ClaudeCliCognition`, which reports the error as data so its
exactly-one-terminal-event contract survives a subprocess that never
starts; that run comes back with `stop_reason="failed"`.

Values: `complete`, `suspended`, `expired`, `budget_exhausted`,
`max_iterations`, `invalid_output`, `terminated`, `failed`. The
free-form detail string is still in `evals["stop_reason"]`.

## Secrets

Mark the request `secret=True` and three things happen:

1. The answer is wrapped in `SecretValue`, whose `repr` and `str` are
   both `'***'` — so it survives an f-string in a log line, an
   exception message, and a debugger pane. `reveal()` is the one
   explicit, greppable way out.
2. The prompt itself is redacted in observations ("enter the code we
   texted to +44…" is revealing too).
3. The working context is **tainted**, and a tainted context is never
   checkpointed again for the rest of the run.

That third one is a real trade, and it's the right one. A one-time
code injected as a tool message would otherwise be serialised into
Postgres, where it outlives by weeks the ten minutes it was valid
for. An un-resumable run can be re-run; a leaked credential cannot be
un-leaked.

## Gotchas

**The `RunPolicy` trifecta gate fires on `resume()` too.** Approving
one tool *call* is not approval of the capability *combination*, so a
deny-mode policy refuses a resume the same way it refuses a fresh run.

**No `Asker` wired + direct `elicit` = denial.** A gate with no human
attached must not silently pass. (The ReAct gate checks for an asker
first and falls back to suspend, so this only affects direct calls.)

**A missing entry in `resume(decisions)` defaults to deny.** An
operator who answered three of four gates has not implicitly approved
the fourth.

**`ask_human_tool` defaults to a 5-minute deadline**, not `None`. A
model-initiated ask is the most likely to hit an abandoned tab, and an
unbounded default would make the hang the easy path.

**`resume` accepts both shapes.** Typed `Decision` for the audit
trail; the legacy `"approve"` / `"reject"` / JSON-args strings are
coerced. Same call site.

## Related

- [Human-in-the-loop tool approval](hitl-tool-approval.md) — the
  simpler approve/deny path
- [Resume from a checkpoint after a crash](resume-after-crash.md)
- [Cap spend with Budget and Quota](spend-budget-and-quota.md) —
  `budget_exhausted` is the other resumable stop reason
