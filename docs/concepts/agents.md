# Agents

An agent is a wired composition: a prompt, a way of looping, a set of
tools, and optionally some memory. `agentkit.agents` is where "a thing
that talks to a model" becomes a thing you can plan, cancel, budget,
coordinate and hand off between.

!!! tip "Is this page for you?"

    **Reach for it when** you are building anything that loops: one
    agent with tools, a coordinator over children, or a fixed
    multi-step `Workflow`.

    **Skip it for now if** you only need a single model call — reach
    for `Chat` and come back later.

## The problem it solves

"Agent" is a loaded word, and most frameworks answer it with a base class
you subclass. That works until you want two things at once: a different
loop with the same prompt and tools, or several agents cooperating
without any of them knowing about the others. Subclassing gives you
neither — the loop is welded to the identity, and coordination means a
shared parent class everyone has to inherit.

agentkit splits them. The `Agent` holds identity and chat configuration;
a `Cognition` holds the *how* of iterating. Changing `SingleCallCognition`
to `ReActCognition` changes the loop and touches nothing else. A
coordinator drives child agents through a `Policy`, and the children do
not know they are being coordinated.

The rest of the package exists because multi-step, multi-agent runs fail
in specific, expensive ways: a run that spends $4 and returns nothing, a
run that parked on a human decision and reported success, a coordinator
whose child deleted its checkpoint on the way out. Every control
primitive here is a named answer to one of those.

## The smallest thing that works

A leaf agent with one tool, driven by `FakeLLM`, so it runs offline:

```python
import asyncio

from agentkit import Agent, ToolCall, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.testing import FakeLLM, Turn, make_test_ctx


@tool(side_effecting=False)
async def add(a: int, b: int) -> int:
    """Add two integers together and return their sum."""
    return a + b


async def main() -> None:
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="c1", name="add", arguments={"a": 6, "b": 7}),)),
        Turn(content="The answer is 13."),
    ])
    ctx = make_test_ctx(llm=llm)
    agent = Agent(
        name="calculator",
        model="fake-model",
        prompt="You answer arithmetic questions using the add tool.",
        cognition=ReActCognition(tools=[add]),
    )
    result = await agent.run("what is 6 + 7?", ctx)
    print(result.output)                          # The answer is 13.
    print(result.stop_reason, result.is_suspended)  # complete False


asyncio.run(main())
```

Swap `ReActCognition(tools=[add])` for the default `SingleCallCognition()`
and the same agent becomes one call and one answer. Nothing else moves.

## How it works — the mental model

```text
Agent  ── identity + chat config (name, model, prompt, output, memory)
  └─ Cognition  ── the loop  (single call | tool loop | coordinator | CLI)
       └─ Policy   ── for a coordinator: who speaks next
```

- **Tools live on the cognition, not on the `Agent`.** A tool set is part
  of a turn-taking regime, not of an identity.
- **Capabilities are not `Agent` kwargs either.** `Compactor`,
  `Guardrail`, `Checkpointer` and `Evaluator` plug in via a
  `RequestBuilder`, the middleware chain, or a policy. See
  [Capabilities](capabilities.md).
- **New shapes are new `Cognition` / `Policy` implementations.** There is
  no agent subclassing anywhere in the framework, and you should not add
  any.

`await agent.run(task, ctx)` collects the stream and returns an
`AgentResult`. `agent.stream(task, ctx)` yields `StreamEvent`s —
`message_delta` tokens, tool events, and a terminal `final`.

## The pieces

### `Agent`

```python
from agentkit import Agent, InMemoryFiles, Prompt, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.memory import FileMemory


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the corpus for passages matching the given query string."""
    return "…"


agent = Agent(
    name="researcher",
    model="claude-sonnet-4-6",
    cognition=ReActCognition(tools=[search]),
    prompt=Prompt(id="researcher", version="1.0.0", template="You research carefully."),
    memory=FileMemory(files=InMemoryFiles()),
)
print(agent.name, type(agent.cognition).__name__)   # researcher ReActCognition
```

Identity and chat configuration live here: `name`, `model`, `prompt` (a
`Prompt` or a plain `str`, wrapped into a one-off inline `Prompt` so
traces still attribute), `temperature`, `max_tokens`, `response_format`,
`output` / `parse` / `max_repairs` for typed output, `memory`, and the
`requires` / `min_context_window` capability contract.

Two checks run at construction, in `__post_init__`, and both exist
because catching a guaranteed failure *before* the run starts is the
whole value:

- `check_prompt()` — refuses a `Prompt` whose declared `inputs` are not
  all bound. See [Prompts](prompts.md#an-unbound-prompt-is-refused-at-construction).
- `check_capabilities()` — refuses a model that lacks something the job
  needs (see below).

`Agent` is a mutable dataclass, so assigning `agent.model` or
`agent.prompt` afterwards bypasses both. Re-assert by calling them.

### `Cognition`

A cognition decides **how many times to call the model, and what happens
between the calls.**

That sounds small, and it is the single biggest behavioural choice you
make. Asking a model one question and returning its answer is one
strategy. Letting it request a tool, running that tool, telling it what
came back, and asking again — until it stops asking — is a completely
different one. Running a team of child agents is a third.

Most frameworks bury that choice inside the agent class. agentkit makes
it a separate object you pass in, so you can change the strategy without
touching your prompt, your tools, or anything downstream.

Four ship in `agentkit.agents.cognition`:

- **`SingleCallCognition`** — ask the model once, return the answer.
  The default, and the right choice for a narrow single-shot skill.
- **`ReActCognition`** — the tool loop. The model either answers or asks
  for a tool; if it asks, the tool runs, the result goes back into the
  conversation, and the model is asked again. ("ReAct" is *Reason +
  Act*, from the 2022 paper that named the pattern.) This is the only
  cognition that can pause for a human and be resumed later via
  `agent.resume(...)`, and the only one that checks for cancellation
  between steps.
- **`CoordinatorCognition`** — runs several child agents, with a
  `Policy` deciding whose turn it is and a `SignalChannel` carrying
  messages between them.
- **`ClaudeCliCognition`** — hands the whole loop to a locally installed
  `claude` CLI, so you manage no API keys at all.
- **`CodexCliCognition`** — the same for a locally installed `codex` CLI.
  Deliberately parallel to the one above, and deliberately not identical:
  its containment is an OS sandbox rather than a tool list. See
  [The Codex CLI](codex-cli.md).

If a term above is new, the [glossary](../glossary.md) defines each one
in a sentence.

Those signals come in two families, and the split is enforced by the type
system rather than by convention: `ControlSignal` is parent → child
(cancel, retask, reduce budget, broadcast context) and `DataSignal` is
child → parent (progress, done, escalate, blocked). A parent cannot emit
a `DoneSignal` and a child cannot emit a `CancelSignal`, because they do
not share a base. Projects subclass whichever direction they need —
`ControlSignal` and `DataSignal` are the extension points — and the
framework dispatches on `isinstance`, so no discriminator field is
required. See [Concepts · Observability](observability.md) for how the
resulting stream is recorded.
- **`ClaudeCliCognition`** — delegates the loop to a locally-installed
  `claude` CLI, so no API key is handled on your server; the CLI's own
  auth is used. Emits the same `StreamEvent`s the others do.
- **`CodexCliCognition`** — the same arrangement for OpenAI's `codex`
  CLI: same `drive` contract, same terminal-event guarantee, same
  `spawn=` seam. Where the two binaries genuinely differ, so does the
  API — see [The Codex CLI](codex-cli.md) for the four places.

Import path: `from agentkit.agents.cognition import ReActCognition`.
Cognitions are deliberately **not** re-exported from the top-level
`agentkit` package — that surface is already dense, and cognitions are
one family among many.

Writing your own is implementing the Protocol: a small async iterator
over a `RunContext` and a task. No `Agent` subclassing.

### `Workflow`

A `Workflow` composes multiple `Agent`s under a `Policy`. It is what lets
you say "planner → many researchers in parallel → synthesizer → critic →
human checkpoint" without any agent knowing about the others. Its
`human_gate` node suspends the workflow — not an individual cognition —
with the same `Suspended` shape.

Explicit control where `Agent` + `Policy` is emergent: a `Workflow` is a
typed, developer-authored DAG.

### Control primitives

- **`SignalChannel`** — the frozen envelope multi-agent signals travel in
  (`ProgressSignal`, `DoneSignal`, `CancelSignal`, `EscalateSignal`, …).
  `Handoff` is separate: it is a routing verb consumed by
  `SelectorPolicy` / `route_by_handoff`, not a member of the
  progress/done family.

    `emit` dual-writes: to the parent's `merge_inbox` (the delivery path,
    where a slow parent applies real backpressure and the child waits)
    and to the channel's own `outbox` (the audit/replay tap, which
    nothing in the framework drains). The tap **never blocks** — it drops
    its oldest entry and counts it on `channel.dropped`. Awaiting a queue
    with no consumer is a deadlock rather than backpressure: before this,
    a run whose parent was keeping up perfectly still stopped dead on its
    257th signal.

- **`Handoff` + routing** — `route_by_handoff(default=...)` reads the
  `HANDOFF:<target>` marker off the last message and routes to that
  child. The target is **checked against the roster**: an invented name
  falls back to `default` (warning once per name) rather than being
  passed to a policy that cannot route it, and matching tolerates a case
  difference and trailing sentence punctuation, because `HANDOFF:Bob.` is
  what a model actually writes. Constrain the model properly with
  `handoff_tool(targets=[...])` — its schema enum makes an invented
  target impossible in the first place.

- **`llm_selector`** — asks a model who speaks next. It resolves the
  reply by *exact match*, else the **last whole-word mention**, longest
  name winning a tie. Reading the last mention follows `parse_handoff`'s
  `rfind` precedent: a model that reasons aloud commits at the end. The
  earlier version scanned the roster in its own order for a substring, so
  `"Not alice — bob should go next"` routed to **alice**, and a roster
  name living inside an ordinary word (`ed` in `proceed`) counted as a
  choice.

- **`RunPolicy`** — the global lethal-trifecta gate: no tool set may
  *both* read external content, write it, and make network calls without
  explicit approval. Fires once before the first cognition drive — and
  also before a `resume()`, because approving one tool call is not
  approval of the capability combination.

- **`ActorBudget`** — a per-agent slice of the run budget with four axes
  (`tokens`, `cost_usd`, `steps`, `wall_seconds`). Raises
  `BudgetExhausted` on the exhausted axis — distinct from `Budget`'s
  `MeterExceeded`. See [Runtime](runtime.md#budget-quota-meter-charge).

- **`Autonomy`** — the tier the run executes at: `"auto"`, `"gated"` or
  `"manual"`. Read by tools and cognitions that gate on human approval;
  the tier plus `@tool(side_effecting=..., requires_approval=...)`
  together decide whether a specific call suspends.

- **`Elicitation` / `Decision` / `Asker`** — pausing for a person as a
  **value request**, not only a veto. `Asker` is injected on `Services`;
  when present, a gated decision **parks in place** (the coroutine
  awaits, live state survives) instead of unwinding to a checkpoint.
  Deadlined, and typed with `actor` + `at`. Works from any cognition,
  because `elicit(ctx, ...)` takes a `Ctx`, not an `Agent`. See
  [the recipe](../recipes/elicit-a-value-from-a-human.md).

### Durable state: one slot per producer

Every producer that checkpoints — the tool loop, `Workflow`, the
coordinator policies — resolves its durable seam through the same order:
an explicit `checkpointer=` on the cognition, then `ctx.checkpointer`,
then a bridge over `ctx.store`. Three separate orders is how a
`Services(store=...)` wiring ended up giving durable tool-loop runs,
durable workflow gates, and coordinator runs that persisted nothing.

Slots are namespaced per producer, so they cannot collide. The tool loop
owns `ReActCognition.checkpoint_slot(run_id, agent_name)` —
`"{run_id}:agent:{name}"` — while a coordinator owns the bare run id.
Without that, a coordinator and its children shared one slot, and a child
finishing normally called `delete(run_id)` and took the coordinator's
in-progress state with it.

| Producer | Slot |
|---|---|
| Tool loop (`ReActCognition`) | `{run_id}:agent:{name}` |
| Coordinator policies | `{run_id}` |
| `PlanPolicy` human gate | `{run_id}:plan` |
| `Workflow` | `{run_id}` |

`Suspended.run_id` still carries the plain run id you pass back to
`Agent.resume`; the slot is re-derived internally. Two children sharing
one agent *name* in a single run still share a slot — name them
distinctly.

**Durable state is encoded, not stored raw.** A producer writes JSON-safe
dicts even when the backing store would happily hold live objects. That
is not ceremony: `PlanPolicy` used to put `Step` / `Usage` /
`AgentResult` instances straight into `ctx.store.set`, so its human gate
tested green on `InMemoryStore` and raised `TypeError: Object of type
Step is not JSON serializable` on a `FileStore` — the feature did not
work on the persistence anyone deploys. Encoding unconditionally is what
keeps an in-memory test honest about the wire.

One consequence worth knowing: a child result's `evals` / `parsed` can
hold anything. If they will not serialize, the plan drops those two
fields with a warning rather than letting the suspend itself raise —
losing the whole run at the gate is the worst available outcome. Return
JSON-safe values from `output=` parsers to keep them.

### How a run ends

`AgentResult.stop_reason` is a closed `Literal`, so the terminal state is
something you branch on rather than sniff out of a dict:

| `stop_reason` | Meaning |
|---|---|
| `complete` | The model produced a final answer |
| `suspended` | **Waiting on a person.** Resumable, and not a failure |
| `expired` | A human-gate deadline passed; the run degraded and continued |
| `budget_exhausted` | A meter ceiling hit. A checkpoint was written *before* stopping |
| `max_iterations` | The tool-loop ceiling was reached with no final answer |
| `invalid_output` | Parse-and-repair exhausted |
| `terminated` | Stopped deliberately: a `TerminationCondition` fired, a person declined at a gate, a run was cancelled (the exact wording is in `evals["stop_reason"]`) |
| `failed` | The run errored and the cognition reported it as *data* rather than raising — only the two CLI cognitions do this, so their guarantee of a terminal event survives a subprocess that never starts |

`result.is_suspended` and `result.is_resumable` are the two convenience
reads. A run that **failed** usually produces no `AgentResult` at all —
the exception propagates — which is what makes "waiting for you" and "it
fell over" distinguishable. `failed` covers the one deliberate exception
to that rule.

**Every producer maps onto this table**, coordinators included. A policy's
own vocabulary is richer than the taxonomy — a plan says
`awaiting_decision`, a ledger says `max_rounds`, a round-robin says
`max_turns` — so each keeps its exact word in `evals["stop_reason"]` and
maps the category onto the typed field with
`agentkit.agents.result.stop_reason_for`. The mapping is *total*: an
unrecognised reason (a custom `TerminationCondition`'s wording) becomes
`terminated` rather than a guess.

That mapping is not decoration. While the policies skipped it, a plan
parked on a human gate reported `stop_reason="complete"` and
`is_suspended is False` with its checkpoint sitting in the store — so an
application branching on the typed field never asked its human and never
resumed. If you write your own policy, map the reason; a source-level
test refuses a new framework reason that is categorised nowhere.

### Plan shape is checked before dispatch

`PlanPolicy` validates a plan against the coordinator's child roster
*before* the first step runs, raising `PlanShapeError` (a `ValueError`)
on three shapes:

| Shape | Why it is refused |
|---|---|
| A step naming a child that isn't on the coordinator | Under `best_effort=False` there is nothing to dispatch. It used to raise a bare `KeyError('reseacher')` from inside the dispatch loop — after the earlier groups had run and spent, and with their results unreachable |
| A step with neither an `agent` nor a `gate_name` | Nothing to dispatch and nothing to wait for |
| A gate sharing a group with dispatch steps | A gate suspends its whole group *before* any step runs, and resume continues at the group **after** it — so those steps were announced in the trace and then never ran, on approve *and* on reject. Whether the work belongs before or after the decision is exactly what the plan failed to say |

`best_effort=True` treats only the first of those as data, not an error: a
live `Planner` names the child it wants, so an unknown name can be a
runtime answer rather than a typo. It lands in `evals["errors"]` as a
`PERMANENT` `Failure` — re-dispatching a name that isn't on the roster
cannot succeed — and the rest of the plan runs. That is the mode's whole
promise, and a mid-loop `KeyError` could not keep it.

`resume` re-validates, because the roster is re-supplied by the caller
and a service that rebuilds its coordinator from config can lose a child
between suspend and resume.

### Model capability contract

`Agent(requires=("vision",), min_context_window=100_000)` is checked at
**construction**, against the
[model registry](../recipes/provider-from-env.md) — before any spend,
because catching it after the bill is worthless. A capability the model
declares as unsupported raises `CapabilityMismatch`; one it doesn't
declare at all is `UNKNOWN` and governed by `on_unknown_capability`
(`"warn"` by default, `"refuse"` for a service that pins its models,
`"allow"` for a caller who verified out of band). `UNKNOWN` is never
treated as present.

A tool-using cognition implies `"tools"` automatically, so you do not
have to declare it.

## What bites people

- **Tools go on the cognition.** `Agent(tools=[...])` is not a thing.
- **A suspended run is not a failed run.** Branch on `stop_reason` /
  `is_suspended`, not on whether you got an exception.
- **Two children with the same `name` in one run share a checkpoint
  slot.** Name them distinctly.
- **`evals` and `parsed` must be JSON-safe** if you want them to survive
  a suspend.
- **Mutating an `Agent` after construction skips its checks.** Call
  `check_prompt()` / `check_capabilities()` again.
- **A `TerminationCondition` you pass in is never advanced.** Every
  producer deep-copies it per run, so reusing a coordinator is safe — but
  reading `MaxTurns(4).turn` on the instance you passed will always show
  zero.

## The invariants it enforces

1. **No agent subclassing.** New shapes are new `Cognition` / `Policy`
   implementations.
2. **Signals are frozen.** A `SignalEnvelope` is immutable; consumers
   read, they don't edit.
3. **Termination is per-run.** A `TerminationCondition` is stateful
   (`MaxTurns.turn`, `Timeout._start`, the latched `Stop`) and lives on a
   long-lived cognition, so every producer — the `ReAct` drive *and* the
   coordinator policies — deep-copies it into a run-local variable.
   Without that, two concurrent `coordinator.run(...)` calls counted into
   one counter: `MaxTurns(4)` gave one run 3 turns and the other 2. The
   instance you passed in is never advanced, so a coordinator is safe to
   reuse.

    `ExternalTermination` opts out of the copy, and that is deliberate: a
    stop switch you hold a handle to is not per-run state, and copying it
    meant `set()` could never reach a loop already running. One switch
    shared by two concurrent runs stops both — use one condition per run
    if you need them independent.

    A judge condition (`judge_termination`) stops only on a **leading**
    affirmative. "Not yet — yesterday's draft is still open." used to
    stop the run, because `YES` was matched as a substring.

4. **Handoff transfers ownership.** After a `Handoff`, the source agent
   stops emitting; there is no shared write.
5. **Suspended is not failed.** A parked run returns a typed
   `AgentResult`; a broken one raises. A reader must be able to tell them
   apart without parsing a message.
6. **A capability is declared, never inferred from a name.** An
   unregistered model reports `UNKNOWN`, never `True` — guessing `True`
   reintroduces the silent, well-formed wrong answer the check exists to
   catch.

!!! abstract "Where this fits in the four themes"
    This page covers the **Cognition** theme (`SingleCallCognition`,
    `ReActCognition`, `CoordinatorCognition`, `ClaudeCliCognition`,
    `CodexCliCognition`, and
    your own `Cognition` implementations) and the **Control** theme
    (`Autonomy`, `RunPolicy`, `ActorBudget`, `Suspended` +
    `agent.resume(...)`, `SignalChannel`, `Handoff`). See the four-theme
    grid on the [landing page](../index.md).

## Related

- [Runtime](runtime.md) — the `RunContext` every cognition takes.
- [Capabilities](capabilities.md) — the collaborators a cognition resolves.
- [Tools](tools.md) — what goes in `ReActCognition(tools=[...])`.
- [Memory](memory.md) and [Context](context.md) — what an agent knows.
- [Human-in-the-loop tool approval](../recipes/hitl-tool-approval.md) and
  [Resume from a checkpoint after a crash](../recipes/resume-after-crash.md).
- The
  [multi-agent mental model](https://github.com/arc-labs-ai/agentkit/blob/main/docs/mental-models/04-multi-agent-coordinated-research.md)
  walks through coordination end to end.
- [API › agents](../api-reference/agents.md) — the generated reference.
