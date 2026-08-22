# Agents

!!! abstract "Where this fits in the four themes"
    This page covers primitives from the **Cognition** theme
    (`SingleCallCognition`, `ReActCognition`, `CoordinatorCognition`,
    `ClaudeCliCognition`, custom `Cognition` Protocol impls) and the
    **Control** theme (`Autonomy`, `RunPolicy`, `ActorBudget`,
    `Suspended` + `agent.resume(...)`, `SignalChannel`, `Handoff`).
    See the four-theme grid on the [landing](../index.md) for the
    top-level mental model.

**What this is.** `agentkit.agents` is where a *thing that talks to a
model* becomes *a thing you can plan, cancel, coordinate, and hand off
between*. It provides `Agent` and `Workflow`, the `Cognition` protocol
and its four first-party implementations (`SingleCallCognition`,
`ReActCognition`, `CoordinatorCognition`, `ClaudeCliCognition`), the
control primitives that make multi-agent flows safe (`SignalChannel`,
`Handoff`, `ActorBudget`, `RunPolicy`), and the policies
(`RoundRobinPolicy`, `SelectorPolicy`, `PlanPolicy`, `LedgerPolicy`)
that decide which agent runs next.

**Why it exists.** "An agent" is a loaded word. In agentkit it means
something very specific: a wired composition of a prompt, a cognition,
a tool set, and (optionally) a memory + capabilities — with a
`Cognition` that owns the *how* of iterating. That split is what makes
it possible to change the loop (`SingleCall` → `ReAct`) without
touching prompts or tools, and to coordinate several agents without any
of them subclassing a common base.

## The pieces

### `Agent`

A single wired composition:

```python
Agent(
    name="researcher",
    cognition=ReActCognition(tools=[search, fetch]),
    prompt=my_prompt,
    memory=composite_memory,
)
```

Tools live on the cognition, not on `Agent`. Cross-cutting
capabilities — `Compactor`, `Guardrail`, `Checkpointer`, `Evaluator` —
are not `Agent` constructor kwargs; they plug in via a
`RequestBuilder`, the middleware chain, or a policy (see
[Capabilities](capabilities.md)).

Calling `await agent.run(task, ctx)` returns an `AgentResult` when the
cognition completes.

### `Cognition`

The Protocol that owns the loop. Lives in `agentkit.agents.cognition`
and ships four implementations:

- **`SingleCallCognition`** — one LLM call, one result. The default;
  the right choice for narrow, single-shot skills.
- **`ReActCognition`** — thought → action → observation loop, with
  tool calling, HITL suspend/resume via `Checkpointer`, and cooperative
  cancellation between steps. The only cognition that supports
  `agent.resume(...)`.
- **`CoordinatorCognition`** — drives many child agents according to a
  `Policy`, merges their signals through a `SignalChannel`.
- **`ClaudeCliCognition`** — delegates the loop to a locally-installed
  `claude` CLI (no API key handling on your server; the CLI's own auth
  is used). Emits the same `StreamEvent`s the other cognitions do.

Import path: `from agentkit.agents.cognition import ReActCognition`.
Cognitions are deliberately NOT re-exported from the top-level
`agentkit` package — the top level is already dense, and cognitions are
one family among many. You can add your own — a cognition is a small
async iterator over a `RunContext` and a task.

### `Workflow`

A `Workflow` composes multiple `Agent`s under a `Policy`. It's the
thing that lets you say "run planner → many researchers in parallel →
synthesizer → critic → human checkpoint" without any agent knowing
about the others. Workflow's `human_gate` node suspends the workflow
(not an individual cognition) with the same `Suspended` shape.

### Control primitives

- **`SignalChannel`** — the frozen envelope multi-agent signals travel
  in (`ProgressSignal`, `DoneSignal`, `CancelSignal`,
  `EscalateSignal`, …). `Handoff` is separate: it's a routing verb
  consumed by `SelectorPolicy` / `route_by_handoff`, not a member of
  the progress/done data-signal family.
- **`Handoff` + routing** — `route_by_handoff(default=...)` reads the
  `HANDOFF:<target>` marker off the last message and routes to that
  child. The target is **checked against the roster**: an invented name
  falls back to `default` (warning once per name) rather than being
  passed to a policy that cannot route it, and matching tolerates a
  case difference and trailing sentence punctuation, because
  `HANDOFF:Bob.` is what a model actually writes. Constrain the model
  properly with `handoff_tool(targets=[...])` — its schema enum makes
  an invented target impossible in the first place.
- **`llm_selector`** — asks a model who speaks next. It resolves the
  reply by *exact match*, else the **last whole-word mention**, longest
  name winning a tie. Reading the last mention follows `parse_handoff`'s
  `rfind` precedent: a model that reasons aloud commits at the end. The
  earlier version scanned the roster in its own order for a substring,
  so `"Not alice — bob should go next"` routed to **alice** and a roster
  name living inside an ordinary word (`ed` in `proceed`) counted as a
  choice.
- **`RunPolicy`** — global lethal-trifecta gate (no tool can *both*
  read external content, write it, and send network calls without
  explicit approval). Fires once before the first cognition drive.
- **`ActorBudget`** — per-agent slice of the run budget with four
  axes (`tokens`, `cost_usd`, `steps`, `wall_seconds`). Raises
  `BudgetExhausted` (distinct from `Budget`'s `MeterExceeded`) on the
  exhausted axis.
- **`Autonomy`** — the tier the run is executing at: `"auto"`,
  `"gated"`, or `"manual"`. Read by tools and cognitions that gate on
  human approval; the tier + `@tool(side_effecting=..., requires_approval=...)`
  together decide whether a specific tool call suspends.
- **`Elicitation` / `Decision` / `Asker`** — pausing for a person as a
  **value request**, not only a veto. `Asker` is injected on
  `Services`; when present a gated decision **parks in place** (the
  coroutine awaits, live state survives) instead of unwinding to a
  checkpoint. Deadlined, and typed with `actor` + `at`. Works from any
  cognition, because `elicit(ctx, ...)` takes a `Ctx`, not an `Agent`.
  See [the recipe](../recipes/elicit-a-value-from-a-human.md).

### Durable state: one slot per producer

Every producer that checkpoints — the tool loop, `Workflow`, the coordinator
policies — resolves its durable seam through the same order: an explicit
`checkpointer=` on the cognition, then `ctx.checkpointer`, then a bridge over
`ctx.store`. Three separate orders is how a `Services(store=...)` wiring ended
up giving durable tool-loop runs, durable workflow gates, and coordinator runs
that persisted nothing.

Slots are namespaced per producer, so they cannot collide. The tool loop owns
`ReActCognition.checkpoint_slot(run_id, agent_name)` —
`"{run_id}:agent:{name}"` — while a coordinator owns the bare run id. Without
that, a coordinator and its children shared one slot, and a child finishing
normally called `delete(run_id)` and took the coordinator's in-progress state
with it.

`Suspended.run_id` still carries the plain run id you pass back to
`Agent.resume`; the slot is re-derived internally. Two children sharing one
agent *name* in a single run still share a slot — name them distinctly.

| Producer | Slot |
|---|---|
| Tool loop (`ReActCognition`) | `{run_id}:agent:{name}` |
| Coordinator policies | `{run_id}` |
| `PlanPolicy` human gate | `{run_id}:plan` |
| `Workflow` | `{run_id}` |

**Durable state is encoded, not stored raw.** A producer writes JSON-safe
dicts even when the backing store would happily hold live objects. That is
not ceremony: `PlanPolicy` used to put `Step` / `Usage` / `AgentResult`
instances straight into `ctx.store.set`, so its human gate tested green on
`InMemoryStore` and raised `TypeError: Object of type Step is not JSON
serializable` on a `FileStore` — the feature did not work on the
persistence anyone deploys. Encoding unconditionally is what keeps an
in-memory test honest about the wire.

One consequence worth knowing: a child result's `evals` / `parsed` can hold
anything. If they will not serialize, the plan drops those two fields with a
warning rather than letting the suspend itself raise — losing the whole run
at the gate is the worst available outcome. Return JSON-safe values from
`output=` parsers to keep them.

### How a run ends

`AgentResult.stop_reason` is a closed `Literal`, so the terminal state
is something you branch on rather than sniff out of a dict:

| `stop_reason` | Meaning |
|---|---|
| `complete` | The model produced a final answer |
| `suspended` | **Waiting on a person.** Resumable, and not a failure |
| `expired` | A human-gate deadline passed; the run degraded and continued |
| `budget_exhausted` | A meter ceiling hit. A checkpoint was written *before* stopping |
| `max_iterations` | The tool-loop ceiling was reached with no final answer |
| `invalid_output` | Parse-and-repair exhausted |
| `terminated` | Stopped deliberately: a `TerminationCondition` fired, a person declined at a gate, a run was cancelled (the exact wording is in `evals["stop_reason"]`) |
| `failed` | The run errored and the cognition reported it as *data* rather than raising — only `ClaudeCliCognition` does this, so its guarantee of a terminal event survives a subprocess that never starts |

`result.is_suspended` and `result.is_resumable` are the two
convenience reads. A run that **failed** usually produces no
`AgentResult` at all — the exception propagates — which is what makes
"waiting for you" and "it fell over" distinguishable. `failed` covers
the one deliberate exception to that rule.

**Every producer maps onto this table**, coordinators included. A
policy's own vocabulary is richer than the taxonomy — a plan says
`awaiting_decision`, a ledger says `max_rounds`, a round-robin says
`max_turns` — so each keeps its exact word in `evals["stop_reason"]`
and maps the category onto the typed field with
`agentkit.agents.result.stop_reason_for`. The mapping is *total*: an
unrecognised reason (a custom `TerminationCondition`'s wording)
becomes `terminated` rather than a guess.

That mapping is not decoration. While the policies skipped it, a plan
parked on a human gate reported `stop_reason="complete"` and
`is_suspended is False` with its checkpoint sitting in the store — so
an application branching on the typed field never asked its human and
never resumed. If you write your own policy, map the reason; a
source-level test refuses a new framework reason that is categorised
nowhere.

### Plan shape is checked before dispatch

`PlanPolicy` validates a plan against the coordinator's child roster
*before* the first step runs, raising `PlanShapeError` (a `ValueError`)
on three shapes:

| Shape | Why it is refused |
|---|---|
| A step naming a child that isn't on the coordinator | Under `best_effort=False` there is nothing to dispatch. It used to raise a bare `KeyError('reseacher')` from inside the dispatch loop — after the earlier groups had run and spent, and with their results unreachable |
| A step with neither an `agent` nor a `gate_name` | Nothing to dispatch and nothing to wait for |
| A gate sharing a group with dispatch steps | A gate suspends its whole group *before* any step runs, and resume continues at the group **after** it — so those steps were announced in the trace and then never ran, on approve *and* on reject. Whether the work belongs before or after the decision is exactly what the plan failed to say |

`best_effort=True` treats only the first of those as data, not an
error: a live `Planner` names the child it wants, so an unknown name
can be a runtime answer rather than a typo. It lands in
`evals["errors"]` as a `PERMANENT` `Failure` — re-dispatching a name
that isn't on the roster cannot succeed — and the rest of the plan
runs. That is the mode's whole promise, and a mid-loop `KeyError`
could not keep it.

`resume` re-validates, because the roster is re-supplied by the caller
and a service that rebuilds its coordinator from config can lose a
child between suspend and resume.

### Model capability contract

`Agent(requires=("vision",), min_context_window=100_000)` is checked
at **construction**, against the
[model registry](../recipes/provider-from-env.md) — before any spend,
because catching it after the bill is worthless. A capability the
model declares as unsupported raises `CapabilityMismatch`; one it
doesn't declare at all is `UNKNOWN` and governed by
`on_unknown_capability` (`"warn"` by default, `"refuse"` for a
service that pins its models). `UNKNOWN` is never treated as present.

## The invariants it enforces

1. **No agent subclassing.** New shapes are new `Cognition` /
   `Policy` implementations.
2. **Signals are frozen.** A `SignalEnvelope` is immutable; consumers
   read, they don't edit.
3. **Termination is per-drive.** A `ReAct` cognition deep-copies its
   termination condition on every drive so cross-drive state can't
   leak.
4. **Handoff transfers ownership.** After a `Handoff`, the source
   agent stops emitting; there is no shared write.
5. **Suspended is not failed.** A parked run returns a typed
   `AgentResult`; a broken one raises. A reader must be able to tell
   them apart without parsing a message.
6. **A capability is declared, never inferred from a name.** An
   unregistered model reports `UNKNOWN`, never `True` — guessing
   `True` reintroduces the silent, well-formed wrong answer the check
   exists to catch.

## Related deep dive

The
[multi-agent mental model](https://github.com/arc-labs-ai/agentkit/blob/main/docs/mental-models/04-multi-agent-coordinated-research.md)
walks through coordination end to end and lists the invariants that
make it correct.

## API

Full generated reference lives at
[API › agents](../api-reference/agents.md).
