# Agents

**What this is.** `agentkit.agents` is where a *thing that talks to a
model* becomes *a thing you can plan, cancel, coordinate, and hand off
between*. It provides `Agent` and `Workflow`, the `Cognition` protocol
(`SingleCallCognition`, `ReActCognition`, `CoordinatorCognition`),
the control primitives that make multi-agent flows safe
(`SignalChannel`, `Handoff`, `ActorBudget`, `RunPolicy`), and the
policies (`RoundRobinPolicy`, `SelectorPolicy`, `PlanPolicy`,
`LedgerPolicy`) that decide which agent runs next.

**Why it exists.** "An agent" is a loaded word. In agentkit it means
something very specific: a wired composition of a prompt, a cognition,
a tool set, and (optionally) a memory + capabilities — with a
`Cognition` that owns the *how* of iterating. That split is what makes
it possible to change the loop (`SingleCall` → `ReAct`) without
touching prompts or tools, and to coordinate several agents without
any of them subclassing a common base.

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
and ships three implementations:

- **`SingleCallCognition`** — one LLM call, one result. The right
  default for narrow, single-shot skills.
- **`ReActCognition`** — thought → action → observation loop, with
  tool calling and cooperative cancellation between steps.
- **`CoordinatorCognition`** — drives many child agents according to a
  `Policy`, merges their signals through a `SignalChannel`.

Import path: `from agentkit.agents.cognition import ReActCognition`.
You can add your own — a cognition is a small async iterator over a
`RunContext` and a task.

### `Workflow`

A `Workflow` composes multiple `Agent`s under a `Policy`. It's the
thing that lets you say "run planner → many researchers in parallel →
synthesizer → critic → human checkpoint" without any agent knowing
about the others.

### Control primitives

- **`SignalChannel`** — the frozen envelope multi-agent signals travel
  in (`ProgressSignal`, `DoneSignal`, `CancelSignal`,
  `EscalateSignal`, …). `Handoff` is separate: it's a routing verb
  consumed by `SelectorPolicy` / handoff-routing, not a member of the
  progress/done data-signal family.
- **`RunPolicy`** — global lethal-trifecta gate (no tool can *both*
  read external content, write it, and send network calls without
  explicit approval).
- **`ActorBudget`** — per-agent slice of the run budget.
- **`Autonomy`** — the tier the run is executing at: `auto`, `gated`,
  or `manual`.

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

## Related deep dive

The
[multi-agent mental model](https://github.com/arc-labs-ai/agentkit/blob/main/docs/mental-models/04-multi-agent-coordinated-research.md)
walks through coordination end to end and lists the invariants that
make it correct.

## API

Full generated reference lives at
[API › agents](../api-reference/agents.md).
