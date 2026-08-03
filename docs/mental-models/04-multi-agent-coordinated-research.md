# Mental Model: Multi-Agent Coordinated Research Report

The coordination + streaming story. This use case exists to prove that
agentkit's multi-agent primitives (`CoordinatorCognition`, `Skill.as_agent`,
`SignalChannel`, `ActorBudget`, streaming through `merge`) compose into a
team of agents that observe each other, share progress upward, and stay
isolated where isolation matters.

## Problem

Given a research question ("what's the state of solid-state batteries in
2026?"), produce a cited brief. The team is:

- **Planner** — decomposes the question into 3–7 sub-questions.
- **Researchers** (one per sub-question, running in parallel) — search,
  read, and score sources.
- **Synthesizer** — waits for researchers, produces a draft synthesis.
- **Critic** — adversarially reviews the synthesis, flags contradictions.

The user watches a live canvas: agents appear, their work streams in,
edges connect sub-questions to sources to claims. A checkpoint pauses
before synthesis so the user can drop irrelevant sources.

## User experience

User submits question. Within seconds a Planner node appears on canvas
with 5 sub-question children. Five Researcher agents spawn in parallel —
each streams its own tokens into its own card. As sources arrive, they
appear under the matching sub-question. Progress bar in each Researcher
card ticks up. After ~90s a checkpoint fires: "Review sources before
synthesis." User drops 2 irrelevant sources, clicks Continue. Synthesizer
starts, streaming a cited draft. Critic runs after. User exports the
final brief.

## How it actually works end-to-end

Walk through the internal state as one successful research run flows
through the framework. The user's `POST /research {question: "..."}`
has already arrived; the api service has constructed a top-level
`RunContext` (`correlation_id="run-xyz"`, `budget=Budget(...)`,
`services=Services(invoker=<shared>, observer=<per-run>, ...)`, shared
`CancellationToken`).

**t=0ms — Coordinator Agent construction.** The api wires four Skills
(`planner_skill`, `researcher_skill`, `synth_skill`, `critic_skill`)
and builds the coordinator:

```
coordinator = Agent(
    name="coordinator",
    cognition=CoordinatorCognition(
        children={
            "planner":     planner_skill.as_agent(),
            "researcher":  researcher_skill.as_agent(),
            "synthesizer": synth_skill.as_agent(),
            "critic":      critic_skill.as_agent(),
        },
        policy=PlanPolicy(planner=<Planner impl>, best_effort=True),
    ),
)
```

Each `Skill.as_agent()` returns a **fresh** `Agent` whose `cognition`
is a `copy.deepcopy(skill.cognition)` (skill.py:106). Two spawns of the
same Skill therefore never share the mutable termination counter that
lives on `ReActCognition.termination`. That deep-copy is the primary
isolation seam.

**t=~10ms — coordinator.run entered.** The Agent's `stream` opens an
`invoke_agent` span and hands off to `CoordinatorCognition.drive`,
which delegates to `policy.execute(coordinator, task, ctx, context)`.

**t=~20ms — PlanPolicy calls Planner.plan(task, ctx).** The Planner
protocol is declared "may be sync or async" — the concrete Planner is
an LLM-backed impl, so it returns a coroutine. `PlanPolicy` checks
`inspect.isawaitable(...)` and `await`s the result. The Planner
internally runs its own agent (small `ReActCognition` or
`SingleCallCognition`) against an LLM to decompose the question, then
returns a list of `Step(agent, input, group)`:

```
[Step("researcher", "solid-state cathode chemistries", group=0),
 Step("researcher", "cycle-life results 2025-2026",   group=0),
 Step("researcher", "sulfide vs oxide electrolytes", group=0),
 Step("researcher", "manufacturability at scale",    group=0),
 Step("researcher", "safety incidents / thermal",    group=0),
 Step("synthesizer", "<merged corpus placeholder>",  group=1),
 Step("critic",      "<synthesis placeholder>",      group=2)]
```

**t=~4s — Group 0 fans out.** PlanPolicy groups by
`sorted({s.group for s in steps})` and, for `group=0`, builds pairs
`[(children["researcher"], step.input), ...]` and calls
`run_agents(pairs, ctx, best_effort=True)`. `run_agents` grabs
`ctx.semaphore()` and dispatches through `gather_best_effort` — a
bounded fan-out where a failing researcher's exception is captured
into the slot instead of cancelling siblings.

**Problem the framework must solve:** all five `pairs` share ONE
`children["researcher"]` Agent instance. Its `cognition` is one
`ReActCognition` object. Five concurrent `.run(...)` calls on that
one instance would otherwise race on `MaxTurns.turn` /
`Timeout._start`.

**t=~4s — Each Researcher's drive isolates termination.** Each
child's `ReActCognition.drive` opens with:

```python
termination = _copy.deepcopy(self.termination) if self.termination is not None else None
```

(react.py:113) — a **drive-local** variable, not `self.termination`.
The clone is threaded into `_iterate` as an argument (react.py:128).
Two concurrent drives on the same cognition therefore each own their
own counter. This is the second isolation seam; the first (per-spawn
Skill deep-copy) covers spawn-from-Skill paths, this one covers
parallel-dispatch-on-shared-Agent paths.

**t=~5s — Researchers start emitting.** Each Researcher runs its
tool loop (`search`, `fetch`, `score_source`). As sources land it
emits `ProgressSignal[Mutation]` on its `SignalChannel`. Every
`emit(signal)` runs:

```python
stamped = dataclasses.replace(
    signal,
    sender_id=signal.sender_id if signal.sender_id is not None else self.agent_id,
    timestamp_us=signal.timestamp_us if signal.timestamp_us != 0 else self._clock(),
)
await self.outbox.put(stamped)
if self._parent_merge_inbox is not None:
    await self._parent_merge_inbox.put((self.agent_id, stamped))
```

(channel.py:165, 171-173). The frozen envelope is **not mutated in
place** — `replace(...)` returns a new stamped copy, and the SAME
reference lands in both the outbox (audit) and the parent's
merge_inbox (live merge). Identity is preserved across those two
consumers; timestamps are stamped once, monotonic per channel.

**t=~5-90s — Coordinator drains merge_inbox.** The coordinator's
side sees ONE queue (`merge_inbox`) into which five children's
signals fan in with `(sender_id, stamped_signal)` pairs. Each signal
becomes a Scene mutation that the api forwards to the canvas over
WebSocket.

**t=~10-90s — StreamEvent flow.** In parallel with the signal
stream, each Researcher's `ReActCognition._iterate` yields
`StreamEvent("message_delta", text=...)` on every LLM token,
`StreamEvent("tool_call", ...)` on every tool dispatch,
`StreamEvent("tool_result", ...)` on every tool completion. The
transport layer (api service) `merge(...)`s the five per-agent
`StreamEvent` iterators into one interleaved stream — the streams
operator uses a bounded queue with dual sentinel paths so a slow
consumer applies backpressure and a cancelled pump doesn't deadlock
(streams.py:119, `merge`).

**t=~90s — All Researchers DoneSignal.** Each Researcher's loop
terminates (either `MaxTurns` in its drive-local clone trips, or the
LLM emits a natural stop). Each emits `DoneSignal[Mutation]` upward.
`run_agents` collects the five `AgentResult`s in input order — a
best-effort slot with an exception is left in place; PlanPolicy then
scans results and appends any failure to
`AgentResult.evals["errors"]`.

**t=~91s — human_gate suspension.** Between group 0 (researchers)
and group 1 (synthesizer), a `Step.gate("review")` in the plan
suspends the coordinator via `Suspended`. `PlanPolicy` checkpoints
the accumulated results + errors + usage to `ctx.store` under a
`plan_policy:<run_id>` key, then returns
`AgentResult.evals["suspended"] = Suspended(run_id="run-xyz", pending=("review",))`
with `stop_reason="awaiting_decision"`. The synthesizer step in
group 2 does NOT run.

**t=<indefinite> — Operator interacts.** The user drops two
irrelevant sources on the canvas. The api posts an approve command
with the trimmed source-id set and the pending decision id
**verbatim**. It calls
`PlanPolicy.resume(coordinator, {"review": "approve"}, ctx)`: the
policy loads the saved state from `ctx.store`, deletes the
checkpoint, and continues at the group AFTER the gate. A
`{"review": "reject"}` decision (or a missing key) instead returns
`stop_reason="rejected"` without running the synthesizer.

**t=~91s (wall) / t=0 (resume) — Synthesizer runs.** PlanPolicy's
next group has one step: `synthesizer` with the approved corpus.
`ReActCognition` (or `SingleCallCognition`, whichever the Synth
Skill wired) drives one LLM call over the approved sources, emitting
`message_delta` tokens as the draft streams. Terminal delta →
`StreamEvent("final", result=AgentResult(...), usage=...)`.

**t=~180s — Critic runs.** Group 2 dispatches the critic on the
synthesizer's output. The critic emits contradiction / weak-citation
nodes as its own progress mutations. When done, another `final`.

**t=~200s — Coordinator returns.** PlanPolicy accumulates
`results[-1].output` as the final brief (the critic's output, since
it ran last; alternate wirings pin the synthesizer's output as the
brief and treat critic outputs as sidecar evals). Usage is merged
across all seven step-runs. The coordinator's `drive` yields exactly
one terminal `StreamEvent("final", result=AgentResult(...), usage=...)`.
`Agent.run` collects that final and returns the top-level result.

## Composition

```
POST /research {question: "..."} 
     │
     ▼
Agent(name="coordinator",
      cognition=CoordinatorCognition(
        children={
          "planner":     planner_skill.as_agent(),     # deep-copy per spawn
          "researcher":  researcher_skill.as_agent(),  #   ↑ isolation seam #1
          "synthesizer": synth_skill.as_agent(),
          "critic":      critic_skill.as_agent(),
        },
        policy=PlanPolicy(planner=<Planner protocol impl>),
      ))
     │
     │  child spawn: coordinator.channel.attach_parent(coordinator's merge_inbox)
     │  each child channel emits upward → parent reads one queue
     │
     ▼
Planner runs → returns Plan(sub_questions=[q1..q5])
     │
     ▼
PlanPolicy dispatches Steps in group 0 → group 1 → …
  group 0: [Step(agent="researcher", input=q1),
            Step(agent="researcher", input=q2),
            ...]
     │
     ▼
run_agents(pairs, ctx, best_effort=True) fans out:
  gather_bounded over ctx.semaphore()
     │
     ▼
Each Researcher = Agent(cognition=ReActCognition(
    tools=[search, fetch, score_source],
    termination=MaxTurns(8) | Timeout(180)))
  → each drive() deep-copies its termination (isolation seam #2)
  → each emits ProgressSignal[Mutation] upward
     │
     ▼
Coordinator's merge_inbox drains signals as they arrive.
Each ProgressSignal is a stamped, frozen envelope.
Coordinator writes each mutation to the Scene → canvas re-renders.
     │
     ▼
When all researchers DoneSignal upward:
  human_gate → suspends → operator drops sources → resume
     │
     ▼
Synthesizer runs on approved corpus → SynthesisNode
Critic runs on the synthesis → CritiqueNodes
     │
     ▼
Return AgentResult(output=<final brief>, usage=<merged>)
```

## The primitives it exercises

| Primitive | Role here | Why load-bearing |
|---|---|---|
| `Coordinator` `Agent` + `CoordinatorCognition` | The team-lead pattern | Emergent multi-agent control |
| `Skill.as_agent()` (deep-copied cognition) | Materialise a named worker per spawn | 5 researchers must not share MaxTurns state |
| `PlanPolicy` + `Planner` (may be async) | Plan → dispatch children in ordered groups | The Planner LLM call is async; PlanPolicy awaits |
| `run_agents` (via `gather_bounded`) | Concurrency-capped child dispatch | Never spawns more researchers than `ctx.budget.semaphore` allows |
| `ReActCognition` (per researcher) | Tool loop for search/fetch/score | Each researcher iterates until done or terminated |
| Per-drive termination clone | 5 researchers, one shared cognition instance, no state race | The drive-local variable, not `self.termination` |
| `SignalChannel` + `MergeInbox` | Child → parent progress stream | Coordinator reads ONE queue instead of polling N children |
| Frozen `SignalEnvelope` + `replace`-on-stamp | Same stamped envelope on outbox + parent's merge_inbox | Audit and merge loop see identical envelope identity |
| Monotonic `timestamp_us` per channel | Replay ordering across a cascade | Sorting by timestamp reconstructs the DAG |
| `StreamEvent` (`Literal` type tag) | Per-agent token streaming to canvas | `message_delta` / `tool_call` / `tool_result` / `final` |
| `merge` (streams operator) | Interleave multiple children's event streams into one | The canvas subscribes to a single unified stream |
| `human_gate` (as a Workflow-style pause) OR `Suspended` | Pre-synthesis review checkpoint | User can drop sources before the synthesizer runs |
| `ActorBudget` per researcher | Slice of parent's Budget to each child | One runaway researcher can't drain the whole run's budget |
| `CancellationToken` (shared) | Cancel any node cancels all its children | User's "abort research" propagates through the tree |
| `output_coerce` + Pydantic schemas per agent | Planner returns `Plan`, Researcher returns `ResearchFindings`, etc. | Typed inter-agent handoffs |
| `Handoff` + `handoff_tool` (optional) | Explicit named routing when coordinator's policy needs it | For the "Critic hands off to Reviser" pattern (not used here but part of the family) |

## What it deliberately doesn't use

- **`Workflow`** — this is emergent (planner decides the plan at runtime).
  Workflow would be right for a research pipeline with a FIXED shape
  (`explore → catalog → cluster → decompose → investigate → synth →
  critique`), which is what research·io's engine actually uses. This doc
  is about the coordinator-cognition shape.
- **`CachedMemory`** — each research run is unique; no repeated queries
  within a session worth caching.
- **`FileTool`** — sources come from the web (via `SearchPort` / `FetchPort`),
  not the local FS.

## Invariants

| Invariant | Concrete failure if violated | Locked by |
|---|---|---|
| **`Skill.as_agent()` returns independent cognition instances** | 5 researchers share MaxTurns counter → any one hitting max ends all of them | `test_as_agent_cognitions_are_independent_instances` |
| **Concurrent `as_agent()` calls produce isolated state** | Two researchers spawned in parallel scribble on each other's termination | `test_concurrent_as_agent_calls_do_not_share_termination_state` |
| **`ReActCognition.drive` uses a drive-local termination clone** | Two concurrent drives on the same cognition (Coordinator dispatch) race on `self.termination = deepcopy(...)` assignment | Termination lives on a local var + threaded into `_iterate`; regression documented in the correctness review that found the assign-to-self race |
| **`SignalEnvelope` is frozen** | Post-emit mutation of `sender_id` → audit trail lies | `test_signal_envelope_is_frozen`, `test_all_signal_types_are_frozen` |
| **`emit` publishes a stamped copy; caller's original untouched** | Caller-side logging shows different values than the queue's | `test_channel_emit_does_not_mutate_caller_signal` |
| **Outbox and merge_inbox see identical stamped reference** | Audit and merge-loop disagree on the envelope's identity (correlation_id / causation_id) | `test_channel_emit_produces_same_stamped_copy_for_outbox_and_parent` |
| **Timestamps within a channel are monotonic** | Replay ordering breaks; UI shows events out of order | `test_channel_emit_stamps_are_monotonic_across_signals` |
| **Forwarder attribution: MergeInbox key = forwarder's agent_id** | A child that forwards a grandchild's signal is invisible in the audit → causation graph broken | `test_channel_emit_stamps_child_id_when_forwarded_to_parent` |
| **`try_send_to` never raises on full inbox** | Best-effort broadcast becomes blocking → parent stalls waiting for a full child inbox | `test_channel_try_send_to_full_inbox_drops_without_raising` |
| **Two independent channels don't cross-talk** | Concurrent emit from two channels crosses envelopes → attribution wrong | `test_two_channels_emit_independently` |
| **`PlanPolicy` awaits async planner output** | Async Planner returns a coroutine; iteration raises TypeError | `test_workflow_with_async_planner_via_planpolicy_awaits_before_iterating` |
| **`Handoff` / `handoff_tool` accepts `MappingProxyType` args** | Tool receives `ToolCall.arguments` (a proxy view); `isinstance(dict)` fails → every handoff rejected as "unknown target" | `test_handoff_tool_accepts_mappingproxy_arguments` |
| **`as_tool` extracts `task` from a `MappingProxyType`** | Skill invoked as a tool receives `str(args)` instead of the task string → sub-run gets garbage | `test_as_tool_forwards_task_from_mappingproxy_arguments` |
| **`ActorBudget` clamps overspend at the reservation** | A child spends beyond its reservation → the parent's budget bookkeeping goes negative | `test_actor_budget_settle_capped_at_reservation` |
| **Cancellation propagates via `ctx.cancel`** | Aborting the coordinator leaves 5 researchers running | `test_cancellation_mid_flight_propagates_to_children` (live coordinator dispatch) + `test_cancel_token_is_shared_reference_across_ctx_child` (structural invariant) in `tests/agents/test_coordinator_agent.py` |
| **`StreamEvent.type` is a `Literal`** | A typo in a consumer's branch (`"finl"` vs `"final"`) silently misses the terminal event | Type-level: `StreamEventType` Literal + mypy exhaustiveness on consumer branches |

## Correctness checklist (for future changes)

- **Coordinator dispatch loop must call `ctx.child()` per spawn.** Otherwise
  budget/cancellation/observation don't flow into children. Grep
  `.run(` in `agents/policies/*.py` — every call should be
  `child_agent.run(input, ctx.child())`, never `ctx` bare.
- **Every child agent's `RunContext` inherits the parent's `cancel` token.**
  Verify in `RunContext.child()` — the token is a shared reference, not
  a copy.
- **`Skill.as_agent()` deep-copy is not skippable.** If someone adds a
  "fast path" that shares cognition, the isolation test catches it. If
  they disable that test to make a benchmark pass, they broke the
  framework.
- **`SignalChannel.emit` uses `dataclasses.replace`, never in-place mutation.**
  Grep `signal.sender_id = ` in `channel.py` — should be zero matches.
- **The `merge` operator's cancellation-safe cleanup**: verified by
  `test_merge_early_break_from_infinite_sources_completes`. If a new
  operator is added to `streams.py`, replicate that pattern (split
  sentinel path on cancelled vs natural completion).
- **Coordinator's Suspended checkpoint carries the pending decision id
  verbatim.** Users approve/reject by that id; a rename or coercion
  breaks resume. Verified by the checkpointer end-to-end test.

## Where it can fail

Enumerated so a code reviewer can walk the list. Each is a real risk if
the framework's or the wiring's invariants slip.

### Framework-level failures (the framework's job to prevent)

1. **`Skill.as_agent()` stops deep-copying `cognition`.** A refactor
   that shares the cognition instance across materialisations → five
   researchers race on `MaxTurns.turn`. *Locked by
   `test_as_agent_cognitions_are_independent_instances` and
   `test_concurrent_as_agent_calls_do_not_share_termination_state` in
   `tests/skills/test_skill.py`.*
2. **`ReActCognition.drive` reassigns `self.termination` instead of
   using a local.** Under parallel dispatch of one shared Agent, two
   drives race on `self.termination = deepcopy(...)` — one clobbers
   the other. *Locked structurally by react.py:113 (local variable
   name `termination`) and threaded into `_iterate` at line 128; the
   test suite's live coordinator dispatch under
   `test_cancellation_mid_flight_propagates_to_children` exercises
   the concurrent-drive path.*
3. **`SignalChannel.emit` mutates the caller's envelope in place.**
   The caller's log line reads different fields than the queue's copy;
   audit and merge-loop disagree on identity. *Locked by
   `test_channel_emit_does_not_mutate_caller_signal` and
   `test_channel_emit_produces_same_stamped_copy_for_outbox_and_parent`
   in `tests/agents/test_signal_protocol.py`.*
4. **`SignalEnvelope` becomes non-frozen.** A helper adds a
   `set_sender` method → downstream code mutates `sender_id` post-emit
   → cascade attribution rewritten silently. *Locked by
   `test_signal_envelope_is_frozen` and `test_all_signal_types_are_frozen`.*
5. **`PlanPolicy` stops awaiting async planners.** A refactor iterates
   `planner.plan(...)` directly → an async planner returns a coroutine
   → `for step in coroutine` raises `TypeError`. *Locked by
   `test_workflow_with_async_planner_via_planpolicy_awaits_before_iterating`
   in `tests/agents/test_workflow.py`.*
6. **`run_agents` drops the semaphore.** Concurrency bound collapses;
   under load the coordinator spawns thousands of parallel researchers
   and hits the provider's rate limit. *Locked structurally by
   `concurrency.py:95-97` (`sem = ctx.semaphore()`; gather_bounded /
   gather_best_effort take `sem=`).*
7. **`ActorBudget.settle_child` stops capping at the reservation.**
   An over-spending child pushes the parent's `used_*` past its cap →
   downstream `exhausted()` never trips. *Locked by
   `test_actor_budget_settle_capped_at_reservation` in
   `tests/agents/test_signal_protocol.py`.*
8. **`merge` deadlocks on early consumer exit.** A pump's `finally`
   awaits a `queue.put` with a full queue and no reader → hang. *Locked
   by the dual-sentinel pattern in `streams.py:merge` (cancelled →
   `put_nowait` + `QueueFull` tolerance; natural → `await put`) and by
   `test_merge_early_break_from_infinite_sources_completes`.*

### Integration-level failures (the wiring's job to prevent)

1. **Researcher Skill wired without a `termination`.** With no smart
   stop, `MaxTurns` on `ReActCognition` is the only ceiling — one
   pathological researcher runs 8 turns and eats the whole budget.
   *Not a framework concern; the research app's Skill construction
   must always set a `termination`.*
2. **A tool is registered on the wrong Skill.** The Synthesizer gets
   `search` in its tool registry → it decides to "verify" a claim
   with a live search mid-synthesis, blowing the citation contract.
   *Wiring test in the research app must assert per-role tool sets.*
3. **`PlanPolicy(planner=None, steps=None)` called with neither.**
   PlanPolicy raises `ValueError`, but that's a runtime failure. A
   wire-up test should construct the coordinator once and dry-run to
   surface this at boot.
4. **Checkpointer not wired on the CoordinatorCognition (or on `ctx`).**
   The pre-synthesis `Suspended` fires; there's nowhere to persist
   the pending decision + corpus → resume can't reconstruct the run.
   *The app must construct with `CoordinatorCognition(..., checkpointer=<store>)`
   or ensure `ctx.checkpointer` is set.*
5. **StreamEvent transport doesn't respect per-agent order.** The api's
   `merge(...)` fan-in is correct; a naive WebSocket forwarder that
   re-orders by wall-clock timestamp corrupts the delta sequence for
   one agent. *Framework only guarantees per-agent stream order;
   transport must preserve that ordering on the wire.*
6. **`best_effort=True` swallowed on `PlanPolicy` construction.** With
   `best_effort=False` (default), one flaky researcher's exception
   cancels the whole group → the research run dies on a transient
   fetch error. *Wiring must pass `best_effort=True` for the research
   group.*

### Application-level failures (the research app's job to prevent)

1. **Planner prompt returns free-text steps instead of typed `Step`
   objects.** The Planner impl must coerce its LLM output through a
   Pydantic schema; a regex-parsed plan silently mis-routes.
2. **Synthesizer prompt drifts and starts synthesizing from prior
   knowledge instead of the approved corpus.** Framework can't detect
   this — the prompt is the app's contract. Only a critic run + a
   citation-existence check catches it.
3. **Researcher scores sources on a subjective "interesting"
   dimension.** Scores should be grounded on relevance + credibility
   heuristics, not aesthetic. The app owns the scoring prompt.
4. **Critic prompt is toothless.** A "review the synthesis" prompt
   with no adversarial framing rubber-stamps every draft. Contradiction
   flagging is an app-level correctness burden.
5. **Coordinator prompt attempts to override PlanPolicy's dispatch.**
   The coordinator Agent's prompt influences only the Planner call in
   this wiring; a stray "call the critic first" instruction is
   ignored, but confuses debugging. Keep the coordinator prompt
   minimal or empty when PlanPolicy owns dispatch.
6. **Sub-question strings echo user PII verbatim into search
   queries.** The research app's Planner prompt must sanitize; the
   framework has no visibility into query content.

## Expected output on a successful run

After the research run completes cleanly, the framework produces the
following concrete artifacts. Any deviation from these shapes on a green
run is a signal.

### The final `AgentResult` from the coordinator

```python
AgentResult(
    output="Solid-state batteries in 2026 are transitioning from lab...",  # the final brief
    usage=Usage(
        input_tokens=42_180,      # planner + 5 researchers + synth + critic
        output_tokens=8_940,
        cost_usd=0.612,
        cache_read_tokens=0,
        cache_write_tokens=0,
    ),
    partial=False,
    evals={
        "results": [
            AgentResult(output="<plan JSON>", ...),         # planner
            AgentResult(output="<findings q1>", ...),        # researcher 1
            AgentResult(output="<findings q2>", ...),        # researcher 2
            AgentResult(output="<findings q3>", ...),        # researcher 3
            AgentResult(output="<findings q4>", ...),        # researcher 4
            AgentResult(output="<findings q5>", ...),        # researcher 5
            AgentResult(output="<synthesis draft>", ...),    # synthesizer
            AgentResult(output="<critique + revised>", ...), # critic
        ],
        "errors": [],              # best_effort=True; empty on a fully-green run
        "stop_reason": "plan_complete",
    },
    parsed=None,                   # no top-level parser on the coordinator
    prompt_version="coordinator-inline",
)
```

Key checks:

- `partial=False` — no researcher failed a repair, no `max_iterations`
  ceiling tripped.
- `evals["errors"]` is empty. A best-effort slot with an exception
  would land here as `(child_name, exc)`. Non-empty on a "green" run
  is a red flag.
- `evals["stop_reason"] == "plan_complete"` — the plan exhausted all
  groups cleanly.
- `usage` sums across every step's LLM turns.

### Per-agent `StreamEvent` sequences

**Planner** (single-pass or short ReAct):

1. `StreamEvent("message_delta", text="{\"sub_")` ... — plan JSON streaming
2. `StreamEvent("message_delta", text="questions\": [")` ...
3. `StreamEvent("final", result=<Plan AgentResult>, usage=...)`

**Each Researcher** (typical shape):

1. `StreamEvent("message_delta", text="I'll search for ")` ...
2. `StreamEvent("tool_call", tool_call=ToolCall("t1", "search", {"query": "..."}))`
3. `StreamEvent("tool_result", tool_call=<same>, tool_result="<top 5 URLs>")`
4. `StreamEvent("step", text="iteration:1")`
5. `StreamEvent("message_delta", text="Fetching first source")` ...
6. `StreamEvent("tool_call", tool_call=ToolCall("t2", "fetch", {"url": "..."}))`
7. `StreamEvent("tool_result", ...)`
8. `StreamEvent("tool_call", tool_call=ToolCall("t3", "score_source", {...}))`
9. `StreamEvent("tool_result", ...)`
10. ... (multiple iterations)
11. `StreamEvent("final", result=<ResearchFindings AgentResult>, usage=...)`

**Synthesizer** (post-checkpoint, streaming prose):

1. `StreamEvent("message_delta", text="Overview\n\n")` — draft tokens
2. ... (many message_delta events)
3. `StreamEvent("final", result=<Synthesis AgentResult>, usage=...)`

**Critic** (analytical, may issue tool calls to re-fetch):

1. `StreamEvent("message_delta", text="Contradiction check:")` ...
2. Optional `tool_call` / `tool_result` for citation verification
3. `StreamEvent("final", result=<Critique AgentResult>, usage=...)`

**Coordinator** (aggregate; one terminal event by design):

1. `StreamEvent("final", result=<aggregate AgentResult>, usage=<merged>)`

The coordinator does NOT re-yield children's message deltas — those
flow through the api's `merge(...)` of the children's own streams. See
the CoordinatorCognition contract: policy runs to completion, one
terminal event.

### `SignalChannel` state (per Researcher)

After a Researcher finishes:

```python
researcher_channel.outbox
# Queue containing ~5-20 stamped ProgressSignal[Mutation] envelopes
# each with sender_id=researcher.agent_id, monotonic timestamp_us,
# then one DoneSignal at the end.

coordinator_channel.merge_inbox
# Queue containing ~25-100 (sender_id, stamped_signal) pairs across
# all 5 researchers. Same stamped references as each researcher's
# outbox — identity preserved.
```

Structural check: for any signal `s` in a researcher's `outbox`, the
corresponding pair `(researcher.agent_id, s)` MUST exist in the
coordinator's `merge_inbox` with an identical object identity (Python
`is`). If they differ, `dataclasses.replace` is being called twice
(regression) or a copy is being made in transit.

### `ActorBudget` bookkeeping per child

After a Researcher settles:

```python
# On the parent (coordinator):
coordinator_actor_budget.reserved_tokens   == 0       # released
coordinator_actor_budget.reserved_cost_usd == 0.0
coordinator_actor_budget.used_tokens       == <up to reservation, never over>
coordinator_actor_budget.used_cost_usd     == <up to reservation, never over>

# On the child (researcher):
researcher_actor_budget.used_tokens        == <actual spend, may exceed reservation>
researcher_actor_budget.used_cost_usd      == <actual spend, may exceed reservation>
```

The over-spend (if any) is visible on the child's own book — for
observability — but the parent's `used_*` MUST NOT exceed what was
reserved. That cap is what stops one runaway researcher from silently
consuming the whole run's envelope.

### Checkpointer state after the pre-synthesis checkpoint clears

Between the checkpoint firing and the operator's approve:

```python
checkpoint = {
    "run_id": "run-xyz",
    "status": "suspended",
    "state": {
        "phase": "pre_synthesis",
        "corpus": [<source_id>, ...],       # the researchers' output
        "pending": [<decision_id>],         # opaque id the operator returns verbatim
        "context": <serialized WorkingContext>,
        "usage": <serialized Usage so far>,
    },
    "created_at": <timestamp>,
}
```

On resume with the operator's approve command:

```python
checkpoint.status == "cleared"     # OR the row is deleted, impl-dependent
# The coordinator's state advances to "synthesis" phase and PlanPolicy
# resumes at group=1.
```

The `pending` id MUST round-trip: the operator returns the exact string
the checkpoint stored. A rename or coercion breaks the handshake.

## Verification protocol

How to actually check the design is working — not just "tests pass" but
"the invariants hold structurally in the current code".

### 1. Automated: run the invariant tests

```bash
cd agentkit
uv run pytest tests/skills/test_skill.py \
              tests/agents/test_signal_protocol.py \
              tests/agents/test_coordinator_agent.py \
              tests/agents/test_agent_loop.py \
              -k "as_agent or termination or channel or signal or coordinator"
```

Expected: all pass. Any failure is a live invariant violation. Adjacent
files that lock further invariants:

- `tests/agents/test_workflow.py` — `test_workflow_with_async_planner_via_planpolicy_awaits_before_iterating`.
- `tests/agents/test_handoff.py` — `test_handoff_tool_accepts_mappingproxy_arguments`.
- `tests/tools/test_function_tool.py` — `test_as_tool_forwards_task_from_mappingproxy_arguments`.
- `tests/agents/test_actor_budget.py` and `test_signal_protocol.py` —
  ActorBudget reservation + settle behaviour.

### 2. Structural: grep for load-bearing patterns

```bash
cd agentkit

# Per-spawn cognition isolation (Skill.as_agent deep-copy).
grep -n "deepcopy" agentkit/skills/skill.py
# Expected: line 106 — cognition=copy.deepcopy(self.cognition),

# Per-drive termination isolation (drive-local clone).
grep -n "termination" agentkit/agents/cognition/react.py
# Expected: 'termination = _copy.deepcopy(self.termination)' as a LOCAL
# variable, threaded into _iterate(...) as an argument. No line
# reassigning ``self.termination`` inside ``drive``.

# SignalChannel.emit uses replace-on-stamp (never mutation).
grep -n "dataclasses.replace" agentkit/agents/control/channel.py
# Expected: three matches (emit, send_to, try_send_to). Zero matches for
# ``signal.sender_id =`` or ``signal.timestamp_us =`` in this file.

# PlanPolicy awaits async planners.
grep -n "isawaitable" agentkit/agents/policies/plan.py
# Expected: 'steps = await plan if inspect.isawaitable(plan) else plan'

# run_agents grabs the tree semaphore.
grep -n "ctx.semaphore" agentkit/kernel/concurrency.py
# Expected: sem = ctx.semaphore() inside run_agents.
```

### 3. Adversarial: parallel dispatch smoke test

Build the smallest possible reproduction that exercises both isolation
seams — one Skill, spawned twice, dispatched in parallel on the same
Agent instance:

```python
# Not committed — one-off verification script.
import asyncio
import copy
from agentkit import Skill
from agentkit.agents.cognition import ReActCognition
from agentkit.agents.control.termination import MaxTurns

# One Skill, one shared cognition template with a mutable counter.
researcher_skill = Skill(
    name="researcher",
    description="finds sources",
    cognition=ReActCognition(tools=[...], termination=MaxTurns(3)),
)

# Materialise twice — deep-copy seam #1.
a1 = researcher_skill.as_agent()
a2 = researcher_skill.as_agent()
assert a1.cognition is not a2.cognition
assert a1.cognition.termination is not a2.cognition.termination

# Now the harder case: ONE Agent instance dispatched twice concurrently
# on its OWN cognition (the coordinator path via `children["researcher"]`).
one_agent = researcher_skill.as_agent()

async def drive_once(task):
    return await one_agent.run(task, ctx.child())

# Two concurrent drives share `one_agent.cognition.termination`.
# The drive-local deep-copy (seam #2) must isolate them.
r1, r2 = await asyncio.gather(
    drive_once("q1"),
    drive_once("q2"),
)
# Verify: each drive's termination counter was independent.
# If one drive's MaxTurns tripped at 3, the other's counter was NOT
# affected — both completed their full loops.
```

If this succeeds, both isolation seams are holding structurally.

### 4. What "failing" would look like

- Five researchers all terminate after 3 turns total when each has
  `MaxTurns(8)` → drive-local termination broke; they're sharing one
  counter.
- `researcher.outbox` and `coordinator.merge_inbox` contain **different
  object identities** for the same logical signal → `dataclasses.replace`
  is being called twice, or a copy is being made in transit.
- `AgentResult.evals["errors"]` contains an entry from a healthy
  researcher → `best_effort` wiring dropped, or a spurious exception
  is being swallowed by `gather_best_effort`.
- The pre-synthesis checkpoint fires, operator approves, and the
  resume path re-runs the researchers → the checkpoint's `pending`
  id was renamed or coerced.
- Coordinator returns `partial=True` on a run where every child
  terminated cleanly → someone's cognition tripped `max_iterations`
  instead of a natural stop; likely an override on the ReAct default
  that lost the termination condition.

## Design tensions to hold in mind

- **`SignalChannel` vs `RunContext.emit`**: two channels for two purposes.
  `emit` is product-facing observation (goes to the UI). `SignalChannel`
  is inter-agent coordination (goes parent → child or child → parent).
  Never conflate them.
- **`PlanPolicy` vs `RoundRobinPolicy` vs `SelectorPolicy`**: policy
  chooses HOW to dispatch, not WHO. All three are Strategy impls of the
  same abstract "given the coordinator + task + children, run something".
  For research, `PlanPolicy` is right because the plan itself is dynamic
  (LLM-generated).
- **`Handoff` vs `SelectorPolicy`**: two ways to route. `Handoff` is a
  typed value emitted by a child ("I'm done, next should be `bob`");
  `SelectorPolicy` is a coordinator-side chooser. For a strict pipeline
  (planner → researcher → synth → critic) neither fits perfectly;
  `PlanPolicy` with groups is cleaner because it separates "what to do"
  from "when to do it".
- **`best_effort=True` at the run_agents level**: one researcher failing
  shouldn't kill the whole run. Errors surface in
  `AgentResult.evals["errors"]` for the coordinator to include in its
  final synthesis.
- **Streaming to the UI**: the canvas subscribes to `ctx.emit` observations
  (Scene-level) AND to a per-agent `stream()` (token deltas). Two channels,
  interleaved via a `merge` in the transport layer (the api service, not
  agentkit). agentkit's job is only to ensure `StreamEvent` order-preserves
  per-agent.

## What this use case tests about agentkit (reverse view)

If a multi-agent research team produces cross-contaminated results, or
loses track of who said what, or dies half-way through and can't resume,
one of these framework invariants failed. The specific things this design
proves the framework must support:

1. `Skill.as_agent()` isolation is real, not just documented. Deep-copy at
   the `as_agent` seam AND at the `drive` seam means two paths where the
   framework must not accidentally re-share state.
2. `SignalChannel`'s frozen-envelope + replace-on-stamp story means the
   audit trail is trustworthy. A signal in the outbox and the same signal
   in the parent's merge_inbox are the SAME reference — no drift, no race.
3. `PlanPolicy` handles both sync and async planners without special-casing
   at the call site. Any Protocol saying "may be sync or async" must have
   every consumer honour it — `LedgerPolicy` did, `PlanPolicy` had to catch
   up (regression documented in the review pass).
4. `MappingProxyType` (arriving via `ToolCall.arguments`) flows through
   every tool boundary: `FunctionTool._invoke`, `as_tool._fn`,
   `handoff_tool._fn`, and the provider serializers
   (`openai_compat._to_msg`, `anthropic._to_msg`). Missing one is a
   silently-broken feature.
5. `ActorBudget` + shared `CancellationToken` mean the parent has real
   control over children — cost, concurrency, and lifecycle.

## Verification snapshot

Last audited against the current tree:

- **Automated invariant tests**: `pytest tests/skills/test_skill.py
  tests/agents/test_signal_protocol.py tests/agents/test_coordinator_agent.py
  tests/agents/test_agent_loop.py -k "as_agent or termination or channel
  or signal or coordinator"` → **44 passed, 0 failed**.
- **`Skill.as_agent()` deep-copies cognition**: `grep -n "deepcopy"
  agentkit/skills/skill.py` returns line 106 —
  `cognition=copy.deepcopy(self.cognition),`. Present as expected.
- **`ReActCognition.drive` uses a drive-local termination clone**: `grep
  -n "termination" agentkit/agents/cognition/react.py` returns 17 matches;
  react.py:113 assigns to a local `termination = _copy.deepcopy(...)`
  and threads it into `_iterate` at line 128. No `self.termination = ...`
  reassignment inside `drive`.
- **`SignalChannel` stamps via `dataclasses.replace`**: `grep -n
  "dataclasses.replace" agentkit/agents/control/channel.py` returns 3
  matches at lines 165 (`emit`), 183 (`send_to`), 195 (`try_send_to`).
  No in-place mutation of `sender_id` / `timestamp_us` in this file.

All three structural checks pass and the 44 tests in scope are green.
The design holds in the current tree.
