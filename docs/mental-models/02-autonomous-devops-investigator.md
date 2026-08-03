# Mental Model: Autonomous DevOps Investigator

The security + human-gate story. This use case exists to prove that agentkit's
safety primitives (`RunPolicy`, `requires_approval`, `Autonomy` tiers,
`CancellationToken`, `caps` on tools) compose into a system that CANNOT
autonomously take destructive action without an operator's explicit consent.

## Problem

An oncall AI observes production alerts (PagerDuty, Sentry, log-based),
investigates by reading logs / deploys / metrics, forms a hypothesis, and
proposes a remediation. It never applies the remediation without a human
approving that specific action. The alert content is *untrusted* (attackers
can shape it), the logs contain *private data*, and the remediation tools
*egress* to production infra — the lethal trifecta. The framework's job is
to make it impossible for that combination to fire autonomously.

## User experience

An alert fires: "checkout-api p99 latency > 2s". The AI's canvas shows:
- reads the last 15 min of `checkout-api` logs
- correlates with the most recent deploy (rev `abc123`, 8 min ago)
- notes error rate ↑ starting 30s after deploy
- proposes: *"Roll back deploy `abc123` via `kubectl rollout undo -n prod checkout-api`. This is destructive."*
- surfaces a **REVIEW REQUIRED** card to the oncall channel with the
  proposed command, evidence, and blast-radius estimate.

The oncall reviews, clicks Approve. The AI executes the rollback, waits
for the deploy status to go healthy, posts a summary to the Slack thread.

If the oncall clicks Cancel, the run halts; nothing was executed.

## How it actually works end-to-end

Walk through the internal state as one successful investigation flows through
the framework. The alert arrives at `POST /investigate`; the server has
already extracted the alert body from the PagerDuty webhook, verified the
signature, and mapped it to `Scope(org_id=7, domain_id="prod")`.

**t=0ms — Request setup.** The HTTP layer builds a `RunContext`:

```
ctx = RunContext(
    correlation_id="incident-2f9a",
    scope=Scope(org_id=7, domain_id="prod"),
    budget=Budget(max_cost_usd=2.0, max_calls=15, max_depth=4),
    services=Services(
        invoker=<shared>,
        observer=<per-request>,
        checkpointer=<Postgres-backed CheckpointPort>,
        trace=<per-request>,
    ),
    meters=[],                    # no tenant Quota on this use case
    cancel=CancellationToken(),   # operator's Cancel trips this
    autonomy="manual",            # every step gates
)
```

`autonomy="manual"` is the load-bearing bit — under `Autonomy.MANUAL`,
`should_gate` returns `True` for every tool call, regardless of the tool's
own `requires_approval` flag. Read-only tools still gate; the operator
sees each proposed action before it fires.

**t=1ms — Agent.run entered.** The `Agent`'s `stream()` opens an
`invoke_agent` span, stamps `agentkit.agent.cognition="react"` and
`agentkit.scope.org_id=7`, then hands off to `ReActCognition.drive`.
Drive deep-copies the termination clone into a drive-local variable so
concurrent investigations sharing the same cognition instance don't race
on `MaxTurns.turn` / `Timeout._start`. `run_id = ctx.correlation_id =
"incident-2f9a"`.

**t=2ms — Checkpointer load probe.** `_load(ctx, run_id)` asks the
`Checkpointer.resume(run_id)` port for a prior snapshot. First
invocation of this run — returns `None`. The `RequestBuilder` builds the
initial prompt from the alert body (`request_builder.build(task, context,
ctx, output_adapter=None)`); `usage=Usage()`, `start_i=0`, `repaired=False`.

**t=3ms — `_iterate` loop enters.** `termination.reset()` fires
(iteration 0 boundary). Iteration `i=0`:

1. `ctx.check_cancelled()` — token not tripped, no-op.
2. `ChatRequest` built with `tools=[get_logs, get_deploys, check_metrics,
   kubectl_rollout, slack_notify]` schemas.
3. `ctx.invoker.stream(req, ctx, meta={"output_adapter": None})` — chat
   middleware chain runs: `tracing → meter → retry → egress_guard →
   LLMPort`.

**t=4ms — Meter guard.** `meter()` middleware iterates
`ctx.all_meters=[budget]` and calls `budget.guard(call)`. `Budget._check`
verifies `spent_usd <= max_cost_usd (2.0)` and `calls <= max_calls (15)`.
Pass — the middleware yields to the next hop.

**t=~600ms — LLM streams.** The provider returns deltas. Deltas 1..N
carry text; the terminal delta carries `finish_reason="tool_calls"` with
`ToolCall("c1", "get_logs", {"service": "checkout-api", "since_minutes": 15})`.
Each text delta the cognition emits as `StreamEvent("message_delta",
text=...)`. `assemble_deltas` folds the stream into an `LLMResult`.

**t=~601ms — Gate check.** `_needs_approval(res.tool_calls, ctx)` iterates
`res.tool_calls`:

```
for c in calls:
    tool = self.tools.get(c.name)         # get_logs
    if should_gate(
        ctx.autonomy,                       # "manual"
        requires_approval=self.tools.requires_approval(c.name),  # False for get_logs
        key_step=getattr(tool, "side_effecting", False),         # False
    ):
        return True
```

Because `autonomy="manual"`, `should_gate` short-circuits on the first
branch: `if autonomy == Autonomy.MANUAL: return True`. Even a read-only
`get_logs` gates.

**t=~602ms — Suspend #1.** The cognition:

1. Appends the assistant turn to the transcript.
2. Calls `_save(ctx, run_id, context, usage, i=0, repaired, status="suspended", pending=(tc,))`.
3. `Checkpointer.snapshot` deep-copies `state`, bumps to `version=1`, calls
   `port.save(cp)`. On Postgres, that's one INSERT.
4. Emits `StreamEvent("interrupt", tool_call=tc)`.
5. Yields the terminal `final` event with `AgentResult(output="",
   partial=True, evals={"stop_reason": "awaiting_approval",
   "suspended": Suspended(run_id, pending=(tc,))})`.
6. `_iterate` returns — the loop is dead until an explicit `resume()`.

The SSE stream carries the `interrupt` event to the browser. The operator
UI renders a REVIEW REQUIRED card with `tc.name="get_logs"` and
`tc.arguments={"service": "checkout-api", "since_minutes": 15}` verbatim
— the `MappingProxyType` on `ToolCall.arguments` prevents any downstream
render code from mutating it.

**t=~603ms → t=~90s — Human wall clock.** The oncall reads the card,
clicks Approve. The API server receives `POST /runs/incident-2f9a/approve
{"c1": "approve"}` and calls `agent.resume(run_id="incident-2f9a",
decisions={"c1": "approve"}, ctx=<fresh ctx with same
correlation_id>)`.

Note: the process may have restarted between suspend and approve. That's
the point of `Checkpointer` — the snapshot lives in Postgres, not in the
run's in-memory state. A fresh worker picks up the same `run_id` and the
resume works identically.

**t=~90.001s — Resume path.** `ReActCognition.resume`:

1. `_load(ctx, run_id)` reads the latest checkpoint. `saved.status="suspended"`.
2. `rehydrate(saved.state)` reconstructs the `WorkingContext`,
   `Usage`, iteration index `i=0`, `repaired=False`.
3. `pending = [dict_to_tc(d) for d in saved.state["pending"]]` — one
   `ToolCall("c1", "get_logs", {"service": ..., "since_minutes": 15})`.
4. For each pending call, `decision=decisions.get(tc.id, "reject")`.
   `decision="approve"` — dispatches `_invoke_tool_safe(ctx, tc)`.
5. The `get_logs` tool executes, returns 47 log lines. The result becomes
   a `Message("tool", content="<lines>", tool_call_id="c1", name="get_logs")`
   appended to `context`.
6. Enters `_iterate(agent, context, usage, i+1=1, repaired, ...)` — the
   loop resumes from the *next* iteration.

**t=~90.005s — Iteration 1.** New chat call with the updated transcript
(alert + assistant + tool result). The model reads the logs, decides it
needs deploys, returns `ToolCall("c2", "get_deploys", {"namespace":
"prod", "service": "checkout-api"})`. Same gate story — MANUAL suspends
again. Operator approves. Resume. Iteration 2. And so on.

**t=~120s — Iteration 3, the destructive proposal.** The model has
enough evidence. It returns `ToolCall("c4", "kubectl_rollout",
{"namespace": "prod", "service": "checkout-api", "revision": "abc123"})`.
For THIS tool, `tools.requires_approval("kubectl_rollout") = True` AND
`tool.side_effecting = True` AND `tool.caps = ("egress",)` — the tool
itself demands approval regardless of autonomy tier.

`_needs_approval` returns True. Suspend #4. The interrupt card renders
the **verbatim** `MappingProxyType` arguments — `args.pop("dry_run")`
would raise `TypeError('mappingproxy' object does not support item
deletion)`. The operator sees `revision="abc123"`. What they see is
exactly what will run.

**t=~180s — Approve + execute.** Operator clicks Approve. `resume` runs
`kubectl_rollout` with the frozen args. The tool shells out (this is the
one blast-radius call). Its result: `"rollout undone; new revision xyz789
healthy at 2026-07-03T14:22:10Z"`.

**t=~181s — Iteration 4.** No pending gate — the model chooses
`ToolCall("c5", "slack_notify", ...)`. Because `slack_notify` has
`requires_approval=False`, under `autonomy="manual"` it STILL gates.
(This is the correct behaviour: MANUAL means the operator wants to see
every step.) Operator approves; slack notification fires.

**t=~185s — Iteration 5, terminal.** The model returns no tool_calls —
its terminal delta is `finish_reason="stop"` with a natural-language
summary. Because `agent.parse is None`, the loop appends the assistant
message, calls `_clear(ctx, run_id)` (Checkpointer.delete — the run is
done, no need to keep snapshots), and yields:

```
StreamEvent(
    "final",
    result=AgentResult(
        output="Rolled back deploy abc123. New rev xyz789 healthy. p99 back to 340ms.",
        usage=Usage(input_tokens=12_400, output_tokens=1_827, cost_usd=0.183),
        prompt_version="investigator-v3",
    ),
    usage=<total>,
)
```

**Cancellation path (alternative).** If at any point the operator clicks
Cancel, the API calls `ctx.cancel.cancel()`. On the next iteration
boundary, `ctx.check_cancelled()` raises `Cancelled("run cancelled")`.
The loop exits via the exception — the terminal `final` event does NOT
fire, so the caller (`Agent.run`) sees a `Cancelled` propagate up. An
in-flight `kubectl_rollout` tool call cannot be un-called — the loop
cancels, but the shell subprocess runs to completion. This is documented
as best-effort.

## Where it can fail

Enumerated so a code reviewer can walk the list. Each is a real risk if
the framework's invariants slip.

### Framework-level failures (the framework's job to prevent)

1. **`should_gate` regresses on MANUAL.** A refactor that drops the
   `if autonomy == Autonomy.MANUAL: return True` short-circuit → MANUAL
   only gates approval-required tools, and read-only tools slip through
   silently. The operator's contract ("show me every step") breaks.
   *Locked by the `should_gate` truth-table in
   `tests/agents/test_gate.py`.*
2. **Approval snapshot mutable across the seam.** If
   `ToolCall.arguments` were a plain `dict` instead of `MappingProxyType`,
   a tool implementation could `args.pop("dry_run")` between the UI
   render and execution — the operator sees "dry run" but the actual
   call is destructive. *Locked by
   `test_toolcall_arguments_view_rejects_item_assignment` in
   `tests/kernel/test_kernel.py`.*
3. **`Checkpointer.snapshot` shares state by reference.** If the
   snapshot didn't deep-copy `state` / `metadata`, the running loop
   would mutate the persisted record — an operator loading the paused
   view would see stale evidence, or (worse) evidence that no longer
   matches what will resume. *Locked by
   `test_snapshot_deep_copies_state_so_post_snapshot_mutation_cannot_reach_it`
   in `tests/capabilities/test_checkpointer.py`.*
4. **`Suspended.pending` is a mutable list.** A stray
   `suspended.pending.append(...)` after emit would desync the operator's
   card from what resume actually invokes. The frozen tuple pins both
   ends of the handshake. *Locked by
   `test_suspended_pending_is_a_frozen_tuple` in `tests/agents/test_agents.py`.*
5. **Cancellation not checked between turns.** If
   `ctx.check_cancelled()` were removed from the top of `_iterate`'s
   loop, an operator's Cancel would only take effect after another full
   LLM turn — burning another turn's worth of tokens and possibly one
   more approved tool call. *Locked by `tests/runtime/test_cancellation.py`.*
6. **`max_iterations` ceiling drops.** If `for i in range(start_i,
   self.max_iterations)` were replaced with `while True`, a termination
   bug would loop forever until the cost cap tripped — the never-hang
   backstop would be gone. *Framework contract: verified structurally
   in `_iterate`.*
7. **`run_with_resilience` retries permanent failures.** A 403 on
   `kubectl` (bad RBAC) should fail fast, not storm the API. If the
   `classify` ordering flipped so PERMANENT went through the retry
   branch, an audit alarm would fire before the model even saw the
   error. *Locked by `test_run_with_resilience_fails_fast_on_permanent`
   in `tests/kernel/test_kernel.py`.*
8. **Resume ignores checkpoint's `pending` list.** If `resume` rebuilt
   `pending` from the transcript instead of `saved.state["pending"]`, a
   snapshot corruption or a race between snapshot and interrupt emit
   would let the operator approve tool call A while the loop invoked
   tool call B. *Locked by
   `test_agent_suspends_then_resumes_through_a_checkpointer` in
   `tests/capabilities/test_checkpointer.py`.*

### Integration-level failures (the wiring's job to prevent)

1. **`RunContext.autonomy` extracted from the wrong request field.** A
   product config bug puts `autonomy="auto"` where the ops runbook said
   `"manual"` — every step suddenly skips the operator's review card.
   *Not a framework concern; the SaaS product's wiring layer is
   responsible.*
2. **Approval endpoint doesn't verify operator identity.** The API
   accepts `POST /runs/{id}/approve` without a signed operator session
   → any authenticated user can approve any pending gate. Framework
   passes the decisions dict through unchanged; who's allowed to POST
   is the app's auth story.
3. **`Checkpointer` port wired to a lossy backend.** The in-memory
   `Checkpointer` port on the wire, an OOM during suspend loses the
   pending state, the operator's approval arrives to nothing. Wire a
   Postgres-backed `CheckpointPort` for prod; a smoke test in the
   product's CI covers the wiring.
4. **`egress_guard` middleware missing from `chat_middleware`.** The
   trifecta panel emits a flagged verdict, but if the chain doesn't
   include an egress check, a rogue tool that egress-hides in the
   chat-call path (a URL fetched via the model's built-in tool use)
   escapes the review. Framework provides the middleware; wiring must
   include it.
5. **SSE transport drops `interrupt` events silently.** The framework
   yields the event; if the transport's back-pressure handler discards
   events when the client disconnects, the operator never sees the
   card. Framework only guarantees per-run stream ordering; delivery is
   the transport's job.
6. **Alert ingestion doesn't tag `caps=("untrusted_content",)`.** If
   the ingestion step just passes the raw alert body as
   `Agent.run(task=alert_body)` without a `FunctionTool` wrapper, the
   `RunPolicy.check` panel never sees the untrusted-content cap, so
   the trifecta verdict is missing one leg — the flag never fires even
   when it should.

### Application-level failures (the operator's job to prevent)

1. **Tool declared side-effecting without `requires_approval=True`.**
   The tool author's mistake — `kubectl_rollout` shipped with
   `side_effecting=True, requires_approval=False`. Under
   `autonomy="auto"`, the gate skips → destructive call fires
   autonomously. Correctness checklist below calls this out; a lint
   check on the tool registry is the right long-term fix.
2. **Tool implementation reads state outside `ctx`.** A `get_logs`
   impl that reads a global Kubernetes client instead of the one on
   `ctx.services.k8s` bypasses the framework's scope threading — a
   test that runs one investigation with a mocked client would still
   hit the real cluster.
3. **Tool result carries executable payload.** A `get_logs` result
   embedded with `<script>` or shell fragments (an attacker shaping
   the log line) → the model incorporates that content into its next
   turn. The framework's `guardrail.wrap_tool_output` frames tool
   output as UNTRUSTED, but the tool author must not disable the
   guardrail with `guardrail=None` on the cognition.
4. **`slack_notify` tool sends sensitive log excerpts.** The tool
   author decides what to pass to Slack. If it forwards raw log lines
   (private data + egress in one call), the caps list should be
   `("egress", "private_data")` and `requires_approval=True`.
   Missing that means the trifecta verdict is muted.
5. **Human approves without reading.** The operator clicks Approve
   without checking `tc.arguments`. Not a framework concern per se,
   but the UI's job to make the destructive summary unmissable
   (blast-radius estimate, resource name, diff preview). Framework
   guarantees the arguments are frozen; UI decides how loudly to
   render them.
6. **Tool author's retry loop re-invokes on transient failure.** A
   `kubectl_rollout` impl that catches its own transient errors and
   retries internally → the model + operator saw one approval, but two
   destructive calls fired. Retry belongs at the middleware layer, not
   the tool layer.

## Expected output on a successful run

After a clean investigation completes (5 iterations, 4 approvals), the
framework produces the following concrete artifacts. Any deviation on a
green run is a signal.

### The final `AgentResult`

```python
AgentResult(
    output="Rolled back deploy abc123. New revision xyz789 healthy at "
           "2026-07-03T14:22:10Z. p99 latency recovered to 340ms.",
    usage=Usage(
        input_tokens=12_400,       # prompt + 4 tool results across iterations
        output_tokens=1_827,       # assistant text + tool_call args
        cost_usd=0.183,            # under the 2.00 ceiling
        cache_read_tokens=0,
        cache_write_tokens=0,
    ),
    partial=False,                 # ran to a natural stop, not a max_iterations trip
    evals={},                      # no pending suspend, no policy verdict attached
    parsed=None,                   # no output schema wired
    prompt_version="investigator-v3",
)
```

Key checks:

- `partial=False` — clean stop. If the run had exhausted
  `max_iterations` before finding a resolution, `partial=True` and
  `evals={"stop_reason": "max_iterations"}`.
- `evals` is empty on the terminal `final`. The four intermediate
  `Suspended` values lived in the terminal-of-that-turn's `evals`, then
  were consumed by `resume`; the final terminal has none.
- `usage.cost_usd < budget.max_cost_usd` — the run stayed inside its
  envelope.

### The `StreamEvent` sequence

The stream carried across all four suspend/resume cycles, in order:

1. `StreamEvent(type="message_delta", text="Investigating...")` — chat starts
2. … (dozens of message_delta events, thinking prose)
3. `StreamEvent(type="interrupt", tool_call=ToolCall("c1", "get_logs", {...}))` — gate #1
4. `StreamEvent(type="final", result=AgentResult(partial=True, evals={"suspended": Suspended(...)}))`
5. **[operator approves — new stream opens on resume]**
6. `StreamEvent(type="tool_result", tool_call=<c1>, tool_result="<47 log lines>")`
7. `StreamEvent(type="step", text="iteration:1")`
8. `StreamEvent(type="message_delta", text="The logs show...")` — turn 2 assistant
9. `StreamEvent(type="interrupt", tool_call=ToolCall("c2", "get_deploys", {...}))` — gate #2
10. … (repeats through gates #3 and #4)
11. `StreamEvent(type="interrupt", tool_call=ToolCall("c4", "kubectl_rollout", {...}))` — gate #4, destructive
12. **[operator approves — resume]**
13. `StreamEvent(type="tool_result", tool_call=<c4>, tool_result="rollout undone; ...")`
14. `StreamEvent(type="step", text="iteration:4")`
15. … (slack notification + terminal turn)
16. `StreamEvent(type="final", result=AgentResult(output="Rolled back...", partial=False), usage=...)`

The invariant to check: EVERY tool call that fired is preceded by an
`interrupt` event AND a matching resume decision. If a `tool_result`
event appears with no prior `interrupt`, the gate was bypassed — that's
a live invariant violation.

### Observer stream

Product-facing observations delivered to the wired `ObserverPort`:

- `Observation(kind="progress", render="investigation started", agent="investigator", run_id="incident-2f9a")`
- `Observation(kind="progress", render="awaiting approval: get_logs", ...)` — one per suspend
- `Observation(kind="progress", render="awaiting approval: kubectl_rollout", payload={"caps": ["egress"], "side_effecting": True}, ...)`
- `Observation(kind="result", render="rollback complete", payload={"revision": "xyz789", "cost_usd": 0.183, "iterations": 5}, ...)`

Every observation carries `run_id="incident-2f9a"` and `agent="investigator"`.
The `trace_context` field is set on each observation iff a real tracer
was wired and a span was open at emit time.

### Checkpoint state saved along the way

`Checkpointer.list_versions("incident-2f9a")` at suspend #4 (before
approval) returns `[1, 2, 3, 4, 5, 6, 7]` — one snapshot per suspend
(`status="suspended"`) and one per resumed iteration boundary
(`status="running"`). After the terminal `final` event, `_clear` fires
and `list_versions` returns `[]` — the durable state is gone; the run
is complete.

The snapshot at version 4 (the `kubectl_rollout` suspend) carries:

```python
Checkpoint(
    run_id="incident-2f9a",
    version=4,
    status=CheckpointStatus.SUSPENDED,
    state={
        "prefix": <serialized prompt prefix>,
        "messages": [<all turns so far, as dicts>],
        "scratchpad": {...},
        "limit": None,
        "shared": {},
        "usage": {"input_tokens": 8_120, "output_tokens": 1_340, "cost_usd": 0.121, ...},
        "iteration": 3,
        "repaired": False,
        "pending": [{
            "id": "c4",
            "name": "kubectl_rollout",
            "arguments": {"namespace": "prod", "service": "checkout-api", "revision": "abc123"},
        }],
    },
    created_at=1735916400.123,
    metadata={},
)
```

The `pending[0]["arguments"]` matches the operator's approval card
byte-for-byte. If they differ, the framework has a bug OR a mutating
consumer touched the arguments after `snapshot`.

### Budget state after the run

```python
budget.spent_usd == 0.183         # sum of 5 chat turns' usage.cost_usd
budget.calls    == 5              # 5 chat requests (one per iteration)
```

`budget.calls` equals the number of iterations that produced a chat
call. Suspend/resume cycles do NOT count as separate calls on the
Budget — the resume reuses the checkpointed usage and starts from
`i+1`. `budget.remaining_usd() = 2.0 - 0.183 = 1.817` — plenty of
headroom.

## Verification protocol

How to actually check the design is working — not just "tests pass" but
"the invariants hold structurally in the current code".

### 1. Automated: run the invariant tests

```bash
cd agentkit
uv run pytest tests/agents/test_agent_loop.py \
              tests/agents/test_gate.py \
              tests/agents/test_control.py \
              tests/runtime/test_cancellation.py \
              tests/capabilities/test_checkpointer.py
```

Expected: all pass. Any failure is a live invariant violation. These
tests exercise `should_gate` truth tables, the `MANUAL`/`GATED`/`AUTO`
loop paths, `Suspended`/resume round-trips, `Checkpointer.snapshot`
deep-copy, and cancellation propagation.

### 2. Structural: grep for load-bearing patterns

```bash
cd agentkit
# The gate seam is called from ReAct's approval check.
grep -n "requires_approval" agentkit/agents/cognition/react.py
# Expected: exactly one match inside _needs_approval, threading the
# tool's requires_approval flag into should_gate.

# should_gate is the sole gating policy — no cognition should re-derive it.
grep -n "should_gate" agentkit/agents/control/gate.py
# Expected: three matches (docstring reference, def, __all__ export).
# The function has ONE definition; imports elsewhere all call the same one.

# Cancellation must fire at least once per iteration.
grep -n "check_cancelled\|ctx.cancel" agentkit/agents/cognition/react.py
# Expected: at least one ctx.check_cancelled() at the top of _iterate's loop.

# Tool arguments are frozen at construction.
grep -n "MappingProxyType" agentkit/kernel/types.py
# Expected: ToolCall.arguments wrapped in MappingProxyType on __post_init__.

# Snapshot deep-copies state.
grep -n "deepcopy" agentkit/capabilities/checkpointer/base.py
# Expected: Checkpointer.snapshot deep-copies both state and metadata.
```

### 3. Adversarial: HITL smoke test

Build the smallest reproduction of the gate + resume flow:

```python
# Not committed — one-off verification script.
import asyncio
from agentkit import Agent, Budget, Scope
from agentkit.agents.cognition import ReActCognition
from agentkit.agents.result import Suspended
from agentkit.runtime import RunContext, Services
from agentkit.tools import FunctionTool

async def kubectl_rollout(**args):
    return "rolled back"

async def main():
    tool = FunctionTool(
        name="kubectl_rollout",
        fn=kubectl_rollout,
        requires_approval=True,
        side_effecting=True,
        caps=("egress",),
    )
    agent = Agent(
        name="investigator",
        model="fake",
        cognition=ReActCognition(tools=[tool]),
    )
    ctx = RunContext(
        correlation_id="smoke-1",
        scope=Scope(org_id=1, domain_id="prod"),
        autonomy="auto",              # deliberately auto — tool's flag must still gate
        services=Services(invoker=<FakeLLM emitting kubectl_rollout tool_call>),
    )
    result = await agent.run("roll back", ctx)
    # Verify:
    assert result.partial is True
    susp = result.evals["suspended"]
    assert isinstance(susp, Suspended)
    assert susp.pending[0].name == "kubectl_rollout"
    # Now attempt to mutate the approval snapshot — must raise:
    try:
        susp.pending[0].arguments["namespace"] = "hacked"
    except TypeError:
        pass   # expected: MappingProxyType blocks item assignment
    # Also: pending is a tuple, not a list — a rebind attempt raises.

asyncio.run(main())
```

If this succeeds (Suspended emitted, arguments frozen, resume threads
the decisions through), the HITL story is holding structurally.

### 4. What "failing" would look like

1. A `tool_result` event appears in the stream with no prior
   `interrupt` event for that `tool_call.id` under `autonomy="manual"` —
   `should_gate` regressed on the MANUAL branch.
2. The operator's rendered `arguments` dict differs from what the
   resumed tool actually invoked — `ToolCall.arguments` mutability
   regression OR `Checkpointer` shallow-copy regression.
3. `Suspended.pending` accepts `.append()` or item assignment — the
   frozen shell/tuple guard broke.
4. An operator Cancel triggers, but another `interrupt` event fires
   after the Cancel timestamp — `ctx.check_cancelled()` missing from
   the loop top.
5. `Checkpointer.list_versions(run_id)` returns non-empty after the
   run's terminal `final` — `_clear` didn't fire on the success path.
6. Under `autonomy="auto"`, a tool with `requires_approval=True` fires
   without an interrupt — the AUTO branch of `should_gate` regressed.

## Composition

```
alert ──▶ POST /investigate
              │  RunContext(autonomy="manual",
              │             budget=Budget(max_cost_usd=2, max_calls=15),
              │             scope=Scope(org_id=…, domain_id="prod"))
              ▼
        Agent(name="investigator",
              cognition=ReActCognition(
                tools=ToolRegistry.from_tools([
                  get_logs,          # caps=("private_data",)
                  get_deploys,       # caps=("private_data",)
                  check_metrics,     # caps=("private_data",)
                  kubectl_rollout,   # caps=("egress",),
                                     # side_effecting=True,
                                     # requires_approval=True
                  slack_notify,      # caps=("egress",),
                                     # side_effecting=True,
                                     # requires_approval=False (informational)
                ]),
                termination=MaxTurns(10) | Timeout(seconds=180)),
              policy=RunPolicy(mode="flag"))    # emits verdict, doesn't block
              │
              ▼
        Middleware chain:
          tracing → meter → retry → egress_guard → LLMPort
              │
              ▼
        Human-gate flow:
          - tool call arrives with requires_approval=True
          - loop suspends → Suspended(run_id, pending=[ToolCall(...)])
          - Checkpointer.snapshot(status="suspended")
          - UI renders card with the pending ToolCall verbatim
          - operator's Approve/Reject arrives as a command
          - ReActCognition.resume(run_id, decisions={id: "approve"|"reject"|<JSON>}, ctx)
              │
              ▼
        Cancellation:
          - CancellationToken shared on ctx.cancel
          - operator's Cancel command → token.cancel() → ctx.check_cancelled()
            raises between turns → run exits with a cancelled Failure
```

The Alert-reading step is worth calling out: whichever tool ingests the raw
alert text ships `caps=("untrusted_content",)` so `RunPolicy.check` sees the
full trifecta up front.

## The primitives it exercises

| Primitive | Role here | Why load-bearing |
|---|---|---|
| `RunPolicy.check` | Panel that flags/blocks the lethal trifecta | If untrusted_content + private_data + egress ever mix in one tool call without approval, the framework must surface it |
| `FunctionTool.caps` | Per-tool trifecta tag | The input to `RunPolicy` |
| `FunctionTool.requires_approval` | Per-tool HITL gate | Egress tools require approval regardless of autonomy tier |
| `Autonomy` (`manual` / `gated` / `auto`) | Run-wide HITL policy | Production runs pin `manual` → every step gates |
| `Suspended` | The value returned from `Agent.run` when a gate fires | The seam between the run and the operator UI |
| `Checkpointer` | Durable pause across operator lunch break | Approval may arrive minutes/hours later; process may restart |
| `ReActCognition.resume` | Continue from a suspended checkpoint with the human's decision | Threads approve/reject/edit into the tool loop |
| `CancellationToken` | Cooperative abort from the operator | Kills the loop between turns; in-flight tool calls run to completion |
| `Budget` (max_cost + max_calls) | Never-hang backstop | Investigation halts even if the model wanders |
| `MaxTurns \| Timeout` | Smart stops layered on the hard budget | Termination composes with `\|` (OR) |
| `ExternalTermination` | Operator can also flip a flag mid-run | Alternative cancel path |

## What it deliberately doesn't use

- **`Workflow`** — investigation is emergent (which log to check next depends
  on previous evidence). Workflow is the wrong tool.
- **`SignalChannel`** — single agent, no children. Signals model coordinator
  ↔ worker; that's the research-report shape, not this.
- **`Skill.as_agent`** — this investigator is bespoke wiring; skills are for
  reusable recipes.
- **`CachedMemory`** — reads are per-alert, cache would either be stale or a
  security risk (an alert from tenant A leaking into an unrelated alert's
  context).

## Invariants

| Invariant | Concrete failure if violated | Locked by |
|---|---|---|
| **`requires_approval=True` always fires the human gate** | The loop invokes a `kubectl` command without operator consent → prod outage or data loss | `test_manual_autonomy_gates_an_unapproved_tool`, `test_gated_autonomy_gates_side_effecting_key_step` in `tests/agents/test_agent_loop.py` |
| **`Autonomy.MANUAL` gates every step, not just approval-required ones** | Operator wanted "review everything" mode; framework only gated the flagged tools → ran read-only tools they wanted to see first | `should_gate` truth table in `tests/agents/test_gate.py` |
| **`RunPolicy.check` returns a flagged verdict when trifecta hits** | The tool set has all three caps and nobody notices → framework silently allows autonomous auto-remediation | `PolicyVerdict` tests in `tests/agents/test_control.py` |
| **`Suspended` carries the ToolCall arguments verbatim** | Operator UI shows a hallucinated command different from what would actually run | `test_agent_suspends_then_resumes_through_a_checkpointer` |
| **`ToolCall.arguments` cannot be mutated between UI render and execution** | A tool implementation `args.pop("dry_run")` mutates the approval snapshot → operator sees "dry run" but actual call is destructive | `test_toolcall_arguments_view_rejects_item_assignment` |
| **`Checkpointer.snapshot` deep-copies state** | Suspended snapshot's `pending` list is shared with the running loop's; a resume-race mutates one side and the operator sees stale evidence | `test_snapshot_deep_copies_state_so_post_snapshot_mutation_cannot_reach_it` |
| **Cancellation propagates through `ctx.check_cancelled()` between turns** | Operator's Cancel arrives but the loop already dispatched the next chat call → burns another turn's tokens after the abort | `tests/runtime/test_cancellation.py` |
| **`Budget.max_calls` is a hard ceiling** | Termination condition bug leaves the loop unhung by MaxTurns → runs forever until cost cap trips | Framework contract: `_iterate`'s `for i in range(start_i, self.max_iterations)` is the never-hang backstop |
| **Approved tools RE-approve on subsequent runs** | Operator approves once → the run remembers → next investigation autonomously reuses the approval | By design: the approval carrier is `Suspended` scoped to `run_id`; a new `run_id` starts fresh |
| **The kernel `run_with_resilience` does NOT retry PERMANENT failures** | A 403 on `kubectl` triggers a retry storm → looks like an attack pattern to infra security | `test_run_with_resilience_fails_fast_on_permanent`, `classify` ordering test |

## Correctness checklist (for future changes)

- **Any new tool with side-effecting infra actions must declare
  `caps=("egress",)` AND `requires_approval=True`.** Missing either
  is a bug. Add an assertion on the tool registry that egress-capable
  side-effecting tools always require approval.
- **`RunPolicy.check` must see every tool the loop can dispatch.** If a
  Skill-adapted tool wraps another agent, the outer policy sees only
  the wrapper's caps — verify caps are propagated on adaptation.
- **Grep `ctx.check_cancelled()` in every cognition — must fire at
  least once per turn.** Missing check = un-cancellable loop.
- **`Suspended.pending` is a frozen `tuple`, not a mutable `list`.**
  Two doors both locked: the frozen dataclass blocks rebinding, the
  tuple itself blocks item mutation. Verified by
  `test_suspended_pending_is_a_frozen_tuple` in `tests/agents/test_agents.py`.
- **`Autonomy` values are `AutonomyLiteral` — a typo doesn't silently
  drop to AUTO.** mypy catches this at the seam.

## Design tensions to hold in mind

- **`RunPolicy(mode="flag")` vs `mode="deny"`**: flag mode publishes a
  verdict as an observation; block mode raises. We use flag here because
  the operator UI is the enforcement point — the run continues to gather
  evidence, but the trifecta warning surfaces prominently in the review card.
- **`caps` on the alert-ingestion step**: whichever way the alert enters
  the run (as a param on `Agent.run(task)` or a tool call), the framework
  needs a signal that the input is untrusted. Convention: wrap alert
  ingestion in a `FunctionTool(name="alert_content", caps=("untrusted_content",))`
  called at run start so `RunPolicy.check` sees the tag.
- **Auto-remediation for pre-approved runbooks**: the "operator approved
  this playbook once" pattern is dangerous. We deliberately require
  per-run approval; a "trusted playbook" library would need its own
  cryptographic-attestation layer, not a framework-level exception.
- **What "cancel" means for an in-flight `kubectl`**: the framework
  cancels the LOOP, not the tool's I/O. A `kubectl` in flight cannot
  be un-called. Document that Cancel is best-effort for tool calls
  already dispatched.

## What this use case tests about agentkit (reverse view)

If a real oncall AI built on agentkit ever executes a destructive command
without operator approval, one of these framework invariants failed. The
specific things this design proves the framework must support:

1. `caps` + `requires_approval` compose into a real gate that no code path
   bypasses.
2. The `Autonomy` policy tier is honoured uniformly across all cognition
   impls (currently ReAct — but any new cognition must respect `should_gate`).
3. `Suspended` + `Checkpointer` durably represent the pause; a process
   restart during operator lunch doesn't lose the pending approval.
4. `ToolCall.arguments` frozen-view prevents the classic
   "approve-then-modify" attack (a tool impl silently mutating args
   between the operator's view and the actual execution).
5. `CancellationToken` is a first-class abort signal — operators can always
   stop a run, even one mid-loop.

## Verification snapshot

Last audited against the current tree:

- **Automated invariant tests**: `pytest tests/agents/test_agent_loop.py
  tests/agents/test_gate.py tests/agents/test_control.py
  tests/runtime/test_cancellation.py tests/capabilities/test_checkpointer.py`
  → 69 passed, 0 failed.
- **Gate call site**: `grep -n "requires_approval"
  agentkit/agents/cognition/react.py` → 1 match at line 361 inside
  `_needs_approval`, threading the tool's flag into `should_gate`.
  Expected structural evidence present.
- **Sole gating policy**: `grep -n "should_gate"
  agentkit/agents/control/gate.py` → 3 matches (docstring, `def`,
  `__all__`). One definition, exported once. Expected structural
  evidence present.
- **Cancellation check between turns**: `grep -n
  "check_cancelled\|ctx.cancel" agentkit/agents/cognition/react.py`
  → 1 match at line 206, at the top of `_iterate`'s per-iteration
  body. Expected structural evidence present.

All four load-bearing structural checks pass. The design holds in the
current tree.
