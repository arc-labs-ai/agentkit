# Mental Model: Multi-Tenant "Chat with Docs" SaaS

The tenant-isolation story. This use case exists to prove that the framework
can safely serve many customers from one process without one tenant's data ever
leaking into another's context — even under concurrent load, even when the
in-process cache is warm, even when a stray retry middleware retries a request
that a different tenant is also making.

## Problem

Multi-tenant SaaS is the archetypal cross-cutting-concern challenge. Data
isolation cannot be a promise the application layer remembers to enforce; it
has to be a property the framework enforces **structurally**. The threat
model isn't primarily malice — the more common failure is a well-meaning
optimization (a shared cache, a shared memoize middleware, a "helpful"
global rate limiter) that reads or writes across the tenant axis by accident.

A tenant is identified by `Scope(org_id, domain_id)`. Every layer that
touches persistent state (memory, cache, quota meter, audit log) must key
its state on that scope. The framework's job is to make that keying:

- **Explicit** — no primitive silently unpartitions across tenants.
- **Ergonomic** — the app doesn't thread scope by hand through every call;
  it lives on `RunContext.scope` and every consumer reads from there.
- **Verifiable** — a grep or a test can prove tenant isolation across
  every hop.

## User experience

Company A signs up. Their SOC 2 evidence PDFs get uploaded and indexed under
`Scope(org_id=1, domain_id=1)`. Company B does the same under `Scope(2, 1)`.

At `t=0`, Ellen (org 1) asks *"What's our incident-response protocol?"* At
`t=3ms`, Priya (org 2) asks the exact same question. Both requests hit the
same server process, the same LLM adapter, the same `InMemoryStore`, the
same warm CachedMemory.

Ellen sees her company's real protocol, cited to her SOC 2 doc. Priya sees
her company's protocol, cited to her SOC 2 doc. Neither sees a hint the
other exists. Their token counts accrue to their own tenant Quota, not to
each other's.

Ellen finishes her chat, refreshes, asks the same question again. This time
CachedMemory has her result — instant reply, no LLM call, no wasted spend
— but *only* for Ellen's tenant. Priya asking the same question a minute
later still misses the cache in her partition.

## How it actually works end-to-end

Walk through the internal state as one successful request flows through the
framework. Ellen's `POST /chat` arrives; the server has already extracted
her `Scope(1, 1)` from her JWT.

**t=0ms — Request setup.** The HTTP layer builds a `RunContext`:

```
ctx = RunContext(
    correlation_id="chat-abc123",
    scope=Scope(org_id=1, domain_id=1),
    budget=Budget(max_cost_usd=0.10, max_calls=8),
    services=Services(invoker=<shared>, observer=<per-request>,
                      trace=<per-request>),
    meters=[quota_per_org],       # tenant-partitioned by scope.key()
    cancel=CancellationToken(),
    autonomy="auto",
)
```

The `invoker` and `quota_per_org` are **shared** across all tenants — that's
what makes the framework cheap. What partitions is `scope`.

**t=1ms — Agent.run entered.** The Agent's ReActCognition begins its drive.
It resolves the RequestBuilder, deep-copies its termination condition (per
`AKT34` — no cross-drive state), then hands off to `_iterate`.

**t=2ms — Invoker.stream called.** ReAct constructs a `ChatRequest`,
threads the output_adapter through `meta={"output_adapter": None}` (nothing
structured here), and calls `ctx.invoker.stream(req, ctx, meta=...)`. The
Invoker wraps this into a `Call("chat", req, ctx, meta={...})` and runs
the chat-middleware chain.

**t=3ms — Meter guard fires.** The `meter()` middleware runs
`on_request`. It iterates `ctx.all_meters` — which is `[budget, quota]` —
and calls `guard(call)` on each. `Budget.guard` checks the run-wide cost
ceiling. `Quota.guard` derives `key = call.ctx.scope.key() = "org1:dom1"`
and checks THAT partition's RPM / TPM / USD windows. If either meter is
over ceiling, `MeterExceeded` raises here, before any spend.

**t=4ms — LLM stream begins.** The chain terminates in the LLM adapter's
`stream(...)`. Deltas start flowing back:
- Delta 1: `text="Your"`, no tool_calls, no usage
- Delta 2: `text=" incident"`, ...
- Delta N (terminal): `text="", finish_reason="tool_calls",
  tool_calls=(ToolCall("c1", "search_docs", {"query": "incident response protocol"}),)`

Each `text` delta the cognition emits as `StreamEvent("message_delta", text=...)`
so the SSE handler can push it to Ellen's browser in real time.

**t=~800ms — Assemble deltas, dispatch tool.** `assemble_deltas` folds
the stream into an `LLMResult(content="Your incident...", tool_calls=(tc,))`.
Because the response has tool_calls, ReAct appends the assistant turn to
the transcript and dispatches the tool via `ctx.invoker.invoke_tool(...)`.

**t=~801ms — Tool executes.** The `search_docs` FunctionTool's inner
callable reaches into memory:

```
memory = ScopedMemory(               # tenant guard
    CachedMemory(                    # in-proc TTL cache
        VectorMemory(index=...)      # concrete backend
    )
)
```

- `ScopedMemory._check(ctx)` runs FIRST — reads `ctx.scope.org_id=1` and
  `ctx.scope.domain_id=1`, both truthy, returns True. If ANY layer forgot
  to thread scope, this is the choke point that raises `PermissionError`.
- `CachedMemory.query(query, k=5, ctx, where=None)` computes
  `key = ("org1:dom1", "incident response protocol", 5, "{}")` and looks
  it up. On the first call — miss. Delegates to `inner.query`.
- `VectorMemory.search(scope=Scope(1,1), query="incident...", k=5)` returns
  three `MemoryItem` objects from Ellen's tenant's index. Different vector
  backends implement scope differently (namespace, filter, separate index);
  the port contract just says "search within this scope".
- CachedMemory stores `(now, list(items))` — a defensive copy — under the
  scope-partitioned key. Returns `list(items)` — another defensive copy —
  to the caller.

**t=~850ms — Tool result flows back.** The `search_docs` result becomes a
`Message(role="tool", content="<chunks joined>", tool_call_id="c1")` and
lands in the transcript. ReAct emits `StreamEvent("tool_result", tool_call=..., tool_result=...)`.

**t=~851ms — Next LLM turn.** The transcript now has the tool result;
another `invoker.stream(...)` call. The LLM synthesizes a final answer citing
the chunks. This time there are no tool_calls — the terminal delta has
`finish_reason="stop"`.

**t=~1400ms — Charge + emit final.** Meter middleware's `on_response`
fires: `budget.charge(call, usage)` and `quota.charge(call, usage)` each
land the spend under `scope.key()="org1:dom1"`. `budget.spent_usd` for THIS
run goes up; `quota._charges["org1:dom1"]` gets a new entry. Ellen's meter
moved; the shared meter's other partitions are untouched.

**t=~1401ms — Final event.** ReAct yields
`StreamEvent("final", result=AgentResult(output="Your incident response...", usage=<total>, ...), usage=<total>)`.
The SSE handler closes the stream.

Meanwhile, Priya's parallel request has been running through the same
Invoker, same LLM adapter, same CachedMemory instance — but with
`ctx.scope=Scope(2, 1)`. Every scope-keyed lookup lands in a different
partition. Ellen's cache entries are invisible to Priya's queries.

## Composition sketch

```
UI  ──▶  chat endpoint
              │  scope = Scope(org_id=A, domain_id=…)
              ▼
        Agent(name="assistant",
              cognition=ReActCognition(tools=[search_docs]),
              output=None)
              │
              ▼
        Middleware chain (shared Invoker; per-tenant Quota):
          tracing → meter(budget + per-tenant Quota) →
          retry → LLMPort
              │
              ▼
        search_docs = FunctionTool wrapping:
          ScopedMemory(               # tenant gate
            CachedMemory(              # in-proc TTL cache, scope-partitioned
              VectorMemory(            # concrete backend
                scope=ctx.scope        # tenant-scoped index namespace
              )))
```

**Adapter contract**: LLM/Store adapters MUST NOT retain state keyed on anything other than the process/instance. Per-tenant isolation lives in `scope`, threaded through `RunContext.child()`, NOT inside adapters.

## Where it can fail

Enumerated so a code reviewer can walk the list. Each is a real risk if
the framework's invariants slip.

### Framework-level failures (the framework's job to prevent)

1. **Cache key drops the scope axis.** A refactor to `CachedMemory._key`
   that forgets the `scope_key` argument → tenant B's identical query hits
   tenant A's cache entry. *Locked by `test_cache_is_partitioned_by_scope`.*
2. **Cache returns internal list by reference.** A caller that mutates
   the returned list corrupts the cache for the whole TTL window,
   affecting subsequent hits from the same tenant. *Locked by
   `test_cached_result_is_isolated_from_caller_mutation`.*
3. **`where` dict with unhashable values crashes the lookup.** Filters
   like `{"tags": ["a", "b"]}` would break the cache key. *Locked by
   `test_cache_accepts_unhashable_where_values`.*
4. **Meter middleware not iterating `ctx.all_meters`.** A regression that
   only guards `Budget` and skips per-tenant `Quota` → runaway spend
   under load. *Locked by
   `test_quota_isolates_tenants_end_to_end_through_meter_middleware`.*
5. **Meter iteration order changes.** Currently `Budget` is first, so
   its trip short-circuits `Quota.charge`. A reordering could double-charge
   or under-charge. *Locked by
   `test_budget_and_quota_both_engaged_through_meter_middleware`.*
6. **ScopedMemory's enforce becomes bypassable.** If `ScopedMemory._check`
   raised only warnings instead of `PermissionError`, a scope-less run
   would silently reach VectorMemory. *Locked by
   `test_scoped_memory_rejects_missing_scope`.*
7. **`RunContext.child()` doesn't inherit scope.** A sub-agent (e.g., a
   Skill invoked as a tool) would run without a tenant identity → the
   ScopedMemory gate would fail loud, but only after the sub-run had
   already started. *Structurally verified: `child()` sets
   `scope=self.scope` (shared reference).*

### Integration-level failures (the wiring's job to prevent)

1. **`Invoker` wiring forgets the meter middleware.** No `meter()` in
   `chat_middleware` → no Quota enforcement → tenant B's runaway consumption
   burns tenant A's budget. *Not lockable at the framework layer; a
   wire-up smoke test in the SaaS product's tests is the right place.*
2. **`RunContext.scope` extracted from the wrong request field.** JWT
   parsing bug puts tenant B's scope into tenant A's `RunContext`. *Not
   a framework concern; the SaaS product's auth layer.*
3. **`VectorMemory` backend doesn't honour the scope on `search`.** A
   Pinecone / Weaviate namespace bug returns cross-tenant vectors even
   though the scope was correctly threaded. *The `VectorPort` contract
   requires per-scope isolation; the adapter carries this contract.*
4. **Streaming transport crosses tenant sessions.** The framework yields
   `StreamEvent`s; whichever transport (SSE, WebSocket) fans them out
   must not route tenant A's tokens onto tenant B's connection. *Transport
   layer's job; the framework only guarantees per-run stream ordering.*

### Application-level failures (the app's job to prevent)

1. **Tools written that ignore `ctx`.** A `search_docs` implementation that
   reads a global connection pool without threading scope would read across
   tenants. Only the tool author can prevent this — but the framework's
   `Tool.run(args, ctx)` signature at least surfaces `ctx` so the author has
   the tenant context available.
2. **Custom `MemorySource` doesn't respect scope in `write`.** The
   `MemorySource.write` contract asks the impl to key by `ctx.scope`; a
   custom impl that forgets writes tenant A's items into the shared
   namespace. Same principle: the framework provides `ctx`; the impl
   must use it.
3. **`scope.org_id=0` is treated as unscoped** by `_default_enforce` (falsy int). If your tenant IDs start at 0, override the enforce callable or use non-zero IDs.

## Invariants matrix

Every row is a property the framework MUST hold. Each row names the test
that locks it in — if you're changing code that touches these areas,
verify the test still asserts what its name claims.

| Invariant | Concrete failure if violated | Locked by |
|---|---|---|
| **Scope partitions the cache key** — `(scope_key, query, k, where)` | Tenant B's identical query hits tenant A's cached items → cross-tenant read | `test_cache_is_partitioned_by_scope` in `tests/memory/test_cached_memory.py` |
| **`ScopedMemory.enforce` cannot be bypassed** at wire-time | A caller layering `CachedMemory(VectorMemory)` without the outer guard reads every tenant's index | `test_scoped_memory_rejects_missing_scope` in `tests/memory/test_scoped_memory.py` |
| **`where` dict may contain unhashable values** without crashing the cache | A filter like `{"tags": ["security","finance"]}` blows up the key lookup, forcing repeated round-trips | `test_cache_accepts_unhashable_where_values` |
| **Cache reads return a defensive copy** | UI dedupes the returned list → mutates the cache → next tenant's read sees the dedup | `test_cached_result_is_isolated_from_caller_mutation` |
| **Cache key stable across dict ordering** | Semantically-equal filters double-fetch → cost overrun + non-deterministic behaviour | `test_cache_key_stable_across_where_dict_ordering` |
| **Per-tenant Quota clamps runaway spend** | A stuck retry loop on tenant B burns tenant A's budget | `test_quota_isolates_tenants_end_to_end_through_meter_middleware`, `test_budget_and_quota_both_engaged_through_meter_middleware` in `tests/runtime/test_meter.py` |
| **Streaming per response is in-order** | Interleaved responses corrupt one user's transcript with another's tokens | Framework-level: `assemble_deltas` reducer + property tests on stream operators |
| **`RunContext.child()` inherits scope** | A sub-agent runs without tenant identity; ScopedMemory then blocks it, but only after the child was already spawned | Structural — `RunContext.child` sets `scope=self.scope` |
| **Every retrieval tool declares `caps=("private_data",)`** | Adding an egress tool later doesn't auto-flag the emergent trifecta | Needs a lint check — currently a gap; a `RunPolicy` panel test would cover it |

## Expected output on a successful run

After Ellen's chat completes cleanly, the framework produces the following
concrete artifacts. Any deviation from these shapes on a green run is a
signal.

### The final `AgentResult`

```python
AgentResult(
    output="Your incident-response protocol at Company A begins with...",
    usage=Usage(
        input_tokens=1247,      # prompt + tool result
        output_tokens=312,      # assistant tokens
        cost_usd=0.0034,        # computed by adapter
        cache_read_tokens=0,    # no prompt-cache hit on cold turn
        cache_write_tokens=0,
    ),
    partial=False,              # ran to a stop, not a repair-exhausted fallback
    evals={},                   # no HITL suspend, no policy verdict attached
    parsed=None,                # no structured output adapter
    prompt_version="researcher-chat-v1",  # from the wired RequestBuilder
)
```

Key checks:

- `partial=False` — a healthy run completes; `partial=True` signals a
  `max_iterations` ceiling, an unrepaired `parse` failure, or a suspended
  HITL run. On a chat, only `parse` failure would produce partial (and this
  use case doesn't wire `parse`).
- `evals` is empty — this run had no HITL gates, no policy verdicts, no
  step-level errors surviving the loop.
- `usage.cost_usd` matches the sum of every LLM turn's charge on this run.

### The `StreamEvent` sequence

The stream that flows to the SSE handler (ordered):

1. `StreamEvent(type="message_delta", text="Your")` — first token
2. `StreamEvent(type="message_delta", text=" incident")` — subsequent tokens
3. … (dozens of message_delta events until the model requests a tool call)
4. `StreamEvent(type="tool_call", tool_call=ToolCall("c1", "search_docs", {"query": "..."}))`
5. `StreamEvent(type="tool_result", tool_call=<same>, tool_result=<chunks joined>)`
6. `StreamEvent(type="step", text="iteration:1")`
7. `StreamEvent(type="message_delta", text="Your")` — start of turn 2
8. … (more text deltas)
9. `StreamEvent(type="final", result=AgentResult(...), usage=Usage(...))`

If you saw an `interrupt` event, that would mean a `requires_approval=True`
tool was invoked. This use case wires no such tool.

### Observer stream

Product-facing observations delivered to the wired `ObserverPort`:

- `Observation(kind="progress", render="chat run starting", agent="assistant", run_id="chat-abc123", ...)`
- Zero-or-more `Observation(kind="partial_result", ...)` if the app emits them
- Exactly one `Observation(kind="result", render="chat done", payload={"tokens": 312}, ...)`

The `run_id` on every observation equals `ctx.correlation_id="chat-abc123"`.
The `trace_context` field is set iff a tracer was wired and a span was open
at emit time.

### Meter state after the run

Assuming tenants 1 and 2 both ran once:

```python
budget.spent_usd    == 0.0034 + ...    # shared: the run-wide budget
budget.calls        == 2                # both chat turns counted
quota._charges["org1:dom1"]  # one entry for Ellen's run
quota._charges["org2:dom1"]  # one entry for Priya's run
```

`quota._charges` MUST have two distinct keys with one entry each. If they
share a key or one is missing an entry, isolation broke.

### CachedMemory state after the run

If Ellen's query hit the memory tool once with `k=5`:

```python
cached_memory._cache
# {("org1:dom1", "incident response protocol", 5, "{}"): (now_ts, [<3 items>])}
```

One entry, keyed on the tenant partition. Priya's identical query would
add a second entry with `("org2:dom1", ...)`.

## Verification protocol

How to actually check the design is working — not just "tests pass" but
"the invariants hold structurally in the current code".

### 1. Automated: run the tests locked to the invariants

```bash
cd agentkit
uv run pytest tests/memory/test_cached_memory.py \
              tests/memory/test_scoped_memory.py \
              tests/runtime/test_meter.py
```

Expected: all pass. Any failure is a live invariant violation.

### 2. Structural: grep for load-bearing scope threading

```bash
cd agentkit
# Every RunContext.child call must NOT drop scope.
grep -n "RunContext(" agentkit/runtime/context.py
# Expected: only one production construction; child() passes scope=self.scope.

# Every MemorySource impl reads ctx.scope in query/write.
grep -rn "async def query" agentkit/memory/
grep -rn "async def write" agentkit/memory/
# Expected: every impl's signature takes `ctx: Ctx` and uses it.

# CachedMemory._key includes scope_key.
grep -n "def _key" agentkit/memory/decorators.py
# Expected: signature has scope_key parameter and it's included in the tuple.
```

### 3. Adversarial: two-tenant smoke test

Build the smallest possible reproduction:

```python
# Not committed — one-off verification script.
import asyncio
from agentkit import Agent, Budget, Scope
from agentkit.memory import CachedMemory, ScopedMemory
from agentkit.runtime import Invoker, Quota, RunContext, Services
from agentkit.middlewares import meter, tracing

quota = Quota(max_usd=0.05, clock=lambda: 1000.0)
invoker = Invoker(llm=<FakeLLM>, chat_middleware=[meter()])

async def one_run(org_id: int):
    ctx = RunContext(
        correlation_id=f"run-{org_id}",
        scope=Scope(org_id=org_id, domain_id=1),
        services=Services(invoker=invoker),
        meters=[quota],
    )
    # ... run an Agent with a memory tool ...

await asyncio.gather(one_run(1), one_run(2))
# Verify: quota._charges has two distinct keys ("org1:dom1" and "org2:dom1"),
# each with exactly the entries that tenant produced. No key overlap.
```

If this succeeds, the tenant-isolation story is holding structurally.

### 4. What "failing" would look like

- `quota._charges["org1:dom1"]` has an entry that came from tenant 2 → cache
  key axis missing scope.
- `CachedMemory._cache` has an entry keyed only on `(query, k, where)` with
  no scope prefix → key regression.
- Tenant 2's Agent returns tenant 1's document text → VectorMemory scope
  filter regression OR ScopedMemory bypass.
- Both tenants exhaust the same quota window at the same time → per-tenant
  partitioning collapsed to a single shared bucket.

## Correctness checklist (for future changes)

When touching any of these files, re-verify against this use case:

- `agentkit/memory/decorators.py` — `_key(...)` must always include the
  scope axis. Grep: `def _key`, verify signature carries a scope arg.
- `agentkit/memory/decorators.py` — `ScopedMemory._check` runs BEFORE
  any inner call in `query` / `write`. Grep: `self._check(ctx)` at
  entry of both methods.
- `agentkit/adapters/store/*.py` — any `get`/`set` where key isn't
  scoped is a leak surface. All `StorePort` writes routed through
  `idempotency_key(scope, ...)` or an equivalent prefix.
- `agentkit/runtime/context.py` — `RunContext.child()` inherits scope
  (verify), so a child agent operates on the same tenant.

## The primitives it exercises

| Primitive | Role here | Why load-bearing |
|---|---|---|
| `RunContext.scope` | Tenant identity, threaded via `ctx.scope` on every call | The only piece of state that tells any layer "who is this for" |
| `ScopedMemory` | Fail-loud tenant guard around VectorMemory | Refuses on missing scope; last line of defence if a caller forgets |
| `CachedMemory` | TTL cache for identical repeated queries | Cheap wins on repeated lookups within a session |
| `VectorMemory` | Backend semantic search, indexed per-tenant | Only surface that actually touches other tenants' bytes |
| `ReActCognition` | Tool-calling loop over the retrieval tool | Multi-turn refinement, "search more" reruns |
| `FunctionTool.caps` | `("private_data",)` on `search_docs` | Feeds `RunPolicy` to catch trifecta later if egress tools get added |
| `Budget` + per-tenant `Quota` meter | Cost/tokens per tenant per window | Prevents one abusive tenant from spending the entire pool |
| `StreamEvent` | Token streaming to the UI | User-facing progressive rendering |

## What it deliberately doesn't use

- **`Checkpointer`** — chats are ephemeral; no crash-resume story.
- **`SignalChannel`** — single agent, no coordination.
- **`RunPolicy` block-mode** — retrieval-only trifecta stays under threshold
  (no egress, no untrusted-content), so flag-mode is enough.
- **`Workflow`** — no deterministic graph; conversation is emergent.
- **`Skill.as_agent`** — this Agent is bespoke to the product, not a
  reusable recipe.

## Design tensions to hold in mind

- **Shared vs per-tenant Invoker**: shared is cheaper (one middleware
  chain, one httpx pool) but harder to reason about isolation. We
  chose shared because `ctx.scope` is threaded end-to-end; if that
  invariant ever breaks, we lose the cheap-shared property.
- **Cache TTL vs eviction on write**: writes invalidate ALL cache
  entries currently (`CachedMemory.write` clears `_cache`). Fine for
  low-write patterns; if writes are hot, a per-tenant partition +
  scoped invalidation is the next step.
- **Streaming ordering when the LLM adapter is shared**: assumes the
  provider's SSE stream is per-request. Never share a Delta stream
  between two Calls. Structurally enforced by `Call` being one-per-request.

## What this use case tests about agentkit (reverse view)

If this design breaks in production, the framework has a bug. The specific
things this design proves the framework must support:

1. Scope threading works end-to-end from HTTP entry → middleware → tool → memory.
2. No in-process shared cache silently unpartitions tenants.
3. The `MemorySource` decorator stack is safe to compose in any order that
   makes semantic sense (`ScopedMemory` outermost, `CachedMemory` inside).
4. `MappingProxyType` (`ToolCall.arguments`) round-trips cleanly through the
   tool boundary even when args carry per-tenant filters.

## Verification snapshot

Last audited against the current tree:

- **Automated invariant tests**: `pytest tests/memory/test_cached_memory.py
  tests/memory/test_scoped_memory.py tests/runtime/test_meter.py` → 29
  passed, 0 failed.
- **`RunContext.child()` inherits scope**: `scope=self.scope` (shared
  reference, read-only) — verified in `agentkit/runtime/context.py:child`.
- **`CachedMemory._key` signature includes scope**: `_key(query, k, where,
  scope_key)` — the scope axis is a required positional argument, not
  optional.
- **`ScopedMemory._check` fires on both entry points**: `query` at line 71
  and `write` at line 75 both call `self._check(ctx)` before delegating to
  the inner source.

All four load-bearing structural checks pass. The design holds in the
current tree.
