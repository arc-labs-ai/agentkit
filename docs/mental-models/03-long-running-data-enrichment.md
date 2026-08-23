# Mental Model: Long-Running Data Enrichment (10k rows)

The durability + cost story. This use case exists to prove that agentkit's
`Workflow` + `Checkpointer` + `Budget` primitives compose into a system that
can process large batches over long durations, survive crashes cleanly, and
never overshoot a budget cap under concurrent workers.

## Problem

A batch job enriches 10,000 vendor records with a summary + a risk category
via an LLM. Runs on a cheap worker for ~2 hours. The worker might crash,
the model might get rate-limited, the cost budget must be respected across
concurrent parallel workers, and re-running after a crash must not
re-enrich rows already done.

## User experience

An operator submits: *"Enrich all vendors updated in the last 7 days.
Budget: $50. Concurrency: 8."*

The job starts. A progress bar reports: *"2,347 / 10,000 done · $12.30
spent · 8 workers active · est. 18 min remaining."*

The worker OS gets OOM-killed at 4,100 rows. The operator sees the last
checkpoint (`4,000 rows`, `$21.15 spent`) and resumes with a single command.
The job picks up at row 4,001, respects the remaining $28.85 of budget, and
completes.

## How it actually works end-to-end

Walk through the internal state as a successful 10k-row batch flows through
the framework. Times are wall-clock at a shared LLM adapter with typical
provider latency; the framework's own overhead per row is single-digit ms.

**t=0 — Workflow assembly.** The batch job's entry point builds the top-level
graph and the per-row worker:

```
worker = Agent(
    name="enricher",
    model="gpt-4o-mini",
    cognition=SingleCallCognition(),
    output=EnrichedVendor,           # Pydantic → typed output
    max_repairs=1,
)

wf = (
    Workflow(name="vendor-enrichment", max_steps=1)
      .fn("load_rows", load_rows_from_db)
      .subworkflow("enrich_batch", per_batch_wf, after="load_rows")
)
```

The `RunContext` for the run carries `budget=Budget(max_cost_usd=50.0,
max_concurrency=8)`, a `store=InMemoryStore()` (Redis in prod), and a
`checkpointer=Checkpointer(port=<CheckpointPort>)`. `ctx.correlation_id`
is `"enrich-2026-07-03-abc"`. Every child context (`ctx.child()`) shares
the same `Budget`, `store`, and `checkpointer` by reference — that is the
whole point of the tree: shared meters, shared durability.

**t=~10ms — load_rows.** The first fn-node runs to completion, returning
10,000 row dicts. Its `Usage()` is zero (no LLM). `Workflow._execute` merges
that into the running `usage` (still zero) and commits `done["load_rows"] =
[<10k rows>]`. The wave advances to `enrich_batch`.

**t=~11ms — per-batch dispatch begins.** `per_batch_wf` reads
`done["load_rows"]` and slices it into chunks of ~100. For chunk 0:

```
pairs = [(worker, row) for row in chunk]     # 100 pairs
results = await run_agents(pairs, ctx, best_effort=True)
```

`run_agents` calls `ctx.semaphore()` which returns the Budget's cached
`asyncio.Semaphore(max_concurrency=8)`. This semaphore is the ONE authority
on how many LLM turns are in flight across the whole tree — every fan-out
in the run acquires from it, so a stray sub-agent cannot double the
concurrency by creating its own semaphore.

**t=~12ms — per-row Agent + SingleCallCognition.** For row 0, `worker.run(row,
ctx.child())` enters. `SingleCallCognition` builds one `ChatRequest` (system
prompt + row content, `response_format=EnrichedVendor.model_json_schema()`),
threads the `output_adapter` through `meta`, and calls
`ctx.invoker.stream(req, ctx, meta=...)`.

**t=~13ms — Budget guard fires.** The `meter()` middleware runs `on_request`
and calls `budget.guard(call)`. Under the budget's lock, `spent_usd` is
compared to `max_cost_usd`. On row 0 with `spent_usd=0.0`, guard passes.
The call proceeds down the chain to the LLM adapter.

**t=~13ms — Semaphore acquire.** Concurrently with the meter guard, up to
8 rows have already crossed into `_run` inside `gather_bounded`, each
holding one permit of the shared `asyncio.Semaphore`. Rows 9…99 are
suspended on `async with sem:` and will resume only as earlier rows'
coroutines release. This is the ONLY concurrency ceiling — there is no
worker pool.

**t=~13ms — idempotency_key computation.** Before dispatching the LLM
call, the per-batch code wraps the worker call in
`ctx.store.get_or_set(key, lambda: worker.run(row, ctx.child()))` where
`key = idempotency_key(ctx.scope, "enrich", row["id"], row["updated_at"])`.
`idempotency_key` is `"idem:" + stable_hash(parts, length=24)`;
`stable_hash` sorts JSON keys and routes unknowns through `_stable_default`
so the digest is identical across processes. On a fresh run, the store
misses, the producer callable runs, and the returned `AgentResult` is
stored under that key.

**t=~800ms — First row's LLM stream returns.** The single-shot response
carries `EnrichedVendor` JSON. `SingleCallCognition` invokes the output
adapter, coerces to the Pydantic model, and returns
`AgentResult(output=<json text>, usage=Usage(input=…, output=…, cost=0.0021),
parsed=EnrichedVendor(...), partial=False)`.

**t=~800ms — Budget.charge fires.** `meter()`'s `on_response` calls
`budget.charge(call, usage)`. Under the lock:
`self.spent_usd = round(self.spent_usd + usage.cost_usd, 6); self.calls += 1`.
Because the lock serialises charges from ALL 8 concurrent workers, the
final total is invariant under interleaving. `Usage.__add__` is
associative + commutative (Hypothesis-verified), so per-worker Usage
folding into the shared meter converges to the same number regardless of
order.

**t=~1s — First 8 rows have returned.** The semaphore releases 8 permits;
rows 9…16 acquire and dispatch. `gather_bounded` preserves input order in
the returned `list[R]` — even if row 12 completes before row 8, the result
slot ordering is preserved because each task has its own index.

**t=~30s — Chunk 0 done (~100 rows).** `run_agents` returns 100 `AgentResult`
entries (best_effort mode; each raised exception is wrapped into a `Failure` —
a plain frozen dataclass carrying `category` / `source` / `message` / `cause`,
NOT an exception — so callers detect with `isinstance(res, Failure)`.
`isinstance(res, BaseException)` misses every one of them, because a `Failure`
is data rather than something raised). `per_batch_wf` folds the results into `done_ids` and
computes cumulative cost.

**t=~30s — First checkpoint save.** `per_batch_wf` calls:

```
await ctx.checkpointer.snapshot(
    run_id=ctx.correlation_id,
    state={"done_ids": list(done_ids), "next_chunk": 1, "spent_usd": budget.spent_usd},
    status=CheckpointStatus.RUNNING,
    metadata={"chunk": 0, "rows_in_chunk": 100},
)
```

Inside `Checkpointer.snapshot`, the per-run lock is acquired, the current
version list is read, `next_version = max+1` is computed, and BOTH `state`
and `metadata` are `copy.deepcopy`'d before being stored. Any live mutation
of `done_ids` in the hot loop after the snapshot returns cannot reach the
persisted record — this is the load-bearing invariant behind resume.

**t=~30s — Checkpoint cadence.** The batch chooses "every chunk" (~100 rows)
as its cadence. Too frequent kills throughput (each snapshot is a
deep-copy + a port write); too infrequent means a crash loses more work.
100 rows per snapshot is the deliberate trade — production tuning depends
on per-row cost.

**t=~5min — 10th chunk done, ~1000 rows.** `budget.spent_usd ≈ $2.10`,
`budget.calls == 1000`, 10 checkpoints (versions 1…10) exist for
`run_id`. The Budget's semaphore has been continuously oversubscribed
(100 rows in the batch, 8 permits) — steady-state 8 workers active.

**t=~10min — Crash.** The worker process is OOM-killed at row 4,100 mid-way
through chunk 41. The 40 chunks already snapshotted are durable; the 100
rows partially in flight in chunk 40 either landed in the store (via
`get_or_set`) or did not. The most recent RUNNING checkpoint is version 40,
state `{"done_ids": [1..4000], "next_chunk": 40, "spent_usd": 21.15}`.

**t=~11min — Resume.** Operator restarts the process; the batch job's
entry point calls `saved = await ctx.checkpointer.resume(ctx.correlation_id)`.
`saved.state` is a deep-copy of what was persisted; the resumer reads
`done_ids = set(saved.state["done_ids"])` and `next_chunk = 40`. A
fresh `Budget` is constructed with `spent_usd = saved.state["spent_usd"]`
so `remaining_usd()` is `50 - 21.15 = $28.85`.

**t=~11min — Row-level idempotency during resume.** The resumer iterates
chunks 40…99. For chunk 40, it dispatches all 100 rows again — but each
call goes through `ctx.store.get_or_set(idempotency_key(...), ...)`. Rows
that landed in the store during the first run (some of chunk 40's rows
completed before the OOM) return the cached `AgentResult` immediately with
`usage=Usage()` (no re-charge). Rows whose producer callable raised or
never ran are re-dispatched — the "producer that raises is NOT stored"
contract on `StorePort.get_or_set` ensures no failure poisons the cache.

**t=~11min+ — Resume proceeds.** New chunks (41…99) snapshot as before,
version numbers continue monotonically from 41. The Budget accumulates
into the merged `spent_usd`, cap intact.

**t=~2h — Terminal.** Chunk 99 completes. `per_batch_wf` returns its final
`WorkflowResult`. The top-level `Workflow` records `stop_reason="complete"`,
merges usage, and returns. The batch job's entry point calls
`await ctx.checkpointer.snapshot(run_id, {...}, status=CheckpointStatus.DONE)`
one last time — this final version's status is terminal, and the persistence
port may prune non-latest versions per its policy.

## Where it can fail

Enumerated so a code reviewer can walk the list. Each is a real risk if
the framework's invariants slip.

### Framework-level failures (the framework's job to prevent)

1. **Snapshot stores a live reference instead of a deep-copy.** A refactor
   to `Checkpointer.snapshot` that drops `copy.deepcopy(state)` → the hot
   loop's continued mutation of `done_ids` corrupts every prior checkpoint
   on `InMemoryStore` (and diverges from serializing backends). *Locked by
   `test_snapshot_deep_copies_state_so_post_snapshot_mutation_cannot_reach_it`.*
2. **Deep-copy fails on nested containers or `MappingProxyType`.** State
   carrying a `ToolCall.arguments` (a `MappingProxyType`) fails pickle on
   Redis/Postgres → resume is broken on any real backend. *Locked by
   `test_snapshot_state_containing_toolcalls_survives_deepcopy`.*
3. **`Usage.__add__` becomes non-associative.** Concurrent workers fold
   Usage in different orders → cost totals drift → cost cap trips
   inconsistently across runs. *Locked by `test_usage_add_is_associative`
   and `test_usage_add_is_commutative` (Hypothesis-generated).*
4. **`Usage` loses `frozen=True`.** A worker mutates a shared Usage
   snapshot → concurrent workers' charges corrupt each other's view of
   spend. *Locked by `test_kernel_value_types_are_frozen`.*
5. **`idempotency_key` becomes non-deterministic across processes.**
   `stable_hash` starts including a memory address or `dict.__repr__`
   → resume computes different keys than the original run → every row
   re-enriches → cost doubles. *Locked by
   `test_idempotency_key_deterministic_across_calls` and
   `test_stable_hash_ignores_dict_iteration_order`.*
6. **`gather_bounded` violates its concurrency cap.** A regression that
   drops the `async with sem:` guard or replaces the shared semaphore
   with a per-task one → provider rate-limits everything → cascading
   429s. *Locked by `test_gather_bounded_preserves_order_and_bounds`.*
7. **`Workflow._execute` drops strict-zip on wave commit.** A silent
   arity mismatch (bounded gather returns fewer items than dispatched)
   silently loses a row's result. *Locked by
   `test_workflow_wave_raises_when_node_run_returns_wrong_arity`.*
8. **`Checkpointer.snapshot` skips the per-run lock.** Two concurrent
   snapshots for the same `run_id` both compute `next_version = max+1`
   off the same read → the second `port.save` either clobbers the first
   (silent loss) or raises a duplicate-key error. *Locked by
   `test_concurrent_snapshots_of_same_run_id_produce_distinct_versions`.*
9. **`Budget.charge` skips its lock.** 8 concurrent charges each read
   `spent_usd` before any writes back → the run overshoots
   `max_cost_usd` by up to 8× the last delta. *Locked by
   `test_concurrent_charges_serialize` in `tests/runtime/test_meter.py`.*
10. **`run_with_resilience` retries a PERMANENT class.** A 403 or
    content-filter refusal retries anyway, burning tokens and possibly
    banning the key. *Locked by
    `test_run_with_resilience_fails_fast_on_permanent`.*

### Integration-level failures (the wiring's job to prevent)

1. **`idempotency_key` parts don't uniquely identify the row.** The
   batch author uses `(scope, "enrich", row["id"])` on a table where
   `id` is stable across schema versions but the `updated_at` column
   changes the desired output. Resume returns stale results because
   the same key is used. *A batch-job smoke test comparing pre- and
   post-schema-migration output is the right place to catch this;
   not a framework concern.*
2. **Checkpoint cadence choice is wrong for the row cost.** Cadence of
   every 100 rows on 500ms-per-row content is fine; on 30s-per-row
   content, an OOM in the middle of a chunk loses 30 minutes of work.
   *The batch author's responsibility to tune; verify with a
   deliberately-killed dry run.*
3. **`Budget` shared instance is not passed to child contexts.** The
   batch job constructs a fresh `Budget` inside `per_batch_wf` instead
   of reading from the parent — every chunk gets its own $50 cap.
   *Not a framework failure; the wiring must reuse the parent's
   `Budget` reference.*
4. **`StorePort` chosen has stale semantics on retry.** `InMemoryStore`
   is fine single-process; a real cluster with Redis must configure
   TTL long enough that a resume 6 hours after a crash still sees the
   cached rows. *Wiring's responsibility; contract on `StorePort` is
   silent about TTL policy.*
5. **The Workflow's `max_steps` is set below the number of chunks.**
   With 100 chunks and `max_steps=1` on the top-level Workflow, the
   nested `per_batch_wf` steps don't count against it — but if a
   refactor flattens the graph, the batch stalls at `max_steps` with
   `stop_reason="max_steps"`. *Not framework failure; wiring choice.*
6. **The `output=EnrichedVendor` schema drifts from the store's cached
   rows.** A resume after a schema change loads cached `AgentResult`
   whose `parsed` field is the old shape. *The batch author must
   invalidate the store or bump the idempotency key on schema change.*

### Application-level failures (the row-tool's job to prevent)

1. **The per-row worker's prompt reads a global variable.** Two
   concurrent workers on rows with different content share prompt
   state via a module-level dict → cross-row contamination in the LLM
   input. Only the worker author can prevent this.
2. **The Pydantic model has a field with a default_factory that mutates
   shared state.** Each row's `EnrichedVendor` instance mutates a
   class-level list → results carry across rows. Framework can't help.
3. **The custom `load_rows_from_db` fn returns a generator instead of
   a list.** The Workflow snapshot deep-copies but the generator is
   exhausted mid-way → downstream chunks are empty. Row tool must
   return a materialised list.
4. **The row worker uses `time.sleep` inside its cognition.** Blocks
   the whole event loop → all 8 workers idle even under a semaphore.
   Framework mandates async everywhere; app code that calls sync I/O
   defeats the concurrency.
5. **The row worker catches every exception and returns a fake
   success.** Every "failure" becomes a stored `AgentResult` with
   garbage content → the store caches garbage → resume returns garbage.
   The framework's `Failure` value type is the right shape for
   best-effort failures; app-level try/except must use it.

## Expected output on a successful run

After the 10k-row job completes cleanly, the framework produces the
following concrete artifacts. Any deviation from these shapes on a green
run is a signal.

### The final `WorkflowResult`

```python
WorkflowResult(
    outputs={
        "load_rows": [<10000 row dicts>],
        "enrich_batch": WorkflowResult(               # nested per_batch_wf result
            outputs={
                "chunk_0": [<100 AgentResult>],
                "chunk_1": [<100 AgentResult>],
                # …
                "chunk_99": [<100 AgentResult>],
            },
            usage=Usage(input_tokens=8_200_000,
                        output_tokens=1_100_000,
                        cost_usd=45.31),
            steps=101,                                # load_rows + 100 chunks
            stop_reason="complete",
            suspended=None,
        ),
    },
    usage=Usage(input_tokens=8_200_000,
                output_tokens=1_100_000,
                cost_usd=45.31),
    steps=2,                                          # load_rows + enrich_batch
    stop_reason="complete",
    suspended=None,
)
```

Key checks:

- `stop_reason == "complete"` — a healthy run terminates here;
  `"suspended"` means a human-gate is open (this use case wires none);
  `"max_steps"` means the graph cycled; `"deadlock"` means the DAG had a
  cycle.
- `usage.cost_usd < budget.max_cost_usd` — should be strictly less; the
  cap trips as a `MeterExceeded`, never as a silent zero-cost row.
- `suspended is None` — non-null means the run paused for a decision.

### The per-row `AgentResult` (one of 10,000)

```python
AgentResult(
    output='{"vendor_id": 42, "summary": "…", "risk": "medium"}',
    usage=Usage(input_tokens=820, output_tokens=110, cost_usd=0.0021),
    partial=False,                        # parse succeeded on first try (or on repair)
    evals={},                             # no HITL, no policy verdicts
    parsed=EnrichedVendor(vendor_id=42, summary="…", risk="medium"),
    prompt_version="enricher-v1",
)
```

If a row's `partial=True`, its output text is the raw model reply but
`parsed` is `None` — the output adapter's parse-and-repair loop exhausted
`max_repairs`. `best_effort=True` on the batch means such rows land as
`AgentResult(partial=True, parsed=None, ...)` (not as `Failure`); the
batch author reconciles the `partial=True` slice separately.

### Checkpoint state after the run completes

The final `snapshot(status=DONE)` writes one terminal version. Prior
RUNNING versions may remain (per-port policy) or may be pruned by an
explicit `ctx.checkpointer.delete(run_id)` call in the batch's terminal
handler. On `InMemoryStore`:

```python
await ctx.checkpointer.list_versions(run_id)
# [1, 2, 3, …, 100, 101]   — 100 RUNNING + 1 DONE

cp = await ctx.checkpointer.resume(run_id)
cp.status                    # CheckpointStatus.DONE
cp.state["done_ids"]         # [1, 2, 3, …, 10000]
cp.state["spent_usd"]        # 45.31
```

If the batch's terminal handler calls `delete(run_id)`, all versions are
gone and `resume(run_id)` returns `None`. That's the "reclaim" pattern.

### Budget state

```python
budget.spent_usd    # 45.31
budget.calls        # 10000       — every per-row Agent counts as one call
budget.max_cost_usd # 50.0
budget.remaining_usd()  # 4.69
```

`budget.calls` must equal `10000` on a green run; if it's less, some rows
were served from `StorePort` cache (a resume situation) and their
`usage=Usage()` charged nothing. If it's more, retry middleware fired on
transient errors — the `retry` middleware's charges accumulate as expected.

### StorePort state (idempotency)

```python
# 10000 keys, one per row.
[k for k in store._data.keys() if k.startswith("idem:")]   # 10000 entries
# Value shape:
store._data["idem:<hash>"]     # AgentResult(parsed=EnrichedVendor(…), …)
```

On a resumed run, the store will end with the same 10k keys — rows served
from cache and rows dispatched fresh both land in the same store shape.

## Verification protocol

How to actually check the design is working — not just "tests pass" but
"the invariants hold structurally in the current code".

### 1. Automated: run the invariant tests

```bash
cd agentkit
uv run pytest tests/capabilities/test_checkpointer.py \
              tests/agents/test_workflow.py \
              tests/runtime/test_meter.py \
              tests/kernel/test_kernel.py \
              -k "usage or idempotency or gather"
```

Expected: all pass. The `-k` filter narrows to the four load-bearing
invariants of this use case; a full run of the four files also exercises
adjacent properties.

### 2. Structural: grep for load-bearing patterns

```bash
cd agentkit

# Checkpointer.snapshot deep-copies BOTH state and metadata at the seam.
grep -n "deepcopy" agentkit/capabilities/checkpointer/base.py
# Expected: at least two hits (state and metadata), both inside snapshot().

# gather_bounded exists and is the only fan-out primitive used by run_agents.
grep -n "gather_bounded\|max_concurrency" agentkit/kernel/concurrency.py
# Expected: gather_bounded function definition + dispatch inside run_agents.
# (max_concurrency lives in Budget, not concurrency.py — that's fine.)

# idempotency_key composes with stable_hash; deterministic across processes.
grep -n "def idempotency_key" agentkit/kernel/resilience.py
# Expected: a single definition delegating to stable_hash on *parts.

# Usage.__add__ is defined on the frozen dataclass.
grep -n "def __add__\|frozen=True" agentkit/kernel/types.py
# Expected: __add__ on Usage, frozen=True on the dataclass.
```

### 3. Adversarial: crash-and-resume smoke test

Build the smallest possible reproduction:

```python
# Not committed — one-off verification script.
import asyncio
from agentkit import Agent, Budget, RunContext, Services
from agentkit.adapters.store import InMemoryStore
from agentkit.capabilities.checkpointer import Checkpointer
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.kernel.resilience import idempotency_key
from agentkit.kernel.concurrency import run_agents

async def enrich_batch(rows, ctx):
    done_ids = []
    for i in range(0, len(rows), 100):
        chunk = rows[i:i+100]
        pairs = [(worker, row) for row in chunk]
        results = await run_agents(pairs, ctx, best_effort=True)
        done_ids.extend(r["id"] for r in chunk)
        await ctx.checkpointer.snapshot(
            ctx.correlation_id,
            {"done_ids": list(done_ids), "spent_usd": ctx.budget.spent_usd},
        )
        if i == 300:              # simulate crash after 300 rows
            raise RuntimeError("boom")

# Run 1 — crashes after 300 rows.
budget = Budget(max_cost_usd=50.0, max_concurrency=8)
store = InMemoryStore()
ckpt = Checkpointer(port=InMemoryCheckpointStore())
ctx = RunContext(correlation_id="run-x", budget=budget, store=store, checkpointer=ckpt, ...)
rows = [{"id": i} for i in range(1000)]
try:
    await enrich_batch(rows, ctx)
except RuntimeError:
    pass

# Verify checkpoint state.
saved = await ckpt.resume("run-x")
assert saved.state["done_ids"] == list(range(1, 301))
assert saved.state["spent_usd"] > 0

# Run 2 — resume; rows 1..300 should be served from store, no re-charge.
before_spent = budget.spent_usd
resumed_rows = rows[300:]     # what a real resumer would compute from done_ids
await enrich_batch(resumed_rows, ctx)
# Verify: budget.spent_usd increased ONLY by rows 301..1000, not by any 1..300.
```

If this succeeds — resumed rows do not re-charge and the final state has
1000 unique idempotency keys — the durability story is holding structurally.

### 4. What "failing" would look like

- `budget.spent_usd > budget.max_cost_usd` at run end → `Budget.guard`
  under-fired OR `Budget.charge` isn't serialised under the lock.
- Two resumed runs of the same job produce different `WorkflowResult.usage`
  totals → `Usage.__add__` is not associative/commutative.
- Row 42's `AgentResult` after resume is not identical to the pre-crash
  result → `idempotency_key` drifts across processes OR store contract
  broken.
- `ctx.checkpointer.list_versions(run_id)` after 100 chunks returns
  `[1, 1, 1, …]` or non-monotonic → per-run lock in `Checkpointer.snapshot`
  regressed.
- `gather_bounded` returns 99 results for a 100-item batch → strict-zip
  guard in `Workflow._execute` should have raised, or per_batch_wf's
  equivalent guard is missing.

## Composition

```
operator ──▶ POST /enrich {source: "vendors", budget: 50, concurrency: 8}
                │
                ▼
        Workflow(name="vendor-enrichment", max_steps=1)
          .fn("load_rows", load_rows_from_db)
          .subworkflow("enrich_batch", per_batch_wf, after="load_rows")
                │
                ▼
        per_batch_wf runs in the parent's ctx.child():
          for each chunk of ~100 rows:
            run_agents([(worker, row) for row in chunk],
                       ctx,
                       best_effort=True)
                │
                ▼
        worker = Agent(
          name="enricher",
          model="gpt-4o-mini",
          cognition=SingleCallCognition(),
          output=EnrichedVendor,               # Pydantic model → typed output
          max_repairs=1,
        )
                │
                ▼
        Middleware chain (per-worker Invoker sharing the same LLMPort):
          tracing → meter(Budget) → memoize → output_coerce → retry → LLMPort
                │
                ▼
        Checkpoint every 100 rows via top-level Workflow's human_gate?
          No — we use ctx.checkpointer.snapshot(
            run_id, {"done_ids": [...], "spent_cost": ...},
            status=RUNNING)
          on our own cadence inside per_batch_wf.
                │
                ▼
        Idempotency:
          for each row, compute idempotency_key(scope, "enrich", row.id)
          use ctx.store.get_or_set(key, lambda: worker.run(row, ctx.child()))
```

The top-level Workflow is boring — two nodes. The interesting work is in
`per_batch_wf`, which fans out per-row calls under a shared semaphore and
snapshots periodically.

## The primitives it exercises

| Primitive | Role here | Why load-bearing |
|---|---|---|
| `Workflow` (subworkflow + fn nodes) | Deterministic top-level graph | Two phases (load / enrich) with data dependency |
| `Agent(output=Pydantic)` + `SingleCallCognition` | Per-row worker, one call each | Parse-and-repair for structured output |
| `gather_bounded` (via `run_agents`) | Concurrency-capped fan-out | Never exceeds N in-flight workers |
| `Budget(max_cost_usd, max_calls, max_concurrency)` | Run-wide cost + call + concurrency ceiling | Central authority that meter middleware charges |
| `Usage.__add__` (associative + commutative) | Per-row charges fold into the run total | Concurrent workers charging in any order produce the same total |
| `Checkpointer` (custom cadence) | Save-every-100 durability | Resume from the latest checkpoint after crash |
| `idempotency_key` + `StorePort.get_or_set` | Row-level dedup on resume | A resumed run re-attempting a completed row returns cached result |
| `output_coerce` + `SchemaAdapter` | Structured typed output per row | The `EnrichedVendor` Pydantic model is the wire shape |
| `retry` middleware | Automatic backoff on transient errors | 429/5xx don't kill a 2-hour job |
| `memoize` middleware (optional) | Cache identical LLM calls | If two rows have identical content, don't pay twice |
| `Failure` value type | Errors as data, aggregated per-batch | `best_effort=True` folds row failures into the result |

## What it deliberately doesn't use

- **`ReActCognition`** — each row is a single call; no tool loop.
- **`SignalChannel`** — workers don't coordinate; they operate independently
  in parallel groups.
- **`RunPolicy` block-mode** — no user-facing egress; internal batch.
- **`ExternalTermination`** / **`Autonomy` gates** — no HITL.
- **`Skill.as_agent`** — the worker Agent is inline; not a reusable recipe.
- **`Coordinator` cognition** — Workflow is the explicit-control choice; the
  emergent coordinator shape would be wrong here (rows don't have a
  team-lead relationship).

## Invariants

| Invariant | Concrete failure if violated | Locked by |
|---|---|---|
| **Checkpoint state independent of live scratch dict** | Live loop's `done_ids` grows after checkpoint → resume from that checkpoint sees over-counted progress | `test_snapshot_deep_copies_state_so_post_snapshot_mutation_cannot_reach_it` |
| **Deep-copy handles nested containers + `MappingProxyType`** | State containing `ToolCall.arguments` fails pickle on serializing backends → job cannot resume across a real (non-InMemory) backend | `test_snapshot_state_containing_toolcalls_survives_deepcopy` |
| **`Usage.__add__` is associative + commutative** | 8 concurrent workers folding usage in different orders produce different totals → cost cap trips inconsistently | `test_usage_add_is_associative`, `test_usage_add_is_commutative` (Hypothesis-generated) |
| **`Usage` is frozen** | A worker mutates a shared Usage snapshot → concurrent workers' charges corrupt each other's view | `test_kernel_value_types_are_frozen` (parametrized) |
| **`idempotency_key` is deterministic across processes** | Resume computes different keys than the original run → every row re-enriches → cost doubles | `test_idempotency_key_deterministic_across_calls`, `test_stable_hash_ignores_dict_iteration_order` |
| **`stable_hash` on a plain class doesn't include memory address** | Same conceptual row hashed differently on resume → dedup lookup misses → double-enrichment | `test_stable_hash_plain_class_no_memory_address` |
| **`gather_bounded(sem=budget.semaphore())` never exceeds cap** | Concurrency accidentally unbounded → provider rate-limits everything → cascade fails | `test_gather_bounded_preserves_order_and_bounds` |
| **`Workflow` wave commit uses strict zip** | A silent semaphore bug returns fewer items than dispatched → some rows are dropped without an error | `Workflow._execute` strict=True + arity guard |
| **`run_with_resilience` retries TRANSIENT and UNKNOWN, not PERMANENT** | 429s retry (good); 403s retry (bad — wastes tokens and might get key banned) | `classify` ordering + `test_run_with_resilience_fails_fast_on_permanent` |
| **`CircuitBreaker` counts only TRANSIENT/UNKNOWN** | Every 400 on a bad row opens the breaker → whole job stalls | `test_breaker_skips_permanent_errors` |
| **`Suspended` is frozen** | Resume path mutates the pending list → a mid-flight retry writes into the check-pointed value | `test_frozen_kernel_value_types` extended to result types |
| **`CircuitOpen` chains the original cause** | Postmortem after a stalled job: you see `CircuitOpen` but the root TransientError is lost → no debugging signal | `test_circuit_open_preserves_original_cause` |

## Correctness checklist (for future changes)

- **Every long-running writer must handle `CancelledError` cleanly.** Grep
  for `except Exception` in worker/handler code — should be
  `except Exception as exc:` with cancellation NOT caught, OR the
  broader `except BaseException` explicitly re-raises `CancelledError`.
- **`Checkpointer.snapshot` cadence never blocks the hot path.** If a
  snapshot takes >100ms, throughput drops noticeably. Verify via a
  test that measures elapsed with a large state.
- **The `StorePort.get_or_set` "producer that raises is NOT stored"
  contract holds.** A transient error on row N must not poison the
  cache; the next attempt should re-run. Verified by
  `test_inmemory_get_or_set_raising_fn_does_not_cache_and_next_call_retries`
  in `tests/adapters/test_stores.py`.
- **Concurrent snapshots of the same run_id assign distinct versions.**
  `test_concurrent_snapshots_of_same_run_id_produce_distinct_versions`
  locks this; if that ever breaks, resume becomes non-deterministic.
- **`ChatRequest` frozen; concurrent workers see distinct request
  objects.** `dataclasses.replace` is the only rewrite path. Grep
  `\.model\s*=` in middlewares — every mutation site should be a
  `replace`.

## Design tensions to hold in mind

- **Checkpoint cadence**: too frequent = throughput dies (each snapshot
  is a deep-copy + a store write); too infrequent = crash loses too
  much work. 100 rows is a starting point; production tuning depends
  on typical row-processing cost.
- **Idempotency backend**: `InMemoryStore` is fine for a single-process
  worker but a real cluster needs Redis/Postgres. The `StorePort`
  contract enforces "failures never cached" uniformly across backends;
  test that invariant on your chosen backend.
- **`memoize` middleware vs `idempotency_key` at the row level**: memoize
  caches identical *LLM calls*; the row-level idempotency caches
  *enrichment results*. Both are wanted here — memoize saves duplicate
  LLM cost inside a run, row idempotency saves duplicate row processing
  across resumes.
- **`best_effort=True` semantics**: a row that fails N retries lands as
  a `Failure` in the batch result. The Workflow's default is fail-fast;
  we explicitly opt into partial progress. Documented so a caller knows
  they need to reconcile the `failed_rows` slice separately.
- **Budget accounting under concurrency**: each worker charges into the
  same `Budget` instance. `Budget.charge` must be atomic — the framework
  serializes via `Budget._lock` (verify). A racy charge would let
  8 workers each see "we have room" and jointly overshoot.

## What this use case tests about agentkit (reverse view)

If a 10k-row batch produces different totals on repeated runs, or fails to
resume cleanly after a crash, one of these framework invariants failed.
The specific things this design proves the framework must support:

1. Deep-copy at the checkpoint seam is the ONLY thing standing between the
   live loop and the durable record. Any shared reference is a bug.
2. `Usage.__add__` associativity + commutativity means "the numbers add up"
   is a real property, not a hope. Verified by Hypothesis.
3. `MappingProxyType` (via `ToolCall.arguments`) survives every backend the
   `Checkpointer` might use — proved by our `tc_to_dict` unwrap + the
   `stable_hash` `_stable_default` Mapping branch.
4. `gather_bounded` respects its semaphore under any interleaving of
   worker starts and completions.
5. `run_with_resilience` retries the right classes, records the right
   causes, and never gets stuck retrying a PERMANENT failure forever.

## Verification snapshot

Last audited against the current tree:

- **Automated invariant tests**: `pytest tests/capabilities/test_checkpointer.py
  tests/agents/test_workflow.py tests/runtime/test_meter.py
  tests/kernel/test_kernel.py -k "usage or idempotency or gather"` →
  12 passed, 0 failed, 104 deselected.
- **`Checkpointer.snapshot` deep-copies both state and metadata**:
  `grep -n "deepcopy" agentkit/capabilities/checkpointer/base.py` returns
  two hits (state at line 93, metadata at line 96) — both inside
  `snapshot()`, exactly as required.
- **`gather_bounded` is the fan-out primitive under `run_agents`**:
  `grep -n "gather_bounded\|max_concurrency" agentkit/kernel/concurrency.py`
  returns two hits — the function definition (line 62) and its dispatch
  inside `run_agents` (line 97). `max_concurrency` itself lives on
  `Budget` in `agentkit/runtime/meter.py` (expected — the semaphore is
  owned by the run-wide meter, not the fan-out primitive).
- **`idempotency_key` is defined once and delegates to `stable_hash`**:
  `grep -n "def idempotency_key" agentkit/kernel/resilience.py` returns
  one hit (line 249), body is `"idem:" + stable_hash(parts, length=24)`.

All three load-bearing structural checks pass, and the 12 invariant tests
in the targeted slice are green. The design holds in the current tree.
