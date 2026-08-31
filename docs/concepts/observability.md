# Observability

Observability is how you find out what a run did, why it was slow, and
what it cost — after it has already happened.

agentkit splits that into three surfaces that are easy to confuse
because they all sound like "telemetry". They are not
interchangeable, they are wired independently, and picking the wrong one
is the usual reason a team ends up with dashboards that cannot answer
their question.

| | **Observations** | **Traces** | **Meters** |
|---|---|---|---|
| Answers | "what is the agent doing *right now*?" | "why did *this run* take 40s?" | "how much did it cost, and who pays?" |
| Audience | your product and its users | you, at 3am | finance, and the ceiling that stops a runaway loop |
| Shape | an ordered stream of `Observation` records | spans with attributes and events | a `Decimal` ledger with ceilings |
| Always on? | yes — the default just drops everything | no — sampled, off by default | yes — `Budget` is always on the context |
| Seam | `ObserverPort` on `Services.observer` | `TracePort` on `Services.trace` | `ctx.budget` and `ctx.meters` |
| Wire it with | `agentkit.adapters.observer` | `agentkit.adapters.observability` | `Budget` / `Quota` + the `meter()` middleware |

The one-line test: **if a user should see it, it is an observation. If
an operator should see it, it is a trace. If an accountant should see
it, it is a meter.**

!!! tip "Is this page for you?"

    **Reach for it when** a run was slow, expensive, or wrong, and
    you need to find out which — after the fact.

    **Skip it for now if** you are still developing locally and
    `print()` is genuinely enough.

## The problem it solves

An agent run is a long-lived, non-deterministic process that spends
money. Without observations, a user stares at a spinner for ninety
seconds with no idea whether anything is happening — and a parent agent
supervising children has nothing to supervise with. Without traces, "the
run was slow" is unanswerable: you cannot tell a slow provider from a
slow tool from six retries. Without meters, a tool loop that never
converges bills until someone notices, and one tenant's runaway job
starves everyone else on the box.

Each one is cheap to wire and expensive to add after the incident.

## The smallest thing that works

An observation stream, with nothing else configured:

```python
import asyncio

from agentkit.adapters.observer import CollectingObserver
from agentkit.testing import make_test_ctx


async def main() -> None:
    obs = CollectingObserver()
    ctx = make_test_ctx(observer=obs)

    await ctx.emit("progress", "read 3 of 10 files", payload={"done": 3, "of": 10})
    await ctx.emit("result", "wrote the summary", payload={"words": 120})

    for o in obs.items:
        print(o.kind, "|", o.render, "|", o.payload, "|", o.run_id)


asyncio.run(main())
```

```text
progress | read 3 of 10 files | {'done': 3, 'of': 10} | test-run
result | wrote the summary | {'words': 120} | test-run
```

`render` is the short human line; `payload` is the machine-readable
record. `run_id` comes from `ctx.correlation_id` automatically.

---

## Observations — the product-facing stream

### The mental model

Every run emits observations whether or not anyone is listening. The
default `Services.observer` is `NoopObserver`, which drops everything,
so emitting is free and zero-config. Attaching an observer is what turns
the stream on.

`ctx.emit(kind, render, *, payload=None, agent="", parent_id=None)` is
the only producer API you need. `kind` is a closed set, so a consumer's
branch table can be exhaustively type-checked:

| kind | emitted when |
|---|---|
| `progress` | incremental work happened |
| `summary` | a rolled-up digest (what `RollupObserver` produces) |
| `partial_result` | a usable-but-incomplete answer |
| `result` | the run finished |
| `error` | the run failed |
| `interrupt` | the run is waiting on a human |
| `signal.emitted` | a `SignalChannel` published to a peer |
| `gate.check` | an autonomy gate was evaluated — see the note below, the payload is not one shape |
| `memory.written` | a memory decorator wrote |

!!! note "`gate.check` has three producers and two payload shapes"

    `ReActCognition` and `elicit()` emit it with three keys.
    `ApprovalServer` emits it with six — `tool`, `allowed`, `reason`,
    `source`, `asked` and `at` — because a CLI permission prompt is
    answered somewhere agentkit cannot see, so the record has to carry
    who decided and whether a person was reached.

    A consumer writing a typed handler off the table above will
    otherwise assume one shape and lose fields on the producer that has
    the most to say. Note also that `arguments` is deliberately absent
    from the broadcast even though `ApprovalDecision` holds it: the
    stream is the wrong place for a payload that may carry a secret.
    Read it off `approvals.decisions` instead.

`result` and `error` are `CRITICAL_KINDS`. Together with `interrupt`
they are never silenced, delayed, or dropped by any cadence wrapper in
the box. That is a guarantee, not a default: an observer that swallows a
result has broken the run's only success channel, and one that swallows
an interrupt has hung it.

**Emitting never breaks the run.** `ctx.emit` wraps the observer call in
`contextlib.suppress(Exception)` unconditionally. A Redis hiccup, a
slow WebSocket subscriber, a Kafka push error — none of it propagates
into the agent loop. Every middleware and pattern in the framework emits
through this seam expecting it to be inert.

### `Observation` is an audit record, so it is frozen — payload included

`Observation` is a frozen dataclass, and `__post_init__` additionally
deep-freezes `payload`.

The reason is fan-out. One record reaches an audit sink, a WebSocket
forwarder and a rollup buffer, all of which retain the reference
alongside the run's live emit loop. `frozen=True` alone only protected
`render`; `obs.payload["status"] = "done"` — on the field that *is* the
machine-readable record — rewrote what the other two sinks had already
queued.

```python
from agentkit.kernel.observation import Observation

obs = Observation(kind="summary", render="wrote intro", payload={"words": 120})
print(type(obs.payload).__name__, obs.payload["words"])
try:
    obs.payload["words"] = 0
except TypeError as exc:
    print("TypeError:", exc)

print("scalar payload passes through:", repr(Observation(kind="progress", payload="tick").payload))
```

```text
FrozenDict 120
TypeError: this payload belongs to a frozen value and cannot be mutated in place. Build a new one instead: dataclasses.replace(obj, field={**obj.field, ...})
scalar payload passes through: 'tick'
```

`deep_freeze` returns non-containers untouched, so a `str` / `int` /
`None` payload is bit-for-bit the object you passed; only dicts and
lists are walked.

!!! note "The measured cost, stated plainly"
    Freezing is O(payload) and runs once per emitted event. Per
    construction, measured on the author's machine:

    | payload | before | after |
    |---|---|---|
    | `None` / `str` / `int` | 1.84 µs | 2.30 µs |
    | `{"step": 3, "of": 10}` | 1.87 µs | 3.23 µs |
    | 4-key summary | 1.84 µs | 3.84 µs |
    | nested signal payload | 1.84 µs | 5.20 µs |
    | full agent result (~40 keys + 10-message transcript) | 1.84 µs | 37.9 µs |

    End to end on the cheapest emit that exists — construct, suppress,
    and `await` into a `NoopObserver` — a 4-key summary went from
    2.62 µs to 4.60 µs. That is +75%, and it is the honest headline
    rather than a footnote.

    What it buys: the same 2 µs sits in front of an `await` into an
    adapter that buffers or writes a row, so it does not show up against
    a real observer; the +36 µs case is a `result`, emitted once per run;
    and scalar payloads — the ones on the tightest progress loops — pay
    0.46 µs. If you profile this as your bottleneck, the fix is a
    shallower payload, not an unfrozen one.

`Observation.__hash__` is deliberately computed on the *stream key* —
`(run_id, agent, kind)` — never on `payload`. A frozen
dataclass would otherwise derive its hash from every compared field, and
`payload` is in practice always a dict, which made every observation a
real run produced unhashable while a test that passed a string proved
the opposite. `__eq__` still compares every field, so a `set`-based
dedup of a replayed stream stays exact; two records sharing a stream key
just land in one bucket, which is what a bucket is for.

### Cadence: how much of the stream reaches the consumer

Four adapters live in `agentkit.adapters.observer`. Two are terminal
sinks, two are cadence wrappers you compose in front of them.

- `CollectingObserver` — appends to `.items`. Unbounded. Tests and
  small in-process consumers.
- `QueueObserver(maxsize=256)` — bounded, non-blocking, async-iterable
  via `stream()`.
- `PolicyObserver(inner, allow=…)` — a kind filter.
- `RollupObserver(inner, every=8, summarize=…)` — buffers non-critical
  observations and emits one `summary` per `every` of them.

`RollupObserver` exists because a step-level stream is too noisy for a
UI and too expensive for a socket. Its cadence rule has three parts, all
visible in one run:

```python
import asyncio

from agentkit.adapters.observer import CollectingObserver, PolicyObserver, RollupObserver
from agentkit.testing import make_test_ctx


async def main() -> None:
    sink = CollectingObserver()
    rollup = RollupObserver(sink, every=4)
    ctx = make_test_ctx(observer=rollup)

    for i in range(6):
        await ctx.emit("progress", f"step {i}")
    print("after 6 progress:", [(o.kind, o.render, o.payload) for o in sink.items])

    await ctx.emit("result", "done")
    print("after the result: ", [(o.kind, o.render, o.payload) for o in sink.items])

    await rollup.close()
    print("dropped:", rollup.dropped)

    # A policy on top: only result / error / interrupt reach the sink.
    quiet_sink = CollectingObserver()
    qctx = make_test_ctx(observer=PolicyObserver.result_only(quiet_sink))
    await qctx.emit("progress", "step 0")
    await qctx.emit("error", "provider 500")
    print("result_only:      ", [(o.kind, o.render) for o in quiet_sink.items])


asyncio.run(main())
```

```text
after 6 progress: [('summary', '4 updates; latest: step 3', {'rolled': 4})]
after the result:  [('summary', '4 updates; latest: step 3', {'rolled': 4}), ('summary', '2 updates; latest: step 5', {'rolled': 2}), ('result', 'done', None)]
dropped: 0
result_only:       [('error', 'provider 500')]
```

Read the second line carefully. Six `progress` observations produced one
summary of 4 and left 2 buffered. The `result` did **not** queue behind
them: it flushed the buffer — producing the second summary — and then
forwarded immediately. `close()` flushes any remaining tail, which is
why `ObserverPort` declares `close()` at all.

`summarize(buffer) -> str` may be sync or async; the async form is the
hook for an LLM judge or a `Compactor`-backed summariser. The default is
dependency-free: a count plus the latest line.

!!! warning "Two failure modes `RollupObserver` had to be taught"
    Both are worth knowing because both look impossible until they
    happen.

    **The flush window.** `_flush` originally awaited `summarize` and
    *then* cleared the buffer. A concurrent `emit` re-entered it, the
    first task cleared the buffer, and the second read `self._buf[-1]`
    on an empty list. Measured with an async summariser, `every=2` and 6
    concurrent observations: `IndexError('list index out of range')`
    raised **into the caller's `emit`**. The same window also silently
    discarded everything appended during the await. The fix detaches the
    buffer before any await, under a lock, so exactly one task owns a
    buffer generation.

    **A telemetry failure killing the run.** A caller-supplied
    `summarize` that raised, or an inner sink that was down, propagated
    straight into `emit`. An agent must not die because its telemetry
    did. Failures are now caught, the batch is counted on `.dropped`,
    and the first one logs a warning. `except Exception`, not
    `BaseException` — `CancelledError` is how a run is torn down and
    must keep propagating.

    `.dropped == 0` means the rollup is complete. Anything else means a
    consumer is reading a partial view.

`QueueObserver` applies the same never-drop-results rule under
backpressure. It bounds *non-critical* observations to `maxsize` and
evicts the oldest when over; `result` and `error` are never dropped and
never count against the bound:

```python
import asyncio

from agentkit.adapters.observer import Hooks, QueueObserver
from agentkit.testing import make_test_ctx


async def main() -> None:
    queue = QueueObserver(maxsize=2)
    hooks = Hooks(inner=queue)
    hooks.on("error", lambda o: print("ALERT:", o.render))
    ctx = make_test_ctx(observer=hooks)

    for i in range(5):
        await ctx.emit("progress", f"step {i}")
    await ctx.emit("result", "done")
    await hooks.close()

    async for o in queue.stream():
        print(o.kind, o.render)


asyncio.run(main())
```

```text
progress step 3
progress step 4
result done
```

`Hooks` is lifecycle subscription over the same stream — `on(kind,
handler)`, or `on("*", …)` for everything. Handlers may be sync or
async, and an exception in one is swallowed: a hook must never
destabilise the run it observes. It chains in front of another observer
via `inner=`, and forwards `close()` down the chain.

!!! tip "Compose them; the order is the cadence"
    `Hooks(inner=PolicyObserver.summaries(RollupObserver(QueueObserver())))`
    is a full stack: alerting hooks in front, a kind filter, rolled-up
    batching, and a bounded queue a WebSocket drains. Every layer
    forwards `close()`, which is what stops the rollup's trailing
    summary from vanishing at process exit.

---

## Traces — the operational timeline

### The mental model

A trace answers "what happened inside *this one run*, in what order, and
how long did each part take". `TracePort` is the seam:

```python
from typing import Any

from agentkit.kernel.observation import TraceContext


class TracePort:  # the real one is a runtime_checkable Protocol
    def span(self, name: str, kind: str, **attrs: Any) -> Any: ...  # sync or async CM
    def current_span_id(self) -> TraceContext | None: ...
    def add_event_to_current_span(self, name: str, **fields: Any) -> None: ...
```

The default is `NoopTrace`. You never open spans by hand — the
`tracing()` middleware owns span lifecycle for every chat and tool call,
so span naming and attribute keys stay consistent across the framework
instead of drifting per pattern.

Names follow the OpenTelemetry GenAI v1.41 operation taxonomy (`chat`,
`execute_tool`). The kernel itself stays OTel-free: only the names and
attribute keys are aligned, so an adapter can lift them into a real
exporter without translation.

```python
import asyncio

from agentkit import ChatRequest, Message
from agentkit.middlewares import tracing
from agentkit.testing import FakeLLM, RecordingTracer, make_test_ctx


async def main() -> None:
    tracer = RecordingTracer()
    ctx = make_test_ctx(
        llm=FakeLLM("Octopuses taste with their arms."),
        trace=tracer,
        chat_middleware=[tracing()],
    )
    req = ChatRequest(messages=[Message("user", "One fact about octopuses.")], model="fake-1")
    await ctx.invoker.chat(req, ctx)

    for span in tracer.spans:
        print(span.name, span.kind)
        for key, value in span.attrs.items():
            print("   ", key, "=", value)
        print("    events:", [name for name, _ in span.events])


asyncio.run(main())
```

```text
chat chat
    gen_ai.request.model = fake-1
    gen_ai.response.first_token_latency_ms = 0.014541990822181106
    gen_ai.usage.input_tokens = 10
    gen_ai.usage.output_tokens = 5
    gen_ai.usage.cost_usd = 0.0001
    gen_ai.response.model = fake-1
    gen_ai.response.finish_reasons = stop
    gen_ai.response.duration_ms = 0.04454198642633855
    events: ['gen_ai.user.message', 'gen_ai.choice']
```

`RecordingTracer` is a real `TracePort` from `agentkit.testing` — the
same object the framework's own tests assert against.

### What the middleware stamps, and the caps it respects

Attributes on a chat span come from the *assembled* result at
stream-end: token usage, `cost_usd`, response model, finish reason,
total duration, and `first_token_latency_ms` (absent, not zero, on
streams that never produce content — a tool-only turn has no first
token, and a misleading `0` is worse than "no value").

Prompt-cache attributes are stamped only when the provider reports a
non-zero value, so the 95% of calls without caching do not carry noisy
zeros. When they are present, the middleware also derives
`gen_ai.usage.cost_saved_by_cache_usd` from a 10% cached-input price
ratio — the rate both OpenAI and Anthropic bill cache reads at.

Events carry content, and content is capped because OTel collectors
enforce per-attribute length limits and silently drop over-long values.
Truncating in agentkit's own code preserves the *leading* content, which
is what an operator wants to see, and signals the cut with a
` [truncated]` suffix:

| constant | value | applies to |
|---|---|---|
| `MAX_MESSAGE_CONTENT_BYTES` | 8 KB | `gen_ai.{role}.message`, `gen_ai.choice` |
| `MAX_TOOL_RESULT_CONTENT_BYTES` | 4 KB | `gen_ai.tool.result` — tool output can be an HTML blob |
| `MAX_TOOL_NAMES_IN_ATTR` | 16 names | `gen_ai.response.tool_calls`, then `,+K more` |

`gen_ai.tool.result_size_bytes` is kept **un**capped, so you can still
see how big the real blob was after the content was truncated. The exact
count of tool calls lives on the separate integer attribute
`gen_ai.response.tool_calls_count`, which is what a histogram wants.

Redaction is deliberately *not* done here — configure it collector-side
(e.g. the OTel collector's `processor/attributes` regex rules).

### Sampling drops spans, never metrics

`ctx.sampler` is consulted before the span opens. When
`should_sample` returns `False` the span, its per-message events, and
its replay write are all skipped — but the underlying handler still runs
and the result still flows, and the **metrics still record**.

That split is the point. Sampling controls trace fidelity; metrics roll
up unconditionally. Without it, every dashboard would under-count token
cost, duration and error rate by exactly the sample rate.

`Services.sampler` defaults to `AlwaysOnSampler` — record everything —
so nothing is silently missing until you decide otherwise.
`otel_sampler(ratio)` builds the in-tree `TraceIdRatioSampler`
(`ratio=1.0` always-on, `0.1` ten percent); it is pure Python with no
OTel dependency, and lives next to the OTel factories only so operators
have one module to import from. For a richer policy — parent-based,
rule-based — write your own `SamplerPort` and inject it directly.

A misbehaving sampler defaults to **keeping** the span. More spans is
the safer failure mode than silent loss.

`Services.metrics` defaults to `NoopMetrics`, so a `Services()` with no
arguments is fully usable and the middleware's metric calls cost
nothing.

```python
"""Sampling drops the SPAN. The metric is still recorded."""

import asyncio
from typing import Any

from agentkit import ChatRequest, Message, RunContext, Scope, Services
from agentkit.adapters.observability import otel_sampler
from agentkit.middlewares import tracing
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM, RecordingTracer


class RecordingMetrics:
    """A MetricsPort: two plain methods, and it never raises into the run."""

    def __init__(self) -> None:
        self.records: list[tuple[str, Any, dict[str, str]]] = []

    def add_counter(self, name: str, value: int | float = 1, *, tags: Any = None) -> None:
        self.records.append((name, value, dict(tags or {})))

    def record_histogram(self, name: str, value: int | float, *, tags: Any = None) -> None:
        self.records.append((name, value, dict(tags or {})))


async def main() -> None:
    tracer, metrics = RecordingTracer(), RecordingMetrics()
    services = Services(
        invoker=Invoker(llm=FakeLLM("hi"), chat_middleware=[tracing()]),
        trace=tracer,
        metrics=metrics,
        sampler=otel_sampler(0.0),  # drop every span
    )
    ctx = RunContext(correlation_id="run-1", scope=Scope(org_id="acme"), services=services)
    await ctx.invoker.chat(ChatRequest(messages=[Message("user", "hi")], model="fake-1"), ctx)

    print("spans:", len(tracer.spans))
    for name, value, tags in metrics.records:
        print(f"  {name} = {value!r} {tags}")


asyncio.run(main())
```

```text
spans: 0
  gen_ai.client.token.usage = 10 {'gen_ai.request.model': 'fake-1', 'agentkit.scope.org_id': 'acme', 'gen_ai.token.type': 'input'}
  gen_ai.client.token.usage = 5 {'gen_ai.request.model': 'fake-1', 'agentkit.scope.org_id': 'acme', 'gen_ai.token.type': 'output'}
  gen_ai.client.operation.duration = 3.1666975701227784e-05 {'gen_ai.request.model': 'fake-1', 'agentkit.scope.org_id': 'acme'}
```

Note the tag set: model plus a small, known `org_id`, and nothing else.
It is kept tiny on purpose — every additional tag multiplies the
time-series count. Durations are in **seconds**, per OTel convention,
while span durations are in milliseconds.

The error path increments `gen_ai.client.error.count` tagged with the
exception class name, so you can alert on a provider 5xx storm without
joining traces.

### A tracer must never break the run

Every observability call in the middleware — metrics, sampler, replay,
per-message events — is wrapped in `contextlib.suppress(Exception)`. The
span itself gets `_safe_trace_span`, which falls back to a `NoopSpan` if
`trace.span(...)` raises on open, enter or exit, and logs the traceback
exactly **once** per process so a misbehaving tracer cannot spam a log
per unit of work.

One asymmetry is deliberate and easy to get wrong: on the error path the
span is closed with the live `(type, value, traceback)` rather than a
clean exit. `ExitStack.close()` is defined as `__exit__(None, None,
None)` — it tells the wrapped context manager the block succeeded,
whatever actually happened. The OTel adapter relies on OTel's own
`__exit__` to call `record_exception` and set the ERROR status, so
closing cleanly meant **a provider failure exported as a successful
span**. Measured: the provider raised `RuntimeError("provider 500")` and
the span came out with `exception=None, status=None`.

The original exception is always re-raised, even if the span's
`__exit__` returns `True`. And only `Exception` is reported as a span
error: `GeneratorExit` / `CancelledError` reach that frame when a
consumer abandons or cancels a stream, which is not a provider fault.

### Wiring OpenTelemetry

`agentkit.adapters.observability` bridges both ports to the OTel SDK.
Install `pip install "arc-agentkit[observability]"`; the factories raise
`ImportError` naming the extra if you have not.

```python
"""A real OTel span, exported in-memory so the example needs no backend."""

import asyncio

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentkit import ChatRequest, Message
from agentkit.adapters.observability import otel_tracer
from agentkit.middlewares import tracing
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    ctx = make_test_ctx(
        llm=FakeLLM("ok"),
        trace=otel_tracer(provider),
        chat_middleware=[tracing()],
    )
    await ctx.invoker.chat(ChatRequest(messages=[Message("user", "hi")], model="fake-1"), ctx)

    for span in exporter.get_finished_spans():
        usage = {k: v for k, v in (span.attributes or {}).items() if k.startswith("gen_ai.usage")}
        print(span.name, span.kind, f"trace_id={span.context.trace_id:032x}")
        print("  ", usage)


asyncio.run(main())
```

```text
chat SpanKind.INTERNAL trace_id=78db743082019e8cf1a337976407ce1e
   {'gen_ai.usage.input_tokens': 10, 'gen_ai.usage.output_tokens': 5, 'gen_ai.usage.cost_usd': 0.0001}
```

In production you call `otel_exporter_otlp_http()` once at startup and
`otel_tracer()` with no argument, which picks up the global provider.
[Wire OpenTelemetry](../recipes/otel-tracing-and-metrics.md) is the
full recipe, including the metrics exporter.

### The replay side channel

Attribute cardinality has a budget; a full prompt does not fit in it.
When a `ReplayStore` is wired on `Services.replay`, the tracing
middleware writes a `ReplayRecord` keyed by the chat span's id carrying
the complete request and assembled response. The span carries the index;
the store carries the bulk, so downstream tooling can rebuild a call
without re-running it. The default `NoopReplayStore` makes this a
zero-cost no-op.

---

## Meters — spend and quota

### The mental model

A meter is one concept — *guard a ceiling, then charge usage* — applied
at two scopes:

- **`Budget`** — per run. One instance shared by reference across the
  whole agent tree via `ctx.child()`, so a fan-out cannot spend the
  parent's ceiling twice. It is also the run's depth and concurrency
  authority.
- **`Quota`** — per tenant. Rolling RPM / TPM / dollar windows keyed by
  `scope.key()`. The noisy-neighbour guard and the chargeback source.
  The shipped one is in-memory; production swaps a Redis-backed `Meter`.

Both implement the `Meter` protocol (`guard` before the work, `charge`
after) and both are driven by the single `meter()` middleware, which
guards every meter on `ctx` before a call and charges them after.

### Money is `Decimal`, with a float mirror

Binary floating point cannot represent `0.01`. A hundred one-cent
charges summed as floats land at `1.0000000000000007`, and a metered run
that cannot be reconciled to the cent is not a ledger.

So `Budget` keeps an exact `Decimal` ledger at `MONEY_SCALE = 6` decimal
places — a tenth of a millicent, the same scale `pricing.cost()` rounds
to — and exposes `spent_usd` as a **float mirror** of it, re-derived
after every charge.

The mirror is not a wart, it is a compatibility decision made
explicitly: keeping the name rather than deprecating it meant ~20 doc
references, 28 test references, and every application reading
`budget.spent_usd` kept working untouched. Rebuilding a `Budget` from a
checkpoint — `Budget(spent_usd=saved.state["spent_usd"])` — is a
documented path, so `spent_usd` stays an init-accepting field that seeds
the exact ledger. From there `_spent` is authoritative and the mirror is
re-derived, so the two cannot drift.

The naming convention is worth memorising, because it is the thing that
stops a float creeping back in:

> **`spent` / `remaining` are `Decimal` everywhere. `*_usd` is `float`
> everywhere.**

```python
import asyncio
from decimal import Decimal

from agentkit import ChatRequest, Message
from agentkit.kernel.types import Usage
from agentkit.middlewares import meter
from agentkit.runtime import Budget
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    budget = Budget(max_cost_usd="0.01", on_exceeded="stop")
    ctx = make_test_ctx(
        llm=FakeLLM("ok", usage=Usage(1000, 500, 0.004)),
        budget=budget,
        chat_middleware=[meter()],
    )
    req = ChatRequest(messages=[Message("user", "hi")], model="fake-1")
    for i in range(3):
        await ctx.invoker.chat(req, ctx)
        print(
            f"call {i + 1}: spent={budget.spent()} remaining={budget.remaining()} "
            f"exhausted={budget.exhausted()} tokens={budget.usage.total_tokens}"
        )
    print("float mirror:", budget.spent_usd, "| cents:", budget.spent_cents())
    print("verdict:", budget.verdict().reason)

    naive = 0.0
    for _ in range(100):
        naive += 0.0001
    exact = sum((Decimal("0.0001") for _ in range(100)), Decimal(0))
    print("100 x $0.0001 as float:", naive, "| as Decimal:", exact)


asyncio.run(main())
```

```text
call 1: spent=0.004000 remaining=0.006000 exhausted=False tokens=1500
call 2: spent=0.008000 remaining=0.002000 exhausted=False tokens=3000
call 3: spent=0.012000 remaining=0 exhausted=True tokens=4500
float mirror: 0.012 | cents: 0.01
verdict: cost $0.012 > $0.01
100 x $0.0001 as float: 0.009999999999999995 | as Decimal: 0.0100
```

Three things in that output:

- **`Budget` accumulates the whole `Usage`**, not just a cost scalar.
  Input / output / cache-read / cache-write counts survive to the end of
  a multi-agent run. Reducing it to one number forced every application
  to re-aggregate what the framework had already seen.
- **`spent_cents()` quantizes at read time**, never per charge. Rounding
  each charge to cents would round every sub-cent call to zero and
  undercount the whole run.
- **The budget overran its ceiling by one call.** That is a known
  property, not a bug: the verdict compares `spent > ceiling` *after* the
  work has run, because there is no pre-flight cost estimate. Set the
  ceiling slightly below your true limit.

### Ceilings are intent; charges are measurement

The two are treated differently on purpose:

| | rule | why |
|---|---|---|
| A **ceiling** with more precision than `MONEY_SCALE` | `MoneyPrecisionError` at construction | silently rounding changes what the operator asked for, and construction is where it is free to fix |
| A **charge** with more precision | quantized, never refused | raising mid-run would abort before the cognition reached its checkpoint — the exact failure this class was rewritten to remove |

There is one exception to the first rule: assigning `max_cost_usd`
*after* construction (`budget.max_cost_usd = 10.0`, the documented way
to raise a ceiling and resume an exhausted run) re-derives
**non-strictly**, because that path is reached from inside `charge()`.

### `charge()` returns a verdict rather than only raising

Raising from inside `charge()` aborts the run mid-call — before the
cognition reaches its checkpoint write — so exhausting a budget
destroyed everything spent up to that point.

`charge()` now returns a `Charge`: `ok`, `reason`, `spent` (exact),
`remaining` (exact), `calls`, and the cumulative `usage`, so a caller
acting on a verdict never has to go back to the meter for the numbers
that justified it.

`on_exceeded` picks what happens at the ceiling:

- `"raise"` (**default**) — `MeterExceeded` from inside `charge`.
- `"stop"` — `charge` returns `Charge(ok=False, …)` and raises nothing.
  The ReAct cognition reads `budget.exhausted()` between units of work,
  writes a `suspended` checkpoint, and ends with
  `stop_reason="budget_exhausted"`. The spend is recoverable.

`"raise"` stays the default deliberately. Flipping it would silently
change the control flow of every existing wiring: a run that used to
abort would continue past its ceiling in any caller that ignores the
return value, which is a worse failure than the one being fixed. Callers
opt into recoverability.

`(await budget.charge(...)).raise_if_exceeded()` converts a verdict back
into the exception at one specific site.

### `Quota` — the tenant axis

`Quota` enforces on `guard` (before the work) rather than on `charge`
(after it), because a rate limit that only notices afterwards is not a
rate limit.

```python
import asyncio

from agentkit import ChatRequest, Message, Scope
from agentkit.kernel.types import Usage
from agentkit.middlewares import meter
from agentkit.runtime import Quota
from agentkit.runtime.meter import MeterExceeded
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    quota = Quota(max_rpm=2, max_usd="1.00")
    ctx = make_test_ctx(
        llm=FakeLLM("ok", usage=Usage(100, 50, 0.02)),
        scope=Scope(org_id="acme"),
        meters=[quota],
        chat_middleware=[meter()],
    )
    req = ChatRequest(messages=[Message("user", "hi")], model="fake-1")
    for i in range(3):
        try:
            await ctx.invoker.chat(req, ctx)
            print(f"call {i + 1}: ok, tenant spend {quota.spent_in_window(ctx.scope.key())}")
        except MeterExceeded as exc:
            print(f"call {i + 1}: refused — {exc}")


asyncio.run(main())
```

```text
call 1: ok, tenant spend 0.020000
call 2: ok, tenant spend 0.040000
call 3: refused — orgacme:domNone: 2 req ≥ 2 rpm
```

`spent_in_window(key)` is the chargeback number, summed in `Decimal` so
it reconciles. `Quota.charge` deliberately leaves `Charge.usage` empty:
it keeps per-window token counts partitioned by tenant, not one
cumulative `Usage`, and returning this call's usage under a field
documented as cumulative would be a quieter lie than leaving it blank.

Two housekeeping behaviours you will not see but should know about.
`_prune` deletes a tenant's key entirely when its window empties,
because a scope carrying a per-user id would otherwise retain one
permanently-empty dict entry per distinct scope ever seen. And `_sweep`
evicts every long-dead tenant at most once per window — measured, 5000
distinct scopes left 5000 retained keys long after every window had
expired.

### `Budget` also owns depth and concurrency

`max_depth` and `max_concurrency` live on `Budget` because they bound
the same runaway the cost ceiling does.

`semaphore(depth)` returns **one semaphore per depth of the agent tree**,
not one for the whole tree. A single shared semaphore deadlocks, and not
hypothetically: a parent's fan-out holds its permits for the entire
duration of each child run, so a nested fan-out draws from a pool its
own ancestors have already drained. With `max_concurrency=2`, an agent
dispatching two `as_tool` sub-agents that each dispatch their own tools
hangs forever.

Keying on depth breaks the cycle structurally — an ancestor at depth *d*
only ever holds permits from pool *d*, and its children draw from pool
*d+1*. The honest trade: the bound is `max_concurrency` **per level**,
so worst-case in-flight work is
`max_concurrency * (max_depth + 1)`. Set it with that in mind.

---

## Where the three surfaces meet

They are separate seams, but two bridges exist so you are not
cross-referencing streams by hand.

**Observation → span.** `ctx.emit` reads `TracePort.current_span_id()`
and stamps the result on `Observation.trace_context`, so a UI-visible
observation can deep-link into the span tree that produced it. For
`CRITICAL_KINDS` it additionally drops an `observation.emitted` event on
the open span, so the trace timeline is complete on its own.

```python
import asyncio

from agentkit.adapters.observer import CollectingObserver
from agentkit.testing import RecordingTracer, make_test_ctx


async def main() -> None:
    sink, tracer = CollectingObserver(), RecordingTracer()
    ctx = make_test_ctx(observer=sink, trace=tracer)

    with ctx.trace.span("chat", "chat", **{"gen_ai.request.model": "fake-1"}):
        await ctx.emit("progress", "thinking")
        await ctx.emit("result", "done")

    for o in sink.items:
        print(o.kind, "->", o.trace_context)
    print("span events:", [name for name, _ in tracer.spans[0].events])


asyncio.run(main())
```

```text
progress -> TraceContext(trace_id='00000000000000000000000000000001', span_id='0000000000000001')
result -> TraceContext(trace_id='00000000000000000000000000000001', span_id='0000000000000001')
span events: ['observation.emitted']
```

`trace_context` is `None` when no tracer is configured or no span is
open — a cold emit is still a valid emit.

**Meter → span.** After charging, the `meter()` middleware drops a
`budget.checkpoint` event on the currently-open chat span with the
post-charge spend:

```text
budget.checkpoint {'spent_usd': 0.0001, 'remaining_usd': 0.0999, 'calls': 1}
```

so you can find the call that crossed a threshold by reading the trace,
without correlating against an external metric.

## What bites people

- **`Observation` has no `seq` or `ts`.** It used to declare both, and
  nothing in agentkit ever set either one — measured,
  `Observation(kind="tool_result", payload={}).seq` was `0` and `.ts`
  was `0.0` on every record any run produced, while `__hash__` folded
  both into the stream key and called `seq` "that emitter's monotonic
  counter". They were removed rather than populated, because ordering
  already has two better owners: in-process, `emit` is awaited and every
  sink preserves arrival order (`CollectingObserver.items` is
  append-ordered, `QueueObserver.stream()` yields in insertion order);
  durably, the Postgres log adapter allocates `seq BIGSERIAL PRIMARY
  KEY` per row and reads back `ORDER BY seq`. If you need a per-sink
  sequence or an arrival stamp, keep it **in the sink** beside the
  record — one shared record cannot carry a different number for each
  fan-out target, and the sink is what knows when it saw the event.
- **Wrapping an observer without forwarding `close()` strands it.** This
  bit the framework itself: a `Hooks -> PolicyObserver -> RollupObserver`
  stack silently dropped the rollup's trailing summary at process exit,
  and `isinstance(PolicyObserver(...), ObserverPort)` was `False` —
  a cadence wrapper failing the very Protocol it exists to compose.
  Every shipped wrapper now forwards `close()`; a custom one must too.
- **A cache hit is traced but not billed.** `memoize` re-emits the
  stored result verbatim, `usage` and all. Metering on "any result
  carrying usage" billed the same provider call once per hit — measured,
  four identical chats behind `memoize` made one provider call but drove
  `spent_usd` from 0.25 to 1.0, and a fifth raised `MeterExceeded` on
  $0.75 that was never spent. `meter()` reads `call.meta["cache_hit"]`
  and skips the charge; `tracing()` sits outside `meter()` in the
  documented chain, so the hit still shows up in the trace.
- **Never put a high-cardinality tag on a metric.** `span.set(k, v)`
  becomes a trace attribute; `add_counter` / `record_histogram` tags
  become time-series dimensions. `user_id` or `run_id` on a metric is
  how a metrics bill becomes the incident.
- **Do not set a global OTel provider twice.** OTel forbids overriding
  `set_tracer_provider` once set; sharing a global across tests gives
  stale-provider failures. Pass an explicit `tracer_provider=` to
  `otel_tracer(...)` in tests.
- **`ObserverPort` is not `TracePort`.** They are wired independently on
  `Services` and neither implies the other. If this page achieves one
  thing, let it be this one.

## Related

- [Wire OpenTelemetry](../recipes/otel-tracing-and-metrics.md) — the
  task-shaped version of the traces half, against a real provider.
- [Cap spend with Budget and Quota](../recipes/spend-budget-and-quota.md)
  — the meters half, end to end.
- [Concepts · Middlewares](middlewares.md) — where `tracing()` and
  `meter()` sit in the chain, and why the order matters.
- [Concepts · Runtime](runtime.md) — `RunContext`, `Services`, and the
  seams these three surfaces plug into.
- [Concepts · Kernel](kernel.md) — `Observation`, `ObserverPort`,
  `TracePort` as value types and Protocols.
