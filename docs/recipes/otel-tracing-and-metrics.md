# How do I wire OpenTelemetry?

## When you'd want this

You want traces (per-request narrative) and metrics (rolled-up
time-series) in a real backend — Tempo, Jaeger, Datadog, Honeycomb,
whatever speaks OTLP. agentkit's `TracePort` and `MetricsPort` are
plain Protocols; the `observability` extra ships one adapter each that
bridges them to the OpenTelemetry SDK.

Traces answer "what did *this* run do?". Metrics answer "what is the
p99 first-token latency doing?". You almost always want both.

## Working code

```python
import asyncio

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentkit import ChatRequest, Message, RunContext, Scope, Services
from agentkit.adapters.observability import otel_tracer
from agentkit.middlewares import meter, tracing
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM


async def main() -> None:
    # Build our own tracer provider so the test is hermetic — production
    # usually calls otel_exporter_otlp_http() at startup and lets the SDK's
    # global provider be picked up automatically.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    services = Services(
        invoker=Invoker(
            llm=FakeLLM("hello"),
            chat_middleware=[tracing(), meter()],
        ),
        trace=otel_tracer(provider),
        # metrics=otel_meter(meter_provider) would light up histograms too.
    )
    ctx = RunContext(
        correlation_id="run-otel-demo",
        scope=Scope(org_id="acme"),
        services=services,
    )

    req = ChatRequest(messages=[Message("user", "hi")], model="gpt-4o-mini")
    await ctx.invoker.chat(req, ctx)

    for span in exporter.get_finished_spans():
        gen_ai = {k: v for k, v in (span.attributes or {}).items() if k.startswith("gen_ai")}
        print(f"[span] {span.name}  {gen_ai}")


asyncio.run(main())
```

## Production wire-up

At process startup:

```python
from agentkit.adapters.observability import (
    otel_exporter_otlp_http,
    otel_metrics_exporter_otlp_http,
    otel_tracer,
    otel_meter,
)

# One-call SDK setup. Reads OTEL_EXPORTER_OTLP_ENDPOINT etc. from env.
otel_exporter_otlp_http()
otel_metrics_exporter_otlp_http(interval_ms=15_000)

services = Services(
    trace=otel_tracer(),        # pulls the global tracer provider
    metrics=otel_meter(),       # pulls the global meter provider
    invoker=my_invoker,
)
```

Install the extra first:

```bash
pip install "arc-agentkit[observability]"
```

## How it works

**`TracePort`** is a plain Protocol: `span(name, kind, **attrs)`
returns a context manager yielding a span with `.set(k, v)` and
`.add_event(name, **fields)`. `NoopTrace` is the default so a
zero-config `RunContext` still works. `otel_tracer(provider)` returns
`OtelTracePort`, which translates `span(...)` into
`tracer.start_as_current_span(...)`, maps our `"client"` /
`"server"` / `"internal"` kind strings to `SpanKind`, and coerces
attribute values into the OTel-accepted set.

**`MetricsPort`** is separate on purpose. Spans and metrics have
different cardinality budgets — a trace backend groans on high-cardinality
tags (`user_id`, `trace_id`); a metrics backend can't tell you why one
specific run failed. `MetricsPort` exposes `add_counter` and
`record_histogram` and gets injected the same way — `Services(metrics=otel_meter())`.

**The `tracing` middleware** owns span lifecycle for every chat and
tool call, so individual patterns never open spans by hand. Span names
align with the OTel GenAI v1.41 taxonomy (`chat`, `execute_tool`) —
the kernel itself stays OTel-free; only the names and attribute keys
match, so any adapter can lift them.

**Sampling** is a kernel-level seam. Head sampling happens in
`ctx.sampler`; `otel_sampler(ratio)` builds one at any ratio between 0
and 1. When `sampler.should_sample` returns False, the span (and its
replay write) is skipped — but the underlying handler still runs.

## Gotchas

- **Don't set global providers more than once.** OTel forbids
  overriding `set_tracer_provider` after it's been set — sharing a
  global across tests gives stale-provider failures. In tests, pass an
  explicit `tracer_provider=` to `otel_tracer(...)`.
- **Metric writes are best-effort.** `OtelMetricsPort` swallows any
  exception from the exporter — a bad instrument name won't crash the
  run, it'll just drop the metric.
- **Attribute cardinality matters.** Anything you `span.set(k, v)`
  becomes a tag on the trace backend; anything you pass to
  `add_counter` / `record_histogram` becomes a tag on the metrics
  backend. Keep tags to low-cardinality axes (`model`, `provider`,
  `agent_name`, `status`) — never `user_id` or `run_id` on metrics.
- **`ObserverPort` is not `TracePort`.** `ObserverPort` is the
  product-facing observation channel (`progress`, `partial_result`,
  `interrupt`); `TracePort` is the operational one. Both are wired
  independently on `Services`.

## Related

- [Concepts · Middlewares](../concepts/middlewares.md) — where
  `tracing()` sits in the chain.
- `agentkit.adapters.observability` — the full public surface
  (`otel_tracer`, `otel_meter`, `otel_sampler`, one-call exporter
  setups).
- `agentkit.kernel.observation` module docstring — the split between
  observations (product-facing) and traces (operational).
