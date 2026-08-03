"""OpenTelemetry adapter for agentkit's TracePort.

Pure OTel SDK — no third-party vendor dependency. Operators wire
this via ``Services(trace=otel_tracer())``; spans flow through
whatever exporter the OTel SDK is configured with (OTLP-HTTP by
default — see ``otel_exporter_otlp_http()`` below).

Behind the ``[observability]`` extra; core agentkit imports nothing
from here.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from agentkit.kernel.observation import TraceContext


def otel_tracer(tracer_provider: Any = None) -> OtelTracePort:
    """Construct a TracePort backed by the OpenTelemetry SDK.

    Production: pass nothing — pulls the global tracer (configured
    via env vars OTEL_SERVICE_NAME / OTEL_EXPORTER_OTLP_ENDPOINT /
    OTEL_EXPORTER_OTLP_HEADERS / ... per the standard OTel SDK
    convention).

    Tests: pass an explicit ``tracer_provider`` (e.g. a
    ``TracerProvider`` wired to ``InMemorySpanExporter``) so each
    test gets its own isolated provider. OTel forbids overriding
    the global ``set_tracer_provider`` once it's been set, so
    sharing a global across tests gives stale-provider failures.

    Raises ImportError when ``opentelemetry-sdk`` isn't installed
    (i.e., the ``[observability]`` extra wasn't installed); the
    error message names the extra so the operator knows what to do.
    """
    try:
        from opentelemetry import trace as ot  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "OTel observability requires the [observability] extra: "
            "pip install arc-agentkit[observability]"
        ) from exc
    if tracer_provider is not None:
        return OtelTracePort(tracer_provider.get_tracer("agentkit"))
    return OtelTracePort(ot.get_tracer("agentkit"))


class OtelTracePort:
    """Concrete TracePort wrapping an OTel ``Tracer``.

    Implements:
    - ``span(name, kind, **attrs)`` — context manager opening an OTel span
    - ``current_span_id()`` — returns the in-context span's ``(trace_id, span_id)`` as a TraceContext
    - ``add_event_to_current_span(name, **fields)`` — drops an event on the open span
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    @contextlib.contextmanager
    def span(self, name: str, kind: str, **attrs: Any) -> Iterator[_OtelSpanAdapter]:
        # Return type is ``Iterator`` (the yielded value's type), not
        # ``ContextManager``. ``@contextmanager`` wraps a generator;
        # typeshed types the decorator so the wrapped fn's
        # ``Iterator[T]`` becomes ``ContextManager[T]``. Declaring the
        # wrapped fn as returning ``ContextManager[Any]`` would break
        # inference at every ``with tracer.span(...) as span:
        # span.set(k, v)`` call site — mypy would see
        # ``ContextManager[Any]`` and complain about ``.set``.
        # Translate our "kind" string into the OTel SpanKind enum
        from opentelemetry.trace import SpanKind  # type: ignore[import-not-found]

        kind_map = {
            "client": SpanKind.CLIENT,
            "server": SpanKind.SERVER,
            "internal": SpanKind.INTERNAL,
        }
        with self._tracer.start_as_current_span(
            name,
            kind=kind_map.get(kind, SpanKind.INTERNAL),
            attributes={k: _coerce_attr(v) for k, v in attrs.items()},
        ) as span:
            yield _OtelSpanAdapter(span)

    def current_span_id(self) -> TraceContext | None:
        from opentelemetry import trace as ot

        span = ot.get_current_span()
        if span is None or not span.is_recording():
            return None
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return None
        return TraceContext(
            trace_id=f"{ctx.trace_id:032x}",
            span_id=f"{ctx.span_id:016x}",
        )

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        from opentelemetry import trace as ot

        span = ot.get_current_span()
        if span is None or not span.is_recording():
            return
        span.add_event(name, attributes={k: _coerce_attr(v) for k, v in fields.items()})


class _OtelSpanAdapter:
    """Thin shim so callers using our ``with ... as span: span.set(k, v)``
    pattern keep working on top of OTel's ``span.set_attribute(k, v)``."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, _coerce_attr(value))

    def add_event(self, name: str, **fields: Any) -> None:
        self._span.add_event(name, attributes={k: _coerce_attr(v) for k, v in fields.items()})


def _coerce_attr(v: Any) -> Any:
    """OTel SDK accepts only str, bool, int, float, or sequences of
    those for attribute values. Coerce everything else to string."""
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v):
        return list(v)
    return str(v)


def otel_exporter_otlp_http(*, endpoint: str | None = None) -> None:
    """One-call setup of the OTel SDK with an OTLP-HTTP exporter.

    Builds a ``TracerProvider`` with a ``BatchSpanProcessor`` wrapping
    an OTLP-HTTP exporter pointed at ``endpoint`` (or the
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var). Call once at process
    startup BEFORE ``otel_tracer()`` is consumed — sets the global
    tracer provider.

    The convenience version: operators who want a "give me OTel
    with sensible defaults" path call this. Operators with their
    own setup (custom processor, multiple exporters, etc.) skip
    this and configure ``opentelemetry.trace.set_tracer_provider``
    themselves.
    """
    try:
        from opentelemetry import trace as ot
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "OTLP-HTTP exporter requires the [observability] extra: "
            "pip install arc-agentkit[observability]"
        ) from exc
    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    ot.set_tracer_provider(provider)


def otel_meter(meter_provider: Any = None, *, instrument_name: str = "agentkit") -> Any:
    """Construct a MetricsPort backed by the OpenTelemetry SDK.

    Production: pass nothing — pulls the global meter (configured
    via env vars OTEL_EXPORTER_OTLP_ENDPOINT /
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT). Tests: pass an explicit
    ``meter_provider`` so each test gets isolated state.

    Raises ImportError when ``opentelemetry-sdk`` isn't installed
    (i.e., the ``[observability]`` extra wasn't installed); the
    error message names the extra so the operator knows what to do.
    """
    try:
        from opentelemetry import metrics as otm
    except ImportError as exc:
        raise ImportError(
            "OTel metrics requires the [observability] extra: pip install arc-agentkit[observability]"
        ) from exc
    if meter_provider is not None:
        return OtelMetricsPort(meter_provider.get_meter(instrument_name))
    return OtelMetricsPort(otm.get_meter(instrument_name))


class OtelMetricsPort:
    """OTel SDK-backed MetricsPort. Lazily creates instruments per name —
    first call to ``add_counter("X")`` creates the counter, subsequent
    calls reuse it. Implements the ``MetricsPort`` Protocol structurally
    (``add_counter`` + ``record_histogram``).

    All operations are best-effort: a misbehaving exporter or invalid
    attribute set MUST NOT raise into the run. Failures are swallowed
    silently (the metric is dropped); the alternative is a single
    misbehaving instrument killing the whole run."""

    def __init__(self, meter: Any) -> None:
        self._meter = meter
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    def add_counter(
        self,
        name: str,
        value: int | float = 1,
        *,
        tags: Any = None,
    ) -> None:
        try:
            counter = self._counters.get(name)
            if counter is None:
                counter = self._meter.create_counter(name)
                self._counters[name] = counter
            counter.add(value, attributes=dict(tags) if tags else None)
        except Exception:  # noqa: BLE001 — never raise into the run
            pass

    def record_histogram(
        self,
        name: str,
        value: int | float,
        *,
        tags: Any = None,
    ) -> None:
        try:
            hist = self._histograms.get(name)
            if hist is None:
                hist = self._meter.create_histogram(name)
                self._histograms[name] = hist
            hist.record(value, attributes=dict(tags) if tags else None)
        except Exception:  # noqa: BLE001 — never raise into the run
            pass


def otel_sampler(ratio: float = 1.0) -> Any:
    """Convenience: build a ``SamplerPort`` from agentkit's pure-Python
    ``TraceIdRatioSampler``. ``ratio=1.0`` = always-on; ``ratio=0.1`` = 10%.

    The sampler is a kernel-level seam (no OTel dependency) so this
    factory just wraps the in-tree impl — it lives next to the OTel
    metrics/tracer factories so operators have ONE module to import
    from when wiring observability. For richer policies (parent-based,
    rule-based), construct your own ``SamplerPort`` impl and inject it
    directly into ``Services(sampler=...)``."""
    from agentkit.kernel.sampling import TraceIdRatioSampler

    return TraceIdRatioSampler(ratio)


def otel_metrics_exporter_otlp_http(
    *,
    endpoint: str | None = None,
    interval_ms: int = 60000,
) -> None:
    """One-call OTel metrics SDK setup with an OTLP-HTTP exporter.

    Sister to ``otel_exporter_otlp_http`` for traces. Call once at
    startup BEFORE ``otel_meter()`` is consumed — sets the global
    meter provider with a ``PeriodicExportingMetricReader`` pushing
    to the OTLP-HTTP endpoint every ``interval_ms`` (default 60s, the
    OTel default).

    The convenience version: operators who want a "give me OTel metrics
    with sensible defaults" path call this. Operators with their own
    setup (custom readers, multiple exporters, etc.) skip this and
    configure ``opentelemetry.metrics.set_meter_provider`` themselves.
    """
    try:
        from opentelemetry import metrics as otm
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore[import-not-found]
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.metrics.export import (  # type: ignore[import-not-found]
            PeriodicExportingMetricReader,
        )
    except ImportError as exc:
        raise ImportError(
            "OTLP-HTTP metrics exporter requires the [observability] extra: "
            "pip install arc-agentkit[observability]"
        ) from exc
    exporter = OTLPMetricExporter(endpoint=endpoint) if endpoint else OTLPMetricExporter()
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=interval_ms)
    provider = MeterProvider(metric_readers=[reader])
    otm.set_meter_provider(provider)


__all__ = [
    "otel_tracer",
    "OtelTracePort",
    "otel_exporter_otlp_http",
    "otel_meter",
    "OtelMetricsPort",
    "otel_sampler",
    "otel_metrics_exporter_otlp_http",
]
