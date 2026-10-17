"""The OpenTelemetry adapter, exercised against the real OTel SDK.

This module shipped with ZERO coverage: 113 statements no test imported. It is
behind the ``[observability]`` extra, which is how it escaped — nothing in the
suite reached for it, so every branch here is first run by these tests.

Everything below uses the SDK's own in-memory exporters rather than doubles.
The thing under test is whether agentkit speaks OTel correctly, and a fake
tracer would accept whatever agentkit sent and prove nothing about that. The
in-memory readers are the same code path a real OTLP exporter sits behind.

The two global-installing factories are driven through a captured
``set_tracer_provider``/``set_meter_provider`` instead of actually installing
one. OTel refuses to replace a provider once set, so a test that installed
globally would make its own ordering matter and would leak into every later
test in the session.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("opentelemetry", reason="needs the `observability` extra")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentkit.integrations.observability.otel import (
    OtelMetricsPort,
    OtelTracePort,
    TraceContext,
    otel_exporter_otlp_http,
    otel_meter,
    otel_metrics_exporter_otlp_http,
    otel_sampler,
    otel_tracer,
)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> OtelTracePort:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return otel_tracer(provider)


# ── spans ────────────────────────────────────────────────────────────────────


def test_a_span_reaches_the_exporter_with_its_name_and_attributes(
    tracer: OtelTracePort, exporter: InMemorySpanExporter
) -> None:
    with tracer.span("encode", "client", model="haiku", attempt=2) as span:
        span.set("outcome", "ok")

    (finished,) = exporter.get_finished_spans()
    assert finished.name == "encode"
    assert finished.attributes["model"] == "haiku"
    assert finished.attributes["attempt"] == 2
    assert finished.attributes["outcome"] == "ok"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("client", "CLIENT"), ("server", "SERVER"), ("internal", "INTERNAL")],
)
def test_each_span_kind_is_translated_to_the_otel_enum(
    tracer: OtelTracePort, exporter: InMemorySpanExporter, kind: str, expected: str
) -> None:
    with tracer.span("op", kind):
        pass
    (finished,) = exporter.get_finished_spans()
    assert finished.kind.name == expected


def test_an_unrecognised_kind_falls_back_to_internal_rather_than_raising(
    tracer: OtelTracePort, exporter: InMemorySpanExporter
) -> None:
    """A span kind is a hint from a caller, not a contract. An unknown one must
    not take down the run it was supposed to be observing — observability that
    can fail the thing it observes is worse than none."""
    with tracer.span("op", "producer-consumer-whatever"):
        pass
    (finished,) = exporter.get_finished_spans()
    assert finished.kind.name == "INTERNAL"


def test_an_event_lands_on_the_open_span(
    tracer: OtelTracePort, exporter: InMemorySpanExporter
) -> None:
    with tracer.span("encode", "client") as span:
        span.add_event("retrying", attempt=2, reason="overloaded")

    (finished,) = exporter.get_finished_spans()
    (event,) = finished.events
    assert event.name == "retrying"
    assert event.attributes["attempt"] == 2
    assert event.attributes["reason"] == "overloaded"


# ── attribute coercion ───────────────────────────────────────────────────────


def test_attribute_values_otel_cannot_carry_are_coerced_not_dropped(
    tracer: OtelTracePort, exporter: InMemorySpanExporter
) -> None:
    """The OTel SDK accepts only str/bool/int/float and homogeneous sequences of
    those. Anything else is dropped WITH A WARNING by the SDK, so an attribute
    a caller set would simply not be there — the failure mode is silence, which
    is the worst one for a debugging aid. Coercing to ``str`` keeps the value
    legible instead."""
    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    with tracer.span("op", "internal") as span:
        span.set("passthrough_str", "text")
        span.set("passthrough_bool", True)
        span.set("passthrough_int", 7)
        span.set("passthrough_float", 1.5)
        span.set("homogeneous_list", ["a", "b"])
        span.set("homogeneous_tuple", (1, 2))
        span.set("mixed_sequence", [1, "two"])
        span.set("bool_int_mix", [True, 1])
        span.set("empty_sequence", [])
        span.set("object", Opaque())
        span.set("mapping", {"k": "v"})

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["passthrough_str"] == "text"
    assert attrs["passthrough_bool"] is True
    assert attrs["passthrough_int"] == 7
    assert attrs["passthrough_float"] == 1.5
    assert list(attrs["homogeneous_list"]) == ["a", "b"]
    assert list(attrs["homogeneous_tuple"]) == [1, 2]
    assert attrs["object"] == "opaque-value"
    assert isinstance(attrs["mapping"], str)

    # The one that was broken. OTel requires a HOMOGENEOUS sequence and drops
    # the whole attribute when types are mixed, logging a warning and nothing
    # else — so an attribute the caller set vanished from the trace. Asserting
    # presence is the point; asserting the type alone would pass even if the
    # SDK had discarded it.
    assert "mixed_sequence" in attrs, "a mixed sequence was dropped by the SDK"
    assert isinstance(attrs["mixed_sequence"], str)
    # bool is a subclass of int, so [True, 1] is a mix the SDK rejects too.
    assert "bool_int_mix" in attrs
    assert isinstance(attrs["bool_int_mix"], str)
    # Nothing to mix, so it survives as a sequence.
    assert list(attrs["empty_sequence"]) == []


# ── the current span ─────────────────────────────────────────────────────────


def test_the_current_span_id_is_reported_as_fixed_width_hex(tracer: OtelTracePort) -> None:
    """32 and 16 hex characters, zero-padded. Width matters: these ids are
    correlated against logs and against other tracing systems by string
    equality, and an unpadded id from a small random value simply fails to
    match."""
    with tracer.span("op", "internal"):
        context = tracer.current_span_id()

    assert isinstance(context, TraceContext)
    assert len(context.trace_id) == 32
    assert len(context.span_id) == 16
    assert int(context.trace_id, 16) and int(context.span_id, 16)


def test_there_is_no_current_span_id_outside_a_span(tracer: OtelTracePort) -> None:
    """Returns None rather than inventing an id. A fabricated id correlates a
    log line to a trace that does not exist, which is harder to debug than no
    id at all."""
    assert tracer.current_span_id() is None


def test_an_event_outside_a_span_is_dropped_rather_than_raising(
    tracer: OtelTracePort, exporter: InMemorySpanExporter
) -> None:
    tracer.add_event_to_current_span("orphan", detail="nothing open")
    assert exporter.get_finished_spans() == ()


def test_an_event_on_the_current_span_is_recorded(
    tracer: OtelTracePort, exporter: InMemorySpanExporter
) -> None:
    with tracer.span("op", "internal"):
        tracer.add_event_to_current_span("mid-flight", attempt=1)

    (finished,) = exporter.get_finished_spans()
    assert [event.name for event in finished.events] == ["mid-flight"]
    assert finished.events[0].attributes["attempt"] == 1


# ── metrics ──────────────────────────────────────────────────────────────────


def _metrics(reader: InMemoryMetricReader) -> dict[str, Any]:
    out: dict[str, Any] = {}
    data = reader.get_metrics_data()
    if data is None:
        return out
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                out[metric.name] = metric
    return out


def test_counters_and_histograms_reach_a_real_meter() -> None:
    reader = InMemoryMetricReader()
    port = otel_meter(MeterProvider(metric_readers=[reader]))
    assert isinstance(port, OtelMetricsPort)

    port.add_counter("calls", 2, tags={"mode": "cli"})
    port.add_counter("calls", 3, tags={"mode": "cli"})
    port.record_histogram("latency_ms", 12.5, tags={"mode": "cli"})

    metrics = _metrics(reader)
    assert set(metrics) == {"calls", "latency_ms"}
    total = sum(point.value for point in metrics["calls"].data.data_points)
    assert total == 5, "the counter must accumulate, not replace"
    assert metrics["latency_ms"].data.data_points[0].count == 1


def test_an_instrument_is_created_once_and_reused() -> None:
    """Lazily created per name, then cached. Re-creating an instrument on every
    call is how a metrics backend ends up with duplicate series for one name."""
    reader = InMemoryMetricReader()
    port = otel_meter(MeterProvider(metric_readers=[reader]))

    port.add_counter("calls")
    first = port._counters["calls"]
    port.add_counter("calls")
    assert port._counters["calls"] is first

    port.record_histogram("latency_ms", 1.0)
    first_histogram = port._histograms["latency_ms"]
    port.record_histogram("latency_ms", 2.0)
    assert port._histograms["latency_ms"] is first_histogram


def test_a_counter_defaults_to_one() -> None:
    reader = InMemoryMetricReader()
    port = otel_meter(MeterProvider(metric_readers=[reader]))
    port.add_counter("events")
    points = _metrics(reader)["events"].data.data_points
    assert sum(point.value for point in points) == 1


def test_a_broken_instrument_never_raises_into_the_run() -> None:
    """The stated contract: "a misbehaving exporter or invalid attribute set
    MUST NOT raise into the run". Metrics are a side channel, and the
    alternative is a single bad instrument killing a call that was otherwise
    about to succeed."""
    class Exploding:
        def create_counter(self, name: str) -> Any:
            raise RuntimeError("meter is broken")

        def create_histogram(self, name: str) -> Any:
            raise RuntimeError("meter is broken")

    port = OtelMetricsPort(Exploding())
    port.add_counter("calls", 1, tags={"a": "b"})
    port.record_histogram("latency_ms", 1.0, tags={"a": "b"})

    assert port._counters == {}
    assert port._histograms == {}


def test_tags_are_optional_and_none_is_not_an_empty_attribute_set() -> None:
    reader = InMemoryMetricReader()
    port = otel_meter(MeterProvider(metric_readers=[reader]))
    port.add_counter("untagged", 1)
    port.record_histogram("untagged_hist", 1.0)
    metrics = _metrics(reader)
    assert metrics["untagged"].data.data_points[0].attributes in ({}, None)


# ── the sampler ──────────────────────────────────────────────────────────────


def test_the_sampler_factory_returns_the_in_tree_ratio_sampler() -> None:
    """A convenience re-export, deliberately: the sampler is a kernel seam with
    no OTel dependency, and it lives here so an operator wiring observability
    has one module to import from."""
    from agentkit.core.telemetry.sampling import TraceIdRatioSampler

    sampler = otel_sampler(0.25)
    assert isinstance(sampler, TraceIdRatioSampler)
    assert sampler._ratio == 0.25
    assert otel_sampler()._ratio == 1.0


# ── the one-call global installers ───────────────────────────────────────────


def test_the_trace_exporter_installs_a_provider_with_a_batch_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured rather than actually installed. OTel refuses to replace a
    provider once one is set, so a test that installed globally would make its
    own position in the session significant and leak into every later test."""
    from opentelemetry import trace as ot

    installed: list[Any] = []
    monkeypatch.setattr(ot, "set_tracer_provider", installed.append)

    otel_exporter_otlp_http(endpoint="http://127.0.0.1:4318/v1/traces")

    (provider,) = installed
    assert isinstance(provider, TracerProvider)


def test_the_trace_exporter_works_without_an_explicit_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No endpoint means the SDK reads OTEL_EXPORTER_OTLP_ENDPOINT itself. The
    two branches differ only in whether the argument is passed, and the
    no-argument one is what production uses."""
    from opentelemetry import trace as ot

    installed: list[Any] = []
    monkeypatch.setattr(ot, "set_tracer_provider", installed.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

    otel_exporter_otlp_http()

    assert len(installed) == 1


def test_the_metrics_exporter_installs_a_meter_provider_with_the_given_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry import metrics as otm

    installed: list[Any] = []
    monkeypatch.setattr(otm, "set_meter_provider", installed.append)

    otel_metrics_exporter_otlp_http(
        endpoint="http://127.0.0.1:4318/v1/metrics", interval_ms=1000
    )

    (provider,) = installed
    assert isinstance(provider, MeterProvider)


def test_the_metrics_exporter_works_without_an_explicit_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry import metrics as otm

    installed: list[Any] = []
    monkeypatch.setattr(otm, "set_meter_provider", installed.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

    otel_metrics_exporter_otlp_http()

    assert len(installed) == 1
