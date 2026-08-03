"""``RecordingTracer`` — an in-process TracePort that records every span for assertion.

Tests use this when they need to either (a) plug *something* into ``Services.trace`` so
``ctx.trace.span(...)`` works without standing up the OTel SDK, or (b) assert on the spans
the framework opened (``chat:<model>``, ``execute_tool:<name>``, attribute stamping by the
``tracing()`` middleware).

This is the test-side equivalent of the production ``OtelTracePort`` adapter — same surface
(``span``, ``current_span_id``, ``add_event_to_current_span``), but instead of forwarding to
the OTel SDK it stashes the spans on ``self.spans``. An optional ``exporter`` callable is
fired with the span object on close, so a test can collect spans into a list with
``RecordingTracer(exporter=spans.append)``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from agentkit.kernel.observation import TraceContext


class RecordingSpan:
    """A span that records its name / kind / attributes / events. Drop-in for the
    production span surface — exposes ``.set(key, value)`` and ``.add_event(name, **fields)``.
    Attributes start from the kwargs passed to ``span()`` and are mutated by ``set()``.
    """

    def __init__(self, name: str, kind: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self.kind = kind
        self.attrs: dict[str, Any] = dict(attrs)
        self.events: list[tuple[str, dict[str, Any]]] = []

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, **fields: Any) -> None:
        self.events.append((name, dict(fields)))


class RecordingTracer:
    """In-process ``TracePort`` that records every span. Optional ``exporter`` callback
    receives each span object on close (mirrors the OTel ``SpanExporter`` shape at the
    granularity tests care about).

    Satisfies the ``TracePort`` Protocol structurally:
    - ``span(name, kind, **attrs)`` — context manager yielding a ``RecordingSpan``
    - ``current_span_id()`` — returns the open span's ``TraceContext`` (synthetic ids) or None
    - ``add_event_to_current_span(name, **fields)`` — drops an event on the open span
    """

    def __init__(self, *, exporter: Callable[[RecordingSpan], None] | None = None) -> None:
        self.spans: list[RecordingSpan] = []
        self._exporter = exporter
        self._stack: list[RecordingSpan] = []

    @contextmanager
    def span(self, name: str, kind: str = "", **attrs: Any) -> Any:
        s = RecordingSpan(name, kind, attrs)
        self.spans.append(s)
        self._stack.append(s)
        try:
            yield s
        finally:
            self._stack.pop()
            if self._exporter is not None:
                self._exporter(s)

    def current_span_id(self) -> TraceContext | None:
        if not self._stack:
            return None
        # Synthetic ids derived from the span's position in the recorded list; stable for the
        # life of the run so a test can correlate observations with the span that produced them.
        idx = len(self.spans)
        return TraceContext(
            trace_id=f"{1:032x}",
            span_id=f"{idx:016x}",
        )

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        if not self._stack:
            return
        self._stack[-1].add_event(name, **fields)


__all__ = ["RecordingTracer", "RecordingSpan"]
