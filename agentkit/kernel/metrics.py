"""Metrics port — counters + histograms decoupled from tracing.

Spans give you a request-level narrative; metrics give you the
time-series rollup. A trace backend (Tempo, Jaeger) tells you
"this specific run failed"; a metrics backend (Prometheus,
Datadog metrics) tells you "the 99th percentile of first-token
latency is climbing."

Two cardinality classes:

- **Counters** — monotonic; for "how many?" (calls, errors,
  retries, cache hits). Tags can grow; the backend aggregates.
- **Histograms** — distributions; for "how big / how slow?" (token
  counts, durations, cost). Backends store bucketed counts and
  derive p50/p95/p99 from them.

NO gauges yet — gauges are stateful and need a separate fetch
contract (the backend pulls the current value). Reach for them
only when a real use case shows up.

Tags MUST stay low-cardinality. Don't tag by user_id or trace_id;
that explodes the time-series count and breaks the backend.
Acceptable tags: model, provider, agent_name, status (success /
error / cancelled). Use ``agentkit.scope.org_id`` ONLY when it's a
small, known set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsPort(Protocol):
    """The metrics seam. Implementations may be sync (OTel SDK does
    its own async dispatch internally) so the methods are plain
    ``def`` — easier to call from middleware without ``await``
    overhead.

    Operations are best-effort: implementations MUST NOT raise into
    the run on metric-write failure (log + drop instead).
    """

    def add_counter(
        self,
        name: str,
        value: int | float = 1,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Increment a monotonic counter by ``value`` (default 1).
        Standard names follow OTel semantic conventions:
        ``gen_ai.client.request.count``, ``gen_ai.client.error.count``."""

    def record_histogram(
        self,
        name: str,
        value: int | float,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Record a single observation in a distribution. Used for
        token counts, durations, costs. Standard names:
        ``gen_ai.client.token.usage``,
        ``gen_ai.client.operation.duration``."""


class NoopMetrics:
    """Default — drops everything. Zero overhead beyond the method call."""

    def add_counter(
        self,
        name: str,
        value: int | float = 1,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        return None

    def record_histogram(
        self,
        name: str,
        value: int | float,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        return None


__all__ = ["MetricsPort", "NoopMetrics"]
