"""Sampling port — decide whether a span is recorded.

Tracing every span at production scale saturates any backend. The
sampler is consulted at span open; when it returns ``False`` the
middleware skips the span entirely. ``ctx.emit`` observations are
NOT sampled — those are always-on (the spec calls this out: spans
are sampled, observations aren't).

Three common strategies:

- **AlwaysOn** (the no-op default) — every span recorded
- **TraceIdRatioBased(p)** — record a deterministic ``p`` fraction
  keyed off ``correlation_id`` (so all spans within one run are
  either all kept or all dropped — coherent traces, no orphan spans)
- **ParentBased(child_sampler)** — defer to the parent's decision
  (the OTel default)

Always-on for sub-1k-runs/day workloads; ratio-based once we're
above that.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class SamplerPort(Protocol):
    """Decides whether a span should be recorded.

    Called by the tracing middleware BEFORE opening the span; when
    it returns ``False`` no span is opened, no attributes stamped,
    no replay record written. The whole observability pipeline
    short-circuits for that call.

    Implementations should be FAST (microseconds) — they're on the
    hot path. A 1ms sampler erases its own value.
    """

    def should_sample(
        self,
        *,
        operation: str,
        correlation_id: str,
        attrs: Mapping[str, str | int | float] | None = None,
    ) -> bool:
        """Return True to record the span, False to skip.

        ``operation`` is the span name (``"chat"``, ``"execute_tool"``,
        ``"invoke_agent"``, ...). ``correlation_id`` is the run id —
        ratio-based samplers hash it so all spans in one run share
        the decision. ``attrs`` is the request-time attribute set
        (useful for rule-based sampling — "always sample errors,"
        "always sample org X")."""


class AlwaysOnSampler:
    """Default — sample every span. Zero overhead; no sampling
    applied."""

    def should_sample(
        self,
        *,
        operation: str,
        correlation_id: str,
        attrs: Mapping[str, str | int | float] | None = None,
    ) -> bool:
        return True


class TraceIdRatioSampler:
    """Sample a deterministic ratio of runs.

    Hashes ``correlation_id`` modulo a fixed denominator; keeps the
    span if the hash falls below ``ratio * denominator``. Result:
    every span within the SAME run shares the keep/drop decision,
    so traces are coherent (no orphan child spans). Across runs,
    you get the configured ratio in aggregate.

    Ratio 0.0 = drop everything; 1.0 = keep everything (same as
    AlwaysOn).
    """

    def __init__(self, ratio: float) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")
        self._ratio = ratio
        # 2**32 is a common denominator; gives us 1-in-4-billion
        # precision which is plenty.
        self._threshold = int(ratio * (1 << 32))

    def should_sample(
        self,
        *,
        operation: str,
        correlation_id: str,
        attrs: Mapping[str, str | int | float] | None = None,
    ) -> bool:
        if self._threshold == 0:
            return False
        if self._threshold >= (1 << 32):
            return True
        # Stable hash — built-in ``hash`` is salted per-process, so
        # use a deterministic one (FNV-1a 32 inline; ~6 lines).
        h = 2166136261
        for b in correlation_id.encode("utf-8"):
            h = ((h ^ b) * 16777619) & 0xFFFFFFFF
        return h < self._threshold


__all__ = ["SamplerPort", "AlwaysOnSampler", "TraceIdRatioSampler"]
