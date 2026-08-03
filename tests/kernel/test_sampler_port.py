"""SamplerPort — head-sampling decision at span open.

The sampler is consulted before opening every span; ``False`` skips
the whole observability pipeline for that call. The default
``AlwaysOnSampler`` matches the legacy no-sampler behaviour;
``TraceIdRatioSampler`` records a deterministic ratio keyed off
``correlation_id`` so all spans within ONE run share the keep/drop
decision (no orphan child spans).
"""

from __future__ import annotations

import pytest

from agentkit.kernel.sampling import AlwaysOnSampler, SamplerPort, TraceIdRatioSampler

# ── AlwaysOn ─────────────────────────────────────────────────────────────────


def test_always_on_sampler_returns_true_for_any_call():
    s = AlwaysOnSampler()
    assert s.should_sample(operation="chat", correlation_id="r-1") is True
    assert s.should_sample(operation="execute_tool", correlation_id="r-2") is True
    assert (
        s.should_sample(
            operation="invoke_agent",
            correlation_id="r-3",
            attrs={"gen_ai.request.model": "m"},
        )
        is True
    )
    assert isinstance(s, SamplerPort)


# ── TraceIdRatioSampler edge cases ───────────────────────────────────────────


def test_trace_id_ratio_sampler_one_keeps_every_call():
    s = TraceIdRatioSampler(1.0)
    for i in range(50):
        assert s.should_sample(operation="chat", correlation_id=f"r-{i}") is True


def test_trace_id_ratio_sampler_zero_drops_every_call():
    s = TraceIdRatioSampler(0.0)
    for i in range(50):
        assert s.should_sample(operation="chat", correlation_id=f"r-{i}") is False


def test_trace_id_ratio_sampler_is_deterministic_per_correlation_id():
    """Same id → same answer. This is THE coherence property: all
    spans in one run share the keep/drop decision so children never
    end up as orphans."""
    s = TraceIdRatioSampler(0.5)
    cid = "run_abc_xyz"
    first = s.should_sample(operation="chat", correlation_id=cid)
    for _ in range(20):
        assert s.should_sample(operation="chat", correlation_id=cid) is first
    # Different operations on the same id also agree (the operation
    # name is metadata; only the id keys the decision).
    assert s.should_sample(operation="execute_tool", correlation_id=cid) is first
    assert s.should_sample(operation="invoke_agent", correlation_id=cid) is first


def test_trace_id_ratio_sampler_approximates_target_ratio_across_ids():
    """Across many distinct correlation_ids the keep-rate approaches
    the configured ratio. 1000 ids is plenty to keep the variance
    inside the +/- 0.1 envelope for a 0.5 target."""
    s = TraceIdRatioSampler(0.5)
    kept = sum(
        1 for i in range(1000) if s.should_sample(operation="chat", correlation_id=f"run-{i}")
    )
    rate = kept / 1000
    assert 0.4 <= rate <= 0.6, f"keep-rate {rate} drifted outside the 0.5 +/- 0.1 envelope"


def test_trace_id_ratio_sampler_rejects_out_of_range_ratio():
    with pytest.raises(ValueError):
        TraceIdRatioSampler(-0.01)
    with pytest.raises(ValueError):
        TraceIdRatioSampler(1.01)


def test_trace_id_ratio_sampler_satisfies_sampler_port_protocol():
    """Structural typing — the concrete impl matches the Protocol."""
    s = TraceIdRatioSampler(0.5)
    assert isinstance(s, SamplerPort)
