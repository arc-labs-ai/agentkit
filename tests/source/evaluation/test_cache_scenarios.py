"""Provider counters and policy facts independently determine cache acceptance."""

import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.types import Binding, CallLimits, Turn
from tests.evals.suites.caching import CASES, decision_checks, execute_workload, policy_prompt
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


def review(*, cached=0, revoked=False):
    value = {"policy_id": "RG-057", "refund_cap_cents": 7109,
             "decision": "refused" if revoked else "approved", "approval_current": not revoked}
    return Turn((json.dumps(value),), value=value,
                raw_usage={"input_tokens": 10000, "input_tokens_details": {"cached_tokens": cached},
                           "output_tokens": 50})


def environment(tmp_path, adapter):
    return SimpleNamespace(runtime=CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES),
                           binding=Binding("openai_api", "test"), limits=CallLimits(max_turns=1),
                           event_sink=lambda event: None, call_id="cache", workspace=tmp_path)


def test_handbook_revision_changes_prefix_at_start_and_oracle_rejects_stale_approval():
    original, warm, revised = policy_prompt("scope"), policy_prompt("scope"), policy_prompt("scope", revoked=True)
    assert original.prefix_digest == warm.prefix_digest != revised.prefix_digest
    assert original.system != revised.system and len(original.system) > 40000
    assert not decision_checks(review().value, revoked=False)
    assert decision_checks(review().value, revoked=True)


@pytest.mark.asyncio
async def test_cache_reuse_requires_actual_usage_counter_and_retains_cold_warm_evidence(tmp_path):
    adapter = ScriptedAdapter(review(), review(cached=8192))
    result = await execute_workload(CASES[0], environment(tmp_path, adapter))
    assert result.checks == () and len(adapter.requests) == 2
    assert result.evidence["phases"][1]["usage"]["input_cache_read"] == 8192
    assert adapter.requests[0][1].prefix_digest == adapter.requests[1][1].prefix_digest


@pytest.mark.asyncio
async def test_cache_misses_remain_failed_after_bounded_retries(tmp_path):
    adapter = ScriptedAdapter(*(review() for _ in range(4)))
    result = await execute_workload(CASES[0], environment(tmp_path, adapter))
    assert "provider did not demonstrate a qualifying real prefix cache read" in result.checks
    assert len(adapter.requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_value,stale_read", [(False, False), (True, False), (False, True)])
async def test_changed_policy_requires_refusal_and_invalidates_previous_prefix(tmp_path, stale_value, stale_read):
    adapter = ScriptedAdapter(review(), review(cached=8192),
                              review(cached=8192 if stale_read else 0, revoked=not stale_value))
    result = await execute_workload(CASES[1], environment(tmp_path, adapter))
    assert bool(result.checks) == (stale_value or stale_read)
    assert len(adapter.requests) == 3
    assert adapter.requests[1][1].prefix_digest != adapter.requests[2][1].prefix_digest


@pytest.mark.asyncio
async def test_shared_cache_workflow_stops_before_dispatch_when_cost_unknown(tmp_path):
    adapter = ScriptedAdapter(replace(review(), usage_complete=False))
    env = environment(tmp_path, adapter)
    env.limits = CallLimits(max_cost_usd=Decimal("0.1"), max_turns=1)
    result = await execute_workload(CASES[0], env)
    assert result.checks and len(adapter.requests) == 1
