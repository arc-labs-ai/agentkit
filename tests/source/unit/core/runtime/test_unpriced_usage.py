"""Unpriced work preserves resource evidence without claiming that spend was zero."""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from agentkit.agents.control.budget import ActorBudget
from agentkit.agents.definitions.results import AgentResult
from agentkit.core.contracts.types import CostSource, LLMResult, Scope, Usage
from agentkit.core.policies._memo_codec import decode_result, encode_result
from agentkit.core.runtime.meter import Budget, Quota
from agentkit.core.serialization.usage import usage_from_dict, usage_to_dict
from agentkit.state.checkpoints.persistence import dict_to_result, result_to_dict


def test_unknown_cost_contaminates_aggregate_but_retains_all_token_buckets():
    known = Usage(10, 20, 0.125, 30, 40, CostSource.PRICE_TABLE)
    unknown = Usage(5, 6, None, 7, 8)
    total = known + unknown
    assert total.cost_usd is None and total.cost_source == CostSource.UNPRICED
    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens, total.cache_write_tokens) == (
        15,
        26,
        37,
        48,
    )
    assert unknown + known == total
    assert Usage() + known == known
    assert (known + Usage(1, 2, 0.25)).cost_source == CostSource.MIXED


@pytest.mark.asyncio
async def test_money_bound_budget_preserves_known_ledger_and_stops_unpriced_continuation():
    budget = Budget(max_cost_usd=Decimal("1"), on_exceeded="stop")
    await budget.charge(None, Usage(cost_usd=0.125))
    verdict = await budget.charge(None, Usage(input_tokens=100, cost_usd=None))
    assert not verdict.ok and not verdict.cost_complete and "unpriced" in verdict.reason
    assert verdict.spent == budget.spent() == Decimal(".125000")
    assert budget.spent_usd is None and not budget.cost_complete
    assert budget.remaining() == Decimal(0)
    assert not (await budget.guard()).ok
    assert budget.usage.input_tokens == 100 and budget.usage.cost_usd is None


@pytest.mark.asyncio
async def test_budget_without_money_ceiling_keeps_unknown_accounting_explicit():
    budget = Budget(on_exceeded="stop")
    verdict = await budget.charge(None, Usage(cost_usd=None))
    assert verdict.ok and not verdict.cost_complete and verdict.usage.cost_usd is None
    assert (await budget.guard()).ok and budget.remaining() is None


@pytest.mark.asyncio
async def test_unpriced_quota_stops_same_tenant_until_the_accounting_window_expires():
    now = [0.0]
    quota = Quota(max_usd=Decimal("1"), window=10, clock=lambda: now[0], on_exceeded="stop")
    call = SimpleNamespace(ctx=SimpleNamespace(scope=Scope()), request=None)
    await quota.charge(call, Usage(cost_usd=0.125))
    charged = await quota.charge(call, Usage(input_tokens=10, cost_usd=None))
    assert not charged.cost_complete and charged.spent == Decimal(".125000")
    denied = await quota.guard(call)
    assert not denied.ok and not denied.cost_complete and "unpriced" in denied.reason
    now[0] = 11
    reopened = await quota.guard(call)
    assert reopened.ok and reopened.cost_complete and reopened.spent == Decimal(0)


def test_actor_envelope_fails_closed_without_destroying_the_known_decimal_ledger():
    budget = ActorBudget(max_tokens=1000, max_cost_usd=1, max_steps=10, max_wall_seconds=60)
    budget.charge(cost_usd=0.125)
    budget.charge(tokens=5, cost_usd=None, steps=1)
    assert budget.exhausted() and budget.remaining_cost() == Decimal(0)
    assert budget.used_cost() == Decimal(".125000") and budget.used_cost_usd is None
    assert not budget.cost_complete and budget.used_tokens == 5 and budget.used_steps == 1


@pytest.mark.parametrize("cost,source", [(None, CostSource.UNPRICED), (0.125, CostSource.PRICE_TABLE)])
def test_shared_cache_checkpoint_codec_retains_price_provenance_and_cache_buckets(cost, source):
    usage = Usage(10, 20, cost, 30, 40, source)
    wire = json.loads(json.dumps(usage_to_dict(usage)))
    assert usage_from_dict(wire) == usage
    cached = decode_result(json.loads(json.dumps(encode_result(LLMResult(content="evidence", usage=usage)))))
    assert cached.usage == usage
    checkpoint = dict_to_result(json.loads(json.dumps(result_to_dict(AgentResult(output="evidence", usage=usage)))))
    assert checkpoint.usage == usage
    assert usage_from_dict({"input": 1, "output": 2}).cost_usd is None


def test_unpriced_cost_cannot_be_disguised_as_numeric_zero():
    with pytest.raises(ValueError, match="requires cost_usd=None"):
        Usage(cost_usd=0, cost_source=CostSource.UNPRICED)
    with pytest.raises(ValueError, match="requires cost_usd=None"):
        Usage(cost_usd=None, cost_source=CostSource.PRICE_TABLE)


def test_text_only_stream_without_terminal_usage_is_unpriced():
    from agentkit.core.contracts.types import Delta, assemble_deltas

    response = assemble_deltas([Delta(text="unmetered reply")])
    assert response.content == "unmetered reply"
    assert response.usage.cost_usd is None and response.usage.cost_source == CostSource.UNPRICED


def test_unknown_child_settlement_preserves_known_spend_and_exhausts_parent_envelope():
    parent = ActorBudget(max_tokens=1000, max_cost_usd=1, max_steps=10, max_wall_seconds=60)
    parent.reserve_for_child(tokens=100, cost_usd=0.5, steps=2)
    parent.settle_child(
        reserved_tokens=100,
        reserved_cost_usd=0.5,
        reserved_steps=2,
        used_tokens=10,
        used_cost_usd=None,
        known_cost_usd=Decimal(".125"),
        used_steps=1,
    )
    assert parent.used_cost() == Decimal(".125000")
    assert parent.used_cost_usd is None and not parent.cost_complete and parent.exhausted()
    assert parent.reserved_cost() == Decimal(0) and parent.used_tokens == 10


def test_unpriced_budget_constructor_does_not_turn_an_unknown_display_amount_into_zero():
    budget = Budget(max_cost_usd=1, spent_usd=None, on_exceeded='stop')
    assert budget.spent_usd is None and not budget.cost_complete
    assert budget.remaining() == Decimal(0) and not budget.verdict().ok
