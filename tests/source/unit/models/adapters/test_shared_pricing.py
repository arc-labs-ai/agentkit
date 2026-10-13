"""Compact provider accounting uses the same dated rates as CallRuntime."""

from decimal import Decimal

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.core.contracts.types import Usage as CompactUsage
from agentkit.models.catalog.compact import cost
from agentkit.models.catalog.pricing import PriceTable, PriceUnavailableError, price, price_compact
from agentkit.models.contracts.constants import CostSource
from agentkit.models.contracts.types import Usage


def test_haiku_drift_is_removed_and_all_counters_agree() -> None:
    table = PriceTable.default()
    compact = CompactUsage(input_tokens=1000, output_tokens=500, cache_read_tokens=2000, cache_write_tokens=100)
    expected = Decimal(".003825")
    assert cost("claude-haiku-4-5", compact, cache_write_ttl="5m") == float(expected)
    assert price("claude-haiku-4-5", Usage(1000, 2000, 100, 0, 500), table) == (expected, CostSource.PRICE_TABLE)
    assert table.date == "2026-10-13" and table.sources


def test_ambiguous_write_duration_is_unpriced_even_for_known_model() -> None:
    usage = CompactUsage(cache_write_tokens=1000)
    assert price_compact("claude-haiku-4-5", usage, PriceTable.default()) == (None, CostSource.UNPRICED)
    with pytest.raises(PriceUnavailableError):
        cost("claude-haiku-4-5", usage)
    assert cost("claude-haiku-4-5", usage, cache_write_ttl="1h") == 0.002
    assert cost("claude-haiku-4-5", usage, cache_write_ttl="5m") == 0.00125


def test_missing_model_or_missing_token_bucket_never_means_free() -> None:
    with pytest.raises(PriceUnavailableError):
        cost("unknown", CompactUsage(input_tokens=1))
    with pytest.raises(PriceUnavailableError):
        cost("gpt-4.1", CompactUsage(cache_write_tokens=1), cache_write_ttl="5m")
    assert cost("gpt-4.1", CompactUsage()) == 0.0


def test_only_declared_snapshot_aliases_have_prices() -> None:
    table = PriceTable.default()
    assert table.has(Mode.OPENAI_API, "gpt-4o-mini-2024-07-18")
    assert not table.has(Mode.OPENAI_API, "gpt-4o-mini-2099-01-01")
    assert not table.has(Mode.ANTHROPIC_API, "anthropic/claude-haiku-4-5")
    host = PriceTable("contract", "2026-10-13", {"m": {"input_uncached": "9"}}, aliases={"vendor/m": "m"})
    assert host.has(Mode.OPENAI_API, "vendor/m")
    assert price_compact("vendor/m", CompactUsage(input_tokens=1_000_000), host)[0] == Decimal("9")


def test_alias_cannot_reference_an_unpriced_family() -> None:
    with pytest.raises(ValueError, match="exact rate row"):
        PriceTable("invalid", "2026-10-13", {}, aliases={"x": "unknown"})
