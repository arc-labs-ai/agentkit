"""Compact provider accounting adapter over the dated host price table."""

from __future__ import annotations

from agentkit.core.contracts.types import ProviderUsage
from agentkit.models.catalog.pricing import PriceTable, PriceUnavailableError, price_compact


def cost(model: str | None, usage: ProviderUsage, *, cache_write_ttl: str | None = None) -> float:
    """Return an estimate, or explicitly refuse unknown/ambiguous pricing.

    Hosts can inject their contractual pricing callable on provider clients.
    This adapter owns no rates and never strips provider prefixes or dates.
    """
    value, _ = price_compact(model or "", usage, PriceTable.default(), cache_write_ttl=cache_write_ttl)
    if value is None:
        raise PriceUnavailableError(f"price unavailable for {model!r}; supply host pricing or full usage buckets")
    return round(float(value), 6)


__all__ = ["cost"]
