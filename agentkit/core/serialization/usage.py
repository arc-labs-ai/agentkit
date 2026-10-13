"""One wire codec for compact usage, including cache buckets and cost provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentkit.core.contracts.types import Usage


def usage_to_dict(usage: Usage) -> dict[str, Any]:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cost": usage.cost_usd,
        "cost_source": usage.cost_source,
        "cache_read": usage.cache_read_tokens,
        "cache_write": usage.cache_write_tokens,
    }


def usage_from_dict(value: Mapping[str, Any] | None) -> Usage:
    """Missing accounting is unpriced; it is never inferred to be a free call."""
    if value is None:
        return Usage(cost_usd=None)
    return Usage(
        int(value.get("input", 0)),
        int(value.get("output", 0)),
        None if value.get("cost") is None else float(value["cost"]),
        int(value.get("cache_read", 0)),
        int(value.get("cache_write", 0)),
        value.get("cost_source"),
    )
