"""Cost from usage — a small, best-effort price table + a cache-aware `cost(model, usage)`.

USD **per 1M tokens** as `(input, output, cache_read, cache_write)`. The provider clients normalise their
`usage` to one convention before pricing: `input_tokens` = fresh (non-cached) input, `cache_read_tokens` =
prompt-cache hits (cheaper), `cache_write_tokens` = cache creation. Unknown models cost `0.0` (no guessing).

This table WILL go stale — it's a convenience default. Inject your own `pricing=` (a `cost(model, usage)`
callable) on a client for authoritative, contractual rates.
"""

from __future__ import annotations

from typing import Any

_PER_M = 1_000_000.0

# model → (input, output, cache_read, cache_write) USD / 1M tokens. Best-effort, public list prices.
_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4-5": (0.80, 4.00, 0.08, 1.00),
    "claude-opus-4-1": (15.00, 75.00, 1.50, 18.75),
    "gpt-4o": (2.50, 10.00, 1.25, 0.0),
    "gpt-4o-mini": (0.15, 0.60, 0.075, 0.0),
    "gpt-4.1": (2.00, 8.00, 0.50, 0.0),
    "gpt-4.1-mini": (0.40, 1.60, 0.10, 0.0),
    "deepseek-chat": (0.27, 1.10, 0.07, 0.0),
    "deepseek-reasoner": (0.55, 2.19, 0.14, 0.0),
}


def _lookup(model: Any) -> tuple[float, float, float, float] | None:
    if not model:
        return None
    name = str(model).lower()
    if name in _PRICES:
        return _PRICES[name]
    bare = name.split("/", 1)[-1]  # strip an OpenRouter-style "anthropic/" prefix
    return _PRICES.get(bare)


def cost(model: Any, usage: Any) -> float:
    """Cache-aware cost in USD for one call's `usage`; `0.0` for an unknown model (override via `pricing=`)."""
    rates = _lookup(model)
    if rates is None:
        return 0.0
    inp, out, cache_read, cache_write = rates
    total: float = round(
        usage.input_tokens / _PER_M * inp
        + usage.output_tokens / _PER_M * out
        + usage.cache_read_tokens / _PER_M * cache_read
        + usage.cache_write_tokens / _PER_M * cache_write,
        6,
    )
    return total


__all__ = ["cost"]
