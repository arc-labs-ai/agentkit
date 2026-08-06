"""Cost from usage — a small, best-effort price table + a cache-aware `cost(model, usage)`.

USD **per 1M tokens** as `(input, output, cache_read, cache_write)`. The provider clients normalise their
`usage` to one convention before pricing: `input_tokens` = fresh (non-cached) input, `cache_read_tokens` =
prompt-cache hits (cheaper), `cache_write_tokens` = cache creation. Unknown models cost `0.0` (no guessing).

This table WILL go stale — it's a convenience default. Inject your own `pricing=` (a `cost(model, usage)`
callable) on a client for authoritative, contractual rates.
"""

from __future__ import annotations

import re
from typing import Any

_PER_M = 1_000_000.0

# Providers return a dated release id on every response — Anthropic uses
# ``claude-haiku-4-5-20251001`` (no dashes) and OpenAI uses
# ``gpt-4o-mini-2024-07-18`` (dashes). The pricing table below is keyed on
# the bare family name, so we strip the trailing date component before
# falling back to a family-level match. This regex matches both shapes.
_DATE_SUFFIX = re.compile(r"-\d{4}-?\d{2}-?\d{2}$")

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
    """Best-effort family-level lookup with date-suffix fallback.

    Providers return dated release ids on every ``LLMResult.model`` — e.g.
    ``claude-haiku-4-5-20251001`` or ``gpt-4o-mini-2024-07-18``. The
    pricing table is keyed on the bare family (``claude-haiku-4-5`` /
    ``gpt-4o-mini``), so we try the exact name first, then strip an
    OpenRouter-style ``provider/`` prefix, then strip the trailing date,
    then both. Order preserves any authoritative dated entry the caller
    registered manually.
    """
    if not model:
        return None
    name = str(model).lower()
    if name in _PRICES:
        return _PRICES[name]
    bare = name.split("/", 1)[-1]  # strip an OpenRouter-style "anthropic/" prefix
    if bare in _PRICES:
        return _PRICES[bare]
    undated = _DATE_SUFFIX.sub("", name)
    if undated in _PRICES:
        return _PRICES[undated]
    bare_undated = _DATE_SUFFIX.sub("", bare)
    return _PRICES.get(bare_undated)


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
