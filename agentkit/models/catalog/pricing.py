"""Versioned, host-supplied prices and disjoint usage accounting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentkit.core.contracts.types import Mode
from agentkit.core.contracts.types import Usage as CompactUsage
from agentkit.core.serialization._immutable import deep_freeze
from agentkit.models.contracts.constants import CostSource
from agentkit.models.contracts.types import Usage


def normalize_usage(mode: Mode, raw: Mapping[str, Any]) -> Usage:
    """Normalize vendor counts without billing cached input twice."""

    def count(key: str, obj: Mapping[str, Any] = raw) -> int:
        return max(0, int(obj.get(key) or 0))

    if mode in (Mode.ANTHROPIC_API, Mode.CLAUDE_CODE):
        cache = raw.get("cache_creation") or {}
        hour = count("ephemeral_1h_input_tokens", cache)
        five = count("ephemeral_5m_input_tokens", cache)
        if not cache:
            five = count("cache_creation_input_tokens")
        reasoning = raw.get("thinking_tokens")
        return Usage(
            count("input_tokens"),
            count("cache_read_input_tokens"),
            five,
            hour,
            count("output_tokens"),
            max(0, int(reasoning)) if reasoning is not None else None,
        )
    details = raw.get("input_tokens_details") or {}
    output = raw.get("output_tokens_details") or {}
    cached = count("cached_input_tokens") if mode == Mode.CODEX else count("cached_tokens", details)
    written = count("cache_write_input_tokens") if mode == Mode.CODEX else count("cache_write_tokens", details)
    reasoning = raw.get("reasoning_output_tokens") if mode == Mode.CODEX else output.get("reasoning_tokens")
    return Usage(
        max(0, count("input_tokens") - cached - written),
        cached,
        written,
        0,
        count("output_tokens"),
        int(reasoning) if reasoning is not None else None,
    )


def usage_delta(current: Usage, previous: Usage) -> Usage:
    """Reconcile cumulative counters; resets are counted as a fresh call."""

    def delta(a: int, b: int) -> int:
        return a - b if a >= b else a

    return Usage(
        delta(current.input_uncached, previous.input_uncached),
        delta(current.input_cache_read, previous.input_cache_read),
        delta(current.input_cache_write_5m, previous.input_cache_write_5m),
        delta(current.input_cache_write_1h, previous.input_cache_write_1h),
        delta(current.output, previous.output),
        None if current.reasoning is None else delta(current.reasoning, previous.reasoning or 0),
    )


@dataclass(frozen=True)
class PriceTable:
    """USD per million tokens keyed by ``mode/model`` or exact model name.

    Rates name the Usage fields. ``output`` includes reasoning; reasoning is
    never charged a second time. Optional modifiers are explicit multipliers.
    """

    version: str
    date: str
    rates: Mapping[str, Mapping[str, str | Decimal]] = field(hash=False)
    modifiers: Mapping[str, str | Decimal] = field(default_factory=dict, hash=False)
    aliases: Mapping[str, str] = field(default_factory=dict, hash=False)
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for row in self.rates.values():
            for value in row.values():
                if not Decimal(value).is_finite() or Decimal(value) < 0:
                    raise ValueError("prices must be finite and nonnegative")
        for value in self.modifiers.values():
            if not Decimal(value).is_finite() or Decimal(value) < 0:
                raise ValueError("price modifiers must be finite and nonnegative")
        for target in self.aliases.values():
            if target not in self.rates:
                raise ValueError("price aliases must reference an exact rate row")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "aliases", deep_freeze(dict(self.aliases)))
        object.__setattr__(self, "rates", deep_freeze(dict(self.rates)))
        object.__setattr__(self, "modifiers", deep_freeze(dict(self.modifiers)))

    @classmethod
    def default(cls) -> PriceTable:
        """Load the dated bundled table; unknown models remain unpriced."""
        return cls.load(Path(__file__).parent / "data/prices.json")

    @classmethod
    def load(cls, path: Path | str) -> PriceTable:
        """Load a versioned JSON price file supplied by the host."""
        return cls(**json.loads(Path(path).read_text()))

    def has(self, mode: Mode, model: str) -> bool:
        return self.row(mode, model) is not None

    def row(self, mode: Mode, model: str) -> Mapping[str, str | Decimal] | None:
        """Resolve exact host-declared aliases, never guessed family/date rates."""
        key = f"{mode}/{model}"
        if key in self.rates:
            return self.rates[key]
        key = self.aliases.get(key, self.aliases.get(model, model))
        return self.rates.get(key)


def price(
    model: str, usage: Usage, table: PriceTable, *, mode: Mode = Mode.OPENAI_API, modifiers: tuple[str, ...] = ()
) -> tuple[Decimal | None, CostSource]:
    rates = table.row(mode, model)
    if rates is None:
        return None, CostSource.UNPRICED
    total = Decimal(0)
    for key in ("input_uncached", "input_cache_read", "input_cache_write_5m", "input_cache_write_1h", "output"):
        count = getattr(usage, key)
        if count and key not in rates:
            return None, CostSource.UNPRICED
        total += Decimal(count) * Decimal(rates.get(key, "0")) / Decimal(1_000_000)
    for modifier in modifiers:
        if modifier not in table.modifiers:
            return None, CostSource.UNPRICED
        total *= Decimal(table.modifiers[modifier])
    return total, CostSource.PRICE_TABLE


class PriceUnavailableError(ValueError):
    """No authoritative rate or representable cache-write TTL is available."""


def price_compact(
    model: str,
    usage: CompactUsage,
    table: PriceTable,
    *,
    mode: Mode = Mode.OPENAI_API,
    cache_write_ttl: str | None = None,
) -> tuple[Decimal | None, CostSource]:
    """Price disjoint compact counters only when cache-write duration is known.

    Compact Usage has one cache-write bucket. The caller must supply its TTL;
    mixed durations require the full models.Usage and cannot be guessed here.
    """
    if cache_write_ttl not in (None, "5m", "1h"):
        raise ValueError("cache_write_ttl must be 5m or 1h")
    if usage.cache_write_tokens and cache_write_ttl is None:
        return None, CostSource.UNPRICED
    normalized = Usage(
        input_uncached=usage.input_tokens,
        input_cache_read=usage.cache_read_tokens,
        input_cache_write_5m=usage.cache_write_tokens if cache_write_ttl == "5m" else 0,
        input_cache_write_1h=usage.cache_write_tokens if cache_write_ttl == "1h" else 0,
        output=usage.output_tokens,
    )
    return price(model, normalized, table, mode=mode)
