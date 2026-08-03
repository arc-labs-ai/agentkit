"""Meter — ONE protocol for governing a unit of work, with two instances at two scopes.

A meter is one concept — *guard a ceiling, then charge usage* — applied at two scopes: `Budget`
per run and `Quota` per tenant. Both implement the `Meter` protocol and are driven by the single
`meter` middleware, which guards every `ctx.meters` before a call and charges them after. `Budget`
is also the run's depth/concurrency authority (it owns the shared semaphore and `max_depth`).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentkit.kernel.types import Usage


class MeterExceeded(RuntimeError):
    """A ceiling (run cost/calls or tenant RPM/TPM/$) was crossed → the caller degrades gracefully."""


@runtime_checkable
class Meter(Protocol):
    async def guard(self, call: Any) -> None: ...  # raises MeterExceeded before the work runs
    async def charge(self, call: Any, usage: Usage) -> None: ...  # accrues after the work returns


def _est_tokens(call: Any) -> int:
    """≈4 chars/token estimate from a chat request's messages (0 for tool calls)."""
    msgs = getattr(getattr(call, "request", None), "messages", None) or []
    return sum(len(getattr(m, "content", "") or "") for m in msgs) // 4


@dataclass
class Budget:
    """The per-run meter + the run's depth/concurrency authority (one instance per agent-tree run)."""

    max_cost_usd: float | None = None
    max_calls: int | None = None
    max_depth: int = 4
    max_concurrency: int = 8
    spent_usd: float = 0.0
    calls: int = 0
    _lock: Any = field(default=None, compare=False, repr=False)
    _sem: Any = field(default=None, compare=False, repr=False)

    def _check(self) -> None:
        if self.max_cost_usd is not None and self.spent_usd > self.max_cost_usd:
            raise MeterExceeded(f"cost ${self.spent_usd} > ${self.max_cost_usd}")
        if self.max_calls is not None and self.calls > self.max_calls:
            raise MeterExceeded(f"calls {self.calls} > {self.max_calls}")

    def _get_lock(self) -> asyncio.Lock:
        # Lazy so the lock binds to the running loop at first use, never to
        # some other loop that happened to exist at Budget-construction time.
        if self._lock is None:
            self._lock = asyncio.Lock()
        lock: asyncio.Lock = self._lock
        return lock

    async def guard(self, call: Any = None) -> None:
        async with self._get_lock():
            self._check()

    async def charge(self, call: Any, usage: Usage) -> None:
        async with self._get_lock():
            self.spent_usd = round(self.spent_usd + usage.cost_usd, 6)
            self.calls += 1
            self._check()

    def remaining_usd(self) -> float | None:
        return None if self.max_cost_usd is None else max(0.0, self.max_cost_usd - self.spent_usd)

    def semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrency)
        sem: asyncio.Semaphore = self._sem
        return sem


@dataclass
class Quota:
    """The per-tenant meter — rolling RPM/TPM/$ windows keyed by `scope.key()`, independent of Budget.
    The noisy-neighbor guard and chargeback source. In-memory ref; prod swaps a Redis-backed Meter."""

    max_rpm: int | None = None
    max_tpm: int | None = None
    max_usd: float | None = None
    window: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _reqs: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _charges: dict[str, list[tuple[float, int, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _lock: Any = field(default=None, compare=False, repr=False)

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - self.window
        self._reqs[key] = [t for t in self._reqs[key] if t > cutoff]
        self._charges[key] = [c for c in self._charges[key] if c[0] > cutoff]

    def _get_lock(self) -> asyncio.Lock:
        # Lazy — see Budget._get_lock for the loop-binding rationale.
        if self._lock is None:
            self._lock = asyncio.Lock()
        lock: asyncio.Lock = self._lock
        return lock

    async def guard(self, call: Any) -> None:
        now = self.clock()
        key = call.ctx.scope.key()
        est = _est_tokens(call)
        async with self._get_lock():  # prune+read+count is one atomic critical section (no awaits)
            self._prune(key, now)
            reqs = len(self._reqs[key])
            toks = sum(t for _, t, _ in self._charges[key])
            cost = sum(c for _, _, c in self._charges[key])
            if self.max_rpm is not None and reqs >= self.max_rpm:
                raise MeterExceeded(f"{key}: {reqs} req ≥ {self.max_rpm} rpm")
            if self.max_tpm is not None and toks + est > self.max_tpm:
                raise MeterExceeded(f"{key}: {toks}+{est} tok > {self.max_tpm} tpm")
            if self.max_usd is not None and cost >= self.max_usd:
                raise MeterExceeded(f"{key}: ${cost} ≥ ${self.max_usd} per window")
            self._reqs[key].append(
                now
            )  # count this attempt toward RPM (a made request, even if it later fails)

    async def charge(self, call: Any, usage: Usage) -> None:
        async with self._get_lock():
            self._charges[call.ctx.scope.key()].append(
                (self.clock(), usage.total_tokens, usage.cost_usd)
            )
