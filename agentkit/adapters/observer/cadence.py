"""Emission-cadence wrappers: `PolicyObserver` (kind filter) and `RollupObserver` (buffer→summary).

Both wrap an inner `ObserverPort` and share `_ALWAYS_FORWARD` — `result`/`error` (CRITICAL_KINDS) plus the
human-gate `interrupt` are never silenced or delayed behind a cadence policy."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agentkit.kernel.observation import CRITICAL_KINDS, Observation, ObservationKind

_ALWAYS_FORWARD = frozenset(CRITICAL_KINDS) | {"interrupt"}  # result/error + the human-gate signal


logger = logging.getLogger(__name__)


class PolicyObserver:
    """The emission-cadence knob: forwards only selected kinds to an `inner` observer.

    `result`/`error`/`interrupt` **always** pass (never silence a result or a human gate); other kinds pass
    only if `allow` is `None` (everything) or contains the kind. Compose it around any `ObserverPort`
    (e.g. a `QueueObserver`); the consumer still streams from the inner observer. Emit-only by design —
    an `ObserverPort` needs only `emit`."""

    def __init__(self, inner: Any, *, allow: frozenset[str] | set[str] | None = None) -> None:
        self._inner = inner
        self._allow = None if allow is None else frozenset(allow)

    async def emit(self, obs: Observation) -> None:
        if self._allow is None or obs.kind in self._allow or obs.kind in _ALWAYS_FORWARD:
            await self._inner.emit(obs)

    async def close(self) -> None:
        """Forward shutdown to `inner`. ``ObserverPort`` declares ``close()``,
        so without it ``isinstance(PolicyObserver(...), ObserverPort)`` was
        ``False`` (measured) — a cadence wrapper did not satisfy the very
        Protocol it exists to compose. Worse, a wrapper that swallows
        ``close`` strands a buffering inner: wrapping a `RollupObserver`
        dropped its trailing summary entirely."""
        inner_close = getattr(self._inner, "close", None)
        if inner_close is not None:
            await inner_close()

    @classmethod
    def everything(cls, inner: Any) -> PolicyObserver:
        """Forward every observation (the streaming/step cadence)."""
        return cls(inner, allow=None)

    @classmethod
    def summaries(cls, inner: Any) -> PolicyObserver:
        """Forward rolled-up `summary`/`progress` (+ always-forwarded), drop finer events."""
        return cls(inner, allow={"summary", "progress"})

    @classmethod
    def result_only(cls, inner: Any) -> PolicyObserver:
        """Send only on a complete result/error (+ interrupt) — the quiet-pipeline cadence."""
        return cls(inner, allow=frozenset())


def _default_rollup(observations: Sequence[Observation]) -> str:
    """Dependency-free roll-up: a count + the latest line. Inject an LLM/`Compactor`-backed `summarize`
    for prose."""
    n = len(observations)
    latest = observations[-1].render if observations else ""
    return f"{n} updates; latest: {latest}"


class RollupObserver:
    """The 'rolled-up summary' cadence: buffer non-critical observations and emit **one**
    `summary` every `every` of them — flushing the buffer first whenever a critical kind
    (`result`/`error`/`interrupt`) passes through, and again on `close()`.

    `summarize(buffer) -> str` builds the roll-up text; it may be **sync or async** (async is the hook for
    an LLM judge or a `Compactor`-backed summariser). Critical observations are always forwarded
    immediately (after the flush), so a result is never delayed behind buffered progress."""

    def __init__(
        self,
        inner: Any,
        *,
        every: int = 8,
        summarize: Callable[[Sequence[Observation]], str | Awaitable[str]] | None = None,
        kind: ObservationKind = "summary",
    ) -> None:
        self._inner = inner
        self._every = max(1, int(every))
        self._summarize = summarize or _default_rollup
        self._kind = kind
        self._buf: list[Observation] = []
        # ``_flush`` awaits a caller-supplied ``summarize`` (the async hook is
        # an LLM/Compactor call), so concurrent ``emit``s re-enter it. Serialise
        # the whole flush so exactly one task owns a buffer generation.
        self._dropped = 0
        self._warned = False
        self._flush_lock = asyncio.Lock()

    async def emit(self, obs: Observation) -> None:
        if obs.kind in _ALWAYS_FORWARD:  # critical → flush the roll-up, then forward it
            await self._flush()
            await self._inner.emit(obs)
            return
        self._buf.append(obs)
        if len(self._buf) >= self._every:
            await self._flush()

    async def _flush(self) -> None:
        # DETACH the buffer before any await, under a lock. The previous order
        # (await ``summarize``, *then* ``self._buf = []``) left a window in
        # which a concurrent ``emit`` re-entered ``_flush``: the first task
        # cleared the buffer and the second read ``self._buf[-1]`` on an empty
        # list. Measured with an async summarizer, ``every=2``, 6 concurrent
        # observations: ``IndexError('list index out of range')`` raised INTO
        # the caller's ``emit`` — an observer is contractually forbidden from
        # breaking the run it observes. The same window also discarded every
        # observation appended during the await, because the post-await
        # ``self._buf = []`` threw them away unsummarised.
        async with self._flush_lock:
            buf, self._buf = self._buf, []
            if not buf:
                return
            # The invariant above is the WHOLE contract, and detaching the
            # buffer only honoured half of it: a summariser supplied by the
            # caller, or an inner sink that is down, still raised straight into
            # the run. Measured: ``summarize`` raising ``RuntimeError`` reached
            # the caller's ``emit`` verbatim, as did a failing sink. An agent
            # must not die because its telemetry did.
            #
            # ``except Exception`` deliberately, not ``BaseException``:
            # ``CancelledError`` is how a run is torn down and must keep
            # propagating.
            try:
                text = self._summarize(buf)
                if inspect.isawaitable(text):
                    text = await text
                src = buf[-1]  # carry correlation from the last buffered item
                rolled = Observation(
                    kind=self._kind,
                    render=text,
                    payload={"rolled": len(buf)},
                    run_id=src.run_id,
                    agent=src.agent,
                    parent_id=src.parent_id,
                )
                await self._inner.emit(rolled)
            except Exception as exc:  # noqa: BLE001 — see above
                # The batch is gone: a summariser that cannot summarise has
                # nothing to emit, and retaining it would re-run the same
                # failure on every subsequent flush. Counted rather than
                # silent, so "no summaries appeared" is distinguishable from
                # "nothing happened" — the same reasoning as
                # ``QueueObserver.dropped``.
                self._dropped += len(buf)
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        "RollupObserver: summarising %d observation(s) failed (%s: %s); "
                        "the batch was dropped and the run continued. Further failures "
                        "are counted on `.dropped` and not logged again.",
                        len(buf),
                        type(exc).__name__,
                        exc,
                    )

    @property
    def dropped(self) -> int:
        """Observations discarded because summarising them failed. ``0`` means
        the rollup is complete; anything else means a consumer is reading a
        partial view."""
        return self._dropped

    async def close(self) -> None:
        """Flush any buffered tail (and close the inner observer if it supports it)."""
        await self._flush()
        inner_close = getattr(self._inner, "close", None)
        if inner_close is not None:
            await inner_close()
