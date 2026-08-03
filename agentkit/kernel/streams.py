"""Reactive stream operators over async iterables — the dependency-free Rx-style algebra.

Reactivity is a *property* the core already has (async generators, `gather_bounded`, the bounded observer
queue). These operators make the observation/event stream **composable** — `map`/`filter`/`scan`/`merge`/
`distinct_until_changed`/`take_until` — without a reactivity dependency, keeping the surface Rx-friendly so
a `reactivex` bridge stays a pure opt-in at the edge, never a core requirement.

Everything is a plain `async def` generator over `AsyncIterator[T]`; predicates/mappers may be sync OR
async. Buffers are bounded and `merge` cancels its pumps on exit — no unbounded history, no leaked tasks.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

# Every operator is parameterised so composition preserves the
# element type. Chaining ``distinct_until_changed(map_stream(events,
# lambda e: e.kind), …)`` keeps ``.kind`` typed through the pipeline
# instead of erasing to ``Any`` at every combinator boundary.
T = TypeVar("T")
U = TypeVar("U")
S = TypeVar("S")


async def _apply(fn: Callable[..., Any], *args: Any) -> Any:
    r = fn(*args)
    return await r if inspect.isawaitable(r) else r


async def map_stream(
    src: AsyncIterator[T], fn: Callable[[T], U | Awaitable[U]]
) -> AsyncIterator[U]:
    """Transform each item (`fn` sync or async)."""
    async for item in src:
        yield await _apply(fn, item)


async def filter_stream(
    src: AsyncIterator[T], pred: Callable[[T], bool | Awaitable[bool]]
) -> AsyncIterator[T]:
    """Keep items where `pred` holds (sync or async)."""
    async for item in src:
        if await _apply(pred, item):
            yield item


async def take(src: AsyncIterator[T], n: int) -> AsyncIterator[T]:
    """Yield at most the first `n` items, then stop."""
    if n <= 0:
        return
    count = 0
    async for item in src:
        yield item
        count += 1
        if count >= n:
            return


async def take_while(
    src: AsyncIterator[T], pred: Callable[[T], bool | Awaitable[bool]]
) -> AsyncIterator[T]:
    """Yield while `pred` holds; stop at the first item that fails it (exclusive)."""
    async for item in src:
        if not await _apply(pred, item):
            return
        yield item


async def take_until(
    src: AsyncIterator[T], pred: Callable[[T], bool | Awaitable[bool]]
) -> AsyncIterator[T]:
    """Yield items, including the first that satisfies `pred`, then stop (inclusive — e.g. up to a
    `result`/`error` observation)."""
    async for item in src:
        yield item
        if await _apply(pred, item):
            return


async def distinct_until_changed(
    src: AsyncIterator[T], key: Callable[[T], Any] | None = None
) -> AsyncIterator[T]:
    """Drop *consecutive* duplicates (coalesce); `key` selects the compared value."""
    sentinel = object()
    prev: Any = sentinel
    async for item in src:
        k = await _apply(key, item) if key is not None else item
        if k != prev:
            yield item
            prev = k


async def scan(
    src: AsyncIterator[T], fn: Callable[[S, T], S | Awaitable[S]], seed: S
) -> AsyncIterator[S]:
    """Rolling accumulate: yield the accumulator after folding each item (`fn(acc, item)`, sync or async).
    The rolling-summary primitive."""
    acc: S = seed
    async for item in src:
        acc = await _apply(fn, acc, item)
        yield acc


async def buffer(src: AsyncIterator[T], n: int) -> AsyncIterator[list[T]]:
    """Batch items into lists of up to `n`; flush a partial tail when the source ends."""
    batch: list[T] = []
    async for item in src:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


async def merge(*sources: AsyncIterator[T], buffer: int = 64) -> AsyncIterator[T]:
    """Interleave several async iterables concurrently (multi-agent stream fan-in). The queue is **bounded**
    (`buffer`) so a slow consumer applies backpressure on the pumps (`put` awaits) instead of growing memory.
    On consumer exit, pumps are cancelled **and awaited** — no leaked tasks, no pending-task warnings.

    Cancellation is deadlock-free. Two distinct sentinel paths keep
    it that way:

    - Natural completion awaits the sentinel put — the consumer is
      still draining, so the put resolves in at most one ``get``
      cycle.
    - A cancelled pump uses ``put_nowait`` and tolerates
      :class:`asyncio.QueueFull`. The consumer is gone (or going);
      it will never read the sentinel, so silently dropping it is
      correct AND avoids the blocking-forever scenario that a
      naive ``await queue.put(done)`` in the ``finally`` block
      would produce on an early-break with a full queue."""
    queue: asyncio.Queue[tuple[bool, Any]] = asyncio.Queue(maxsize=max(1, buffer))
    done = object()

    async def _pump(source: AsyncIterator[T]) -> None:
        cancelled = False
        try:
            async for item in source:
                try:
                    await queue.put((True, item))
                except asyncio.CancelledError:
                    cancelled = True
                    raise
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if cancelled:
                # Consumer is gone (or going) — best-effort sentinel, never
                # block. ``put_nowait`` succeeds if there's room; otherwise
                # the sentinel is silently dropped and the consumer (which
                # is no longer reading) never needed it anyway.
                try:
                    queue.put_nowait((False, done))
                except asyncio.QueueFull:
                    pass
            else:
                # Natural completion — sentinel must reach the consumer so
                # ``remaining`` advances. The consumer is still pulling, so
                # this awaits at most one ``get`` cycle.
                await queue.put((False, done))

    tasks = [asyncio.create_task(_pump(s)) for s in sources]
    remaining = len(sources)
    try:
        while remaining:
            ok, item = await queue.get()
            if ok:
                yield item
            else:
                remaining -= 1
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(
            *tasks, return_exceptions=True
        )  # await the cancellations — no leaked pumps


async def collect(src: AsyncIterator[T]) -> list[T]:
    """Drain a stream to a list (convenience for tests / terminal consumers)."""
    return [item async for item in src]


__all__ = [
    "map_stream",
    "filter_stream",
    "take",
    "take_while",
    "take_until",
    "distinct_until_changed",
    "scan",
    "buffer",
    "merge",
    "collect",
]
