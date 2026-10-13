"""Bounded in-process subscriptions with optional journal-before-delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from typing import Any
from uuid import uuid4

from agentkit.core.events.ports import EventJournal
from agentkit.models.contracts.types import CallEvent

EventHandler = Callable[[CallEvent], Awaitable[None]]


class EventDeliveryError(RuntimeError):
    """Names the failing subscription without exposing event content."""


@dataclass
class _Subscription:
    id: str
    handler: EventHandler
    kinds: tuple[str, ...]
    call_id: str | None


class EventHub:
    """EventSink-compatible local hub; a durable broker implements EventPublisher.

    Handlers run in publication order, with a timeout and bounded queue. A
    handler can publish a follow-up event: it is queued after the current
    event, preventing recursive calls and lock deadlocks. External publishers
    await delivery acknowledgement. Failures are surfaced, never dropped.

    This is not distributed exactly-once delivery. Durable subscribers use
    event IDs for idempotence and a host journal/broker for restart recovery.
    """

    def __init__(
        self, *, journal: EventJournal | None = None, queue_size: int = 1024, handler_timeout_s: float = 30
    ) -> None:
        if queue_size < 1 or handler_timeout_s <= 0:
            raise ValueError("event queue and handler timeout must be positive")
        self.journal = journal
        self.handler_timeout_s = handler_timeout_s
        self._subscriptions: dict[str, _Subscription] = {}
        self._queue: asyncio.Queue[tuple[CallEvent, asyncio.Future[None]]] = asyncio.Queue(queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._errors: list[EventDeliveryError] = []
        self._closed = False
        self._publication_lock = asyncio.Lock()
        self._sequences: dict[str, int] = {}
        self._explicit_streams: set[str] = set()
        self._in_handler: ContextVar[bool] = ContextVar("agentkit_event_handler", default=False)

    def subscribe(
        self, id: str, handler: EventHandler, *, kinds: tuple[str, ...] = ("*",), call_id: str | None = None
    ) -> Callable[[], None]:
        if not id or id in self._subscriptions or self._closed:
            raise ValueError("subscription ID must be unique and the hub must be open")
        self._subscriptions[id] = _Subscription(id, handler, kinds, call_id)

        def unsubscribe() -> None:
            self._subscriptions.pop(id, None)

        return unsubscribe

    async def __call__(self, event: CallEvent) -> None:
        await self.publish(event)

    async def publish(self, event: CallEvent) -> None:
        if self._closed:
            raise RuntimeError("event hub is closed")
        source_sequenced = event.sequence != 0
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        # Observe late failures even if an external publisher is cancelled;
        # acknowledgement still raises normally and drain retains the errors.
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        nested = self._in_handler.get()
        async with self._publication_lock:
            if self._closed:
                raise RuntimeError("event hub is closed")
            if self._queue.full():
                raise EventDeliveryError("event publication queue is full")
            previous = self._sequences.get(event.call_id, 0)
            if event.sequence == 0:
                if event.call_id in self._explicit_streams:
                    raise EventDeliveryError(
                        "automatic event sequence conflicts with a source-sequenced stream; "
                        "use publish_reaction() or a distinct call_id"
                    )
                event = replace(event, sequence=previous + 1)
            elif event.sequence <= previous:
                raise EventDeliveryError("event sequence must increase within a call")
            if not event.timestamp:
                event = replace(event, timestamp=datetime.now(UTC).isoformat())
            if self.journal is not None:
                await self.journal.append(event)
            if source_sequenced:
                self._explicit_streams.add(event.call_id)
            self._sequences[event.call_id] = event.sequence
            self._queue.put_nowait((event, future))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._dispatch())
        if nested:
            # Detached descendant tasks also inherit handler context. Restart
            # an idle worker before returning, while avoiding self-ack waits.
            # Their delivery failures remain visible through drain().
            return
        await asyncio.shield(future)

    async def publish_reaction(
        self, parent: CallEvent, kind: str, fields: Mapping[str, Any] | None = None, *, reaction_id: str | None = None
    ) -> None:
        """Publish a child stream without consuming the source call's sequence.

        Hosts can provide a stable reaction ID for durable idempotency. An
        omitted ID generates a fresh stream for each publication.
        """
        await self.publish(
            CallEvent(
                kind,
                f"{parent.call_id}/reaction-{reaction_id or uuid4().hex}",
                fields or {},
                parent_call_id=parent.call_id,
            )
        )

    async def _dispatch(self) -> None:
        try:
            while not self._queue.empty():
                event, future = self._queue.get_nowait()
                errors = []
                for subscription in tuple(self._subscriptions.values()):
                    if subscription.call_id is not None and subscription.call_id != event.call_id:
                        continue
                    if not any(fnmatchcase(event.kind, pattern) for pattern in subscription.kinds):
                        continue
                    token = self._in_handler.set(True)
                    try:
                        async with asyncio.timeout(self.handler_timeout_s):
                            # A handler gets its own task so self-cancellation
                            # cannot kill the dispatch worker or strand its queue.
                            await asyncio.create_task(_invoke_handler(subscription.handler, event))
                    except asyncio.CancelledError as exc:
                        worker = asyncio.current_task()
                        if worker is not None and worker.cancelling():
                            if not future.done():
                                future.cancel()
                            self._queue.task_done()
                            raise
                        error = EventDeliveryError(
                            f"subscription {subscription.id!r} failed for {event.event_id}: {type(exc).__name__}"
                        )
                        errors.append(error)
                        self._errors.append(error)
                    except Exception as exc:
                        error = EventDeliveryError(
                            f"subscription {subscription.id!r} failed for {event.event_id}: {type(exc).__name__}"
                        )
                        errors.append(error)
                        self._errors.append(error)
                    finally:
                        self._in_handler.reset(token)
                if not future.done():
                    if errors:
                        future.set_exception(errors[0])
                    else:
                        future.set_result(None)
                self._queue.task_done()
        except BaseException:
            self._errors.append(EventDeliveryError("event dispatch worker stopped before delivery completed"))
            while not self._queue.empty():
                _, pending = self._queue.get_nowait()
                if not pending.done():
                    pending.cancel()
                self._queue.task_done()
            raise

    async def drain(self) -> None:
        await self._queue.join()
        if self._errors:
            errors, self._errors = self._errors, []
            raise ExceptionGroup("event delivery failures", errors)

    async def aclose(self) -> None:
        self._closed = True
        await self.drain()
        if self._worker:
            await self._worker

    async def __aenter__(self) -> EventHub:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


async def _invoke_handler(handler: EventHandler, event: CallEvent) -> None:
    await handler(event)
