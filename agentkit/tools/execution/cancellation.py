"""Join asynchronous tool bodies when their run is cancelled."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")


async def run_cancellable(work: Awaitable[T], ctx: Any) -> T:
    """Observe cooperative run cancellation even while a tool is awaiting I/O.

    The invocation owns its body task. Cancelling the caller, a deadline, or
    the run always cancels and joins that task before returning or raising.
    Synchronous tools still require their own cooperative cancellation checks.
    """
    body = asyncio.ensure_future(work)
    try:
        while not body.done():
            ctx.check_cancelled()
            await asyncio.wait((body,), timeout=0.025)
        ctx.check_cancelled()
        return body.result()
    finally:
        if not body.done() and (not isinstance(body, asyncio.Task) or not body.cancelling()):
            body.cancel()
        cleanup = asyncio.gather(body, return_exceptions=True)
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Server shutdown can cancel an invocation whose run-token
                # cancellation already entered its finally block. Do not send
                # that second cancellation into the body cleanup.
                interrupted = True
        if interrupted:
            raise asyncio.CancelledError
