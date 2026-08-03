"""Compactors — policies for bounding a transcript before it hits the model.

When a transcript exceeds a token limit, a `Compactor` is applied to reduce its size. Built-in
implementations include summarization, truncation, sliding window, and importance-based filtering.

The `Compactor` Protocol matches `RequestBuilder`/`Retriever`/`Grounder`: a noun-form-of-verb
capability that exposes a single `compact()` method. Each concrete `Compactor` handles its own
threshold check internally — calling `compact()` on a small-enough transcript is a no-op.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message


def _approx_tokens(messages: list[Message]) -> int:
    return sum(len(m.content or "") for m in messages) // 4


@runtime_checkable
class Compactor(Protocol):
    """The single seam every compaction policy implements.

    A `Compactor` takes a transcript and returns a (possibly smaller) transcript. Implementations
    own their own threshold check — a no-op return on a transcript that is already under budget is
    expected, and lets the caller call `compact()` unconditionally without a guard at the call site.
    """

    async def compact(self, messages: list[Message], ctx: Ctx) -> list[Message]: ...


__all__ = ["Compactor"]
