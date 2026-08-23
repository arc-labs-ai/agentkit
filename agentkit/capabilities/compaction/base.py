"""Compactors — policies for bounding a transcript before it hits the model.

When a transcript exceeds a token limit, a `Compactor` is applied to reduce its size. Built-in
implementations include summarization, truncation, sliding window, and importance-based filtering.

The `Compactor` Protocol matches `RequestBuilder`/`Retriever`/`Grounder`: a noun-form-of-verb
capability that exposes a single `compact()` method. Each concrete `Compactor` handles its own
threshold check internally — calling `compact()` on a small-enough transcript is a no-op.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentkit.context.tokens import estimate_message_tokens
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message


def _approx_tokens(messages: list[Message]) -> int:
    """The default ``estimate`` every built-in compactor uses.

    Delegates to ``context.tokens.estimate_message_tokens`` instead of
    keeping its own chars/4 sum. The two used to be independent copies and
    both ignored ``tool_calls``: on a measured 80-message transcript
    carrying 324,420 chars of tool arguments (~81k tokens) this returned
    **20**, so ``TruncationCompactor(max_tokens=1000)`` kept 80/80 messages
    and the provider rejected the request with a 400. Sharing one
    implementation is also what makes a caller's ``ApproxTokenCounter``
    pre-check and the compactor's own threshold check agree — disagreeing
    estimators are how this shipped unnoticed."""
    return estimate_message_tokens(messages)


@runtime_checkable
class Compactor(Protocol):
    """The single seam every compaction policy implements.

    A `Compactor` takes a transcript and returns a (possibly smaller) transcript. Implementations
    own their own threshold check — a no-op return on a transcript that is already under budget is
    expected, and lets the caller call `compact()` unconditionally without a guard at the call site.
    """

    async def compact(self, messages: list[Message], ctx: Ctx) -> list[Message]: ...


__all__ = ["Compactor"]
