"""Pluggable token-count estimators for a message list.

The counter is the seam that lets ``WorkingContext.tokens()`` answer
"how big is this transcript right now?" without forcing a dependency
on ``tiktoken`` or a provider-side SDK. The default
``ApproxTokenCounter`` is the chars/4 heuristic the rest of agentkit
already uses; ``TiktokenCounter`` opt-in upgrades to provider-accurate
counts when ``tiktoken`` is installed (extras: ``arc-agentkit[fast]``).

The counter is intentionally async — a real counter might call a remote
counting endpoint (provider-side) or warm up an encoder lazily. The
in-process counters are ``async def`` purely to keep the seam uniform
(every counter await is a single ``await`` at the call site).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentkit.kernel.types import Message


@runtime_checkable
class TokenCounter(Protocol):
    """Estimates token usage of a message list.

    Pluggable so a cheap-but-fuzzy ``ApproxTokenCounter`` (chars/4) can
    be swapped for a provider-accurate ``TiktokenCounter`` without
    changing the ``WorkingContext`` API. Implementations MUST be
    pure-ish — same input, same output — so a caller can pre-check a
    budget against the same number the invoker will produce.
    """

    async def estimate(self, messages: list[Message]) -> int: ...


@dataclass(frozen=True)
class ApproxTokenCounter:
    """Default — chars/4 approximation.

    Free, no deps, ~ok within 30% of actual for English. Use
    ``TiktokenCounter`` when accuracy matters (right before a borderline
    context-limit decision). Matches the heuristic ``RequestBuilder``
    uses for its ``approx_tokens`` field, so a RequestBuilder-side
    pre-check stays consistent with the WorkingContext-side view.
    """

    chars_per_token: float = 4.0

    async def estimate(self, messages: list[Message]) -> int:
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        total_chars = sum(len(m.content or "") for m in messages)
        return int(total_chars // self.chars_per_token)


@dataclass(frozen=True)
class TiktokenCounter:
    """Provider-accurate when ``tiktoken`` is installed.

    Opt-in dep under ``arc-agentkit[fast]``. Falls back to
    ``ApproxTokenCounter`` when ``tiktoken`` is absent or fails to load
    (no import error — just less accurate). The encoding name keys
    into tiktoken's registry; ``cl100k_base`` matches GPT-4/3.5 and
    is a reasonable default for cross-provider estimates.
    """

    encoding: str = "cl100k_base"

    async def estimate(self, messages: list[Message]) -> int:
        try:
            import tiktoken  # type: ignore[import-not-found]
        except Exception:
            return await ApproxTokenCounter().estimate(messages)

        try:
            enc = tiktoken.get_encoding(self.encoding)
        except Exception:
            return await ApproxTokenCounter().estimate(messages)

        total = 0
        for m in messages:
            content = m.content or ""
            if content:
                total += len(enc.encode(content))
        return total


__all__ = ["ApproxTokenCounter", "TiktokenCounter", "TokenCounter"]
