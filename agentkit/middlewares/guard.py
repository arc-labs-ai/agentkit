"""Input guards — `BaseMiddleware`s that inspect a request *before* it reaches the model and refuse it
(raise `Blocked`) on a policy violation. A guard lives in `on_request`, emits an event, and
short-circuits — it never needs to touch `next`, so it's a textbook `BaseMiddleware` rather than a raw
`(call, next)` function.
"""

from __future__ import annotations

import re

from agentkit.kernel.middleware import BaseMiddleware, Blocked, MiddlewareContext
from agentkit.kernel.types import Operation

# Prompt-injection / unsafe-input signatures. Override via `SecurityMiddleware(patterns=[...])`.
_DEFAULT_PATTERNS = (
    r"ignore.*previous.*instructions",  # classic instruction-override injection
    r"system.*prompt.*injection",
    r"<script.*?>.*?</script>",  # script injection
    r"\\x[0-9a-fA-F]{2}",  # hex-encoding attempts
)


class SecurityMiddleware(BaseMiddleware):
    """Blocks malicious input before it reaches the model.

    Scans the latest prompt against injection/unsafe-content patterns; on a match it emits an `error`
    observation and raises `Blocked` (refusing the call). Clean input passes through untouched. Drop it in
    the chat chain ahead of the model: `Invoker(llm=…, chat_middleware=[SecurityMiddleware(), tracing(), …])`.
    """

    def __init__(self, patterns: list[str] | None = None) -> None:
        self._patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in (patterns or _DEFAULT_PATTERNS)
        ]

    async def on_request(self, ctx: MiddlewareContext) -> None:
        if ctx.operation != Operation.MODEL_CALL:  # only model calls carry a prompt to guard
            return
        for msg in ctx.data:  # ctx.data = the messages of a model call
            content = getattr(msg, "content", "") or ""
            for rx in self._patterns:
                if rx.search(content):
                    await ctx.emit("error", "blocked: malicious input detected")
                    raise Blocked("malicious input detected", {"pattern": rx.pattern})


def security(*, patterns: list[str] | None = None) -> SecurityMiddleware:
    """Factory returning a configured ``SecurityMiddleware``.

    Matches the lowercase-factory convention every other middleware
    in this package follows (``tracing()``, ``retry()``, ``meter()``,
    etc.). Kept alongside the class so callers wanting fine-grained
    control can still pass ``SecurityMiddleware(...)`` directly."""
    return SecurityMiddleware(patterns=patterns)


__all__ = ["SecurityMiddleware", "security"]
