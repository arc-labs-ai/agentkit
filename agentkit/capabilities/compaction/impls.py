"""Concrete `Compactor` implementations: summarization, truncation, sliding window, importance filtering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentkit.capabilities.compaction.base import _approx_tokens
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message
from agentkit.prompts.builtin import COMPACTION_IMPORTANCE, COMPACTION_SUMMARY


@dataclass
class SummarizationCompactor:
    """Summarizes the older middle of the conversation using an LLM."""

    summarizer: Any  # an LLMPort (cheap model) — uses .chat
    model: str = ""
    max_tokens: int = 12000
    keep_recent: int = 4
    estimate: Callable[[list[Message]], int] = field(default=_approx_tokens)

    async def compact(self, messages: list[Message], ctx: Ctx) -> list[Message]:
        if self.estimate(messages) <= self.max_tokens or len(messages) <= self.keep_recent + 1:
            return messages
        system = messages[0] if messages and messages[0].role == "system" else None
        head = 1 if system else 0
        keep = self.keep_recent

        # Start the kept tail at len-keep, then walk BACK over any leading `tool` messages so a tool result
        # is never split from the assistant tool-call that produced it (orphaned tool → provider 400).
        start = max(head, len(messages) - keep) if keep else len(messages)
        while keep and start > head and messages[start].role == "tool":
            start -= 1
        recent = messages[start:]
        middle = messages[head:start]
        if not middle:
            return messages

        transcript = "\n".join(f"[{m.role}] {m.content}" for m in middle)
        with ctx.trace.span(
            "compact.summarize", "compact", **{"agentkit.compact.messages": len(middle)}
        ):
            res = await self.summarizer.chat(
                messages=[
                    Message("system", COMPACTION_SUMMARY.render()),
                    Message("user", transcript),
                ],
                model=self.model,
            )

        summary = Message("system", f"[Summary of {len(middle)} earlier turns]\n{res.content}")
        return ([system] if system else []) + [summary] + list(recent)


@dataclass
class TruncationCompactor:
    """Drops the oldest messages (excluding system prompt) to fit within token limits."""

    max_tokens: int = 12000
    keep_recent: int = 4
    estimate: Callable[[list[Message]], int] = field(default=_approx_tokens)

    async def compact(self, messages: list[Message], ctx: Ctx) -> list[Message]:
        if self.estimate(messages) <= self.max_tokens:
            return messages

        system = messages[0] if messages and messages[0].role == "system" else None
        head = 1 if system else 0

        # Keep removing from the head until we are under the limit or hit keep_recent
        current_messages = list(messages)
        while (
            len(current_messages) > (head + self.keep_recent)
            and self.estimate(current_messages) > self.max_tokens
        ):
            current_messages.pop(head)

        return current_messages


@dataclass
class SlidingWindowCompactor:
    """Strictly keeps only the system prompt and the N most recent turns."""

    keep_recent: int = 10

    async def compact(self, messages: list[Message], _ctx: Ctx) -> list[Message]:
        # Clamped, because ``keep_recent`` is a COUNT and a negative one has no
        # meaning to express. Left unclamped it doesn't degrade, it inverts:
        # ``len(messages) - keep_recent`` reaches PAST the end of the list, and
        # the walk-back below subscripts there.
        keep = max(0, self.keep_recent)
        if len(messages) <= keep + 1:
            return messages

        system = messages[0] if messages and messages[0].role == "system" else None
        head = 1 if system else 0

        start = max(head, len(messages) - keep)
        # Ensure we don't orphan tool results.
        #
        # ``keep and`` is load-bearing, not a micro-optimisation. At ``keep == 0``
        # (the honest way to say "system prompt only") ``start`` is exactly
        # ``len(messages)`` — a valid slice bound, but not a valid subscript, so
        # ``messages[start].role`` raised ``IndexError`` on every transcript of
        # two or more messages. There is also nothing to walk back FROM: no
        # message is being kept, so no tool result can be orphaned.
        # ``ImportanceFilteringCompactor`` guards its identical loop the same way.
        while keep and start > head and messages[start].role == "tool":
            start -= 1

        return ([system] if system else []) + list(messages[start:])


@dataclass
class ImportanceFilteringCompactor:
    """Uses an LLM to identify and keep only the most important turns."""

    filterer: Any  # an LLMPort
    model: str = ""
    max_tokens: int = 12000
    keep_recent: int = 2
    estimate: Callable[[list[Message]], int] = field(default=_approx_tokens)

    async def compact(self, messages: list[Message], ctx: Ctx) -> list[Message]:
        if self.estimate(messages) <= self.max_tokens:
            return messages

        system = messages[0] if messages and messages[0].role == "system" else None
        head = 1 if system else 0
        keep = self.keep_recent

        start = max(head, len(messages) - keep)
        while keep and start > head and messages[start].role == "tool":
            start -= 1
        recent = messages[start:]
        middle = messages[head:start]

        if not middle:
            return messages

        transcript = "\n".join(f"[{m.role}] {m.content}" for m in middle)
        with ctx.trace.span(
            "compact.importance", "compact", **{"agentkit.compact.messages": len(middle)}
        ):
            res = await self.filterer.chat(
                messages=[
                    Message("system", COMPACTION_IMPORTANCE.render()),
                    Message("user", transcript),
                ],
                model=self.model,
            )

        important_note = Message("system", f"[Key points from earlier context]\n{res.content}")
        return ([system] if system else []) + [important_note] + list(recent)


__all__ = [
    "ImportanceFilteringCompactor",
    "SlidingWindowCompactor",
    "SummarizationCompactor",
    "TruncationCompactor",
]
