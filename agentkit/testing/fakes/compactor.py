"""`FakeCompactor` — RequestBuilder test helper.

Replaces every transcript with a single sentinel message so a test can detect that compaction
ran. Lighter than the LLM-shaped ``FakeLLM`` — no streaming/usage machinery, just call
recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentkit.kernel.types import Message


@dataclass
class FakeCompactor:
    """RequestBuilder test helper. Replaces every transcript with a single
    sentinel message so a test can detect that compaction ran."""

    sentinel: str = "[COMPACTED]"
    called: bool = False

    async def compact(self, messages: list[Message], ctx: Any) -> list[Message]:
        del messages, ctx
        self.called = True
        return [Message("system", self.sentinel)]


__all__ = ["FakeCompactor"]
