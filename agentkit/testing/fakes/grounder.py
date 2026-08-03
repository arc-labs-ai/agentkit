"""`FakeGrounder` — RequestBuilder test helper.

Lighter than the LLM-shaped ``FakeLLM`` (`agentkit.testing.fakes.llm.FakeLLM`) — exists for
tests that need to capture the exact arguments a capability sent (e.g., the ``(ctx, task)``
tuple a grounder received) without dragging in the streaming/usage/script machinery of a full
LLM adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeGrounder:
    """RequestBuilder test helper. Records ``(ctx, task)`` calls; returns a
    canned block (empty string allowed to test the skip path)."""

    block: str = ""
    calls: list[tuple[Any, str]] = field(default_factory=list)

    async def __call__(self, ctx: Any, task: str) -> str:
        self.calls.append((ctx, task))
        return self.block


__all__ = ["FakeGrounder"]
