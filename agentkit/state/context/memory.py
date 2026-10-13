"""Memory retrieval contracts for untrusted prompt context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentkit.core.contracts.types import Scope
from agentkit.core.serialization._immutable import deep_freeze


@dataclass(frozen=True)
class MemoryContext:
    """One attributed recall result; content is always treated as untrusted input."""

    source: str
    content: str
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not self.source or not self.content:
            raise ValueError("memory context requires source and content")
        object.__setattr__(self, "metadata", deep_freeze(dict(self.metadata)))


@runtime_checkable
class MemoryProvider(Protocol):
    """Retrieve scoped evidence for prompt composition; no run or model access."""

    async def retrieve(self, query: str, *, scope: Scope, limit: int) -> Sequence[MemoryContext]: ...
