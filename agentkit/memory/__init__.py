"""agentkit.memory — the unified memory/RAG seam.

One Protocol (``MemorySource``), one composite (``CompositeMemory``),
many backends. An agent declares ``Agent.memory: MemorySource | None``;
the cognition decides when to query and how to fold results into the
working context.

First-party sources and decorators live in this package; concrete
adapters land alongside as they're written:

- ``VectorMemory``   — vector-store backed (the canonical RAG case).
- ``FileMemory``     — file-system backed (a memory tool's read side).
- ``JournalMemory``  — wraps ``MutationJournal`` as a queryable source.
- ``ToolMemory``     — adapts any ``Tool`` into a ``MemorySource``.
- ``ScopedMemory``   — Decorator enforcing ``ctx.scope`` at the boundary.
- ``CompactedMemory``— Decorator summarising results via a ``Compactor``.
- ``CachedMemory``   — Decorator caching hot queries with a TTL.
"""

from agentkit.memory.base import (
    MemoryItem,
    MemorySource,
    Reranker,
    score_sort_rerank,
)
from agentkit.memory.composite import CompositeMemory, SequentialMemory
from agentkit.memory.decorators import CachedMemory, CompactedMemory, ScopedMemory
from agentkit.memory.file import FileMemory
from agentkit.memory.grounder import as_grounder
from agentkit.memory.journal import JournalMemory
from agentkit.memory.scratchpad import ScratchpadMemory
from agentkit.memory.tool import ToolMemory, default_parse
from agentkit.memory.vector import VectorMemory

__all__ = [
    "CachedMemory",
    "CompactedMemory",
    "CompositeMemory",
    "FileMemory",
    "JournalMemory",
    "MemoryItem",
    "MemorySource",
    "Reranker",
    "ScopedMemory",
    "ScratchpadMemory",
    "SequentialMemory",
    "ToolMemory",
    "VectorMemory",
    "as_grounder",
    "default_parse",
    "score_sort_rerank",
]
