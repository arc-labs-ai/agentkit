"""agentkit.state.memory — the unified memory/RAG seam.

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
- ``ReadOnlyMemory`` — Decorator refusing (or dropping) writes, for a
  source that is read-only by policy rather than by backend.
"""

from agentkit.state.memory.base import MemoryItem, MemorySource, Reranker, score_sort_rerank

# ``CompositeWriteError`` is re-exported alongside the two sources that raise
# it. It was the only public name in ``memory.composite`` that was not, so the
# error a caller has to catch lived at a deeper import path than the class that
# raises it — a caller writing ``except CompositeWriteError`` after
# ``from agentkit.state.memory import CompositeMemory`` got an ImportError.
from agentkit.state.memory.composite import (
    DEDUPE_COUNT_KEY,
    DEDUPE_SOURCES_KEY,
    CompositeMemory,
    CompositeWriteError,
    DedupeMode,
    SequentialMemory,
)

# ``MemoryWriteRefused`` rides alongside ``ReadOnlyMemory`` for the same
# reason ``CompositeWriteError`` rides alongside ``CompositeMemory``: the
# error a caller has to catch must not live at a deeper import path than
# the class that raises it.
from agentkit.state.memory.decorators import (
    CachedMemory,
    CompactedMemory,
    MemoryWriteRefused,
    OnWritePolicy,
    ReadOnlyMemory,
    ScopedMemory,
)
from agentkit.state.memory.file import FileMemory
from agentkit.state.memory.grounder import as_grounder, as_grounding_source
from agentkit.state.memory.journal import JournalMemory
from agentkit.state.memory.scratchpad import ScratchpadMemory
from agentkit.state.memory.tool import ToolMemory, default_parse
from agentkit.state.memory.vector import VectorMemory

__all__ = [
    "render_grounding",
    # The two metadata keys a dedupe survivor is stamped with. Named constants
    # rather than string literals in a doc page: a consumer branching on
    # "did more than one source agree?" should not have to keep a magic string
    # in sync with this module by hand.
    "DEDUPE_COUNT_KEY",
    "DEDUPE_SOURCES_KEY",
    "CompositeWriteError",
    "DedupeMode",
    "CachedMemory",
    "CompactedMemory",
    "CompositeMemory",
    "FileMemory",
    "JournalMemory",
    "MemoryItem",
    "MemorySource",
    "MemoryWriteRefused",
    "OnWritePolicy",
    "ReadOnlyMemory",
    "Reranker",
    "ScopedMemory",
    "ScratchpadMemory",
    "SequentialMemory",
    "ToolMemory",
    "VectorMemory",
    "as_grounder",
    "as_grounding_source",
    "default_parse",
    "score_sort_rerank",
]

from agentkit.state.memory.rendering import render_grounding
