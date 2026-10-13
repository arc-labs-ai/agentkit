"""Compactors — policies for bounding a transcript before it hits the model.

See ``base`` for the ``Compactor`` Protocol and ``impls`` for the built-in concrete implementations.
"""

from __future__ import annotations

# Re-exported for back-compat: pre-split, ``compaction.py`` imported ``Message`` at
# module top level, so ``from agentkit.state.compaction import Message`` worked.
# Tests rely on this; keep the name reachable (not part of ``__all__``).
from agentkit.core.contracts.types import Message as Message
from agentkit.state.compaction.base import Compactor
from agentkit.state.compaction.impls import (
    ImportanceFilteringCompactor,
    SlidingWindowCompactor,
    SummarizationCompactor,
    TruncationCompactor,
)

__all__ = [
    "Compactor",
    "ImportanceFilteringCompactor",
    "SlidingWindowCompactor",
    "SummarizationCompactor",
    "TruncationCompactor",
]
