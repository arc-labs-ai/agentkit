"""Compactors — policies for bounding a transcript before it hits the model.

See ``base`` for the ``Compactor`` Protocol and ``impls`` for the built-in concrete implementations.
"""

from __future__ import annotations

from agentkit.capabilities.compaction.base import Compactor
from agentkit.capabilities.compaction.impls import (
    ImportanceFilteringCompactor,
    SlidingWindowCompactor,
    SummarizationCompactor,
    TruncationCompactor,
)

# Re-exported for back-compat: pre-split, ``compaction.py`` imported ``Message`` at
# module top level, so ``from agentkit.capabilities.compaction import Message`` worked.
# Tests rely on this; keep the name reachable (not part of ``__all__``).
from agentkit.kernel.types import Message as Message  # noqa: F401  (back-compat re-export)

__all__ = [
    "Compactor",
    "ImportanceFilteringCompactor",
    "SlidingWindowCompactor",
    "SummarizationCompactor",
    "TruncationCompactor",
]
