"""RequestBuilder capability — the single seam where the four engineering disciplines meet.

See ``base`` for the ``RequestBuilder`` / ``BuiltRequest`` / ``BudgetCheck`` surface and
``grounder`` for the two grounding seams — the flattened ``Grounder`` and the
typed ``GroundingSource`` that keeps a retrieved item's provenance alive as far
as the prompt.
"""

from __future__ import annotations

from agentkit.capabilities.request_builder.base import (
    BudgetCheck,
    BuiltRequest,
    RequestBuilder,
)
from agentkit.capabilities.request_builder.grounder import (
    GROUNDING_RECORD_KEY,
    Grounder,
    GroundingAdmit,
    GroundingRender,
    GroundingSource,
    render_grounding,
)

__all__ = [
    "GROUNDING_RECORD_KEY",
    "BudgetCheck",
    "BuiltRequest",
    "Grounder",
    "GroundingAdmit",
    "GroundingRender",
    "GroundingSource",
    "RequestBuilder",
    "render_grounding",
]
