"""Optional collaborators over the seams."""

from agentkit.capabilities.checkpointer import Checkpointer
from agentkit.capabilities.compaction import (
    Compactor,
    ImportanceFilteringCompactor,
    SlidingWindowCompactor,
    SummarizationCompactor,
    TruncationCompactor,
)
from agentkit.capabilities.eval import Evaluator
from agentkit.capabilities.guardrails import Guardrail
from agentkit.capabilities.output_schema import OutputCoercionError, SchemaAdapter, adapt
from agentkit.capabilities.request_builder import (
    GROUNDING_RECORD_KEY,
    BuiltRequest,
    Grounder,
    GroundingAdmit,
    GroundingRender,
    GroundingSource,
    RequestBuilder,
    render_grounding,
)

__all__ = [
    "GROUNDING_RECORD_KEY",
    "BuiltRequest",
    "Checkpointer",
    "Compactor",
    "Evaluator",
    "Grounder",
    "GroundingAdmit",
    "GroundingRender",
    "GroundingSource",
    "Guardrail",
    "ImportanceFilteringCompactor",
    "OutputCoercionError",
    "RequestBuilder",
    "SchemaAdapter",
    "SlidingWindowCompactor",
    "SummarizationCompactor",
    "TruncationCompactor",
    "adapt",
    "render_grounding",
]
