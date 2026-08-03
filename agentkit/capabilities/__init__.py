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
from agentkit.capabilities.request_builder import BuiltRequest, Grounder, RequestBuilder

__all__ = [
    "BuiltRequest",
    "Checkpointer",
    "Compactor",
    "Evaluator",
    "Grounder",
    "Guardrail",
    "ImportanceFilteringCompactor",
    "OutputCoercionError",
    "RequestBuilder",
    "SchemaAdapter",
    "SlidingWindowCompactor",
    "SummarizationCompactor",
    "TruncationCompactor",
    "adapt",
]
