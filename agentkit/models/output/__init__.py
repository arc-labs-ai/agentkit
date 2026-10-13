"""Output-schema adapter layer — uniform bridge between "user declared this
output shape on their Agent" and "framework can ship the matching JSON Schema to
the LLM and coerce the response back into a typed instance."

Public surface:

    * :class:`SchemaAdapter`        — the Protocol every flavour satisfies.
    * :class:`OutputCoercionError`  — the single exception type the retry
                                       middleware catches across all flavours.
    * :func:`adapt`                 — dispatcher: takes any supported spec and
                                       returns the right adapter.

The individual adapter classes (:class:`PydanticAdapter`,
:class:`DataclassAdapter`, :class:`AttrsAdapter`, :class:`JsonSchemaAdapter`)
are re-exported for type-hinting and direct construction; in normal use the
caller goes through ``adapt()`` and never names a concrete adapter.

Optional dependencies (pydantic, attrs, jsonschema, msgspec) are NEVER imported
at module load — every probe is guarded so this subpackage stays importable in
a minimal environment. The corresponding flavour simply becomes unavailable
until its dep is installed."""

from __future__ import annotations

from agentkit.models.output.adapt import adapt
from agentkit.models.output.attrs_adapter import AttrsAdapter
from agentkit.models.output.dataclass_adapter import DataclassAdapter
from agentkit.models.output.json_schema_adapter import JsonSchemaAdapter
from agentkit.models.output.protocol import OutputCoercionError, SchemaAdapter
from agentkit.models.output.pydantic_adapter import PydanticAdapter

__all__ = [
    "AttrsAdapter",
    "DataclassAdapter",
    "JsonSchemaAdapter",
    "OutputCoercionError",
    "PydanticAdapter",
    "SchemaAdapter",
    "adapt",
]
