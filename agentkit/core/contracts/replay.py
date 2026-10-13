"""ReplayStore — side-channel storage for full LLM call inputs/outputs.

Span attributes have a cardinality budget (OTel SDKs cap at 32-128
attributes per span). High-cardinality payloads — full prompt
messages, tool arguments, response bodies — live HERE instead.
Keyed by ``span_id`` so the trace is the index, the store is the
bulk.

The default impl is a no-op so the kernel works without storage.
Adapters (Langfuse, Phoenix, S3, disk) implement this Protocol and
plug it in via Services. The chat middleware writes; downstream
tooling reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ReplayRecord:
    """One LLM-call replay payload. ``span_id`` ties it to the trace so the
    trace acts as the index and the store carries the bulk.

    ``operation`` is the unit of work — ``"chat"`` for an LLM turn,
    ``"execute_tool"`` for a tool execution — matching the OTel GenAI
    operation taxonomy span names. ``request`` is the input shape
    (messages, tools, response_format for chat; arguments for tools).
    ``response`` is the assembled ``LLMResult`` or tool output; ``None``
    when the call is still in flight or errored before producing one.
    """

    span_id: str
    operation: str
    request: Any
    response: Any | None = None


@runtime_checkable
class ReplayStore(Protocol):
    """Optional side-channel store for full LLM call replay.

    Implementations are expected to be best-effort: a failed write
    must NOT break the run. Adapters wrap their writes in suppress
    blocks at the call site (see ``middlewares/tracing.py``). The
    default ``NoopReplayStore`` drops writes silently — the canonical
    zero-config path.
    """

    async def put(self, record: ReplayRecord) -> None: ...
    async def get(self, span_id: str) -> ReplayRecord | None: ...


class NoopReplayStore:
    """Default replay store: drops writes, reads return None.

    Structurally satisfies ``ReplayStore`` so a ``Services()`` with no
    args is fully usable. Production wires a concrete adapter
    (Langfuse / Phoenix / S3 / disk)."""

    async def put(self, record: ReplayRecord) -> None:
        return None

    async def get(self, span_id: str) -> ReplayRecord | None:
        return None


__all__ = [
    "ReplayRecord",
    "ReplayStore",
    "NoopReplayStore",
]
