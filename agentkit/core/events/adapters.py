"""Event-sink adapters for masking and OpenTelemetry trace export."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from agentkit.models.contracts.constants import EventKind
from agentkit.models.contracts.types import CallEvent, EventSink


class MaskingSink:
    """Apply a host masker and bound every nested event field before delivery."""

    def __init__(
        self,
        inner: EventSink,
        masker: Callable[[CallEvent], CallEvent] | None = None,
        *,
        max_field_bytes: int = 4096,
    ) -> None:
        if max_field_bytes < 1:
            raise ValueError("max_field_bytes must be positive")
        self.inner = inner
        self.masker = masker
        self.max_field_bytes = max_field_bytes

    def _bound(self, value: Any) -> Any:
        if isinstance(value, str):
            raw = value.encode("utf-8")
            if len(raw) > self.max_field_bytes:
                return {"truncated": True, "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            return value
        if isinstance(value, bytes):
            if len(value) > self.max_field_bytes:
                return {"truncated": True, "length": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            return value
        if isinstance(value, Mapping):
            return {str(key): self._bound(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(self._bound(item) for item in value)
        return value

    async def __call__(self, event: CallEvent) -> None:
        masked = self.masker(event) if self.masker is not None else event
        bounded = CallEvent(
            masked.kind,
            masked.call_id,
            self._bound(masked.fields),
            sequence=masked.sequence,
            timestamp=masked.timestamp,
            parent_call_id=masked.parent_call_id,
            schema_version=masked.schema_version,
        )
        result = self.inner(bounded)
        if inspect.isawaitable(result):
            await result


class TraceEventSink:
    """GenAI spans through the existing TracePort, without a required OTel SDK."""

    def __init__(self, trace: Any) -> None:
        self.trace = trace
        self._spans: dict[str, tuple[Any, Any]] = {}

    async def __call__(self, event: CallEvent) -> None:
        fields = event.fields
        if event.kind == EventKind.CALL_STARTED:
            manager = self.trace.span(
                "invoke_agent",
                kind="agent",
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.provider.name": fields.get("mode"),
                    "gen_ai.request.model": fields.get("model"),
                    "agentkit.call.id": event.call_id,
                },
            )
            self._spans[event.call_id] = (manager, manager.__enter__())
        entry = self._spans.get(event.call_id)
        if entry is None:
            return
        manager, span = entry
        if event.kind == EventKind.USAGE:
            usage = fields.get("usage") or {}
            span.set(
                "gen_ai.usage.input_tokens",
                sum(
                    usage.get(k, 0)
                    for k in ("input_uncached", "input_cache_read", "input_cache_write_5m", "input_cache_write_1h")
                ),
            )
            span.set("gen_ai.usage.output_tokens", usage.get("output", 0))
        span.add_event(str(event.kind), **dict(fields))
        if event.kind == EventKind.CALL_FINISHED:
            span.set("agentkit.cost.usd", fields.get("cost_usd"))
            span.set("agentkit.cost.source", fields.get("cost_source"))
            manager.__exit__(None, None, None)
            del self._spans[event.call_id]
