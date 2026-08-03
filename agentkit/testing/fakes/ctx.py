"""``FakeCtx`` — minimal Ctx that RECORDS spans for assertion.

Distinct from ``agentkit.runtime.NullCtx``: ``NullCtx`` absorbs operations and records
nothing; ``FakeCtx`` records spans so tests can assert on what attributes were set.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any

from agentkit.kernel.types import Scope


class _FakeSpan:
    """Records what was set on it. Drop-in for the production span surface."""

    def __init__(self, name: str, kind: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self.kind = kind
        self.attrs: dict[str, Any] = dict(attrs)
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    async def __aenter__(self) -> _FakeSpan:
        return self

    async def __aexit__(self, *_a: object) -> None:
        return None

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, **fields: Any) -> None:
        self.events.append((name, fields))


class _FakeTracer:
    """Records every ``span()`` call. Tests assert via ``tracer.spans``.

    Also surfaced as the ``spans`` attribute on ``FakeCtx`` itself
    (``ctx.spans``) so existing tests that fold tracer + ctx into one
    object keep working."""

    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    @contextmanager
    def span(self, name: str, kind: str = "", **attrs: Any) -> Any:
        s = _FakeSpan(name, kind, attrs)
        self.spans.append(s)
        yield s


class FakeCtx:
    """Minimal Ctx that RECORDS spans + observations.

    Use when a test needs to assert on what spans were opened (e.g.,
    RequestBuilder stamps ``agentkit.prompt.version`` — a test for that
    contract needs a tracer that captures attributes). Distinct from
    ``agentkit.runtime.NullCtx``: ``NullCtx`` absorbs operations and
    records nothing; ``FakeCtx`` records spans for assertion. Both are
    valid — pick based on whether you need to assert on what was
    recorded.

    ``tracer.spans`` is also exposed as ``ctx.spans`` for tests that
    fold the tracer onto the ctx itself. Either access form works.
    """

    def __init__(self, *, scope: Scope | None = None) -> None:
        self.correlation_id = "fake-run"
        self.scope = scope if scope is not None else Scope()
        self.autonomy = "manual"
        self.trace = _FakeTracer()
        self.invoker: Any = None
        self.store: Any = None
        self.checkpointer: Any = None

    @property
    def spans(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Convenience alias mirroring ``(name, kind, attrs)`` tuples on
        the ctx itself, so tests can assert on ``ctx.spans`` directly
        instead of reaching through ``ctx.trace.spans``."""
        return [(s.name, s.kind, s.attrs) for s in self.trace.spans]

    def check_cancelled(self) -> None:
        return None

    def semaphore(self) -> Any:
        @asynccontextmanager
        async def _grant() -> Any:
            yield self

        return _grant()

    def child(self) -> FakeCtx:
        return self

    async def emit(
        self,
        kind: str,
        render: str = "",
        *,
        payload: Any = None,
        agent: str = "",
        parent_id: Any = None,
    ) -> None:
        del kind, render, payload, agent, parent_id
        return None


__all__ = ["FakeCtx"]
