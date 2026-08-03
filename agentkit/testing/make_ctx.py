"""``make_test_ctx`` — canonical factory for a real ``RunContext`` in tests.

Replaces the per-file ``_ctx(llm)`` factories scattered across the test suite. One factory,
all the knobs as kwargs, defensive defaults so the no-arg form is usable for trivial tests.

This is NOT a fake — it builds a real ``RunContext`` wired with real ``Services`` /
``Invoker``. Test doubles (``FakeCtx``, ``FakeLLM``, etc.) live under ``agentkit.testing.fakes``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agentkit.kernel.concurrency import CancellationToken
from agentkit.kernel.protocols import AutonomyLiteral
from agentkit.kernel.types import Scope
from agentkit.runtime import Budget, Invoker, RunContext, Services


def make_test_ctx(
    *,
    llm: Any = None,
    invoker: Any = None,
    scope: Scope | None = None,
    budget: Budget | None = None,
    store: Any = None,
    vector: Any = None,
    checkpointer: Any = None,
    observer: Any = None,
    trace: Any = None,
    cancel: CancellationToken | None = None,
    chat_middleware: Sequence[Any] = (),
    tool_middleware: Sequence[Any] = (),
    meters: Sequence[Any] = (),
    correlation_id: str = "test-run",
    autonomy: AutonomyLiteral = "auto",
) -> RunContext:
    """Build a ``RunContext`` for tests with the knobs the call site cares
    about and sensible no-op defaults for everything else.

    Two ways to wire the LLM seam: pass ``invoker=`` (a fully-built
    ``Invoker``) OR pass ``llm=`` and we'll wrap it with
    ``Invoker(llm=..., chat_middleware=chat_middleware)``. Tests that
    don't need an Invoker at all leave both unset — they get a
    ``RunContext`` with ``services.invoker = None``, which is fine for
    capabilities tests (RequestBuilder, Compactor) that never invoke an LLM.

    ``Scope()`` is the zero-tenant default. Override with
    ``scope=Scope(org_id, domain_id)`` for tenant-isolation tests.

    ``observer`` / ``trace`` / ``checkpointer`` / ``store`` / ``vector``
    default to ``None`` here, which we *omit* when constructing
    ``Services`` so its built-in ``NoopObserver`` / ``NoopTrace``
    factories still kick in — passing ``None`` explicitly would clobber
    them and break code that calls ``ctx.trace.span(...)`` or
    ``ctx.observer.emit(...)``.
    """
    if invoker is None and llm is not None:
        invoker = Invoker(
            llm=llm,
            chat_middleware=list(chat_middleware),
            tool_middleware=list(tool_middleware),
        )

    # Only forward observer/trace if explicitly given — otherwise let
    # Services' default_factory install the no-op implementations.
    extra: dict[str, Any] = {}
    if observer is not None:
        extra["observer"] = observer
    if trace is not None:
        extra["trace"] = trace

    services = Services(
        invoker=invoker,
        store=store,
        vector=vector,
        checkpointer=checkpointer,
        **extra,
    )
    return RunContext(
        correlation_id=correlation_id,
        scope=scope if scope is not None else Scope(),
        budget=budget if budget is not None else Budget(),
        services=services,
        cancel=cancel,
        autonomy=autonomy,
        meters=list(meters),
    )


__all__ = ["make_test_ctx"]
