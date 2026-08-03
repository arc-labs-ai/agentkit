"""Adapter: ``MemorySource`` → ``Grounder``.

The framework's ``RequestBuilder`` accepts a ``Grounder`` — an async
callable ``(ctx, task) -> str`` that returns grounding text injected
into the prefix on the first turn (or every turn if
``reground_every_turn``). This module bridges the two abstractions:
give it a ``MemorySource`` and it returns a Grounder closure.

The retrieval policy (``k``, ``where``, formatting) is baked into
the closure at wiring time. RequestBuilder stays oblivious to where
grounding came from — vector store, file system, tool-wrapped probe,
or a composite of all three.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem, MemorySource


def _default_format(items: list[MemoryItem]) -> str:
    """Format items as `[source] content` lines.

    Sources are bracket-prefixed so the LLM sees attribution naturally
    and can cite. Empty list returns an empty string (RequestBuilder
    treats this as "no grounding for this turn")."""
    if not items:
        return ""
    return "\n".join(f"[{i.source}] {i.content}" for i in items)


def as_grounder(
    memory: MemorySource,
    *,
    k: int = 5,
    where: dict[str, Any] | None = None,
    format: Callable[[list[MemoryItem]], str] | None = None,
) -> Callable[[Ctx, str], Awaitable[str]]:
    """Adapt a ``MemorySource`` into a ``RequestBuilder.Grounder``.

    ``k`` and ``where`` are baked into the returned closure. ``format``
    overrides the default ``[source] content`` rendering — pass a
    custom formatter when the prompt expects a specific shape (Markdown
    bullets, numbered citations, JSON, etc.).

    The query passed to the grounder is the user task — the same string
    the agent is about to answer. For "query rewriting" before retrieval
    (HyDE, query decomposition) wrap the memory itself rather than
    transforming inside the grounder.
    """
    renderer = format if format is not None else _default_format

    async def grounder(ctx: Ctx, task: str) -> str:
        items = await memory.query(task, k=k, ctx=ctx, where=where)
        return renderer(items)

    return grounder


__all__ = ["as_grounder"]
