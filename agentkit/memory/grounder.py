"""Adapters: ``MemorySource`` → the two grounding seams.

``RequestBuilder`` accepts either an async ``(ctx, task) -> str``
(``Grounder``) or an async ``(ctx, task) -> Sequence[MemoryItem]``
(``GroundingSource``). This module bridges a ``MemorySource`` to both:

- ``as_grounder(memory)``        — queries and FLATTENS to text.
- ``as_grounding_source(memory)``— queries and hands the items over intact.

The retrieval policy (``k``, ``where``) is baked into the returned closure at
wiring time either way. RequestBuilder stays oblivious to where grounding came
from — vector store, file system, tool-wrapped probe, or a composite of all
three.

Which to reach for: ``as_grounding_source`` whenever the application has a
rule about WHICH retrieved items may be used, because the flattening in
``as_grounder`` destroys the ``source`` / ``score`` / ``metadata`` such a rule
would read — most sharply the rule that a memory a model wrote is not
evidence. ``as_grounder`` stays right when the grounding is a corpus with no
provenance worth preserving and the prompt just wants the text.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
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


def as_grounding_source(
    memory: MemorySource,
    *,
    k: int = 5,
    where: dict[str, Any] | None = None,
) -> Callable[[Ctx, str], Awaitable[Sequence[MemoryItem]]]:
    """Adapt a ``MemorySource`` into a ``RequestBuilder.GroundingSource``.

    The same closure ``as_grounder`` builds, minus the flattening step — which
    is the whole implementation and also the whole point. There is no
    ``format=`` here because rendering has moved to where it can see what it
    is rendering: ``RequestBuilder(render=...)``, downstream of
    ``RequestBuilder(admit=...)``. Keeping a formatter on this side too would
    give two places to decide the same thing, and the one on this side runs
    BEFORE admission, so a rule like "drop model-authored memories" would be
    applied to text that had already been joined.

    ``k`` and ``where`` are baked in at wiring time exactly as before, and the
    query is still the user task verbatim — for query rewriting (HyDE,
    decomposition) wrap the ``MemorySource`` rather than reaching inside this
    closure.

    Note the return annotation is the structural ``Callable[...]`` rather than
    ``GroundingSource``: ``agentkit.memory`` must not import
    ``agentkit.capabilities`` (the capabilities side imports ``MemoryItem``
    from here, and the pair would cycle). ``as_grounder`` above spells its
    return type out for the same reason.
    """

    async def grounding_source(ctx: Ctx, task: str) -> Sequence[MemoryItem]:
        return await memory.query(task, k=k, ctx=ctx, where=where)

    return grounding_source


__all__ = ["as_grounder", "as_grounding_source"]
