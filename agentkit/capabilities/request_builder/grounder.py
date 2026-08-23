"""Grounder — the small callable seam that lets ``RequestBuilder`` pull
retrieval text without knowing where it came from."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agentkit.kernel.protocols import Ctx

Grounder = Callable[[Ctx, str], Awaitable[str]]
"""A grounder is anything async-callable as `(ctx, task) -> str`.

Why a callable and not a `MemorySource` plus k/where knobs: those knobs
are retrieval mechanics, not prompt-assembly concerns. RequestBuilder should
not know that the grounding came from a vector store at all — only
that *some* text is available for this task. The caller bakes the
retrieval policy (k, where filter, query derivation, even the choice
of source — vector index, MCP server, static fixtures) in once at
wiring time, then hands the bound callable to the RequestBuilder.

A `VectorMemory` (or any `MemorySource`) is adapted to a `Grounder`
by writing a small async wrapper that queries it and formats the
returned `MemoryItem`s:

    async def grounder(ctx, task):
        items = await mem.query(task, k=5, ctx=ctx)
        return "\\n".join(f"[{i.source}] {i.content}" for i in items)
    builder = RequestBuilder(prompt=..., grounder=grounder)

For role-specific tuning, the wrapper stays adjacent to the agent
definition without leaking into RequestBuilder's signature."""


# The `name=` value stamped on a grounding system message. It is a
# LABEL, not a matching key: `reground_every_turn=True` drops the stale
# block by rebuilding `PrefixContext.grounding` wholesale, so nothing in
# the framework ever looks this string up. It is a module constant so a
# downstream caller introspecting an assembled transcript can pick the
# grounding message out by the same name the builder wrote.
_GROUNDING_NAME = "grounding"


__all__ = ["Grounder"]
