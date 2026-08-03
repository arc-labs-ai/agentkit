"""``ToolMemory`` — adapt any ``Tool`` into a ``MemorySource``.

A tool (``FunctionTool`` or anything matching the Tool Protocol — i.e.
has a ``name`` and an async ``run(args, ctx)``) is, semantically, a
knowledge probe. Web search, internal docs lookup, a SQL-backed
"find_customer" function — each is a way for the agent to *reach for*
information. ``ToolMemory`` exposes that probe through the unified
``MemorySource`` seam so the cognition can include tool-fetched
context next to vector/file/journal hits in a single ``CompositeMemory``.

Discipline:
- Read-only by adapter contract. ``write`` is a no-op — tools that mutate
  the world stay tools (the cognition calls them mid-loop), not memory
  sources. A tool can be both registered as a ``FunctionTool`` (for
  LLM-decided calls) AND wrapped as a ``ToolMemory`` (for cognition-
  decided pre-fetch); the same underlying callable serves both.
- The default parser is best-effort and handles the four common return
  shapes (``list[dict]``, ``list[str]``, single string, dict with a
  ``"results"`` key). For anything richer, pass ``result_to_items=`` —
  the adapter never tries to be clever about an unknown shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem


def default_parse(result: Any, query: str) -> list[MemoryItem]:
    """Best-effort parser for tool-call return values.

    Handles the four shapes a search-ish tool commonly returns:

    - ``dict`` with a ``"results"`` key → recurse on the ``results`` list
    - ``list[dict]`` → each dict is treated as ``{content, score?, metadata?}``
      (anything else in the dict becomes metadata; ``"text"`` and ``"snippet"``
      are accepted as content aliases for convenience)
    - ``list[str]`` → each string becomes a ``MemoryItem.content``
    - ``str`` → split on blank lines (``"\\n\\n"``) into multiple items;
      a single-paragraph string yields one item

    Anything else is wrapped as a single item with ``content=str(result)`` —
    enough to be visible in traces, not enough to be useful: callers with
    richer return shapes should pass ``result_to_items=``.

    The ``query`` argument is accepted for parity with custom parsers
    (which may want to highlight matches) but the default parser ignores it.
    """
    del query  # the default parser doesn't use the query — kept for parser-API parity

    # dict-with-"results" — unwrap and recurse
    if isinstance(result, dict):
        if "results" in result and isinstance(result["results"], list):
            return default_parse(result["results"], query="")
        # bare dict with a content-shaped payload
        return [_item_from_dict(result)]

    if isinstance(result, list):
        items: list[MemoryItem] = []
        for entry in result:
            if isinstance(entry, dict):
                items.append(_item_from_dict(entry))
            elif isinstance(entry, str):
                items.append(MemoryItem(content=entry, source="tool"))
            else:
                items.append(MemoryItem(content=str(entry), source="tool"))
        return items

    if isinstance(result, str):
        # split on blank lines; preserves a single-paragraph string as one item
        parts = [p.strip() for p in result.split("\n\n") if p.strip()]
        if not parts:
            return []
        return [MemoryItem(content=p, source="tool") for p in parts]

    # unknown shape — be visible, don't be clever
    return [MemoryItem(content=str(result), source="tool")]


def _item_from_dict(entry: dict[str, Any]) -> MemoryItem:
    """Adapt a single result dict to a ``MemoryItem``. Accepts ``content`` /
    ``text`` / ``snippet`` as content aliases; ``score`` is lifted out; everything
    else is preserved as metadata."""
    data = dict(entry)
    content = data.pop("content", None)
    if content is None:
        content = data.pop("text", None)
    if content is None:
        content = data.pop("snippet", None)
    if content is None:
        # No recognized content key — stringify the whole dict so it's at
        # least visible.
        content = str(entry)
    score = data.pop("score", None)
    metadata = data.pop("metadata", None)
    if metadata is None:
        # remaining keys become metadata (so a result like
        # {"content": "...", "url": "..."} keeps the url)
        metadata = data
    return MemoryItem(
        content=str(content),
        source="tool",
        score=float(score) if isinstance(score, (int, float)) else None,
        metadata=dict(metadata) if metadata else {},
    )


@dataclass(slots=True)
class ToolMemory:
    """Adapter wrapping a ``FunctionTool`` (or any Tool-Protocol object)
    as a ``MemorySource``. The tool becomes a knowledge probe the
    cognition can query alongside vector/file/journal sources.

    The tool is invoked with a dict of args built from
    ``{query_arg: query, "limit": k, **(where or {})}``. ``where`` keys
    are passed through unmodified so a tool with a typed parameter like
    ``site:`` can use them directly.

    Parsing the tool's return into ``MemoryItem``s defaults to
    :func:`default_parse` and can be overridden per-instance with
    ``result_to_items=``. Every emitted item has its ``source`` stamped
    with this instance's ``name`` so downstream consumers can attribute
    correctly.

    Writes are no-ops. A wrapped tool is read-only as memory; the same
    underlying callable can be registered as a ``FunctionTool`` for
    LLM-decided side effects without conflict.
    """

    tool: Any  # FunctionTool or Tool-Protocol-compliant (has .name + async .run)
    query_arg: str = "query"
    result_to_items: Callable[[Any, str], list[MemoryItem]] | None = None
    name: str = field(default="tool")

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        args: dict[str, Any] = {self.query_arg: query, "limit": k}
        if where:
            # ``where`` keys merge in last so a caller-provided ``limit``
            # overrides the default ``k``, which is the intuitive precedence.
            args.update(where)
        result = await self.tool.run(args, ctx)

        parser = self.result_to_items if self.result_to_items is not None else default_parse
        items = list(parser(result, query))

        # Stamp ``source`` on every item so attribution is consistent even
        # when a custom parser forgot to set it.
        stamped = [
            MemoryItem(
                content=item.content,
                source=self.name,
                score=item.score,
                metadata=item.metadata,
            )
            for item in items
        ]
        return stamped[:k]

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        """No-op. Tools are query-only as memory; a side-effecting tool
        belongs in the ``ToolRegistry``, not as a memory write sink."""
        del items, ctx
        return None


__all__ = ["ToolMemory", "default_parse"]
