"""Shared, attributed rendering for retrieved memory records."""

from collections.abc import Sequence

from agentkit.state.memory.base import MemoryItem


def render_grounding(items: Sequence[MemoryItem]) -> str:
    """Render records as attributed evidence; callers decide admission first."""
    return "\n".join(f"[{item.source}] {item.content}" for item in items)
