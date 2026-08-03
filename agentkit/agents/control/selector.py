"""Selector — the who-speaks-next contract for ``SelectorPolicy``.

A Selector picks the next agent to take a turn given the transcript so
far + the coordinator's roster. Returning the agent's ``name`` (or
``None`` to fall back to round-robin in ``SelectorPolicy``'s default
behavior) is the contract. Both sync and async forms are supported —
policies that need to ask a model "who's best next" use the async form
(see ``llm_selector`` in ``agents/policies/selector_policy.py``).

This module defines the type alias + the Protocol. Concrete selectors
(handoff-by-marker, LLM-driven, round-robin-by-default) live in
``agents/policies/selector_policy.py`` and ``agents/control/handoff.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from agentkit.kernel.types import Message

# Sync form. The `Sequence[Any]` for agents is intentional — the coordinator
# carries duck-typed agents (anything with `.name` + `.run(...)`); the
# selector only reads `.name`. PEP 695 `type` statements (Python 3.12+)
# are lazy and play well with `from __future__ import annotations`.
type SyncSelector = Callable[[Sequence[Message], Sequence[Any]], str | None]

# Async form. Same shape, returns an awaitable.
type AsyncSelector = Callable[[Sequence[Message], Sequence[Any]], Awaitable[str | None]]

# The union — `SelectorPolicy.selector` accepts either.
type Selector = SyncSelector | AsyncSelector


@runtime_checkable
class SelectorProtocol(Protocol):
    """The Protocol form, for callers that want a class-based selector
    (state, configuration, etc.) instead of a function. Both `__call__`
    forms are accepted — sync return `str | None`, async return
    `Awaitable[str | None]`."""

    def __call__(
        self,
        transcript: Sequence[Message],
        agents: Sequence[Any],
    ) -> str | None | Awaitable[str | None]: ...


__all__ = [
    "AsyncSelector",
    "Selector",
    "SelectorProtocol",
    "SyncSelector",
]
