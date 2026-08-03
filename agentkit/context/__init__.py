"""agentkit.context — in-flight reasoning state.

The context subpackage owns the **data agents reason over**. It is the
third leg of agentkit's run-state tripod:

* ``RunContext`` (``agentkit.runtime``)  — execution wiring: identity,
  budgets, services, cooperative cancellation. The *how* a run runs.
* ``WorkingContext`` (this package)      — in-flight reasoning state.
  A unified state object with four orthogonal axes:
    - ``prefix``     — cache-stable system + grounding head
    - ``messages``   — per-turn LLM transcript tail
    - ``scratchpad`` — cross-step notes
    - ``journal``    — append-only authored history (MutationJournal)

Long-term recall lives on the agent — ``Agent.memory: MemorySource``
(see ``agentkit.memory``). The cognition queries memory and grounds
the prompt via the ``RequestBuilder`` seam.

This package is deliberately narrow: it owns the data shapes and the
pure operations on them (slice, fork, merge, freeze, diff, record,
journal-diff, token estimate). It does NOT compact, retrieve, or
call the LLM — those policies stay with their owners (``Compactor``,
``MemorySource``, ``Invoker``).

The split mirrors the KV-cache discipline: the prefix is the
cache-stable head, the messages tail is per-turn churn, the journal
is structured authored history, and they never bleed into each
other.
"""

from __future__ import annotations

from agentkit.context.context import (
    ContextDiff,
    FrozenContext,
    JournalEntryT,
    MutationJournal,
    WorkingContext,
)
from agentkit.context.prefix import PrefixContext
from agentkit.context.scope import (
    AllOf,
    AnyOf,
    ContextScope,
    LastNTurns,
    Not,
    RoleFilter,
    Since,
    Tagged,
)
from agentkit.context.tokens import ApproxTokenCounter, TiktokenCounter, TokenCounter

__all__ = [
    "AllOf",
    "AnyOf",
    "ApproxTokenCounter",
    "ContextDiff",
    "ContextScope",
    "FrozenContext",
    "JournalEntryT",
    "LastNTurns",
    "MutationJournal",
    "Not",
    "PrefixContext",
    "RoleFilter",
    "Since",
    "Tagged",
    "TiktokenCounter",
    "TokenCounter",
    "WorkingContext",
]
