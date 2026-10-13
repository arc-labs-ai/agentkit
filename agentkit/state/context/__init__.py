"""In-flight conversation state and host policies for context and memory.

``WorkingContext`` keeps the stable prefix, transcript messages, scratchpad
and journal as separate values. ``ContextPolicy`` transforms that state
between slices or before portable replay. ``MemoryProvider`` supplies scoped,
untrusted context parts during agent prompt composition. Neither policy calls
a model; ``CallRuntime`` and ``AgentRunner`` own execution.
"""

from __future__ import annotations

from agentkit.state.context.context import ContextDiff, FrozenContext, JournalEntryT, MutationJournal, WorkingContext
from agentkit.state.context.memory import MemoryContext, MemoryProvider
from agentkit.state.context.policy import ContextPolicy, SlidingWindowContextPolicy
from agentkit.state.context.prefix import PrefixContext
from agentkit.state.context.replay import BoundedReplay, prepare_bounded_replay
from agentkit.state.context.scope import AllOf, AnyOf, ContextScope, LastNTurns, Not, RoleFilter, Since, Tagged
from agentkit.state.context.tokens import ApproxTokenCounter, TiktokenCounter, TokenCounter

__all__ = [
    "AllOf",
    "AnyOf",
    "ApproxTokenCounter",
    "ContextDiff",
    "ContextPolicy",
    "SlidingWindowContextPolicy",
    "ContextScope",
    "BoundedReplay",
    "FrozenContext",
    "JournalEntryT",
    "LastNTurns",
    "MutationJournal",
    "MemoryContext",
    "MemoryProvider",
    "Not",
    "PrefixContext",
    "RoleFilter",
    "Since",
    "Tagged",
    "TiktokenCounter",
    "TokenCounter",
    "WorkingContext",
    "prepare_bounded_replay",
]
