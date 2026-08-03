"""agents.control — coordination primitives policies + leaf agents compose.

A control primitive doesn't run agents; it defines the contracts agents use to
interact: typed parent <-> child signals, per-actor budgets, Handoff verbs,
Selector callbacks, termination conditions, HITL approval gates, and run-wide
safety/autonomy policies.
"""

from agentkit.agents.control.budget import ActorBudget, BudgetExhausted
from agentkit.agents.control.channel import MergeInbox, SignalChannel
from agentkit.agents.control.gate import Autonomy, should_gate
from agentkit.agents.control.handoff import (
    HANDOFF_MARKER,
    Handoff,
    handoff_tool,
    parse_handoff,
    route_by_handoff,
)
from agentkit.agents.control.safety import PolicyVerdict, RunPolicy
from agentkit.agents.control.selector import (
    AsyncSelector,
    Selector,
    SelectorProtocol,
    SyncSelector,
)
from agentkit.agents.control.signals import (
    BlockedSignal,
    BudgetReducedSignal,
    CancelSignal,
    ContextUpdateSignal,
    ControlSignal,
    DataSignal,
    DoneSignal,
    EscalateSignal,
    MergeWithPeerSignal,
    ProgressSignal,
    RedirectSignal,
    SignalEnvelope,
)
from agentkit.agents.control.termination import (
    ExternalTermination,
    FunctionalTermination,
    FunctionCall,
    MaxMessages,
    MaxTurns,
    SourceMatch,
    Stop,
    TerminationCondition,
    TextMention,
    Timeout,
)

__all__ = [
    "ActorBudget",
    "AsyncSelector",
    "Autonomy",
    "BlockedSignal",
    "BudgetExhausted",
    "BudgetReducedSignal",
    "CancelSignal",
    "ContextUpdateSignal",
    "ControlSignal",
    "DataSignal",
    "DoneSignal",
    "EscalateSignal",
    "ExternalTermination",
    "FunctionCall",
    "FunctionalTermination",
    "HANDOFF_MARKER",
    "Handoff",
    "MaxMessages",
    "MaxTurns",
    "MergeInbox",
    "MergeWithPeerSignal",
    "PolicyVerdict",
    "ProgressSignal",
    "RedirectSignal",
    "RunPolicy",
    "Selector",
    "SelectorProtocol",
    "SignalChannel",
    "SignalEnvelope",
    "SourceMatch",
    "Stop",
    "SyncSelector",
    "TerminationCondition",
    "TextMention",
    "Timeout",
    "handoff_tool",
    "parse_handoff",
    "route_by_handoff",
    "should_gate",
]
