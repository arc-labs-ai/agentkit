"""Opinionated compositions built FROM the runtime + middlewares.

A single ``Agent`` is the foundational pattern: a leaf runs its own ReAct loop over
its tools (or short-circuits to a single chat call when ``tools=None``); a coordinator
dispatches to its ``children`` according to a ``Policy``. ``Workflow`` is the typed,
developer-authored DAG counterpart — explicit control where ``Agent``/``Policy`` is
emergent.

**Orchestrator-Worker.** A central coordinator decomposes a task and dispatches it
to specialised workers, then reduces their outputs. In agentkit this is
``Agent(children=..., policy=PlanPolicy(planner=...))`` (planned, named steps) or
``Agent(children=..., policy=LedgerPolicy(assessor=...))`` (planner + ledger). The
planner's "send this sub-task to that worker" verb IS a ``Handoff`` — a typed transfer
of control with a named target and a reason — and the policy interprets it via
``route_by_handoff``. The worker's reply lands back on the shared transcript /
scratchpad, not in a parent-side return channel: coordination is via the blackboard.

**Routing.** ``SelectorPolicy`` is the seam: a ``Selector`` returns the next speaker
by name. The pre-built selectors stack three control regimes — ``route_by_handoff``
for the typed ``Handoff`` verb; ``handoff_selector`` for the marker-only legacy path;
and ``llm_selector`` for fully emergent "ask a model" routing.
"""

from agentkit.agents.agent import Agent
from agentkit.agents.control import (
    ActorBudget,
    Autonomy,
    BlockedSignal,
    BudgetExhausted,
    BudgetReducedSignal,
    CancelSignal,
    ContextUpdateSignal,
    ControlSignal,
    DataSignal,
    DoneSignal,
    EscalateSignal,
    Handoff,
    MergeInbox,
    MergeWithPeerSignal,
    PolicyVerdict,
    ProgressSignal,
    RedirectSignal,
    RunPolicy,
    SignalChannel,
    SignalEnvelope,
)
from agentkit.agents.policies import (
    LedgerPolicy,
    PlanPolicy,
    RoundRobinPolicy,
    SelectorPolicy,
)
from agentkit.agents.result import AgentResult, Suspended, WorkflowResult
from agentkit.agents.workflow import Workflow

__all__ = [
    "ActorBudget",
    "Agent",
    "AgentResult",
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
    "Handoff",
    "LedgerPolicy",
    "MergeInbox",
    "MergeWithPeerSignal",
    "PlanPolicy",
    "PolicyVerdict",
    "ProgressSignal",
    "RedirectSignal",
    "RoundRobinPolicy",
    "RunPolicy",
    "SelectorPolicy",
    "SignalChannel",
    "SignalEnvelope",
    "Suspended",
    "Workflow",
    "WorkflowResult",
]
