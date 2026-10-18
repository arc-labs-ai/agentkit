"""Resource permission contribution; AgentKit owns decision enforcement."""

from agentkit.core import Scope
from agentkit.tools import PermissionDecision, ToolInvocationRequest


def record_permission(call_id: str, scope: Scope, granted_ids: frozenset[str], lifecycle):
    def permission(request: ToolInvocationRequest) -> PermissionDecision:
        allowed = (
            request.name == "lookup_record"
            and request.invocation.call_id == call_id
            and request.scope == scope
            and request.arguments.get("record_id") in granted_ids
        )
        decision = PermissionDecision.ALLOW if allowed else PermissionDecision.DENY
        lifecycle.permission(request.invocation, decision)
        return decision

    return permission
