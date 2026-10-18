"""Observers retain safe correlation metadata, not tool arguments/results."""

from dataclasses import dataclass, field

from agentkit.models import CallEvent
from agentkit.tools import ToolLifecycleEvent


@dataclass
class Lifecycle:
    events: list[dict] = field(default_factory=list)

    def call(self, event: CallEvent):
        if event.kind in ("call_started", "call_finished"):
            self.events.append(
                {
                    "kind": event.kind,
                    "call_id": event.call_id,
                    "outcome": event.fields.get("outcome") if event.kind == "call_finished" else None,
                }
            )

    def tool(self, event: ToolLifecycleEvent):
        self.events.append(
            {
                "kind": "tool_" + event.phase,
                "call_id": event.invocation.call_id,
                "invocation_id": event.invocation.host_invocation_id,
                "provider_tool_call_id": event.invocation.provider_tool_call_id,
                "code": event.code,
            }
        )

    def permission(self, invocation, decision):
        self.events.append(
            {
                "kind": "permission_decision",
                "call_id": invocation.call_id,
                "invocation_id": invocation.host_invocation_id,
                "decision": decision.value,
            }
        )
