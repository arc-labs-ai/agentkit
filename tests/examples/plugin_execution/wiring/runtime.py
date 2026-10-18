"""Compose plugin contributions through public AgentKit contracts only."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agentkit.core import Scope
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CallLimits
from agentkit.tools import ToolSet
from tests.examples.plugin_execution.configuration.providers import REQUIREMENTS
from tests.examples.plugin_execution.plugins.hooks.lifecycle import Lifecycle
from tests.examples.plugin_execution.plugins.hooks.permissions import record_permission
from tests.examples.plugin_execution.plugins.tools.catalog import RecordAnswer, tool_catalog
from tests.examples.plugin_execution.plugins.tools.handlers import lookup_handler


@dataclass
class PluginCall:
    call_id: str
    scope: Scope
    tools: ToolSet
    lifecycle: Lifecycle
    effects: list[str]
    prompt: str


def compose_call(
    call_id: str,
    *,
    records: Mapping[str, str] | None = None,
    granted_ids: frozenset[str] = frozenset({"record-1"}),
    scope: Scope = Scope(org_id=1),
) -> PluginCall:
    records = {"record-1": "ready"} if records is None else records
    lifecycle, effects = Lifecycle(), []
    registry = tool_catalog(lookup_handler(records, scope, effects))
    tools = ToolSet(
        registry, permissions=(record_permission(call_id, scope, granted_ids, lifecycle),), observers=(lifecycle.tool,)
    )
    guidance = (Path(__file__).parents[1] / "plugins" / "skills" / "instructions.md").read_text()
    return PluginCall(call_id, scope, tools, lifecycle, effects, guidance + "\nLook up record-1.")


async def execute(runtime: CallRuntime, binding: Binding, call: PluginCall, *, cancel=None):
    return await runtime.run(
        binding,
        call.prompt,
        call.tools,
        output=RecordAnswer,
        call_id=call.call_id,
        scope=call.scope,
        events=call.lifecycle.call,
        cancel=cancel,
        execution=REQUIREMENTS,
        limits=CallLimits(max_turns=3, wall_clock_s=60),
    )
