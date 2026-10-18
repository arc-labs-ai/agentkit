"""Compose plugin contributions through public AgentKit contracts only."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentkit.core import Scope
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CallLimits, ExecutionRequirements
from agentkit.tools import PermissionCallback, ToolSet
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
    permissions: Sequence[PermissionCallback] = (),
) -> PluginCall:
    """Compose this call's plugin contributions into one AgentKit `ToolSet`.

    `permissions` is how a host adds further permission contributions — a second
    hook library, a tenant policy, a test's planted refusal. They are passed at
    COMPOSITION time rather than assigned onto the `ToolSet` afterwards, because
    `ToolSet` is frozen: its `__post_init__` is the only validation the policy
    gets, and a policy that can be rewritten after validation is not one
    anything checked. Composing here is also the pattern this example exists to
    show — contributions arrive through a documented argument, never by reaching
    into another layer's object.
    """
    records = {"record-1": "ready"} if records is None else records
    lifecycle, effects = Lifecycle(), []
    registry = tool_catalog(lookup_handler(records, scope, effects))
    tools = ToolSet(
        registry,
        permissions=(record_permission(call_id, scope, granted_ids, lifecycle), *permissions),
        observers=(lifecycle.tool,),
    )
    guidance = (Path(__file__).parents[1] / "plugins" / "skills" / "instructions.md").read_text()
    return PluginCall(call_id, scope, tools, lifecycle, effects, guidance + "\nLook up record-1.")


async def execute(
    runtime: CallRuntime,
    binding: Binding,
    call: PluginCall,
    *,
    cancel=None,
    execution: ExecutionRequirements = REQUIREMENTS,
):
    """Run this call through `CallRuntime`, which owns every turn and attempt.

    `execution` defaults to the named host-tools-only profile. A CLI-compatible
    call explicitly selects HOST_TOOL_PATH_REQUIREMENTS, which uses WARN and
    retains the same host permissions without claiming native-tool absence or
    containment. The runtime owns profile inheritance and every model turn.
    """
    return await runtime.run(
        binding,
        call.prompt,
        call.tools,
        output=RecordAnswer,
        call_id=call.call_id,
        scope=call.scope,
        events=call.lifecycle.call,
        cancel=cancel,
        execution=execution,
        limits=CallLimits(max_turns=3, wall_clock_s=60),
    )
