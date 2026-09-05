"""Static and runtime checks for AgentKit's public context boundary."""

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Scope
from agentkit.runtime.context import RunContext


def accepts_ctx(ctx: Ctx) -> Ctx:
    """Make mypy verify the concrete runtime context structurally."""
    return ctx


def test_run_context_satisfies_the_public_context_protocol() -> None:
    ctx = RunContext(correlation_id="run-1", scope=Scope())

    assert accepts_ctx(ctx) is ctx
