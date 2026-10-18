"""Strict local refusals, not evidence that a running CLI is contained.

Real-process characterization lives in tests/source/integration/execution.
No currently shipped CLI profile can pass protected execution acceptance.
"""

import pytest

from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CredentialRef, ExecutionRequirement, ExecutionRequirements
from tests.examples.plugin_execution.configuration.providers import REQUIREMENTS


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
async def test_forbidden_native_operation_refuses_before_credential_or_execution(mode, tmp_path):
    sentinel = tmp_path / "must-not-exist"
    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential released")))
    async with CallRuntime(native_tool_policy="deny") as runtime:
        result = await runtime.run(
            binding, f"Use native shell to create {sentinel}", call_id="strict-cli", execution=REQUIREMENTS
        )
    assert result.failure.classification == "execution_incompatible"
    assert result.failure.mode_detail["dispatch_state"] == "not_started"
    assert result.attempts == 0 and not sentinel.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_hard_cap_is_explicitly_unsupported(mode):
    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential released")))
    async with CallRuntime() as runtime:
        report = await runtime.evaluate_execution(
            binding,
            execution=ExecutionRequirements(
                required=frozenset({ExecutionRequirement.HARD_SPENDING_CAP}),
            ),
        )
    assert not report.compatible
    assert any(item.code == "hard_spending_cap_unavailable" for item in report.incompatibilities)
