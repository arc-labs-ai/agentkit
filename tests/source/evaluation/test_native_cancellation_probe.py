"""The native probe cancels real tool execution without assuming partial CLI text."""

import pytest

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.types import Binding, CallLimits, Turn
from tests.evals.suites.conformance import CASES, execute
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["codex", "claude_code"])
async def test_native_probe_requires_diagnostic_entry_and_actual_cleanup(mode, tmp_path):
    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "sampling-1", "name": "diagnostic_sampling", "arguments": {}},),
            raw_usage={"input_tokens": 10, "output_tokens": 2},
        )
    )
    case = next(case for case in CASES if case.id == "cancellation")
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer, checks = await execute(
            case,
            runtime,
            Binding(mode, "test"),
            CallLimits(wall_clock_s=2),
            lambda event: None,
            tmp_path,
            "native-probe",
        )
    assert answer.outcome == "cancelled" and checks == []
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_native_probe_rejects_textual_answer_without_diagnostic_execution(tmp_path):
    adapter = ScriptedAdapter(Turn(("Sampling complete.",), raw_usage={"input_tokens": 1, "output_tokens": 1}))
    case = next(case for case in CASES if case.id == "cancellation")
    async with CallRuntime(adapters={"codex": adapter}, pricing=PRICES) as runtime:
        answer, checks = await execute(
            case,
            runtime,
            Binding("codex", "test"),
            CallLimits(wall_clock_s=2),
            lambda event: None,
            tmp_path,
            "native-probe",
        )
    assert answer.outcome == "answered"
    assert "native CLI never entered the actual diagnostic tool" in checks
    assert "cancelled native diagnostic did not finish host cleanup" in checks
    assert "in-flight native CLI cancellation was not reported" in checks
