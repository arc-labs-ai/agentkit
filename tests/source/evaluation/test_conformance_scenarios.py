"""Pre-dispatch rejection must be precise; task-scoped context survives handoff."""

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.types import Binding, CallLimits
from tests.evals.suites.conformance import CASES, execute


@pytest.mark.asyncio
@pytest.mark.parametrize("message,accepted", (
    ("openai_api/test has no verified support for top_k", True),
    ("openai_api/test has no verified support for temperature", False),
    ("invalid credential", False),
))
async def test_settings_probe_only_accepts_its_exact_unsupported_setting(tmp_path, message, accepted):
    class Runtime:
        async def ask(self, *args, **kwargs):
            raise ValueError(message)

    case = next(case for case in CASES if case.id == "settings_refusal")
    assert case.expected_pre_dispatch_rejection
    assert all(not entry.expected_pre_dispatch_rejection for entry in CASES if entry is not case)
    arguments = (case, Runtime(), Binding(Mode.OPENAI_API, "test"), CallLimits(), lambda _: None, tmp_path, "probe")
    if accepted:
        assert await execute(*arguments) == (None, [])
    else:
        with pytest.raises(ValueError, match=message):
            await execute(*arguments)


@pytest.mark.asyncio
async def test_context_switch_scope_authorizes_second_completion_and_retains_tool_only_evidence(tmp_path, monkeypatch):
    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.contracts.types import Turn
    from tests.source.conformance.test_calls import PRICES, ScriptedAdapter
    from tests.source.evaluation.test_knowledge_scenarios import call, turn

    class Adapter(ScriptedAdapter):
        async def turn(self, binding, prompt, **kwargs):
            if self.turns:
                return await super().turn(binding, prompt, **kwargs)
            self.requests.append((binding, prompt, kwargs))
            import re
            token = re.search(r"context-evidence-[0-9a-f]{16}", str(kwargs["session"].history)).group()
            return Turn(value={"token": token}, messages=(token,), raw_usage={"input_tokens": 1, "output_tokens": 1})

    from tests.evals.engine.config import EvalConfig

    monkeypatch.setattr(EvalConfig, "binding", lambda self, mode, credential: Binding(mode, "test", credential))
    adapter = Adapter(turn(calls=(call("read_diagnostic_context_evidence"),)), turn({"status": "complete"}))
    runtime = CallRuntime(adapters={Mode.OPENAI_API: adapter, Mode.CLAUDE_CODE: adapter}, pricing=PRICES)
    case = next(case for case in CASES if case.id == "tool_context_switch")
    answer, checks = await execute(case, runtime, Binding(Mode.OPENAI_API, "test"), CallLimits(),
                                   lambda _: None, tmp_path, "context-probe")
    assert not checks and answer.value["token"].startswith("context-evidence-"), repr(answer.failure)
    assert "first task/completion only" in str(adapter.requests[0][1])
    assert "explicitly authorizes" in str(adapter.requests[-1][1])


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", (True, False))
async def test_tool_error_recovery_requires_a_real_successful_retry(tmp_path, retry):
    import json

    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.contracts.types import Turn
    from tests.source.conformance.test_calls import PRICES

    class EvidenceAdapter:
        def __init__(self):
            self.requests = []

        async def turn(self, binding, prompt, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1 or len(self.requests) == 2 and retry:
                return Turn(tool_calls=({"id": str(len(self.requests)), "name": "fetch_evidence",
                                         "arguments": {}},), raw_usage={"input_tokens": 1, "output_tokens": 1})
            results = [m for m in kwargs["session"].messages if m.role == "tool"]
            assert results[0].metadata["is_error"] is True
            token = results[-1].content if retry else "invented-token"
            return Turn((json.dumps({"token": token}),), raw_usage={"input_tokens": 1, "output_tokens": 1})

    adapter = EvidenceAdapter()
    runtime = CallRuntime(adapters={Mode.ANTHROPIC_API: adapter}, pricing=PRICES)
    case = next(case for case in CASES if case.id == "tool_error_recovery")
    answer, checks = await execute(case, runtime, Binding(Mode.ANTHROPIC_API, "test"), CallLimits(max_turns=4),
                                   lambda _: None, tmp_path, "recover-probe")
    assert answer.outcome == "answered"
    assert (not checks) == retry
    if not retry:
        assert "tool failure was not recovered" in checks


@pytest.mark.asyncio
async def test_skill_format_feedback_retains_loaded_evidence_without_disclosing_the_expected_answer(tmp_path):
    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.contracts.types import Turn
    from tests.source.conformance.test_calls import PRICES, ScriptedAdapter

    calculated = str(6 * 7)
    usage = {"input_tokens": 1, "output_tokens": 1}
    adapter = ScriptedAdapter(
        Turn(tool_calls=({"id": "load", "name": "load_skill",
                          "arguments": {"name": "filesystem:arithmetic"}},), raw_usage=usage),
        Turn(("The calculation is complete: answer=" + calculated,), raw_usage=usage),
        Turn(("answer=" + calculated,), raw_usage=usage),
    )
    case = next(case for case in CASES if case.id == "discoverable_skill")
    answer, checks = await execute(case, CallRuntime(adapters={Mode.ANTHROPIC_API: adapter}, pricing=PRICES),
                                   Binding(Mode.ANTHROPIC_API, "test"), CallLimits(max_turns=4),
                                   lambda _: None, tmp_path, "skill-feedback")
    assert checks == [] and answer.outcome == "answered"
    session = adapter.requests[-1][2]["session"]
    feedback = next(m.content for m in session.messages if m.role == "user" and "strict final output" in m.content)
    assert "answer=" + calculated not in feedback and "literal output pattern" in feedback
    assert any(m.role == "tool" and m.name == "load_skill" and not m.metadata.get("is_error")
               for m in session.messages)
