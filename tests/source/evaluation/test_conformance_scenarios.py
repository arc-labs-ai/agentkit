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
    from agentkit.models.contracts.types import CredentialRef

    destination = Binding(Mode.CLAUDE_CODE, "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "claude-test"}))
    answer, checks = await execute(case, runtime, Binding(Mode.OPENAI_API, "test"), CallLimits(),
                                   lambda _: None, tmp_path, "context-probe",
                                   bindings={Mode.OPENAI_API: Binding(Mode.OPENAI_API, "test"),
                                             Mode.CLAUDE_CODE: destination})
    assert not checks and answer.value["token"].startswith("context-evidence-"), repr(answer.failure)
    assert adapter.requests[-1][0].credential.resolve() == {"ANTHROPIC_API_KEY": "claude-test"}
    assert "first task/completion only" in str(adapter.requests[0][1])
    assert "explicitly authorizes" in str(adapter.requests[-1][1])


@pytest.mark.asyncio
async def test_model_switch_uses_only_the_selected_destination_modes(tmp_path):
    import re
    from types import SimpleNamespace

    from tests.evals.suites.conformance import execute

    class Runtime:
        def __init__(self):
            self.bindings = []

        async def ask(self, binding, prompt, *args, **kwargs):
            self.bindings.append(binding)
            if len(self.bindings) == 1:
                code = re.search(r"Remember this exact code: ([^.]+)", prompt).group(1)
                session = SimpleNamespace(history=f"exact code assigned above: {code}")
                return SimpleNamespace(outcome="answered", text="OK", session=session, value=None)
            code = re.search(r"exact code assigned above: (.*)", kwargs["resume"].history).group(1)
            return SimpleNamespace(outcome="answered", text=code, session="session", value={"code": code})

    runtime = Runtime()
    case = next(case for case in CASES if case.id == "model_switch")
    claude = Binding(Mode.CLAUDE_CODE, "claude-test")
    anthropic = Binding(Mode.ANTHROPIC_API, "anthropic-test")
    openai = Binding(Mode.OPENAI_API, "openai-test")
    answer, checks = await execute(
        case, runtime, claude, CallLimits(), lambda _: None, tmp_path, "selected-model-switch",
        bindings={Mode.CLAUDE_CODE: claude, Mode.ANTHROPIC_API: anthropic, Mode.OPENAI_API: openai},
    )
    assert runtime.bindings == [claude, anthropic]
    assert answer.value["code"]
    assert not checks


@pytest.mark.asyncio
async def test_working_context_state_survives_two_real_runtime_calls(tmp_path):
    import json
    import re

    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.contracts.types import Turn
    from tests.evals.suites.conformance import execute
    from tests.source.conformance.test_calls import PRICES

    class ContextAdapter:
        def __init__(self):
            self.requests = []

        async def turn(self, binding, prompt, **kwargs):
            self.requests.append((binding, prompt, kwargs))
            if kwargs["schema"] is None:
                return Turn(("READY",), raw_usage={"input_tokens": 1, "output_tokens": 1})
            session = kwargs["session"]
            transcript = repr(session.history) + repr(session.messages)
            marker = re.search(r"state-context-[0-9a-f]{16}", transcript).group()
            value = {"token": marker}
            return Turn((json.dumps(value),), value=value, raw_usage={"input_tokens": 1, "output_tokens": 1})

    case = next(case for case in CASES if case.id == "working_context_state")
    adapter = ContextAdapter()
    answer, checks = await execute(
        case,
        CallRuntime(adapters={Mode.OPENAI_API: adapter}, pricing=PRICES),
        Binding(Mode.OPENAI_API, "test"),
        CallLimits(max_turns=4),
        lambda _: None,
        tmp_path,
        "working-context-state",
    )
    assert not checks, checks
    assert answer.value["token"].startswith("state-context-")
    assert len(adapter.requests) == 2
    first, second = (request[2]["session"] for request in adapter.requests)
    assert first is None and second is not None
    assert second.prefix.grounding
    assert any("READY" in str(item) for item in second.history)


@pytest.mark.asyncio
async def test_tool_error_recovery_reprompts_after_a_model_guesses_from_tool_failure(tmp_path):
    import json

    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.contracts.types import Turn
    from tests.source.conformance.test_calls import PRICES

    class EvidenceAdapter:
        def __init__(self):
            self.requests = []

        async def turn(self, binding, prompt, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) in (1, 3):
                return Turn(tool_calls=({"id": str(len(self.requests)), "name": "fetch_evidence",
                                         "arguments": {}},), raw_usage={"input_tokens": 1, "output_tokens": 1})
            if len(self.requests) == 2:
                return Turn((json.dumps({"token": "invented-token"}),), value={"token": "invented-token"},
                            raw_usage={"input_tokens": 1, "output_tokens": 1})
            results = [m for m in kwargs["session"].messages if m.role == "tool"]
            assert results[0].metadata["is_error"] is True
            token = results[-1].content
            return Turn((json.dumps({"token": token}),), value={"token": token},
                        raw_usage={"input_tokens": 1, "output_tokens": 1})

    adapter = EvidenceAdapter()
    runtime = CallRuntime(adapters={Mode.ANTHROPIC_API: adapter}, pricing=PRICES)
    case = next(case for case in CASES if case.id == "tool_error_recovery")
    answer, checks = await execute(case, runtime, Binding(Mode.ANTHROPIC_API, "test"), CallLimits(max_turns=4),
                                   lambda _: None, tmp_path, "recover-probe")
    assert answer.outcome == "answered"
    assert checks == []
    assert len(adapter.requests) == 4


@pytest.mark.asyncio
async def test_delegation_eval_reserves_child_budget_and_uses_host_controlled_brief(tmp_path, monkeypatch):
    from decimal import Decimal
    from types import SimpleNamespace

    from agentkit.agents.execution.runner import AgentRunner

    captured = {}

    async def run_agent(self, definition, task, inputs, **kwargs):
        captured["share"] = definition.delegation.budget_share
        child_ref, objective = definition.delegation.allowed[0], "model-supplied objective"
        captured["brief"] = self.delegation_brief(
            await self.agents.get(child_ref), objective
        ).goal
        return SimpleNamespace(outcome="answered", text="unimportant for this policy assertion")

    monkeypatch.setattr(AgentRunner, "run_agent", run_agent)
    case = next(case for case in CASES if case.id == "delegation")

    class Runtime:
        pass

    await execute(case, Runtime(), Binding(Mode.ANTHROPIC_API, "test"), CallLimits(max_cost_usd=Decimal("0.75")),
                  lambda _: None, tmp_path, "delegation-policy")
    assert captured["share"] >= Decimal("0.65")
    assert "model-supplied objective" not in captured["brief"]
    assert "fetch_delegated_evidence" in captured["brief"]


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
