import json
from decimal import Decimal
from pathlib import Path

import pytest

from agentkit.models.contracts.constants import EventKind
from agentkit.models.contracts.types import CallEvent
from tests.evals.engine.config import EvalConfig
from tests.evals.engine.reporting import case_spend, redact, write_report


def event(kind, call_id, cost=None):
    return CallEvent(kind, call_id, {"cost_usd": cost})


def test_resume_case_accounts_for_both_calls_and_deduplicates_delivery():
    first = event(EventKind.CALL_FINISHED, "first", "0.02")
    second = event(EventKind.CALL_FINISHED, "second", "0.03")
    events = [event(EventKind.CALL_STARTED, "first"), first, event(EventKind.CALL_STARTED, "second"), second, second]
    assert case_spend(events, Decimal("0.5"), dispatched=True) == Decimal("0.05")


@pytest.mark.parametrize("cost", [None, "None", "NaN", "Infinity", "-1", "invalid"])
def test_unknown_or_invalid_cost_reserves_full_case_envelope(cost):
    assert case_spend([event(EventKind.CALL_FINISHED, "a", cost)], Decimal("0.5"), dispatched=True) == Decimal("0.5")


def test_missing_terminal_event_reserves_envelope_even_after_first_completed_call():
    events = [event(EventKind.CALL_FINISHED, "first", "0.01"), event(EventKind.CALL_STARTED, "second")]
    assert case_spend(events, Decimal("0.5"), dispatched=True) == Decimal("0.5")
    assert case_spend([], Decimal("0.5"), dispatched=True) == Decimal("0.5")
    assert case_spend([], Decimal("0.5"), dispatched=False) == 0


def test_confirmed_auth_rejection_without_generation_has_no_billable_spend():
    events = [
        event(EventKind.CALL_STARTED, "auth"),
        event(EventKind.ATTEMPT_STARTED, "auth"),
        event(EventKind.ATTEMPT_FINISHED, "auth"),
        event(EventKind.CALL_FINISHED, "auth", None),
    ]
    assert case_spend(
        events,
        Decimal("0.25"),
        dispatched=True,
        confirmed_no_billable_work=True,
    ) == Decimal(0)


def test_report_checkpoint_redacts_credentials_recursively_and_replaces_atomically(tmp_path: Path):
    target = tmp_path / "reports" / "matrix.json"
    secret = "synthetic-private-test-token"
    report = {"rows": [{"answer_text": f"echo {secret}", "failure": {"detail": secret}}], secret: (secret,)}
    write_report(target, report, (secret,))
    data = target.read_text()
    assert secret not in data
    assert json.loads(data)["rows"][0]["answer_text"] == "echo [REDACTED]"
    assert not target.with_suffix(".tmp").exists()
    assert target.stat().st_mode & 0o777 == 0o600
    write_report(target, {"complete": True}, (secret,))
    assert json.loads(target.read_text()) == {"complete": True}
    assert redact("unchanged", ()) == "unchanged"


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
def test_eval_budget_rejects_invalid_envelopes(value):
    with pytest.raises(ValueError):
        EvalConfig(max_total_usd=Decimal(value))
    with pytest.raises(ValueError):
        EvalConfig(max_case_usd=Decimal(value))


@pytest.mark.asyncio
async def test_matrix_stops_before_next_two_call_case_exceeds_remaining_budget(monkeypatch, tmp_path):
    import tests.evals.engine.runner as runner
    from agentkit.core.contracts.types import Mode
    from agentkit.models.contracts.types import CredentialRef
    from tests.evals.suites.conformance import EvalCase

    class FakeRuntime:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def execute(case, runtime, binding, limits, events, directory, call_id):
        for suffix in ("", "/resume"):
            events(event(EventKind.CALL_STARTED, call_id + suffix))
            events(event(EventKind.CALL_FINISHED, call_id + suffix, "0.1"))

        class Answer:
            text = "synthetic-private-test-token"
            value = None
            outcome = "answered"
            cost_usd = Decimal("0.1")
            failure = None
            attempts = 1
            from agentkit.models.contracts.types import Usage

            usage = Usage()

        return Answer(), []

    monkeypatch.setattr(runner, "CallRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(
        runner, "CASES", (EvalCase("resume_one", (), "contract", 2), EvalCase("resume_two", (), "contract", 2))
    )
    monkeypatch.setattr(
        EvalConfig,
        "credentials",
        lambda self: CredentialRef(lambda: {"OPENAI_API_KEY": "synthetic-private-test-token"}),
    )
    config = EvalConfig(
        modes=(Mode.OPENAI_API,), report_dir=tmp_path, max_total_usd=Decimal("0.6"), max_case_usd=Decimal("0.25")
    )
    target, passed = await runner.run_matrix(config, suites=("contract",))
    data = json.loads(target.read_text())
    assert not passed
    assert data["complete"]
    assert data["spend_usd"] == "0.2"
    assert [row["status"] for row in data["rows"]] == ["passed", "not_run"]
    assert data["rows"][0]["case_spend_usd"] == "0.2"
    assert data["rows"][0]["answer_text"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_backend_fanout_reserves_global_workflow_budget_until_prior_case_finishes(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace

    import tests.evals.engine.runner as runner
    from agentkit.core.contracts.types import Mode
    from agentkit.models.contracts.types import CredentialRef, Usage
    from tests.evals.engine.contracts import EvalDomain, WorkflowResult, WorkloadCase

    class FakeRuntime:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    starts = []

    async def execute_workflow(case, env):
        starts.append(env.binding.mode)
        env.event_sink(event(EventKind.CALL_STARTED, env.call_id))
        if len(starts) == 1:
            first_started.set()
            await release_first.wait()
        env.event_sink(event(EventKind.CALL_FINISHED, env.call_id, "0.1"))
        answer = SimpleNamespace(
            text="done",
            value={"done": True},
            outcome="answered",
            failure=None,
            cost_usd=Decimal("0.1"),
            usage=Usage(),
            attempts=1,
        )
        return WorkflowResult(answer, evidence={"actual_state": "committed"})

    case = WorkloadCase(
        "real_budget_case",
        EvalDomain.OPERATIONS,
        "Apply one approved transaction",
        ("state",),
        max_cost_usd=Decimal("1"),
    )
    monkeypatch.setattr(runner, "CallRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "CASES", (case,))
    monkeypatch.setattr(runner, "WORKLOAD_EXECUTORS", {case.id: execute_workflow})
    monkeypatch.setattr(
        EvalConfig,
        "credentials",
        lambda self: CredentialRef(
            lambda: {"OPENAI_API_KEY": "synthetic-openai", "ANTHROPIC_API_KEY": "synthetic-anthropic"}
        ),
    )
    config = EvalConfig(
        modes=(Mode.ANTHROPIC_API, Mode.OPENAI_API), report_dir=tmp_path, max_total_usd=Decimal("1.1"), jobs=2
    )
    task = asyncio.create_task(runner.run_matrix(config))
    await first_started.wait()
    await asyncio.sleep(0.02)
    assert len(starts) == 1
    partial = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert partial["spend_upper_bound_usd"] == "1"
    release_first.set()
    target, passed = await task
    assert passed
    data = json.loads(target.read_text())
    assert data["spend_usd"] == "0.2" and len(data["rows"]) == 2
    assert not data["in_progress"]
    assert not data["source_changed"]
    for mode in config.modes:
        artifacts = target.parent / "cases" / mode / case.id / "trial-1"
        assert json.loads((artifacts / "evidence.json").read_text()) == {"actual_state": "committed"}
        trace = [json.loads(line) for line in (artifacts / "events.jsonl").read_text().splitlines()]
        assert [record["kind"] for record in trace] == ["call_started", "call_finished"]


@pytest.mark.asyncio
async def test_unsupported_modes_are_not_applicable_and_never_reserve_or_dispatch(monkeypatch, tmp_path):
    import tests.evals.engine.runner as runner
    from agentkit.core.contracts.types import Mode
    from agentkit.models.contracts.types import CredentialRef
    from tests.evals.engine.contracts import EvalDomain, WorkflowResult, WorkloadCase

    supported = WorkloadCase("api_only", EvalDomain.OPERATIONS, "API-only application witness", ("tools",),
                             supported_modes=(Mode.OPENAI_API,))
    calls = []

    async def execute(case, env):
        from types import SimpleNamespace

        from agentkit.models.contracts.types import Usage

        calls.append(env.binding.mode)
        env.event_sink(event(EventKind.CALL_STARTED, env.call_id))
        env.event_sink(event(EventKind.CALL_FINISHED, env.call_id, "0"))
        answer = SimpleNamespace(
            text="done", value=None, outcome="answered", failure=None,
            cost_usd=Decimal("0"), usage=Usage(), attempts=1,
        )
        return WorkflowResult(answer)

    monkeypatch.setattr(runner, "CASES", (supported,))
    monkeypatch.setattr(runner, "WORKLOAD_EXECUTORS", {supported.id: execute})
    monkeypatch.setattr(EvalConfig, "credentials", lambda self: CredentialRef(lambda: {
        "OPENAI_API_KEY": "synthetic-test", "ANTHROPIC_API_KEY": "synthetic-test",
    }))
    config = EvalConfig(modes=(Mode.OPENAI_API, Mode.CLAUDE_CODE), report_dir=tmp_path)
    target, passed = await runner.run_matrix(config)
    rows = json.loads(target.read_text())["rows"]
    assert passed
    assert calls == [Mode.OPENAI_API]
    assert {row["mode"]: row["status"] for row in rows} == {
        "openai_api": "passed", "claude_code": "not_applicable",
    }
    assert not any("case_spend_usd" in row for row in rows if row["status"] == "not_applicable")


@pytest.mark.asyncio
@pytest.mark.parametrize("flag,dispatch,executor_checks,expected_status,expected_spend", (
    (True, False, [], "passed", "0"),
    (False, False, [], "failed", "0.1"),
    (True, True, [], "failed", "0.1"),
    (True, False, ["unexpected refusal"], "failed", "0"),
))
async def test_only_declared_clean_pre_dispatch_rejection_can_omit_an_answer(
    monkeypatch, tmp_path, flag, dispatch, executor_checks, expected_status, expected_spend,
):
    import tests.evals.engine.runner as runner
    from agentkit.core.contracts.types import Mode
    from agentkit.models.contracts.types import CredentialRef
    from tests.evals.suites.conformance import EvalCase

    class FakeRuntime:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass

    case = EvalCase("rejection_contract", ("C3",), "probes", expected_pre_dispatch_rejection=flag)
    async def execute(case, runtime, binding, limits, events, directory, call_id):
        if dispatch:
            events(event(EventKind.CALL_STARTED, call_id))
        return None, executor_checks

    monkeypatch.setattr(runner, "CallRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "CASES", (case,))
    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(EvalConfig, "credentials", lambda self: CredentialRef(lambda: {"OPENAI_API_KEY": "synthetic-test"}))
    config = EvalConfig(modes=(Mode.OPENAI_API,), report_dir=tmp_path,
                        max_total_usd=Decimal(".1"), max_case_usd=Decimal(".1"))
    target, passed = await runner.run_matrix(config, suites=("probes",))
    report = json.loads(target.read_text())
    row = report["rows"][0]
    assert row["status"] == expected_status and passed == (expected_status == "passed")
    assert row["case_spend_usd"] == expected_spend
    assert row["answer_value"] is None and row["attempts"] == 0
    assert row["events"] == (["call_started"] if dispatch else [])
