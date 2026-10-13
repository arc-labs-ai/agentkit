"""Bounded concurrent backend matrix with deterministic workflow acceptance."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime, RetryPolicy
from agentkit.models.contracts.constants import EventKind, FailureClass
from agentkit.models.providers.auth import anthropic_credential
from tests.evals.catalog import CASES, WORKLOAD_EXECUTORS
from tests.evals.engine.config import EvalConfig
from tests.evals.engine.contracts import (
    BudgetScope,
    EvalDomain,
    EvalStatus,
    WorkflowEnvironment,
    WorkflowResult,
    WorkloadCase,
)
from tests.evals.engine.environment import runtime_versions
from tests.evals.engine.reporting import case_spend, redact, source_digest, write_events, write_report
from tests.evals.sandbox.workspace import SandboxUnavailable, WorkspaceSandbox
from tests.evals.suites.conformance import EvalCase, execute


async def run_matrix(
    config: EvalConfig,
    *,
    suites: tuple[str, ...] = ("workload",),
    case_filter: tuple[str, ...] = (),
    domain_filter: tuple[EvalDomain, ...] = (),
) -> tuple[Path, bool]:
    credential = config.credentials()
    prices = config.prices()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.report_dir / timestamp
    target = run_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    spent = Decimal(0)
    reserved = Decimal(0)
    secrets = tuple(credential.resolve().values())
    active: dict[str, Any] = {}
    report = {
        "date": timestamp,
        "price_version": prices.version,
        "spend_usd": str(spent),
        "requested_modes": config.modes,
        "requested_suites": suites,
        "requested_domains": domain_filter,
        "rows": rows,
        "complete": False,
        "passed": False,
        "source_digest": source_digest(),
        "in_progress": active,
        "runtime_versions": await runtime_versions(config.modes),
    }
    write_report(target, report, secrets)
    budget = asyncio.Condition()
    concurrency = asyncio.Semaphore(config.jobs)

    def checkpoint() -> None:
        report["spend_usd"] = str(spent)
        report["spend_upper_bound_usd"] = str(spent + reserved)
        write_report(target, report, secrets)

    selected = tuple(
        case
        for case in CASES
        if case.suite in suites
        and (not case_filter or case.id in case_filter)
        and (not domain_filter or isinstance(case, WorkloadCase) and case.domain in domain_filter)
    )
    missing = set(case_filter) - {case.id for case in selected}
    if missing:
        raise ValueError("no selected evaluation matches: " + ", ".join(sorted(missing)))

    async with CallRuntime(pricing=prices, retry=RetryPolicy(max_attempts=2)) as runtime:

        async def backend(mode: Mode) -> None:
            nonlocal spent, reserved
            binding = config.binding(mode, credential)
            values = credential.resolve()
            available = (
                anthropic_credential(values) is not None
                if mode in (Mode.CLAUDE_CODE, Mode.ANTHROPIC_API)
                else bool(values.get("OPENAI_API_KEY"))
            )
            binary = "claude" if mode == Mode.CLAUDE_CODE else "codex" if mode == Mode.CODEX else None
            unavailable = not available or (binary and not shutil.which(binary))
            for trial, case in ((trial, case) for trial in range(1, config.repetitions + 1) for case in selected):
                is_workload = isinstance(case, WorkloadCase)
                case_budget = min(
                    config.max_case_usd, case.max_cost_usd if isinstance(case, WorkloadCase) else Decimal("0.25")
                )
                envelope = case_budget
                # Contract cases declare a per-call envelope; workload metadata
                # declares the complete workflow envelope including child calls.
                if not is_workload:
                    envelope *= case.max_calls
                call_id = f"eval/{timestamp}/{mode}/{case.id}/trial-{trial}"
                domain = case.domain if isinstance(case, WorkloadCase) else "conformance"
                reason = "credentials/binary unavailable" if unavailable else None
                supported_modes = case.supported_modes
                unsupported = bool(supported_modes) and mode not in supported_modes
                if unsupported:
                    reason = "scenario unsupported on this mode"
                if is_workload and domain == EvalDomain.CODING and not WorkspaceSandbox.process_available():
                    reason = "required OS process sandbox unavailable"
                async with concurrency:
                    async with budget:
                        if reason is None:
                            while spent + reserved + envelope > config.max_total_usd and reserved:
                                await budget.wait()
                            if spent + reserved + envelope > config.max_total_usd:
                                reason = "matrix spend ceiling"
                        if reason:
                            rows.append(
                                {
                                    "mode": mode,
                                    "case": case.id,
                                    "domain": domain,
                                    "suite": case.suite,
                                    "trial": trial,
                                    "expectation": case.expectation if isinstance(case, WorkloadCase) else "success",
                                    "pair_id": case.pair_id if isinstance(case, WorkloadCase) else None,
                                    "status": EvalStatus.NOT_APPLICABLE if unsupported else EvalStatus.NOT_RUN,
                                    "reason": reason,
                                }
                            )
                            checkpoint()
                            continue
                        reserved += envelope
                        active[call_id] = {"mode": mode, "case": case.id, "reserved_usd": str(envelope)}
                        checkpoint()
                    events: list[Any] = []
                    started = time.monotonic()
                    result = WorkflowResult(None)
                    dispatched = False
                    setup_unavailable = False
                    if isinstance(case, WorkloadCase):
                        limits = replace(
                            config.limits(),
                            max_cost_usd=case_budget
                            if case.budget_scope == BudgetScope.WORKFLOW
                            else case_budget / case.max_calls,
                            max_turns=case.max_turns,
                            max_output_tokens=case.max_output_tokens,
                            wall_clock_s=case.wall_clock_s,
                            idle_s=case.idle_s if case.idle_s is not None else config.limits().idle_s,
                            first_event_s=case.first_event_s if case.first_event_s is not None
                            else config.limits().first_event_s,
                        )
                    else:
                        limits = replace(config.limits(), max_cost_usd=case_budget)
                    artifacts = run_dir / "cases" / str(mode) / case.id / f"trial-{trial}"
                    with tempfile.TemporaryDirectory(prefix="agentkit-workload-") as directory:
                        workspace = Path(directory)
                        try:
                            dispatched = not (isinstance(case, EvalCase) and case.expected_pre_dispatch_rejection)
                            if isinstance(case, WorkloadCase):
                                environment = WorkflowEnvironment(
                                    runtime,
                                    binding,
                                    limits,
                                    events.append,
                                    workspace,
                                    call_id,
                                    WorkspaceSandbox(workspace),
                                )
                                result = await WORKLOAD_EXECUTORS[case.id](case, environment)
                            else:
                                answer, checks = await execute(
                                    case, runtime, binding, limits, events.append, workspace, call_id
                                )
                                result = WorkflowResult(answer, tuple(checks))
                        except SandboxUnavailable:
                            setup_unavailable = True
                            dispatched = bool(events)
                            result = WorkflowResult(None, ("required OS process sandbox unavailable",))
                        except Exception as exc:
                            # Exceptions may echo request/auth bodies. Record type only.
                            result = WorkflowResult(None, (f"runner exception: {type(exc).__name__}",))
                        answer = result.answer
                        checks = list(result.checks)
                        dispatched = dispatched or bool(events) or answer is not None
                        expected_rejection = (
                            isinstance(case, EvalCase)
                            and case.expected_pre_dispatch_rejection
                            and answer is None
                            and not events
                            and not checks
                            and not dispatched
                        )
                        if answer is None and not setup_unavailable and not expected_rejection:
                            checks.append("workflow produced no model answer")
                        if answer is not None and (
                            not events
                            or events[0].kind != EventKind.CALL_STARTED
                            or events[-1].kind != EventKind.CALL_FINISHED
                        ):
                            checks.append("missing lifecycle events")
                        usage = answer.usage if answer is not None else None
                        confirmed_auth_rejection = (
                            isinstance(case, EvalCase)
                            and case.id == "provider_auth_rejection"
                            and answer is not None
                            and answer.failure is not None
                            and answer.failure.classification == FailureClass.AUTH
                            and answer.attempts == 1
                            and usage is not None
                            and not any(
                                (
                                    usage.input_uncached,
                                    usage.input_cache_read,
                                    usage.input_cache_write_5m,
                                    usage.input_cache_write_1h,
                                    usage.output,
                                )
                            )
                        )
                        amount = case_spend(
                            events,
                            envelope,
                            dispatched=dispatched,
                            confirmed_no_billable_work=confirmed_auth_rejection,
                        )
                        if amount > envelope:
                            checks.append("case spend exceeded its declared envelope")
                        row = {
                            "mode": mode,
                            "model": binding.model,
                            "case": case.id,
                            "domain": domain,
                            "suite": case.suite,
                            "trial": trial,
                            "description": case.description if isinstance(case, WorkloadCase) else None,
                            "expectation": case.expectation if isinstance(case, WorkloadCase) else "contract_check",
                            "pair_id": case.pair_id if isinstance(case, WorkloadCase) else None,
                            "contract": () if isinstance(case, WorkloadCase) else case.contract,
                            "aspects": case.aspects if isinstance(case, WorkloadCase) else case.contract,
                            "status": EvalStatus.NOT_RUN
                            if setup_unavailable and not events
                            else EvalStatus.FAILED
                            if checks
                            else EvalStatus.PASSED,
                            "checks": checks,
                            "duration_s": round(time.monotonic() - started, 3),
                            "outcome": answer.outcome if answer else None,
                            "answer_text": redact(answer.text, secrets)[:4000] if answer else None,
                            "answer_messages": [redact(message, secrets)[:4000] for message in answer.messages[-20:]]
                            if answer and hasattr(answer, "messages") else [],
                            "answer_value": answer.value if answer else None,
                            "failure": asdict(answer.failure) if answer and answer.failure else None,
                            "cost_usd": str(answer.cost_usd) if answer and answer.cost_usd is not None else None,
                            "case_spend_usd": str(amount),
                            "usage": asdict(answer.usage) if answer else None,
                            "attempts": answer.attempts if answer else 0,
                            "events": [event.kind for event in events],
                            "evidence_path": str((artifacts / "evidence.json").relative_to(run_dir)),
                        }
                        write_report(artifacts / "evidence.json", dict(result.evidence), secrets)
                        write_events(artifacts / "events.jsonl", events, secrets)
                        write_report(artifacts / "result.json", row, secrets)
                    async with budget:
                        reserved -= envelope
                        spent += amount
                        active.pop(call_id, None)
                        rows.append(row)
                        checkpoint()
                        budget.notify_all()
                    print(f"{mode:15} {case.id:38} {row['status']:7} {row['duration_s']}s", flush=True)

        await asyncio.gather(*(backend(mode) for mode in config.modes))
    passed = any(row["status"] == EvalStatus.PASSED for row in rows) and all(
        row["status"] in (EvalStatus.PASSED, EvalStatus.NOT_APPLICABLE) for row in rows
    )
    report["source_changed"] = source_digest() != report["source_digest"]
    passed = passed and not report["source_changed"]
    from tests.evals.coverage import summarize_coverage

    report.update(complete=True, passed=passed, coverage=summarize_coverage(rows).as_dict())
    checkpoint()
    return target, passed
