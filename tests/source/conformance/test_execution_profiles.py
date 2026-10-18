"""Explicit profiles use the existing runtime and host policy path.

SDK HTTP and CLI binaries are scripted here; the CLI bridge uses actual local
MCP HTTP. Real binary characterization is separate integration coverage.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from agentkit.core.runtime import CallRuntime
from agentkit.models import (
    Binding,
    CallLimits,
    CredentialRef,
    ExecutionRequirement,
    ExecutionRequirements,
    NativeExecution,
    NativeToolPolicy,
    SessionHandle,
)


def requirements(profile):
    return ExecutionRequirements(profile=profile)


@pytest.mark.parametrize("profile", ["host_tools_only", "cli_compatible"])
def test_profiles_are_first_class_public_contracts(profile):
    import agentkit.models as models
    import agentkit.models.contracts as contracts

    assert hasattr(models, "ExecutionProfile"), "profiles must be part of the public API"
    assert hasattr(contracts, "ExecutionProfile"), "model contracts must export the same profile type"
    assert contracts.ExecutionProfile is models.ExecutionProfile
    request = requirements(profile)
    assert request.profile == models.ExecutionProfile(profile)
    assert {
        ExecutionRequirement.HOST_TOOLS,
        ExecutionRequirement.SCHEMA,
        ExecutionRequirement.USAGE,
    } <= request.required
    assert (ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE in request.required) is (profile == "host_tools_only")
    assert request.native_execution == NativeExecution.UNCONSTRAINED
    assert ExecutionRequirement.CREDENTIALS not in request.required


def test_unknown_profile_is_refused_instead_of_defaulting_to_a_weaker_profile():
    with pytest.raises(ValueError, match="profile"):
        requirements("almost_host_tools_only")


@pytest.mark.parametrize(
    "requested,ceiling,expected",
    [
        ("cli_compatible", "host_tools_only", "host_tools_only"),
        ("host_tools_only", "cli_compatible", "host_tools_only"),
        ("cli_compatible", "cli_compatible", "cli_compatible"),
        (None, "cli_compatible", "cli_compatible"),
        ("host_tools_only", None, "host_tools_only"),
    ],
)
def test_profile_intersection_preserves_the_stronger_contract(requested, ceiling, expected):
    result = requirements(requested).constrain(requirements(ceiling))
    assert result.profile == expected
    assert (ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE in result.required) is (expected == "host_tools_only")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
@pytest.mark.parametrize("profile", ["host_tools_only", "cli_compatible"])
async def test_profile_assessment_is_credential_free_and_truthful(mode, profile):
    from agentkit.models.providers.cli import CliAdapter

    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential resolved")))
    adapters = {} if mode.endswith("_api") else {mode: CliAdapter(spawn=lambda *args, **kwargs: None)}
    async with CallRuntime(adapters=adapters) as runtime:
        report = await runtime.evaluate_execution(binding, execution=requirements(profile))
    assert report.profile == profile
    assert report.compatible is (mode.endswith("_api") or profile == "cli_compatible")
    assert {fact.requirement for fact in report.assessments} == set(ExecutionRequirement)
    if not report.compatible:
        absence = next(fact for fact in report.incompatibilities if fact.requirement == "native_tools_unavailable")
        assert absence.status == ("unsupported" if mode == "codex" else "unverified")
        assert absence.code and absence.explanation
    if profile == "cli_compatible" and not mode.endswith("_api"):
        containment = next(fact for fact in report.assessments if fact.requirement == ExecutionRequirement.CREDENTIALS)
        assert containment.status != "supported"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
async def test_cli_compatible_cannot_weaken_a_runtime_deny_ceiling(mode):
    from agentkit.models.providers.cli import CliAdapter

    adapter = CliAdapter(spawn=lambda *args, **kwargs: pytest.fail("CLI dispatched"))
    async with CallRuntime(adapters={mode: adapter}, native_tool_policy="deny") as runtime:
        answer = await runtime.run(
            Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential resolved"))),
            "Do the job",
            call_id="deny-ceiling",
            execution=requirements("cli_compatible"),
        )
    assert answer.failure.classification == "execution_incompatible"
    assert answer.attempts == 0 and answer.cost_usd is None and not answer.usage_complete
    assert answer.failure.mode_detail["dispatch_state"] == "not_started"


@pytest.mark.asyncio
async def test_codex_spending_limit_is_assessed_independently_of_profile():
    from agentkit.models.providers.cli import CliAdapter

    binding = Binding("codex", "test", CredentialRef(lambda: pytest.fail("credential resolved")))
    limits = CallLimits(max_cost_usd=Decimal("0.01"), max_turns=2)
    async with CallRuntime(
        adapters={"codex": CliAdapter(spawn=lambda *a, **k: pytest.fail("CLI dispatched"))}
    ) as runtime:
        clean = await runtime.evaluate_execution(binding, execution=requirements("cli_compatible"))
        planted = await runtime.evaluate_execution(binding, execution=requirements("cli_compatible"), limits=limits)
        answer = await runtime.run(
            binding, "job", call_id="cap", execution=requirements("cli_compatible"), limits=limits
        )
    assert clean.compatible and not planted.compatible
    assert any(fact.code == "observed_spending_threshold_refused_for_mode" for fact in planted.incompatibilities)
    assert answer.failure.classification == "execution_incompatible" and answer.attempts == 0
    assert limits.max_cost_usd == Decimal("0.01")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
async def test_profiles_do_not_promote_requested_containment(mode):
    from agentkit.models.providers.cli import CliAdapter

    request = replace(requirements("cli_compatible"), native_execution=NativeExecution.CONTAINED)
    async with CallRuntime(adapters={mode: CliAdapter(spawn=lambda *a, **k: None)}) as runtime:
        report = await runtime.evaluate_execution(Binding(mode, "test"), execution=request)
    assert not report.compatible
    assert any(fact.requirement == ExecutionRequirement.CREDENTIALS for fact in report.incompatibilities)


# Reuse the example's transport doubles, not its model loop or permission logic.
from contextlib import asynccontextmanager

from tests.examples.plugin_execution.tests.conftest import scripted_cli as scripted_cli
from tests.examples.plugin_execution.tests.conftest import scripted_client as scripted_client


@asynccontextmanager
async def provider_runtime(mode, scripted_client, scripted_cli, *, ceiling="warn", **script):
    from agentkit.models.providers.cli import CliAdapter
    from agentkit.models.providers.frontiers import ApiAdapter

    client = None
    if mode.endswith("_api"):
        client, _ = scripted_client(mode, **script)
        adapter = ApiAdapter(client=client)
    else:
        adapter = CliAdapter(spawn=scripted_cli(mode, **script).spawn)
    try:
        async with CallRuntime(
            adapters={mode: adapter},
            native_tool_policy=ceiling,
            unpriced_policy="proceed_unpriced",
        ) as runtime:
            yield runtime
    finally:
        if client is not None:
            await client.aclose()


def binding(mode):
    key = "ANTHROPIC_API_KEY" if mode in ("anthropic_api", "claude_code") else "OPENAI_API_KEY"
    return Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
@pytest.mark.parametrize("defect", [None, "denied", "unattended", "unknown", "invalid", "scope"])
async def test_cli_compatible_uses_one_host_policy_path(mode, defect, scripted_client, scripted_cli):
    from agentkit.core import Scope
    from agentkit.tools import PermissionDecision
    from tests.examples.plugin_execution.wiring.runtime import compose_call, execute

    call = compose_call(
        "policy",
        granted_ids=frozenset() if defect == "denied" else frozenset({"record-1"}),
        permissions=(lambda _: PermissionDecision.ASK,) if defect == "unattended" else (),
    )
    if defect == "scope":
        call = replace(call, scope=Scope(org_id=2))
    script = {
        "tool_name": "undeclared" if defect == "unknown" else "lookup_record",
        "record_id": 42 if defect == "invalid" else "record-1",
        "answer_record_id": "record-1",
        "status": "unavailable" if defect else "ready",
    }
    async with provider_runtime(mode, scripted_client, scripted_cli, **script) as runtime:
        events = []
        original = call.lifecycle.call

        def observe(event):
            events.append(event)
            original(event)

        call.lifecycle.call = observe
        answer = await execute(runtime, binding(mode), call, execution=requirements("cli_compatible"))
    assert answer.session.execution.profile == "cli_compatible"
    assert answer.session.native_tool_policy == NativeToolPolicy.WARN
    assert call.effects == ([] if defect else ["record-1"])
    assert sum(event.kind == "call_finished" for event in events) == 1
    assert (
        next(event for event in events if event.kind == "call_started").fields["execution_profile"] == "cli_compatible"
    )
    if defect is None:
        assert answer.outcome == "answered" and answer.value.status == "ready"
        tool_events = [event for event in call.lifecycle.events if event["kind"].startswith("tool_")]
        assert [event["kind"] for event in tool_events] == ["tool_requested", "tool_finished"]
        assert tool_events[0]["invocation_id"] == tool_events[1]["invocation_id"]
        assert any(event["kind"] == "permission_decision" for event in call.lifecycle.events)
    elif defect != "unknown":
        refused = next(event for event in call.lifecycle.events if event["kind"] == "tool_finished")
        assert (
            refused["code"]
            == {
                "denied": "permission_denied",
                "scope": "permission_denied",
                "unattended": "permission_unattended",
                "invalid": "tool_arguments_invalid",
            }[defect]
        )
    if mode == "codex":
        assert any(event.kind == "degraded" and event.fields.get("setting") == "native_tools" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_concurrent_cli_compatible_scope_isolation(mode, scripted_client, scripted_cli):
    import asyncio

    from agentkit.core import Scope
    from tests.examples.plugin_execution.wiring.runtime import compose_call, execute

    left = compose_call("left", scope=Scope(org_id=1))
    right = compose_call(
        "right", records={"record-2": "private"}, granted_ids=frozenset({"record-2"}), scope=Scope(org_id=2)
    )
    async with (
        provider_runtime(mode, scripted_client, scripted_cli) as runtime_left,
        provider_runtime(mode, scripted_client, scripted_cli, status="unavailable") as runtime_right,
    ):
        answers = await asyncio.gather(
            execute(runtime_left, binding(mode), left, execution=requirements("cli_compatible")),
            execute(runtime_right, binding(mode), right, execution=requirements("cli_compatible")),
        )
    assert left.effects == ["record-1"] and not right.effects
    assert any(event.get("code") == "permission_denied" for event in right.lifecycle.events)
    assert all(event["call_id"] == "left" for event in left.lifecycle.events)
    assert all(event["call_id"] == "right" for event in right.lifecycle.events)
    assert all(answer.session.execution.profile == "cli_compatible" for answer in answers)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
@pytest.mark.parametrize("fork", [False, True])
async def test_resumed_strict_profile_cannot_silently_become_cli_compatible(mode, fork):
    from agentkit.models.providers.cli import CliAdapter

    session = SessionHandle(
        mode, "strict-parent", cli_session_id="strict-session", execution=requirements("host_tools_only")
    )
    async with CallRuntime(adapters={mode: CliAdapter(spawn=lambda *a, **k: pytest.fail("CLI dispatched"))}) as runtime:
        answer = await runtime.run(
            Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential resolved"))),
            "resume",
            call_id="child",
            resume=session,
            fork=fork,
            execution=requirements("cli_compatible"),
        )
    assert answer.failure.classification == "execution_incompatible" and answer.attempts == 0
    assert answer.failure.mode_detail["execution_profile"] == "host_tools_only"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_named_profile_cannot_drop_an_explicit_host_permission_deny(mode, scripted_client, scripted_cli):
    from agentkit.tools import PermissionDecision
    from tests.examples.plugin_execution.wiring.runtime import compose_call, execute

    call = compose_call(
        "deny-final", permissions=(lambda _: PermissionDecision.DENY, lambda _: PermissionDecision.ALLOW)
    )
    async with provider_runtime(mode, scripted_client, scripted_cli, status="unavailable") as runtime:
        await execute(runtime, binding(mode), call, execution=requirements("cli_compatible"))
    assert not call.effects
    assert any(event.get("code") == "permission_denied" for event in call.lifecycle.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_cli_compatible_uses_warn_even_under_an_allow_ceiling(mode, scripted_client, scripted_cli):
    from tests.examples.plugin_execution.wiring.runtime import compose_call, execute

    call = compose_call("explicit-warn")
    async with provider_runtime(mode, scripted_client, scripted_cli, ceiling="allow") as runtime:
        answer = await execute(runtime, binding(mode), call, execution=requirements("cli_compatible"))
    assert answer.outcome == "answered" and call.effects == ["record-1"]
    assert answer.session.native_tool_policy == NativeToolPolicy.WARN


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
@pytest.mark.parametrize("profile", ["host_tools_only", "cli_compatible"])
@pytest.mark.parametrize("path", ["retry", "correction", "resume"])
async def test_profile_preserved_on_all_api_continuations(mode, profile, path):
    import json
    from typing import Literal

    import httpx
    from pydantic import BaseModel

    from agentkit.core.runtime import RetryPolicy
    from agentkit.models.providers.frontiers import ApiAdapter
    from tests.source.unit.models.adapters.test_sdk_execution_assessment import _final_frames

    class Answer(BaseModel):
        status: Literal["ok"]

    seen, requests = [], []

    class Recording(ApiAdapter):
        async def turn(self, binding, prompt, **kwargs):
            seen.append((kwargs.get("execution"), kwargs.get("native_tool_policy")))
            return await super().turn(binding, prompt, **kwargs)

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1 and path == "retry":
            return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})
        text = '{"status":"invalid"}' if len(requests) == 1 and path == "correction" else '{"status":"ok"}'
        frames = _final_frames(mode, text)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as client:
        async with CallRuntime(
            adapters={mode: Recording(client=client)},
            unpriced_policy="proceed_unpriced",
            retry=RetryPolicy(base_s=0, cap_s=0),
        ) as runtime:
            answer = await runtime.run(
                binding(mode),
                "job",
                output=Answer,
                call_id="first",
                execution=requirements(profile),
                limits=CallLimits(max_turns=3),
            )
            if path in ("resume", "fork"):
                answer = await runtime.run(
                    binding(mode),
                    "continue",
                    output=Answer,
                    call_id="second",
                    resume=answer.session,
                    fork=path == "fork",
                    limits=CallLimits(max_turns=3),
                )
    assert answer.outcome == "answered", answer.failure
    assert len(seen) == 2 and len(requests) == 2
    assert all(execution.profile == profile for execution, _ in seen)
    expected = NativeToolPolicy.DENY if profile == "host_tools_only" else NativeToolPolicy.WARN
    assert all(policy == expected for _, policy in seen)
    assert answer.session.execution.profile == profile


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
@pytest.mark.parametrize("restriction", ["filesystem", "mcp", "spending"])
async def test_resume_cannot_drop_additional_resource_requirements(mode, tmp_path, restriction):
    from agentkit.models.providers.cli import CliAdapter

    extra = {
        "filesystem": ExecutionRequirement.FILESYSTEM_READ,
        "mcp": ExecutionRequirement.MCP_ALLOWLIST,
        "spending": ExecutionRequirement.HARD_SPENDING_CAP,
    }[restriction]
    request = replace(
        requirements("cli_compatible"),
        required=requirements("cli_compatible").required | {extra},
        **({"read_roots": (tmp_path,)} if restriction == "filesystem" else {}),
    )
    session = SessionHandle(mode, "parent", execution=request)
    adapters = {} if mode.endswith("_api") else {mode: CliAdapter(spawn=lambda *a, **k: pytest.fail("CLI dispatched"))}
    async with CallRuntime(adapters=adapters) as runtime:
        answer = await runtime.run(
            Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential resolved"))),
            "resume",
            resume=session,
            execution=requirements("cli_compatible"),
            call_id="child",
        )
    assert answer.attempts == 0 and answer.failure.classification == "execution_incompatible"
    assert any(item["requirement"] == extra for item in answer.failure.mode_detail["incompatibilities"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_configuration_receipt_is_bound_to_the_profile(mode):
    from agentkit.models.providers.cli import CliAdapter

    adapters = {} if mode.endswith("_api") else {mode: CliAdapter(spawn=lambda *a, **k: None)}
    async with CallRuntime(adapters=adapters, native_tool_policy="deny") as runtime:
        strict = await runtime.evaluate_execution(binding(mode), execution=requirements("host_tools_only"))
        compatible = await runtime.evaluate_execution(binding(mode), execution=requirements("cli_compatible"))
    assert strict.configuration_id != compatible.configuration_id
    changed = compatible.require_configuration(strict.configuration_id)
    assert not changed.compatible
    assert changed.incompatibilities[0].code == "execution_configuration_changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
@pytest.mark.parametrize("profile", ["host_tools_only", "cli_compatible"])
async def test_profiles_do_not_make_api_native_fork_supported(mode, profile):
    session = SessionHandle(mode, "parent", execution=requirements(profile))
    async with CallRuntime() as runtime:
        with pytest.raises(ValueError, match="fork requires a resumable CLI session"):
            await runtime.run(
                Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential resolved"))),
                "fork",
                call_id="fork",
                resume=session,
                fork=True,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code"])
async def test_supported_observed_limit_keeps_profile_and_configuration_receipt(mode, scripted_client, scripted_cli):
    from agentkit.models.providers.cli import CliAdapter
    from agentkit.models.providers.frontiers import ApiAdapter
    from tests.examples.plugin_execution.wiring.runtime import compose_call
    from tests.source.unit.models.adapters.test_sdk_execution_assessment import _PRICES

    client = None
    if mode.endswith("_api"):
        client, _ = scripted_client(mode)
        adapter = ApiAdapter(client=client)
    else:
        adapter = CliAdapter(spawn=scripted_cli(mode).spawn)
    limits = CallLimits(max_cost_usd=Decimal("1"), max_turns=3)
    call = compose_call("observed-limit")
    try:
        async with CallRuntime(adapters={mode: adapter}, pricing=_PRICES) as runtime:
            answer = await runtime.run(
                binding(mode),
                call.prompt,
                call.tools,
                call_id=call.call_id,
                scope=call.scope,
                execution=requirements("cli_compatible"),
                limits=limits,
            )
    finally:
        if client is not None:
            await client.aclose()
    assert answer.outcome == "answered", answer.failure
    assert call.effects == ["record-1"]
    assert answer.session.execution.profile == "cli_compatible"
    assert ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD in answer.session.execution.required
    assert limits.max_cost_usd == Decimal("1") and answer.cost_usd is not None
