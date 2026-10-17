"""Execution requests describe enforced scope, across assessment and dispatch."""

from dataclasses import replace

import httpx
import pytest

from agentkit.core.contracts.types import Mode, ToolSchema
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models import Binding, CredentialRef, ExecutionRequirement, ExecutionRequirements
from agentkit.models.contracts.types import Turn
from agentkit.models.providers.frontiers.calls import ApiAdapter
from agentkit.tools import ToolRegistry
from agentkit.tools.execution.policy import ToolSet
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter, sse
from tests.source.conformance.test_tool_policy_revocation import channel


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(Mode))
async def test_requested_hard_cap_refuses_before_credentials_for_every_mode(mode):
    def forbidden():
        pytest.fail("must not resolve credentials for an unsupported hard cap")

    async with CallRuntime() as runtime:
        result = await runtime.ask(
            Binding(mode, "test", CredentialRef(forbidden)),
            "q",
            call_id="hard-cap",
            execution=ExecutionRequirements(required=frozenset({ExecutionRequirement.HARD_SPENDING_CAP})),
        )
    assert result.failure.classification == "execution_incompatible" and result.attempts == 0
    assert result.cost_usd is None and not result.usage_complete


@pytest.mark.asyncio
async def test_existing_codex_cost_limit_refusal_has_honest_local_accounting():
    from decimal import Decimal

    from agentkit.core.testing.fakes import FakeCodexCli
    from agentkit.models import CallLimits
    from agentkit.models.providers.cli import CliAdapter

    cli = FakeCodexCli.script([])
    async with CallRuntime(adapters={"codex": CliAdapter(spawn=cli)}, pricing=PRICES) as runtime:
        result = await runtime.ask(
            Binding("codex", "test"), "q", call_id="cost-limit", limits=CallLimits(max_cost_usd=Decimal("1"))
        )
    assert result.failure.classification == "invalid_request"
    assert result.failure.mode_detail["dispatch_state"] == "not_started"
    assert result.attempts == 0 and not cli.invocations
    assert result.cost_usd is None and not result.usage_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_confirmed_api_http_failure_keeps_provider_classification(mode):
    from agentkit.models.providers.frontiers.base import ProviderError

    class Transport:
        async def _stream_events(self, path, payload):
            raise ProviderError("private-body", status_code=500, body='{"error":{"type":"api_error"}}')
            yield  # pragma: no cover

    adapter = ApiAdapter()
    result = await (adapter._openai if mode == Mode.OPENAI_API else adapter._anthropic)(
        Transport(), {}, lambda event: None
    )
    assert result.failure.classification == "transient_network" and result.failure.retryable
    assert "private-body" not in repr(result.failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(Mode))
async def test_unknown_adapter_cannot_satisfy_implicit_strict_native_denial(mode):
    adapter = ScriptedAdapter(Turn(("must not run",)))
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES, native_tool_policy="deny") as runtime:
        report = await runtime.evaluate_execution(Binding(mode, "test"))
        assert not report.compatible
        result = await runtime.ask(Binding(mode, "test"), "q", call_id="unknown-adapter")
    assert not adapter.requests and result.attempts == 0
    assert result.failure.classification == "execution_incompatible"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_api_host_tool_loop_typed_answer_grants_validation_approval_and_accounting(mode):
    import json

    from pydantic import BaseModel

    from agentkit.tools import tool

    class Answer(BaseModel):
        status: str

    effects, requests, events = [], [], []

    @tool(side_effecting=True)
    def granted(count: int, ctx) -> str:
        """Commit the count under the host's current invocation authority."""
        effects.append((count, ctx.tool_invocation))
        return "committed"

    @tool(side_effecting=True, requires_approval=True)
    def unapproved() -> str:
        """Commit only with host approval, which this call has not granted."""
        pytest.fail("unapproved tool caused an effect")

    def respond(request):
        requests.append(json.loads(request.content))
        calls = (
            [("granted", {"count": 2}), ("granted", {"count": "invalid"}), ("unknown", {}), ("unapproved", {})]
            if len(requests) == 1
            else []
        )
        if mode == Mode.OPENAI_API:
            output = [
                {"type": "function_call", "call_id": f"provider-{i}", "name": name, "arguments": json.dumps(args)}
                for i, (name, args) in enumerate(calls)
            ]
            output = output or [{"type": "message", "content": [{"type": "output_text", "text": '{"status":"ok"}'}]}]
            wire = [
                {
                    "type": "response.completed",
                    "response": {
                        "output": output,
                        "usage": {"input_tokens": 10, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 2}},
                    },
                }
            ]
        else:
            blocks = [
                {"type": "tool_use", "id": f"provider-{i}", "name": name, "input": args}
                for i, (name, args) in enumerate(calls)
            ] or [{"type": "text", "text": '{"status":"ok"}'}]
            wire = [
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 1,
                            "cache_read_input_tokens": 2,
                            "cache_creation_input_tokens": 7,
                            "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 4},
                        }
                    },
                }
            ]
            for i, block in enumerate(blocks):
                wire.extend(
                    [
                        {"type": "content_block_start", "index": i, "content_block": block},
                        {"type": "content_block_stop", "index": i},
                    ]
                )
            wire.extend(
                [
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use" if calls else "end_turn"},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                ]
            )
        return httpx.Response(200, content=sse(wire))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with CallRuntime(
            adapters={mode: ApiAdapter(client=client)}, pricing=PRICES, native_tool_policy="deny"
        ) as runtime:
            result = await runtime.run(
                Binding(
                    mode,
                    "test",
                    CredentialRef(
                        lambda: {
                            "OPENAI_API_KEY" if mode == Mode.OPENAI_API else "ANTHROPIC_API_KEY": "test-only",
                        }
                    ),
                ),
                "q",
                ToolSet(ToolRegistry.from_tools([granted, unapproved])),
                output=Answer,
                call_id="host-loop",
                events=events.append,
                execution=ExecutionRequirements(
                    required=frozenset(
                        {
                            ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE,
                            ExecutionRequirement.HOST_TOOLS,
                            ExecutionRequirement.SCHEMA,
                            ExecutionRequirement.USAGE,
                        }
                    )
                ),
            )
    assert result.value == Answer(status="ok") and result.outcome == "answered"
    assert len(effects) == 1 and effects[0][0] == 2
    assert effects[0][1].provider_tool_call_id == "provider-0"
    assert len(requests) == result.attempts == 2
    assert result.usage_complete and result.usage.output == 2 and result.usage.input_cache_read == 4
    if mode == Mode.ANTHROPIC_API:
        assert result.usage.input_cache_write_5m == 6 and result.usage.input_cache_write_1h == 8
    assert result.cost_usd is not None
    assert [
        (event.kind, event.fields["attempt"])
        for event in events
        if event.kind in ("attempt_started", "attempt_finished")
    ] == [("attempt_started", 1), ("attempt_finished", 1), ("attempt_started", 2), ("attempt_finished", 2)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_backend_receives_runtime_attempt_and_execution_contract(mode):
    from agentkit.core.testing.fakes import FakeClaudeCli, FakeCodexCli, codex_turn
    from agentkit.models.providers.cli import BackendReport, CliAdapter

    spawn = (
        FakeClaudeCli.script(
            [
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "done",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ]
        )
        if mode == Mode.CLAUDE_CODE
        else FakeCodexCli.script(codex_turn(text="done", usage=(1, 0, 1)))
    )
    launches = []

    class Backend:
        async def inspect(self):
            return BackendReport("test", "1")

        async def start(self, invocation):
            launches.append(invocation)
            return await spawn(*invocation.argv, **invocation.options)

    request = ExecutionRequirements(required=frozenset({ExecutionRequirement.HOST_TOOLS}))
    async with CallRuntime(adapters={mode: CliAdapter(execution_backend=Backend())}, pricing=PRICES) as runtime:
        result = await runtime.ask(
            Binding(
                mode,
                "test",
                CredentialRef(
                    lambda: {
                        "ANTHROPIC_API_KEY" if mode == Mode.CLAUDE_CODE else "OPENAI_API_KEY": "test-only",
                    }
                ),
            ),
            "q",
            call_id="backend-attempt",
            execution=request,
        )
    assert result.outcome == "answered" and len(launches) == 1
    invocation = launches[0]
    assert invocation.execution == request and invocation.native_tool_policy == "warn"
    assert invocation.mode == mode and invocation.call_id == "backend-attempt" and invocation.attempt == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["deny", "contained"])
async def test_direct_subprocess_backend_cannot_silently_accept_protected_invocation(policy, tmp_path):
    import sys

    from agentkit.models import NativeExecution
    from agentkit.models.providers.cli import CliInvocation, SubprocessExecutionBackend

    marker = tmp_path / "must-not-run"
    invocation = CliInvocation(
        (sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)),
        {"env": {}, "start_new_session": True},
        execution=ExecutionRequirements(native_execution=NativeExecution.CONTAINED)
        if policy == "contained"
        else ExecutionRequirements(),
        native_tool_policy="deny" if policy == "deny" else "allow",
    )
    with pytest.raises(ValueError, match="cannot establish"):
        await SubprocessExecutionBackend().start(invocation)
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(Mode))
async def test_effective_execution_policy_identity_is_recorded_before_attempt_start(mode):
    from tests.source.conformance.test_calls import real_adapter

    events = []
    async with CallRuntime(adapters={mode: real_adapter(mode, "done")}, pricing=PRICES) as runtime:
        result = await runtime.ask(
            Binding(
                mode,
                "test",
                CredentialRef(
                    lambda: {
                        "ANTHROPIC_API_KEY"
                        if mode in (Mode.CLAUDE_CODE, Mode.ANTHROPIC_API)
                        else "OPENAI_API_KEY": "test-only",
                    }
                ),
            ),
            "q",
            call_id="policy",
            events=events.append,
        )
    assert result.outcome == "answered"
    policy = [event for event in events if event.fields.get("mode_event") == "execution_policy"]
    assert len(policy) == 1
    assert policy[0].fields["adapter"] and policy[0].fields["backend"]
    assert policy[0].fields["native_tool_policy"] == "warn"
    assert events.index(policy[0]) < next(i for i, event in enumerate(events) if event.kind == "attempt_started")


def test_public_tool_invocation_identity_is_typed_and_task_local():
    from agentkit.tools import ToolInvocation, current_tool_invocation

    identity = ToolInvocation("host-1", provider_tool_call_id=None, transport_request_id=7)
    assert identity.provider_tool_call_id is None and identity.transport_request_id == 7
    assert current_tool_invocation() is None


def test_resumed_authority_intersects_roots_and_destinations_without_dropping_requirements():
    from agentkit.models import NativeExecution

    original = ExecutionRequirements(
        native_execution=NativeExecution.CONTAINED,
        read_roots=("/jobs/one",),
        write_roots=("/jobs/one/output",),
        network_destinations=("api.example",),
    )
    requested = ExecutionRequirements(
        read_roots=("/jobs/one/data", "/jobs/two"),
        write_roots=("/jobs",),
        network_destinations=("api.example", "private.local"),
    )
    result = requested.constrain(original)
    assert result.native_execution == NativeExecution.CONTAINED
    assert result.required == original.required
    assert tuple(map(str, result.read_roots)) == ("/jobs/one/data",)
    assert tuple(map(str, result.write_roots)) == ("/jobs/one/output",)
    assert result.network_destinations == ("api.example",)


@pytest.mark.asyncio
async def test_backend_self_declaration_cannot_promote_codex_native_absence_to_unverified():
    from agentkit.models.providers.cli import BackendReport, CliAdapter

    class Declared:
        async def inspect(self):
            return BackendReport("self-declared", "1", frozenset(ExecutionRequirement))

    report = await CliAdapter(execution_backend=Declared(), native_tool_policy="deny").evaluate_execution(
        Binding(Mode.CODEX, "test")
    )
    fact = next(fact for fact in report.assessments if fact.requirement == "native_tools_unavailable")
    assert fact.status == "unsupported" and not report.compatible


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_unrestricted_api_request_hook_cannot_attest_native_tool_absence(mode):
    def forbidden():
        pytest.fail("capability refusal must precede credential resolution")

    def expand(payload):
        payload["tools"] = [{"type": "web_search"}]
        return payload

    async with CallRuntime(
        adapters={mode: ApiAdapter(request_hook=expand)}, native_tool_policy="deny", pricing=PRICES
    ) as runtime:
        report = await runtime.evaluate_execution(Binding(mode, "test"))
        assert not report.compatible
        assert report.incompatibilities[0].code == "request_hook_authority_unverified"
        result = await runtime.ask(Binding(mode, "test", CredentialRef(forbidden)), "q", call_id="hook")
    assert result.attempts == 0 and result.failure.classification == "execution_incompatible"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(Mode))
async def test_runner_keeps_execution_requirements_on_the_runtime_path(mode):
    from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet

    adapter = ScriptedAdapter()
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        result = await AgentRunner(runtime).run_agent(
            AgentDefinition("worker", "1", "Use only the requested execution authority."),
            InstructionSet("Complete the task."),
            {},
            binding=Binding(mode, "test"),
            call_id="runner",
            execution=ExecutionRequirements(read_roots=()),
        )
    assert result.failure.classification == "execution_incompatible"
    assert result.attempts == 0 and not adapter.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_cli_missing_usage_is_not_manufactured_as_measured_zero(mode):
    from agentkit.core.testing.fakes import FakeClaudeCli, FakeCodexCli, codex_turn
    from agentkit.models.providers.cli import CliAdapter

    if mode == Mode.CLAUDE_CODE:
        spawn = FakeClaudeCli.script([{"type": "result", "subtype": "success", "result": "done"}])
    else:
        spawn = FakeCodexCli.script(codex_turn(text="done"))
    async with CallRuntime(
        adapters={mode: CliAdapter(spawn=spawn)}, pricing=PRICES, unpriced_policy="proceed_unpriced"
    ) as runtime:
        result = await runtime.ask(
            Binding(
                mode,
                "test",
                CredentialRef(
                    lambda: {
                        "ANTHROPIC_API_KEY" if mode == Mode.CLAUDE_CODE else "OPENAI_API_KEY": "test-only",
                    }
                ),
            ),
            "q",
            call_id="missing-usage",
        )
    assert result.outcome == "answered"
    assert not result.usage_complete and result.cost_usd is None
    assert result.raw_usage == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_api_tool_handler_receives_provider_identity_and_trusted_call_scope(mode):
    from agentkit.core.contracts.types import Scope
    from agentkit.tools import tool

    seen = []

    @tool(side_effecting=False)
    def inspect_invocation(ctx) -> str:
        """Record the trusted invocation scope and tool identity."""
        seen.append((ctx.scope, ctx.correlation_id, ctx.tool_invocation))
        return "ok"

    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "provider-call-7", "name": "inspect_invocation", "arguments": {}},),
            raw_usage={"input_tokens": 1, "output_tokens": 1},
        ),
        Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    scope = Scope(org_id=91)
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        result = await runtime.run(
            Binding(mode, "test"),
            "q",
            ToolSet(ToolRegistry.from_tools([inspect_invocation])),
            scope=scope,
            call_id="host-call",
        )
    assert result.outcome == "answered" and len(seen) == 1
    assert seen[0][:2] == (scope, "host-call")
    identity = seen[0][2]
    assert identity.provider_tool_call_id == "provider-call-7"
    assert identity.host_invocation_id and identity.transport_request_id is None


@pytest.mark.asyncio
async def test_mcp_identity_is_preserved_without_manufacturing_missing_provider_id():
    from mcp.shared.memory import create_connected_server_and_client_session

    from agentkit.core.testing import make_test_ctx
    from agentkit.tools import tool
    from agentkit.tools.execution.policy import serve

    seen = []

    @tool(side_effecting=False)
    def inspect_invocation(ctx) -> str:
        """Inspect the call association without trusting client arguments."""
        seen.append((ctx.scope, ctx.tool_invocation))
        return "ok"

    ctx = make_test_ctx()
    spec = serve(ToolSet(ToolRegistry.from_tools([inspect_invocation])), ctx=ctx)
    try:
        async with create_connected_server_and_client_session(spec.build_server()) as client:
            for meta in (
                {"claudecode/toolUseId": "toolu-actual", "scope": {"org_id": 999}},
                {"callId": "call-actual"},
                {},
                {"claudecode/toolUseId": {"malformed": True}},
                {"claudecode/toolUseId": "one", "callId": "two"},
            ):
                result = await client.call_tool("inspect_invocation", {}, meta=meta)
                assert not result.isError
    finally:
        await spec.stop()
    assert len(seen) == 5 and all(scope == ctx.scope for scope, _ in seen)
    assert [identity.provider_tool_call_id for _, identity in seen] == ["toolu-actual", "call-actual", None, None, None]
    assert all(identity.transport_request_id is not None for _, identity in seen)
    assert len({identity.host_invocation_id for _, identity in seen}) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_api_assessment_has_full_matrix_and_explicit_host_callback_scope(mode):
    request = ExecutionRequirements(
        required=frozenset(
            {
                ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE,
                ExecutionRequirement.HOST_TOOLS,
                ExecutionRequirement.SCHEMA,
                ExecutionRequirement.USAGE,
            }
        )
    )
    async with CallRuntime(native_tool_policy="deny") as runtime:
        report = await runtime.evaluate_execution(Binding(mode, "test"), execution=request)
    assert report.compatible
    assert {fact.requirement for fact in report.assessments} == set(ExecutionRequirement)
    host = next(fact for fact in report.assessments if fact.requirement == "host_tools")
    assert "callback" in host.explanation and "sandbox" in host.explanation
    assert report.adapter.endswith("ApiAdapter")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_api_supported_requirements_reach_dispatch_and_boundary_requests_do_not(mode):
    requests = []
    events = (
        [
            {
                "type": "response.completed",
                "response": {
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        ]
        if mode == Mode.OPENAI_API
        else [
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "done"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
    )

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=sse(events))

    credentials = []

    def resolve():
        credentials.append(True)
        return {"OPENAI_API_KEY" if mode == Mode.OPENAI_API else "ANTHROPIC_API_KEY": "test-only"}

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with CallRuntime(
            adapters={mode: ApiAdapter(client=client)}, pricing=PRICES, native_tool_policy="deny"
        ) as runtime:
            binding = Binding(mode, "test", CredentialRef(resolve))
            answer = await runtime.ask(
                binding,
                "q",
                call_id="allowed",
                execution=ExecutionRequirements(required=frozenset({ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE})),
            )
            assert answer.outcome == "answered" and answer.attempts == 1
            rejected = await runtime.ask(
                binding, "q", call_id="blocked", execution=ExecutionRequirements(read_roots=())
            )
    assert len(requests) == len(credentials) == 1
    assert rejected.failure.classification == "execution_incompatible"
    assert rejected.attempts == 0 and rejected.cost_usd is None and not rejected.usage_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["api_direct", "api_dispatcher", "mcp_direct", "mcp_dispatcher"])
async def test_toolset_validates_declared_json_schema_before_custom_tool_effects(route):
    effects = []

    class Write:
        name = "write"
        description = "Write a bounded count of records."
        schema = ToolSchema(
            name,
            description,
            {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["count"],
                "additionalProperties": False,
            },
        )
        output_schema = None
        side_effecting = True
        requires_approval = False

        async def run(self, args, ctx):
            effects.append(args["count"])
            return "written"

    tools = ToolSet(ToolRegistry.from_tools([Write()]), deferred=("write",))
    async with channel(tools, route) as execute:
        await execute("search_tools", {"query": "write"})
        import json

        for args, error in [
            ({"count": 2}, False),
            ({"count": 4}, True),
            ({"count": "2"}, True),
            ({"count": 2, "extra": True}, True),
        ]:
            result = await execute(
                "call_loaded_tool" if route.endswith("dispatcher") else "write",
                {"name": "write", "arguments_json": json.dumps(args)} if route.endswith("dispatcher") else args,
            )
            assert (result.isError if route.startswith("mcp") else result.is_error) is error
    assert effects == [2]


@pytest.mark.asyncio
async def test_resume_inherits_execution_requirements_before_credential_resolution():
    adapter = ScriptedAdapter(Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}))
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        first = await runtime.ask(Binding("openai_api", "test"), "q", call_id="first")
        # A portable session carries the host's ceiling even if the next call
        # omits execution=. The adapter cannot silently regain filesystem access.
        session = replace(first.session, execution=ExecutionRequirements(read_roots=()))
        answer = await runtime.ask(Binding("openai_api", "test"), "q", call_id="resume", resume=session)
    assert answer.failure.classification == "execution_incompatible"
    assert answer.attempts == 0 and len(adapter.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
async def test_api_partial_stream_disconnect_is_uncertain_not_retried(mode):
    class Broken(ApiAdapter):
        async def _openai(self, transport, payload, emit):
            return await super()._openai(transport, payload, emit)

    class Transport:
        async def _stream_events(self, path, payload):
            yield (
                {"type": "response.output_text.delta", "delta": "partial"}
                if mode == Mode.OPENAI_API
                else {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}
            )
            raise ConnectionError("planted-private-diagnostic")

    adapter = Broken()
    turn = await (adapter._openai if mode == Mode.OPENAI_API else adapter._anthropic)(
        Transport(), {}, lambda event: None
    )
    assert turn.failure.classification == "execution_uncertain"
    assert not turn.failure.retryable and not turn.usage_complete
    assert "planted-private-diagnostic" not in repr(turn.failure)
    if mode == Mode.ANTHROPIC_API:
        assert turn.raw_usage["input_tokens"] == 9
