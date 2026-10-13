"""AgentRunner keeps model-call authority on the CallRuntime path."""

import asyncio
from decimal import Decimal

import pytest

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet
from agentkit.core.contracts.types import Mode, Scope
from agentkit.core.runtime.calls import CallRuntime
from agentkit.core.testing.fakes import CliRun, FakeClaudeCli, FakeCodexCli, codex_turn
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef, SessionHandle, Turn, Workspace
from agentkit.models.providers.cli.calls import CliAdapter
from agentkit.state.context import MemoryContext, SlidingWindowContextPolicy, WorkingContext
from agentkit.tools import FunctionTool, ToolRegistry
from agentkit.tools.execution.policy import ToolSet
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.asyncio
async def test_agent_runner_forwards_workspace_price_table_and_turn_slice(tmp_path):
    scope = Scope(org_id=2, domain_id=9)
    native_agents = {"reviewer": {"description": "Review code"}}
    adapter = ScriptedAdapter(Turn(("done",), raw_usage={"input_tokens": 10, "output_tokens": 2}))
    prices = PriceTable("runner-test", "2026-10-13", {"test": {
        "input_uncached": "2", "input_cache_read": "0", "output": "4",
    }})
    workspace = Workspace(working_dir=tmp_path)
    async with CallRuntime(
        adapters={"claude_code": adapter}, pricing=prices,
        native_tool_policy="warn", result_masker=lambda _name, text: text,
    ) as runtime:
        runtime_kwargs = {}
        original_run = runtime.run

        async def capture_run(*args, **kwargs):
            runtime_kwargs.update(kwargs)
            return await original_run(*args, **kwargs)

        runtime.run = capture_run
        runner = AgentRunner(runtime)
        answer = await runner.run_agent(
            AgentDefinition("worker", "1", "Run in the supplied workspace.", limits=CallLimits(max_turns=2)),
            InstructionSet("Return done."), {}, binding=Binding("claude_code", "test"),
            call_id="authority", workspace=workspace, pricing=prices, turns_per_slice=1,
            resume=SessionHandle(Mode.CLAUDE_CODE, "authority", cli_session_id="existing-session"),
            stable_prefix=True, fork=True, native_agents=native_agents, scope=scope,
            capture_text=True, native_tool_policy="deny",
        )
    assert answer.outcome == "answered"
    assert answer.cost_usd == Decimal("0.000028")
    _, _, kwargs = adapter.requests[0]
    assert kwargs["workspace"] == workspace
    assert kwargs["limits"].max_turns == 1
    assert runtime_kwargs["pricing"] == prices and runtime_kwargs["turns_per_slice"] == 1
    assert runtime_kwargs["stable_prefix"] is True and runtime_kwargs["fork"] is True
    assert runtime_kwargs["native_agents"] == native_agents and runtime_kwargs["scope"] == scope
    assert runtime_kwargs["capture_text"] is True
    assert runtime_kwargs["native_tool_policy"] == "deny"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.ANTHROPIC_API, Mode.OPENAI_API])
async def test_tool_calls_execute_concurrently_with_the_invocation_scope(mode):
    scope = Scope(org_id=17, domain_id=4)
    active = 0
    peak = 0
    observed_scopes = []

    async def operation(ctx) -> str:
        """Run a bounded operation under the current tenant scope."""
        nonlocal active, peak
        observed_scopes.append(ctx.scope)
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.03)
            return "ok"
        finally:
            active -= 1

    first = FunctionTool.from_callable(operation, name="first", side_effecting=False)
    second = FunctionTool.from_callable(operation, name="second", side_effecting=False)
    tool_set = ToolSet(ToolRegistry.from_tools([first, second]), max_concurrency=2)
    adapter = ScriptedAdapter(
        Turn(tool_calls=(
            {"id": "one", "name": "first", "arguments": {}},
            {"id": "two", "name": "second", "arguments": {}},
        ), raw_usage={"input_tokens": 1, "output_tokens": 1}),
        Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.run(
            Binding(mode, "test"), "Run both.", tool_set,
            call_id="parallel", scope=scope,
        )
    assert answer.outcome == "answered"
    assert peak == 2
    assert observed_scopes == [scope, scope]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.ANTHROPIC_API, Mode.OPENAI_API])
async def test_a_side_effecting_tool_turn_runs_sequentially(mode):
    active = 0
    peak = 0

    async def operation() -> str:
        """Track whether tools in a mixed turn overlap."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return "ok"
        finally:
            active -= 1

    pure = FunctionTool.from_callable(operation, name="pure", side_effecting=False)
    mutation = FunctionTool.from_callable(operation, name="mutation", side_effecting=True)
    tools = ToolSet(ToolRegistry.from_tools([pure, mutation]), max_concurrency=2)
    adapter = ScriptedAdapter(
        Turn(tool_calls=(
            {"id": "one", "name": "pure", "arguments": {}},
            {"id": "two", "name": "mutation", "arguments": {}},
        ), raw_usage={"input_tokens": 1, "output_tokens": 1}),
        Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.run(Binding(mode, "test"), "Run safely.", tools, call_id="serial-tools")
    assert answer.outcome == "answered"
    assert peak == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_cli_hosted_tools_receive_the_invocation_scope(mode, monkeypatch):
    scope = Scope(org_id=31, domain_id=6)
    observed = []

    async def inspect_scope(ctx) -> str:
        """Return the invocation's host-owned scope."""
        observed.append(ctx.scope)
        return f"{ctx.scope.org_id}:{ctx.scope.domain_id}"

    tools = ToolSet(ToolRegistry.from_tools([FunctionTool.from_callable(inspect_scope, side_effecting=False)]))

    class Server:
        timeout_s = 60
        auto_approve = ()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def check_ready(self):
            return None

        def cli_kwargs(self, **_):
            return {"tools": ("",)}

        def codex_kwargs(self, **_):
            return {}

    def fake_serve(tool_set, *, ctx, **_):
        observed.append(ctx.scope)
        return Server()

    import agentkit.tools.execution.policy as policy

    monkeypatch.setattr(policy, "serve", fake_serve)
    fake_cli = (
        FakeClaudeCli.script(
            [
                {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "done"}]}},
                {"type": "result", "subtype": "success", "result": "done", "usage": {"input_tokens": 1, "output_tokens": 1}},
            ]
        )
        if mode == Mode.CLAUDE_CODE
        else FakeCodexCli.script(codex_turn(text="done"))
    )
    credentials = (
        {"ANTHROPIC_API_KEY": "test"} if mode == Mode.CLAUDE_CODE else {"OPENAI_API_KEY": "test"}
    )
    events = []
    async with CallRuntime(adapters={mode: CliAdapter(spawn=fake_cli)}, pricing=PRICES) as runtime:
        answer = await runtime.run(
            Binding(mode, "test", CredentialRef(lambda: credentials)), "q", tools,
            call_id=f"cli-scope-{mode.value}", scope=scope, events=events.append,
        )
    assert answer.outcome == "answered", (answer.failure, events, fake_cli.invocations, observed)
    assert observed == [scope]


@pytest.mark.asyncio
async def test_raising_event_sink_isolated_and_answer_retained():
    adapter = ScriptedAdapter(Turn(("kept answer",), raw_usage={"input_tokens": 1, "output_tokens": 1}))
    calls = 0

    def broken_sink(event):
        nonlocal calls
        calls += 1
        raise RuntimeError("subscriber unavailable")

    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding("openai_api", "test"), "Answer.", events=broken_sink,
                                   call_id="subscriber-failure")
    assert answer.outcome == "answered"
    assert answer.text == "kept answer"
    assert calls == 3  # two failures trigger DEGRADED; the third disables a persistently failing sink
    assert answer.degradations[-1].setting == "event_sink"


@pytest.mark.asyncio
async def test_memory_provider_is_scoped_and_composed_as_untrusted_context():
    scope = Scope(org_id=8, domain_id=3)

    class Provider:
        def __init__(self):
            self.request = None

        async def retrieve(self, query, *, scope, limit):
            self.request = (query, scope, limit)
            return (MemoryContext("customer-records", "Tenant-specific fact."),)

    memory = Provider()
    adapter = ScriptedAdapter(Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}))
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, memory=memory, memory_limit=3)
        await runner.run_agent(
            AgentDefinition("worker", "1", "Use scoped recall."),
            InstructionSet("Find the relevant fact."), {}, binding=Binding("openai_api", "test"),
            call_id="memory-provider", scope=scope,
        )
    assert memory.request == ("Find the relevant fact.", scope, 3)
    assert "Untrusted memory context: source=customer-records" in adapter.requests[0][1].user
    assert "Tenant-specific fact." in adapter.requests[0][1].user


@pytest.mark.asyncio
async def test_context_policy_runs_between_tool_turns():
    observations = []

    def policy(context: WorkingContext) -> WorkingContext:
        observations.append(len(context.messages))
        context.note("policy_applied", True)
        return context

    async def operation() -> str:
        """Return a short operation result for the compaction boundary test."""
        return "ok"

    tool = FunctionTool.from_callable(operation, side_effecting=False)
    adapter = ScriptedAdapter(
        Turn(tool_calls=({"id": "one", "name": "operation", "arguments": {}},),
             raw_usage={"input_tokens": 1, "output_tokens": 1}),
        Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        runner = AgentRunner(
            runtime,
            tools=ToolRegistry.from_tools([tool]),
            context_policy=policy,
        )
        answer = await runner.run_agent(
            AgentDefinition("worker", "1", "Run the operation."), InstructionSet("Go."), {},
            binding=Binding("openai_api", "test"), call_id="context-policy",
        )
    assert answer.outcome == "answered"
    assert len(observations) == 1 and observations[0] >= 2
    assert dict(answer.context.scratchpad)["policy_applied"] is True


@pytest.mark.asyncio
async def test_cli_context_compaction_keeps_one_native_session_across_three_slices(tmp_path):
    from agentkit.models.contracts.types import CredentialRef

    texts = ("draft one", "draft two", "complete")
    native_session_id = "11111111-1111-4111-8111-111111111111"
    runs = [
        CliRun.of((
            {"type": "system", "subtype": "init", "session_id": native_session_id},
            {"type": "assistant", "message": {"id": f"message-{index}", "content": [{"type": "text", "text": text}]}},
            {"type": "result", "subtype": "success", "result": text,
             "usage": {"input_tokens": 1, "output_tokens": 1}},
        ))
        for index, text in enumerate(texts)
    ]
    cli = FakeClaudeCli(runs)
    policy_observations = []
    verifier_calls = 0

    def compact(context: WorkingContext) -> WorkingContext:
        policy_observations.append(len(context.messages))
        return SlidingWindowContextPolicy(max_turns=1)(context)

    def verify(answer) -> bool:
        nonlocal verifier_calls
        verifier_calls += 1
        return answer.text == "complete"

    async with CallRuntime(adapters={Mode.CLAUDE_CODE: CliAdapter(spawn=cli)}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, context_policy=compact)
        answer = await runner.run_agent(
            AgentDefinition("builder", "1", "Complete the task.", limits=CallLimits(max_turns=3)),
            InstructionSet("Build it.", verifier=verify), {},
            binding=Binding(Mode.CLAUDE_CODE, "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            call_id="compact-native-session", workspace=Workspace(working_dir=tmp_path),
        )

    assert answer.outcome == "answered" and answer.text == "complete"
    assert verifier_calls == 3 and len(policy_observations) == 2
    assert len(cli.invocations) == 3
    assert "--resume" not in cli.invocations[0].argv
    assert all("--resume" in inv.argv for inv in cli.invocations[1:]), [inv.argv for inv in cli.invocations]
    assert [inv.argv[inv.argv.index("--resume") + 1] for inv in cli.invocations[1:]] == [
        native_session_id, native_session_id,
    ]


def test_sliding_window_policy_is_deterministic_and_does_not_summarize():
    from agentkit.core.contracts.types import Message

    context = WorkingContext()
    context.extend((
        Message("user", "old question"), Message("assistant", "old answer"),
        Message("user", "new question"), Message("assistant", "new answer"),
    ))
    policy = SlidingWindowContextPolicy(max_turns=1)
    first = policy(context)
    second = policy(context)
    assert first.messages == second.messages
    assert [message.content for message in first.messages] == ["new question", "new answer"]
