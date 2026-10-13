from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from agentkit.agents.definitions.agent import AgentDefinition, ToolGrant
from agentkit.agents.definitions.instructions import InstructionSet
from agentkit.agents.definitions.store import MemoryAgentStore
from agentkit.agents.execution.runner import AgentRunner
from agentkit.agents.minds import ChainOfThoughtMind, Mind, ReActMind, SingleCallMind
from agentkit.agents.prompts import Prompt
from agentkit.agents.prompts.composition import PromptPart, compose, digest
from agentkit.agents.skills.resources import FileSkillSource, SkillRef
from agentkit.core.contracts.types import Scope
from agentkit.core.runtime import RunContext
from agentkit.core.runtime.calls import CallRuntime
from agentkit.integrations.mcp.gateway import McpGateway, McpPolicy, ServerConfig
from agentkit.models.contracts.failures import classify, retry_after
from agentkit.models.contracts.settings import CapabilityTable
from agentkit.models.contracts.types import Binding, CallLimits, ModelSettings, Turn
from agentkit.models.providers.auth import anthropic_credential
from agentkit.models.providers.cli.isolation import IsolatedHome
from agentkit.tools import ToolRegistry
from agentkit.tools.execution.policy import OverflowPolicy, ToolResult, ToolSet
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter


@pytest.mark.parametrize(
    "values,kind",
    [
        ({"ANTHROPIC_API_KEY": "sk-ant-api-test"}, "api_key"),
        ({"ANTHROPIC_API_KEY": "sk-ant-oat01-test"}, "bearer"),
        ({"ANTHROPIC_AUTH_TOKEN": "rotating", "ANTHROPIC_API_KEY": "ignored"}, "bearer"),
    ],
)
def test_auth_kinds_never_put_oauth_in_api_key_header(values, kind):
    auth = anthropic_credential(values)
    assert auth.kind == kind
    assert ("Authorization" in auth.headers()) == (kind == "bearer")
    assert ("ANTHROPIC_AUTH_TOKEN" in auth.cli_env()) == (kind == "bearer")
    assert auth.secret not in repr(auth)
    assert anthropic_credential({}) is None


@pytest.mark.parametrize(
    "event,expected,retry",
    [
        (
            {"status": 429, "error": {"code": "insufficient_quota", "type": "rate_limit_error"}},
            "quota_exhausted",
            False,
        ),
        (
            {"status": 429, "error": {"details": {"error_code": "enforced_spend_limit_reached"}}},
            "quota_exhausted",
            False,
        ),
        ({"status": 429, "headers": {"Retry-After": "2"}}, "rate_limited", True),
        ({"error": {"type": "overloaded_error"}}, "overloaded", True),
        ({"status": 401}, "auth", False),
        ({"status": 400}, "invalid_request", False),
        ({"status": 500}, "transient_network", True),
        ({"status": 504}, "timeout", True),
        ({"error": {"code": "context_length_exceeded"}}, "context_exceeded", False),
        ({"subtype": "refusal"}, "refused", False),
        ({"subtype": "cancelled"}, "cancelled", False),
    ],
)
def test_failure_codes_override_http_status(event, expected, retry):
    failure = classify(event)
    assert failure.classification == expected and failure.retryable is retry


def test_retry_after_formats_and_structural_settings():
    assert retry_after("2.5") == 2.5
    assert retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0
    assert retry_after("unknown") is None
    caps = CapabilityTable("1", "today", {"codex": {"effort": ["low", "high"]}})
    effective = caps.check(Binding("codex", "test"), ModelSettings(effort="max", tier="fast"))
    assert effective.settings.effort == "high" and effective.settings.tier is None
    assert len(effective.degradations) == 2
    with pytest.raises(ValueError):
        caps.check(Binding("codex", "test"), ModelSettings(top_p=0.5))


def test_composition_is_stable_strict_and_keeps_untrusted_skills_out_of_system():
    parts = [
        PromptPart(Prompt("task", "1", "Do {goal}", inputs=("goal",))),
        PromptPart(Prompt("rules", "1", "Engine rules"), "rules", True, True),
        PromptPart(Prompt("skill", "1", "Untrusted persona"), "persona", True),
    ]
    first = compose(parts, {"goal": "work"})
    second = compose(parts, {"goal": "other work"})
    assert first.system == "Engine rules" and "Untrusted persona" in first.user
    assert first.prefix_digest == second.prefix_digest and first.manifest.digest != second.manifest.digest
    with pytest.raises(ValueError):
        compose(parts, {})
    with pytest.raises(ValueError):
        compose(parts, {"goal": "work"}, max_chars=2)
    with pytest.raises(ValueError):
        compose([PromptPart(Prompt("rules", "1", "evil"), "rules")], {})


@pytest.mark.asyncio
async def test_skill_manifest_revocation_and_path_boundaries(tmp_path):
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: demo\ndescription: A useful skill.\n---\nDo useful work.")
    (directory / "notes.txt").write_text("first")
    source = FileSkillSource(tmp_path)
    doc = await source.get(SkillRef("filesystem", "demo"))
    assert await source.read(doc.ref, "notes.txt") == "first"
    assert (await source.list())[0].ref.digest == doc.ref.digest
    with pytest.raises(PermissionError):
        await source.read(doc.ref, "../outside")
    (directory / "notes.txt").write_text("changed")
    with pytest.raises(PermissionError, match="changed"):
        await source.get(doc.ref)
    with pytest.raises(PermissionError):
        await source.get(SkillRef("filesystem", "../escape"))


@pytest.mark.asyncio
async def test_tool_policy_is_shared_for_error_overflow_images_and_deferred_loading(tmp_path):
    async def long_result() -> str:
        """Read a large result that is spilled to a file when it exceeds the configured context bound."""
        return "x" * 2000

    async def bad_result() -> str:
        """Fetch a failing result, whose execution error must be returned to the model for recovery."""
        raise ValueError("try another input")

    tools = ToolSet(
        ToolRegistry.from_tools([long_result, bad_result]), OverflowPolicy(256, tmp_path), deferred=("long_result",)
    )
    ctx = RunContext(correlation_id="test", scope=Scope())
    assert "long_result" not in [s.name for s in tools.schemas()]
    refused = await tools.execute("long_result", {}, ctx)
    assert refused.is_error
    await tools.execute("search_tools", {"query": "large"}, ctx)
    existing_spills = set(tmp_path.glob("*.json"))
    spilled = await tools.execute("long_result", {}, ctx)
    assert spilled.overflowed and len(set(tmp_path.glob("*.json")) - existing_spills) == 1
    assert (await tools.execute("bad_result", {}, ctx)).is_error
    image = ToolResult(({"type": "image", "data": "eA==", "mimeType": "image/png"},))
    assert (await tools.handle("image", image)).content[0]["type"] == "image"
    tools.images = False
    with pytest.raises(ValueError):
        await tools.handle("image", image)


@pytest.mark.asyncio
async def test_gateway_pins_raw_definitions_and_deny_wins():
    definition = {"name": "read", "description": "external read", "inputSchema": {"type": "object"}}

    class Client:
        async def list_tools(self):
            return [definition]

        async def call_tool(self, name, args):
            return {"content": [{"type": "text", "text": "ok"}]}

    config = ServerConfig("display-name", url="https://public.example/mcp")
    policy = McpPolicy(allow=("https://public.example/mcp#*",), approved={"read": digest(definition)})
    async with McpGateway(client_factory=lambda _: Client()) as gateway:
        connected = await gateway.connect(config, policy)
        tools = gateway.tools()
        name = tools.registry.names()[0]
        assert (await tools.execute(name, {}, RunContext(correlation_id="test", scope=Scope()))).text == "ok"
        definition["description"] = "changed"
        await connected.refresh()
        assert not gateway.tools().registry.names()
        assert (await tools.execute(name, {}, RunContext(correlation_id="test", scope=Scope()))).is_error
        connected.policy = replace(policy, deny=("https://public.example/mcp#*",))
        assert not gateway.tools(approved_only=False).registry.names()


def test_owned_home_lock_excludes_live_roots_and_sweeps_orphans(tmp_path):
    owner = IsolatedHome(tmp_path)
    live_root = owner.root
    home = owner.create()
    (home / "auth.json").write_text("secret")
    IsolatedHome.sweep(tmp_path)
    assert live_root.exists()
    owner.close()
    owner.close()
    assert not live_root.exists()
    orphan = tmp_path / "owner-orphan"
    orphan.mkdir()
    (orphan / "owner.lock").touch()
    (orphan / "auth.json").write_text("secret")
    IsolatedHome.sweep(tmp_path)
    assert not orphan.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mind", [SingleCallMind(), ReActMind(), ChainOfThoughtMind()])
@pytest.mark.parametrize("mode", ["claude_code", "codex", "anthropic_api", "openai_api"])
async def test_mind_runs_on_every_mode_without_transport_knowledge(mind, mode):
    assert isinstance(mind, Mind)
    adapter = ScriptedAdapter(Turn(("42",)))
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES)
    result = await mind.run(runtime, Binding(mode, "test"), "6 times 7", call_id="mind")
    assert result.outcome == "answered" and result.text == "42"
    if mind.name == "chain_of_thought":
        assert adapter.requests[0][1].manifest.parts[-1][0] == "mind.chain_of_thought"


@pytest.mark.asyncio
async def test_agent_versions_grants_and_done_verifier():
    assert ToolGrant(("*",), ("write*",)).permits("read")
    assert not ToolGrant(("*",), ("write*",)).permits("write")
    store = MemoryAgentStore()
    definition = AgentDefinition("agent", "1", "description", Binding("openai_api", "test"), mind="chain_of_thought")
    await store.put(definition)
    assert await store.get("agent@1") == definition
    with pytest.raises(ValueError):
        await store.put(replace(definition, description="changed"))
    runtime = CallRuntime(adapters={"openai_api": ScriptedAdapter(Turn(("wrong",)))}, pricing=PRICES)
    runner = AgentRunner(runtime)
    answer = await runner.run_agent(
        replace(definition, limits=CallLimits(max_turns=1)),
        InstructionSet("task", verifier=lambda _: False),
        {},
        call_id="agent",
    )
    assert answer.outcome == "limit_reached"


@pytest.mark.asyncio
async def test_limit_and_cache_invalidation_keep_resume_history():
    adapter = ScriptedAdapter(
        Turn(
            ("first",),
            raw_usage={"input_tokens": 10, "output_tokens": 5},
            history=({"role": "assistant", "content": "first"},),
        ),
        Turn(("second",)),
    )
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES)
    first = await runtime.ask(
        Binding("openai_api", "test"), "first", limits=CallLimits(max_cost_usd=Decimal("0.00001")), call_id="one"
    )
    assert first.outcome == "limit_reached" and first.session is not None
    events = []
    second = await runtime.ask(
        Binding("openai_api", "test"), "second", resume=first.session, events=events.append, call_id="two"
    )
    assert second.outcome == "answered"
    assert adapter.requests[1][2]["session"].history[0]["content"] == "first"
