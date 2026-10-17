"""Regressions reproducible through the pre-existing public contract."""

import pytest

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models import Binding, CallLimits, NativeToolPolicy
from agentkit.models.providers.cli import CliAdapter


@pytest.mark.asyncio
async def test_local_codex_policy_rejection_has_typed_predispatch_reason():
    async def forbidden(*args, **kwargs):
        raise AssertionError("no dispatch permitted")

    async with CallRuntime(
        adapters={"codex": CliAdapter(spawn=forbidden)},
        native_tool_policy=NativeToolPolicy.DENY,
    ) as runtime:
        answer = await runtime.ask(Binding("codex", "test"), "q", limits=CallLimits(max_turns=1), call_id="strict")
    assert answer.failure.classification == "execution_incompatible"
    assert answer.failure.mode_detail["dispatch_state"] == "not_started"
    assert answer.cost_usd is None
    assert not answer.usage_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,config_var", [("claude_code", "CLAUDE_CONFIG_DIR"), ("codex", "CODEX_HOME")])
async def test_cli_home_and_xdg_paths_never_inherit_unrelated_host_files(mode, config_var, monkeypatch, tmp_path):
    from agentkit.core.testing.fakes import FakeClaudeCli, FakeCodexCli, codex_turn
    from agentkit.models import CredentialRef

    monkeypatch.setenv("HOME", str(tmp_path / "unrelated-home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "unrelated-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "unrelated-cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "unrelated-data"))
    if mode == "claude_code":
        cli = FakeClaudeCli.script(
            [{"type": "result", "subtype": "success", "result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}]
        )
        key = "ANTHROPIC_API_KEY"
    else:
        cli = FakeCodexCli.script(codex_turn(text="ok"))
        key = "OPENAI_API_KEY"
    async with CallRuntime(adapters={mode: CliAdapter(spawn=cli)}, unpriced_policy="proceed_unpriced") as runtime:
        await runtime.ask(
            Binding(mode, "test", CredentialRef(lambda: {key: "test-only"})),
            "q",
            limits=CallLimits(max_turns=1),
            call_id="home",
        )
    env = cli.invocations[0].env
    assert env["HOME"] == env[config_var]
    for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        assert env[name].startswith(env["HOME"] + "/")


@pytest.mark.asyncio
async def test_adapter_explicitly_disables_ambient_claude_skills():
    from agentkit.core.testing.fakes import FakeClaudeCli
    from agentkit.models import CredentialRef

    cli = FakeClaudeCli.script(
        [{"type": "result", "subtype": "success", "result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}]
    )
    async with CallRuntime(
        adapters={"claude_code": CliAdapter(spawn=cli)}, unpriced_policy="proceed_unpriced"
    ) as runtime:
        await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test-only"})),
            "q",
            limits=CallLimits(max_turns=1),
            call_id="no-skills",
        )
    assert "--disable-slash-commands" in cli.invocations[0].argv
