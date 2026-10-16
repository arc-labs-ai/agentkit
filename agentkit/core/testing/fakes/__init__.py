"""Fakes used by source tests and downstream harness conformance suites."""

from agentkit.core.testing.fakes.claude_cli import CliRun, FakeClaudeCli
from agentkit.core.testing.fakes.clock import FakeClock
from agentkit.core.testing.fakes.codex_cli import FakeCodexCli, codex_turn
from agentkit.core.testing.fakes.llm import FakeLLM, FakeTurn, ScriptExhausted
from agentkit.core.testing.fakes.tool import FakeTool

__all__ = [
    "CliRun", "FakeClaudeCli", "FakeClock", "FakeCodexCli", "FakeLLM", "FakeTool",
    "FakeTurn", "ScriptExhausted", "codex_turn",
]
