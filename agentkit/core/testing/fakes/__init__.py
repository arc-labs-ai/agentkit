"""Test doubles — fakes that satisfy port Protocols with deterministic
canned behavior for unit + integration tests.

Two exceptions, and they are deliberate: `FakeClaudeCli` and
`FakeCodexCli` fake a SUBPROCESS, not a port, because the CLI cognitions
have no port to stand behind — they spawn the `claude` / `codex` binary
and parse its JSON stdout. Each double replaces the spawn and nothing
above it, so a test still runs the real parser, the real event mapping
and the real budget charge. They share their value types (`CliRun`,
`CliStderr`, `CliInvocation`) because a recorded spawn is a recorded
spawn whichever binary produced it.

Each fake records calls (where useful) so tests can assert on what the
agent sent. All exports here are re-exported from `agentkit.core.testing`
itself, so the user-facing import path is:

    from agentkit.core.testing import FakeLLM, FakeCtx, FakeGrounder, ...

The submodule path (`agentkit.core.testing.fakes.X`) is also valid for
callers who prefer per-module imports."""

from agentkit.core.testing.fakes.claude_cli import CliInvocation, CliRun, CliStderr, FakeClaudeCli
from agentkit.core.testing.fakes.clock import FakeClock
from agentkit.core.testing.fakes.codex_cli import FakeCodexCli, codex_turn
from agentkit.core.testing.fakes.compactor import FakeCompactor
from agentkit.core.testing.fakes.ctx import FakeCtx
from agentkit.core.testing.fakes.fetch import FakeFetch
from agentkit.core.testing.fakes.grounder import FakeGrounder
from agentkit.core.testing.fakes.llm import FakeLLM, ScriptExhausted, Turn
from agentkit.core.testing.fakes.memory import FakeMemory
from agentkit.core.testing.fakes.search import FakeSearch
from agentkit.core.testing.fakes.tool import FakeTool
from agentkit.core.testing.fakes.tracer import RecordingSpan, RecordingTracer

__all__ = [
    "CliInvocation",
    "CliRun",
    "CliStderr",
    "FakeClaudeCli",
    "FakeClock",
    "FakeCodexCli",
    "FakeCompactor",
    "FakeCtx",
    "FakeFetch",
    "FakeGrounder",
    "FakeLLM",
    "FakeMemory",
    "FakeSearch",
    "FakeTool",
    "RecordingSpan",
    "RecordingTracer",
    "ScriptExhausted",
    "Turn",
    "codex_turn",
]
