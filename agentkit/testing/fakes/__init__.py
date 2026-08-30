"""Test doubles — fakes that satisfy port Protocols with deterministic
canned behavior for unit + integration tests.

One exception, and it is deliberate: `FakeClaudeCli` fakes a SUBPROCESS,
not a port, because `ClaudeCliCognition` has no port to stand behind —
it spawns the `claude` binary and parses its stream-json stdout. The
double replaces the spawn and nothing above it, so a test still runs the
real parser, the real event mapping and the real budget charge.

Each fake records calls (where useful) so tests can assert on what the
agent sent. All exports here are re-exported from `agentkit.testing`
itself, so the user-facing import path is:

    from agentkit.testing import FakeLLM, FakeCtx, FakeGrounder, ...

The submodule path (`agentkit.testing.fakes.X`) is also valid for
callers who prefer per-module imports."""

from agentkit.testing.fakes.claude_cli import CliInvocation, CliRun, CliStderr, FakeClaudeCli
from agentkit.testing.fakes.clock import FakeClock
from agentkit.testing.fakes.compactor import FakeCompactor
from agentkit.testing.fakes.ctx import FakeCtx
from agentkit.testing.fakes.fetch import FakeFetch
from agentkit.testing.fakes.grounder import FakeGrounder
from agentkit.testing.fakes.llm import FakeLLM, ScriptExhausted, Turn
from agentkit.testing.fakes.memory import FakeMemory
from agentkit.testing.fakes.search import FakeSearch
from agentkit.testing.fakes.tool import FakeTool
from agentkit.testing.fakes.tracer import RecordingSpan, RecordingTracer

__all__ = [
    "CliInvocation",
    "CliRun",
    "CliStderr",
    "FakeClaudeCli",
    "FakeClock",
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
]
