"""Test doubles — fakes that satisfy port Protocols with deterministic
canned behavior for unit + integration tests.

Each fake records calls (where useful) so tests can assert on what the
agent sent. All exports here are re-exported from `agentkit.testing`
itself, so the user-facing import path is:

    from agentkit.testing import FakeLLM, FakeCtx, FakeGrounder, ...

The submodule path (`agentkit.testing.fakes.X`) is also valid for
callers who prefer per-module imports."""

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
