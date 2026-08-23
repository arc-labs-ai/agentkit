"""Canonical test utilities for agentkit consumers.

What lives here:
- `make_test_ctx(...)` — factory that builds a REAL `RunContext` for
  tests, wired with the FakeLLM-backed Invoker + noop Trace/Observer
  defaults. Suitable for any test that needs a real ctx.

What lives under `fakes/`:
- `FakeCtx` — minimum Ctx that RECORDS spans (different from
  `agentkit.runtime.NullCtx`, which records nothing). Use when a test
  needs to assert on what spans were opened.
- `FakeLLM`, `FakeClock`, `FakeFetch`, `FakeSearch` — port doubles.
- `FakeGrounder`, `FakeCompactor` — capability doubles for
  RequestBuilder tests.

Naming convention: `Fake*` is the standard for test doubles. `Null*` /
`Noop*` are production-grade null-object patterns; they live in their
respective runtime/kernel modules, NOT here.
"""

from agentkit.testing.fakes import (
    FakeClock,
    FakeCompactor,
    FakeCtx,
    FakeFetch,
    FakeGrounder,
    FakeLLM,
    FakeMemory,
    FakeSearch,
    FakeTool,
    RecordingSpan,
    RecordingTracer,
    ScriptExhausted,
    Turn,
)
from agentkit.testing.make_ctx import make_test_ctx

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
    "make_test_ctx",
]
