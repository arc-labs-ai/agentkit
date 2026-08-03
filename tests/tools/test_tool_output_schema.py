"""Tool-result schema validation on ``FunctionTool``.

The fire site for the ``tool.shape_mismatch`` span event (B4b): when a tool
declares an ``output_schema`` and its return value doesn't match, the framework
drops the event on the currently-open span and raises :class:`ToolShapeError` —
caught by the Agent loop and reflected back to the model as a tool-call failure.

Cover both EXPLICIT ``output_schema=`` wiring and AUTO-INFERENCE from the
function's return-type annotation. Both must yield the same behaviour; the
auto-inference path is just an ergonomic shortcut on top of the same dispatch.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import Scope, ToolCall
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.testing import FakeLLM, Turn
from agentkit.tools import FunctionTool, ToolShapeError, tool


def _run(coro):
    return asyncio.run(coro)


# Module-scope Pydantic class for auto-inference tests. Defining the model
# inside a test function and using ``from __future__ import annotations``
# leaves ``get_type_hints`` unable to resolve the forward-stringified
# annotation against the function's module globals, so we keep it module-level
# for the auto-inference happy-path test.
_pydantic = pytest.importorskip("pydantic")


class _ModuleReport(_pydantic.BaseModel):
    title: str
    score: int


def _make_module_scope_report() -> _ModuleReport:
    """Return a payload the framework should coerce into a `_ModuleReport`."""
    return {"title": "auto", "score": 1}  # type: ignore[return-value]


# ---- Capturing tracer that records add_event_to_current_span on a span stack ----------------
# Mirrors the shape of `_CapturingTrace` in test_span_events.py so the assertion
# pattern is identical.


@dataclass
class _Span:
    name: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value


class _CapturingTrace:
    def __init__(self) -> None:
        self.spans: list[_Span] = []
        self._stack: list[_Span] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "", **attrs: Any):
        s = _Span(name=name, kind=kind, attrs=dict(attrs))
        self.spans.append(s)
        self._stack.append(s)
        try:
            yield s
        finally:
            self._stack.pop()

    def current_span_id(self):
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        if self._stack:
            self._stack[-1].events.append((name, fields))

    def all_events(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [(s.name, n, f) for s in self.spans for (n, f) in s.events]


def _make_ctx(*, trace: _CapturingTrace | None = None) -> RunContext:
    services_kwargs: dict[str, Any] = {}
    if trace is not None:
        services_kwargs["trace"] = trace
    services = Services(**services_kwargs)
    return RunContext(
        correlation_id="r",
        scope=Scope(),
        budget=Budget(max_cost_usd=10.0),
        services=services,
    )


# ---- 1. Pydantic output_schema: valid instance passes through unchanged --------------------


def test_pydantic_output_schema_passes_valid_instance_through():
    """A tool typed/declared with a Pydantic output_schema that returns the
    exact instance class → fast-path: the same object is returned, no
    re-validation needed."""
    pydantic = pytest.importorskip("pydantic")

    class Report(pydantic.BaseModel):
        title: str
        score: int

    instance = Report(title="ok", score=42)

    def make_report() -> Report:
        """Build a fully-typed Report instance for the model to read."""
        return instance

    t = FunctionTool.from_callable(make_report, output_schema=Report)
    out = _run(t.run({}, ctx=_make_ctx()))
    assert out is instance  # fast-path: the same object


# ---- 2. Pydantic output_schema: valid dict gets coerced ------------------------------------


def test_pydantic_output_schema_coerces_valid_dict():
    """A tool that returns a plain dict matching the Pydantic schema → the dict
    gets coerced into a typed instance via ``model_validate``."""
    pydantic = pytest.importorskip("pydantic")

    class Report(pydantic.BaseModel):
        title: str
        score: int

    def make_dict() -> Any:
        """Return a dict the framework should coerce into a Report."""
        return {"title": "ok", "score": 7}

    t = FunctionTool.from_callable(make_dict, output_schema=Report)
    out = _run(t.run({}, ctx=_make_ctx()))
    assert isinstance(out, Report)
    assert out.title == "ok" and out.score == 7


# ---- 3. Pydantic output_schema: INVALID dict raises + drops span event ---------------------


def test_pydantic_output_schema_invalid_dict_raises_and_emits_event():
    """A dict missing required Pydantic fields → ``ToolShapeError`` is raised
    and a ``tool.shape_mismatch`` span event is dropped on the open span."""
    pydantic = pytest.importorskip("pydantic")

    class Report(pydantic.BaseModel):
        title: str
        score: int

    def bad() -> Any:
        """Return a malformed payload (missing required `score`)."""
        return {"title": "incomplete"}

    t = FunctionTool.from_callable(bad, output_schema=Report)
    trace = _CapturingTrace()
    ctx = _make_ctx(trace=trace)

    # Open a span so add_event_to_current_span has something to land on
    # (mirroring B4b's tracing middleware opening the ``execute_tool`` span).
    async def go():
        with trace.span("execute_tool", "execute_tool", **{"gen_ai.tool.name": "bad"}):
            with pytest.raises(ToolShapeError) as ei:
                await t.run({}, ctx=ctx)
        return ei.value

    exc = _run(go())
    assert exc.tool_name == "bad"
    assert exc.expected == "Report"
    assert exc.errors  # the per-field error list is populated

    # The span event landed on the execute_tool span.
    mismatch_events = [f for (_s, n, f) in trace.all_events() if n == "tool.shape_mismatch"]
    assert len(mismatch_events) == 1
    ev = mismatch_events[0]
    assert ev["tool_name"] == "bad"
    assert ev["expected_shape"] == "Report"


# ---- 4. Dataclass output_schema: valid instance passes through ----------------------------


@dataclass
class _Hit:
    url: str
    title: str


@dataclass
class _SearchResults:
    query: str
    hits: list[_Hit]


def test_dataclass_output_schema_passes_valid_instance_through():
    """A dataclass output_schema with a matching instance → returned as-is."""
    expected = _SearchResults(query="redis", hits=[_Hit(url="https://r.io/1", title="One")])

    def search() -> _SearchResults:
        """Run a fake search and return a typed SearchResults instance."""
        return expected

    t = FunctionTool.from_callable(search, output_schema=_SearchResults)
    out = _run(t.run({}, ctx=_make_ctx()))
    assert out is expected


# ---- 5. Dataclass output_schema: wrong-type return raises ToolShapeError -------------------


def test_dataclass_output_schema_wrong_type_raises():
    """A tool declared to return a dataclass but returning a list → ``ToolShapeError``
    with the expected schema name carried on the exception."""

    def wrong() -> Any:
        """Return a value that doesn't match the declared dataclass shape."""
        return [1, 2, 3]  # not a dict, not a SearchResults

    t = FunctionTool.from_callable(wrong, output_schema=_SearchResults)
    with pytest.raises(ToolShapeError) as ei:
        _run(t.run({}, ctx=_make_ctx()))
    assert ei.value.tool_name == "wrong"
    assert ei.value.expected == "_SearchResults"


# ---- 6. Auto-inference from return-type annotation ----------------------------------------


def test_auto_inferred_output_schema_from_return_annotation():
    """A tool with a Pydantic return annotation and NO explicit ``output_schema``
    → the framework auto-infers and enforces the schema."""
    # Module-scope class — ``get_type_hints`` needs the class in the function's
    # module globals to resolve a forward-stringified annotation (this file uses
    # ``from __future__ import annotations``).
    t = FunctionTool.from_callable(_make_module_scope_report)
    assert t.output_schema is _ModuleReport
    out = _run(t.run({}, ctx=_make_ctx()))
    assert isinstance(out, _ModuleReport) and out.title == "auto"


def test_auto_inferred_output_schema_dataclass_return_annotation():
    """Same auto-inference behaviour for a stdlib dataclass return type."""

    def make() -> _SearchResults:
        """Return a typed SearchResults via auto-inference."""
        return _SearchResults(query="q", hits=[])

    t = FunctionTool.from_callable(make)
    assert t.output_schema is _SearchResults
    out = _run(t.run({}, ctx=_make_ctx()))
    assert isinstance(out, _SearchResults)


# ---- 7. Auto-inference skipped on primitive return ----------------------------------------


def test_auto_inference_skipped_for_primitive_return_annotation():
    """Functions typed ``-> str`` (etc.) MUST NOT trigger validation — the
    validation overhead would be silly and the schema is trivially "any string"."""

    def echo(s: str) -> str:
        """Echo back the input string — primitive return type, no schema check."""
        return s

    t = FunctionTool.from_callable(echo)
    assert t.output_schema is None
    assert t._output_adapter is None
    # Returning anything (even a non-string) doesn't trigger ToolShapeError
    # because no schema was auto-inferred.
    out = _run(t.run({"s": "hi"}, ctx=_make_ctx()))
    assert out == "hi"


def test_auto_inference_skipped_for_generic_dict_return():
    """A ``-> dict[str, Any]`` annotation isn't a typed-struct flavour ``adapt()``
    supports, so auto-inference quietly skips it (no crash on tool construction)."""

    def fetch(url: str) -> dict[str, Any]:
        """Fetch a URL and return a plain dict of the response."""
        return {"ok": True, "url": url}

    t = FunctionTool.from_callable(fetch)
    assert t.output_schema is None


def test_auto_inference_skipped_when_return_annotation_missing():
    """No return annotation at all → no auto-inference, no validation."""

    def loose(x):
        """A loose-typed tool with no return annotation — no schema enforced."""
        return x

    t = FunctionTool.from_callable(loose)
    assert t.output_schema is None


# ---- 8. Explicit `output_schema=None` opt-out --------------------------------------------


def test_explicit_output_schema_none_overrides_auto_inference():
    """A user can pass ``output_schema=None`` to the decorator to OPT OUT of
    auto-inference even when the return annotation IS enforceable."""
    pydantic = pytest.importorskip("pydantic")

    class Report(pydantic.BaseModel):
        title: str

    @tool(side_effecting=False, output_schema=None)
    def make() -> Report:
        """Return a wildly wrong value — the explicit None means no check fires."""
        return "not a Report"  # type: ignore[return-value]

    assert make.output_schema is None
    assert make._output_adapter is None
    out = _run(make.run({}, ctx=_make_ctx()))
    assert out == "not a Report"  # passes through unchanged — no validation


# ---- 9. Span event payload shape ----------------------------------------------------------


def test_shape_mismatch_event_carries_tool_name_and_expected_shape():
    """The span event payload MUST carry both ``tool_name`` and ``expected_shape``
    so the timeline + downstream tooling can attribute the failure cleanly."""
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        x: int

    def bad() -> Any:
        """Return wrong-shape data to force a schema mismatch."""
        return {"y": 5}

    t = FunctionTool.from_callable(bad, output_schema=Out)
    trace = _CapturingTrace()

    async def go():
        with trace.span("execute_tool", "execute_tool", **{"gen_ai.tool.name": "bad"}):
            with contextlib.suppress(ToolShapeError):
                await t.run({}, ctx=_make_ctx(trace=trace))

    _run(go())

    events = [(s, f) for (s, n, f) in trace.all_events() if n == "tool.shape_mismatch"]
    assert len(events) == 1
    span_name, fields = events[0]
    assert span_name == "execute_tool"  # landed on the open execute_tool span
    assert fields["tool_name"] == "bad"
    assert fields["expected_shape"] == "Out"


# ---- 10. Agent loop reflects ToolShapeError back to the model -----------------------------


def test_agent_loop_reflects_tool_shape_error_to_model():
    """When a tool inside an Agent run raises ``ToolShapeError``, the loop must
    catch and reflect the failure to the model as a tool-message — instead of
    crashing the run. The model then has the chance to recover on the next
    assistant turn."""
    pydantic = pytest.importorskip("pydantic")

    class Hit(pydantic.BaseModel):
        url: str
        title: str

    @tool(side_effecting=False, output_schema=Hit)
    def lookup(q: str) -> Any:
        """Look up a Hit for query `q`. (Will return a malformed payload to
        trigger the schema mismatch and exercise the reflect-to-model path.)"""
        return {"url": "https://x"}  # missing required `title` field

    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "lookup", {"q": "redis"}),)),
            Turn(content="recovered after the tool error"),
        ]
    )
    inv = Invoker(llm=llm)
    ctx = RunContext("r", Scope(1, 2), Budget(max_cost_usd=10.0), Services(invoker=inv))

    agent = Agent(name="a", model="m", cognition=ReActCognition(tools=[lookup]))
    res = _run(agent.run("q", ctx))

    # The loop didn't crash; the model produced its second turn after seeing the error.
    assert res.output == "recovered after the tool error"
    # Two LLM calls: the first emitted the tool call, the second saw the error
    # reflection in its transcript and produced the recovery answer.
    assert llm.calls == 2
