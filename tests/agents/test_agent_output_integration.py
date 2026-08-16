"""End-to-end wiring for Agent ``output=`` — the B3b integration contract.

These tests pin the round trip the Track B3b plan documents:

1. The Agent builds a ``SchemaAdapter`` from ``output=`` once in
   ``__post_init__`` (via the ``adapt()`` dispatcher).
2. The adapter's JSON Schema lands in ``WorkingContext.prefix.schema_block``
   — the cache-stable head — NOT in the per-turn ``messages`` tail.
3. The Agent's parse-and-repair loop uses ``adapter.parse`` to coerce the
   model response; failures reflect the validation diagnostics back to the
   model on the next turn and the loop retries.
4. ``AgentResult.parsed`` carries the typed instance on success and stays
   ``None`` when no ``output=`` was declared.

The tests use the canonical ``FakeLLM`` / ``make_test_ctx`` fixtures so the
focus is on Agent-layer wiring, not provider behaviour.

Deliberately NO ``output_coerce()`` in the chain: this module pins that the
strict ``AgentResult.parsed`` path works on the Agent's own
``parse``-and-repair loop, independent of the middleware. That wiring makes
the Agent emit its "no output_coerce, so no streamed partials" warning,
which is correct here and suppressed below. The middleware-wired path —
including that the two compose without breaking repair — is covered in
``tests/agents/test_stream_partial_output.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.agents import Agent
from agentkit.context import WorkingContext
from agentkit.testing import FakeLLM, Turn, make_test_ctx

pydantic = pytest.importorskip("pydantic")

pytestmark = pytest.mark.filterwarnings("ignore:agent .* has no output_coerce:UserWarning")


class PlannerOutput(pydantic.BaseModel):
    """Pydantic schema used in the happy-path and retry tests.

    Two required fields (``subject``, ``steps``) and one optional
    (``rationale``) give us enough surface to exercise both shape
    validation and the per-field error reflection."""

    subject: str
    steps: list[str]
    rationale: str | None = None


@dataclass
class TaskOutput:
    """Stdlib dataclass schema — exercises the DataclassAdapter path
    end-to-end."""

    title: str
    priority: int


def _run(agent: Agent, ctx: Any, task: str = "go") -> Any:
    return asyncio.run(agent.run(task, ctx))


# ── 1. Happy path with Pydantic ─────────────────────────────────────────────


def test_happy_path_pydantic_yields_typed_parsed_object() -> None:
    """`output=PlannerOutput` + valid JSON → `result.parsed` is a `PlannerOutput`."""
    valid = '{"subject": "ship B3b", "steps": ["wire", "test"]}'
    agent = Agent(name="planner", model="m", output=PlannerOutput)
    result = _run(agent, make_test_ctx(llm=FakeLLM(valid)))
    assert isinstance(result.parsed, PlannerOutput)
    assert result.parsed.subject == "ship B3b"
    assert result.parsed.steps == ["wire", "test"]
    assert not result.partial


# ── 2. Happy path with stdlib dataclass ─────────────────────────────────────


def test_happy_path_dataclass_yields_typed_parsed_object() -> None:
    """`output=TaskOutput` (dataclass) + valid JSON → `result.parsed` is a `TaskOutput`."""
    valid = '{"title": "ship", "priority": 1}'
    agent = Agent(name="planner", model="m", output=TaskOutput)
    result = _run(agent, make_test_ctx(llm=FakeLLM(valid)))
    assert isinstance(result.parsed, TaskOutput)
    assert result.parsed.title == "ship"
    assert result.parsed.priority == 1
    assert not result.partial


# ── 3. No `output=` declared → no parsed object ─────────────────────────────


def test_no_output_declared_leaves_parsed_none() -> None:
    """`output=` unset means the Agent is back to its unstructured shape:
    `result.parsed is None`, no error, no schema in the prefix."""
    agent = Agent(name="a", model="m")
    ctx = make_test_ctx(llm=FakeLLM('{"x": 1}'))
    result = _run(agent, ctx)
    assert result.parsed is None
    assert not result.partial


# ── 4. Invalid JSON triggers reflect-and-retry ──────────────────────────────


def test_invalid_json_triggers_reflect_and_retry() -> None:
    """First reply fails validation → the loop reflects the OutputCoercionError
    diagnostics back as a user message → the second reply parses → we end up
    with two LLM calls and the second user message contains the validation
    error so the model can correct itself."""
    bad = '{"subject": "x"}'  # missing required `steps`
    good = '{"subject": "x", "steps": ["a"]}'
    llm = FakeLLM.script([Turn(content=bad), Turn(content=good)])
    agent = Agent(name="planner", model="m", output=PlannerOutput, max_repairs=1)
    # Use a captured WorkingContext so we can inspect the reflected user message.
    wc = WorkingContext()
    ctx = make_test_ctx(llm=llm)
    result = asyncio.run(agent.run("plan", ctx, context=wc))
    assert isinstance(result.parsed, PlannerOutput)
    assert result.parsed.steps == ["a"]
    assert llm.calls == 2  # one repair round
    # The reflected error becomes a user message on the tail. The contents
    # should mention the validation failure so the model has something to
    # correct on.
    reflections = [m for m in wc.messages if m.role == "user" and "previous response" in m.content]
    assert reflections, "expected a reflect-and-retry user message on the tail"
    assert "steps" in reflections[0].content or "failed to coerce" in reflections[0].content


# ── 5. Schema lands in the prefix (cache-stable head) ───────────────────────


def test_schema_block_lands_in_prefix() -> None:
    """After Agent.run, `context.prefix.schema_block` is non-None and contains
    the JSON Schema. This is the cache-stable invariant: the schema is part of
    the prefix, not the per-turn tail."""
    wc = WorkingContext()
    agent = Agent(name="planner", model="m", output=PlannerOutput)
    ctx = make_test_ctx(llm=FakeLLM('{"subject": "x", "steps": []}'))
    _run(agent, ctx)
    # The Agent uses its own WorkingContext when one isn't passed — re-run
    # explicitly passing wc so we can inspect.
    wc = WorkingContext()
    ctx = make_test_ctx(llm=FakeLLM('{"subject": "x", "steps": []}'))
    asyncio.run(agent.run("go", ctx, context=wc))
    assert wc.prefix.schema_block is not None
    # The schema_block contains the adapter's JSON Schema as text — at minimum
    # the field names and the schema header should be present.
    assert "subject" in wc.prefix.schema_block
    assert "steps" in wc.prefix.schema_block
    assert "PlannerOutput" in wc.prefix.schema_block


# ── 6. Schema does NOT leak into the messages tail ──────────────────────────


def test_schema_does_not_leak_into_messages() -> None:
    """The schema text MUST live in the prefix only. The per-turn `messages`
    tail (the mutable, post-cache part) MUST NOT contain it — this is the
    KV-cache-discipline invariant: mutating the tail is cheap, mutating the
    prefix invalidates the cache.

    We check the user/tool/system tail entries — the assistant message
    will of course carry the model's emitted JSON, which is expected
    and is NOT a schema-text leak (it's the parsed output)."""
    wc = WorkingContext()
    agent = Agent(name="planner", model="m", output=PlannerOutput)
    ctx = make_test_ctx(llm=FakeLLM('{"subject": "x", "steps": []}'))
    asyncio.run(agent.run("go", ctx, context=wc))
    # The schema header sentence is the distinguishing marker — it can
    # ONLY come from `_render_schema_block`. Check no tail message
    # (regardless of role) carries it.
    for m in wc.messages:
        assert "Respond with JSON matching this schema" not in (m.content or ""), (
            f"schema preamble leaked into tail: role={m.role!r}"
        )
    # And no NON-assistant message should carry the schema-name marker.
    non_assistant = [m for m in wc.messages if m.role != "assistant"]
    for m in non_assistant:
        assert "PlannerOutput" not in (m.content or ""), (
            f"schema leaked into tail message: role={m.role!r} content={m.content!r}"
        )


# ── 7. AgentResult.parsed is the same instance the adapter produced ────────


def test_parsed_round_trips_through_adapter() -> None:
    """The typed object on `AgentResult.parsed` came from `adapter.parse` —
    re-serialising it via `adapter.serialize` should round-trip the public
    fields. Guards against silent string-leaking ("parsed" really is a
    typed instance, not just the raw content again)."""
    agent = Agent(name="planner", model="m", output=PlannerOutput)
    valid = '{"subject": "ship", "steps": ["a", "b"], "rationale": "why"}'
    ctx = make_test_ctx(llm=FakeLLM(valid))
    result = _run(agent, ctx)
    assert isinstance(result.parsed, PlannerOutput)
    # Pydantic's model_dump is what PydanticAdapter.serialize uses.
    assert result.parsed.model_dump() == {
        "subject": "ship",
        "steps": ["a", "b"],
        "rationale": "why",
    }


# ── 8. Adapter is exposed on Agent for caller introspection ────────────────


def test_output_adapter_is_built_at_post_init() -> None:
    """The Agent builds the adapter once at construction. This guards against
    a silent regression where the adapter would be re-built per turn (which
    would invalidate caches and double the schema-rendering cost)."""
    agent = Agent(name="planner", model="m", output=PlannerOutput)
    assert agent._output_adapter is not None
    assert agent._output_adapter.python_type is PlannerOutput
    assert agent._output_adapter.name == "PlannerOutput"
    # Stays None when output= isn't declared.
    bare = Agent(name="a", model="m")
    assert bare._output_adapter is None


# ── 9. response_format auto-inference (OpenAI only for now) ────────────────


def test_response_format_inferred_for_openai_models() -> None:
    """`gpt-*` models get an OpenAI `json_schema` response_format wired
    automatically. Other providers default to None and rely on the
    prompt-injection path. An explicit `response_format=` wins over the
    inference."""
    agent_gpt = Agent(name="a", model="gpt-4o-mini", output=PlannerOutput)
    assert agent_gpt.response_format is not None
    assert agent_gpt.response_format["type"] == "json_schema"
    assert agent_gpt.response_format["json_schema"]["name"] == "PlannerOutput"
    assert agent_gpt.response_format["json_schema"]["strict"] is True

    agent_claude = Agent(name="a", model="claude-3-5-sonnet", output=PlannerOutput)
    # No native wiring yet for Anthropic — fall back to prompt-injection only.
    assert agent_claude.response_format is None

    explicit = {"type": "json_object"}
    agent_explicit = Agent(
        name="a", model="gpt-4o-mini", output=PlannerOutput, response_format=explicit
    )
    assert agent_explicit.response_format is explicit
