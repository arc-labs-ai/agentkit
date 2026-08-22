"""``output=`` types a CLI-delegated run, exactly like it types a normal one.

Before this, `ClaudeCliCognition` ignored `agent.output` entirely: the schema
was never sent, `AgentResult.parsed` was always `None`, and a caller who
declared `output=Invoice` got prose back with no indication that the typing
they asked for had been dropped on the floor.

The CLI has first-class support for this — `--json-schema` makes it validate
its own final answer and re-prompt itself on a mismatch, returning the value in
the result payload's `structured_output` field. So the schema goes out through
the same `SchemaAdapter` that types the rest of the framework, and the
validated dict comes back through that same adapter into a real Python object.

Three outcomes, and only the first is a success — the other two are the ones
that used to look like success:

* value present  → `parsed` is the declared type
* retries burnt  → subtype `error_max_structured_output_retries`
* absent, exit 0 → the CLI docs are explicit that this is a failure too
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.context import WorkingContext
from agentkit.kernel.types import StreamEvent
from agentkit.testing.fakes.ctx import FakeCtx
from tests.agents.cognition.test_claude_cli import _FakeProcess, _line

pydantic = pytest.importorskip("pydantic")

real_cli = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


class Invoice(pydantic.BaseModel):
    vendor: str
    total: float


def _stream(*, structured: Any = ..., subtype: str = "success", text: str = "done") -> list[bytes]:
    """A minimal CLI stream: one assistant text block, then a result payload.

    ``structured=...`` (the sentinel) omits ``structured_output`` entirely —
    the shape a CLI run takes when it never produced one.
    """
    result: dict[str, Any] = {
        "type": "result",
        "subtype": subtype,
        "is_error": False,
        "session_id": "sess-1",
        "duration_ms": 12,
        "total_cost_usd": 0.001,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "result": text,
    }
    if structured is not ...:
        result["structured_output"] = structured
    return [
        _line({"type": "system", "subtype": "init", "session_id": "sess-1"}),
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}),
        _line(result),
    ]


def _drive(cog: ClaudeCliCognition, agent: Agent, lines: list[bytes]) -> tuple[Any, tuple[str, ...]]:
    proc = _FakeProcess(stdout_lines=lines)
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:

        async def _go() -> list[StreamEvent]:
            return [ev async for ev in cog.drive(agent, "extract it", FakeCtx(), WorkingContext())]

        events = asyncio.run(_go())
    return events[-1].result, tuple(spawn.await_args.args)


# ── 1. the schema goes out ──────────────────────────────────────────────────


def test_an_agents_output_schema_reaches_the_cli() -> None:
    """THE gap. ``output=Invoice`` used to be silently ignored here."""
    agent = Agent(name="x", output=Invoice, cognition=(cog := ClaudeCliCognition()))
    _, argv = _drive(cog, agent, _stream(structured={"vendor": "ACME", "total": 42.5}))

    sent = json.loads(argv[argv.index("--json-schema") + 1])
    assert sent["type"] == "object"
    assert set(sent["properties"]) == {"vendor", "total"}
    assert sent["required"] == ["vendor", "total"]


def test_an_explicit_schema_overrides_the_agents() -> None:
    """The cognition's own ``json_schema=`` wins — it is the escape hatch for a
    shape the CLI should produce that is not the agent's Python type."""
    explicit = {"type": "object", "properties": {"n": {"type": "integer"}}}
    cog = ClaudeCliCognition(json_schema=explicit)
    agent = Agent(name="x", output=Invoice, cognition=cog)
    _, argv = _drive(cog, agent, _stream(structured={"n": 1}))
    assert json.loads(argv[argv.index("--json-schema") + 1]) == explicit


def test_no_schema_no_flag() -> None:
    """The negative control: an agent with no ``output=`` must not gain a flag,
    since ``--json-schema`` forces the CLI into a validating mode."""
    cog = ClaudeCliCognition()
    _, argv = _drive(cog, Agent(name="x", cognition=cog), _stream(structured=...))
    assert "--json-schema" not in argv


# ── 2. the value comes back TYPED ───────────────────────────────────────────


def test_the_validated_value_is_coerced_to_the_declared_type() -> None:
    """``parsed`` is an ``Invoice``, not the raw dict — the CLI validated the
    shape, the adapter builds the object, and the caller gets what
    ``output=`` promised. The raw dict stays available in ``evals``."""
    cog = ClaudeCliCognition()
    agent = Agent(name="x", output=Invoice, cognition=cog)
    result, _ = _drive(cog, agent, _stream(structured={"vendor": "ACME", "total": 42.5}))

    assert isinstance(result.parsed, Invoice)
    assert result.parsed.vendor == "ACME" and result.parsed.total == 42.5
    assert result.evals["structured_output"] == {"vendor": "ACME", "total": 42.5}
    assert result.partial is False and result.stop_reason == "complete"


def test_without_a_python_type_the_dict_is_the_parsed_value() -> None:
    """An explicit ``json_schema=`` on an agent that declares no ``output=``:
    there is no type to build, so the validated dict IS the answer."""
    cog = ClaudeCliCognition(json_schema={"type": "object"})
    result, _ = _drive(cog, Agent(name="x", cognition=cog), _stream(structured={"a": 1}))
    assert result.parsed == {"a": 1}


# ── 3. the three ways it fails ──────────────────────────────────────────────


def test_a_value_that_does_not_fit_the_python_type_is_a_failure() -> None:
    """The CLI validated against the JSON Schema, which is looser than the
    Python type (a missing required field the schema did not require, a
    stricter validator). Reported, never raised — the run happened."""
    cog = ClaudeCliCognition()
    agent = Agent(name="x", output=Invoice, cognition=cog)
    result, _ = _drive(cog, agent, _stream(structured={"vendor": "ACME"}))  # no total

    assert result.parsed is None
    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_mismatch"
    assert result.stop_reason == "invalid_output"
    assert "total" in result.evals["structured_output_error"]


def test_a_success_with_no_structured_output_is_a_failure() -> None:
    """Called out explicitly in the CLI docs. Without this branch the run reads
    ``partial=False`` with ``parsed=None``, and a caller who declared
    ``output=`` sees exactly what they would see if the wiring did not exist."""
    cog = ClaudeCliCognition()
    agent = Agent(name="x", output=Invoice, cognition=cog)
    result, _ = _drive(cog, agent, _stream(structured=...))

    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_missing"
    assert result.stop_reason == "invalid_output"


def test_exhausted_retries_are_reported_as_invalid_output() -> None:
    """``error_max_structured_output_retries`` is the CLI giving up after
    re-prompting itself. Not ``failed`` — the run worked, the shape did not."""
    cog = ClaudeCliCognition()
    agent = Agent(name="x", output=Invoice, cognition=cog)
    result, _ = _drive(
        cog, agent, _stream(structured=..., subtype="error_max_structured_output_retries")
    )
    assert result.evals["stop_reason"] == "error_max_structured_output_retries"
    assert result.stop_reason == "invalid_output"
    assert result.partial is True


def test_an_unschemad_run_is_untouched_by_all_of_this() -> None:
    """The additive guarantee: a caller who never asked for structured output
    sees exactly the previous behaviour — no flag, no parsed, no partial."""
    cog = ClaudeCliCognition()
    result, argv = _drive(cog, Agent(name="x", cognition=cog), _stream(structured=...))
    assert "--json-schema" not in argv
    assert result.parsed is None and result.partial is False
    assert "structured_output_error" not in result.evals


# ── 4. against the real binary ──────────────────────────────────────────────


@real_cli
def test_the_real_cli_returns_a_typed_object() -> None:
    """End to end: the adapter's schema is accepted by the CLI, the CLI returns
    a conforming value, and it comes back as an ``Invoice``.

    A mock proves the plumbing; only the binary proves the schema dialect is
    one it accepts (it validates JSON Schema draft-07 and rejects newer).
    """
    cog = ClaudeCliCognition(
        model="claude-haiku-4-5-20251001", tools=("",), permission_mode="dontAsk", max_turns=1
    )
    agent = Agent(name="x", prompt="Be terse.", output=Invoice, cognition=cog)

    async def _go() -> Any:
        events = [
            ev
            async for ev in cog.drive(
                agent,
                "The vendor is ACME and the total is 42.50. Return it.",
                FakeCtx(),
                WorkingContext(),
            )
        ]
        return events[-1].result

    result = asyncio.run(_go())
    assert result.stop_reason == "complete", result.evals
    assert isinstance(result.parsed, Invoice)
    assert result.parsed.vendor == "ACME"
    assert result.parsed.total == 42.5
