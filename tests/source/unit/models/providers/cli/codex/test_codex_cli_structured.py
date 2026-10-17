"""``--output-schema``: a typed Codex run, and the two ways it goes quiet.

WHY THIS FILE EXISTS
--------------------
Codex does not return a separate validated field the way ``claude`` does. It
constrains its FINAL MESSAGE to the schema, so the object has to be parsed back
out of the answer text — and when the model answers in prose instead, there is
nothing to parse and nothing that raises. Reporting that as a clean success with
``parsed=None`` is the worst available outcome: a caller who declared
``output=Invoice`` reads the silence as if the type had simply not been wired,
and goes looking at their own adapter.

The second failure is housekeeping with teeth. The flag takes a FILE PATH, not
inline JSON, so every typed run writes the schema into a 0700 scratch directory
and must remove it — otherwise each run leaves behind a file whose contents name
the caller's data model, on whatever host the service happens to run on.

These tests use ``json_schema=`` rather than an ``output=``-carrying agent so
the subject is this cognition's decode-and-report path and not a schema
adapter's: a coercion failure is a different branch with a different stop
reason, and mixing them would make a passing test ambiguous about which one it
pinned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agentkit.core.contracts.types import StreamEvent
from agentkit.core.testing.fakes import FakeCodexCli, codex_turn
from agentkit.models.providers.cli.codex.driver import CodexCliCognition, _strict_object_schema

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"total": {"type": "integer"}},
    "required": ["total"],
}


@dataclass
class _Agent:
    prompt: str | None = None


async def _drive(cognition: CodexCliCognition) -> list[StreamEvent]:
    return [ev async for ev in cognition.drive(_Agent(), "total it up", None, None)]


def _final(events: list[StreamEvent]) -> Any:
    assert [e.type for e in events].count("final") == 1, [e.type for e in events]
    return events[-1].result


def _surviving(schema_path: str) -> tuple[str, ...]:
    """Whichever of the schema file and its 0700 scratch directory still exist.

    A plain sync helper so the blocking stat stays out of the async test body,
    which the suite lints for.
    """
    path = Path(schema_path)
    return tuple(str(p) for p in (path, path.parent) if p.exists())


def _typed(answer: str) -> tuple[CodexCliCognition, FakeCodexCli]:
    cli = FakeCodexCli.script(codex_turn(text=answer, usage=(10, 0, 5)))
    return CodexCliCognition(spawn=cli, json_schema=_SCHEMA), cli


@pytest.mark.asyncio
async def test_a_final_message_that_is_not_the_requested_json_is_a_partial_run() -> None:
    """The run worked; the SHAPE did not. Reporting it as complete hands back
    ``parsed=None`` with ``partial=False`` and no stop reason — indistinguishable
    from a run where no output type was declared at all, which is where a caller
    who declared one starts debugging their own schema adapter instead of
    reading what the model actually said."""
    cognition, _ = _typed("I'd rather explain it in words.")
    result = _final(await _drive(cognition))

    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_missing"
    assert result.stop_reason == "invalid_output"
    assert result.parsed is None
    # ``undecodable`` rather than ``missing``: there WAS text and it was not
    # JSON, which is the case where what the model said is the whole diagnosis.
    failure = result.evals["structured_output_failure"]
    assert failure["kind"] == "undecodable"
    assert "I'd rather explain it in words." in result.evals["structured_output_error"]


@pytest.mark.asyncio
async def test_an_empty_final_message_is_a_missing_output_rather_than_an_undecodable_one() -> None:
    """The two have different fixes. "Nothing came back" is a run that died or a
    model that refused; "this is not JSON" is a prompt or schema problem with an
    excerpt worth reading. Collapsing them loses the only clue either one
    carries."""
    cognition, _ = _typed("")
    result = _final(await _drive(cognition))

    assert result.partial is True
    assert result.evals["structured_output_failure"]["kind"] == "missing"
    assert result.evals["stop_reason"] == "structured_output_missing"


@pytest.mark.asyncio
async def test_the_schema_file_the_run_wrote_does_not_outlive_the_run() -> None:
    """``--output-schema`` takes a path, so a typed run writes the caller's data
    model to disk. Leaking it means every typed run leaves behind a file naming
    that model — accumulating, on a host the caller may not control, for a file
    they were never told exists and cannot manage."""
    cognition, cli = _typed('{"total": 7}')
    await _drive(cognition)

    argv = cli.argv(0)
    schema_path = argv[argv.index("--output-schema") + 1]
    survivors = _surviving(schema_path)

    assert survivors == (), f"the run left {survivors} behind"


@pytest.mark.asyncio
async def test_a_json_final_message_is_parsed_back_out_of_the_answer_text() -> None:
    """The success path, and the reason the failure paths above have to be
    distinguishable from it: with no output adapter the validated dict IS the
    parsed value, so ``parsed`` and ``tests.evals["structured_output"]`` are the
    only evidence the schema did anything."""
    cognition, _ = _typed('{"total": 7}')
    result = _final(await _drive(cognition))

    assert result.partial is False
    assert result.stop_reason == "complete"
    assert result.parsed == {"total": 7}
    assert result.evals["structured_output"] == {"total": 7}
    assert "structured_output_error" not in result.evals


@pytest.mark.asyncio
async def test_a_fenced_json_answer_is_accepted_because_a_fence_is_a_formatting_habit() -> None:
    """A model told to answer in JSON will wrap it in a ``` fence. That is
    compliance with a decoration on it, not a failure to comply, and it is the
    ONLY accommodation made — scanning for the first ``{`` was rejected, because
    it turns "the model ignored the schema and explained itself" into a
    confident parse of whatever JSON-shaped fragment was in the explanation."""
    cognition, _ = _typed('```json\n{"total": 7}\n```')
    result = _final(await _drive(cognition))

    assert result.parsed == {"total": 7}
    assert result.partial is False


@pytest.mark.asyncio
async def test_prose_wrapped_around_a_json_object_is_reported_rather_than_salvaged() -> None:
    """The counterpart to the fence: a model that answered in sentences with an
    example object in them has NOT answered the schema, and the caller needs to
    see that. Salvaging the fragment would produce a confident, wrong object and
    a successful-looking run."""
    cognition, _ = _typed('Here is my answer: {"total": 7} — hope that helps.')
    result = _final(await _drive(cognition))

    assert result.partial is True
    assert result.parsed is None
    assert result.evals["structured_output_failure"]["kind"] == "undecodable"


@pytest.mark.asyncio
async def test_no_scratch_directory_is_created_for_a_run_that_declared_no_schema() -> None:
    """The cleanup has to be conditional on there being something to clean, and
    an untyped run must not pay for a temp directory it never used — this is the
    other side of the leak test, and the branch a mutated cleanup inverts."""
    cli = FakeCodexCli.script(codex_turn(text="plain answer", usage=(1, 0, 1)))
    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert "--output-schema" not in cli.argv(0)
    assert result.output == "plain answer"
    assert result.parsed is None
    assert "structured_output" not in result.evals


def test_every_object_in_the_schema_is_closed_because_openai_strict_mode_demands_it() -> None:
    """Strict mode refuses a schema whose objects do not close themselves, and
    the two adapters disagree about it: the dataclass adapter emits
    ``additionalProperties: false`` and Pydantic's does not. So
    ``output=SomeDataclass`` worked on Codex and ``output=SomePydanticModel``
    failed every time with a 400 the caller could only find by reading
    ``tests.evals["stderr"]`` — an arbitrary difference between two ways of
    declaring the same shape."""
    closed = _strict_object_schema(
        {
            "type": "object",
            "properties": {
                "line": {"type": "object", "properties": {"qty": {"type": "integer"}}},
                "tags": {"type": "array", "items": {"properties": {"name": {"type": "string"}}}},
            },
        }
    )

    assert closed["additionalProperties"] is False
    assert closed["properties"]["line"]["additionalProperties"] is False
    # Detected by ``properties`` too, not only by an explicit ``type``.
    assert closed["properties"]["tags"]["items"]["additionalProperties"] is False
    # A leaf that is not an object is left exactly as it was.
    assert closed["properties"]["line"]["properties"]["qty"] == {"type": "integer"}


def test_an_explicitly_stated_additional_properties_is_left_alone_even_when_wrong() -> None:
    """A caller who wrote ``additionalProperties: true`` meant it, and silently
    inverting a stated instruction is worse than the provider error that names
    it — the error is readable, the inversion is a schema that does not match
    the one in the source."""
    assert _strict_object_schema({"type": "object", "additionalProperties": True})["additionalProperties"] is True
