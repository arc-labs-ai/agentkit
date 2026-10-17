"""``output=`` on an agent the CLI runs, and what happens when it is not honoured.

WHY THIS FILE EXISTS
--------------------
A caller who writes ``Agent(output=Invoice, cognition=ClaudeCliCognition(...))``
has made a claim about the type they will get back. On the CLI path that claim
passes through four places, and each of them has a failure mode that produces a
RESULT rather than an error:

    the schema goes out      ``agent.output`` → ``--json-schema``. Ignore it and
                             the CLI never validates its answer at all; the run
                             returns prose and ``parsed=None``, which reads
                             exactly like an output type nobody wired up.

    the value comes back     the CLI validates against the schema and hands
                             back a plain ``dict``. Hand that dict straight on
                             and ``result.parsed`` is a mapping where the
                             caller's own type annotation says ``Invoice``, so
                             the ``AttributeError`` lands in their code.

    the value is rejected    an answer the JSON Schema accepted but the Python
                             type will not build is a FAILED run. Reported as a
                             success it is a null object presented as data.

    nothing comes back       the CLI can exit zero having produced no
                             ``structured_output`` at all, and the docs are
                             explicit that this is a failure too. Without that
                             branch the result is ``partial=False`` with
                             ``parsed=None`` — the most convincing wrong answer
                             on the whole path.

Each of those is silent. None of them raises, none logs, and all four return a
terminal event a caller can read without noticing. So the tests below assert on
the three things that stay observable: the argv, the TYPE of ``result.parsed``,
and the stop reason / failure record an application branches on.

THE ``from __future__ import annotations`` TRAP
----------------------------------------------
Schema inference reads annotations. Under postponed evaluation an annotation is
a string, and an unresolvable one falls back to ``{"type": "string"}`` without
complaint — so a schema built from a type this module cannot resolve by name is
a schema that silently accepts anything, and a test over it passes for the
wrong reason. Every output type here is a module-level dataclass with direct,
resolvable annotations for that reason. Do not parametrize over an annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.contracts.types import Mode, StreamEvent
from agentkit.core.testing.fakes import FakeClaudeCli
from agentkit.models.contracts.schema import OutputSpec, compile_schema
from agentkit.models.output.adapt import adapt
from agentkit.models.providers.cli._common import CliResult, CliTimeouts, StructuredOutputFailure
from agentkit.models.providers.cli.claude.driver import ClaudeCliCognition


@dataclass
class Invoice:
    """The declared output type. Annotations are direct and resolvable — see
    the module docstring on why that is load-bearing rather than incidental."""

    number: str
    total: int


# One finished run, without its ``structured_output``. Each test adds the
# structured half it is about, so the one that adds nothing is a genuine
# "the CLI exited zero and produced no object".
FINISHED_RUN: dict[str, Any] = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "Invoice A-1 totals 7.",
    "session_id": "01HSESSION",
    "duration_ms": 1234,
    "total_cost_usd": 0.25,
    "usage": {"input_tokens": 100, "output_tokens": 50},
}

# A raw JSON Schema, for the ``json_schema=`` escape hatch: a caller who wants
# the CLI to validate an answer without declaring a Python type for it.
RAW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ticket": {"type": "string"}},
    "required": ["ticket"],
    "additionalProperties": False,
}


@dataclass
class FakeCtx:
    """The execution context, holding only what this cognition reads off it."""

    scope: Any = None
    correlation_id: str = "01HCORRELATION"
    cancel: Any = None

    def check_cancelled(self) -> None:
        return None


class FakeAgent:
    """An agent-like carrying a declared output type, as ``Agent`` would.

    ``_output_adapter`` is built through the framework's own ``adapt()`` rather
    than hand-rolled: it is the SAME adapter that renders the schema and
    validates the answer, and a double that rendered one shape while accepting
    another would make the round trip these tests exist to check untestable.
    """

    prompt = "You are a concise assistant."
    model = "claude-sonnet-4-6"

    def __init__(self, output: Any = None) -> None:
        self._output_adapter = adapt(output) if output is not None else None


def cognition(cli: FakeClaudeCli, **kwargs: Any) -> ClaudeCliCognition:
    """A cognition wired to the offline CLI, with every liveness bound off."""
    return ClaudeCliCognition(spawn=cli, timeouts=CliTimeouts.off(), **kwargs)


async def drive(cog: ClaudeCliCognition, agent: Any, ctx: Any = None) -> CliResult:
    """Drive one run and return the result off its single terminal event."""
    events: list[StreamEvent] = []
    async for event in cog.drive(agent, "invoice me", ctx if ctx is not None else FakeCtx(), None):
        events.append(event)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1, f"expected exactly one terminal event, got {[e.type for e in events]}"
    result = finals[0].result
    assert isinstance(result, CliResult)
    return result


def sent_schema(cli: FakeClaudeCli) -> dict[str, Any]:
    """The schema the CLI was actually launched with, parsed back out of argv."""
    import json

    argv = cli.invocations[0].argv
    assert "--json-schema" in argv, f"--json-schema missing from argv: {argv}"
    parsed: dict[str, Any] = json.loads(argv[argv.index("--json-schema") + 1])
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# The schema goes out.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_agents_declared_output_type_becomes_the_clis_output_schema() -> None:
    """``output=Invoice`` on a CLI-delegated agent used to be ignored outright:
    the flag never appeared, the CLI validated nothing, and the run came back as
    prose with ``parsed=None``. That is indistinguishable from forgetting to
    declare an output type at all, which is why it survived. The guarantee is
    that declaring a type on this cognition types the run exactly as it types a
    normal one — so the schema in the argv has to carry the declared FIELDS,
    not merely be present."""
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": {"number": "A-1", "total": 7}}])

    await drive(cognition(cli), FakeAgent(Invoice))

    schema = sent_schema(cli)
    assert schema["type"] == "object"
    assert schema["properties"]["number"] == {"type": "string"}
    # The trap in the module docstring, caught: an unresolved annotation
    # degrades to ``{"type": "string"}``, so an integer field that arrives as a
    # string schema means the inference silently gave up.
    assert schema["properties"]["total"] == {"type": "integer"}
    assert sorted(schema["required"]) == ["number", "total"]


@pytest.mark.asyncio
async def test_an_explicit_json_schema_on_the_cognition_overrides_the_agents_type() -> None:
    """The two are different claims and the explicit one is the more specific:
    a caller passing ``json_schema=`` is overriding what the CLI validates
    against, usually because the vendor's dialect needs a shape the adapter
    does not render. Letting the agent's type win would make the override
    silently do nothing."""
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": {"ticket": "T-9"}}])

    await drive(cognition(cli, json_schema=RAW_SCHEMA), FakeAgent(Invoice))

    assert sorted(sent_schema(cli)["properties"]) == ["ticket"]


@pytest.mark.asyncio
async def test_an_agent_with_no_declared_output_asks_the_cli_to_validate_nothing() -> None:
    """Without a declared type there is no schema to send, and inventing one
    would make the CLI refuse answers the caller never asked it to constrain —
    plus turn every ordinary prose run into a structured-output failure, since
    an absent ``structured_output`` is only a failure when one was requested."""
    cli = FakeClaudeCli.script([FINISHED_RUN])

    result = await drive(cognition(cli), FakeAgent())

    assert "--json-schema" not in cli.invocations[0].argv
    assert result.stop_reason == "complete"
    assert result.partial is False
    assert result.parsed is None


# ─────────────────────────────────────────────────────────────────────────────
# The value comes back.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_validated_payload_comes_back_as_the_declared_python_type() -> None:
    """The CLI validates against the schema and hands back a plain ``dict``.
    That dict satisfies the schema and will satisfy any test that reads it with
    ``[]`` — but a caller who declared ``output=Invoice`` wrote
    ``result.parsed.number``, and a mapping answers that with an
    ``AttributeError`` in THEIR code, far from here. The round trip back through
    the same adapter that produced the schema is what makes the declared type
    the type that arrives."""
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": {"number": "A-1", "total": 7}}])

    result = await drive(cognition(cli), FakeAgent(Invoice))

    assert isinstance(result.parsed, Invoice)
    assert result.parsed == Invoice(number="A-1", total=7)
    assert result.partial is False
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_the_raw_validated_dict_is_kept_alongside_the_typed_object() -> None:
    """A caller who declared a Python type still sometimes wants the payload as
    the CLI produced it — to persist it, to diff it, to read a field the type
    drops. Coercing in place and discarding the original would make that
    impossible to recover, and the coercion is lossy by design."""
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": {"number": "A-1", "total": 7}}])

    result = await drive(cognition(cli), FakeAgent(Invoice))

    assert result.evals["structured_output"] == {"number": "A-1", "total": 7}


@pytest.mark.asyncio
async def test_a_raw_schema_with_no_python_type_hands_back_the_validated_dict() -> None:
    """``json_schema=`` on an agent that declares no ``output=`` is the
    escape hatch: there is no Python type to build, so the validated dict IS
    the parsed value. Returning ``None`` instead — on the grounds that no
    adapter ran — would make the escape hatch produce nothing at all, and the
    caller would have to dig the data back out of ``tests.evals``."""
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": {"ticket": "T-9"}}])

    result = await drive(cognition(cli, json_schema=RAW_SCHEMA), FakeAgent())

    assert result.parsed == {"ticket": "T-9"}
    assert result.partial is False


# ─────────────────────────────────────────────────────────────────────────────
# The value is rejected, or never arrives.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_payload_the_declared_type_rejects_is_never_reported_as_a_clean_parse() -> None:
    """A JSON Schema and a Python type do not agree on everything, so a payload
    the CLI accepted can still fail to build the declared class. The failure is
    reported rather than raised — the run happened and its prose is real — but
    reporting it with ``partial=False`` and the ordinary stop reason hands the
    caller a ``parsed=None`` that looks exactly like a run which never declared
    an output type, while the diagnostics sit unread in ``tests.evals``."""
    broken = {"number": "A-1", "total": "seven"}
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": broken}])

    result = await drive(cognition(cli), FakeAgent(Invoice))

    assert result.parsed is None
    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_mismatch"
    assert result.stop_reason == "invalid_output"
    # The prose is still handed back: the run was paid for and its text is real.
    assert result.output == "Invoice A-1 totals 7."


@pytest.mark.asyncio
async def test_a_success_with_no_structured_output_is_not_a_clean_run() -> None:
    """This is the most convincing wrong answer on the path. The process exits
    zero, the CLI publishes an ordinary ``success`` result, and nothing in it is
    flagged — but ``--json-schema`` was passed and no object came back. Without
    this branch the caller gets ``partial=False``, ``stop_reason`` unremarkable
    and ``parsed=None``, and reads the prose as though the output type had
    simply not been wired. The CLI's own docs call this a failure; it has to be
    one here too."""
    cli = FakeClaudeCli.script([FINISHED_RUN])

    result = await drive(cognition(cli), FakeAgent(Invoice))

    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_missing"
    assert result.stop_reason == "invalid_output"
    failure = StructuredOutputFailure.of(result.evals)
    assert failure is not None and failure.kind == "missing"


@pytest.mark.asyncio
async def test_exhausted_structured_output_retries_are_categorised_as_invalid_output() -> None:
    """The CLI re-prompts itself against ``--json-schema`` and reports
    ``error_max_structured_output_retries`` when it gives up. agentkit never saw
    the attempts, so the vendor subtype is all there is — and leaving it to the
    generic lifecycle mapping puts a run that produced no valid object in the
    same bucket as a run that merely ended. ``invalid_output`` is the closed
    taxonomy's name for "there is an answer and it is not usable", which is the
    one a retry policy or an alert branches on."""
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "subtype": "error_max_structured_output_retries"}])

    result = await drive(cognition(cli), FakeAgent(Invoice))

    assert result.stop_reason == "invalid_output"
    assert result.evals["stop_reason"] == "error_max_structured_output_retries"
    assert result.partial is True
    failure = StructuredOutputFailure.of(result.evals)
    assert failure is not None and failure.kind == "retries_exhausted"


@pytest.mark.asyncio
async def test_the_field_level_coercion_diagnostics_survive_onto_the_failure() -> None:
    """``OutputCoercionError.__str__`` only summarises — "2 error(s)" — and the
    part an application can act on is the per-field list behind it. Those paths
    are what a form highlights, what a repair prompt names and what an operator
    reads; dropping them leaves a failure record that says a number and nothing
    about which field was wrong. They are carried as structured violations
    rather than a joined sentence precisely so nobody has to parse English a
    library upgrade can reword."""
    broken = {"number": "A-1", "total": "seven", "paid": True}
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": broken}])

    result = await drive(cognition(cli), FakeAgent(Invoice))

    failure = StructuredOutputFailure.of(result.evals)
    assert failure is not None
    paths = {v.path for v in failure.violations}
    assert "$.total" in paths
    assert "$.paid" in paths
    assert any("expected int" in v.message for v in failure.violations)
    # And the repair prompt is built from them, so an empty violation list
    # would leave a caller re-prompting with no feedback at all.
    assert "$.total" in failure.repair_prompt()


# ─────────────────────────────────────────────────────────────────────────────
# The declaration itself — one contract, compiled per transport.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_schema_the_cli_receives_is_the_one_compiled_for_its_own_dialect() -> None:
    """The CLI parses ``--json-schema`` itself and refuses an invalid one at
    startup, on a subprocess's stderr that this cognition only surfaces for a
    run which already failed. So the schema cannot go out as the adapter's raw
    rendering: it goes through the same ``compile_schema`` pass every other
    transport uses, for the CLI's mode, which is what adds the dialect bits the
    binary expects. Sending the uncompiled form would fail the run three
    seconds in with a diagnosis nobody sees — and byte-equality here is also
    what keeps one declaration from becoming two contracts, where the same
    agent is validated differently depending on which cognition ran it."""
    cli = FakeClaudeCli.script([{**FINISHED_RUN, "structured_output": {"number": "A-1", "total": 7}}])
    agent = FakeAgent(Invoice)

    await drive(cognition(cli), agent)

    adapter = adapt(Invoice)
    # Carried as an ``OutputSpec`` on every non-CLI path: the same declaration,
    # deep-frozen, which is the form the rest of the framework compiles from.
    spec = OutputSpec(schema=adapter.json_schema(), name=adapter.name)
    assert sent_schema(cli) == compile_schema(dict(spec.schema), Mode.CLAUDE_CODE, agent.model)
