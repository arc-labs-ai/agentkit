"""``output=`` types a Codex-delegated run, exactly like it types a normal one.

The CLI's seam for this is ``--output-schema``, and it differs from
``claude --json-schema`` in two ways that both change what this cognition has
to do:

**It takes a FILE PATH, not inline JSON.** So the schema is written to a temp
file under a 0700 directory for the life of the spawn and removed afterwards. A
caller must never be asked to manage one — a schema file left in ``/tmp`` after
every run is a leak, and a schema file the caller has to create is an API that
makes ``output=Invoice`` worse than writing the prompt by hand.

**The validated value is the final MESSAGE, not a separate field.** ``claude``
hands back a ``structured_output`` key alongside its prose; Codex constrains the
answer itself. So the object has to be parsed back out of the answer text, and
"the model explained itself instead of answering in JSON" has to be reported
rather than repaired — that is the case a caller most needs to see.

Three outcomes, and only the first is a success — the other two are the ones
that would otherwise look like one:

* parses and coerces → ``parsed`` is the declared type
* parses, wrong shape → ``structured_output_mismatch``
* does not parse      → ``structured_output_missing``
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CodexCliCognition
from agentkit.testing.fakes import FakeCodexCli, codex_turn
from tests.agents.cognition.test_codex_cli import drive, final_of

pydantic = pytest.importorskip("pydantic")

real_codex = pytest.mark.skipif(
    shutil.which("codex") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="codex CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


class Invoice(pydantic.BaseModel):
    vendor: str
    total: float


def schema_path_of(cli: FakeCodexCli) -> Path | None:
    argv = list(cli.invocations[-1].argv)
    return Path(argv[argv.index("--output-schema") + 1]) if "--output-schema" in argv else None


def still_there(path: Path) -> bool:
    """Whether the path exists — a sync helper so ``ASYNC240`` does not fire on
    a single ``stat`` inside an async test body."""
    return path.exists()


def _read_schema(argv: tuple[str, ...]) -> tuple[bool, object]:
    """``(the file was there, its parsed contents)`` for a spawn's argv.

    Synchronous, and called from the double below, because the whole point is
    to look at the file WHILE the spawn is happening — after the run it is
    gone, which is its own assertion further down.
    """
    if "--output-schema" not in argv:
        return False, None
    path = Path(argv[list(argv).index("--output-schema") + 1])
    if not path.is_file():
        return False, None
    return True, json.loads(path.read_text())


class PeekingCodexCli(FakeCodexCli):
    """A ``FakeCodexCli`` that records the schema file it was handed.

    The only way to assert on a file whose entire lifetime is one spawn. One
    class rather than one per test, because three tests want the same look and
    three copies of a filesystem read inside an ``async def`` is also three
    lint suppressions.
    """

    seen: dict[str, object]

    def __init__(self, *a: object, **kw: object) -> None:
        super().__init__(*a, **kw)  # type: ignore[arg-type]
        self.seen = {}

    async def __call__(self, *argv: str, **kw: object):  # type: ignore[no-untyped-def]
        existed, document = _read_schema(argv)
        self.seen["exists"] = existed
        self.seen["document"] = document
        return await super().__call__(*argv, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# 1. the schema goes out
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_agents_output_type_becomes_the_output_schema() -> None:
    """The gap this closes: declaring ``output=Invoice`` used to be silently
    dropped on a CLI-delegated run, so a caller got prose back with no
    indication that the typing they asked for had gone nowhere."""
    cli = FakeCodexCli.script(codex_turn(text='{"vendor": "Acme", "total": 12.5}', usage=(10, 0, 8)))
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="reader", cognition=cog, output=Invoice)

    result = final_of(await drive(cog, agent=agent))

    assert isinstance(result.parsed, Invoice)
    assert (result.parsed.vendor, result.parsed.total) == ("Acme", 12.5)
    assert result.partial is False
    assert result.evals["structured_output"] == {"vendor": "Acme", "total": 12.5}


@pytest.mark.asyncio
async def test_the_schema_is_written_to_a_file_the_flag_names() -> None:
    """``--output-schema`` takes a path. Passing the JSON inline — which is what
    the sibling CLI wants — fails at CLI startup with an error about a file."""
    cli = PeekingCodexCli.script(codex_turn(text='{"vendor": "A", "total": 1}', usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    await drive(cog, agent=Agent(name="r", cognition=cog, output=Invoice))

    assert cli.seen["exists"] is True
    document = cli.seen["document"]
    assert isinstance(document, dict)
    assert set(document.get("properties", {})) == {"vendor", "total"}


@pytest.mark.asyncio
async def test_the_schema_file_is_removed_when_the_run_ends() -> None:
    """A 0700 directory created for the spawn and torn down after it. Left
    behind, every typed run leaks a file naming the caller's data model."""
    cli = FakeCodexCli.script(codex_turn(text='{"vendor": "A", "total": 1}', usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    await drive(cog, agent=Agent(name="r", cognition=cog, output=Invoice))

    path = schema_path_of(cli)
    assert path is not None
    assert not still_there(path), "the schema file outlived the run"
    assert not still_there(path.parent), "the 0700 scratch directory outlived the run"


@pytest.mark.asyncio
async def test_an_explicit_json_schema_wins_over_the_agents_output() -> None:
    """The override, for the caller who wants a shape the Python type does not
    express — a stricter enum, an additionalProperties clause."""
    explicit = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    cli = PeekingCodexCli.script(codex_turn(text='{"n": 3}', usage=(1, 0, 1)))
    cog = CodexCliCognition(json_schema=explicit, spawn=cli)
    result = final_of(await drive(cog, agent=Agent(name="r", cognition=cog)))

    assert cli.seen["document"] == explicit
    # No agentkit adapter exists (the agent declared no ``output=``), so the
    # validated dict IS the parsed value — there is no Python type to build.
    assert result.parsed == {"n": 3}


@pytest.mark.asyncio
async def test_no_schema_means_no_flag_and_no_parsed_value() -> None:
    cli = FakeCodexCli.script(codex_turn(text="just prose", usage=(1, 0, 1)))
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert "--output-schema" not in cli.invocations[-1].argv
    assert result.parsed is None
    assert "structured_output" not in result.evals


# ─────────────────────────────────────────────────────────────────────────────
# 2. the answer comes back
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_fenced_json_answer_is_still_read() -> None:
    """A model told to answer in JSON will sometimes wrap it in a ``` fence.
    That is a formatting habit, not a failure to comply, and it is the one
    accommodation made here."""
    cli = FakeCodexCli.script(
        codex_turn(text='```json\n{"vendor": "Acme", "total": 3.0}\n```', usage=(1, 0, 1))
    )
    cog = CodexCliCognition(spawn=cli)
    result = final_of(await drive(cog, agent=Agent(name="r", cognition=cog, output=Invoice)))
    assert isinstance(result.parsed, Invoice)
    assert result.parsed.vendor == "Acme"


@pytest.mark.asyncio
async def test_prose_instead_of_json_is_reported_not_repaired() -> None:
    """Scanning for the first ``{`` and hoping was considered and rejected: it
    turns "the model ignored the schema and explained itself" — which the caller
    needs to see — into a confident parse of whatever JSON-shaped fragment
    happened to be in the explanation."""
    cli = FakeCodexCli.script(
        codex_turn(text="I could not find an invoice in this repo, so here is what I did instead.", usage=(1, 0, 1))
    )
    cog = CodexCliCognition(spawn=cli)
    result = final_of(await drive(cog, agent=Agent(name="r", cognition=cog, output=Invoice)))

    assert result.parsed is None
    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_missing"
    assert result.stop_reason == "invalid_output"
    # The prose is kept: the run happened and its text is real.
    assert "could not find an invoice" in result.output
    assert "not the JSON" in result.evals["structured_output_error"]


@pytest.mark.asyncio
async def test_an_empty_answer_against_a_schema_is_a_failure() -> None:
    """Without this branch the run returns ``partial=False`` and
    ``parsed=None``, and a caller who declared ``output=Invoice`` reads the
    silence as if the object had simply not been wired."""
    cli = FakeCodexCli.script(codex_turn(text="", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    result = final_of(await drive(cog, agent=Agent(name="r", cognition=cog, output=Invoice)))
    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_missing"
    assert "no final message" in result.evals["structured_output_error"]


@pytest.mark.asyncio
async def test_valid_json_of_the_wrong_shape_is_a_mismatch_not_a_miss() -> None:
    """Different stop reason on purpose: "the model answered in JSON but got the
    fields wrong" and "the model did not answer in JSON at all" have different
    fixes — tighten the schema versus tighten the prompt."""
    cli = FakeCodexCli.script(codex_turn(text='{"vendor": "Acme"}', usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    result = final_of(await drive(cog, agent=Agent(name="r", cognition=cog, output=Invoice)))

    assert result.parsed is None
    assert result.partial is True
    assert result.evals["stop_reason"] == "structured_output_mismatch"
    assert result.stop_reason == "invalid_output"
    # The raw value is still handed back — a caller may be able to use it even
    # though the declared type could not be built.
    assert result.evals["structured_output"] == {"vendor": "Acme"}
    assert "total" in result.evals["structured_output_error"]


@pytest.mark.asyncio
async def test_a_structured_failure_does_not_mask_a_worse_one() -> None:
    """A run that also died has to report THAT. The stop-reason priority puts
    the process failure above the shape failure, because a caller looking at
    ``structured_output_missing`` on a run whose CLI exited 137 would go and
    tighten a prompt."""
    from agentkit.testing.fakes import CliRun

    cli = FakeCodexCli([CliRun.of(codex_turn(text="", usage=(1, 0, 1)), returncode=137)])
    cog = CodexCliCognition(spawn=cli)
    result = final_of(await drive(cog, agent=Agent(name="r", cognition=cog, output=Invoice)))
    assert result.evals["stop_reason"] == "cli_exit_137"


@pytest.mark.asyncio
async def test_an_output_last_message_path_is_passed_through(tmp_path: Path) -> None:
    """Not needed by this cognition — the JSON stream carries the answer — and
    exposed because a caller may want the file for something else."""
    out = tmp_path / "answer.json"
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    await drive(CodexCliCognition(output_last_message=out, spawn=cli))
    argv = list(cli.invocations[-1].argv)
    assert argv[argv.index("--output-last-message") + 1] == str(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. per-turn schemas, which the Claude session cannot do
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_session_turn_can_carry_its_own_output_type() -> None:
    """The advantage of resuming rather than holding a process:
    ``--output-schema`` is chosen at spawn and every turn IS a spawn, so an
    ``output=``-carrying agent may be passed to any turn. ``ClaudeCliSession``
    has to refuse the same thing, because there the flag is fixed for the life
    of one process."""
    cli = FakeCodexCli(
        [
            FakeCodexCli.script(codex_turn(text="hello", usage=(1, 0, 1)))._runs[0],
            FakeCodexCli.script(
                codex_turn(text='{"vendor": "Acme", "total": 9.0}', thread_id=None, usage=(1, 0, 1))
            )._runs[0],
        ]
    )
    cog = CodexCliCognition(spawn=cli)
    typed = Agent(name="r", cognition=cog, output=Invoice)

    async with cog.session() as chat:
        plain = [ev async for ev in chat.turn("say hello")][-1].result
        structured = [ev async for ev in chat.turn("now the invoice", agent=typed)][-1].result

    assert plain.output == "hello"
    assert plain.parsed is None
    assert isinstance(structured.parsed, Invoice)
    assert "--output-schema" not in cli.invocations[0].argv
    assert "--output-schema" in cli.invocations[1].argv


# ─────────────────────────────────────────────────────────────────────────────
# 4. against the real binary
# ─────────────────────────────────────────────────────────────────────────────


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_real_cli_returns_a_typed_object(tmp_path: Path) -> None:
    """One real call proving the whole path: the schema is written where the CLI
    can read it, the CLI constrains its answer to it, and the answer becomes a
    Python object."""
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
    )
    agent = Agent(name="reader", cognition=cog, output=Invoice)
    result = final_of(
        await drive(cog, agent=agent, task="The vendor is Acme and the total is 12.50. Report it.")
    )

    assert isinstance(result.parsed, Invoice), result.evals
    assert result.parsed.vendor == "Acme"
    assert result.parsed.total == pytest.approx(12.50)


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_real_cli_accepts_the_schema_path_we_write(tmp_path: Path) -> None:
    """Narrower than the test above and it fails differently: this one only
    asserts the CLI did not reject the flag at startup, which is what a schema
    written in the wrong place or the wrong dialect looks like."""
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    result = final_of(await drive(cog, task="Answer 'ok'."))
    assert result.evals["cli_return_code"] == 0, result.evals
    assert result.evals.get("stop_reason") not in {"spawn_failed", "cli_exit_1"}, result.evals
