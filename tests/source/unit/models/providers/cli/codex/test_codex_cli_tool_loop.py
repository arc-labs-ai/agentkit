"""What bounds the tool loop INSIDE one ``codex exec`` turn, and what a failed tool call looks like.

WHY THIS FILE EXISTS
--------------------
``codex exec`` is one spawn per agentkit turn, and everything the model does
lives inside it: the model calls a tool, the CLI runs it, the model calls
another, and nothing in agentkit sees a turn boundary in between.
``CallLimits.max_turns`` counts ADAPTER turns, so it counted ``1`` while the
following happened live (codex-cli 0.162.1, ``gpt-4o-mini``, the live
execution-contract gate scenario):

* 126 tool calls in that single turn, every one of them byte-identical —
  ``read_mcp_resource(server="mcp__agentkit", …)``, the CLI's own MCP resource
  shim, called with the tool NAMESPACE name where the CLI wanted the server
  name, failing the same way every time;
* ``max_turns=8`` in force throughout, because the runtime had seen one turn;
* the run ended only when the 180s wall clock killed it, with
  ``outcome="limit_reached"`` and no answer.

So the wall clock was the only real bound, and a wall clock prices a loop in
seconds while the invoice is in tool calls. The tests below are about making the money
stop: a declared cap the driver can actually enforce, and a no-progress guard
that fires without anyone declaring anything — where "no progress" means the
same call returning the same answer, because the first attempt at this counted
only FAILURES and the live re-run showed the loop was half successes.

The same run exposed the second subject here. All 126 of those failures reached
the host's event stream as ``is_error=False`` with an EMPTY
``result_preview``, because the driver handed the ``tool_result`` event a bare
string: ``CliAdapter`` reads ``.text`` / ``.is_error`` off that payload (the
shape the SDK path's ``ToolResult`` has), and a ``str`` has neither. A host
watching the stream could therefore not tell a refused call from a successful
one — which is exactly the distinction the live gate test exists to make, and
the one a security reviewer would rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.contracts.types import StreamEvent
from agentkit.core.testing.fakes import FakeCodexCli, codex_turn
from agentkit.models.contracts.constants import StopReason
from agentkit.models.providers.cli.codex.driver import CodexCliCognition


@dataclass
class _Agent:
    """The only attributes this cognition reads off an agent."""

    prompt: str | None = None
    model: str | None = None


async def _drive(cognition: CodexCliCognition) -> list[StreamEvent]:
    return [ev async for ev in cognition.drive(_Agent(), "go", None, None)]


def _final(events: list[StreamEvent]) -> Any:
    assert [e.type for e in events].count("final") == 1, [e.type for e in events]
    assert events[-1].type == "final"
    return events[-1].result


def _tool_calls(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "tool_call"]


def _tool_results(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "tool_result"]


def _failing_shim_call(uri: str = "r") -> dict[str, Any]:
    """One repetition of the loop that was measured live.

    The arguments are identical every time and the id is not, which is why the
    no-progress guard keys on the call rather than on the item: the CLI numbers
    every repetition ``item_1``, ``item_2``, … and a guard that deduped on the
    id would see 126 distinct calls.
    """
    return {
        "type": "mcp_tool_call",
        "server": "mcp__agentkit",
        "tool": "read_mcp_resource",
        "arguments": {"server": "mcp__agentkit", "uri": uri},
        "result": None,
        "error": {"message": "resources/read failed: unknown MCP server 'mcp__agentkit'"},
        "status": "failed",
    }


def _working_call(n: int) -> dict[str, Any]:
    """A tool call that did something, so a cap can be told from a loop."""
    return {
        "type": "mcp_tool_call",
        "server": "agentkit",
        "tool": "read_dispatch_code",
        "arguments": {"record": f"record-{n}"},
        "result": {"content": [{"type": "text", "text": f"code-{n}"}]},
        "status": "completed",
    }


# ── a failed tool call is an error, not an empty success ─────────────────────


@pytest.mark.asyncio
async def test_a_tool_call_the_cli_refused_reaches_the_host_as_an_error_carrying_its_reason() -> None:
    """A refusal that arrives as ``is_error=False`` with no text is a refusal nobody can audit.

    ``CliAdapter`` builds every TOOL_RESULT event's ``is_error``,
    ``result_preview``, ``result_size`` and ``result_digest`` by reading
    ``.text`` / ``.is_error`` / ``.overflowed`` off this payload — the shape the
    SDK path's ``ToolResult`` has. A bare ``str`` has none of those attributes,
    so every CLI tool result reached a host as ``is_error=False``,
    ``result_preview=""``, ``result_size=0``: a denied call and a successful one
    were the same event.

    That is not cosmetic. ``approval_policy="never"`` makes the Codex CLI refuse
    an approval-gated MCP tool ITSELF — measured verbatim, codex-cli 0.162.1:
    ``"MCP tool call requires approval, but approval policy is never"`` — before
    the call reaches agentkit's server, so this payload is the ONLY place a host
    can see that a gated tool was stopped rather than run.
    """
    cli = FakeCodexCli.script(
        codex_turn(
            items=[
                {
                    "type": "mcp_tool_call",
                    "server": "agentkit",
                    "tool": "commit_dispatch",
                    "arguments": {"record": "r"},
                    "result": None,
                    "error": {"message": "MCP tool call requires approval, but approval policy is never"},
                    "status": "failed",
                }
            ],
            text="blocked",
            usage=(10, 0, 5),
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))

    (result_event,) = _tool_results(events)
    payload = result_event.tool_result
    assert payload.is_error is True, "a refused tool call must not read as a success"
    assert "requires approval" in payload.text, payload.text
    # Still a string, so every existing consumer of this event — and the Claude
    # driver, which hands out plain strings — is unaffected by the two
    # attributes being there.
    assert payload == "MCP tool call requires approval, but approval policy is never"
    assert isinstance(payload, str)


@pytest.mark.asyncio
async def test_a_shell_command_that_exited_non_zero_is_flagged_without_losing_its_output() -> None:
    """The exit code decides ``is_error``; the output is still the text.

    A failing command is the one whose output a reader needs most, so flagging
    it must not replace it — and a flag derived from the exit code is the only
    signal available: the CLI does not mark the item itself.
    """
    cli = FakeCodexCli.script(
        codex_turn(
            items=[
                {
                    "type": "command_execution",
                    "command": "pytest -q",
                    "aggregated_output": "1 failed",
                    "exit_code": 1,
                    "status": "completed",
                }
            ],
            text="done",
            usage=(10, 0, 5),
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))

    (result_event,) = _tool_results(events)
    assert result_event.tool_result.is_error is True
    assert "1 failed" in result_event.tool_result.text
    assert "[exit 1]" in result_event.tool_result.text


@pytest.mark.asyncio
async def test_a_tool_call_that_worked_is_not_flagged_as_an_error() -> None:
    """The direction that keeps the flag worth reading.

    A guard that marked everything would make ``is_error`` useless in exactly
    the same way as one that marked nothing — and it is the shape this fix
    would rot into first, because every item type has to opt in to the flag
    separately.
    """
    cli = FakeCodexCli.script(codex_turn(items=[_working_call(1)], text="code-1", usage=(10, 0, 5)))
    events = await _drive(CodexCliCognition(spawn=cli))

    (result_event,) = _tool_results(events)
    assert result_event.tool_result.is_error is False
    assert result_event.tool_result.text == "code-1"
    assert _final(events).stop_reason == StopReason.COMPLETE


# ── the no-progress guard, on by default ─────────────────────────────────────


@pytest.mark.asyncio
async def test_the_same_failing_tool_call_repeated_is_stopped_before_it_can_run_a_hundred_more() -> None:
    """The measured runaway: 126 identical failing calls, and a wall clock as the only bound.

    Each repetition is a real model turn (Codex sends
    ``parallel_tool_calls: false`` — captured off the wire), so each one is
    billed input and output tokens, and nothing in the stream says the model is
    stuck. The host's only lever was ``wall_clock_s``, which prices a loop in
    seconds rather than in calls.

    The guard is deliberately narrow: identical name, identical arguments and
    an identical result. Nothing was learned between those calls, so there is
    no reading of "making progress" that includes them — which is what makes
    it safe to have on by default, where a plain cap on tool calls would kill
    a legitimately long coding run.
    """
    cli = FakeCodexCli.script(
        codex_turn(items=[_failing_shim_call() for _ in range(60)], text="never reached", usage=(999, 0, 999))
    )
    cognition = CodexCliCognition(spawn=cli, max_repeated_tool_calls=4)
    events = await _drive(cognition)
    result = _final(events)

    assert len(_tool_calls(events)) == 4, "the guard let the loop keep going"
    assert result.stop_reason == StopReason.FAILED
    assert result.evals["stop_reason"] == StopReason.ERROR_MAX_TURNS
    assert result.evals["tool_call_bound"] == "max_repeated_tool_calls"
    assert result.evals["tool_call_limit"] == 4
    assert result.evals["tool_calls"] == 4
    assert "read_mcp_resource" in result.evals["tool_call_repeated"]
    # Reading STOPS, which is the half that can be observed offline: this
    # double's process exits the instant it is waited on, so the shared
    # driver's terminate-the-group step (the half that stops a real CLI from
    # going on spending) has nothing left to signal. Six lines of sixty-four.
    assert cli.invocations[0].lines_read == 6, "the driver kept reading past the bound"
    # The tail of the recording — the answer and the usage payload — was never
    # read, which is the evidence that the run really ended at the cap.
    assert result.output == ""
    assert result.partial is True


@pytest.mark.asyncio
async def test_a_call_repeated_with_a_changing_answer_is_progress_however_often_it_is_made() -> None:
    """The false positive that would break real runs, and it is the likely one.

    An agent polling a build, watching a log, or waiting on a port runs the
    SAME command with the SAME arguments over and over, and that is not a
    runaway: each result is something the model did not have. The result is
    therefore part of the fingerprint, and a guard that is on by default has to
    get this direction right or it is worse than no guard at all.
    """
    cli = FakeCodexCli.script(
        codex_turn(
            items=[
                {
                    "type": "command_execution",
                    "command": "tail -n1 build.log",
                    "aggregated_output": f"compiling unit {n}",
                    "exit_code": 0,
                    "status": "completed",
                }
                for n in range(40)
            ],
            text="built",
            usage=(10, 0, 5),
        )
    )
    cognition = CodexCliCognition(spawn=cli, max_repeated_tool_calls=4)
    events = await _drive(cognition)
    result = _final(events)

    assert len(_tool_calls(events)) == 40
    assert result.output == "built"
    assert result.stop_reason == StopReason.COMPLETE
    assert "tool_call_bound" not in result.evals


@pytest.mark.asyncio
async def test_a_call_that_keeps_succeeding_with_the_same_empty_answer_is_still_a_loop() -> None:
    """The half a failure-only guard let through, and it was the expensive half.

    Measured on the second live run, after the guard counted failures only: the
    model alternated a failing ``read_mcp_resource`` with
    ``list_mcp_resources``, which SUCCEEDED every single time and returned
    ``{"resources":[]}`` — 127 times, 88 of them consecutively, one billed model
    turn each. A success that returns the same nothing it returned a moment ago
    is not progress, and a guard that waits for an error never fires on it.
    """
    empty_list = {
        "type": "mcp_tool_call",
        "server": "codex",
        "tool": "list_mcp_resources",
        "arguments": {},
        "result": {"content": [{"type": "text", "text": '{"resources":[]}'}]},
        "status": "completed",
    }
    cli = FakeCodexCli.script(
        codex_turn(items=[dict(empty_list) for _ in range(60)], text="never reached", usage=(999, 0, 999))
    )
    cognition = CodexCliCognition(spawn=cli, max_repeated_tool_calls=4)
    events = await _drive(cognition)
    result = _final(events)

    assert len(_tool_calls(events)) == 4
    assert result.evals["stop_reason"] == StopReason.ERROR_MAX_TURNS
    assert "list_mcp_resources" in result.evals["tool_call_repeated"]


@pytest.mark.asyncio
async def test_distinct_failing_calls_are_progress_even_though_they_all_fail() -> None:
    """A model recovering from failures is not a model stuck in one.

    Fifty failed calls with fifty different arguments is a search — bad at its
    job, maybe, but each failure is new information. Firing here would stop
    every exploratory run that touches a missing path, which is why the
    fingerprint includes the arguments and not just the name.
    """
    cli = FakeCodexCli.script(
        codex_turn(
            items=[_failing_shim_call(uri=f"uri-{n}") for n in range(50)],
            text="gave up",
            usage=(10, 0, 5),
        )
    )
    cognition = CodexCliCognition(spawn=cli, max_repeated_tool_calls=4)
    result = _final(await _drive(cognition))

    assert result.output == "gave up"
    assert result.stop_reason == StopReason.COMPLETE


# ── the declared cap ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_declared_max_turns_bounds_the_tool_calls_inside_the_one_codex_turn() -> None:
    """``max_turns`` has to mean something here or it must not be accepted here.

    Codex has no ``--max-turns`` flag and its in-turn loop is invisible to the
    runtime's turn counter, so a caller who sets ``max_turns=8`` and gets 126
    tool calls was told a bound was in force that was not. The driver counts
    the tool calls it sees — one per model turn, since Codex sends
    ``parallel_tool_calls: false`` — and ends the turn past the cap.

    The cap allows exactly ``max_turns`` calls and fires on the one after, the
    same way ``claude --max-turns`` does: a run that fits in its budget must
    not be reported as having blown it.
    """
    cli = FakeCodexCli.script(
        codex_turn(items=[_working_call(n) for n in range(10)], text="never reached", usage=(999, 0, 999))
    )
    cognition = CodexCliCognition(spawn=cli, max_turns=3)
    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == StopReason.ERROR_MAX_TURNS
    assert result.evals["tool_call_bound"] == "max_turns"
    assert result.evals["tool_call_limit"] == 3
    assert result.evals["tool_calls"] == 4, "the call that crossed the cap is the one we saw"
    # Two preamble payloads plus the four calls, and nothing after: the stream
    # was abandoned where the bound said, not read to the end and judged later.
    assert cli.invocations[0].lines_read == 6


@pytest.mark.asyncio
async def test_a_run_that_fits_inside_its_declared_cap_is_not_failed_for_reaching_it() -> None:
    """The off-by-one that would make the cap unusable.

    A host that sizes ``max_turns`` to the work ("two lookups, then answer")
    would otherwise have every successful run reported as
    ``error_max_turns`` — and the obvious response is to raise the cap, which
    takes the bound back off.
    """
    cli = FakeCodexCli.script(
        codex_turn(items=[_working_call(n) for n in range(3)], text="code-2", usage=(10, 0, 5))
    )
    result = _final(await _drive(CodexCliCognition(spawn=cli, max_turns=3)))

    assert result.output == "code-2"
    assert result.stop_reason == StopReason.COMPLETE
    assert "tool_call_bound" not in result.evals


@pytest.mark.asyncio
async def test_no_cap_is_declared_by_default_so_a_long_legitimate_run_is_not_cut_short() -> None:
    """``max_turns=None`` means what it says, and the no-progress guard is the
    only thing that fires without being asked for. A default numeric cap here
    would silently truncate the long autonomous runs this CLI exists for, and
    an operator would read the resulting ``error_max_turns`` as the model
    giving up."""
    cli = FakeCodexCli.script(codex_turn(items=[_working_call(n) for n in range(30)], text="done", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli)

    assert cognition.max_turns is None
    result = _final(await _drive(cognition))
    assert result.output == "done"
    assert result.stop_reason == StopReason.COMPLETE


@pytest.mark.parametrize("field", ["max_turns", "max_repeated_tool_calls"])
def test_a_bound_of_zero_or_less_is_refused_at_construction_rather_than_stopping_every_run(field: str) -> None:
    """A cap of 0 would end every turn on its first tool call, and a negative
    one reads as "off" while behaving the same way. Both are a typo, and the
    cheap place to find out is here rather than in a run that dies instantly
    with a stop reason that says the model hit a limit."""
    with pytest.raises(ValueError, match=field):
        CodexCliCognition(**{field: 0})
    with pytest.raises(ValueError, match=field):
        CodexCliCognition(**{field: -1})
