"""The tool-event surface of a Codex turn, and the hole it is honest about.

WHY THIS FILE EXISTS
--------------------
A Codex tool item is announced twice — ``item.started`` then
``item.completed`` for a shell command, once only for a patch or a web search —
and a consumer has to see one ``tool_call`` and one ``tool_result`` from that
regardless of which shape arrived. Emitting the call on both phases does not
raise and does not change the answer; it doubles every number an observer
counts tool calls with, which is the kind of defect that is discovered in a
billing dispute rather than in a test run.

The other half of the file pins the warning. ``CodexCliCognition`` bypasses the
``Invoker``, so no middleware runs for a native CLI tool call: no egress or SSRF
check, no input guard, no audit record, no idempotency key. The failure is
invisible by construction, which is exactly why the cognition has to NAME the
middlewares that will not apply — and name them once, because a warning that
repeats per iteration is what teaches people to silence it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.contracts.types import StreamEvent
from agentkit.core.testing.fakes import FakeCodexCli, codex_turn
from agentkit.models.providers.cli.codex.driver import CodexCliCognition


@dataclass
class _Agent:
    prompt: str | None = None


@dataclass
class _Invoker:
    """Just the attribute the bypass warning reads off a context's invoker."""

    tool_middleware: tuple[Any, ...] = ()


@dataclass
class _Ctx:
    """A structural context: an invoker and the cancel poll, nothing else."""

    invoker: Any = None
    all_meters: tuple[Any, ...] = ()
    correlation_id: str = "run-1"

    def check_cancelled(self) -> None:
        return None


@dataclass
class _Egress:
    """A middleware that only has to be nameable."""

    name: str = "Egress"


async def _drive(cognition: CodexCliCognition, *, ctx: Any = None) -> list[StreamEvent]:
    return [ev async for ev in cognition.drive(_Agent(), "go", ctx, None)]


def _final(events: list[StreamEvent]) -> Any:
    assert [e.type for e in events].count("final") == 1, [e.type for e in events]
    return events[-1].result


def _calls(events: list[StreamEvent]) -> list[Any]:
    return [e.tool_call for e in events if e.type == "tool_call"]


def _results(events: list[StreamEvent]) -> list[Any]:
    return [e for e in events if e.type == "tool_result"]


@pytest.mark.asyncio
async def test_a_tool_item_announced_twice_still_emits_exactly_one_tool_call() -> None:
    """A shell command arrives as a ``started``/``completed`` pair and a patch
    arrives only completed, so the call event has to be emitted by whichever
    phase gets there first — and by only one of them. Emitting on both leaves
    the answer and the result text correct while every observer counting tool
    calls, every audit tally and every per-tool rate limit reads double."""
    cli = FakeCodexCli.script(
        (
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.started"},
            {"type": "item.started", "item": {"id": "c1", "type": "command_execution", "command": "ls -l"}},
            {
                "type": "item.completed",
                "item": {
                    "id": "c1",
                    "type": "command_execution",
                    "command": "ls -l",
                    "aggregated_output": "a\nb",
                    "exit_code": 0,
                },
            },
            {"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}},
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))

    calls = _calls(events)
    assert len(calls) == 1, [c.name for c in calls]
    assert calls[0].name == "shell"
    assert calls[0].arguments == {"command": "ls -l"}
    assert [r.tool_result for r in _results(events)] == ["a\nb"]
    assert _final(events).partial is False


@pytest.mark.asyncio
async def test_a_command_that_failed_silently_still_reports_its_exit_code() -> None:
    """A command that fails with no output has an empty ``aggregated_output``
    and a non-zero code, so without the code appended the ``tool_result`` is the
    empty string — a failure that reads exactly like a success that printed
    nothing. The code is appended only on failure, because on success it is
    noise on every single command."""
    cli = FakeCodexCli.script(
        codex_turn(
            text="gave up",
            usage=(1, 0, 1),
            items=[
                {"id": "c1", "type": "command_execution", "command": "nope", "aggregated_output": "", "exit_code": 127}
            ],
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))

    assert [r.tool_result for r in _results(events)] == ["[exit 127]"]


@pytest.mark.asyncio
async def test_an_mcp_tool_keeps_the_clis_own_name_so_an_audit_record_matches_a_transcript() -> None:
    """``mcp__<server>__<tool>`` is Codex's own naming convention. Renaming it on
    the way through would leave an agentkit audit record that cannot be joined
    to the Codex transcript of the same call, which is the one thing both
    records exist for."""
    cli = FakeCodexCli.script(
        codex_turn(
            text="read it",
            usage=(1, 0, 1),
            items=[
                {
                    "id": "t1",
                    "type": "mcp_tool_call",
                    "server": "files",
                    "tool": "read",
                    "arguments": {"path": "README.md"},
                    "result": {"content": [{"type": "text", "text": "# Title"}]},
                }
            ],
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))

    assert [c.name for c in _calls(events)] == ["mcp__files__read"]
    assert _calls(events)[0].arguments == {"path": "README.md"}
    # The MCP content-block envelope is flattened, not handed on as JSON.
    assert [r.tool_result for r in _results(events)] == ["# Title"]


@pytest.mark.asyncio
async def test_a_patch_item_names_the_files_it_touched_rather_than_dumping_a_diff() -> None:
    """``apply_patch`` is the tool whose effects outlive the run, so the one
    thing its result has to carry is WHICH paths changed. A patch item only ever
    completes — it has no ``started`` phase — which is why the call event cannot
    be owned by ``started`` alone."""
    cli = FakeCodexCli.script(
        codex_turn(
            text="patched",
            usage=(1, 0, 1),
            items=[
                {
                    "id": "p1",
                    "type": "file_change",
                    "changes": [{"kind": "add", "path": "a.py"}, {"kind": "update", "path": "b.py"}],
                }
            ],
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))

    assert [c.name for c in _calls(events)] == ["apply_patch"]
    assert [r.tool_result for r in _results(events)] == ["add a.py\nupdate b.py"]


@pytest.mark.asyncio
async def test_a_plan_update_reports_progress_rather_than_the_whole_plan() -> None:
    """``update_plan`` fires repeatedly through a long run. The useful signal is
    how much of the plan is done, which is what distinguishes a run that is
    working from one that is looping."""
    cli = FakeCodexCli.script(
        codex_turn(
            text="planned",
            usage=(1, 0, 1),
            items=[
                {
                    "id": "l1",
                    "type": "todo_list",
                    "items": [{"completed": True}, {"completed": False}, {"completed": False}],
                }
            ],
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))

    assert [e.text for e in events if e.type == "step"] == ["plan:1/3"]


@pytest.mark.asyncio
async def test_a_web_search_is_reported_as_the_tool_call_the_chain_never_saw() -> None:
    """The CLI's own ``web_search`` is the tool that brings untrusted content
    into the session, and it runs inside the CLI's process where no egress or
    allowlist check reaches it. Surfacing it as a ``tool_call`` is the only
    record an observer gets that it happened at all."""
    cli = FakeCodexCli.script(
        codex_turn(
            text="found it",
            usage=(1, 0, 1),
            items=[{"id": "w1", "type": "web_search", "query": "agentkit"}],
        )
    )
    cognition = CodexCliCognition(spawn=cli, web_search=True)
    events = await _drive(cognition)

    assert [c.name for c in _calls(events)] == ["web_search"]
    assert _calls(events)[0].arguments == {"query": "agentkit"}
    # The warning the cognition raises has to be able to name it.
    assert "web_search" in cognition.native_tools()


def test_the_native_tool_list_reports_apply_patch_even_under_a_read_only_sandbox() -> None:
    """The tool is PRESENT and the model still calls it; the sandbox is what
    makes the write fail. Dropping it from the list would tell a reader the
    middleware chain has nothing to miss, when what actually happens is an
    attempted edit the chain never saw."""
    assert CodexCliCognition(sandbox="read-only").native_tools() == ("apply_patch", "shell", "update_plan")


@pytest.mark.asyncio
async def test_the_first_drive_on_a_context_with_tool_middleware_names_what_will_not_run() -> None:
    """The bypass is invisible by construction: the CLI runs ``shell`` inside its
    own process, so there is no failing call to notice and no audit record whose
    absence shows up. A generic "middleware may not apply" would leave the
    reader no better off, so the warning names both halves — the middlewares and
    the tools they are not applying to."""
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(1, 0, 1)))
    cognition = CodexCliCognition(spawn=cli)
    ctx = _Ctx(invoker=_Invoker(tool_middleware=(_Egress(),)))

    with pytest.warns(UserWarning) as caught:
        await _drive(cognition, ctx=ctx)

    message = str(caught[0].message)
    assert "Egress" in message
    assert "shell" in message
    assert "sandbox='read-only'" in message


@pytest.mark.asyncio
async def test_the_bypass_warning_is_latched_per_instance_so_a_loop_does_not_train_people_to_silence_it() -> None:
    """A warning that repeats on every iteration of an agent loop is noise, and
    noise is what gets a ``filterwarnings`` line added that then hides the next
    one. Latched per INSTANCE rather than per process, because two cognitions in
    one service are usually two configurations and the second is the one nobody
    has audited."""
    cli = FakeCodexCli.answering("one", "two")
    cognition = CodexCliCognition(spawn=cli)
    ctx = _Ctx(invoker=_Invoker(tool_middleware=(_Egress(),)))

    with pytest.warns(UserWarning):
        await _drive(cognition, ctx=ctx)
    # A second drive on the SAME instance must say nothing; `filterwarnings =
    # ["error"]` turns a repeat into this test's failure.
    await _drive(cognition, ctx=ctx)

    assert cli.spawns == 2


@pytest.mark.asyncio
async def test_a_context_with_no_middleware_is_not_warned_about_a_chain_it_does_not_have() -> None:
    """Silent when there is nothing being skipped. A safety warning that fires
    for every run is indistinguishable from a decoration, and the whole point of
    this one is that it marks a configuration worth auditing."""
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(1, 0, 1)))
    # No ``pytest.warns``: `filterwarnings = ["error"]` makes a stray warning
    # here the failure, which is the assertion.
    events = await _drive(CodexCliCognition(spawn=cli), ctx=_Ctx(invoker=_Invoker()))

    assert _final(events).output == "ok"
