"""`FakeCodexCli` — an offline double for the ``codex`` CLI path.

The sibling of :class:`~agentkit.testing.fakes.FakeClaudeCli`, at the same seam
and for the same reasons: :class:`~agentkit.agents.cognition.CodexCliCognition`
spawns a binary and parses its JSONL stdout, so without a double a test of
anything Codex-shaped either spends real money or needs a real ``codex``
install, and the interesting failures on that path (a truncated line, a
non-zero exit mid-answer, a ``turn.completed`` that never arrives, a
``turn.failed``) have no test at all because nothing can produce them.

    from agentkit.testing.fakes import FakeCodexCli, codex_turn

    cli = FakeCodexCli.script(codex_turn(text="done", usage=(1200, 1000, 40)))
    cognition = CodexCliCognition(spawn=cli)

The design argument for putting the double at the spawn seam rather than one
level up — and for injecting a callable rather than generating a fake binary —
is made once, in ``FakeClaudeCli``'s module docstring, and applies here
verbatim. Two points from it are worth repeating because they shape how tests
here are written:

* **Exhaustion is loud.** Asking for a spawn the recording does not have raises
  :class:`~agentkit.testing.ScriptExhausted`, a ``BaseException``, so it
  escapes the cognition's terminal-event guarantee instead of arriving as a
  tidy ``AgentResult(partial=True)``. A recording is a claim about how many
  times the binary is invoked, and this cognition invokes it once per TURN, so
  the claim is load-bearing here in a way it is not for Claude: a Codex session
  spawns per turn, and a session that spawns twice for one ``turn()`` has a
  defect.
* **There is no suspension point.** Reading a line from this double never
  awaits, so a task reading a replayed stdout runs to EOF without another task
  getting a turn. ``asyncio.wait_for`` cannot fire mid-stream; cancellation
  through this double is ``ctx.check_cancelled()``, which the cognition polls
  once per line and which does work.

WHY THERE IS A ``codex_turn`` BUILDER AND NOT JUST ``script``
------------------------------------------------------------
A well-formed Codex turn is five payloads in a fixed order — ``thread.started``,
``turn.started``, the items, ``turn.completed`` — and the order carries meaning
the cognition depends on: the thread id arrives before anything else (it is what
a session resumes), and the usage arrives last (it is what the budget is charged
from). Hand-writing that in every test is both tedious and the place a test
accidentally asserts against a stream the real binary would never emit.

It matters most for the multi-turn tests. ``CodexCliCognition`` sessions resume
rather than hold a process, so a three-turn conversation is three spawns and
therefore three complete turns of payloads — fifteen dicts written by hand,
where the interesting part of the test is one ``resume`` argument in an argv.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from agentkit.testing.fakes._cli_replay import (
    CliInvocation,
    CliRun,
    CliStderr,
    _exhausted_message,
    _ReplayProcess,
)
from agentkit.testing.fakes.llm import ScriptExhausted

__all__ = ["FakeCodexCli", "codex_turn"]

# The thread id a builder uses when a test does not care which one it gets.
# A real Codex thread id is a UUIDv7; this is shaped like one so a test that
# round-trips it through ``resume_session_id`` is exercising the same string
# length and character class a real one would.
DEFAULT_THREAD_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"


def codex_turn(
    *,
    text: str | None = None,
    reasoning: str | None = None,
    thread_id: str | None = DEFAULT_THREAD_ID,
    usage: tuple[int, int, int] | None = None,
    items: Iterable[Mapping[str, Any]] = (),
    failed: str | None = None,
    error: str | None = None,
    duration_ms: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """One complete, well-formed ``codex exec --json`` turn, as payloads.

    Emits the real event order: ``thread.started`` (unless ``thread_id=None``,
    which is how a test builds the resumed turn of a CLI version that does not
    re-announce the thread), ``turn.started``, one ``item.completed`` per item,
    then exactly one terminal ``turn.completed`` or ``turn.failed``.

    ``usage`` is ``(input_tokens, cached_input_tokens, output_tokens)`` in
    Codex's own convention — input INCLUSIVE of the cached prefix. Passing it
    that way rather than as agentkit's split is deliberate: the subtraction is
    the cognition's job, and a builder that pre-split it would make the one
    test that exists to pin that subtraction assert against its own arithmetic.

    ``reasoning`` and ``text`` are shorthand for the two items nearly every
    test wants; ``items`` takes the rest (a ``command_execution``, a
    ``file_change``, an ``mcp_tool_call``) as raw item dicts, appended after
    them. ``error`` inserts a top-level stream error before the terminal event —
    the shape the CLI uses for a broken pipe, as opposed to ``failed``, which is
    the turn itself going wrong.
    """
    out: list[dict[str, Any]] = []
    if thread_id is not None:
        out.append({"type": "thread.started", "thread_id": thread_id})
    out.append({"type": "turn.started"})
    n = 0
    if reasoning is not None:
        out.append({"type": "item.completed", "item": {"id": f"item_{n}", "type": "reasoning", "text": reasoning}})
        n += 1
    for item in items:
        entry = dict(item)
        entry.setdefault("id", f"item_{n}")
        out.append({"type": "item.completed", "item": entry})
        n += 1
    if text is not None:
        out.append({"type": "item.completed", "item": {"id": f"item_{n}", "type": "agent_message", "text": text}})
    if error is not None:
        out.append({"type": "error", "message": error})
    if failed is not None:
        out.append({"type": "turn.failed", "error": {"message": failed}})
    else:
        completed: dict[str, Any] = {"type": "turn.completed"}
        if usage is not None:
            inp, cached, output = usage
            completed["usage"] = {
                "input_tokens": inp,
                "cached_input_tokens": cached,
                "output_tokens": output,
            }
        if duration_ms is not None:
            completed["duration_ms"] = duration_ms
        out.append(completed)
    return tuple(out)


class FakeCodexCli:
    """Deterministic offline ``codex`` CLI. Pass it as ``spawn=`` to a
    :class:`~agentkit.agents.cognition.CodexCliCognition`.

    Holds an ordered list of :class:`~agentkit.testing.fakes.CliRun`\\ s — one
    per **spawn**. That unit is worth being precise about, because it differs
    from the Claude double's in the way that matters for writing tests:
    ``codex exec`` is one-shot, so a spawn is a TURN. ``drive()`` consumes one
    run; a three-turn :class:`~agentkit.agents.cognition.CodexCliSession`
    consumes three.

    Each spawn consumes the next run; asking for one the recording does not
    have raises :class:`~agentkit.testing.ScriptExhausted`. ``repeat_last=True``
    replays the final run forever, which is the shape a session test that does
    not care how many turns it takes wants.
    """

    def __init__(self, runs: Sequence[CliRun] = (), *, repeat_last: bool = False) -> None:
        self._runs = tuple(runs)
        self._repeat_last = repeat_last
        self._cursor = 0
        self.invocations: list[CliInvocation] = []

    # ---- construction ------------------------------------------------------

    @classmethod
    def replay(cls, path: Path | str, *more: Path | str, repeat_last: bool = False) -> FakeCodexCli:
        """Replay recorded JSONL session file(s), one spawn per file.

        The bytes are read **now**, once, and held immutably — not streamed off
        disk per spawn. A recording that vanished or changed between
        construction and the spawn three layers down would surface as
        ``evals["error"]`` inside the cognition's ``except BaseException``,
        where a missing fixture is nearly unreadable; and "replays
        byte-identically twice" is trivially true of an immutable buffer and
        merely probable of a file. Streaming happens on the other side — the
        buffer is handed to the consumer one line at a time.
        """
        paths = [Path(p) for p in (path, *more)]
        return cls([CliRun(stdout=p.read_bytes()) for p in paths], repeat_last=repeat_last)

    @classmethod
    def script(
        cls,
        payloads: Iterable[Mapping[str, Any] | CliStderr | str | bytes],
        *,
        stderr: bytes = b"",
        returncode: int = 0,
        repeat_last: bool = False,
    ) -> FakeCodexCli:
        """Assemble ONE spawn's output from payloads — the case nobody recorded.

        One spawn rather than one-per-payload because that is what the shapes
        this is for need: a refusal, a stream error and a malformed final answer
        are all a single CLI invocation that went wrong. Several spawns with
        different exit codes are ``FakeCodexCli([CliRun.of(...), ...])``.

        See :meth:`~agentkit.testing.fakes.CliRun.of` for how a payload becomes
        bytes — in particular that a raw ``str``/``bytes`` is written verbatim,
        which is how a truncated or non-JSON line is built.
        """
        return cls([CliRun.of(payloads, stderr=stderr, returncode=returncode)], repeat_last=repeat_last)

    @classmethod
    def answering(cls, *texts: str, thread_id: str = DEFAULT_THREAD_ID) -> FakeCodexCli:
        """One spawn per text, each a complete well-formed turn saying it.

        The shorthand for a multi-turn session test, where the conversation's
        CONTENT is not what is under test — the argv, the thread id, the stop
        reason or the metering is — and five hand-written payloads per turn are
        five chances to write a stream the binary would not emit::

            cli = FakeCodexCli.answering("first", "second", "third")

        Every turn re-announces the same ``thread_id``, which is what a real
        resumed run does; a session that failed to thread its id would show up
        as a missing ``resume`` in ``invocations[1].argv``, not as a mismatched
        id here.
        """
        return cls([CliRun.of(codex_turn(text=t, thread_id=thread_id)) for t in texts])

    # ---- the seam ----------------------------------------------------------

    async def __call__(
        self,
        *argv: str,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        **_ignored: Any,
    ) -> asyncio.subprocess.Process:
        """Stand in for ``asyncio.create_subprocess_exec``.

        ``**_ignored`` swallows ``stdin``/``stdout``/``stderr`` pipe modes and
        whatever flags a future call site adds: they describe how the REAL
        implementation should wire file descriptors, and there are none here.
        """
        run = self._next_run()
        invocation = CliInvocation(argv=tuple(argv), cwd=cwd, env=dict(env or {}))
        self.invocations.append(invocation)
        # Structurally a Process for every attribute the cognition touches
        # (stdout/stderr/stdin/returncode/wait/terminate/kill) and nothing
        # more. The cast is the honest spelling of that: there is no Protocol
        # for "the part of Process this module uses", and inventing one would
        # put a type in the production module purely to describe a test double.
        return cast("asyncio.subprocess.Process", _ReplayProcess(run, invocation))

    def _next_run(self) -> CliRun:
        if self._cursor >= len(self._runs):
            if not self._repeat_last or not self._runs:
                raise ScriptExhausted(
                    _exhausted_message(double="FakeCodexCli", have=len(self._runs), idx=self._cursor)
                )
            # Incremented past the end too, so a ``repeat_last`` test can
            # assert HOW MANY spawns a loop made, not merely that it made
            # extra ones.
            self._cursor += 1
            return self._runs[-1]
        run = self._runs[self._cursor]
        self._cursor += 1
        return run

    # ---- inspection --------------------------------------------------------

    @property
    def remaining(self) -> int:
        """Runs not yet spawned. Negative is impossible; zero after the last
        spawn is the normal end of a fully-consumed recording."""
        return max(0, len(self._runs) - self._cursor)

    @property
    def spawns(self) -> int:
        return len(self.invocations)

    def argv(self, index: int = -1) -> tuple[str, ...]:
        """One spawn's argv, defaulting to the most recent.

        A convenience with a purpose: the flag assertions in this cognition's
        tests are all "what did spawn N ask for", and
        ``cli.invocations[-1].argv`` reads as plumbing at every one of them.
        """
        return self.invocations[index].argv
