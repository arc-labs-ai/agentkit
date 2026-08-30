"""`FakeClaudeCli` — an offline double for the ``claude`` CLI path.

Every other fake in this package fakes a PORT. The CLI is not behind one:
:class:`~agentkit.agents.cognition.ClaudeCliCognition` spawns a binary and
parses its stream-json stdout, so before this existed a test of anything
CLI-shaped either spent real money or stood up a real ``claude`` install —
which is why the CLI tests in this repo carry a ``shutil.which("claude")``
skipif and why the interesting failures on that path (a truncated line, a
non-zero exit mid-answer, a ``result`` payload that never arrives) had no test
at all. Nothing could produce them.

    from agentkit.testing.fakes import FakeClaudeCli

    cli = FakeClaudeCli.replay(Path("sessions/adds_endpoint.jsonl"))
    cognition = ClaudeCliCognition(spawn=cli)

Two shapes. :meth:`FakeClaudeCli.replay` takes a recorded stream-json session
and replays it event for event, which is what turns one real dispatch into a
reusable fixture. :meth:`FakeClaudeCli.script` assembles payloads for the cases
nobody has recorded and nobody wants to: a refusal, a session limit, a
malformed final answer, a process killed halfway through a JSON object.


WHERE THE DOUBLE SITS, AND WHY IT IS NOT A BINARY
-------------------------------------------------
The spec that asked for this offered two seams: generate a fake executable and
point ``claude_bin=`` at it, or inject a transport seam. A generated executable
is more faithful — it exercises ``create_subprocess_exec``, real pipes, real
exit codes — and it was rejected for one reason that overrides all of that:
**a subprocess cannot raise into the test's stack.**

Exhaustion has to be loud (see :class:`~agentkit.testing.ScriptExhausted`,
reused here verbatim). A fake binary asked for a run its recording does not
have can only exit non-zero, and ``ClaudeCliCognition`` deliberately converts a
non-zero exit into ``AgentResult(partial=True, stop_reason="failed")`` and a
``final`` event. That is precisely the "stable, plausible, wrong answer" that
``ScriptExhausted`` was created to stop a test double from producing. The
double whose whole job is catching a loop that does not terminate would have
been the thing hiding it — again.

Three smaller reasons point the same way:

* **Determinism.** ``replay`` promises byte-identical output twice. A binary
  reintroduces an interpreter, a filesystem, a PATH, a ``chmod`` bit and a
  platform; the recording would be deterministic and the *transport* would not.
* **Speed.** A spawn is 30–80 ms. This package's fakes back most of a
  ~3700-test suite; the CLI ones would be the only fakes with a process cost.
* **Scope.** The existing CLI tests reach this seam with
  ``patch("agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec")``,
  which resolves through the module's ``asyncio`` reference to the real
  ``asyncio`` module and so disables subprocess spawning **process-wide** while
  held. An injected callable is scoped to one cognition instance.

What that costs, stated plainly: this double does not exercise
``create_subprocess_exec`` itself, OS-level argv handling, or real pipe
backpressure. Those belong to the ``@real_cli`` tests, which still exist and
still run against the binary when one is installed. Everything below the seam —
``_parse_line``, ``_events_from_payload``, ``_TurnState.fold``, the stop-reason
priority, the structured-output decision and the meter charge — is exercised
exactly as a real run exercises it, because the double supplies bytes and
nothing else. A double that returned finished ``AgentResult``s would cover none
of it, and all six of those are where this cognition's shipped bugs have been.

THE ONE LIMITATION THAT WILL SURPRISE YOU: NO SUSPENSION POINT
--------------------------------------------------------------
Reading a line from this double never awaits anything. A real
``StreamReader`` blocks on the pipe and hands the event loop back; this one
returns from an already-materialised buffer. So a task reading a replayed
stdout runs to EOF without another task getting a turn, and three things a
reader might reasonably expect do **not** happen by themselves:

* ``asyncio.gather(drive_a(), drive_b())`` runs *a* to completion and then
  *b*. Measured — 100 deltas from two runs, one changeover. If a test needs
  them genuinely interleaved, it awaits in its OWN consuming loop
  (``await asyncio.sleep(0)`` after each event); measured, the same pair then
  alternates on every event.
* ``asyncio.wait_for(drive(...), timeout=...)`` cannot fire mid-stream.
  Measured: a 0.05 s timeout around a 200 000-payload replay consumed all
  200 001 events and returned. **Cancellation through this double is
  ``ctx.check_cancelled()``**, which the cognition polls once per line and
  which does work — it terminates the replayed process and reports
  ``stop_reason="cancelled"``.
* :meth:`~agentkit.agents.cognition.ClaudeCliSession.interrupt` launched with
  ``asyncio.create_task`` does not get to run until the consumer awaits. Put
  one ``await asyncio.sleep(0)`` in the loop consuming ``turn()`` and the full
  round trip works, acknowledgement included.

A per-line yield inside the double would remove all three, and was rejected:
measured at 34x on the read path (89 ms → 3074 ms for 200 000 lines), paid by
every test, to buy something three of them want and can ask for themselves.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from agentkit.testing.fakes.llm import ScriptExhausted

__all__ = ["CliInvocation", "CliRun", "CliStderr", "FakeClaudeCli"]


@dataclass(frozen=True, slots=True)
class CliStderr:
    """A chunk the CLI wrote to **stderr** at this point in its stdout stream.

    Positional, not a lump attached to the run, because the cognition reads
    stderr exactly once — at the end, after stdout hits EOF. A diagnostic
    emitted in the middle of a long run therefore has to survive until then,
    and "did it?" is a real question: stderr is the only channel that explains
    a bare ``cli_exit_1`` to an operator. Placing chunks in the stream is what
    lets a test ask it.
    """

    data: bytes


@dataclass(frozen=True, slots=True)
class CliRun:
    """One spawn of the CLI, whole: its stdout, its stderr, its exit code.

    Frozen because it is the recording. Two ``FakeClaudeCli``s over one
    ``CliRun`` must not be able to affect each other, and the cheapest way to
    guarantee that is for the shared thing to be immutable and the per-spawn
    cursor to live in the iterator.
    """

    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int = 0
    # stderr chunks written PART WAY through stdout, as (stdout lines already
    # emitted, chunk). Built by ``of()``; a file recording has none, since a
    # ``.jsonl`` capture holds only the one stream.
    interleaved_stderr: tuple[tuple[int, bytes], ...] = ()

    @classmethod
    def of(
        cls,
        payloads: Iterable[Mapping[str, Any] | CliStderr | str | bytes],
        *,
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> CliRun:
        """Assemble a run from payloads.

        A ``Mapping`` is serialised as one JSON line — the ordinary case. A
        ``str`` or ``bytes`` is written **verbatim**, newline and all, which is
        the whole malformed-session story in one rule: ``'{"type":"result",'``
        is a process killed mid-object, ``'\\n'`` is the CLI's blank warm-up
        line, ``'warning: ...\\n'`` is a plain-text diagnostic on stdout. No
        ``truncate=`` knob, no ``corrupt=`` flag; every shape a real binary can
        emit is already expressible, and a knob per shape would only enumerate
        the ones someone thought of.
        """
        out = bytearray()
        interleaved: list[tuple[int, bytes]] = []
        for item in payloads:
            if isinstance(item, CliStderr):
                interleaved.append((_count_lines(out), item.data))
            elif isinstance(item, bytes):
                out += item
            elif isinstance(item, str):
                out += item.encode()
            else:
                out += (json.dumps(dict(item), separators=(",", ":")) + "\n").encode()
        return cls(
            stdout=bytes(out),
            stderr=stderr,
            returncode=returncode,
            interleaved_stderr=tuple(interleaved),
        )


@dataclass(slots=True)
class CliInvocation:
    """What one spawn was asked to do, and what happened to it.

    Records argv/cwd/env so a flags test can assert on them without the
    process-wide ``create_subprocess_exec`` patch, and ``stdin`` so a session
    test can assert on the NDJSON turns the cognition wrote — which is
    otherwise invisible, because a session's input side never reaches the
    filesystem or the event stream.
    """

    argv: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    stdin: bytes = b""
    # stdout lines the consumer has actually pulled. The streaming assertion:
    # a double that buffered a run to completion before handing over its first
    # event would show this at the recording's full length by the time the
    # first ``message_delta`` arrives.
    lines_read: int = 0
    terminated: bool = False
    killed: bool = False
    returncode: int | None = None


def _count_lines(buf: bytes | bytearray) -> int:
    """How many results ``readline()`` will yield from ``buf``.

    A trailing fragment with no newline counts, because that is what a
    ``StreamReader`` does at EOF — and a truncated final line reaching the
    parser is the behaviour half this module exists to reproduce.
    """
    if not buf:
        return 0
    return buf.count(b"\n") + (0 if buf.endswith(b"\n") else 1)


def _iter_lines(blob: bytes) -> Iterator[bytes]:
    """Split ``blob`` the way ``StreamReader.readline`` does — separator kept,
    trailing fragment yielded as its own line.

    Lazily, one line materialised at a time, so a 20k-payload recording is not
    turned into a 20k-element list before the consumer sees line one. The
    consumer is an ``async for`` over stdout that is meant to be incremental;
    a double that eagerly split would still *look* right on every assertion
    about content and quietly destroy the only property a streaming test cares
    about.
    """
    start = 0
    n = len(blob)
    while start < n:
        cut = blob.find(b"\n", start)
        if cut == -1:
            yield blob[start:]
            return
        yield blob[start : cut + 1]
        start = cut + 1


class _ReplayStdout:
    """``proc.stdout`` — an async line iterator over one immutable recording."""

    def __init__(self, run: CliRun, invocation: CliInvocation, stderr: _ReplayStderr) -> None:
        self._lines = _iter_lines(run.stdout)
        # Keyed on the POSITION alone. A bare ``sorted()`` compares the whole
        # tuple, so two chunks written at the same point in the stdout stream
        # fell through to comparing their BYTES and came back in alphabetical
        # order — a two-line traceback emitted as two writes was reassembled
        # backwards, and the run still looked fine. Python's sort is stable, so
        # keying on the position is exactly "ties keep their write order".
        self._pending = sorted(run.interleaved_stderr, key=lambda pair: pair[0])
        self._invocation = invocation
        self._stderr = stderr

    def __aiter__(self) -> _ReplayStdout:
        return self

    async def __anext__(self) -> bytes:
        self._flush_stderr_up_to(self._invocation.lines_read)
        try:
            line = next(self._lines)
        except StopIteration:
            # EOF: anything the process still had to say on stderr was said
            # before it exited, so it must be readable now.
            self._flush_stderr_up_to(None)
            raise StopAsyncIteration from None
        self._invocation.lines_read += 1
        return line

    def _flush_stderr_up_to(self, emitted: int | None) -> None:
        while self._pending and (emitted is None or self._pending[0][0] <= emitted):
            self._stderr.write(self._pending.pop(0)[1])


class _ReplayStderr:
    """``proc.stderr``. Drains once, like the real pipe: a second ``read()``
    returns ``b""`` rather than the same bytes again, so a session that reads
    stderr on two consecutive failed turns does not attribute the first turn's
    diagnostic to the second as well."""

    def __init__(self, initial: bytes = b"") -> None:
        self._buf = bytearray(initial)

    def write(self, data: bytes) -> None:
        self._buf += data

    async def read(self, n: int = -1) -> bytes:
        del n  # the cognition only ever reads to EOF
        out, self._buf = bytes(self._buf), bytearray()
        return out


class _RecordingStdin:
    """``proc.stdin`` — a writer that appends to the invocation record.

    Present even for a ``drive()`` spawn, which passes ``stdin=DEVNULL`` and
    never writes: handing back ``None`` there would be more faithful and would
    also make the fake's shape depend on the caller's flags, which is one more
    thing for a test to get wrong for no benefit.
    """

    def __init__(self, invocation: CliInvocation) -> None:
        self._invocation = invocation
        self._closing = False

    def write(self, data: bytes) -> None:
        self._invocation.stdin += bytes(data)

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True


class _ReplayProcess:
    """The ``asyncio.subprocess.Process`` surface the cognition actually uses.

    ``returncode`` stays ``None`` until ``wait()``/``terminate()``/``kill()``,
    which is the property the cognition leans on hardest: a persistent session
    checks ``proc.returncode is not None`` before every turn to decide whether
    the conversation is still alive, so a double that reported its exit code
    from the start would make every session look dead on turn two.
    """

    def __init__(self, run: CliRun, invocation: CliInvocation) -> None:
        self._run = run
        self._invocation = invocation
        self.stderr = _ReplayStderr(run.stderr)
        self.stdout = _ReplayStdout(run, invocation, self.stderr)
        self.stdin = _RecordingStdin(invocation)

    @property
    def returncode(self) -> int | None:
        return self._invocation.returncode

    async def wait(self) -> int:
        if self._invocation.returncode is None:
            self._invocation.returncode = self._run.returncode
        return self._invocation.returncode

    def terminate(self) -> None:
        self._invocation.terminated = True
        # A well-behaved process exits on SIGTERM; the native encoding of that
        # is a negative return code, and ``_finalise`` reports it verbatim.
        self._invocation.returncode = -15

    def kill(self) -> None:
        self._invocation.killed = True
        self._invocation.returncode = -9


class FakeClaudeCli:
    """Deterministic offline ``claude`` CLI. Pass it as ``spawn=`` to a
    :class:`~agentkit.agents.cognition.ClaudeCliCognition`.

    Holds an ordered list of :class:`CliRun`\\ s — one per **spawn**, not per
    turn, because that is the unit the CLI actually has: ``drive()`` spawns
    once per run, while a session spawns once and serves many turns off the
    same stdout. Each spawn consumes the next run; asking for one the recording
    does not have raises :class:`~agentkit.testing.ScriptExhausted`, for the
    reason that class exists — a recording is a claim about how many times the
    CLI is invoked, and code that invokes it more has a defect (a retry that
    should not have fired, a loop that should have stopped) that replaying the
    last run would answer with a plausible, wrong result.

    ``repeat_last=True`` replays the final run forever, for the test that
    deliberately drives an unbounded number of spawns.
    """

    def __init__(self, runs: Sequence[CliRun] = (), *, repeat_last: bool = False) -> None:
        self._runs = tuple(runs)
        self._repeat_last = repeat_last
        self._cursor = 0
        self.invocations: list[CliInvocation] = []

    # ---- construction ------------------------------------------------------

    @classmethod
    def replay(cls, path: Path | str, *more: Path | str, repeat_last: bool = False) -> FakeClaudeCli:
        """Replay recorded stream-json session file(s), one spawn per file.

        The bytes are read **now**, once, and held immutably — not streamed off
        disk per spawn. Two reasons, and neither is performance: a recording
        that vanishes or changes between construction and the spawn three
        layers down would surface as ``evals["error"]`` inside the cognition's
        ``except BaseException``, where a missing fixture is nearly unreadable;
        and "replays byte-identically twice" is trivially true of an immutable
        buffer and merely probable of a file. Streaming happens on the other
        side — the buffer is handed to the consumer one line at a time.
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
    ) -> FakeClaudeCli:
        """Assemble ONE spawn's output from payloads — the case nobody recorded.

        One spawn rather than one-per-payload because that is what the shapes
        this is for need: a refusal, a session limit and a malformed final
        answer are all a single CLI invocation that went wrong. Several spawns
        with different exit codes are ``FakeClaudeCli([CliRun.of(...), ...])``.

        See :meth:`CliRun.of` for how a payload becomes bytes — in particular
        that a raw ``str``/``bytes`` is written verbatim, which is how a
        truncated or non-JSON line is built.
        """
        return cls([CliRun.of(payloads, stderr=stderr, returncode=returncode)], repeat_last=repeat_last)

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
        Declaring them would make this double break whenever the cognition
        changed a pipe mode, which is not a change any test of parsing or event
        mapping should notice.
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
                raise ScriptExhausted(_exhausted_message(len(self._runs), self._cursor))
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


def _exhausted_message(have: int, idx: int) -> str:
    """All the value of raising is in this string, so it carries both numbers
    and both ways out. "recording exhausted" on its own sends the reader into
    this module, and the defect is almost never here."""
    return (
        f"FakeClaudeCli exhausted: the recording has {have} run(s), but the CLI was spawned "
        f"{idx + 1} time(s). A recording is a claim about how many times the binary is "
        "invoked — spawning more usually means the code under test retried, looped or "
        "re-entered when it should not have, which is the defect, not a shortage of "
        f"recording. Either supply the missing run(s) if {idx + 1} spawns are genuinely "
        "expected — FakeClaudeCli([CliRun.of(...), CliRun.of(...)]) — or pass "
        "repeat_last=True to replay the final run forever when the test deliberately "
        "drives an unbounded number of spawns."
    )
