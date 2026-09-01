"""The subprocess-replay machinery behind ``FakeClaudeCli`` and ``FakeCodexCli``.

Both doubles stand in for ``asyncio.create_subprocess_exec`` and both replay a
recording of one binary's stdout, so everything about *being a fake process* is
shared: line splitting the way a ``StreamReader`` does it, a stderr pipe that
drains once, a ``returncode`` that stays ``None`` until the process is waited
on, and a per-spawn record of the argv/cwd/env/stdin a test wants to assert on.

None of that is CLI-specific, and the parts of it that are subtle are subtle in
ways that already cost this codebase a bug each:

* ``_count_lines`` counts a trailing fragment with no newline, because that is
  what a real reader yields at EOF and a truncated final line reaching the
  parser is behaviour these doubles exist to reproduce.
* interleaved stderr chunks sort on POSITION ONLY. A bare ``sorted()`` compares
  the whole tuple, so two chunks written at the same point fell through to
  comparing their BYTES — a two-line traceback emitted as two writes came back
  in alphabetical order, reassembled backwards, and the run still looked fine.
* ``returncode`` staying ``None`` while the process is alive is the property the
  persistent-session paths lean on hardest: a session checks it before every
  turn to decide whether the conversation is still there, so a double that
  reported its exit code from the start would make every session look dead on
  turn two.

A second copy of any of those would be a second place to get them wrong, and
the wrongness is invisible — every one of them produces a fake that still
passes every assertion about content. The per-CLI modules keep what is genuinely
theirs: the payload vocabulary, the class docstring that explains the seam, and
the exhaustion message, which names the binary and the constructor a reader has
to go fix.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CliInvocation",
    "CliRun",
    "CliStderr",
    "_ReplayProcess",
    "_count_lines",
    "_exhausted_message",
    "_iter_lines",
]


def _exhausted_message(*, double: str, have: int, idx: int) -> str:
    """All the value of raising is in this string, so it carries both numbers
    and both ways out. "recording exhausted" on its own sends the reader into
    the double's module, and the defect is almost never there.

    ``double`` names the class so the two suggested fixes are copy-pasteable
    rather than a shape the reader has to translate.
    """
    return (
        f"{double} exhausted: the recording has {have} run(s), but the CLI was spawned "
        f"{idx + 1} time(s). A recording is a claim about how many times the binary is "
        "invoked — spawning more usually means the code under test retried, looped or "
        "re-entered when it should not have, which is the defect, not a shortage of "
        f"recording. Either supply the missing run(s) if {idx + 1} spawns are genuinely "
        f"expected — {double}([CliRun.of(...), CliRun.of(...)]) — or pass "
        "repeat_last=True to replay the final run forever when the test deliberately "
        "drives an unbounded number of spawns."
    )


@dataclass(frozen=True, slots=True)
class CliStderr:
    """A chunk the CLI wrote to **stderr** at this point in its stdout stream.

    Positional, not a lump attached to the run, because a cognition reads
    stderr exactly once — at the end, after stdout hits EOF. A diagnostic
    emitted in the middle of a long run therefore has to survive until then,
    and "did it?" is a real question: stderr is the only channel that explains
    a bare ``cli_exit_1`` to an operator. Placing chunks in the stream is what
    lets a test ask it.
    """

    data: bytes


@dataclass(frozen=True, slots=True)
class CliRun:
    """One spawn of a CLI, whole: its stdout, its stderr, its exit code.

    Frozen because it is the recording. Two doubles over one ``CliRun`` must not
    be able to affect each other, and the cheapest way to guarantee that is for
    the shared thing to be immutable and the per-spawn cursor to live in the
    iterator.
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
        is a process killed mid-object, ``'\\n'`` is a blank warm-up line,
        ``'warning: ...\\n'`` is a plain-text diagnostic on stdout. No
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
    test can assert on what the cognition wrote — which is otherwise invisible,
    because a session's input side never reaches the filesystem or the event
    stream.
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
    parser is the behaviour half of this module exists to reproduce.
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
        del n  # the cognitions only ever read to EOF
        out, self._buf = bytes(self._buf), bytearray()
        return out


class _RecordingStdin:
    """``proc.stdin`` — a writer that appends to the invocation record.

    Present even for a spawn that passes ``stdin=DEVNULL`` and never writes:
    handing back ``None`` there would be more faithful and would also make the
    fake's shape depend on the caller's flags, which is one more thing for a
    test to get wrong for no benefit.
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
    """The ``asyncio.subprocess.Process`` surface the cognitions actually use.

    ``returncode`` stays ``None`` until ``wait()``/``terminate()``/``kill()``,
    which is the property they lean on hardest: a persistent session checks
    ``proc.returncode is not None`` before every turn to decide whether the
    conversation is still alive, so a double that reported its exit code from
    the start would make every session look dead on turn two.
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
