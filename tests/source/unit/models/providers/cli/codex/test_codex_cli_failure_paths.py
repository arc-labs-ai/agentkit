"""What `CodexCliCognition` reports when the run did NOT work.

WHY THIS FILE EXISTS
--------------------
``codex exec`` fails differently from ``claude``, and the differences are
exactly where this cognition's own failure handling has to be its own:

* **the CLI reports no cost at all.** ``ProviderUsage.cost_usd`` on this path comes
  from a price table or a caller's rate card, so "the run failed" and "we never
  saw a terminal usage payload" have to produce an UNKNOWN cost rather than
  ``$0.00``. A silent zero is worse than a missing number: it is a positive
  claim, it keeps a budget's headroom intact, and it shows a ten-minute run as
  free on every dashboard that sums the field;
* **the structured answer IS the final message**, not a separate validated
  field, so "there was text and it was not JSON" is a different fault from
  "there was nothing" — and the raw excerpt is the whole diagnosis of the first;
* **there is no ``--max-budget-usd``**, so a budget cannot stop a run
  mid-flight. All a ceiling can do here is refuse to spawn and charge
  afterwards, and saying that plainly is the only honest option;
* **every session has ``shell``**, so ``native_tool_policy="deny"`` is
  unsatisfiable — the containment that does apply is the OS sandbox, and the
  caps table is keyed on it;
* a turn ends with ``turn.completed`` or ``turn.failed``, and a stream that has
  neither must never read as a finished answer. Measured before
  ``malformed_output`` existed: ``stop_reason="complete"``, ``partial=False``,
  half an answer handed back as a whole one.

``FakeCodexCli`` supplies the bytes and nothing else, so ``_parse_line``,
``_events_from_payload``, the item fold, the stop-reason priority and the
pricing all run as a real run runs them. Where the shape is not expressible as a
recording — a stdout that never speaks, a reader whose buffer overran — there is
a hand-written process double, because those are the branches a recording cannot
reach.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agentkit.core.contracts.types import CostSource, ProviderUsage, StreamEvent
from agentkit.core.testing.fakes import CliRun, FakeCodexCli, codex_turn
from agentkit.models.providers.cli._common import CliTimeouts
from agentkit.models.providers.cli.codex import driver as cx
from agentkit.models.providers.cli.codex.driver import CodexCliCognition

# ── stubs ────────────────────────────────────────────────────────────────────


@dataclass
class _Agent:
    """The only attributes this cognition reads off an agent."""

    prompt: Any = None
    model: str | None = None
    _output_adapter: Any = None


class _Adapter:
    """The output-schema surface ``_resolve_json_schema`` reads off an agent."""

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

    def validate(self, value: Any) -> Any:
        return ("coerced", value)


class _RejectingAdapter(_Adapter):
    def validate(self, value: Any) -> Any:
        raise _Coercion("1 error(s)")


class _Coercion(Exception):
    errors = ["answer: Input should be a valid string"]
    raw = {"answer": 42}


class _StopButton:
    """A ``Ctx``-shaped stub carrying nothing but cooperative cancellation.

    The driver polls ``check_cancelled()`` once per stdout line, which is the
    only cancellation route through a replayed process — ``FakeCodexCli``'s
    reads never await, so ``wait_for`` cannot fire mid-stream.
    """

    def __init__(self, *, after_lines: int = 0) -> None:
        self.checks = 0
        self._after = after_lines

    def check_cancelled(self) -> None:
        self.checks += 1
        if self.checks > self._after:
            raise RuntimeError("the caller pressed stop")


class _ExhaustedCtx:
    class _Budget:
        def remaining(self) -> float:
            return 0.0

    budget = _Budget()

    def check_cancelled(self) -> None:
        return None


class _RecordingMeter:
    def __init__(self) -> None:
        self.charges: list[tuple[float | None, Any]] = []

    async def charge(self, _call: Any, usage: ProviderUsage) -> None:
        self.charges.append((usage.cost_usd, usage.cost_source))


class _MeteredCtx:
    def __init__(self, meter: _RecordingMeter) -> None:
        self.all_meters = [meter]
        self.actor_budget = None

    def check_cancelled(self) -> None:
        return None


def retry() -> Any:
    """A raw ``(call, next)`` middleware in the shape the real ones have.

    The factories return closures all literally named ``mw``; the name a caller
    typed survives only on ``__qualname__``. Module level, so that qualname is
    ``retry.<locals>.mw`` rather than nested inside a test's.
    """

    def mw(call: Any, nxt: Any) -> Any:  # pragma: no cover - never invoked
        return nxt(call)

    return mw


# ── a process double, for the shapes no recording can express ────────────────


class _Stderr:
    def __init__(self, data: bytes = b"") -> None:
        self._buf = bytearray(data)

    async def read(self, n: int = -1) -> bytes:
        del n
        out, self._buf = bytes(self._buf), bytearray()
        return out


class _Stdin:
    def write(self, data: bytes) -> None:
        self.written = getattr(self, "written", b"") + data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _NeverSpeaks:
    """A stdout that is open and silent — the shape of a CLI that cannot
    authenticate and does not say so."""

    def __aiter__(self) -> _NeverSpeaks:
        return self

    async def __anext__(self) -> bytes:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _OverrunsTheReader:
    """A stdout whose first line is bigger than the reader will assemble.

    This is the measured codex case: one ``item.completed`` carrying a very
    large file's contents.
    """

    def __aiter__(self) -> _OverrunsTheReader:
        return self

    async def __anext__(self) -> bytes:
        raise ValueError("Separator is not found, and chunk exceed the limit")


class _Proc:
    def __init__(self, stdout: Any, *, stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.stderr = _Stderr(stderr)
        self.stdin = _Stdin()
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.killed += 1
        if self.returncode is None:
            self.returncode = -9


def _spawning(proc: Any) -> Any:
    async def spawn(*_argv: str, **_kwargs: Any) -> Any:
        return proc

    return spawn


# ── drivers ──────────────────────────────────────────────────────────────────


async def _drive(
    cognition: CodexCliCognition, *, agent: Any = None, ctx: Any = None, task: str = "summarise the diff"
) -> list[StreamEvent]:
    """Every event one ``drive`` produced, terminal event last."""
    return [ev async for ev in cognition.drive(agent or _Agent(), task, ctx, None)]


def _final(events: list[StreamEvent]) -> Any:
    """The one terminal result, asserting the exactly-one-``final`` contract."""
    types = [e.type for e in events]
    assert types.count("final") == 1, types
    assert types[-1] == "final", types
    return events[-1].result


async def _run_to_result(cognition: CodexCliCognition, **kwargs: Any) -> Any:
    """One ``drive``'s terminal result, for a SYNC test driving an inner
    coroutine — the house idiom when the assertions are filesystem ones."""
    return _final(await _drive(cognition, **kwargs))


def _half_a_turn() -> tuple[dict[str, Any], ...]:
    """A thread that started, answered, and never reached its terminal event."""
    return (
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "The three risks are"}},
    )


# ── the terminal event exists at all ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_binary_that_is_not_on_path_still_produces_one_terminal_event() -> None:
    """``drive`` is an async generator, so a spawn failure has nowhere to raise
    that a caller's ``async for`` would survive.

    A missing ``codex`` is the most common way this cognition fails, and it fails
    before the binary can say anything. ``execution_started=False`` is the fact a
    retry policy needs: nothing ran, so nothing was half-done — and the cost must
    be unknown rather than zero, because "we did not spend anything" and "we have
    no idea what was spent" are different claims and only one of them is
    defensible without a terminal usage payload.
    """

    async def missing(*_argv: str, **_kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", "codex")

    result = _final(await _drive(CodexCliCognition(spawn=missing)))

    assert result.stop_reason == "failed"
    assert result.evals["stop_reason"] == "spawn_failed"
    assert result.partial is True
    assert result.evals["execution_started"] is False
    assert result.evals["execution_uncertain"] is False
    assert result.usage.cost_usd is None
    assert result.evals["cost_source"] == CostSource.UNPRICED


@pytest.mark.asyncio
async def test_a_mistyped_working_dir_is_not_reported_as_a_missing_binary(tmp_path: Any) -> None:
    """One exception type, two problems, two different fixes.

    ``create_subprocess_exec`` raises ``FileNotFoundError`` for a missing ``cwd``
    AND for a missing binary, so an operator who typo'd ``working_dir=`` is
    otherwise told to install the CLI. The spawn must not even be attempted,
    because the error it would produce is the ambiguous one.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli, working_dir=tmp_path / "not-here")

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "working_dir_missing"
    assert len(cli.invocations) == 0, "the check is a pre-flight; the subprocess must not be paid for"


# ── the worst shape: a truncated stream reported as a finished answer ────────


@pytest.mark.asyncio
async def test_a_stream_with_no_terminal_turn_event_is_never_reported_complete() -> None:
    """A turn ends at ``turn.completed`` or ``turn.failed``, and this had
    neither.

    Measured before this reason existed: ``stop_reason="complete"``,
    ``partial=False``, half an answer handed back as a finished one — the worst
    shape a failure can take, because the text is well-formed and the result
    carries nothing a caller could branch on. The fragment is kept, because it
    is real; what changes is the claim made about it.
    """
    cli = FakeCodexCli.script(_half_a_turn())

    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "malformed_output"
    assert result.stop_reason == "failed"
    assert result.partial is True
    assert result.output == "The three risks are"


@pytest.mark.asyncio
async def test_a_final_line_cut_mid_object_is_a_truncated_stream_and_not_a_parse_failure() -> None:
    """A process killed partway through writing ``turn.completed``.

    The half-line is skipped by ``_parse_line`` (it is not a JSON object), so
    nothing raises and the run looks like one that simply ended — which is why
    the terminal-event check has to be what catches it. ``parse_failed`` would
    also be wrong: nothing was corrupt, the writer was interrupted.
    """
    cli = FakeCodexCli.script([*_half_a_turn(), '{"type":"turn.completed","usage":'])

    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "malformed_output"
    assert result.partial is True


@pytest.mark.asyncio
async def test_a_run_that_reached_its_terminal_event_is_reported_complete() -> None:
    """The other half of the contract, and the reason the check above cannot
    just be "always partial".

    A ``malformed_output`` that fired on healthy runs would make ``partial``
    meaningless in the other direction — every successful run flagged as
    suspect, and callers taught to ignore the flag.
    """
    cli = FakeCodexCli.script(codex_turn(text="all three risks", usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.stop_reason == "complete"
    assert result.partial is False
    assert result.output == "all three risks"


# ── the stop-reason ranking ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_turn_the_cli_itself_failed_carries_the_clis_own_explanation() -> None:
    """``turn.failed`` is the CLI saying the turn went wrong, with a message.

    That message is the whole diagnosis — a rate limit, a provider error, a
    refused tool — and collapsing it to a generic failure throws away the only
    actionable half. The partial answer is still kept, because text that arrived
    before the failure is real.
    """
    cli = FakeCodexCli.script(codex_turn(text="got this far", failed="rate limit exceeded"))

    events = await _drive(CodexCliCognition(spawn=cli))
    result = _final(events)

    assert result.stop_reason == "failed"
    assert result.partial is True
    assert result.output == "got this far"
    assert "turn_failed:rate limit exceeded" in [e.text for e in events if e.type == "step"]


@pytest.mark.asyncio
async def test_an_item_level_error_is_recorded_without_failing_a_run_that_finished() -> None:
    """"command output truncated" is not a failure.

    Promoting an item-level warning to a failed run would make
    ``partial``/``stop_reason`` fire on healthy runs — and a run that took forty
    seconds and looks fine is explained by exactly these and nothing else in the
    result, so dropping them instead loses the only trace of them.
    """
    cli = FakeCodexCli.script(codex_turn(text="done", error="command output truncated", usage=(10, 0, 5)))

    events = await _drive(CodexCliCognition(spawn=cli))
    result = _final(events)

    assert result.stop_reason == "complete"
    assert result.partial is False
    assert result.evals["cli_errors"] == ["command output truncated"]
    assert "error:command output truncated" in [e.text for e in events if e.type == "step"]


@pytest.mark.asyncio
async def test_a_death_by_signal_keeps_its_own_reason_rather_than_a_numeric_exit() -> None:
    """A negative return code is not an exit the program chose.

    ``process_crashed`` is a refinement of ``cli_exit_<n>``: an OOM kill and a
    deliberate ``exit(9)`` are different events, and ``cli_exit_-9`` would read
    as the latter.
    """
    cli = FakeCodexCli([CliRun.of(codex_turn(text="partial", usage=(10, 0, 5)), returncode=-9)])

    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "process_crashed"
    assert result.evals["cli_return_code"] == -9
    assert result.partial is True


@pytest.mark.asyncio
async def test_a_clap_usage_error_keeps_its_exit_code_and_its_stderr() -> None:
    """``codex exec`` exits 2 on an argument it does not recognise, before
    starting a thread.

    That has happened twice in this cognition's own history — ``--search`` and
    ``--ask-for-approval`` both moved — and in both cases the ONLY diagnosis was
    a clap message on stderr with no stdout at all. Both the number and the text
    have to reach the result or a flag regression is invisible.
    """
    cli = FakeCodexCli.script([], returncode=2, stderr=b"error: unexpected argument '--search' found")

    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "cli_exit_2"
    assert result.evals["cli_return_code"] == 2
    assert "unexpected argument" in result.evals["stderr"]


@pytest.mark.asyncio
async def test_a_schema_the_cli_refused_is_named_rather_than_left_as_an_exit_code() -> None:
    """Codex is handed the schema as a FILE, so its refusal is a bare non-zero
    exit with the complaint on stderr — even less legible than Claude's, which
    at least names the flag.

    ``schema_rejected`` is what turns that into something to go fix, and it is
    read off stderr because the exit code is identical to every other startup
    refusal.
    """
    cli = FakeCodexCli.script([], returncode=1, stderr=b"invalid json schema in output schema file")
    cognition = CodexCliCognition(spawn=cli, json_schema={"type": "object", "properties": {"a": {"type": "string"}}})

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "schema_rejected"
    assert result.stop_reason == "failed"


@pytest.mark.asyncio
async def test_a_single_payload_too_big_for_the_reader_gets_a_size_reason_not_a_corruption_reason() -> None:
    """Measured against real codex on "read every file under agentkit/ and
    summarise each": ``failed`` / ``parse_failed`` after three events, with the
    CLI exiting 0 and holding the answer.

    Nothing was malformed — one ``item.completed`` carried a large file and
    outgrew the reader's buffer — so the fix is a size, where ``parse_failed``
    sends an operator hunting for a corrupt payload. ``execution_uncertain`` is
    True because the CLI had already started doing work, and under a
    workspace-write sandbox some of it may have landed.
    """
    proc = _Proc(_OverrunsTheReader())

    result = _final(await _drive(CodexCliCognition(spawn=_spawning(proc))))

    assert result.evals["stop_reason"] == "output_line_too_long"
    assert result.stop_reason == "failed"
    assert result.evals["execution_uncertain"] is True
    assert "read the file in parts" in result.evals["error"]


# ── liveness ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cli_that_never_speaks_is_bounded_killed_and_told_apart_from_a_slow_model() -> None:
    """``startup`` covers LOCAL work only — boot, read config, resolve auth —
    so it cannot false-positive on model latency.

    A silent process looks active for as long as the caller is willing to wait,
    and the process must be killed when the bound fires: a timeout that gives up
    on reading while leaving the child alive is a leak dressed as a bound. The
    number goes on the result so the fix does not need the construction site to
    hand.
    """
    proc = _Proc(_NeverSpeaks(), stderr=b"warming up")
    cognition = CodexCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=0.05))

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "startup_timeout"
    assert result.evals["timeout_kind"] == "startup_timeout"
    assert result.evals["timeout_s"] == 0.05
    assert proc.terminated and proc.killed, "a bound that leaves the child running is a leak"


@pytest.mark.asyncio
async def test_a_startup_hang_whose_stderr_says_log_in_says_that_instead() -> None:
    """``startup_timeout`` alone sends an operator to look at the network when
    the answer is a credential.

    The refinement may only ever make an already-failed run's label more
    specific — the fallback is the bound's own name, which is always correct —
    and the markers must be phrases the CLI prints when auth has FAILED, never
    ones it prints while working.
    """
    proc = _Proc(_NeverSpeaks(), stderr=b"Not logged in. Run `codex login` to authenticate.")
    cognition = CodexCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=0.05))

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "authentication_failed"
    assert result.evals["timeout_kind"] == "startup_timeout", "the bound that fired is still recorded"


@pytest.mark.asyncio
async def test_a_run_that_goes_quiet_after_speaking_crosses_the_idle_bound_not_the_startup_one() -> None:
    """Which bound fired is the whole diagnosis.

    ``startup_timeout`` says the binary never came up; ``idle_timeout`` says a
    shell command is stuck. They send an operator to two different places, and
    the phase transition between them is the part that is easy to get subtly
    wrong — so it is pinned through the driver and not only on the deadline
    object.
    """

    class _OneLineThenSilence:
        def __init__(self) -> None:
            self._spoken = False

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> bytes:
            if not self._spoken:
                self._spoken = True
                return b'{"type":"thread.started","thread_id":"thread-1"}\n'
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    proc = _Proc(_OneLineThenSilence())
    cognition = CodexCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=None, idle=0.05))

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "idle_timeout"
    assert proc.terminated and proc.killed


# ── cancellation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tripped_stop_token_terminates_the_process_and_reports_a_partial_answer() -> None:
    """A stop button must leave no ``codex`` holding a pipe.

    The group — not the pid — because a cancelled run is exactly the one whose
    CLI may have forked a test suite and stopped reporting on it. ``cancelled``
    rather than ``failed``: somebody chose to stop this, and ``partial=True``
    because whatever text arrived is a fragment.
    """
    cli = FakeCodexCli.script(codex_turn(text="the answer", reasoning="thinking", usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli), ctx=_StopButton(after_lines=1)))

    assert result.evals["stop_reason"] == "cancelled"
    assert result.stop_reason == "cancelled"
    assert result.partial is True
    assert cli.invocations[0].terminated is True
    assert cli.invocations[0].killed is True, "the SIGKILL sweep is what reaps what the CLI forked"


@pytest.mark.asyncio
async def test_an_asyncio_cancel_delivers_the_terminal_event_and_then_keeps_propagating() -> None:
    """Both halves, in that order.

    Swallowing ``CancelledError`` — which a bare ``except BaseException`` does —
    breaks ``asyncio.wait_for``'s timeout and a ``TaskGroup``'s sibling
    cancellation, so the caller's own cancel mechanism silently stops working.
    Not delivering the event first breaks the exactly-one-``final`` contract on
    the path callers most need it.
    """

    async def cancelled_mid_spawn(*_argv: str, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    seen: list[StreamEvent] = []
    with pytest.raises(asyncio.CancelledError):
        async for event in CodexCliCognition(spawn=cancelled_mid_spawn).drive(_Agent(), "go", None, None):
            seen.append(event)

    assert [e.type for e in seen] == ["final"]
    assert seen[-1].result.evals["stop_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_a_keyboard_interrupt_is_reported_as_a_result_and_then_re_raised() -> None:
    """Ctrl-C used to come back as a tidy result while the interpreter kept
    running, so the caller's own ``except KeyboardInterrupt`` never fired.

    The terminal-event guarantee is about the EVENT, not about swallowing.
    ``Exception`` is the dividing line because that is the line the language
    draws between "the operation failed" and "this program is being taken down".
    """

    async def interrupted(*_argv: str, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt()

    seen: list[StreamEvent] = []
    with pytest.raises(KeyboardInterrupt):
        async for event in CodexCliCognition(spawn=interrupted).drive(_Agent(), "go", None, None):
            seen.append(event)

    assert [e.type for e in seen] == ["final"]
    assert seen[-1].result.partial is True


# ── cost: never a silent zero ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_run_whose_terminal_usage_never_arrived_reports_an_unknown_cost() -> None:
    """The CLI reports no cost, so the only inputs to one are a model name and a
    token count — and a truncated run has no token count either.

    Pricing the default zeros would produce ``$0.00``, which is not a missing
    number but the WRONG number stated as fact: the meters are charged nothing,
    a budget keeps headroom it has already spent, and a dashboard summing
    ``cost_usd`` shows a ten-minute run as free. ``UNPRICED`` is the framework's
    word for "no valid measurement is available" and this is exactly that case.
    """
    meter = _RecordingMeter()
    cli = FakeCodexCli.script(_half_a_turn())

    result = _final(await _drive(CodexCliCognition(spawn=cli), ctx=_MeteredCtx(meter)))

    assert result.evals["stop_reason"] == "malformed_output", "precondition: no terminal usage arrived"
    assert result.usage.cost_usd is None
    assert result.usage.cost_source is CostSource.UNPRICED
    assert result.evals["cost_source"] == CostSource.UNPRICED
    assert meter.charges == [(None, CostSource.UNPRICED)], "a meter must not be told the run was free"


@pytest.mark.asyncio
async def test_a_pricing_callable_that_raises_costs_the_caller_a_cost_and_not_the_answer() -> None:
    """The tokens are real and the terminal-event guarantee is unconditional.

    A caller-supplied rate card that blows up — a missing model key, a network
    lookup — must not take the run's answer with it. The result is marked
    unpriced, which preserves BOTH facts: the answer and the token counts stay,
    and the cost is honestly absent rather than silently zero.
    """
    cli = FakeCodexCli.script(codex_turn(text="the answer", usage=(1200, 1000, 40)))

    def no_rate_card(_model: str | None, _usage: ProviderUsage) -> float:
        raise KeyError("gpt-whatever")

    result = _final(await _drive(CodexCliCognition(spawn=cli, pricing=no_rate_card)))

    assert result.output == "the answer"
    assert result.stop_reason == "complete"
    assert result.usage.cost_usd is None
    assert result.usage.cost_source is CostSource.UNPRICED
    assert result.usage.input_tokens == 200, "the token facts survive a pricing failure"
    assert result.usage.cache_read_tokens == 1000


@pytest.mark.parametrize(
    ("returned", "why"),
    [
        (float("nan"), "NaN makes every later comparison in the ledger false"),
        (float("inf"), "infinity would exhaust every ceiling at once"),
        (-1.0, "a negative cost would CREDIT the budget for money just spent"),
    ],
)
@pytest.mark.asyncio
async def test_a_pricing_callable_returning_an_unusable_number_is_treated_as_no_measurement(
    returned: float, why: str
) -> None:
    """A rate card can return a number a ledger cannot add up.

    These are worse than an exception, because they propagate silently: a budget
    handed NaN can no longer answer whether it is over, and a negative cost
    hands headroom back. Refusing them here is the only place the cognition can,
    and the honest replacement is "unknown", not zero.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli, pricing=lambda m, u: returned)))

    assert result.usage.cost_usd is None, why
    assert result.usage.cost_source is CostSource.UNPRICED


@pytest.mark.asyncio
async def test_a_callers_own_rate_card_is_labelled_as_theirs_and_not_as_a_reported_figure() -> None:
    """``cost_source`` is what stops a host estimate being read as provider
    billing.

    The CLI reports nothing, so every number on this path is an estimate — and a
    caller reconciling against an invoice needs to know whether the figure came
    from their contractual rates or from agentkit's public table. "Reported"
    would be a claim nobody made.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli, pricing=lambda m, u: 1.25)))

    assert result.usage.cost_usd == pytest.approx(1.25)
    assert result.usage.cost_source is CostSource.HOST_PRICING
    assert result.evals["cost_source"] == CostSource.HOST_PRICING


@pytest.mark.asyncio
async def test_a_budget_with_no_headroom_refuses_before_paying_for_a_subprocess() -> None:
    """This is the WHOLE of the budget's in-flight power here.

    ``codex`` has no ``--max-budget-usd``, so a run that starts with headroom can
    spend past it and only be caught by the charge at the end. The pre-flight
    refusal is therefore the only ceiling that exists, and ``budget_exhausted``
    is deliberately NOT a failure — it is the resumable one: raise the ceiling
    and run again.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli), ctx=_ExhaustedCtx()))

    assert result.evals["stop_reason"] == "budget_exhausted"
    assert result.stop_reason == "budget_exhausted", "resumable, not failed"
    assert len(cli.invocations) == 0


@pytest.mark.asyncio
async def test_a_runs_wall_time_is_measured_because_the_cli_reports_no_duration() -> None:
    """``evals["cli_duration_ms"]`` has to mean the same thing in both
    cognitions, and ``claude`` gets it from its ``result`` payload.

    A duration field that is always zero is worse than an absent one, because a
    dashboard plots it. The clock starts before the concurrency permit is
    acquired, so the number includes time spent WAITING for a spawn slot —
    which is the figure an operator debugging a slow run under load actually
    needs.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.evals["cli_duration_ms"] >= 0
    assert isinstance(result.evals["cli_duration_ms"], int)

    reported = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5), duration_ms=4321))
    assert _final(await _drive(CodexCliCognition(spawn=reported))).evals["cli_duration_ms"] == 4321, (
        "a CLI version that does report one must win over the measurement"
    )


# ── structured output, where the answer IS the schema's payload ──────────────


@pytest.mark.asyncio
async def test_a_final_message_that_is_not_json_keeps_the_text_as_the_diagnosis() -> None:
    """For Codex the structured answer IS the final message.

    So "there was text and it was not JSON" is a different fault from "there was
    nothing", and it is the one where the raw excerpt is the whole diagnosis —
    what the model actually said is what tells a caller whether the schema was
    too strict, the prompt was ambiguous, or the model refused. ``invalid_output``
    and not ``failed``: the run worked, the shape did not.
    """
    cli = FakeCodexCli.script(codex_turn(text="I'd rather explain it in prose.", usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli), agent=_Agent(_output_adapter=_Adapter())))

    assert result.stop_reason == "invalid_output"
    assert result.evals["stop_reason"] == "structured_output_missing"
    assert result.partial is True
    failure = result.evals["structured_output_failure"]
    assert failure["kind"] == "undecodable", "there WAS text; 'missing' would be the wrong fault"
    assert failure["raw_excerpt"] == "I'd rather explain it in prose."


@pytest.mark.asyncio
async def test_no_final_message_at_all_is_reported_as_missing_rather_than_undecodable() -> None:
    """The mirror case, and the reason the two kinds are kept apart.

    ``undecodable`` points a reader at what the model said; ``missing`` says
    there is nothing to look at, so the question is why the turn produced no
    final message. One excerpt and one empty string would make them
    indistinguishable.
    """
    cli = FakeCodexCli.script(codex_turn(text=None, usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli), agent=_Agent(_output_adapter=_Adapter())))

    assert result.evals["structured_output_failure"]["kind"] == "missing"
    assert result.stop_reason == "invalid_output"


@pytest.mark.asyncio
async def test_a_fenced_code_block_is_unwrapped_because_models_wrap_json_in_one() -> None:
    """A model told to answer in JSON very often answers in ```` ```json ````.

    Refusing that would report ``structured_output_missing`` for an answer that
    is right, which is the failure mode most likely to be blamed on the schema.
    The unwrapping is deliberately narrow — a fence, not a search for the first
    ``{`` — because scanning for JSON-shaped fragments inside prose turns a
    refusal into a confident parse of whatever happened to be there.
    """
    cli = FakeCodexCli.script(codex_turn(text='```json\n{"answer": "yes"}\n```', usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli), agent=_Agent(_output_adapter=_Adapter())))

    assert result.stop_reason == "complete"
    assert result.parsed == ("coerced", {"answer": "yes"})
    assert result.evals["structured_output"] == {"answer": "yes"}


@pytest.mark.asyncio
async def test_json_that_does_not_fit_the_declared_type_is_a_mismatch_with_field_paths() -> None:
    """It parsed and it did not fit — the only kind that can carry violations.

    Those per-field paths are what an application needs to mark a form field or
    group failures across a fleet; reporting the exception's summary alone
    ("1 error(s)") leaves a caller parsing English that a library upgrade can
    reword.
    """
    cli = FakeCodexCli.script(codex_turn(text='{"answer": 42}', usage=(10, 0, 5)))

    result = _final(await _drive(CodexCliCognition(spawn=cli), agent=_Agent(_output_adapter=_RejectingAdapter())))

    assert result.evals["stop_reason"] == "structured_output_mismatch"
    assert result.stop_reason == "invalid_output"
    assert result.evals["structured_output_failure"]["violations"] == [
        {"path": "$.answer", "message": "Input should be a valid string"}
    ]


@pytest.mark.asyncio
async def test_every_object_in_the_schema_is_closed_because_the_providers_mode_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI's structured-output mode refuses a schema whose objects do not
    close themselves.

    The two adapters disagreed about this: the dataclass adapter already emits
    ``additionalProperties: false`` and Pydantic's ``model_json_schema()`` does
    not — so ``output=SomeDataclass`` worked and ``output=SomePydanticModel``
    failed EVERY time with a 400 the caller could only see by reading
    ``evals["stderr"]`` on a failed run. An arbitrary difference between two ways
    of declaring the same shape, and it has to be closed at every nesting level,
    not just the root.
    """
    written: dict[str, str] = {}
    real_write = Path.write_text

    def spy(self: Path, text: str, **kwargs: Any) -> int:
        written[self.name] = text
        return real_write(self, text, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy)

    cli = FakeCodexCli.script(codex_turn(text='{"nested": {"a": "x"}}', usage=(10, 0, 5)))
    cognition = CodexCliCognition(
        spawn=cli,
        json_schema={
            "type": "object",
            "properties": {"nested": {"type": "object", "properties": {"a": {"type": "string"}}}},
        },
    )
    await _drive(cognition)

    schema = json.loads(written["output_schema.json"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["nested"]["additionalProperties"] is False, "every level, not only the root"


def test_the_schema_file_is_removed_when_the_run_ends_however_it_ended() -> None:
    """``--output-schema`` takes a path, so the cognition owns a temp file a
    caller never sees and must not be asked to manage.

    It is torn down however the run ends — a FAILED run included, which is the
    case that matters: the alternative is a service that leaks a 0700 directory
    per failed run until the disk fills, and presents as an unrelated outage.

    Synchronous, with the filesystem assertions outside the coroutine: blocking
    ``pathlib`` calls on the event loop are what ruff's ASYNC240 exists to
    catch, and this suite is held to it too.
    """
    cli = FakeCodexCli.script([], returncode=1, stderr=b"boom")
    cognition = CodexCliCognition(spawn=cli, json_schema={"type": "object", "properties": {"a": {"type": "string"}}})

    result = asyncio.run(_run_to_result(cognition))

    assert result.evals["stop_reason"] == "cli_exit_1", "precondition: the run failed"
    argv = cli.invocations[0].argv
    path = Path(argv[argv.index("--output-schema") + 1])
    assert not path.exists()
    assert not path.parent.exists()


# ── configuration refused before it costs a subprocess ───────────────────────


@pytest.mark.parametrize(
    ("kwargs", "fragment", "why"),
    [
        (
            {"continue_session": True, "resume_session_id": "t-1"},
            "pass one, not both",
            "two resume mechanisms means one of them is being ignored",
        ),
        (
            {"fork_session": True},
            "requires a specific resume_session_id",
            "a fork of nothing is a flag with no effect",
        ),
        (
            {"bypass_sandbox": True, "sandbox": "read-only"},
            "states a policy that will not be in force",
            "the bypass turns the sandbox OFF, so the stated sandbox is a fiction the CLI "
            "resolves silently — a containment decision that reads as made and is not",
        ),
        (
            {"network_access": True, "sandbox": "read-only"},
            "only applies under sandbox='workspace-write'",
            "under read-only it does nothing, so the caller believes they opened the network "
            "and did not",
        ),
        (
            {"ephemeral": True, "continue_session": True},
            "nothing for resume_session_id/continue_session to resume",
            "ephemeral writes no session file, so the resume silently starts fresh",
        ),
        (
            {"env_policy": "strict"},
            "env_policy must be",
            "a typo'd policy would fall through to inherit, which is the unsafe one",
        ),
        (
            {"mcp_servers": {"ledger": {}}},
            "must be a non-empty mapping",
            "an empty server table is a server the CLI cannot start",
        ),
        (
            {"mcp_servers": {"ledger": {"bearer_token_env_var": "T"}}},
            "declares neither 'command'",
            "with no command and no url there is nothing to start or reach",
        ),
    ],
)
def test_a_configuration_the_cli_itself_refuses_is_refused_at_construction(
    kwargs: dict[str, Any], fragment: str, why: str
) -> None:
    """Every one of these is cheaper to catch here than as a subprocess that
    exits non-zero three seconds later with the message on a stderr this
    cognition surfaces only on an already-failed run.

    Three of them are worse than merely late — ``bypass_sandbox`` with a
    ``sandbox``, ``network_access`` under ``read-only`` and ``ephemeral`` with a
    resume all run SUCCESSFULLY while doing the opposite of what the caller
    stated, so there is no failed run to read a stderr from.
    """
    with pytest.raises(ValueError, match=fragment):
        CodexCliCognition(**kwargs)
    assert why


def test_a_deny_policy_is_refused_outright_because_every_codex_session_has_a_shell() -> None:
    """Unsatisfiable, and said so rather than accepted and ignored.

    There is no tool allow-list and no PreToolUse hook, so agentkit's tool
    middleware can never reach ``shell`` — a deny policy here is not a
    configuration, it is the whole cognition. Accepting the flag silently would
    be the worst outcome: a service that believes it refused ungovernable tools
    and runs them anyway. The refusal has to name both ways out.
    """
    with pytest.raises(ValueError) as caught:
        CodexCliCognition(native_tool_policy="deny")

    message = str(caught.value)
    assert "cannot be satisfied" in message
    assert "ClaudeCliCognition(tools=('',)" in message, "the configuration that CAN hold the policy"
    assert "native_tool_policy='warn'" in message, "and the way to accept the gap deliberately"
    assert "sandbox=" in message, "naming the containment that does apply"


def test_a_secret_that_has_to_travel_in_argv_is_warned_about_and_the_safe_pattern_is_named() -> None:
    """``-c key=value`` is the ONLY way to set a Codex config override, and an
    argument list is readable by any local account with ``ps``.

    Unlike the Claude cognition there is nowhere safe to move it to — ``codex
    exec`` has no config-file flag — so this is warned rather than refused: the
    value may be a placeholder or an already-scoped token, and refusing would
    break a working configuration over a name heuristic. The warning is only
    worth having if it names Codex's own answer.
    """
    with pytest.warns(UserWarning, match="mcp_servers.ledger.bearer_token") as caught:
        CodexCliCognition(mcp_servers={"ledger": {"url": "https://x.test", "bearer_token": "planted"}})

    message = str(caught[0].message)
    assert "bearer_token_env_var" in message, "the pattern that does work has to be in the message"
    assert "`ps`" in message


def test_a_config_key_that_only_names_an_environment_variable_is_not_warned_about() -> None:
    """``*_env_var`` keys name a variable to read the secret FROM.

    That is the documented safe pattern — the one the warning above points at —
    so warning about it would mean warning on the fix, which is how a warning
    teaches people to silence the category.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CodexCliCognition(mcp_servers={"ledger": {"url": "https://x.test", "bearer_token_env_var": "MY_TOKEN"}})


def test_the_trifecta_tags_follow_the_sandbox_that_will_actually_be_in_force() -> None:
    """Codex's containment is a sandbox, not a tool list, so the caps table is
    keyed on the sandbox — including the defaults.

    A ``caps`` that answered "nothing" for an unset ``--sandbox`` would leave the
    trifecta gate blind on the most common wiring there is. The case that matters
    is the innocuous-looking one: ``sandbox="read-only", web_search=True`` is
    private data plus untrusted content plus egress — the lethal trifecta —
    because a "read-only" sandbox is read-only about WRITES and is not blind.
    """
    assert CodexCliCognition().effective_sandbox == "read-only", "the CLI's own default for exec"
    assert CodexCliCognition().caps == ("private_data",)

    trifecta = CodexCliCognition(sandbox="read-only", web_search=True)
    assert set(trifecta.caps) == {"private_data", "untrusted_content", "egress"}

    # The bypass flag IS danger-full-access by another name.
    assert CodexCliCognition(bypass_sandbox=True).effective_sandbox == "danger-full-access"
    assert set(CodexCliCognition(bypass_sandbox=True).caps) == {"private_data", "egress"}

    # ``network_access`` is what turns workspace-write's network back on.
    assert "egress" not in CodexCliCognition(sandbox="workspace-write").caps
    assert "egress" in CodexCliCognition(sandbox="workspace-write", network_access=True).caps


# ── the middleware bypass is invisible by construction ───────────────────────


@pytest.mark.asyncio
async def test_the_first_run_names_what_the_chain_will_not_reach_and_then_stays_quiet() -> None:
    """A caller who wires the documented tool chain gets a session where none of
    it applies to ``shell``, with nothing saying so.

    Unlike the Claude cognition there is no "no native tools" escape, so the
    warning always has content — which makes the once-per-instance latch the only
    thing stopping it becoming noise, and noise is what teaches people to add a
    ``filterwarnings`` line. It has to name the OS sandbox, because that is the
    containment that does apply and the reader's only remaining lever.
    """

    class _Ctx:
        invoker = type("_Inv", (), {"tool_middleware": (retry(),)})()

        def check_cancelled(self) -> None:
            return None

    cli = FakeCodexCli.answering("first", "second")
    cognition = CodexCliCognition(spawn=cli)

    with pytest.warns(UserWarning, match="bypasses the Invoker") as caught:
        await _drive(cognition, ctx=_Ctx())
    message = str(caught[0].message)
    assert "retry" in message, "a chain entry the caller cannot name is one they cannot act on"
    assert "shell" in message
    assert "sandbox='read-only'" in message, "the containment that DOES apply"
    assert "no PreToolUse hook" in message, "and why there is no fix, only a trade"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await _drive(cognition, ctx=_Ctx())  # the latch holds: a second warning would raise here


# ── which auth the child resolves ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_profile_policy_strips_the_ambient_key_and_names_what_it_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``config_home=`` implies ``"profile"``, matching the Claude rule so one
    wiring reads the same across both.

    Codex was measured NOT to prefer an ambient ``OPENAI_API_KEY`` over the
    ``CODEX_HOME`` login, so this is hygiene here rather than a fix — and it is
    kept anyway because precedence is a config key rather than a law, and "the
    child gets exactly what we chose" is the property a reproducible service
    wants from BOTH cognitions rather than from whichever one happens to need it
    today. A silent strip is still as invisible as the silent inherit it
    replaces, so it is announced once.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-planted-not-a-real-key-0000000")
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli, config_home=tmp_path)

    with pytest.warns(UserWarning, match="removed OPENAI_API_KEY") as caught:
        await _drive(cognition)

    env = cli.invocations[0].env
    assert "OPENAI_API_KEY" not in env
    assert env["CODEX_HOME"] == str(tmp_path)
    assert "implied by config_home=" in str(caught[0].message), "the caller never typed this policy"
    assert "env_policy='inherit'" in str(caught[0].message), "the way back has to be in the message"


@pytest.mark.asyncio
async def test_no_invented_trace_variable_is_put_in_the_childs_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Claude cognition bridges ``correlation_id`` into
    ``CLAUDE_TRACE_EXTERNAL_ID`` because that variable exists.

    ``codex`` has no counterpart, and setting an invented one would put a value
    in the child's environment that nothing reads while reading, from the
    outside, exactly like a working trace bridge — so an operator would join on
    an id the CLI never emitted. The id is still on the result, which is the
    honest place for it.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _Ctx:
        correlation_id = "run-42"

        def check_cancelled(self) -> None:
            return None

    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    result = _final(await _drive(CodexCliCognition(spawn=cli), ctx=_Ctx()))

    assert not [k for k in cli.invocations[0].env if "TRACE" in k or "EXTERNAL" in k]
    assert result.evals["external_run_id"] == "run-42"


# ── the prompt, which has no flag on this CLI ────────────────────────────────


@pytest.mark.asyncio
async def test_prepend_mode_separates_the_prompt_from_the_task_with_a_rule() -> None:
    """Codex has no ``--append-system-prompt``, so both modes are constructions
    rather than flags.

    The rule is not decoration: without a separator a one-line prompt and a
    one-line task read to the model as one run-on instruction, which is how a
    "be terse" prompt ends up being treated as part of the question.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))

    await _drive(CodexCliCognition(spawn=cli), agent=_Agent(prompt="Be terse."), task="go")

    assert cli.invocations[0].stdin == b"Be terse.\n\n---\n\ngo"
    assert "-" == cli.invocations[0].argv[-1], "the prompt travels over stdin, never in argv"


def test_replace_mode_writes_the_prompt_to_an_instructions_file_and_sends_the_task_alone() -> None:
    """``replace`` overrides Codex's base instructions — tool guidance, sandbox
    explanation, apply_patch format — entirely.

    That is why it is not the default: a replacement turns a capable coding
    agent into a chat model that still has tools it no longer knows how to
    drive. It also means the prompt must NOT also be prepended to the task, or
    the model receives it twice and the duplication reads to it as escalating
    emphasis. The file belongs to the spawn and goes with it.

    Synchronous, with the filesystem assertion outside the coroutine: blocking
    ``pathlib`` calls on the event loop are what ruff's ASYNC240 exists to
    catch, and this suite is held to it too.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli, system_prompt_mode="replace")

    asyncio.run(_run_to_result(cognition, agent=_Agent(prompt="You are an auditor."), task="go"))

    argv = cli.invocations[0].argv
    override = next(a for a in argv if a.startswith("experimental_instructions_file="))
    assert cli.invocations[0].stdin == b"go", "the prompt is in the file, so the task travels alone"
    assert "You are an auditor." not in cli.invocations[0].stdin.decode()
    assert not Path(override.split("=", 1)[1].strip('"')).exists()


@pytest.mark.asyncio
async def test_replace_mode_with_no_prompt_does_not_blank_the_clis_base_instructions() -> None:
    """An empty instructions file would replace Codex's base instructions with
    nothing.

    That is a strictly worse agent than the one the caller started with, and it
    would look like the mode had no effect rather than like it had removed
    everything — so no prompt means no file.
    """
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))

    await _drive(CodexCliCognition(spawn=cli, system_prompt_mode="replace"), agent=_Agent(prompt="   "))

    assert not [a for a in cli.invocations[0].argv if "experimental_instructions_file" in a]


@pytest.mark.asyncio
async def test_a_task_beginning_with_a_dash_is_not_read_as_a_flag() -> None:
    """``codex exec`` would read ``"--force a rewrite"`` as arguments and die on
    an unknown one.

    The prompt travels over stdin now, which removes the problem for a real run
    — but the argv builder is still reachable (the flags matrix, a caller who
    does not stream input) and the ``--`` separator is what keeps it correct
    there. ``-`` itself must NOT get the separator, or it becomes a literal
    one-character prompt: exactly backwards.
    """
    cognition = CodexCliCognition()

    argv = cognition._build_argv("--force a rewrite")
    assert argv[-2:] == ["--", "--force a rewrite"]

    streamed = cognition._build_argv("--force a rewrite", stream_input=True)
    assert streamed[-1] == "-"
    assert "--" not in streamed[-2:], "the stdin marker must not be quoted into a literal prompt"


# ── a session's failures, which are a different question ────────────────────


@pytest.mark.asyncio
async def test_a_turn_asked_of_a_closed_session_refuses_and_names_the_thread_it_could_resume() -> None:
    """``codex exec`` is one-shot, so a closed session has no process to be
    dead — but it does have a CLI-side thread that still exists on disk.

    Silently starting a fresh one would answer from an empty context while the
    caller believes the history is there, so the model appears to have forgotten
    everything. The refusal carries the thread id precisely because the
    conversation is recoverable — that is the whole difference from the Claude
    session, whose context dies with its process.
    """
    cli = FakeCodexCli.answering("first")
    cognition = CodexCliCognition(spawn=cli)

    async with cognition.session() as chat:
        first = _final([ev async for ev in chat.turn("one")])
        thread_id = first.evals["session_id"]

    after = _final([ev async for ev in chat.turn("two")])

    assert first.stop_reason == "complete"
    assert after.evals["stop_reason"] == "session_closed"
    assert after.stop_reason == "failed"
    assert thread_id in after.evals["error"], "the thread is resumable, so its id has to be in the refusal"
    assert "resume_session_id=" in after.evals["error"]
    assert after.usage.cost_usd is None, "a turn that never ran has no cost, known or otherwise"
    assert len(cli.invocations) == 1, "a closed session must not quietly start a new conversation"


@pytest.mark.asyncio
async def test_a_later_turn_resumes_the_thread_the_first_turn_named() -> None:
    """A session that failed to thread its id would silently hold one-turn
    conversations.

    Every turn would answer correctly, in isolation, and the model would appear
    to have no memory — which reads as a model problem rather than as a missing
    ``resume`` in the second spawn's argv. Every option goes BEFORE the
    subcommand, so the resume fragment and the stdin marker are the tail.
    """
    cli = FakeCodexCli.answering("first", "second")
    cognition = CodexCliCognition(spawn=cli)

    async with cognition.session() as chat:
        async for _ in chat.turn("one"):
            pass
        async for _ in chat.turn("two"):
            pass
        assert chat.turns_taken == 2, "a session reporting one turn where three ran is its own bug"

    first, second = cli.invocations
    assert "resume" not in first.argv, "turn one starts the thread"
    assert second.argv[-3:-1] == ("resume", "0199a213-81c0-7800-8aa1-bbab2a035a53")
    assert second.argv[-1] == "-"


@pytest.mark.asyncio
async def test_a_first_turn_that_died_before_naming_a_thread_starts_fresh_rather_than_resuming_nothing() -> None:
    """A conversation with no history is exactly what a first turn that produced
    none left behind.

    Resuming an id the session never learned would be a CLI invocation with an
    empty positional — a clap error several seconds in, for a session that is
    merely empty. Starting fresh is the honest reading of "we have nothing to
    continue".
    """
    cli = FakeCodexCli([CliRun.of([], returncode=1, stderr=b"boom"), CliRun.of(codex_turn(text="second"))])
    cognition = CodexCliCognition(spawn=cli)

    async with cognition.session() as chat:
        failed = _final([ev async for ev in chat.turn("one")])
        recovered = _final([ev async for ev in chat.turn("two")])

    assert failed.evals["stop_reason"] == "cli_exit_1"
    assert failed.evals["session_id"] == ""
    assert "resume" not in cli.invocations[1].argv, "there was no thread id to resume"
    assert recovered.stop_reason == "complete"


@pytest.mark.asyncio
async def test_a_session_can_be_an_agents_cognition_so_consecutive_runs_continue_one_thread() -> None:
    """``drive`` is the ``Cognition``-shaped door onto a session.

    Without it a caller who wants consecutive ``agent.run(...)`` calls to
    continue one CLI-side thread has to drive ``turn()`` by hand and reimplement
    everything ``Agent`` does around a cognition — and the shape that makes the
    continuation possible is exactly the shape ``Agent`` already calls.
    """
    cli = FakeCodexCli.answering("hello", "again")

    async with CodexCliCognition(spawn=cli).session() as chat:
        first = _final([ev async for ev in chat.drive(_Agent(), "one", None, None)])
        second = _final([ev async for ev in chat.drive(_Agent(), "two", None, None)])

    assert (first.output, second.output) == ("hello", "again")
    assert "resume" in cli.invocations[1].argv, "the second run continued the first one's thread"


@pytest.mark.asyncio
async def test_a_fold_that_raises_mid_stream_is_a_parse_failure_and_an_uncertain_run() -> None:
    """A bug on OUR side of the pipe, after the CLI has started working.

    ``parse_failed`` is the honest label — nothing about the process failed —
    and ``execution_uncertain`` is the fact that stops a caller retrying:
    under a workspace-write sandbox the CLI may already have edited files, and
    a retry would apply them twice. The cost has to be unknown for the same
    reason the truncated case does: the terminal usage never arrived.
    """
    real_parser = cx._events_from_payload

    def explode(payload: dict[str, Any], log: Any) -> Any:
        if payload.get("type") == "item.completed":
            raise RuntimeError("a planted fold bug")
        return real_parser(payload, log)

    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cx, "_events_from_payload", explode)
        result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "parse_failed"
    assert result.stop_reason == "failed"
    assert result.evals["execution_uncertain"] is True
    assert result.usage.cost_usd is None
    assert "RuntimeError: a planted fold bug" in result.evals["error"]


@pytest.mark.asyncio
async def test_a_rate_limited_turn_carries_the_status_code_and_retry_after_the_cli_reported() -> None:
    """A 429 is the one failure a caller can do something specific about.

    "the turn failed" tells them to retry blindly; the status, the code and the
    ``retry_after`` tell them WHEN, and they are the difference between a backoff
    that works and one that hammers a provider that already said how long to
    wait. Each lands on its own eval key so a caller branches on a number rather
    than grepping a sentence.
    """
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.started"},
            {
                "type": "turn.failed",
                "error": {
                    "message": "rate limited",
                    "status": 429,
                    "code": "rate_limit_exceeded",
                    "retry_after": 30,
                },
            },
        ]
    )

    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "turn_failed"
    assert result.evals["api_error_status"] == 429
    assert result.evals["cli_error_code"] == "rate_limit_exceeded"
    assert result.evals["retry_after_s"] == 30
    assert result.partial is True


@pytest.mark.asyncio
async def test_a_session_reports_its_own_name_so_a_trace_can_tell_the_two_regimes_apart() -> None:
    """``Agent._span_attrs`` reads ``cognition.name`` on EVERY run.

    A session without one raised ``AttributeError: 'CodexCliSession' object has
    no attribute 'name'`` straight out of ``agent.run(...)`` — the exact usage
    the class advertises. Suffixed rather than delegated verbatim because the two
    regimes have genuinely different behaviour to debug.
    """
    cognition = CodexCliCognition(spawn=FakeCodexCli.answering("x"), web_search=True)
    session = cognition.session()

    assert session.name == "codex_cli_session"
    assert session.caps == cognition.caps, "a session can BE the cognition handed to RunPolicy"
    await session.close()


# ── diagnostics on the way out ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stderr_reaches_the_result_only_on_a_failure_and_with_credentials_removed() -> None:
    """``evals["stderr"]`` is the CLI's own diagnostics, and an ``AgentResult``
    is checkpointed, logged and fanned out to observers.

    Codex writes human progress to stderr on every run, so attaching it
    unconditionally would make every result carry noise — and
    ``--color never`` is passed precisely so a stored diagnostic has no ANSI
    escapes in it. Credential-shaped runs are replaced because a CLI printing a
    failing request's header is ordinary behaviour.
    """
    failing = FakeCodexCli.script([], returncode=1, stderr=b"POST /v1/responses\napi_key=0123456789abcdefghij\n")
    failed = _final(await _drive(CodexCliCognition(spawn=failing)))

    assert "POST /v1/responses" in failed.evals["stderr"]
    assert "0123456789abcdefghij" not in failed.evals["stderr"]
    assert "api_key=" in failed.evals["stderr"], "which credential leaked is not itself the secret"

    healthy = FakeCodexCli(
        [CliRun.of(codex_turn(text="ok", usage=(10, 0, 5)), stderr=b"[2026-01-01] thinking...")]
    )
    assert "stderr" not in _final(await _drive(CodexCliCognition(spawn=healthy))).evals


@pytest.mark.asyncio
async def test_the_model_a_run_was_asked_for_is_never_passed_off_as_the_one_that_served_it() -> None:
    """``evals["cli_model"]`` means "which model ACTUALLY ran", and Codex
    resolves aliases and profiles on its side.

    So defaulting the field to ``self.model`` would make an alias look like a
    served model name — a run that silently fell back would be indistinguishable
    from one that did not, in the result AND in the cost estimate that was
    computed from it. Absent is the honest answer while the stream says nothing,
    and absence is what a reader has to be able to rely on.

    (Note for whoever reads this next: no parser branch populates
    ``_TurnState.model`` today — see the finding on the dead ``cli_model`` path —
    so this currently pins the fallback half only. It is still the half that
    would do damage if someone "fixed" the empty field by filling it from the
    request.)
    """
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "thread-1", "model": "gpt-5.1-codex-max"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "ok"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}},
        ]
    )

    result = _final(await _drive(CodexCliCognition(spawn=cli, model="codex-latest")))

    assert result.evals.get("cli_model") != "codex-latest", "the request is not evidence of what served the run"
    argv = cli.invocations[0].argv
    assert argv[argv.index("--model") + 1] == "codex-latest", "the request itself still goes out"


@pytest.mark.asyncio
async def test_a_reasoning_summary_is_kept_beside_the_answer_and_not_inside_it() -> None:
    """Reasoning summaries are how the run got there, not what it answered.

    Folding them into ``output`` puts scratch work in the answer a caller shows
    a user; dropping the event makes a long quiet stretch unexplained. They are
    streamed live AND folded onto the result, because a consumer that joined late
    saw none of the deltas.
    """
    cli = FakeCodexCli.script(codex_turn(reasoning="weighing the two options", text="Option B.", usage=(10, 0, 5)))

    events = await _drive(CodexCliCognition(spawn=cli))
    result = _final(events)

    assert result.evals["thinking"] == "weighing the two options"
    assert result.output == "Option B.", "the reasoning must not become the answer"
    assert "weighing the two options" in [e.text for e in events if e.type == "message_delta"]
