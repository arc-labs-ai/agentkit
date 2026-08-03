"""``RunContext.emit`` is best-effort.

The docstring promise: emit "must never break the run". Every path
inside ``emit`` — the trace helpers AND the observer's ``await`` —
is wrapped in ``contextlib.suppress(Exception)`` so a misbehaving
observer adapter (Redis hiccup, kafka push error, slow subscriber
timeout) can't propagate into the agent loop. These tests pin
that contract at both ends: the trace side and the observer side.
"""

from __future__ import annotations

import asyncio

from agentkit.kernel.observation import Observation
from agentkit.kernel.types import Scope
from agentkit.runtime import Budget, RunContext, Services


def _run(coro):
    return asyncio.run(coro)


class _CrashingObserver:
    """Always raises on emit — simulates a broken adapter (provider down,
    network error, encoder failure)."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.attempts = 0

    async def emit(self, obs: Observation) -> None:  # noqa: ARG002
        self.attempts += 1
        raise self.exc


class _RecordingObserver:
    """Captures every emission — the happy path."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    async def emit(self, obs: Observation) -> None:
        self.observations.append(obs)


def _ctx(observer) -> RunContext:
    return RunContext(
        "run-x",
        Scope(),
        Budget(),
        Services(observer=observer),
    )


def test_emit_swallows_observer_exception():
    """A broken observer must NOT propagate. Otherwise the agent loop
    would crash for an observability side-effect."""
    obs = _CrashingObserver(RuntimeError("redis publish failed"))
    ctx = _ctx(obs)

    # Must not raise.
    _run(ctx.emit("info", "hello", payload={"x": 1}))
    # And the observer WAS called — we didn't skip emit; we caught.
    assert obs.attempts == 1


def test_emit_swallows_observer_exception_on_critical_kind():
    """Defensive — CRITICAL kinds (result/error) take an extra trace path
    after the observer.emit call. A failing observer must not prevent
    that path from running; the suppress is around emit only, the trace
    add_event happens regardless (also wrapped in its own suppress)."""
    obs = _CrashingObserver(RuntimeError("kafka timeout"))
    ctx = _ctx(obs)

    # ``error`` is in CRITICAL_KINDS — exercises both the observer.emit
    # path AND the post-emit trace path. Neither should raise.
    _run(ctx.emit("error", "the run failed", payload={"e": "X"}))
    assert obs.attempts == 1


def test_emit_happy_path_still_records_observation():
    """Regression check — the happy path must still actually deliver
    the Observation. The suppress is around a single call site, not the
    whole method."""
    obs = _RecordingObserver()
    ctx = _ctx(obs)

    _run(ctx.emit("info", "hello", payload={"v": 42}, agent="planner"))
    assert len(obs.observations) == 1
    o = obs.observations[0]
    assert o.kind == "info"
    assert o.render == "hello"
    assert o.payload == {"v": 42}
    assert o.agent == "planner"
    assert o.run_id == "run-x"


def test_emit_swallows_async_only_exception_types():
    """An observer that raises a non-trivial async exception
    (TimeoutError from asyncio.wait_for, etc.) is still handled. The
    suppress catches ``Exception`` broadly so any non-BaseException
    class is contained."""
    obs = _CrashingObserver(TimeoutError("subscriber slow"))
    ctx = _ctx(obs)
    _run(ctx.emit("info", "ping"))
    assert obs.attempts == 1


def test_emit_does_not_swallow_cancellation():
    """Critical defensive property — ``CancelledError`` is a
    BaseException subclass (not Exception). It MUST propagate so
    cooperative cancellation still works. The ``suppress(Exception)``
    catches the wide class but leaves BaseException alone."""

    class _CancelObserver:
        async def emit(self, obs: Observation) -> None:  # noqa: ARG002
            raise asyncio.CancelledError()

    obs = _CancelObserver()
    ctx = _ctx(obs)

    cancelled = False
    try:
        _run(ctx.emit("info", "should not be swallowed"))
    except asyncio.CancelledError:
        cancelled = True
    assert cancelled
