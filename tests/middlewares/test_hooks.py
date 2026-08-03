"""Lifecycle hooks (ch22): `Hooks` is a thin ObserverPort — `on(stage, handler)` dispatches each
observation by kind (or "*"), sync or async, and a handler never breaks the run. It's sugar over the
observation stream (ch23), not a new mechanism. Offline & deterministic."""

import asyncio

from agentkit.adapters.observer import CollectingObserver, Hooks
from agentkit.agents import Agent, RoundRobinPolicy
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.control import MaxMessages
from agentkit.kernel.observation import Observation
from agentkit.kernel.types import Scope
from agentkit.runtime import Budget, Invoker, RunContext
from agentkit.runtime.context import Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


def test_dispatches_handlers_by_stage():
    seen = []
    hooks = (
        Hooks()
        .on("summary", lambda o: seen.append(("s", o.render)))
        .on("result", lambda o: seen.append(("r", o.render)))
    )

    async def go():
        await hooks.emit(Observation(kind="summary", render="a"))
        await hooks.emit(Observation(kind="result", render="done"))
        await hooks.emit(Observation(kind="progress", render="ignored"))  # no handler → no-op

    _run(go())
    assert seen == [("s", "a"), ("r", "done")]


def test_wildcard_fires_for_every_observation():
    kinds = []
    hooks = Hooks().on("*", lambda o: kinds.append(o.kind))
    _run(_emit_each(hooks, ["run_start", "summary", "result"]))
    assert kinds == ["run_start", "summary", "result"]


def test_async_handlers_are_awaited():
    seen = []

    async def handler(o):
        seen.append(o.kind)

    hooks = Hooks().on("error", handler)
    _run(hooks.emit(Observation(kind="error", render="boom")))
    assert seen == ["error"]


def test_a_raising_handler_never_breaks_the_run():
    seen = []
    hooks = (
        Hooks()
        .on("summary", lambda o: (_ for _ in ()).throw(RuntimeError("bad hook")))  # raises
        .on("summary", lambda o: seen.append("second still ran"))
    )  # must still fire
    _run(hooks.emit(Observation(kind="summary", render="x")))  # must not raise
    assert seen == ["second still ran"]


def test_forwards_to_inner_observer():
    sink = CollectingObserver()
    hooks = Hooks(inner=sink).on("summary", lambda o: None)
    _run(hooks.emit(Observation(kind="summary", render="kept")))
    assert [o.render for o in sink.items] == ["kept"]  # chained through to the inner observer


def test_hooks_capture_coordinator_lifecycle_stages():
    """A coordinator Agent's `RoundRobinPolicy` emits the standard lifecycle observations
    (`run_start` → `summary`(s) → `result`) on the observer threaded through `RunContext`.
    Hooks is an ObserverPort, so wiring it as the run's observer captures the whole tape."""
    stages = []
    hooks = Hooks().on("*", lambda o: stages.append(o.kind))
    ctx = RunContext(
        "r", Scope(1, 2), Budget(), Services(invoker=Invoker(llm=FakeLLM("ok")), observer=hooks)
    )
    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children={"a": Agent("a", "m")},
            policy=RoundRobinPolicy(),
            termination=MaxMessages(1),
        ),
    )
    _run(coord.run("start", ctx))
    assert stages[0] == "run_start" and "summary" in stages and stages[-1] == "result"


async def _emit_each(observer, kinds):
    for k in kinds:
        await observer.emit(Observation(kind=k, render=k))
