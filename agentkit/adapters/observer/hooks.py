"""`Hooks` — lifecycle subscription over the observation stream."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from agentkit.kernel.observation import Observation


class Hooks:
    """Lifecycle subscription: a thin `ObserverPort` that dispatches each observation to handlers
    registered with `on(stage, handler)`, where `stage` is the observation `kind` (e.g. `run_start`,
    `summary`, `interrupt`, `error`, `result`) or `"*"` for every observation.

    Lifecycle is **not a new mechanism** — it's the observation stream, named. Handlers are sync or
    async and **never break the run**: an exception in a handler is swallowed.
    Optionally forwards to an `inner` observer, so `Hooks` chains in front of a `QueueObserver`."""

    def __init__(self, inner: Any | None = None) -> None:
        self._handlers: dict[str, list[Callable[[Observation], Any]]] = {}
        self._inner = inner

    def on(self, stage: str, handler: Callable[[Observation], Any]) -> Hooks:
        self._handlers.setdefault(stage, []).append(handler)
        return self

    async def emit(self, obs: Observation) -> None:
        for stage in (obs.kind, "*"):
            for handler in self._handlers.get(stage, ()):
                try:
                    r = handler(obs)
                    if inspect.isawaitable(r):
                        await r
                except Exception:        # noqa: BLE001 — a hook must never destabilise the observed run
                    pass
        if self._inner is not None:
            await self._inner.emit(obs)
