"""Checkpointer — the capability over `CheckpointPort`.

A producer (leaf `Agent`, coordinator `Agent`, a custom workflow) that wants durable,
resumable runs takes a
`Checkpointer` and calls `snapshot(state, status=...)` at meaningful transitions. The
Checkpointer handles version numbering (monotonic, owned by the latest-version of the
port — one producer per `run_id`) and clock stamping (deterministic via injected
`ClockPort`, else `time.time()`).

Resume reads `latest(run_id)`; time-travel/replay reads `at_version(run_id, n)`. Neither
interprets `state` — that's the producer's contract.

Why this is a capability, not the port itself. The port is the seam (save/latest/version
range/delete); the capability is the *ergonomic facade* every producer reaches for so each
one doesn't re-derive "next version = latest + 1" and "stamp time from the clock if I have
one". Keeping it thin means the port stays substitutable (in-memory today, Postgres/Redis
tomorrow) without re-touching every pattern.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import time
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.ports import (
    Checkpoint,
    CheckpointPort,
    CheckpointStatus,
    ClockPort,
)

# Duplicated from ``agents.control.elicitation.SECRET_TAINT_KEY`` rather than
# imported: ``capabilities`` sits BELOW ``agents`` and importing upward would
# invert the layering (and drag the agents tree into every checkpoint wiring).
# ``tests/agents/test_hitl_elicitation.py`` pins the two constants equal, so
# the duplication cannot drift silently.
_SECRET_TAINT_KEY = "_agentkit_secret_taint"


def _is_tainted(state: dict[str, Any]) -> bool:
    """Does this state blob carry a secret-taint marker?

    The marker is set on a ``WorkingContext.scratchpad`` by
    ``agents.control.elicitation.mark_context_tainted`` when a person supplies a
    value to a ``secret=True`` elicitation. Producers serialise the scratchpad
    into ``state``, so the marker travels with the thing that must not be
    persisted — which is why the check can live at this single seam instead of
    being re-implemented at all seven ``snapshot`` call sites.

    Checks the top level AND a nested ``scratchpad`` because producers differ:
    the tool loop nests it under ``"scratchpad"``, a caller passing a bare
    scratchpad as the whole state would have it flat.
    """
    if state.get(_SECRET_TAINT_KEY):
        return True
    nested = state.get("scratchpad")
    return bool(isinstance(nested, dict) and nested.get(_SECRET_TAINT_KEY))


def resolve_checkpointer(ctx: Any, explicit: Checkpointer | None = None) -> Checkpointer | None:
    """The ONE durable-state resolution order, shared by every producer.

    ``explicit`` (a producer's own ``checkpointer=``) > ``ctx.checkpointer`` >
    a legacy bridge synthesized over ``ctx.store`` (so a wiring that only
    injected a ``StorePort`` keeps working). ``None`` when nothing is wired,
    which callers treat as "durable resume disabled".

    This lives here rather than on any one producer because having two
    resolution orders is how a seam silently stops working: ``Workflow``
    persisted only through ``ctx.store`` while the ReAct cognition preferred
    ``ctx.checkpointer``, so wiring the documented durable seam — a
    ``Checkpointer`` — left workflow human-gates unpersisted, and the failure
    surfaced later as "no suspended workflow <id> to resume". One function
    means a new producer cannot invent a third order.
    """
    if explicit is not None:
        return explicit
    found: Checkpointer | None = getattr(ctx, "checkpointer", None)
    if found is not None:
        return found
    store = getattr(ctx, "store", None)
    if store is not None:
        from agentkit.capabilities.checkpointer.persistence import StoreBackedCheckpointStore

        return Checkpointer(port=StoreBackedCheckpointStore(store))
    return None


def _snapshot_span(ctx: Any, run_id: str, status: CheckpointStatus | str) -> Any:
    """Best-effort span for ``Checkpointer.snapshot``. Yields a
    context manager (real span or a nullcontext) — the checkpointer is
    used outside a trace-bearing ctx in tests, and a misbehaving tracer
    must never break persistence. ``status`` may be a ``CheckpointStatus``
    enum OR a raw string (some producers pass strings like ``"running"``
    directly); both flatten to a plain string on the span attribute."""
    trace = getattr(ctx, "trace", None) if ctx is not None else None
    if trace is None:
        return contextlib.nullcontext()
    status_str = status.value if hasattr(status, "value") else str(status)
    try:
        return trace.span(
            "checkpointer.snapshot",
            "internal",
            run_id=run_id,
            status=status_str,
        )
    except Exception:
        return contextlib.nullcontext()


@dataclass
class Checkpointer:
    """Capability over a `CheckpointPort`.

    `snapshot()` bumps the version, stamps `created_at`, and saves. `resume()` returns the
    latest checkpoint (or `None` if there's nothing to resume from). `at_version()` is the
    time-travel hook — return a specific historical snapshot, or `None` if it doesn't
    exist. `delete()` clears all versions for a `run_id` (terminal cleanup).
    """

    port: CheckpointPort
    clock: ClockPort | None = None
    # Per-run lock. ``snapshot`` reads the current version list then
    # writes a new one — two concurrent snapshots for the same
    # ``run_id`` would otherwise both compute ``next_version = max + 1``
    # off the same read, then race on ``port.save``. On a real
    # (suspending) backend the second save either overwrites the first
    # (silent data loss) or hits a duplicate-key error (production
    # noise). The lock serialises the read-compute-write cycle
    # per-``run_id`` so concurrent callers see monotonic distinct
    # versions.
    _run_locks: dict[str, asyncio.Lock] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    async def snapshot(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        status: CheckpointStatus = CheckpointStatus.RUNNING,
        metadata: dict[str, Any] | None = None,
        ctx: Any = None,
    ) -> Checkpoint:
        """Persist a new versioned snapshot. Returns the saved Checkpoint so the caller can
        log/expose its version. Failures from `port.save` propagate — the caller decides
        whether a checkpoint-save failure should abort the run.

        ``state`` and ``metadata`` are deep-copied at the seam.
        ``Checkpoint`` is a frozen shell, but its ``state`` / ``metadata``
        dicts are stored by reference; ``InMemoryStore`` returns those
        refs on resume. Without the copy, a caller that keeps a live
        handle to the scratch dict (or a resumer that pops entries from
        ``cp.state``) would mutate the durable record — parity with any
        serializing backend (Redis, Postgres) would break silently.
        Deep-copy models the wire semantics locally so in-memory tests
        stay honest about the durability contract.

        ``ctx`` is optional: when supplied and it carries a ``TracePort``
        via ``ctx.trace``, a ``checkpointer.snapshot`` span is opened
        around the save so operators can see checkpoint activity in the
        trace timeline. Callers without a ctx (bare unit tests, offline
        replayers) leave it unset and the span helper degrades to a
        ``nullcontext``.

        **A SECRET-TAINTED state is never persisted** — see
        :func:`_is_tainted`. This check lives here, at the single seam
        every producer passes through, rather than in each cognition or
        policy: there are seven ``snapshot`` call sites across the tool
        loop and the coordinator policies, and a containment rule that
        has to be re-implemented at each one is a rule that will be
        missed at the eighth."""
        # The per-run lock scopes to ONE Checkpointer instance in
        # one process; cross-process concurrent producers still race
        # (the docstring warns "one producer per run_id"), but the
        # far more common in-process race — a run's own hot loop
        # plus a background health-snapshot task, two policy
        # branches both writing at a phase boundary — is closed
        # here so callers can compose without threading a lock
        # through their own code.
        if _is_tainted(state):
            # Refuse the write and return the shape the caller expects, so a
            # producer that never opted into secrets needs no new branch. The
            # run continues; only its durability is given up — an un-resumable
            # run can be re-run, a leaked one-time code cannot be un-leaked.
            return Checkpoint(
                run_id=run_id,
                version=0,  # 0 = never persisted; a real version starts at 1
                state={},
                status=status,
                created_at=self.clock.now() if self.clock is not None else time.time(),
                metadata={"skipped": "secret_taint"},
            )
        with _snapshot_span(ctx, run_id, status):
            lock = self._run_locks.setdefault(run_id, asyncio.Lock())
            async with lock:
                versions = await self.port.list_versions(run_id)
                next_version = (max(versions) + 1) if versions else 1
                now = self.clock.now() if self.clock is not None else time.time()
                cp = Checkpoint(
                    run_id=run_id,
                    version=next_version,
                    state=copy.deepcopy(state),
                    created_at=now,
                    status=status,
                    metadata=copy.deepcopy(dict(metadata)) if metadata else {},
                )
                await self.port.save(cp)
                return cp

    async def resume(self, run_id: str, *, include_terminal: bool = False) -> Checkpoint | None:
        """Return the latest RESUMABLE checkpoint for `run_id`, or `None`.

        By default, a terminal snapshot (`DONE` / `FAILED`) is treated as
        "no resumable state" and this returns `None` — a naive
        "resume if any checkpoint exists" wiring MUST NOT re-run a
        finished job. Callers that legitimately want to inspect the
        terminal snapshot (an audit UI, an operator tool, replay /
        time-travel debugging) opt in explicitly with
        `include_terminal=True`.

        ``status`` on the loaded checkpoint may be a raw string (some
        producers, and the ``Checkpoint`` shape's own tests, pass strings
        directly) rather than a ``CheckpointStatus`` enum instance;
        because ``CheckpointStatus`` is a ``StrEnum``, wrapping the
        value normalises both cases without an ``isinstance`` branch."""
        cp = await self.port.latest(run_id)
        if cp is None:
            return None
        if not include_terminal and CheckpointStatus(cp.status).is_terminal():
            return None
        return cp

    async def at_version(self, run_id: str, version: int) -> Checkpoint | None:
        """Return the checkpoint at exactly `version`, or `None` if missing.
        The time-travel / replay seam."""
        return await self.port.at_version(run_id, version)

    async def list_versions(self, run_id: str) -> list[int]:
        """Return all known versions for `run_id` in ascending order."""
        return await self.port.list_versions(run_id)

    async def delete(self, run_id: str) -> None:
        """Clear all checkpoints for `run_id`. Idempotent — deleting an unknown run is a no-op.

        Also evicts the per-run ``asyncio.Lock`` stashed in
        ``_run_locks`` — without this, a long-lived Checkpointer serving
        many short-lived runs (a shared engine pool, a coordinator
        spawning throwaway children) would accumulate one lock per
        distinct ``run_id`` for the process lifetime. Deletion is the
        producer's signal that a run is over, so it's the natural
        eviction point. The lock is dropped only AFTER ``port.delete``
        succeeds — a failed backend delete leaves the lock in place so a
        retry stays serialised with any in-flight snapshot for the same
        run."""
        await self.port.delete(run_id)
        self._run_locks.pop(run_id, None)


__all__ = ["Checkpointer"]
