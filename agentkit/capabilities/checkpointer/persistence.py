"""Durable-checkpoint helpers shared by `Agent`'s tool-loop branch and the coordinator loop.

A run that uses tools must be **resumable**: a crash mid-tool-call should leave a
checkpoint a second `stream()`/`resume()` pass can pick up from. These helpers are the pure
plumbing — the agent-state shape conversion (messages/usage/tool-calls ↔ dict) — so the loop
module reads as policy only. The coordinator-state helpers (`coord_state_to_dict` /
`coord_state_from_dict`) follow the same shape language so leaf-agent and coordinator
checkpoints read consistently.

The *transport* (where the snapshot lives and how versions are numbered) is no longer here;
that's the `Checkpointer` capability + `CheckpointPort`. A wiring that injects only
`ctx.store` still gets durable resume through `StoreBackedCheckpointStore`, which
`capabilities.checkpointer.resolve_checkpointer` synthesizes on demand — that is the
ergonomic default, not a compatibility shim.

EVERY producer resolves through that one function — the tool loop, `Workflow`, the
coordinator policies, `PlanPolicy`'s human gate. An earlier version of this docstring
claimed coordinator runs deliberately excluded the store-backed port and "require a real
`Checkpointer`"; that divergence was not a simplification, it was a silent failure — a
`Services(store=...)` wiring left a completed coordinator run with zero keys in the store
and no warning. It is exactly as durable as the store behind it, and its one limitation
(a single slot per run, no version history) costs nothing to a producer that only ever
reads `latest`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentkit.context import PrefixContext, WorkingContext
from agentkit.kernel._frozen import thaw
from agentkit.kernel.ports import Checkpoint
from agentkit.kernel.types import Message, ToolCall, Usage

if TYPE_CHECKING:
    from agentkit.agents.result import AgentResult


def tc_to_dict(t: ToolCall) -> dict[str, Any]:
    # The PORTABILITY reason this unwrap was written for is gone.
    # ``arguments`` is a ``FrozenDict`` now, not a ``MappingProxyType``, and a
    # ``FrozenDict`` json-dumps, deep-copies and pickles like the plain dict it
    # subclasses — so no backend needs the copy.
    #
    # Not merely weaker than it was: the specific hole an earlier note pointed
    # at is CLOSED. ``deep_freeze`` now normalises a
    # ``MappingProxyType`` into a ``FrozenDict``, nested ones included, so the
    # "a caller passed a proxy and it is still sitting here" scenario cannot
    # happen through ``ToolCall`` any more (measured:
    # ``ToolCall("c", "s", MappingProxyType({...})).arguments`` is a
    # ``FrozenDict``). What ``deep_freeze`` still hands back by identity is any
    # OTHER ``Mapping`` — rewriting a caller's own type is the line it refuses
    # to cross — and ``t`` is annotated ``ToolCall`` with ``arguments:
    # dict[str, Any]``, so landing one of those here means a caller mypy would
    # already have rejected. Real, but not the reason this line exists.
    #
    # The reason it exists is the CONTRACT. These helpers are exported
    # (``agentkit.capabilities.checkpointer.tc_to_dict``): they are the shape an
    # app builds its own snapshot out of, and the annotation promises a plain
    # ``dict``. Without the unwrap ``arguments`` comes back frozen, and a caller
    # assembling state — ``state["pending"][0]["arguments"]["x"] = ...`` — gets a
    # ``TypeError`` from inside a value it built itself and never asked to be
    # immutable. Measured both ways: with the unwrap the top level is a mutable
    # ``dict`` and that assignment lands; without it, a ``FrozenDict`` that
    # refuses it. The copy is SHALLOW and deliberately so — measured, the nested
    # ``arguments["n"]`` stays a ``FrozenDict`` and ``["n"]["y"] = 1`` still
    # raises. The freeze belongs on the ToolCall, which is the record; this
    # function only unwraps the one layer the caller is handed.
    return {"id": t.id, "name": t.name, "arguments": dict(t.arguments)}


def dict_to_tc(d: dict[str, Any]) -> ToolCall:
    return ToolCall(d["id"], d["name"], d.get("arguments", {}))


def msg_to_dict(m: Message) -> dict[str, Any]:
    return {
        "role": m.role,
        "content": m.content,
        "tool_call_id": m.tool_call_id,
        "name": m.name,
        "tool_calls": [tc_to_dict(t) for t in m.tool_calls],
    }


def dict_to_msg(d: dict[str, Any]) -> Message:
    # Build by keyword — Message's field order is (role, content, name, tool_call_id,
    # tool_calls), which doesn't match the natural "shape" order we serialize. Positional
    # construction here would silently assign the tool-call tuple to `name`, which
    # round-trips fine through Agent's loop (rehydrated, never re-serialized inside the
    # run) but breaks the team's snapshot path on resume (`msg_to_dict` would iterate a
    # tuple-shaped `name` and a `None` tool_calls).
    return Message(
        role=d["role"],
        content=d.get("content", ""),
        name=d.get("name"),
        tool_call_id=d.get("tool_call_id"),
        tool_calls=tuple(dict_to_tc(t) for t in (d.get("tool_calls") or [])),
    )


def usage_to_dict(u: Usage) -> dict[str, Any]:
    return {"input": u.input_tokens, "output": u.output_tokens, "cost": u.cost_usd}


def prefix_to_dict(p: PrefixContext) -> dict[str, Any]:
    """Serialize a ``PrefixContext`` (the cache-stable head of a
    ``WorkingContext``). Round-trips the system prompt + grounding
    messages + schema block — the three fields a v2 checkpoint must
    rehydrate to reconstruct an identical prefix.
    """
    return {
        "system_prompt": p.system_prompt,
        "grounding": [msg_to_dict(m) for m in p.grounding],
        "schema_block": p.schema_block,
    }


def dict_to_prefix(d: dict[str, Any]) -> PrefixContext:
    """Inverse of ``prefix_to_dict``.

    Takes the block ``prefix_to_dict`` writes, and only that. An earlier
    version also accepted ``None``/absent and returned an empty
    ``PrefixContext``, so that a snapshot written before the context split
    (system prompt inline as the first ``messages`` entry) still resumed.
    That second shape has no writer: ``ReActCognition._save`` is the only
    thing in the tree that produces a ``{"messages": ...}`` state and it has
    always emitted ``prefix`` since the split, which landed unreleased — the
    only tagged version is 0.1.0 and it predates ``PrefixContext``
    entirely. A decoder branch that no writer can reach is not tolerance,
    it is a silent default: a state that lost its ``prefix`` on the wire
    would have rehydrated with an EMPTY system prompt and re-run the agent
    with no instructions. Requiring the key turns that into a ``KeyError``
    naming the field, which is what the rest of ``rehydrate``'s required
    set already does.

    The per-field ``.get`` defaults below are a different thing and stay:
    they let a caller hand-assembling a prefix block omit the parts it does
    not use (these helpers are exported for exactly that), and they cost
    nothing because each field's absence has one obvious meaning.
    """
    if not isinstance(d, dict):
        # An explicit ``None`` used to be the second accepted shape, so it is
        # the one malformed value a caller is most likely to still send. Reject
        # it by NAME: without this it reaches ``d.get`` and surfaces as
        # ``AttributeError: 'NoneType' object has no attribute 'get'`` from
        # inside the decoder, which says nothing about which field was wrong.
        # An absent key already fails clearly with ``KeyError('prefix')``; this
        # makes the null case fail just as clearly.
        raise TypeError(
            f"checkpoint field 'prefix' must be the block prefix_to_dict writes, "
            f"got {type(d).__name__}. A prefix-less state is not resumable — it "
            f"would restore an empty system prompt and re-run the agent with no "
            f"instructions."
        )
    return PrefixContext(
        system_prompt=d.get("system_prompt", "") or "",
        grounding=tuple(dict_to_msg(m) for m in d.get("grounding") or []),
        schema_block=d.get("schema_block"),
    )


def rehydrate(saved: dict[str, Any]) -> tuple[WorkingContext, Usage, int, bool]:
    """Rebuild ``(WorkingContext, Usage, next_iteration, repaired)`` from
    a checkpoint dict.

    ONE shape, the one ``ReActCognition._save`` writes. Four keys are
    REQUIRED — the ``prefix`` block (system prompt + grounding + schema),
    the tail ``messages``, the accrued ``usage`` and the next
    ``iteration`` — and a missing one raises a ``KeyError`` naming it — an operator gets a diagnosable failure rather
    than a run that quietly resumes with half its state. ``scratchpad`` /
    ``limit`` / ``shared`` / ``repaired`` stay optional because each has a
    single unambiguous default (empty, unbounded, unshared, unrepaired)
    that is indistinguishable from having been written.
    """
    context = WorkingContext(
        prefix=dict_to_prefix(saved["prefix"]),
        messages=[dict_to_msg(d) for d in saved["messages"]],
        # ``thaw``, not a bare pass-through and not ``dict(...)``. The stored
        # state is deep-frozen (a durable record should be), but what we are
        # building here is a LIVE working context that the resumed run writes
        # to. Measured before this call existed: the first ``ctx.note(...)``
        # after a durable resume raised ``TypeError: this payload belongs to a
        # frozen value``. A top-level ``dict(...)`` only moves that failure to
        # the first NESTED write.
        scratchpad=thaw(saved.get("scratchpad") or {}),
        limit=saved.get("limit"),
        shared=bool(saved.get("shared", False)),
    )
    u = saved["usage"]
    return (
        context,
        Usage(u["input"], u["output"], u["cost"]),
        saved["iteration"],
        saved.get("repaired", False),
    )


# ---- the store-backed CheckpointPort ------------------------------------------------------


def ckpt_key(run_id: str) -> str:
    """The KV key `StoreBackedCheckpointStore` reads and writes for `run_id`.

    Public because "where is this run's checkpoint in my store?" is an operator
    question — an admin tool clearing a stuck suspend, or a test asserting a
    snapshot really reached the backend, needs the same derivation the port
    uses, and hardcoding `checkpoint:<id>` in a second place is how the two
    drift. `tests/adapters/test_durable_resume_backends.py` asserts against
    this key precisely so the store round trip stays pinned to it.
    """
    return f"checkpoint:{run_id}"


class StoreBackedCheckpointStore:
    """A minimal `CheckpointPort` over a generic `StorePort`.

    This is what makes `Services(store=...)` a durable wiring. `resolve_checkpointer`
    synthesizes one when a `ctx.store` is present and no `Checkpointer` was injected,
    which is the ergonomic path every producer depends on: the tool loop, `Workflow`,
    the coordinator policies and `PlanPolicy`'s human gate all resolve through that one
    function, and `Workflow`'s own unpersisted-gate warning names `Services(store=...)`
    to the user as one of the two ways to make a suspend resumable. Measured by deleting
    the synthesis and running the suite: 23 tests fail across workflow gates, plan gates,
    tool-loop approve/resume and coordinator slot isolation. It is the DEFAULT, not a
    compatibility shim — an app with a `StorePort` already wired should not also have to
    hand-construct a `CheckpointPort` to get resume.

    Wire a real `CheckpointPort` adapter (`InMemoryCheckpointStore`, `PostgresCheckpointStore`)
    when you want version HISTORY, which is the one thing this cannot give you.

    The single-slot-per-run model is deliberate: only the latest version survives on the
    store, so time-travel / replay is unavailable here and `Checkpointer.list_versions()`
    returns at most one element. A generic KV cannot losslessly model a version history
    without leaking a schema into `StorePort`, and resume — which reads `latest` and
    nothing else — pays nothing for the limitation.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    @staticmethod
    def _key(run_id: str) -> str:
        return ckpt_key(run_id)

    @staticmethod
    def _to_dict(cp: Checkpoint) -> dict[str, Any]:
        return {
            "run_id": cp.run_id,
            "version": cp.version,
            "state": cp.state,
            "created_at": cp.created_at,
            "status": cp.status,
            "metadata": cp.metadata,
        }

    @staticmethod
    def _from_dict(d: dict[str, Any]) -> Checkpoint:
        return Checkpoint(
            run_id=d["run_id"],
            version=d["version"],
            state=d["state"],
            created_at=d["created_at"],
            status=d["status"],
            metadata=d.get("metadata") or {},
        )

    async def save(self, cp: Checkpoint) -> None:
        await self._store.set(self._key(cp.run_id), self._to_dict(cp))

    async def latest(self, run_id: str) -> Checkpoint | None:
        raw = await self._store.get(self._key(run_id))
        return self._from_dict(raw) if raw else None

    async def at_version(self, run_id: str, version: int) -> Checkpoint | None:
        cp = await self.latest(run_id)
        return cp if cp and cp.version == version else None

    async def list_versions(self, run_id: str) -> list[int]:
        cp = await self.latest(run_id)
        return [cp.version] if cp else []

    async def delete(self, run_id: str) -> None:
        await self._store.delete(self._key(run_id))


# ---- coordinator-state shape ------------------------------------------------------------


def result_to_dict(r: Any) -> dict[str, Any]:
    """Serialize an `AgentResult` into a flat dict. The `parsed` field is passed through
    raw — for a serializing port the caller is responsible for ensuring parsers produce
    primitive/JSON-safe values (matches the workflow durable-resume contract)."""
    return {
        "output": r.output,
        "usage": usage_to_dict(r.usage),
        "partial": r.partial,
        "evals": dict(r.evals or {}),
        "parsed": r.parsed,
        "prompt_version": r.prompt_version,
        # Persisted explicitly: it is a CLOSED taxonomy a reader branches on,
        # and re-deriving it from ``evals`` on the way back only works for
        # producers that also record a free-form reason. Omitting it silently
        # downgraded every rehydrated result to ``"complete"``.
        "stop_reason": r.stop_reason,
    }


def dict_to_result(d: dict[str, Any]) -> AgentResult:
    """Inverse of `result_to_dict`."""
    from agentkit.agents.result import (  # local import — avoids any forward-ref churn
        AgentResult,
        stop_reason_for,
    )

    u = d.get("usage") or {"input": 0, "output": 0, "cost": 0.0}
    return AgentResult(
        output=d.get("output", ""),
        usage=Usage(u.get("input", 0), u.get("output", 0), u.get("cost", 0.0)),
        partial=d.get("partial", False),
        evals=dict(d.get("evals") or {}),
        parsed=d.get("parsed"),
        prompt_version=d.get("prompt_version", ""),
        # A record written before ``stop_reason`` was persisted has no key. Fall
        # back to deriving it from the free-form ``evals`` reason, which those
        # records DO carry — so an old checkpoint upgrades rather than reading
        # back as a bare "complete".
        stop_reason=d.get("stop_reason") or stop_reason_for((d.get("evals") or {}).get("stop_reason")),
    )


def coord_state_to_dict(
    *,
    turn: int,
    transcript: list[Message],
    scratchpad: dict[str, Any] | None,
    results: list[Any],
    usage: Usage,
    stop_reason: str | None,
    status: str,
) -> dict[str, Any]:
    """Serialize a coordinator Agent's mid-run state into a flat dict suitable for
    `Checkpoint.state`.

    Shape: `turn` is the **next** turn index to execute on resume (so a snapshot taken
    after completing turn N stores `turn=N+1`); `transcript` is the full message list
    (works for blackboard mode too — on resume the caller's `WorkingContext.messages` is
    repopulated from this); `scratchpad` is the blackboard scratchpad (or `{}` when no
    blackboard); `results` is the per-turn `AgentResult` list serialized so far; `usage`
    is the accumulated total; `stop_reason` is the termination/ceiling reason (typically
    `None` mid-run, set on a `"done"` snapshot); `status` is `"running"` / `"done"` /
    `"failed"`. Re-uses `msg_to_dict` and `usage_to_dict` so leaf-agent and coordinator
    checkpoints share a shape language.

    Notes on what is intentionally NOT serialized:
        * The children dict — children are passed at construction; the same coordinator
          Agent must be rebuilt to resume.
        * Termination-condition state — most built-ins are stateless or re-derivable
          from the transcript (`MaxMessages`/`MaxTurns` counters == turn count).
          `Timeout` (wall-clock start), `ExternalTermination` (flag), and a stateful
          `FunctionalTermination` predicate are NOT re-derivable; on resume we
          `.reset()` and replay the rehydrated assistant deltas through the condition
          so cumulative counters catch up. Wall-clock conditions effectively restart
          their timer on resume — flagged for follow-up.
        * Policy state — selectors and assessors are treated as stateless (the
          built-ins read the last message / call a model). A policy with private state
          would need its own serialization seam.
    """
    return {
        "turn": int(turn),
        "transcript": [msg_to_dict(m) for m in transcript],
        "scratchpad": dict(scratchpad or {}),
        "results": [result_to_dict(r) for r in results],
        "usage": usage_to_dict(usage),
        "stop_reason": stop_reason,
        "status": status,
    }


def coord_state_from_dict(
    d: dict[str, Any],
) -> tuple[int, list[Message], dict[str, Any], list[Any], Usage]:
    """Rehydrate `(turn, transcript, scratchpad, results, usage)` from a `Checkpoint.state`.

    The inverse of `coord_state_to_dict`. `stop_reason` and `status` aren't returned —
    callers read them off the surrounding `Checkpoint`/state dict directly when they
    matter (e.g. gating resume on `status != "done"`)."""
    u = d.get("usage") or {"input": 0, "output": 0, "cost": 0.0}
    return (
        int(d.get("turn", 0)),
        [dict_to_msg(m) for m in d.get("transcript", [])],
        thaw(d.get("scratchpad") or {}),  # same reason as ``rehydrate`` above
        [dict_to_result(r) for r in d.get("results", [])],
        Usage(u.get("input", 0), u.get("output", 0), u.get("cost", 0.0)),
    )


__all__ = [
    "StoreBackedCheckpointStore",
    "ckpt_key",
    "coord_state_from_dict",
    "coord_state_to_dict",
    "dict_to_msg",
    "dict_to_prefix",
    "dict_to_result",
    "dict_to_tc",
    "msg_to_dict",
    "prefix_to_dict",
    "rehydrate",
    "result_to_dict",
    "tc_to_dict",
    "usage_to_dict",
]
