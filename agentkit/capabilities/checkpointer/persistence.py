"""Durable-checkpoint helpers shared by `Agent`'s tool-loop branch and the coordinator loop.

A run that uses tools must be **resumable**: a crash mid-tool-call should leave a
checkpoint a second `stream()`/`resume()` pass can pick up from. These helpers are the pure
plumbing — the agent-state shape conversion (messages/usage/tool-calls ↔ dict) — so the loop
module reads as policy only. The coordinator-state helpers (`coord_state_to_dict` /
`coord_state_from_dict`) follow the same shape language so leaf-agent and coordinator
checkpoints read consistently.

The *transport* (where the snapshot lives and how versions are numbered) is no longer here;
that's the `Checkpointer` capability + `CheckpointPort`. `Agent` keeps a thin legacy bridge
(`StoreBackedCheckpointStore`) so an existing wiring that injects only `ctx.store` still
gets durable resume — the bridge is constructed on demand inside `Agent`, not exported.
Coordinator runs do NOT carry the legacy bridge — they require a real `Checkpointer` if
they want durability. This deliberate divergence keeps the coordinator path simpler (one
resolution rule instead of three) while preserving Agent's back-compat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentkit.context import PrefixContext, WorkingContext
from agentkit.kernel.ports import Checkpoint
from agentkit.kernel.types import Message, ToolCall, Usage

if TYPE_CHECKING:
    from agentkit.agents.result import AgentResult


def tc_to_dict(t: ToolCall) -> dict[str, Any]:
    # ``ToolCall.arguments`` is a ``MappingProxyType``. Serialize it as
    # a plain ``dict`` so downstream deep-copy / JSON / pickle backends
    # stay portable — ``mappingproxy`` is neither pickleable nor
    # JSON-serializable by default.
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


def dict_to_prefix(d: dict[str, Any] | None) -> PrefixContext:
    """Inverse of ``prefix_to_dict``. Tolerates ``None`` so a
    pre-context-split checkpoint round-trips into an empty prefix
    instead of crashing on missing keys."""
    if not d:
        return PrefixContext()
    return PrefixContext(
        system_prompt=d.get("system_prompt", "") or "",
        grounding=tuple(dict_to_msg(m) for m in d.get("grounding") or []),
        schema_block=d.get("schema_block"),
    )


def rehydrate(saved: dict[str, Any]) -> tuple[WorkingContext, Usage, int, bool]:
    """Rebuild ``(WorkingContext, Usage, next_iteration, repaired)`` from
    a checkpoint dict.

    The shape is implicitly v2 after the context-split: the dict now
    carries a ``prefix`` block (system prompt + grounding + schema)
    alongside the tail messages and scratchpad. A missing ``prefix``
    key (older snapshot) round-trips into an empty ``PrefixContext``,
    so a checkpoint written before the split still rehydrates — its
    transcript will have the prompt as the first ``system`` message in
    ``messages`` (the legacy shape), which is functionally equivalent
    on resume.
    """
    context = WorkingContext(
        prefix=dict_to_prefix(saved.get("prefix")),
        messages=[dict_to_msg(d) for d in saved["messages"]],
        scratchpad=saved.get("scratchpad", {}),
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


# ---- legacy compatibility ---------------------------------------------------------------


def ckpt_key(run_id: str) -> str:
    """Legacy KV key used by the original `ctx.store`-backed checkpoint format.
    Kept for back-compat: the `StoreBackedCheckpointStore` bridge writes at this key so an
    older wiring that only injected `ctx.store` continues to round-trip cleanly."""
    return f"checkpoint:{run_id}"


class StoreBackedCheckpointStore:
    """A minimal `CheckpointPort` over a generic `StorePort`.

    Built ONLY as a legacy bridge inside `Agent`: when neither `Agent.checkpointer` nor
    `ctx.checkpointer` is wired but `ctx.store` exists, we synthesize one of these so the
    pre-existing `InMemoryStore`-driven tests keep working. New code wires a real
    `CheckpointPort` adapter (e.g. `InMemoryCheckpointStore`) directly.

    The bridge keeps a single-slot-per-run model — only the latest version survives on the
    store. Time-travel / history is NOT available through this adapter; callers that need
    versions list `Checkpointer.list_versions()`, which returns at most one element. This
    is intentional: a generic KV cannot losslessly model a version history without leaking
    a schema, and we don't want to retrofit that into `StorePort`.
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
        dict(d.get("scratchpad") or {}),
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
