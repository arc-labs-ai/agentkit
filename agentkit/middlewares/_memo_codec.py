"""JSON-shaped codec for a memoized chat result — the wire format `memoize()` stores.

`memoize()` on a chat chain stores an assembled `LLMResult`. `InMemoryStore` keeps the object
by reference so that worked; every DURABLE adapter serializes with `json.dumps`
(`adapters/store/file.py:95`, `redis.py:59`, `postgres.py:86`), so it did not. Measured on
`memoize()` over a two-call `FakeLLM("hi")` chat::

    InMemoryStore : ok
    FileStore     : TypeError: Object of type LLMResult is not JSON serializable

That is the FIRST cache WRITE, i.e. the production configuration failed before it ever had a
hit to be wrong about, while every test passed because the tests wire `InMemoryStore`.

This module is the same shape language as
`capabilities/checkpointer/persistence.py` (`result_to_dict` / `dict_to_result`), and
deliberately reuses its `tc_to_dict` / `dict_to_tc` so a `ToolCall` has ONE serialized shape in
this codebase rather than two that drift.

`usage_to_dict` is the one helper NOT reused, and the reason is a measured data loss:
`usage_to_dict(Usage(1, 2, 0.5, 7, 3))` returns `{'input': 1, 'output': 2, 'cost': 0.5}` — it
drops `cache_read_tokens` / `cache_write_tokens`. Those are load-bearing HERE in a way they are
not in a checkpoint: `meter`/`Budget` price a cached read differently from a fresh prompt, so a
cache hit that read back `cache_read_tokens=0` would misreport spend on every replay. This
codec carries all five fields (:func:`_usage_to_dict`).
"""

from __future__ import annotations

import math
from typing import Any, Final

from agentkit.capabilities.checkpointer.persistence import dict_to_tc, tc_to_dict
from agentkit.kernel.types import LLMResult, ToolCall, Usage

# Marker + version on the envelope. `decode_result` is a no-op on anything without the marker,
# which is what makes this safe to run over a store whose entries it did not write: an
# `InMemoryStore` holding a raw `LLMResult` from a pre-fix process comes back untouched, and a
# TOOL result — an opaque caller payload that may be any JSON dict — is never mistaken for an
# envelope. The version field is not read yet; it exists so a v2 shape can be told apart from a
# v1 entry instead of being guessed at from which keys are present.
_MARKER: Final = "__agentkit_memo__"
_VERSION: Final = 1

# Depth cap for the JSON-native walk below. `parsed` is caller data and may be cyclic;
# `json.dumps` itself raises `ValueError: Circular reference detected` on one. Rather than
# recurse until Python's own stack limit, anything deeper than this is declared not-safe, which
# costs at most a skipped cache entry. 20 is well past any realistic parsed schema.
_MAX_DEPTH: Final = 20


class NotJSONSafe:
    """A wrapper whose ONLY job is to make `json.dumps` refuse the value inside it.

    This is how the codec asks a question it cannot ask directly: *is the store behind this
    call durable?* `StorePort` has no capability flag, and adding one would only cover the
    adapters in this repo — a third-party `StorePort` would still get it wrong. So the store
    answers by behaving: an object store (`InMemoryStore`) keeps this wrapper by reference and
    `decode_result` unwraps it on the way back out, while a serializing store raises
    ``TypeError: Object of type NotJSONSafe is not JSON serializable`` on the write, which
    `memoize` catches and turns into "this call is not cacheable here" — a miss, which is
    always safe.

    Two classes of value get wrapped, and the second is the one that matters.

    1. Values `json.dumps` would reject anyway (a Pydantic model, a dataclass, an arbitrary
       object). Wrapping changes nothing for them; they were already going to raise.
    2. Values `json.dumps` would silently ACCEPT and hand back as something else. This is the
       real hazard, because it is the failure mode that reports success. Measured::

           json.dumps({"steps": ("a", "b")})  -> '{"steps": ["a", "b"]}'   round-trips != original
           json.dumps({1: "a"})               -> '{"1": "a"}'              int key becomes str
           json.dumps({"x": float("nan")})    -> '{"x": NaN}'              not valid JSON; Postgres
                                                                           jsonb rejects it outright

       A cache that stores a tuple and returns a list is exactly "stored something lossy that
       later reads back as a wrong answer". Refusing to store beats storing a near-miss.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"NotJSONSafe({type(self.value).__name__})"


def _json_native(value: Any, depth: int = 0) -> bool:
    """Is `value` composed ENTIRELY of types that survive `json.dumps` → `json.loads` unchanged?

    Strict on purpose: `isinstance(True, int)` is True in Python and `bool` round-trips fine, so
    both are accepted, but a `tuple`, a `set`, a `Decimal`, a non-`str` mapping key and a
    non-finite float are all rejected — see the measurements on :class:`NotJSONSafe`. `dict`
    subclasses (a `FrozenDict` sitting in a caller's parsed object) are accepted; they dump as
    the plain dict they subclass and come back as one.
    """
    if depth > _MAX_DEPTH:
        return False
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)  # NaN/Infinity are not JSON, whatever `json.dumps` emits
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _json_native(v, depth + 1) for k, v in value.items()
        )
    if isinstance(value, list):
        return all(_json_native(v, depth + 1) for v in value)
    return False


def _usage_to_dict(u: Usage) -> dict[str, Any]:
    """All FIVE `Usage` fields — see the module docstring for why this is not
    `persistence.usage_to_dict`."""
    return {
        "input": u.input_tokens,
        "output": u.output_tokens,
        "cost": u.cost_usd,
        "cache_read": u.cache_read_tokens,
        "cache_write": u.cache_write_tokens,
    }


def _usage_from_dict(d: Any) -> Usage:
    """Inverse of :func:`_usage_to_dict`. Tolerant of missing keys so an envelope written by a
    build that carried fewer fields rehydrates to zeros rather than a `KeyError`."""
    if not isinstance(d, dict):
        return Usage()
    return Usage(
        int(d.get("input", 0) or 0),
        int(d.get("output", 0) or 0),
        float(d.get("cost", 0.0) or 0.0),
        int(d.get("cache_read", 0) or 0),
        int(d.get("cache_write", 0) or 0),
    )


def encode_result(result: Any) -> Any:
    """`LLMResult` → a JSON-shaped envelope. Anything else is passed through untouched.

    Every field of the seven is carried. `parsed` is the only one that cannot be carried
    unconditionally: it is an arbitrary caller object — a Pydantic model, a dataclass, a plain
    dict — and no generic decoder can rebuild the first two from JSON without importing a class
    name out of a cache entry, which is a code-execution seam a cache has no business opening.

    So the rule is carry-or-refuse, never downgrade. A `parsed` that is natively JSON
    (`None`, dict/list/str/int/float/bool) is stored verbatim and reads back `==` to what the
    miss returned. Anything else is wrapped in :class:`NotJSONSafe`, which an object store keeps
    intact and a serializing store rejects — so on a durable store that call is simply not
    cached. It is never stored as `None`, because "a cache HIT returned `parsed=None`" is the
    exact bug class commit `5bb104a` fixed, and re-introducing it one layer down would be worse
    than not caching at all: an uncached call costs a provider round trip, a wrongly-cached one
    costs a wrong answer.
    """
    if not isinstance(result, LLMResult):
        return result
    parsed = result.parsed
    return {
        _MARKER: _VERSION,
        "content": result.content,
        "model": result.model,
        "provider": result.provider,
        "finish_reason": result.finish_reason,
        "usage": _usage_to_dict(result.usage),
        "tool_calls": [tc_to_dict(t) for t in result.tool_calls],
        "parsed": parsed if _json_native(parsed) else NotJSONSafe(parsed),
    }


def decode_result(value: Any) -> Any:
    """Envelope → `LLMResult`. Anything else — including a raw `LLMResult` — is returned as-is.

    `tool_calls` is rebuilt through `ToolCall`, so `arguments` comes back FROZEN
    (`ToolCall.__post_init__` deep-freezes it) rather than as the plain dict JSON handed us.
    That is not cosmetic: the same `ToolCall` flows into the ReAct approval snapshot, the
    idempotency-key hash and the audit trail, and a cache hit that yielded a mutable one would
    hand a replayed turn a weaker guarantee than the original. Measured end to end: a
    `ToolCall("c1", "search", {"q": "hi", "n": {"deep": 1}})` through `json.dumps`/`json.loads`
    and back into `ToolCall` is `==` the original, `arguments` and the nested `arguments["n"]`
    are both `FrozenDict`, and an in-place write raises.
    """
    if isinstance(value, LLMResult) or not isinstance(value, dict) or _MARKER not in value:
        return value
    parsed = value.get("parsed")
    if isinstance(parsed, NotJSONSafe):
        # Only reachable on an object store, which kept the wrapper by reference — the whole
        # point being that the caller's typed object survives a hit UNCHANGED there.
        parsed = parsed.value
    return LLMResult(
        content=value.get("content", "") or "",
        model=value.get("model"),
        provider=value.get("provider"),
        finish_reason=value.get("finish_reason"),
        usage=_usage_from_dict(value.get("usage")),
        tool_calls=tuple(_to_tool_call(t) for t in (value.get("tool_calls") or ())),
        parsed=parsed,
    )


def _to_tool_call(raw: Any) -> ToolCall:
    """A stored tool call → `ToolCall`. Tolerates an already-rebuilt one, which is what an
    object store hands back when it kept our envelope by reference and something upstream had
    already decoded it."""
    if isinstance(raw, ToolCall):
        return raw
    return dict_to_tc(raw)


__all__ = ["NotJSONSafe", "decode_result", "encode_result"]
