"""`_result_to_stream` must re-express an assembled result WITHOUT losing a field.

`_result_to_stream(call, result)` is the way back INTO the single streaming contract: a memoize hit
(and miss — the producer collects `next` to store it, then re-streams), a `BaseMiddleware` with
`buffers=True` re-emitting a transformed result, and `on_error` recovery all hand it a finished
`LLMResult` and expect the stream it returns to reassemble into that same result.

It is a hand-written field-by-field copy across a type boundary, so it drifts silently. It did:
six of `LLMResult`'s seven fields were copied and `parsed` — the typed object `output_coerce()`
exists to produce — was dropped. Measured on a `memoize()`d chat with `output=Plan` declared::

    original LLMResult.parsed    : Plan(subject='ship', steps=['a', 'b'])
    Delta rebuilt for the stream : parsed=None
    reassembled LLMResult.parsed : None

Nothing raises; only the one call site that reads `.parsed` sees it, and it sees `None` instead of
the typed object.

The load-bearing test here is not the single-field assertion — it is
`test_every_llmresult_field_survives_the_round_trip`, a DRIFT RATCHET. It enumerates
`dataclasses.fields(LLMResult)` rather than a hand-written list, so adding an eighth field upstream
and forgetting this function fails HERE, at the copy, instead of at some consumer months later.
`test_every_delta_field_reaches_llmresult` is the same ratchet pointed the other way, across
`assemble_deltas`.
"""

from __future__ import annotations

import asyncio
from dataclasses import Field, fields
from typing import Any

from agentkit.kernel.middleware import (
    BaseMiddleware,
    Call,
    MiddlewareContext,
    _result_to_stream,
    collect,
)
from agentkit.kernel.middleware import chain as compose
from agentkit.kernel.types import Delta, LLMResult, ToolCall, Usage, assemble_deltas


def _chat_call() -> Call:
    """`_result_to_stream` reads only `call.kind`; request/ctx are irrelevant to the copy."""
    return Call("chat", request=None, ctx=None)


# ── the reported bug ───────────────────────────────────────────────────────────────────────


def test_parsed_survives_the_rebuild() -> None:
    """The reported drop, at the seam that dropped it."""
    parsed = {"subject": "ship", "steps": ["a", "b"]}  # any typed object; identity is what matters
    original = LLMResult(content='{"subject": "ship"}', parsed=parsed)

    (delta,) = _result_to_stream(_chat_call(), original)

    assert delta.parsed is parsed, "the replayed terminal Delta lost the typed object"
    assert assemble_deltas([delta]).parsed is parsed, "…and so did the reassembled LLMResult"


def test_the_replayed_delta_is_not_marked_partial() -> None:
    """`Delta.partial` staying unset is CORRECT — this test exists so nobody "fixes" it.

    `partial` means "in-progress, tolerantly parsed, required fields may be unset". A replayed
    TERMINAL delta is the opposite: the result is already complete and `parsed` is its strict
    answer. `assemble_deltas` also never lifts `partial` onto `LLMResult`, so a `partial` stamped
    here would be write-only. See `assemble_deltas`' docstring in `kernel/types.py`."""
    (delta,) = _result_to_stream(_chat_call(), LLMResult(content="x", parsed={"a": 1}))
    assert delta.partial is None


# ── the drift ratchets ─────────────────────────────────────────────────────────────────────


def _sentinel(f: Field[Any], seen: dict[str, Any]) -> Any:
    """A distinguishable non-default value for one field, chosen by its annotation.

    Deliberately table-driven with NO generic fallback: a field of an unrecognised type fails the
    test with instructions rather than being quietly skipped, because "skipped" is exactly how the
    `parsed` drop survived. `types.py` uses `from __future__ import annotations`, so `f.type` is the
    annotation SOURCE TEXT, not the runtime type."""
    ann = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    if ann in ("str", "str | None"):
        return f"<{f.name}>"
    if ann in ("Usage", "Usage | None"):
        return Usage(7, 11, 0.13)
    if ann == "tuple[ToolCall, ...]":
        return (ToolCall("call-1", "weather", {"city": "SF"}),)
    if ann == "Any":
        return seen.setdefault(f.name, object())  # unique per field, identity-comparable
    raise AssertionError(
        f"{f.name}: {f.type!r} has no sentinel. A new field type reached this round-trip test — "
        "teach _sentinel() how to build a non-default value for it, then make sure the converters "
        "under test actually carry the field."
    )


def _all_fields_set(cls: type) -> Any:
    """Instantiate `cls` with every field holding a distinct non-default sentinel."""
    seen: dict[str, Any] = {}
    return cls(**{f.name: _sentinel(f, seen) for f in fields(cls)})


def test_every_llmresult_field_survives_the_round_trip() -> None:
    """RATCHET: `LLMResult → _result_to_stream → assemble_deltas → LLMResult` is lossless.

    Fails automatically when `LLMResult` gains a field that `_result_to_stream` does not copy — the
    eighth-field case. If such a field genuinely cannot ride a `Delta` (a per-delta concept, say),
    the honest fix is to give it an entry in a documented exception list here with the reason,
    exactly as `Delta.partial` has one below — not to delete the assertion."""
    original = _all_fields_set(LLMResult)

    items = _result_to_stream(_chat_call(), original)
    rebuilt = assemble_deltas(items)

    dropped = [f.name for f in fields(LLMResult) if getattr(rebuilt, f.name) != getattr(original, f.name)]
    assert not dropped, (
        f"_result_to_stream lost {dropped} on the way back into the stream. Every LLMResult field "
        "must be copied onto the replayed Delta — a memoize hit, a buffered BaseMiddleware and an "
        "on_error recovery all reassemble the result from exactly these items."
    )


# `Delta` fields `assemble_deltas` deliberately does NOT lift onto `LLMResult`, with the reason.
# Anything else missing is drift, not design.
_DELTA_FIELDS_NOT_ASSEMBLED = {
    "partial": (
        "per-delta, tolerantly-parsed, in-progress object with possibly-unset required fields. "
        "Lifting it would give LLMResult two competing typed fields where the tolerant one could "
        "shadow the strict `parsed`. In-flight partials reach consumers via "
        "StreamEvent.partial_output instead."
    ),
}


def test_every_delta_field_reaches_llmresult() -> None:
    """RATCHET, the other direction: `assemble_deltas` carries every `Delta` field or documents why.

    `text` lands on `content` (the concatenation target); every other shared name maps 1:1."""
    delta = _all_fields_set(Delta)
    result = assemble_deltas([delta])

    result_field_names = {f.name for f in fields(LLMResult)}
    lost: list[str] = []
    for f in fields(Delta):
        if f.name in _DELTA_FIELDS_NOT_ASSEMBLED:
            assert f.name not in result_field_names, (
                f"Delta.{f.name} is on the deliberate-drop list but LLMResult now HAS that field — "
                "the exception list is stale; decide whether it should be carried."
            )
            continue
        target = "content" if f.name == "text" else f.name
        assert target in result_field_names, f"Delta.{f.name} maps to no LLMResult field"
        if getattr(result, target) != getattr(delta, f.name):
            lost.append(f.name)
    assert not lost, f"assemble_deltas dropped {lost}"


# ── positive controls (these pass BEFORE and AFTER the fix) ────────────────────────────────


def test_a_chat_result_without_parsed_still_round_trips() -> None:
    """The unstructured case — the overwhelming majority of calls — is untouched."""
    original = LLMResult(
        content="hello",
        model="m",
        provider="fake",
        finish_reason="stop",
        usage=Usage(10, 5, 0.0001),
        tool_calls=(ToolCall("c1", "weather", {"city": "SF"}),),
    )

    rebuilt = assemble_deltas(_result_to_stream(_chat_call(), original))

    assert rebuilt == original
    assert rebuilt.parsed is None


def test_a_tool_result_is_passed_through_verbatim() -> None:
    """Non-chat kinds are opaque: the one-item stream is the value itself, never a Delta."""
    payload = {"sent": True, "n": 1}
    assert _result_to_stream(Call("tool", request=None, ctx=None), payload) == [payload]


def test_a_none_result_is_passed_through_verbatim() -> None:
    """`None` is not a chat result to rebuild — an empty inner stream assembles to it, and the
    caller must see `None`, not a `Delta` full of defaults."""
    assert _result_to_stream(_chat_call(), None) == [None]
    assert _result_to_stream(Call("tool", request=None, ctx=None), None) == [None]


# ── end-to-end through the chain the kernel itself owns ────────────────────────────────────


class _Buffering(BaseMiddleware):
    """`buffers=True` collects the inner stream and RE-EMITS the result via `_result_to_stream` —
    the second live caller of the broken copy, alongside memoize. It transforms nothing here; a
    pass-through middleware must be invisible in the result."""

    buffers = True

    async def on_response(self, ctx: MiddlewareContext, result: Any) -> Any:
        return result


def test_a_buffering_middleware_does_not_eat_the_typed_output() -> None:
    """END-TO-END inside the kernel: identical chains that differ ONLY in `buffers` must produce
    identical results. Before the fix the buffered one returned `parsed=None` while the streaming
    one returned the typed object — the same bug memoize surfaces, reachable without any store."""
    parsed = {"subject": "ship"}

    async def terminal(call: Call) -> Any:
        yield Delta(text="{}", model="m", provider="fake")
        yield Delta(finish_reason="stop", usage=Usage(1, 1, 0.0), parsed=parsed)

    async def run(buffers: bool) -> LLMResult:
        mw = _Buffering()
        mw.buffers = buffers
        handler = compose([mw], terminal)
        result: LLMResult = await collect(handler(_chat_call()), kind="chat")
        return result

    buffered, streamed = asyncio.run(run(True)), asyncio.run(run(False))

    assert streamed.parsed is parsed  # passes before AND after: pass-through never rebuilt
    assert buffered.parsed is parsed, "a buffering middleware dropped the typed output"
    assert buffered == streamed  # and nothing else drifted either
