"""End-to-end streaming partial_parse — the in-flight typed-object pipe.

These tests pin the streaming contract end-to-end:

    * As text deltas arrive, the ``output_coerce`` middleware calls
      ``adapter.partial_parse(accumulated_so_far)`` and lifts the
      partial typed object onto the delta as ``Delta.partial``.
    * Subsequent deltas may carry the SAME partial structure — the
      middleware only stamps when the partial value CHANGED, sparing
      downstream consumers reprocessing identical partials.
    * The terminal delta carries the strict ``parsed`` value (the B3b
      contract) and ``LLMResult.parsed`` is the validated result.
    * The streamed partial type is the same as the parsed type —
      callers can render against ``partial`` without waiting for
      ``parsed``.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.capabilities.output_schema import adapt
from agentkit.kernel.types import ChatRequest, Delta, Message, Usage
from agentkit.middlewares import output_coerce
from agentkit.runtime import Invoker
from agentkit.testing import make_test_ctx

pydantic = pytest.importorskip("pydantic")


class PlannerOutput(pydantic.BaseModel):
    """The canonical streaming-output shape — a single list of strings
    so we can watch items accumulate item-by-item."""

    sub_questions: list[str]


class TokenScriptedLLM:
    """A minimal LLM seam that emits a SCRIPTED sequence of text
    deltas, one per ``stream()`` yield. Production FakeLLM chunks
    content into fixed-size pieces — fine for transport tests but
    too coarse for verifying per-delta partial lift. This tiny shim
    gives the test exact control over the per-delta cut points so
    we can prove ``partial`` advances in step with the bytes."""

    def __init__(self, chunks: list[str], *, model: str = "fake-stream") -> None:
        self._chunks = list(chunks)
        self._model = model

    async def stream(self, **_kw):
        for chunk in self._chunks:
            yield Delta(text=chunk, model=self._model, provider="fake")
        yield Delta(
            model=self._model,
            provider="fake",
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


def _collect_deltas(llm: TokenScriptedLLM, ctx, request: ChatRequest) -> list[Delta]:
    """Run an Invoker stream end-to-end and collect every delta the
    middleware chain produced. We need the raw delta list (not the
    assembled ``LLMResult``) to inspect per-delta ``partial`` lifting."""
    invoker = Invoker(llm=llm, chat_middleware=[output_coerce()])

    async def _drain():
        return [d async for d in invoker.stream(request, ctx)]

    return asyncio.run(_drain())


def test_streaming_partial_advances_per_delta() -> None:
    """As the JSON arrives token-by-token, ``Delta.partial`` carries a
    progressively-richer ``PlannerOutput`` — the consumer never has to
    wait for the terminal delta to render."""
    # Cut-points chosen so each chunk advances the partial visible to
    # the consumer: open object, first key+value (one item), then the
    # closing tokens.
    chunks = [
        '{"sub',
        '_questions": ["why is sky',
        ' blue?"',
        ', "why is grass green?"]',
        "}",
    ]
    llm = TokenScriptedLLM(chunks)
    ctx = make_test_ctx(llm=llm)
    ctx._output_adapter = adapt(PlannerOutput)  # type: ignore[attr-defined]
    req = ChatRequest(messages=[Message("user", "plan")], model="m")

    deltas = _collect_deltas(llm, ctx, req)

    # Collect every non-None partial we observed across the stream.
    partials = [d.partial for d in deltas if d.partial is not None]
    assert partials, "expected at least one delta carrying a partial"
    # Every partial is the typed object — not a dict or a stray string.
    for p in partials:
        assert isinstance(p, PlannerOutput)
    # The first useful partial appears once the second chunk has been
    # accumulated (the key+first-value arrived). It carries the
    # first sub-question — the consumer can render at this point.
    # Use ``model_dump(exclude_unset=True)`` because partials built via
    # ``model_construct`` leave un-streamed fields unset, and direct
    # attribute access on an unset field raises AttributeError.
    first_with_items = next(
        (p for p in partials if "sub_questions" in p.model_dump(exclude_unset=True)),
        None,
    )
    assert first_with_items is not None
    assert first_with_items.sub_questions == ["why is sky"]

    # By the end of the stream the partial reflects both items.
    last_partial = partials[-1]
    assert last_partial.sub_questions == [
        "why is sky blue?",
        "why is grass green?",
    ]


def test_terminal_delta_carries_strict_parsed() -> None:
    """The B3b contract: the terminal synthetic delta carries
    ``parsed`` — the strict-validated typed object. ``assemble_deltas``
    lifts it onto ``LLMResult.parsed``."""
    chunks = ['{"sub_questions": ["a", "b"]}']
    llm = TokenScriptedLLM(chunks)
    ctx = make_test_ctx(llm=llm)
    ctx._output_adapter = adapt(PlannerOutput)  # type: ignore[attr-defined]
    req = ChatRequest(messages=[Message("user", "plan")], model="m")

    deltas = _collect_deltas(llm, ctx, req)
    parsed_deltas = [d for d in deltas if d.parsed is not None]
    assert len(parsed_deltas) == 1, "exactly one synthetic terminal delta carries parsed"
    parsed = parsed_deltas[0].parsed
    assert isinstance(parsed, PlannerOutput)
    assert parsed.sub_questions == ["a", "b"]


def test_partial_is_not_emitted_when_unchanged() -> None:
    """Whitespace-only deltas don't change the parsed structure — the
    middleware withholds the duplicate partial so downstream consumers
    don't reprocess identical state. We script a stream where one chunk
    contributes only whitespace AFTER a partial was already lifted."""
    chunks = [
        '{"sub_questions": ["a"]',
        "  ",  # pure whitespace — parsed structure identical
        "}",
    ]
    llm = TokenScriptedLLM(chunks)
    ctx = make_test_ctx(llm=llm)
    ctx._output_adapter = adapt(PlannerOutput)  # type: ignore[attr-defined]
    req = ChatRequest(messages=[Message("user", "plan")], model="m")

    deltas = _collect_deltas(llm, ctx, req)
    partials_emitted = [d for d in deltas if d.partial is not None]
    # The middleware lifts each NEW partial onto the delta in place;
    # equality comparison on Pydantic models is field-by-field, so a
    # whitespace-only delta whose partial is identical to the prior
    # one MUST NOT have ``partial`` stamped.
    seen_values = [p.partial for p in partials_emitted]
    # No two consecutive partials are equal — dedupe is in force.
    for a, b in zip(seen_values, seen_values[1:], strict=False):
        assert a != b, f"middleware emitted duplicate partial: {a!r} == {b!r}"


def test_no_adapter_means_no_partial_lift() -> None:
    """Strict no-op fast-path: no adapter on the ctx → no partial
    lifting, deltas pass through unchanged."""
    chunks = ['{"sub_questions": ["a"]}']
    llm = TokenScriptedLLM(chunks)
    ctx = make_test_ctx(llm=llm)  # no _output_adapter
    req = ChatRequest(messages=[Message("user", "plan")], model="m")
    deltas = _collect_deltas(llm, ctx, req)
    assert all(d.partial is None for d in deltas)
    assert all(d.parsed is None for d in deltas)


def test_terminal_parsed_is_strict_partial_is_construct() -> None:
    """The terminal ``parsed`` came through ``adapter.parse`` (strict
    ``model_validate`` — would reject missing required fields). The
    in-flight ``partial`` came through ``model_construct`` (which
    permits unset required fields). The middleware MUST NOT confuse
    the two: with a stream that completes, both end up populated, but
    only ``parsed`` carries the strict validation guarantee."""
    chunks = ['{"sub_questions": ["only one"]}']
    llm = TokenScriptedLLM(chunks)
    ctx = make_test_ctx(llm=llm)
    ctx._output_adapter = adapt(PlannerOutput)  # type: ignore[attr-defined]
    req = ChatRequest(messages=[Message("user", "plan")], model="m")
    deltas = _collect_deltas(llm, ctx, req)
    parsed = next(d.parsed for d in deltas if d.parsed is not None)
    last_partial = [d.partial for d in deltas if d.partial is not None][-1]
    # Strict validate vs construct — both end up structurally equal
    # for a complete stream.
    assert parsed.sub_questions == last_partial.sub_questions == ["only one"]
