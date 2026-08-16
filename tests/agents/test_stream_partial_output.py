"""``StreamEvent.partial_output`` — streaming a typed object through ``Agent.stream`` alone.

The gap this closes: the framework could already *parse* a partial
structured output (``capabilities/output_schema/_partial_json.py``) and
*lift* it onto ``Delta.partial`` (``middlewares/output_coerce.py``), but
``StreamEvent`` had nowhere to put it and both cognitions dropped it. An
application that wanted the in-progress object had to bypass ``Agent``
entirely — reach into ``_resolve_request_builder()`` / ``_output_adapter``,
rebuild the ``ChatRequest`` by hand, and drive ``ctx.invoker.stream``.

The load-bearing assertion in this module is therefore a NEGATIVE one about
the caller: every test here goes through the public ``Agent.stream`` and
touches no private attribute.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.agents import Agent
from agentkit.kernel.types import Delta, ToolCall, Usage
from agentkit.middlewares import output_coerce
from agentkit.testing import FakeLLM, make_test_ctx

pydantic = pytest.importorskip("pydantic")


class Article(pydantic.BaseModel):
    """Two fields, one of them long — so the long field's value is still
    growing across many deltas while the short field is already final.
    That is the shape a streaming UI actually renders."""

    title: str
    body: str


# The JSON the model "writes", cut at points that land mid-string inside
# ``body`` — the tolerant parser must cope with an unterminated string.
_ARTICLE_JSON = '{"title": "Ships", "body": "the tide came in and then it went back out again"}'


class ScriptedLLM:
    """Emits an exact, caller-chosen sequence of text deltas.

    ``FakeLLM`` chunks at a fixed width, which is fine for transport tests
    but gives no control over WHERE the cuts fall. Partial-parse behaviour
    is entirely about the cut points, so this shim takes them literally.
    """

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)

    async def stream(self, **_kw):
        for chunk in self._chunks:
            yield Delta(text=chunk, model="scripted", provider="fake")
        yield Delta(
            model="scripted",
            provider="fake",
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


class _no_warnings:
    """Record every warning raised in the block, assert on it afterwards.

    ``pytest.warns(None)`` is an error on modern pytest, so "assert this
    specific warning did NOT fire" has to be spelled out by hand.
    """

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings(record=True)
        caught = self._cm.__enter__()
        warnings.simplefilter("always")
        return caught

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def _chunk(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


def _collect(agent: Agent, ctx) -> list:
    async def go():
        return [ev async for ev in agent.stream("write it", ctx)]

    return asyncio.run(go())


def _structured_ctx(chunks: list[str]):
    """A ctx whose chat chain includes ``output_coerce()`` — the wiring that
    makes partials flow. Without it the partial pipe is inert (and the Agent
    warns; see ``test_missing_output_coerce_warns_once``)."""
    return make_test_ctx(llm=ScriptedLLM(chunks), chat_middleware=[output_coerce()])


# ── the growing partial ──────────────────────────────────────────────────────


def test_successive_events_carry_a_growing_partial() -> None:
    """As bytes arrive, ``partial_output`` shows steadily more of the object.

    Two properties are asserted, and they are different claims:

    1. MONOTONIC — the set of fields that have arrived only ever grows.
       A partial that lost a field would mean the tolerant parser had
       re-interpreted earlier bytes, which would make a UI flicker.
    2. PREFIX-CONSISTENT — a string field's value at step N is a prefix of
       its value at step N+1. Strings grow character-by-character as the
       bytes land (an unterminated ``"Shi`` yields ``title="Shi"``), so this
       is what proves the partial tracks the stream rather than being
       re-guessed each time.
    """
    agent = Agent("writer", "m", output=Article)
    events = _collect(agent, _structured_ctx(_chunk(_ARTICLE_JSON, 12)))

    partials = [ev.partial_output for ev in events if ev.partial_output is not None]
    assert len(partials) >= 2, "expected the partial to be re-stamped as the object filled in"

    seen_fields: set[str] = set()
    last = {"title": "", "body": ""}
    for p in partials:
        assert isinstance(p, Article)
        fields = set(p.model_fields_set)
        assert seen_fields <= fields, f"a field disappeared: had {seen_fields}, now {fields}"
        seen_fields = fields
        for name in fields:
            value = getattr(p, name)
            assert value.startswith(last[name]), f"{name} is not prefix-consistent across partials"
            last[name] = value

    # The short field completes long before the long one — the whole point of
    # streaming a partial rather than waiting for the result. By the time
    # ``body`` first appears, ``title`` is already whole.
    first_with_body = next(p for p in partials if "body" in p.model_fields_set)
    assert first_with_body.title == "Ships"
    assert len(first_with_body.body) < len(partials[-1].body)


def test_last_partial_agrees_with_result_parsed() -> None:
    """The final partial and the strict ``result.parsed`` describe the same object.

    Note the assertion is on the last ``message_delta`` carrying a partial,
    NOT on the terminal ``final`` event. ``output_coerce`` emits ``parsed``
    on a synthetic post-terminal ``Delta`` that carries no text, so no
    partial is ever computed for it — the ``final`` event's own
    ``partial_output`` is legitimately ``None``.

    The agreement is asserted field-by-field over ``model_fields_set``
    rather than with ``==``. Whole-object equality only holds when the last
    text delta happens to close the JSON; the subset property holds for
    every possible cut point, so it is the invariant worth pinning.
    """
    agent = Agent("writer", "m", output=Article)
    events = _collect(agent, _structured_ctx(_chunk(_ARTICLE_JSON, 12)))

    last_partial = [ev.partial_output for ev in events if ev.partial_output is not None][-1]
    final = next(ev for ev in events if ev.type == "final")
    parsed = final.result.parsed

    assert isinstance(parsed, Article)
    for name in last_partial.model_fields_set:
        assert getattr(last_partial, name) == getattr(parsed, name), (
            f"partial disagrees with parsed on {name!r}"
        )
    # And the strict result is complete where the partial may not have been.
    assert parsed.title == "Ships" and parsed.body.endswith("again")


def test_final_event_itself_carries_no_partial() -> None:
    """Pins the asymmetry documented above, so nobody "fixes" it later:
    the terminal event's payload is ``result`` (with ``parsed``), never a
    partial."""
    agent = Agent("writer", "m", output=Article)
    events = _collect(agent, _structured_ctx(_chunk(_ARTICLE_JSON, 12)))
    final = next(ev for ev in events if ev.type == "final")
    assert final.partial_output is None
    assert final.result.parsed is not None


def test_partial_may_have_unset_required_fields() -> None:
    """The documented consumer contract: ``partial_output`` is built through
    the bypass-init path, so a required field can be genuinely ABSENT.

    A consumer that trusted attribute access would crash here. This test
    exists to prove the hazard is real, not theoretical — if a future change
    made partials always-complete, the framework would be silently
    fabricating field values and this test should fail loudly.
    """
    agent = Agent("writer", "m", output=Article)
    # Cut early enough that the first partial has ``title`` but not ``body``.
    events = _collect(agent, _structured_ctx(_chunk(_ARTICLE_JSON, 6)))
    first = next(ev.partial_output for ev in events if ev.partial_output is not None)

    assert "body" not in first.model_fields_set
    with pytest.raises(AttributeError):
        _ = first.body


# ── the negative: no output schema changes nothing ───────────────────────────


def test_no_output_schema_yields_partial_none_and_nothing_else_changes() -> None:
    """The additive guarantee. An agent with no schema must be bit-identical
    to its pre-change behaviour: same text, same event types, same final
    output — and ``partial_output`` uniformly ``None``."""
    agent = Agent("plain", "m")
    ctx = make_test_ctx(llm=FakeLLM("hello there, world"))
    events = _collect(agent, ctx)

    assert all(ev.partial_output is None for ev in events)
    assert [ev.type for ev in events] == ["message_delta"] * (len(events) - 1) + ["final"]
    assert "".join(ev.text for ev in events if ev.type == "message_delta") == "hello there, world"

    final = next(ev for ev in events if ev.type == "final")
    assert final.result.output == "hello there, world"
    assert final.result.parsed is None
    assert final.result.stop_reason == "complete"


def test_schema_declared_but_no_coerce_middleware_still_parses() -> None:
    """Without ``output_coerce()`` the partial pipe is inert, but the strict
    result is unaffected — the cognition runs ``agent.parse`` itself. This is
    the silent hole the warning exists to announce: nothing LOOKS broken."""
    agent = Agent("writer", "m", output=Article)
    ctx = make_test_ctx(llm=ScriptedLLM([_ARTICLE_JSON]))  # no output_coerce
    with pytest.warns(UserWarning, match="no output_coerce"):
        events = _collect(agent, ctx)

    assert all(ev.partial_output is None for ev in events)
    final = next(ev for ev in events if ev.type == "final")
    assert isinstance(final.result.parsed, Article)  # strict path unharmed


def test_missing_output_coerce_warns_once() -> None:
    """One warning per Agent instance, not one per call — a hot loop must not
    flood the log."""
    agent = Agent("writer", "m", output=Article)

    with pytest.warns(UserWarning, match="no output_coerce"):
        _collect(agent, make_test_ctx(llm=ScriptedLLM([_ARTICLE_JSON])))

    with _no_warnings() as caught:
        _collect(agent, make_test_ctx(llm=ScriptedLLM([_ARTICLE_JSON])))
    assert not [w for w in caught if "output_coerce" in str(w.message)]


def test_no_warning_when_no_output_schema() -> None:
    """An unstructured agent has nothing to coerce, so the warning must stay
    silent regardless of the chain."""
    agent = Agent("plain", "m")
    with _no_warnings() as caught:
        _collect(agent, make_test_ctx(llm=FakeLLM("hi")))
    assert not [w for w in caught if "output_coerce" in str(w.message)]


# ── the interaction bug: output_coerce must not defeat parse-and-repair ──────


def test_coercion_failure_reflects_and_repairs_with_middleware_wired() -> None:
    """Regression: ``output_coerce()`` in the chain used to ABORT the run on
    the first malformed response.

    The middleware strict-parses at end-of-stream and re-raises, which
    escaped past the cognition's reflect-and-retry branch — so the very
    wiring that enables streamed partials broke the repair loop that output
    schemas exist for. (This is why every pre-existing structured-output
    test omitted the middleware.) The cognitions now catch it and let
    ``agent.parse`` re-raise inside the repair loop.
    """
    calls = {"n": 0}

    class FlakyLLM:
        async def stream(self, **_kw):
            calls["n"] += 1
            text = "definitely not json" if calls["n"] == 1 else _ARTICLE_JSON
            yield Delta(text=text, model="flaky", provider="fake")
            yield Delta(finish_reason="stop", usage=Usage(1, 1), model="flaky", provider="fake")

    agent = Agent("writer", "m", output=Article, max_repairs=2)
    ctx = make_test_ctx(llm=FlakyLLM(), chat_middleware=[output_coerce()])
    final = next(ev for ev in _collect(agent, ctx) if ev.type == "final")

    assert calls["n"] == 2, "the model should have been re-prompted exactly once"
    assert isinstance(final.result.parsed, Article)
    assert final.result.stop_reason == "complete"
    assert final.result.partial is False


def test_repair_exhaustion_still_reports_invalid_output() -> None:
    """The repair budget still terminates. With the middleware wired and the
    model never recovering, the run ends as a recorded ``invalid_output``
    outcome — not an escaped exception."""

    class BadLLM:
        async def stream(self, **_kw):
            yield Delta(text="nope", model="bad", provider="fake")
            yield Delta(finish_reason="stop", usage=Usage(1, 1), model="bad", provider="fake")

    agent = Agent("writer", "m", output=Article, max_repairs=1)
    ctx = make_test_ctx(llm=BadLLM(), chat_middleware=[output_coerce()])
    final = next(ev for ev in _collect(agent, ctx) if ev.type == "final")

    assert final.result.stop_reason == "invalid_output"
    assert final.result.partial is True
    assert final.result.parsed is None
    assert final.result.evals["stop_reason"] == "invalid_output"


def test_streaming_a_typed_object_needs_no_private_attribute_access() -> None:
    """Brief 1's "done when", asserted as executable documentation.

    Everything below is public surface: construct an Agent, stream it,
    read ``ev.partial_output``. No ``_resolve_request_builder()``, no
    ``_output_adapter``, no hand-built ``ChatRequest``, no direct
    ``ctx.invoker.stream``.
    """
    agent = Agent("writer", "m", output=Article)
    ctx = _structured_ctx(_chunk(_ARTICLE_JSON, 10))

    async def render() -> list[str]:
        titles: list[str] = []
        async for ev in agent.stream("write it", ctx):
            p = ev.partial_output
            if p is not None and "title" in p.model_fields_set:
                titles.append(p.title)
        return titles

    assert asyncio.run(render())[0] == "Ships"


def test_a_tool_loop_with_an_output_schema_survives_the_coerce_middleware() -> None:
    """The ReAct half of the repair-interaction fix.

    A tool-calling turn's assistant text is a preamble ("Let me look that
    up…"), not the final JSON — so ``output_coerce``'s end-of-stream strict
    parse fails on EVERY tool turn. Uncaught, that aborted any tool-using
    agent with an ``output=`` schema the moment the middleware was wired,
    which is the wiring required for streamed partials in the first place.
    """
    from agentkit.agents.cognition import ReActCognition
    from agentkit.tools import tool

    @tool(side_effecting=False)
    def lookup(q: str) -> str:
        """Look up a fact about the given query string and return it."""
        return "42"

    class ToolThenJson:
        def __init__(self) -> None:
            self.n = 0

        async def stream(self, **_kw):
            self.n += 1
            if self.n == 1:
                yield Delta(text="Let me look that up.", model="m", provider="fake")
                yield Delta(
                    tool_calls=(ToolCall("c1", "lookup", {"q": "x"}),),
                    usage=Usage(1, 1, 0.0),
                    finish_reason="tool_calls",
                    model="m",
                    provider="fake",
                )
            else:
                yield Delta(text=_ARTICLE_JSON, model="m", provider="fake")
                yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="fake")

    llm = ToolThenJson()
    agent = Agent("x", "m", output=Article, cognition=ReActCognition(tools=[lookup]))
    ctx = make_test_ctx(llm=llm, chat_middleware=[output_coerce()])

    events = _collect(agent, ctx)
    final = next(ev for ev in events if ev.type == "final")

    assert llm.n == 2, "the loop should have run the tool then answered"
    assert final.result.stop_reason == "complete"
    assert isinstance(final.result.parsed, Article)
    # And partials still flowed on the answering turn.
    assert any(ev.partial_output is not None for ev in events)
