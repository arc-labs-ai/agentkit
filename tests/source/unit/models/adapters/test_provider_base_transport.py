"""The ONE transport every frontier provider sits on (`providers/frontiers/base.py`).

Two terminals — `_post` (one-shot JSON) and `_stream_events` (server-sent events) — and the
non-incremental `stream()` fallback that lets a provider with no SSE support satisfy the streaming
primitive anyway. Each one decides something a caller can never see from the outside:

* whether a failure RAISES at all, or ends the stream as a complete-looking answer;
* which of the TWO readers of that failure gets the facts it needs — `resilience.classify`, which
  reads the MESSAGE as substrings, and `contracts.failures.classify`, which json-decodes `body` and
  reads the STRUCTURED `error.type`/`error.code`. `frontiers/calls.py` builds its `Turn.failure`
  from the second one, so a fact that reaches only the message reaches only half the framework;
* whether the request that goes on the wire is the one the caller asked for.

`test_provider_base_client.py` covers the owned/injected client lifecycle, and
`test_provider_error_classification.py` covers the 4xx/5xx map as the SUBSTRING reader sees it.
This file covers the parts neither touches: the fallback stream, the structured reader, the SSE
framing cases that only appear when a server skips an event boundary, and the transport-level
failures that arrive with no status at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.core.contracts.types import LLMResult, Message, ProviderUsage, ToolCall
from agentkit.core.execution.resilience import ErrorClass
from agentkit.core.execution.resilience import classify as resilience_classify
from agentkit.models.contracts.constants import FailureClass
from agentkit.models.contracts.failures import classify as failure_classify
from agentkit.models.providers.frontiers import base
from agentkit.models.providers.frontiers.base import (
    HttpLLM,
    ProviderAuthError,
    ProviderError,
    _aclose_quietly,
    _classify_4xx,
    _error_type,
    raise_if_error_frame,
)

# `contracts.failures.classify` marks exactly these four retryable. Every other member of the
# 15-class taxonomy means "stop" — so the class a transport error lands in IS the retry decision.
_RETRYABLE = (
    FailureClass.RATE_LIMITED,
    FailureClass.OVERLOADED,
    FailureClass.TIMEOUT,
    FailureClass.TRANSIENT_NETWORK,
)


class _Probe(HttpLLM):
    """The smallest conforming subclass — `_headers()` is the only piece a provider must fill."""

    def _headers(self) -> dict[str, str]:
        return {"authorization": "Bearer k"}


class _NonStreamingLLM(_Probe):
    """A provider that implements `chat()` and NOTHING else — the case `HttpLLM.stream` exists for."""

    def __init__(self, outcome: Any, **kw: Any) -> None:
        super().__init__(api_key="k", base_url="http://x", **kw)
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


@contextlib.asynccontextmanager
async def _transport(handler: Any, **kw: Any) -> AsyncIterator[_Probe]:
    """A `_Probe` on a MockTransport client. The client is INJECTED, so the transport never owns or
    replaces it and the context manager's close is the only one — no unclosed-pool ResourceWarning."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        yield _Probe(api_key="k", base_url="http://x", client=client, **kw)


def _ok_response(terminal: str) -> httpx.Response:
    """A 200 each terminal can actually read: JSON for `_post`, an SSE frame for `_stream_events`."""
    if terminal == "post":
        return httpx.Response(200, json={"ok": True})
    return httpx.Response(
        200,
        content=b'data: {"ok": true}\n\ndata: [DONE]\n\n',
        headers={"content-type": "text/event-stream"},
    )


async def _drive(llm: HttpLLM, terminal: str, payload: dict[str, Any] | None = None) -> Any:
    """Run one terminal to completion. Both must behave identically on every failure path, so every
    failure test here is parametrised over the two rather than written against whichever came first."""
    body = {"model": "m"} if payload is None else payload
    if terminal == "post":
        return await llm._post("/v1/x", body)
    return [event async for event in llm._stream_events("/v1/x", body)]


# ── the non-incremental `stream()` fallback ─────────────────────────────────


@pytest.mark.asyncio
async def test_a_provider_without_sse_support_still_yields_one_complete_terminal_delta() -> None:
    """`stream()` is the ONE streaming primitive: every caller — `collect`, the cognition loop, the
    streaming middleware chain — iterates it, and a provider that only speaks one-shot JSON must
    still satisfy it. The fallback wraps `chat()`'s finished result as a single terminal delta, so
    every field the caller would have read off the `LLMResult` has to survive the wrap. A field
    dropped here is a field that exists on `chat()` and vanishes on `stream()` — one seam, two
    answers, decided by which method the caller happened to use.
    """
    result = LLMResult(
        content="the answer",
        model="m-1",
        provider="probe",
        finish_reason="stop",
        usage=ProviderUsage(input_tokens=11, output_tokens=3, cost_usd=0.5),
        tool_calls=(ToolCall("call_0", "search", {"q": "x"}),),
        parsed={"answer": 42},
    )
    llm = _NonStreamingLLM(result)

    deltas = [d async for d in llm.stream(messages=[Message("user", "q")], model="m-1")]

    assert len(deltas) == 1  # one TERMINAL delta, not a terminal plus an empty one
    (delta,) = deltas
    assert delta.text == "the answer"
    assert delta.finish_reason == "stop"
    assert delta.model == "m-1" and delta.provider == "probe"
    assert delta.usage == result.usage
    assert delta.tool_calls == result.tool_calls
    assert delta.parsed == {"answer": 42}
    # `partial` means "an in-progress partial parse", which a delta rebuilt from a COMPLETE result
    # is not; setting it would make the streaming coerce middleware re-emit a finished answer.
    assert delta.partial is None


@pytest.mark.asyncio
async def test_the_streaming_fallback_forwards_every_call_parameter_to_chat() -> None:
    """The fallback is a pass-through, and a pass-through that quietly drops an argument is the
    worst kind: `tools=None` means "no tool use available", `response_format=None` means "prose
    instead of JSON", `max_tokens=None` means "the provider's default". Each is a silently
    DIFFERENT request that still returns a plausible answer, so nothing downstream can notice.
    """
    llm = _NonStreamingLLM(LLMResult(content="ok"))
    messages = [Message("user", "q")]
    tools = [{"name": "search"}]
    response_format = {"type": "json_object"}

    async for _ in llm.stream(
        messages=messages,
        model="m-1",
        tools=tools,
        response_format=response_format,
        temperature=0.7,
        max_tokens=256,
        cache_hint="aggressive",
    ):
        pass

    assert llm.calls == [
        {
            "messages": messages,
            "model": "m-1",
            "tools": tools,
            "response_format": response_format,
            "temperature": 0.7,
            "max_tokens": 256,
            "cache_hint": "aggressive",
        }
    ]


@pytest.mark.asyncio
async def test_a_chat_failure_surfaces_out_of_the_streaming_fallback() -> None:
    """The failure mode this whole module is built against: a stream that ENDS instead of raising
    hands the agent a half-answer it treats as the model's final word. The fallback awaits `chat()`
    before yielding anything, so a provider failure must come out of the first `__anext__` — not as
    an empty one-item stream, and not swallowed because the generator body never ran.
    """
    llm = _NonStreamingLLM(ProviderError("provider transient error (status 503; overloaded): busy"))

    with pytest.raises(ProviderError) as exc:
        [d async for d in llm.stream(messages=[Message("user", "q")], model="m-1")]

    assert resilience_classify(exc.value) is ErrorClass.TRANSIENT  # still retryable through the fallback
    assert llm.calls  # the request WAS attempted; the stream did not fail open


# ── in-band SSE error frames: the shapes with no `error` object to read ─────


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "error"},  # the envelope alone — Anthropic's minimal error event
        {"type": "error", "error": "Overloaded"},  # `error` as prose, not an object
        {"type": "error", "error": None},
        {"type": "error", "error": ["overloaded"]},
    ],
)
def test_an_error_envelope_with_no_error_object_still_raises(frame: dict[str, Any]) -> None:
    """A provider that marks the frame `type: "error"` has told us the stream failed, and that is
    enough — the `error` payload is a DETAIL, not the signal. If the missing or non-dict payload
    made this function return (or raise AttributeError reaching into a string), the frame would
    fall through the translator's `elif` chain exactly as it did before the function existed: loop
    ends normally, partial text, `finish_reason=None`, nothing for `retry()` to see.
    """
    with pytest.raises(ProviderError) as exc:
        raise_if_error_frame(frame)

    assert "mid-stream" in str(exc.value)  # it still says what happened
    assert exc.value.body is not None
    assert json.loads(exc.value.body) == frame  # the raw frame is kept for whoever can read it


def test_a_frame_that_is_not_an_error_passes_through_untouched() -> None:
    """POSITIVE CONTROL for the guard above. Every ordinary delta goes through this function, so a
    broadened error test that started raising on `{"usage": ...}` or a `content_block_delta` would
    turn every successful stream into a failure."""
    for frame in (
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "message_stop"},
        {"choices": [{"delta": {"content": "hi"}}]},
        {"error": None},  # no `type: error` marker and no error object → not an error
    ):
        assert raise_if_error_frame(frame) is None


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        ("overloaded_error", FailureClass.OVERLOADED),
        ("timeout_error", FailureClass.TIMEOUT),
        ("rate_limit_error", FailureClass.RATE_LIMITED),
        ("api_error", FailureClass.TRANSIENT_NETWORK),
        ("authentication_error", FailureClass.AUTH),
        ("invalid_request_error", FailureClass.INVALID_REQUEST),
    ],
)
def test_an_in_band_error_frame_classifies_through_its_body_not_its_status(
    error_type: str, expected: FailureClass
) -> None:
    """An in-band frame arrives inside a 200, so it carries NO status — and `failures.classify`
    maps most conditions BY status. Its only other key is the structured `error.type`, which lives
    in the frame body. That is why the frame travels on `ProviderError(body=...)` as well as in the
    message: the message feeds the substring reader, the body feeds this one, and they are two
    different readers with two different jobs. `overloaded_error` and `timeout_error` are precisely
    what providers report this way, and both are retryable only if this reader sees them.
    """
    frame = {"type": "error", "error": {"type": error_type, "message": "provider said so"}}

    with pytest.raises(ProviderError) as exc:
        raise_if_error_frame(frame)

    failure = failure_classify(exc.value)
    assert failure is not None
    assert failure.classification is expected
    assert failure.retryable is (expected in _RETRYABLE)


def test_an_error_frame_stripped_of_its_body_is_no_longer_classifiable() -> None:
    """NEGATIVE CONTROL, and the measurement that justifies the `body=` argument. The same frame
    carried only in the message classifies UNKNOWN and non-retryable — an overload the resilience
    layer is told to give up on. Delete `body=` from `raise_if_error_frame` and this is what every
    in-band failure degrades to, with the message still reading perfectly well to a human.
    """
    frame = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    with pytest.raises(ProviderError) as exc:
        raise_if_error_frame(frame)

    bodyless = ProviderError(str(exc.value))
    blind = failure_classify(bodyless)
    assert blind is not None and blind.classification is FailureClass.UNKNOWN and blind.retryable is False


# ── the structured reader's view of an HTTP failure ─────────────────────────


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (429, {"type": "rate_limit_error", "message": "slow down"}, FailureClass.RATE_LIMITED),
        (503, {"type": "overloaded_error", "message": "busy"}, FailureClass.OVERLOADED),
        (500, {"type": "api_error", "message": "boom"}, FailureClass.TRANSIENT_NETWORK),
        (502, {"type": "api_error", "message": "bad gateway"}, FailureClass.TRANSIENT_NETWORK),
        (401, {"type": "authentication_error", "message": "bad key"}, FailureClass.AUTH),
        (403, {"type": "permission_error", "message": "forbidden"}, FailureClass.AUTH),
        (400, {"type": "invalid_request_error", "message": "nope"}, FailureClass.INVALID_REQUEST),
        (404, {"type": "not_found_error", "message": "no such model"}, FailureClass.INVALID_REQUEST),
        (402, {"type": "billing_error", "message": "pay up"}, FailureClass.QUOTA_EXHAUSTED),
        (
            429,
            {"type": "insufficient_quota", "code": "insufficient_quota", "message": "no credit"},
            FailureClass.QUOTA_EXHAUSTED,
        ),
    ],
)
def test_a_classified_http_failure_keeps_the_facts_the_structured_reader_needs(
    status: int, error: dict[str, Any], expected: FailureClass
) -> None:
    """`_classify_4xx` composes its MESSAGE for the substring reader and deliberately withholds
    anything that would make a permanent failure read transient — which is exactly why the
    machine-readable facts must survive elsewhere. `frontiers/calls.py` builds `Turn.failure` from
    `failures.classify`, and that reader uses only `status_code` and `body`. A branch that forgot
    either (or truncated the body past its JSON, as `_error_type` once was) leaves this reader with
    UNKNOWN/non-retryable for a rate limit, or retryable for a reached spend cap.

    The retryable flag is asserted from the class rather than written per row: exactly four of the
    taxonomy's fifteen classes are retryable, so the class a failure lands in IS the retry decision.
    """
    body = json.dumps({"error": error})
    exc = _classify_4xx(status, body)

    assert isinstance(exc, ProviderError)
    assert exc.status_code == status
    assert exc.body == body  # whole, parseable, untouched by the message's truncation
    failure = failure_classify(exc)
    assert failure is not None
    assert failure.classification is expected
    assert failure.retryable is (expected in _RETRYABLE)


@pytest.mark.parametrize("body", ["[]", "null", "42", '"rate limited"', '["error"]', "true"])
def test_a_body_that_is_valid_json_but_not_an_object_has_no_error_type(body: str) -> None:
    """`_error_type` reaches into `parsed["error"]["type"]`, and a provider (or a proxy in front of
    one) can answer with a JSON array, string or literal. Only a dict has a `.get`, so without the
    shape check this raises AttributeError from inside the error path — a transport bug thrown
    while handling a provider error, which loses the provider error entirely.

    Returning None means "no structured type", so the response falls through to the STATUS map:
    a shapeless 429 body stays a retryable rate limit, a shapeless 400 stays permanent.
    """
    assert _error_type(body) is None

    transient = _classify_4xx(429, body)
    assert not isinstance(transient, ProviderAuthError)
    assert resilience_classify(transient) is ErrorClass.TRANSIENT
    assert resilience_classify(_classify_4xx(400, body)) is ErrorClass.PERMANENT


@pytest.mark.parametrize("terminal", ["post", "stream"])
@pytest.mark.asyncio
async def test_both_terminals_attach_the_response_headers_to_the_failure(terminal: str) -> None:
    """`Retry-After` is the provider TELLING us how long to wait, and it lives in the headers —
    nowhere else. `failures.classify` reads it off `exc.headers` into `Failure.retry_after_s`, which
    is what a scheduler waits on; without the graft the field is None and the retry layer falls back
    to its own guess, hammering a provider that already said when to come back. Both terminals do
    the graft because a caller does not choose a terminal based on how errors are reported.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"retry-after": "7", "x-request-id": "abc123"},
        )

    async with _transport(handler) as llm:
        with pytest.raises(ProviderError) as exc:
            await _drive(llm, terminal)

    assert exc.value.headers["retry-after"] == "7"
    assert exc.value.headers["x-request-id"] == "abc123"  # the whole header set, for the operator
    failure = failure_classify(exc.value)
    assert failure is not None
    assert failure.classification is FailureClass.RATE_LIMITED
    assert failure.retry_after_s == 7.0


@pytest.mark.parametrize("terminal", ["post", "stream"])
@pytest.mark.asyncio
async def test_both_terminals_raise_a_non_provider_classification_untouched(
    terminal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_classify_4xx`'s contract is "return the exception to RAISE", not "return a ProviderError" —
    the transport raises whatever comes back. The header graft is therefore guarded: `headers` is
    part of `ProviderError`'s documented surface, and grafting it onto an unrelated exception would
    invent an attribute `failures.classify` then reads as authoritative-but-empty, overwriting
    nothing and explaining nothing. Guarded, the classifier stays free to return any exception type
    and the transport never rewrites or re-wraps it.
    """
    monkeypatch.setattr(base, "_classify_4xx", lambda status, body: RuntimeError(f"unmapped status {status}"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(418, text="teapot")

    async with _transport(handler) as llm:
        with pytest.raises(RuntimeError) as exc:
            await _drive(llm, terminal)

    assert not isinstance(exc.value, ProviderError)  # not re-wrapped on the way out
    assert str(exc.value) == "unmapped status 418"
    assert not hasattr(exc.value, "headers")


# ── SSE framing: the cases that only appear when a server skips a boundary ──


@pytest.mark.asyncio
async def test_a_data_line_with_no_space_after_the_colon_is_not_mangled() -> None:
    """The SSE spec strips EXACTLY ONE optional leading space from a `data:` value. A server that
    writes `data:{...}` with no space is spec-legal and common from hand-rolled proxies; stripping a
    character unconditionally would eat the opening brace and leave `{"a": 1}` as `"a": 1}` — not
    valid JSON, so the frame is dropped by `except JSONDecodeError` and the answer silently
    truncates. The surrounding file pins the one-space case; this is the other side of the branch.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data:{"seq": 1}\n\ndata:  {"seq": 2}\n\ndata:[DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with _transport(handler) as llm:
        events = await _drive(llm, "stream")

    assert events == [{"seq": 1}, {"seq": 2}]  # no-space and two-space both arrive intact


@pytest.mark.asyncio
async def test_done_ends_the_stream_even_when_the_server_omits_the_event_boundary() -> None:
    """`[DONE]` is the provider saying "that was the whole answer". Servers that omit the blank line
    between events are common enough that this parser handles them — which means `[DONE]` can be
    discovered while FLUSHING the buffer rather than at an event boundary, and it must terminate
    there too. If that flush only recognised events and not the sentinel, `[DONE]` would be
    accumulated as a payload prefix and every frame a chatty proxy appends after it — keep-alives,
    a repeated terminal chunk — would be reassembled into the model's answer.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(b'data: {"seq": 1}\ndata: [DONE]\ndata: {"seq": 2}\n'),
            headers={"content-type": "text/event-stream"},
        )

    async with _transport(handler) as llm:
        events = await _drive(llm, "stream")

    assert events == [{"seq": 1}]  # everything before DONE, nothing after it


# ── transport failures: no status, no body, and two readers to satisfy ──────


@pytest.mark.parametrize("terminal", ["post", "stream"])
@pytest.mark.asyncio
async def test_a_connect_failure_before_any_event_reads_transient_to_the_retry_layer(terminal: str) -> None:
    """A refused connection is the most retryable failure there is and the one with the least to go
    on: no status, no body, nothing but an httpx exception type. The message is the only channel
    left, so it is WORDED for the substring reader — "network/timeout" is what makes
    `resilience.classify` say TRANSIENT and `retry()` re-invoke. Both terminals use the same wording
    because a fallover must not depend on whether the caller streamed.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _transport(handler) as llm:
        with pytest.raises(ProviderError) as exc:
            await _drive(llm, terminal)

    assert resilience_classify(exc.value) is ErrorClass.TRANSIENT
    assert exc.value.status_code is None  # there was no response to take one from
    # The original httpx error is kept as the cause: it is the only place the transport-level fact
    # ("ConnectError", not "ReadTimeout") still exists after the wrap.
    assert isinstance(exc.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
async def test_a_read_failure_after_the_first_event_raises_rather_than_truncating() -> None:
    """The dangerous half of a stream failure: the socket dies with tokens already delivered, so
    the caller HAS a plausible answer and the generator could simply stop. It must not — a stream
    that ends quietly mid-sentence is indistinguishable from a finished one, and `finish_reason`
    stays None where nobody looks. The deltas already yielded are kept (they were real) and the
    failure follows them.
    """

    async def body() -> AsyncIterator[bytes]:
        yield b'data: {"seq": 1}\n\n'
        raise httpx.ReadError("socket closed mid-answer")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    seen: list[dict[str, Any]] = []
    async with _transport(handler) as llm:
        with pytest.raises(ProviderError) as exc:
            async for event in llm._stream_events("/v1/x", {"model": "m"}):
                seen.append(event)

    assert seen == [{"seq": 1}]  # what arrived is not thrown away
    assert resilience_classify(exc.value) is ErrorClass.TRANSIENT
    assert isinstance(exc.value.__cause__, httpx.ReadError)


# ── the request hook: one rewrite, honoured by both terminals ──────────────


@pytest.mark.parametrize("terminal", ["post", "stream"])
@pytest.mark.asyncio
async def test_the_request_hook_rewrites_the_payload_on_both_terminals(terminal: str) -> None:
    """The hook is the only seam a host has for a field agentkit does not model — a routing
    preference, a provider beta flag, a tenant tag. It is applied at BOTH terminals because a hook
    honoured on `chat()` and ignored on `stream()` sends two different requests from one
    configuration, and the difference shows up only as a provider-side behaviour change nobody can
    attribute. Its RETURN VALUE is what ships, so a hook may build a new dict instead of mutating
    the caller's — and the caller's own payload must come back unchanged.
    """
    sent: list[dict[str, Any]] = []
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _ok_response(terminal)

    def hook(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        return {**payload, "metadata": {"tenant": "acme"}}

    payload = {"model": "m"}
    async with _transport(handler, request_hook=hook) as llm:
        await _drive(llm, terminal, payload)

    assert sent == [{"model": "m", "metadata": {"tenant": "acme"}}]
    assert len(calls) == 1  # once per request, not once per retry of the same payload
    assert payload == {"model": "m"}  # the hook's return shipped; the caller's dict was not touched


@pytest.mark.parametrize("terminal", ["post", "stream"])
@pytest.mark.asyncio
async def test_without_a_hook_the_payload_goes_on_the_wire_verbatim(terminal: str) -> None:
    """POSITIVE CONTROL. `request_hook` defaults to None and the overwhelmingly common case is no
    hook at all — a rewrite that leaked into that path (a normalising pass, an injected default)
    would change every request every provider makes, which is the one thing this transport must
    never do on its own initiative."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _ok_response(terminal)

    async with _transport(handler) as llm:
        await _drive(llm, terminal, {"model": "m", "temperature": 0.0})

    assert sent == [{"model": "m", "temperature": 0.0}]


# ── releasing a client whose event loop is already gone ────────────────────


class _Unclosable:
    """A client whose pool primitives belong to a loop that has already been closed, so teardown
    raises. This is the NORMAL case for an orphan, not an exotic one."""

    def __init__(self) -> None:
        self.attempts = 0

    async def aclose(self) -> None:
        self.attempts += 1
        raise RuntimeError("Event loop is closed")


@pytest.mark.asyncio
async def test_closing_an_orphan_client_swallows_a_teardown_failure() -> None:
    """The orphan close runs as a fire-and-forget task on the NEW loop, so an exception escaping it
    is never awaited by anyone: it surfaces as a loop-level "exception was never retrieved" report
    attached to no request, long after the call that triggered it. Swallowing is deliberate and
    documented — a best-effort close that may raise is still strictly better than the measured leak
    of dropping the reference outright, and the attempt itself is the part that matters.
    """
    client = _Unclosable()

    assert await _aclose_quietly(client) is None
    assert client.attempts == 1  # best effort means it was actually tried


def test_an_orphan_that_cannot_be_closed_does_not_break_the_next_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scenario end to end: one provider instance driven from two `asyncio.run(...)` calls, with
    the first client's loop gone by the time the close runs. The second call must get a working
    client — a teardown failure for a connection pool nobody can reach any more is not a reason to
    fail the request that triggered the cleanup.
    """
    made: list[_Unclosable] = []

    def fake_make_client(timeout: float) -> _Unclosable:
        made.append(_Unclosable())
        return made[-1]

    monkeypatch.setattr(base, "_make_client", fake_make_client)
    llm = _Probe(api_key="k", base_url="http://x")

    async def touch() -> Any:
        client = llm._http()
        await asyncio.sleep(0)  # let the scheduled orphan close run (and fail)
        return client

    first = asyncio.run(touch())
    second = asyncio.run(touch())

    assert first is not second and made == [first, second]
    assert first.attempts == 1  # the orphan's release was attempted
    assert second.attempts == 0  # the live client was left alone
