"""`ApiAdapter` — what reaches the wire, what never does, and what a half-finished stream means.

WHY THIS FILE EXISTS
--------------------
`ApiAdapter` is the single place where a composed prompt, a `ModelSettings` and a tool registry
become an actual HTTPS request body, and almost every decision it encodes is invisible from the
outside. A caller that asks for `effort="high"` gets an answer either way: the difference between
`{"thinking": {"type": "adaptive"}}` and silence is a quality and a bill, not an error. A caller
that attaches an image to a resumed conversation gets an answer either way: the difference between
upgrading the user turn already in history and appending a second one is whether the model is asked
the same question twice. Nothing raises in either case, which is exactly why these have to be
pinned on the wire rather than through the answer.

The same asymmetry runs through the refusals. `turn` has three ways to decline before it opens a
socket — CLI workspace features, home-file credentials, a missing key — and each one is a decision
about what must NOT be attempted: a workspace request dispatched to the Responses API would silently
drop the filesystem authority the caller asked for, and a keyless request would spend a round trip
to be told what the adapter already knew.

Two parsers close the file. `_openai` and `_anthropic` read a server-sent stream whose terminal
frame is the only evidence the provider finished, so their hostile case is the stream that simply
STOPS: no error, no terminal event, a body that just ends. That is a real network, and the usage
already counted on the way through is a real bill.

HARNESS
-------
Payload decisions go through `httpx.MockTransport` (as the neighbouring provider tests do), because
the request body must survive `json.dumps` — a `StrEnum` setting that serialized as a repr, or a
`FrozenDict` schema that did not serialize at all, is a bug a dict-level assertion cannot see. The
two stream parsers take the transport by duck type instead: they touch nothing but
`_stream_events`, so `_Stream` is the honest seam, and it also records exactly which frames were
pulled — which is how "stops reading at the error frame" is observable at all.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import pytest

from agentkit.agents.prompts.composition import ImagePart, inline_prompt
from agentkit.core.contracts.types import Mode, ToolSchema
from agentkit.models.contracts.constants import EventKind, FailureClass
from agentkit.models.contracts.types import (
    Binding,
    CallLimits,
    Credential,
    CredentialRef,
    ModelSettings,
    SessionHandle,
    Turn,
    Workspace,
)

httpx = pytest.importorskip("httpx")

from agentkit.models.providers.frontiers.calls import ApiAdapter

_OPENAI_CREDENTIAL = CredentialRef(lambda: {"OPENAI_API_KEY": "sk-test"})
_ANTHROPIC_CREDENTIAL = CredentialRef(lambda: {"ANTHROPIC_API_KEY": "sk-ant-test"})
_LIMITS = CallLimits(max_output_tokens=64)
_PNG = ImagePart("image/png", b"\x89PNG\r\n\x1a\nfixture")

_OPENAI_DONE = (
    {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    },
)
_ANTHROPIC_DONE = (
    {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
    {"type": "message_stop"},
)


class _Wire:
    """A `MockTransport` client plus every request the adapter put through it."""

    def __init__(self, events: tuple[dict[str, Any], ...]) -> None:
        self._events = events
        self.payloads: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.urls: list[str] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._respond))

    def _respond(self, request: Any) -> Any:
        self.payloads.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        self.urls.append(str(request.url))
        body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in self._events)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    @property
    def payload(self) -> dict[str, Any]:
        assert len(self.payloads) == 1, f"expected exactly one request, saw {len(self.payloads)}"
        return self.payloads[0]


@asynccontextmanager
async def _wire(events: tuple[dict[str, Any], ...] = ()) -> AsyncIterator[_Wire]:
    wire = _Wire(events)
    try:
        yield wire
    finally:
        await wire.client.aclose()


async def _turn(
    adapter: ApiAdapter,
    *,
    mode: Mode = Mode.OPENAI_API,
    prompt: Any = None,
    settings: Any = None,
    credential: Any = None,
    schema: Any = None,
    tools: Any = None,
    session: Any = None,
    emit: Any = None,
    native_agents: Any = None,
    workspace: Any = None,
) -> Turn:
    """Drive one turn, defaulting everything the test is not talking about."""
    if credential is None:
        credential = _ANTHROPIC_CREDENTIAL if mode == Mode.ANTHROPIC_API else _OPENAI_CREDENTIAL
    return await adapter.turn(
        Binding(mode, "test", credential),
        prompt if prompt is not None else inline_prompt("question"),
        settings=settings if settings is not None else ModelSettings(),
        limits=_LIMITS,
        schema=schema,
        tools=tools,
        session=session,
        cancel=None,
        emit=emit if emit is not None else (lambda _event: None),
        native_agents=native_agents,
        workspace=workspace,
    )


class _Tools:
    """The only thing `turn` asks a tool registry for is `schemas()`."""

    def schemas(self) -> tuple[ToolSchema, ...]:
        return (ToolSchema("lookup", "Look a fact up", {"type": "object", "properties": {"q": {"type": "string"}}}),)


class _Stream:
    """A duck-typed transport: the two parsers touch nothing but `_stream_events`.

    `yielded` records which frames the parser actually pulled, so a parser that
    stops early is distinguishable from one that drains the stream and then
    reports the same thing.
    """

    def __init__(self, *events: dict[str, Any], then: BaseException | None = None) -> None:
        self._events = events
        self._then = then
        self.yielded: list[str] = []

    async def _stream_events(self, _path: Any, _payload: Any) -> AsyncIterator[dict[str, Any]]:
        for event in self._events:
            self.yielded.append(event["type"])
            yield event
        if self._then is not None:
            raise self._then


# ── refusals that must happen before a socket is opened ─────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"native_agents": {"reviewer": object()}},
        {"workspace": Workspace(working_dir="/tmp/fixture")},
        {"native_agents": {"reviewer": object()}, "workspace": Workspace()},
    ],
)
async def test_cli_workspace_features_are_refused_instead_of_quietly_dropped(extra: dict[str, Any]) -> None:
    """Native sub-agents and a workspace are filesystem and tool AUTHORITY. The Responses and
    Messages APIs have nowhere to put either, so dispatching the call anyway would answer the
    caller's question while silently ignoring the sandbox they scoped it to — the one failure mode
    a security-relevant parameter must never have. The refusal also has to be free: it names the
    reason and never reaches the transport."""
    async with _wire(_OPENAI_DONE) as wire:
        turn = await _turn(ApiAdapter(client=wire.client), **extra)

    assert turn.failure is not None
    assert turn.failure.classification == FailureClass.INVALID_REQUEST
    assert "CLI workspace features" in turn.failure.mode_detail["reason"]
    assert wire.payloads == [], "a refused call must not spend a round trip"


@pytest.mark.asyncio
async def test_home_file_credentials_are_refused_loudly_even_when_an_env_key_is_also_present() -> None:
    """Home-file credentials are an installation: the CLI adapters write them into a HOME the
    subprocess then reads. An API mode cannot install anything, so a `Credential` carrying files is
    a caller who believes a credential source is in play that is not. Ignoring the files and using
    the env key alongside them would authenticate as somebody else and look like success, so this
    raises rather than returning a `Failure` — a misconfiguration, not a transport fault."""
    credential = CredentialRef(
        lambda: Credential(env={"OPENAI_API_KEY": "sk-test"}, home={".credentials.json": b"{}"})
    )
    async with _wire(_OPENAI_DONE) as wire:
        with pytest.raises(ValueError, match="environment values"):
            await _turn(ApiAdapter(client=wire.client), credential=credential)

    assert wire.payloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,environment",
    [
        (Mode.OPENAI_API, {}),
        (Mode.OPENAI_API, {"OPENAI_API_KEY": ""}),
        (Mode.OPENAI_API, {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        (Mode.ANTHROPIC_API, {}),
        (Mode.ANTHROPIC_API, {"ANTHROPIC_API_KEY": ""}),
        (Mode.ANTHROPIC_API, {"OPENAI_API_KEY": "sk-test"}),
    ],
)
async def test_a_missing_key_is_an_auth_failure_decided_without_asking_the_provider(
    mode: Mode, environment: dict[str, str]
) -> None:
    """The adapter knows it has no key, so sending the request to find out is a round trip spent
    learning nothing — and, against a provider that rate-limits unauthenticated traffic, a round
    trip that costs the NEXT call. The cases that matter are the quiet ones: an env var set to the
    empty string looks configured, and the other vendor's key looks like a key. Neither is one, and
    classifying this as AUTH (rather than, say, INVALID_REQUEST) is what routes the host to its
    credential resolver."""
    async with _wire(_OPENAI_DONE) as wire:
        turn = await _turn(ApiAdapter(client=wire.client), mode=mode, credential=CredentialRef(lambda: environment))

    assert turn.failure is not None and turn.failure.classification == FailureClass.AUTH
    assert wire.payloads == [], "a keyless call must not reach the provider"


# ── the OpenAI Responses payload ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unconfigured_openai_turn_sends_nothing_it_was_not_asked_to_send() -> None:
    """POSITIVE CONTROL for every optional block below. `reasoning` and `text` are assembled
    incrementally and only attached when non-empty; an always-attached `{}` would be a different
    request for every caller who configured nothing — and on the Responses API an empty
    `reasoning` object is not a no-op, it is a value. The required keys are listed exactly so that
    a new one cannot be introduced without a reader noticing: `store=False` and the explicit
    `include` are the zero-retention posture that lets encrypted reasoning survive continuation
    without the provider keeping the conversation."""
    async with _wire(_OPENAI_DONE) as wire:
        turn = await _turn(ApiAdapter(client=wire.client))

    assert turn.failure is None
    assert sorted(wire.payload) == [
        "include",
        "input",
        "max_output_tokens",
        "model",
        "prompt_cache_key",
        "store",
        "stream",
    ]
    assert wire.payload["store"] is False
    assert wire.payload["include"] == ["reasoning.encrypted_content"]
    assert wire.payload["stream"] is True


@pytest.mark.asyncio
async def test_top_k_is_never_forwarded_to_the_responses_api() -> None:
    """The Responses API has no `top_k`, and an unknown key there is a 400 for the whole call —
    so a loop that merely copied every non-None setting would turn a harmless sampling preference
    into a hard failure for every turn. `temperature` and `top_p` are named explicitly for exactly
    this reason, and the assertion is the absence."""
    settings = ModelSettings(temperature=0.25, top_p=0.9, top_k=40)
    async with _wire(_OPENAI_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), settings=settings)

    assert wire.payload["temperature"] == 0.25
    assert wire.payload["top_p"] == 0.9
    assert "top_k" not in wire.payload, "the Responses API would reject the whole request"


@pytest.mark.asyncio
async def test_effort_and_a_reasoning_summary_travel_in_one_reasoning_object() -> None:
    """Two independent settings share one wire key, so whichever is written second must merge
    rather than replace. `effort` buys thinking tokens the caller pays for and `summary` is the
    only way those tokens become visible, so losing either is a silent downgrade: the answer still
    arrives, just shallower or opaque."""
    settings = ModelSettings(effort="high", reasoning_summary="detailed")
    async with _wire(_OPENAI_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), settings=settings)

    assert wire.payload[EventKind.REASONING] == {"effort": "high", "summary": "detailed"}


@pytest.mark.asyncio
async def test_verbosity_and_a_json_schema_share_the_text_object() -> None:
    """Same merge, different key — and here the two halves come from different arguments
    (`settings` and `schema`), which is the version of this bug that survives review. A schema that
    overwrote `verbosity` would still produce valid JSON, so nothing downstream would notice. The
    schema goes out `strict`: the caller asked for a shape, and a non-strict format makes it a
    suggestion."""
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "additionalProperties": False}
    async with _wire(_OPENAI_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), settings=ModelSettings(verbosity="low"), schema=schema)

    assert wire.payload["text"]["verbosity"] == "low"
    assert wire.payload["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "schema": schema,
        "strict": True,
    }


@pytest.mark.asyncio
async def test_a_service_tier_is_forwarded_under_the_name_the_api_uses() -> None:
    """`tier` is priced differently per tier, so dropping it is a billing difference the caller
    cannot see in the answer — `flex` silently served at the standard rate is money, and `priority`
    silently served as `flex` is latency. The rename (`tier` → `service_tier`) is the part that
    rots: a passthrough of `settings` by key would send `tier` and be ignored."""
    async with _wire(_OPENAI_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), settings=ModelSettings(tier="flex"))

    assert wire.payload["service_tier"] == "flex"


@pytest.mark.asyncio
async def test_openai_tool_schemas_are_advertised_non_strict() -> None:
    """`strict: False` is deliberate and is the opposite of the output schema's `strict: True`
    above. Strict function schemas require every property to be required and
    `additionalProperties: false` throughout, which the tool schemas agentkit accepts do not
    guarantee — under strict mode the provider refuses to CALL the tool at all, so a tool loop
    would stall with no error anyone could attribute to this flag."""
    async with _wire(_OPENAI_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), tools=_Tools())

    assert wire.payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look a fact up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            "strict": False,
        }
    ]


# ── the Anthropic Messages payload ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unconfigured_anthropic_turn_sends_nothing_it_was_not_asked_to_send() -> None:
    """The mirror control. `output_config` is attached only when an effort or a schema put
    something in it, and `max_tokens` is REQUIRED by the Messages API — a request without it is a
    400, which is why the cap falls back through settings, limits and a floor rather than being
    conditional."""
    async with _wire(_ANTHROPIC_DONE) as wire:
        turn = await _turn(ApiAdapter(client=wire.client), mode=Mode.ANTHROPIC_API)

    assert turn.failure is None
    assert sorted(wire.payload) == ["max_tokens", "messages", "model", "stream"]
    assert wire.payload["max_tokens"] == 64


@pytest.mark.asyncio
async def test_top_k_is_forwarded_to_the_messages_api() -> None:
    """The deliberate asymmetry with the Responses API: Anthropic DOES take `top_k`, so dropping
    it here would be an unrequested change to the sampling distribution. Pinned alongside its
    opposite so the two lists cannot be "unified" into one shared loop without a failure."""
    settings = ModelSettings(temperature=0.25, top_p=0.9, top_k=40)
    async with _wire(_ANTHROPIC_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), mode=Mode.ANTHROPIC_API, settings=settings)

    assert (wire.payload["temperature"], wire.payload["top_p"], wire.payload["top_k"]) == (0.25, 0.9, 40)


@pytest.mark.asyncio
async def test_effort_turns_on_adaptive_thinking_and_not_only_an_output_config() -> None:
    """Asking for effort on the Messages API is two coupled edits: the budget in `output_config`
    and the `thinking` block that makes the model actually use it. Sending the first alone is the
    dangerous half-fix — the request is accepted, the caller is billed for the model they chose,
    and no extended thinking happens."""
    async with _wire(_ANTHROPIC_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), mode=Mode.ANTHROPIC_API, settings=ModelSettings(effort="high"))

    assert wire.payload["output_config"]["effort"] == "high"
    assert wire.payload["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_a_system_prompt_is_cached_with_no_ttl_key_when_no_ttl_was_requested() -> None:
    """The stable prefix is marked `ephemeral` ALWAYS — caching the system prompt is free and the
    cache is what makes a multi-turn agent affordable. The `ttl` sub-key is the part that is
    conditional, and it has to be absent rather than null: a `ttl` the caller never asked for is a
    different (and differently priced) cache entry, and `"ttl": None` is a 400."""
    prompt = replace(inline_prompt("question"), system="Stable research rules")
    async with _wire(_ANTHROPIC_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), mode=Mode.ANTHROPIC_API, prompt=prompt)

    system = wire.payload["system"]
    assert system == [
        {"type": "text", "text": "Stable research rules", "cache_control": {"type": "ephemeral"}}
    ], "no ttl was requested, so no ttl key"


@pytest.mark.asyncio
async def test_anthropic_tool_schemas_are_advertised_under_input_schema() -> None:
    """Same three facts as the OpenAI tool payload, in a different shape and under a different
    key. A tool whose `input_schema` is missing is advertised to the model as taking no arguments,
    so it gets called with `{}` and the tool fails on its own validation — a failure that reads as
    the model's mistake."""
    async with _wire(_ANTHROPIC_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), mode=Mode.ANTHROPIC_API, tools=_Tools())

    assert wire.payload["tools"] == [
        {
            "name": "lookup",
            "description": "Look a fact up",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


# ── images on a resumed conversation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_images_upgrade_the_resumed_user_turn_instead_of_asking_twice() -> None:
    """A resumed session already carries the user's question as plain text (the runtime appends it
    on resume), so attaching images has to REPLACE that turn's content rather than append a second
    user turn. Appending would send the question twice — the model sees a repeat and the caller
    pays input tokens for it, and on the Messages API two consecutive user turns is also a shape
    the provider may reject outright.

    The base64 payload is the other half: Anthropic wants the bytes alone, so shipping the whole
    `data:` URL is a silently corrupt image the model cannot decode."""
    session = SessionHandle(Mode.ANTHROPIC_API, "resumed", history=({"role": "user", "content": "question"},))
    prompt = replace(inline_prompt("question"), images=(_PNG,))
    async with _wire(_ANTHROPIC_DONE) as wire:
        turn = await _turn(ApiAdapter(client=wire.client), mode=Mode.ANTHROPIC_API, prompt=prompt, session=session)

    messages = wire.payload["messages"]
    assert len(messages) == 1, "the question must not be asked twice"
    content = messages[0]["content"]
    assert [block["type"] for block in content] == ["text", "image"]
    assert content[0]["text"] == "question"
    assert content[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(_PNG.data).decode("ascii"),
    }
    assert turn.history[0]["content"] == content, "the upgraded turn is what the next call resumes from"


@pytest.mark.asyncio
async def test_openai_images_upgrade_the_resumed_user_turn_with_the_data_url_form() -> None:
    """Same decision, opposite encoding. The Responses API takes the full `data:` URL in
    `image_url`, where Anthropic takes the bare base64 — so the two paths cannot share one block
    builder, and a copy from the other branch produces a request the provider accepts and an image
    it cannot read."""
    session = SessionHandle(Mode.OPENAI_API, "resumed", history=({"role": "user", "content": "question"},))
    prompt = replace(inline_prompt("question"), images=(_PNG,))
    async with _wire(_OPENAI_DONE) as wire:
        await _turn(ApiAdapter(client=wire.client), prompt=prompt, session=session)

    history = wire.payload["input"]
    assert len(history) == 1, "the question must not be asked twice"
    content = history[0]["content"]
    assert [block["type"] for block in content] == ["input_text", "input_image"]
    assert content[1]["image_url"] == _PNG.data_url()
    assert content[1]["image_url"].startswith("data:image/png;base64,")


# ── the turn's own error boundary and client ownership ──────────────────────


@pytest.mark.asyncio
async def test_a_malformed_function_call_becomes_a_failure_and_still_returns_the_request_history() -> None:
    """The response parsers decode provider output OUTSIDE their own try block, so a frame that
    violates the vendor's documented shape — here a `function_call` with no `call_id` — raises a
    bare `KeyError` out of the parser. `turn`'s boundary is what stops that from reaching the
    caller's event loop as an unhandled exception mid-agent-run, which would take down the whole
    run for one bad frame instead of failing one attempt.

    The history still has to come back: it is the record of what was actually asked, and the
    runtime needs it to resume or report the attempt."""
    malformed = (
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{"type": "function_call", "name": "lookup", "arguments": "{}"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )
    async with _wire(malformed) as wire:
        turn = await _turn(ApiAdapter(client=wire.client))

    assert turn.failure is not None
    assert turn.messages == (), "a failure must not look like an empty answer"
    assert turn.history == ({"role": "user", "content": "question"},)


@pytest.mark.asyncio
async def test_an_injected_client_survives_the_turn_for_the_next_one_to_reuse() -> None:
    """The adapter must not close what it does not own. The runtime injects one client and drives
    many turns through it — an agent loop is dozens of calls — so closing it in the `finally` would
    make turn two fail on a closed client, and would also throw away the connection pool and the
    TLS session that make the second call fast."""
    async with _wire(_OPENAI_DONE) as wire:
        adapter = ApiAdapter(client=wire.client)
        first = await _turn(adapter)
        second = await _turn(adapter)

        assert first.failure is None and second.failure is None
        assert len(wire.payloads) == 2
        assert not wire.client.is_closed, "the injector owns the lifecycle"


@pytest.mark.asyncio
async def test_a_client_the_adapter_built_itself_is_closed_when_the_turn_ends(monkeypatch: Any) -> None:
    """The other half of the same rule. An adapter constructed with no client builds one per turn,
    and the `HttpLLM` holding it is local to `turn` — so nothing else can ever close it. Leaking
    one per call means leaking sockets and file descriptors for the life of the process, which
    surfaces much later as a fan-out that cannot open connections."""
    import agentkit.models.providers.frontiers.base as base

    built: list[Any] = []

    async with _wire(_OPENAI_DONE) as wire:

        def _factory(_timeout: float) -> Any:
            built.append(wire.client)
            return wire.client

        monkeypatch.setattr(base, "_make_client", _factory)
        turn = await _turn(ApiAdapter())

        assert turn.failure is None and turn.messages == ("ok",)
        assert built, "the adapter never built a client of its own"
        assert wire.client.is_closed, "an owned client must be closed by the turn that built it"


# ── the OpenAI stream parser ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_reasoning_summary_delta_is_emitted_as_reasoning_and_never_as_message_text() -> None:
    """Summary deltas and answer deltas arrive on the same stream and must not be concatenated
    into one visible string: a UI that rendered them together would show the model's private
    deliberation to the user, and a transcript that captured it would persist it. The kind is also
    what lets a host display reasoning differently (or not at all)."""
    stream = _Stream(
        {"type": "response.reasoning_summary_text.delta", "delta": "weighing the options"},
        {"type": "response.output_text.delta", "delta": "ok"},
        *_OPENAI_DONE,
    )
    events: list[Any] = []

    turn = await ApiAdapter()._openai(stream, {}, events.append)

    reasoning = [event for event in events if event.kind == EventKind.REASONING]
    deltas = [event for event in events if event.kind == EventKind.MESSAGE_DELTA]
    assert turn.failure is None
    assert [event.fields["summary"] for event in reasoning] == ["weighing the options"]
    assert [event.fields["text"] for event in deltas] == ["ok"]
    assert "weighing" not in json.dumps([dict(event.fields) for event in deltas])


@pytest.mark.asyncio
async def test_an_in_band_openai_error_frame_ends_the_turn_without_draining_the_stream() -> None:
    """An `error` frame arrives mid-stream with no HTTP status to classify from, so the frame's own
    `type` is the only evidence — and it has to be read, or a rate limit becomes an UNKNOWN that
    the runtime will not retry. Returning immediately matters too: the provider may keep the
    response open, and continuing to read it holds a connection open waiting for a terminal frame
    that is never coming."""
    stream = _Stream(
        {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        {"type": "response.completed", "response": {"status": "completed", "output": []}},
    )

    turn = await ApiAdapter()._openai(stream, {}, lambda _event: None)

    assert turn.failure is not None
    assert turn.failure.classification == FailureClass.RATE_LIMITED
    assert turn.failure.retryable is True
    assert stream.yielded == ["error"], "the parser kept reading after a terminal error"


@pytest.mark.asyncio
async def test_openai_cancellation_keeps_the_usage_seen_so_far_and_marks_it_incomplete() -> None:
    """A cancelled turn is still billed for the tokens the provider already produced, so dropping
    the usage understates the spend of every cancelled call — and a budget built from it drifts
    further out with every cancellation. `usage_complete=False` is the other half: the numbers are
    real but not final, so the ledger records them without treating them as the turn's total.
    `CancelledError` is also re-raised nowhere here — it becomes a classified `Failure`, because
    the runtime decides what a cancellation means for the call."""
    stream = _Stream(
        {
            "type": "response.incomplete",
            "response": {"status": "incomplete", "output": [], "usage": {"input_tokens": 42, "output_tokens": 7}},
        },
        then=asyncio.CancelledError(),
    )

    turn = await ApiAdapter()._openai(stream, {}, lambda _event: None)

    assert turn.failure is not None and turn.failure.classification == FailureClass.CANCELLED
    assert turn.raw_usage == {"input_tokens": 42, "output_tokens": 7}
    assert turn.usage_complete is False


@pytest.mark.asyncio
async def test_an_openai_stream_that_ends_with_no_terminal_frame_is_a_failure_not_an_empty_answer() -> None:
    """The hostile case: the body just ends. No error, no `response.completed`, no exception — a
    dropped connection looks exactly like a model that chose to say nothing. Without the terminal
    frame there is no usage, no status and no output list, so treating it as success would report a
    blank answer with zero cost as a finished turn, and a validator would then blame the model.

    Only the CLASSIFICATION is pinned here. Whether this failure is marked retryable is NOT
    asserted, because the adapter currently constructs it non-retryable while every
    `transient_network` produced by `classify` is retryable — reported separately rather than
    frozen in place here."""
    stream = _Stream({"type": "response.output_text.delta", "delta": "partial"})

    turn = await ApiAdapter()._openai(stream, {}, lambda _event: None)

    assert turn.messages == ()
    assert turn.failure is not None
    assert turn.failure.classification == FailureClass.TRANSIENT_NETWORK


@pytest.mark.asyncio
async def test_an_unknown_openai_event_kind_is_reported_as_transport_and_otherwise_ignored() -> None:
    """The Responses stream has upwards of thirty event types and gains more; only four are acted
    on. The rest must flow through to the transport event — which is the host's only window onto
    what the provider is doing — without being mistaken for text, for an error or for the end of
    the stream. A parser that raised on an unrecognised kind would fail turns the day the vendor
    ships a new frame."""
    stream = _Stream(
        {"type": "response.created", "response": {}},
        {"type": "response.output_item.added", "item": {"type": "message"}},
        {"type": "response.output_text.delta", "delta": "ok"},
        *_OPENAI_DONE,
    )
    events: list[Any] = []

    turn = await ApiAdapter()._openai(stream, {}, events.append)

    assert turn.failure is None and turn.messages == ("ok",)
    assert [event.fields["mode_event"] for event in events if event.kind == EventKind.TRANSPORT] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.completed",
    ]
    assert [event.fields["text"] for event in events if event.kind == EventKind.MESSAGE_DELTA] == ["ok"]


@pytest.mark.asyncio
async def test_an_openai_output_item_that_is_neither_message_nor_call_survives_in_history_only() -> None:
    """Encrypted reasoning items are the reason this file's module docstring says history is kept
    verbatim: they are opaque, they are what makes a signature survive into the next turn, and
    they carry no text and no call. Folding them into the answer would emit ciphertext as the
    model's reply; dropping them from history would break continuation on the following turn,
    which is the failure that shows up somewhere else entirely."""
    reasoning_item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}
    stream = _Stream(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    reasoning_item,
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    )

    turn = await ApiAdapter()._openai(stream, {}, lambda _event: None)

    assert turn.messages == ("ok",), "an opaque item must not become visible text"
    assert turn.tool_calls == ()
    assert turn.history[0] == reasoning_item


@pytest.mark.asyncio
async def test_openai_function_call_items_become_tool_calls_with_decoded_arguments() -> None:
    """Arguments arrive as a JSON STRING, and the rest of the framework (masking, policy checks,
    the tool's own signature) works on a mapping — so a forgotten decode hands every tool a single
    positional string instead of its parameters. `call_id`, not `id`, is the field a result is
    keyed by on the way back, and an absent `arguments` means no arguments, not the string
    `"null"`."""
    stream = _Stream(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {"type": "function_call", "call_id": "fc_1", "name": "lookup", "arguments": '{"q": "a"}'},
                    {"type": "function_call", "call_id": "fc_2", "name": "lookup", "arguments": ""},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    )

    turn = await ApiAdapter()._openai(stream, {}, lambda _event: None)

    assert turn.tool_calls == (
        {"id": "fc_1", "name": "lookup", "arguments": {"q": "a"}},
        {"id": "fc_2", "name": "lookup", "arguments": {}},
    )
    assert turn.history == tuple(stream._events[0]["response"]["output"]), "vendor items survive verbatim"


# ── the Anthropic stream parser ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_anthropic_error_frame_keeps_the_input_tokens_already_counted() -> None:
    """`message_start` carries the input tokens — the prompt was read and WILL be billed, whatever
    happens next. An error frame after it is the case where a naive parser returns a bare failure
    and the cost of the attempt disappears: retried three times, that is three invisible prompt
    charges on a call the ledger believes cost nothing."""
    stream = _Stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 99}}},
        {"type": "error", "error": {"type": "overloaded_error"}},
        {"type": "message_stop"},
    )

    turn = await ApiAdapter()._anthropic(stream, {}, lambda _event: None)

    assert turn.failure is not None and turn.failure.classification == FailureClass.OVERLOADED
    assert turn.raw_usage == {"input_tokens": 99}
    assert stream.yielded == ["message_start", "error"], "the parser kept reading after a terminal error"


@pytest.mark.asyncio
async def test_an_unknown_anthropic_event_kind_is_ignored_rather_than_ending_the_turn() -> None:
    """The Messages stream already sends `ping` keep-alives and gains new event types without a
    version bump, so an unrecognised kind has to be skipped. A parser that raised, or that treated
    an unknown kind as terminal, would fail a turn whose answer arrived perfectly — and it would
    start doing so the day the vendor adds a frame, on a code path no test touched."""
    stream = _Stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "some_future_frame", "payload": {"unknown": True}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )

    turn = await ApiAdapter()._anthropic(stream, {}, lambda _event: None)

    assert turn.failure is None
    assert turn.messages == ("ok",)
    assert turn.raw_usage == {"input_tokens": 1, "output_tokens": 1}


@pytest.mark.asyncio
async def test_an_unknown_content_block_delta_type_leaves_the_block_untouched() -> None:
    """Same forward compatibility one level down, where it is sharper: the four known delta types
    MUTATE the accumulating block, so a fifth one falling through to the last branch would append
    a citation's payload onto the answer text or onto a reasoning signature. A signature with extra
    bytes in it is rejected when the block is replayed on the next turn, which breaks continuation
    rather than this turn — the worst place for it to show up."""
    stream = _Stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": {"url": "x"}}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )

    turn = await ApiAdapter()._anthropic(stream, {}, lambda _event: None)

    assert turn.failure is None
    assert turn.messages == ("ok",), "an unknown delta must not be concatenated into the answer"
    assert turn.history == ({"role": "assistant", "content": [{"type": "text", "text": "ok"}]},)


@pytest.mark.asyncio
async def test_an_anthropic_stream_without_message_stop_is_a_failure_with_its_usage_preserved() -> None:
    """`message_stop` is the ONLY evidence the provider finished. Everything else can be present —
    text, a stop reason, usage — and the response still be a truncated body, because
    `message_delta` arrives before the end of the stream. Accepting it would hand the caller a
    half-sentence as a complete answer with `usage_complete` set, which is both a wrong answer and
    a wrong bill.

    As above, only the classification is pinned; the retryable flag is reported separately."""
    stream = _Stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "half a sent"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
    )

    turn = await ApiAdapter()._anthropic(stream, {}, lambda _event: None)

    assert turn.messages == (), "a truncated body is not an answer"
    assert turn.failure is not None
    assert turn.failure.classification == FailureClass.TRANSIENT_NETWORK
    assert turn.raw_usage == {"input_tokens": 7, "output_tokens": 3}
    assert turn.usage_complete is False


@pytest.mark.asyncio
async def test_a_paused_turn_is_reported_as_a_synthetic_tool_call_so_the_loop_continues() -> None:
    """`pause_turn` is a complete, well-formed response that is NOT the end of the turn: the model
    stopped mid-work (a long server-side tool run) and expects to be called again with its own
    content handed back untouched. There is no tool call to drive that continuation, so the adapter
    manufactures one — without it the runtime sees a finished turn with a partial answer and
    returns it as the result.

    The assistant content has to come back verbatim in the same breath, because that is what gets
    replayed. A synthetic call with no content to resume from would restart the work the caller
    already paid for."""
    stream = _Stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "still working"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "pause_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )

    turn = await ApiAdapter()._anthropic(stream, {}, lambda _event: None)

    assert turn.failure is None
    assert turn.tool_calls == ({"id": "", "name": "__pause_turn__", "arguments": {}},)
    assert turn.refused is False and turn.truncated is False
    assert turn.history == ({"role": "assistant", "content": [{"type": "text", "text": "still working"}]},)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_reason,expected",
    [("refusal", "refused"), ("max_tokens", "truncated"), ("end_turn", None)],
)
async def test_the_stop_reason_decides_refusal_and_truncation_independently(
    stop_reason: str, expected: str | None
) -> None:
    """Two booleans read from one string, and they mean opposite things to the runtime: `refused`
    is terminal (re-asking burns money on the same refusal) while `truncated` is resumable. Both
    false for `end_turn` is the case that makes the pair meaningful — a parser that defaulted
    either to true would mark every normal answer as a refusal or as cut off."""
    stream = _Stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )

    turn = await ApiAdapter()._anthropic(stream, {}, lambda _event: None)

    assert (turn.refused, turn.truncated) == (expected == "refused", expected == "truncated")
    assert turn.tool_calls == (), "only pause_turn manufactures a call"
