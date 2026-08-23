"""Shared httpx plumbing for the provider clients — the ONE transport (httpx-only, no per-provider
SDKs). Two terminals: `_post` (one-shot JSON, used by `chat()`) and `_stream_events` (the server-sent
events terminal the streaming primitive iterates). Both map HTTP errors to a pre-classified
`ProviderError` (rate-limit/5xx/network → transient; 4xx → permanent) so agentkit's `retry`/`fallback`
middleware reacts correctly. Resilience and observability are the middleware chain's job, NOT this client's.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]  — the optional [http] extra (arc-agentkit[http])

from agentkit.kernel._json import JSONDecodeError
from agentkit.kernel._json import loads as _json_loads
from agentkit.kernel.errors import ProviderAuthError as _KernelProviderAuthError


def raise_if_error_frame(event: dict[str, Any]) -> None:
    """Turn a provider's in-band SSE error frame into a raised ``ProviderError``.

    Both providers can deliver a failure INSIDE a 200 response, part-way
    through a stream, once headers are long gone:

        Anthropic:  {"type": "error", "error": {"type": "overloaded_error", ...}}
        OpenAI:     {"error": {"message": "...", "type": "server_error", ...}}

    Neither translator had a branch for it, so the frame fell through every
    ``elif``, the loop ended normally, and the caller received a TRUNCATED
    answer presented as a complete one — ``finish_reason=None``, partial text,
    no exception anywhere. An agent takes that half-sentence as the model's
    final word. Worse, because nothing raised, ``retry()`` never fired: the
    single most retryable provider failure there is (``overloaded_error``) was
    the one the resilience layer never saw.

    The provider's own error type and message go into the exception text
    because ``kernel.resilience.classify`` is substring-based — ``overloaded``,
    ``rate limit`` and ``429`` are already TRANSIENT there, so an overload
    frame now routes to a retry, and ``invalid_request_error`` routes to
    fail-fast, without this function needing its own error taxonomy.
    """
    err = event.get("error")
    if not isinstance(err, dict) and event.get("type") != "error":
        return
    if not isinstance(err, dict):
        err = {}
    kind = err.get("type") or event.get("type") or "error"
    message = err.get("message") or "provider reported an error mid-stream"
    code = err.get("code")
    detail = f" (code {code})" if code is not None else ""
    raise ProviderError(f"provider stream error [{kind}]{detail}: {message}")


def _make_client(timeout: float) -> httpx.AsyncClient:
    """Build the owned httpx.AsyncClient.

    HTTP/2 multiplexes concurrent streaming requests over one TCP
    connection — meaningful when the Planner fans out to N
    Researchers all hitting the same provider host. Requires the
    ``h2`` package via the ``arc-agentkit[http]`` extra; we probe and
    silently fall back to HTTP/1.1 when absent.

    Explicit Limits prevent a runaway fan-out from exhausting file
    descriptors. Defaults below cover ~20 concurrent agents on one
    provider; raise for higher fan-out.
    """
    try:
        import h2  # type: ignore[import-not-found]  # noqa: F401 — probe only; optional [http] extra, no stubs

        http2 = True
    except ImportError:
        http2 = False
    return httpx.AsyncClient(
        timeout=timeout,
        http2=http2,
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        ),
    )


class ProviderError(Exception):
    """A provider HTTP/transport failure. Its message is crafted so `kernel.resilience.classify` maps it:
    rate-limit/5xx/network → TRANSIENT (retried); 4xx → PERMANENT (fail fast).

    ``status_code``/``body`` carry the machine-readable facts. They exist because the MESSAGE is a
    classification channel first and a debugging channel second — a PERMANENT failure must not print
    the substrings ``classify`` reads as TRANSIENT (see ``_permanent_message``), so the untouched
    provider body lives here instead of inside ``str(exc)``."""

    def __init__(
        self, message: str, *, status_code: int | None = None, body: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ProviderAuthError(_KernelProviderAuthError, ProviderError):
    """Raised when the provider returns 401/403 (bad credentials or forbidden), or a 429 whose body
    names an account/billing condition that no amount of retrying will clear.

    Multi-inherits from both the kernel taxonomy (``agentkit.kernel.errors.ProviderAuthError`` →
    ``AgentkitError``) and the transport-level ``ProviderError``, so a raised instance satisfies
    ``except AgentkitError``, ``except ProviderAuthError``, AND legacy ``except ProviderError`` blocks.
    Message keeps the "401/403/unauthorized/forbidden" substrings so ``kernel.resilience.classify``
    maps it to PERMANENT (fail-fast, don't retry a stale key)."""

    def __init__(
        self, message: str, *, status_code: int | None = None, body: str | None = None
    ) -> None:
        # The MRO puts ``_KernelProviderAuthError`` (plain ``Exception.__init__``) ahead of
        # ``ProviderError``, so a bare ``super().__init__`` would drop the keyword fields and
        # ``ProviderAuthError(msg, status_code=…)`` would raise TypeError. Delegate explicitly.
        ProviderError.__init__(self, message, status_code=status_code, body=body)


# Provider body ``error.type`` values that are PERMANENT despite arriving with
# a 429 (rate-limit) HTTP status. OpenAI returns 429 for both true
# rate-limiting AND for account/billing conditions that will never resolve
# without human action. Retrying these burns the full retry budget on a
# guaranteed-permanent failure. We upgrade them to ``ProviderAuthError`` so
# ``kernel.resilience.classify`` treats them as PERMANENT.
_PERMANENT_429_TYPES: frozenset[str] = frozenset(
    {
        "billing_not_active",
        "insufficient_quota",
        "invalid_api_key",
        "account_deactivated",
    }
)


# Substrings ``kernel.resilience.classify`` reads as TRANSIENT. Mirrors its
# ``_TRANSIENT`` tuple, which we may not edit and deliberately do not import
# (a private cross-module symbol would be a worse coupling than a copy whose
# drift is pinned by a test asserting the CLASSIFICATION, not the list).
_TRANSIENT_SUBSTRINGS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "server_error",
    "overloaded",
    "529",
    "429",
    "timeout",
    "timed out",
    "503",
    "502",
    "504",
    "temporarily",
    "connection reset",
    "connection aborted",
    "circuit open",
)

_BODY_EXCERPT = 500  # how much of the body goes INTO the message (never into the parse)


def _reads_transient(text: str) -> bool:
    """Would ``classify`` call a message containing this text TRANSIENT?"""
    low = text.lower()
    return any(m in low for m in _TRANSIENT_SUBSTRINGS)


def _permanent_message(kind: str, markers: str, status_code: int, excerpt: str, error_type: str | None) -> str:
    """Compose a message ``classify`` cannot mistake for TRANSIENT.

    ``classify`` is substring-based and checks its TRANSIENT list FIRST, so a
    single "429" anywhere in ``str(exc)`` outranks every PERMANENT marker.
    That silently defeated the whole ``_PERMANENT_429_TYPES`` upgrade below:
    measured, ``429/billing_not_active`` raised ``ProviderAuthError`` but
    classified TRANSIENT, and ``run_with_resilience`` spent all 5 attempts
    (measured: "HTTP attempts for a permanent billing error: 5") on a failure
    only a human topping up the account can clear.

    ``resilience.py`` is not ours to change, so the fix is on this side: a
    PERMANENT exception must not SAY anything transient. The status code is
    omitted when it is itself a transient marker, and the provider body is
    withheld from the message when it contains one ("You exceeded your current
    quota … rate limit"). Nothing is lost — the untouched body and the status
    are on ``exc.body`` / ``exc.status_code``.
    """
    status = "" if _reads_transient(str(status_code)) else f"status {status_code}; "
    typed = f"type={error_type}; " if error_type and not _reads_transient(error_type) else ""
    detail = (
        excerpt
        if not _reads_transient(excerpt)
        else "<body withheld: it contains substrings classify() reads as TRANSIENT — see exc.body>"
    )
    return f"{kind} ({status}{typed}{markers}): {detail}"


def _error_type(body: str) -> str | None:
    """The provider's ``error.type``, or None.

    Parses the FULL body. It used to be handed a 500-char truncation, so a
    longer error body was not valid JSON any more and ``error.type`` vanished
    with it — measured: the same 429/``insufficient_quota`` response mapped to
    ``ProviderError`` (transient) when truncated and ``ProviderAuthError``
    (permanent) when whole. Truncation is a DISPLAY concern; it now happens
    after the parse, in the message only.
    """
    try:
        parsed = _json_loads(body)
    except JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    err = parsed.get("error")
    if not isinstance(err, dict):
        return None
    t = err.get("type")
    return t if isinstance(t, str) else None


def _classify_4xx(status_code: int, body: str) -> Exception:
    """Return the exception to raise for a 4xx/5xx response.

    Peeks the JSON body for ``error.type`` — providers that return 429
    for account/billing conditions (OpenAI ``billing_not_active`` /
    ``insufficient_quota``) get upgraded to ``ProviderAuthError`` so
    retry doesn't burn the budget on a permanent failure. A 429 that is a
    GENUINE rate limit keeps its transient message and is still retried.
    """
    error_type = _error_type(body)
    excerpt = body[:_BODY_EXCERPT]
    if error_type in _PERMANENT_429_TYPES:
        return ProviderAuthError(
            _permanent_message(
                "provider auth error", "unauthorized/forbidden", status_code, excerpt, error_type
            ),
            status_code=status_code,
            body=body,
        )
    if status_code == 429 or status_code >= 500:
        return ProviderError(
            f"provider transient error (status {status_code}; rate limit/overloaded): {excerpt}",
            status_code=status_code,
            body=body,
        )
    if status_code in (401, 403):
        return ProviderAuthError(
            _permanent_message(
                "provider auth error", "unauthorized/forbidden", status_code, excerpt, error_type
            ),
            status_code=status_code,
            body=body,
        )
    return ProviderError(
        _permanent_message(
            "provider invalid request", "unauthorized/bad request", status_code, excerpt, error_type
        ),
        status_code=status_code,
        body=body,
    )


def _sse_decode(lines: list[str]) -> tuple[str, Any]:
    """Decode one server-sent event from its accumulated `data:` field lines.

    The SSE spec allows a single `data` field to be split over several lines,
    joined with "\\n". Both providers emit one-line JSON today, but a proxy
    that re-wraps long frames does not — and the previous line-at-a-time parse
    dropped such a payload WHOLE through `except JSONDecodeError: continue`.
    Measured: `multi-line data field -> text deltas: []`, expected `['hello']`
    — a truncated answer with no error raised anywhere, exactly the failure
    `raise_if_error_frame` exists to prevent.

    Returns `("done", None)` for the `[DONE]` sentinel, `("event", obj)` for a
    decoded frame, and `("partial", None)` when the joined text is not (yet)
    valid JSON, meaning the caller should keep accumulating.
    """
    raw = "\n".join(lines).strip()
    if not raw:
        return ("partial", None)
    if raw == "[DONE]":
        return ("done", None)
    try:
        return ("event", _json_loads(raw))
    except JSONDecodeError:
        return ("partial", None)


# Strong refs to in-flight orphan-client closes, so the tasks are not garbage
# collected mid-await (asyncio only holds weak references to running tasks).
_ORPHAN_CLOSES: set[Any] = set()


async def _aclose_quietly(client: Any) -> None:
    """Close a client whose event loop we have already left. Best effort: its pool's
    synchronisation primitives belong to the dead loop, so teardown may raise — which is
    still strictly better than the measured leak of dropping the reference outright."""
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001 — teardown of an orphan must never fail the new call
        pass


class HttpLLM:
    """Base for an httpx-backed `LLMPort`. Subclasses set `_headers()` and the request/response mapping;
    an owned AsyncClient is created lazily and reused for connection pooling — inject one for tests/pooling.

    An owned client is bound to the event loop it was created on, so it is re-created if the provider is
    later driven from a *different* loop (e.g. one instance reused across separate `asyncio.run(...)` calls);
    an injected client is left untouched (the injector owns its lifecycle and loop)."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        default_model: str | None = None,
        pricing: Any = None,
        client: Any = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._pricing = pricing  # cost(model, usage) -> float; None → bundled table
        self._client = client  # injectable httpx.AsyncClient (tests use a MockTransport one)
        self._owns_client = client is None
        self._loop: Any = None  # the loop an OWNED client is bound to (for cross-loop reuse)
        self._timeout = timeout
        self._extra_headers = extra_headers or {}

    def _headers(self) -> dict[str, str]:  # provider-specific auth/version
        raise NotImplementedError

    async def chat(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any = None,
        response_format: Any = None,
        temperature: float = 0.0,
        max_tokens: Any = None,
        cache_hint: Any = None,
    ) -> Any:
        raise NotImplementedError  # subclasses implement the provider request/response mapping

    async def stream(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any = None,
        response_format: Any = None,
        temperature: float = 0.0,
        max_tokens: Any = None,
        cache_hint: Any = None,
    ) -> AsyncIterator[Any]:
        """Streaming primitive. The base yields a single terminal `Delta` wrapping `chat()`; providers that
        support server-sent events override this to emit incremental token deltas."""
        from agentkit.adapters.llm._mapping import result_to_delta

        res = await self.chat(
            messages=messages,
            model=model,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_hint=cache_hint,
        )
        yield result_to_delta(res)

    async def _stream_events(self, path: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """POST a streaming request and yield each server-sent `data:` JSON object as it arrives. A
        connection error or a non-2xx status surfaces as a pre-classified `ProviderError` before the first
        event, so the retry/fallover middleware can re-invoke. Errors after streaming begins propagate.

        Cancellation is cooperative — ``ctx.cancel.cancel()`` does NOT abort a mid-stream response.
        The outer task must be cancelled (``asyncio.CancelledError``) to release the httpx
        connection. The cognition loop polls ``ctx.check_cancelled()`` between turns and stops
        before the NEXT LLM call, but the current one runs to completion. See
        ``docs/mental-models/02-autonomous-devops-investigator.md`` for the cooperative-cancel
        model."""
        try:
            async with self._http().stream(
                "POST", self._base_url + path, json=payload, headers=self._headers()
            ) as resp:
                if resp.status_code >= 400:
                    # NOT truncated here: `_classify_4xx` parses the body for `error.type`, and a
                    # 500-char cut turned a long error body into invalid JSON (see `_error_type`).
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise _classify_4xx(resp.status_code, body)
                pending: list[str] = []
                async for line in resp.aiter_lines():
                    s = line.rstrip("\r\n")
                    if not s.strip():  # blank line — SSE event boundary
                        kind, ev = _sse_decode(pending)
                        pending = []
                        if kind == "done":
                            return
                        if kind == "event":
                            yield ev
                        continue
                    if s.startswith(":") or not s.startswith("data:"):
                        continue  # comment/keep-alive, or an `event:`/`id:`/`retry:` field
                    value = s[5:]
                    if value.startswith(" "):  # spec: exactly ONE leading space is stripped
                        value = value[1:]
                    if pending:
                        # Servers that omit the blank line between events are common enough that
                        # this parser must still handle them (measured: a no-blank-line stream
                        # yielded ['a', 'b'] before this change and must keep doing so). Flush the
                        # buffer only when it ALREADY parses; otherwise this line continues a
                        # payload split across `data:` fields.
                        kind, ev = _sse_decode(pending)
                        if kind == "done":
                            return
                        if kind == "event":
                            pending = []
                            yield ev
                    pending.append(value)
                kind, ev = _sse_decode(pending)  # EOF with no trailing blank line
                if kind == "event":
                    yield ev
        except httpx.HTTPError as exc:  # connect/read/timeout → transient (before any token)
            raise ProviderError(f"provider request failed (network/timeout): {exc}") from exc

    def _http(self) -> Any:
        if self._client is not None and not self._owns_client:
            return self._client  # injected → caller owns lifecycle/loop, reuse as-is
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:  # first use, or driven from a new loop
            if self._client is not None:
                # Measured: the replacement client overwrote the old one with no `aclose()`, so its
                # pooled sockets leaked for the life of the process every time one instance was
                # driven from a second `asyncio.run(...)`. The owning loop is normally already
                # closed by then, so schedule the close on the CURRENT loop and swallow failures.
                task = loop.create_task(_aclose_quietly(self._client))
                _ORPHAN_CLOSES.add(task)
                task.add_done_callback(_ORPHAN_CLOSES.discard)
            self._client = _make_client(self._timeout)
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
            self._loop = None

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http().post(self._base_url + path, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:  # connect/read/timeout → transient
            raise ProviderError(f"provider request failed (network/timeout): {exc}") from exc
        if resp.status_code >= 400:
            # Full body — `_classify_4xx` must PARSE it; truncation happens in the message only.
            raise _classify_4xx(resp.status_code, resp.text)
        return resp.json()  # type: ignore[no-any-return]  # httpx returns Any-typed JSON at boundary

    def _cost(self, model: Any, usage: Any) -> float:
        if self._pricing is not None:
            return float(self._pricing(model, usage))
        from agentkit.adapters.llm.providers.pricing import cost

        return cost(model, usage)
