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
    rate-limit/5xx/network → TRANSIENT (retried); 4xx → PERMANENT (fail fast)."""


class ProviderAuthError(_KernelProviderAuthError, ProviderError):
    """Raised when the provider returns 401/403 (bad credentials or forbidden).

    Multi-inherits from both the kernel taxonomy (``agentkit.kernel.errors.ProviderAuthError`` →
    ``AgentkitError``) and the transport-level ``ProviderError``, so a raised instance satisfies
    ``except AgentkitError``, ``except ProviderAuthError``, AND legacy ``except ProviderError`` blocks.
    Message keeps the "401/403/unauthorized/forbidden" substrings so ``kernel.resilience.classify``
    maps it to PERMANENT (fail-fast, don't retry a stale key)."""


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


def _classify_4xx(status_code: int, body: str) -> Exception:
    """Return the exception to raise for a 4xx/5xx response.

    Peeks the JSON body for ``error.type`` — providers that return 429
    for account/billing conditions (OpenAI ``billing_not_active`` /
    ``insufficient_quota``) get upgraded to ``ProviderAuthError`` so
    retry doesn't burn the budget on a permanent failure.
    """
    error_type: str | None = None
    try:
        parsed = _json_loads(body)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                t = err.get("type")
                if isinstance(t, str):
                    error_type = t
    except JSONDecodeError:
        pass
    if error_type in _PERMANENT_429_TYPES:
        return ProviderAuthError(
            f"provider auth error (status {status_code}; type={error_type}; unauthorized/forbidden): {body}"
        )
    if status_code == 429 or status_code >= 500:
        return ProviderError(f"provider transient error (status {status_code}; rate limit/overloaded): {body}")
    if status_code in (401, 403):
        return ProviderAuthError(f"provider auth error (status {status_code}; unauthorized/forbidden): {body}")
    return ProviderError(f"provider invalid request (status {status_code}; unauthorized/bad request): {body}")


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
                    body = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise _classify_4xx(resp.status_code, body)
                async for line in resp.aiter_lines():
                    s = line.strip()
                    if not s.startswith("data:"):
                        continue
                    data = s[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        yield _json_loads(data)
                    except JSONDecodeError:
                        continue
        except httpx.HTTPError as exc:  # connect/read/timeout → transient (before any token)
            raise ProviderError(f"provider request failed (network/timeout): {exc}") from exc

    def _http(self) -> Any:
        if self._client is not None and not self._owns_client:
            return self._client  # injected → caller owns lifecycle/loop, reuse as-is
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:  # first use, or driven from a new loop
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
            body = resp.text[:500]
            raise _classify_4xx(resp.status_code, body)
        return resp.json()  # type: ignore[no-any-return]  # httpx returns Any-typed JSON at boundary

    def _cost(self, model: Any, usage: Any) -> float:
        if self._pricing is not None:
            return float(self._pricing(model, usage))
        from agentkit.adapters.llm.providers.pricing import cost

        return cost(model, usage)
