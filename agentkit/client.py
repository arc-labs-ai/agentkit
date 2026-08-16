"""agentkit.client — the batteries-included entry point.

`claude()/openai()/deepseek()/openrouter()` return a ready `Chat`: a provider client (httpx) wrapped in the
**standard resilience chain** (`tracing → meter → retry`) on a `RunContext`. `await chat(user, system=…)`
→ an `LLMResult` (`.content`, `.usage` with cache-aware cost). This is built **on** the middleware chain,
not bypassing it — pass `middleware=[...]` to override, or drop to `Invoker`/`Agent` for tools & loops.

`Chat` owns the provider's httpx client, so use it as an async context manager (`async with claude(...) as chat:`)
or call `await chat.aclose()` when done, to close pooled connections.

`Chat` returns a complete `LLMResult` (it collects the stream under the hood). To consume tokens
incrementally, use `Invoker.stream(...)` or `Agent.stream(...)` directly.
"""

from __future__ import annotations

from typing import Any

from agentkit.kernel.resilience import CircuitBreaker
from agentkit.kernel.types import ChatRequest, LLMResult, Message, Scope
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services


class Chat:
    """A ready-to-use chat callable over a provider LLM, with `tracing → meter → retry` pre-wired."""

    def __init__(
        self,
        llm: Any,
        *,
        model: str | None = None,
        budget: Budget | None = None,
        breaker: bool = True,
        trace: Any = None,
        store: Any = None,
        middleware: list[Any] | None = None,
    ):
        chain = (
            list(middleware)
            if middleware is not None
            else [
                tracing(),
                meter(),
                retry(breaker=CircuitBreaker("agentkit.llm") if breaker else None),
            ]
        )
        self._llm = llm
        self._invoker = Invoker(llm=llm, chat_middleware=chain)
        self._model = model
        self._budget = budget or Budget()
        self._trace = trace
        self._store = store

    async def aclose(self) -> None:
        """Close the underlying provider's httpx client (no-op for a client the caller injected/owns)."""
        closer = getattr(self._llm, "aclose", None)
        if closer is not None:
            await closer()

    async def __aenter__(self) -> Chat:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def __call__(
        self,
        user: str,
        *,
        system: str = "",
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        svc: dict[str, Any] = {"invoker": self._invoker, "store": self._store}
        if self._trace is not None:  # else RunContext's no-op trace default applies
            svc["trace"] = self._trace
        resolved = model or self._model
        if not resolved:
            raise ValueError("no model: pass model= to the preset or to the call")
        ctx = RunContext("chat", Scope(), self._budget, Services(**svc))
        req = ChatRequest(
            messages=[Message("system", system), Message("user", user)],
            model=resolved,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await self._invoker.chat(req, ctx)


def _provider(preset: str, **kw: Any) -> Any:
    # lazy → keeps the httpx extra off the import path
    from agentkit.adapters.llm import providers

    return getattr(providers, preset)(**kw)


def claude(*, api_key: str, model: str = "claude-sonnet-4-6", **chat_opts: Any) -> Chat:
    return Chat(_provider("claude", api_key=api_key, model=model), model=model, **chat_opts)


def openai(*, api_key: str, model: str | None = None, **chat_opts: Any) -> Chat:
    return Chat(_provider("openai", api_key=api_key, model=model), model=model, **chat_opts)


def deepseek(*, api_key: str, model: str = "deepseek-chat", **chat_opts: Any) -> Chat:
    return Chat(_provider("deepseek", api_key=api_key, model=model), model=model, **chat_opts)


def openrouter(*, api_key: str, model: str | None = None, **chat_opts: Any) -> Chat:
    return Chat(_provider("openrouter", api_key=api_key, model=model), model=model, **chat_opts)


def from_env(
    model: str | None = None,
    *,
    provider: str | None = None,
    fallback: str | None = None,
    **chat_opts: Any,
) -> Chat:
    """The zero-bootstrap preset: pick the provider from the model name and read
    the credential from the environment.

        async with from_env("claude-sonnet-4-6") as chat:   # ANTHROPIC_API_KEY
            print((await chat("hi")).content)

    This is the last mile every application was writing by hand — read the
    key, pick the provider, handle the optional-extra ImportError, decide what
    to do when nothing is configured. It composes with the explicit factories
    rather than replacing them: ``claude(api_key=...)`` is untouched and stays
    the right call when the caller already holds the credential.

    ``fallback="fake"`` degrades to a canned LLM (warning once) instead of
    raising when no credential is present — opt-in only, because a fake
    quietly serving production traffic is worse than a startup crash. See
    ``adapters.llm.model_registry.ModelRegistry.resolve``.

    Raises ``UnknownModel`` / ``ProviderNotConfigured`` / ``MissingProviderExtra``
    — all subclasses of ``RegistryError``, so one ``except`` covers the whole
    bootstrap.
    """
    from agentkit.adapters.llm.model_registry import registry as _registry

    reg = _registry()
    llm = reg.resolve(model, provider=provider, fallback=fallback)
    # Resolve the model the Chat will send. An explicit model wins; otherwise
    # fall back to the provider's declared default so ``from_env(provider="…")``
    # doesn't produce a Chat that raises "no model" on first call.
    resolved = model
    if resolved is None and provider is not None:
        entry = reg.provider_entry(provider)
        resolved = entry.default_model if entry is not None else None
    return Chat(llm, model=resolved, **chat_opts)


__all__ = ["Chat", "claude", "openai", "deepseek", "openrouter", "from_env"]
