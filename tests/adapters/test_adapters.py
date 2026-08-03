"""Adapters — FakeLLM scripting, InMemoryStore (single-flight + never-cache-failure + log),
InMemoryVector (scored search + scope isolation), + the typed error taxonomy that adapter
boundaries raise (ProviderAuthError on 401/403, CheckpointerError wrapping malformed JSON,
and the top-level ``from agentkit import`` export."""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.store import InMemoryStore
from agentkit.adapters.vector import InMemoryVector
from agentkit.kernel.types import Chunk, Message, Scope, ToolCall
from agentkit.testing import FakeLLM, Turn


def _run(coro):
    return asyncio.run(coro)


def test_fakellm_script_replays_turns():
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),)), Turn(content="done")])
    r1 = _run(llm.chat(messages=[Message("user", "x")], model="m"))
    r2 = _run(llm.chat(messages=[Message("user", "x")], model="m"))
    assert r1.tool_calls[0].name == "fetch" and r2.content == "done"


def test_store_get_or_set_single_flight_and_never_caches_failure():
    store = InMemoryStore()
    n = {"c": 0}

    async def produce():
        n["c"] += 1
        return {"v": n["c"]}

    a = _run(store.get_or_set("k", produce))
    b = _run(store.get_or_set("k", produce))
    assert a == b == {"v": 1} and n["c"] == 1  # cached

    async def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        _run(store.get_or_set("k2", boom))
    assert _run(store.get("k2")) is None  # failure not stored


def test_store_append_and_list():
    store = InMemoryStore()
    _run(store.append("log", {"a": 1}))
    _run(store.append("log", {"a": 2}))
    assert _run(store.list("log")) == [{"a": 1}, {"a": 2}]


def test_vector_scored_search_and_isolation():
    v = InMemoryVector()
    _run(v.upsert(Scope(1, 1), [Chunk("1", "pricing and billing", {"source": "d"})]))
    hits = _run(v.search(Scope(1, 1), "pricing", k=3))
    assert hits and hits[0][0] > 0 and hits[0][1].id == "1"  # (score, chunk), score > 0
    assert _run(v.search(Scope(2, 2), "pricing")) == []  # other tenant isolated


# ── Typed error taxonomy at adapter boundaries ───────────────────────────────
#
# Adapters must never leak backend types (httpx / asyncpg / redis) past their
# port. The tests below pin down the three concrete boundaries where wrapping
# lives, plus the top-level re-export so callers can ``from agentkit import
# ProviderAuthError`` and pattern-match on it without reaching into the
# adapter package.


def test_provider_auth_error_on_401():
    """A provider that answers 401 raises ``ProviderAuthError`` — which
    multi-inherits from the kernel taxonomy AND the transport-level
    ``ProviderError``, so legacy ``except ProviderError`` blocks keep firing
    while new callers can pattern-match on the auth-specific type."""
    from agentkit import AgentkitError, ProviderAuthError
    from agentkit.adapters.llm.providers import OpenAICompatibleLLM, ProviderError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_api_key"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = OpenAICompatibleLLM(api_key="sk-bad", base_url="http://x", client=client)

    with pytest.raises(ProviderAuthError) as exc_info:
        _run(llm.chat(messages=[Message("user", "q")], model="m"))

    exc = exc_info.value
    assert isinstance(exc, ProviderAuthError)  # the taxonomy the caller pattern-matches on
    assert isinstance(
        exc, ProviderError
    )  # backward-compat: legacy `except ProviderError` still fires
    assert isinstance(exc, AgentkitError)  # under the top-level framework base


def test_provider_auth_error_also_fires_for_403():
    """403 (forbidden) belongs to the same auth-failure family as 401
    (unauthorized) — both are "credentials the server won't accept",
    neither retriable. Same class, distinct status in the message so
    ``classify`` still maps it to PERMANENT."""
    from agentkit import ProviderAuthError
    from agentkit.adapters.llm.providers import OpenAICompatibleLLM

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = OpenAICompatibleLLM(api_key="sk-x", base_url="http://x", client=client)

    with pytest.raises(ProviderAuthError):
        _run(llm.chat(messages=[Message("user", "q")], model="m"))


def test_checkpointer_error_wraps_malformed_json():
    """A row with unparseable JSON in the ``state`` column raises
    ``CheckpointerError`` from ``_row_to_checkpoint``, with the original
    ``json.JSONDecodeError`` preserved as ``__cause__`` for debugging.
    Tested at the decode function directly (no live Postgres needed) —
    the port method wraps the same way at the outer boundary."""
    from agentkit import CheckpointerError
    from agentkit.adapters.checkpoint.postgres import _row_to_checkpoint

    corrupt_row = {
        "run_id": "run-corrupt",
        "version": 1,
        "state": '{"phase": "planning"',  # unclosed brace + missing quote
        "created_at": 1_700_000_000.0,
        "status": "running",
        "metadata": "{}",
    }

    with pytest.raises(CheckpointerError) as exc_info:
        _row_to_checkpoint(corrupt_row)

    # __cause__ preserved via ``raise CheckpointerError(...) from exc`` — the caller
    # can drill into the underlying JSON parser error if it wants.
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_error_taxonomy_exports_from_top_level():
    """The four typed error classes are reachable from the top-level
    ``agentkit`` package — callers should never have to reach into
    ``agentkit.kernel.errors`` to except on a framework error."""
    from agentkit import (
        AgentkitError,
        CheckpointerError,
        ProviderAuthError,
        StoreUnavailable,
    )

    # Subclass relationships hold — a single ``except AgentkitError`` catches all four.
    assert issubclass(CheckpointerError, AgentkitError)
    assert issubclass(StoreUnavailable, AgentkitError)
    assert issubclass(ProviderAuthError, AgentkitError)
    # And ``AgentkitError`` itself is a plain ``Exception`` subclass — an ``except Exception:``
    # sweep-up doesn't miss it.
    assert issubclass(AgentkitError, Exception)
