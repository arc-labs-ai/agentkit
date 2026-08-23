"""Tests for the http2-enabled client builder.

Verifies the defensive fallback when h2 isn't installed — the
client must still be constructible (over HTTP/1.1).
"""

import asyncio
import sys

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers.base import _make_client


def test_client_constructs_with_default_timeout():
    client = _make_client(30.0)
    try:
        assert isinstance(client, httpx.AsyncClient)
        assert client.timeout.connect == 30.0
    finally:
        # close() is idempotent; sync close is fine for the test
        asyncio.run(client.aclose())


def test_client_falls_back_when_h2_missing(monkeypatch):
    """Hide ``h2`` from import resolution; client must still build."""
    monkeypatch.setitem(sys.modules, "h2", None)  # raises ImportError on import
    client = _make_client(15.0)
    try:
        assert isinstance(client, httpx.AsyncClient)
    finally:
        asyncio.run(client.aclose())


# ── client lifecycle: owned, injected, and re-created across event loops ─────


class _FakeClient:
    """Stands in for the owned httpx.AsyncClient so a close can be observed."""

    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class _Probe:
    """Minimal HttpLLM subclass — `_headers` is the only abstract piece."""

    def _headers(self):
        return {}


def _probe(monkeypatch, **kw):
    """An HttpLLM whose owned client is a _FakeClient, plus the list of ones handed out."""
    from agentkit.adapters.llm.providers import base

    made: list[_FakeClient] = []

    def fake_make_client(timeout):
        made.append(_FakeClient())
        return made[-1]

    monkeypatch.setattr(base, "_make_client", fake_make_client)
    cls = type("_ProbeLLM", (_Probe, base.HttpLLM), {})
    return cls(api_key="k", base_url="http://x", **kw), made


def test_a_client_from_a_dead_event_loop_is_closed_not_leaked(monkeypatch):
    """Regression: one provider instance driven from two `asyncio.run(...)` calls re-created the
    owned client and OVERWROTE the old one with no `aclose()` — measured
    `new client on new loop? True; old client aclose()d? False`, i.e. its pooled sockets leaked
    for the life of the process. The close is best-effort (the owning loop is normally gone), but
    it must be attempted."""
    llm, made = _probe(monkeypatch)

    async def touch():
        llm._http()
        await asyncio.sleep(0)  # let the scheduled orphan close run

    asyncio.run(touch())
    asyncio.run(touch())

    assert len(made) == 2 and made[0] is not made[1]
    assert made[0].closed == 1  # the orphan was released
    assert made[1].closed == 0  # the live one was not


def test_repeated_use_on_one_loop_reuses_a_single_client(monkeypatch):
    """POSITIVE CONTROL. Connection pooling is the whole reason the client is held. A "fix" that
    closed and re-made it per call would pass the leak test and destroy the pooling."""
    llm, made = _probe(monkeypatch)

    async def touch_twice():
        first = llm._http()
        second = llm._http()
        return first is second

    assert asyncio.run(touch_twice()) is True
    assert len(made) == 1 and made[0].closed == 0


def test_an_injected_client_is_never_replaced_or_closed(monkeypatch):
    """POSITIVE CONTROL. The injector owns the lifecycle AND the loop — the documented contract
    tests rely on when they pass a MockTransport client."""
    injected = _FakeClient()
    llm, made = _probe(monkeypatch, client=injected)

    async def touch():
        return llm._http()

    assert asyncio.run(touch()) is injected
    assert asyncio.run(touch()) is injected
    assert made == [] and injected.closed == 0


def test_aclose_releases_an_owned_client(monkeypatch):
    """Positive control for the explicit teardown path, which the orphan close must not disturb."""
    llm, made = _probe(monkeypatch)

    async def use_then_close():
        llm._http()
        await llm.aclose()

    asyncio.run(use_then_close())
    assert made[0].closed == 1 and llm._client is None
