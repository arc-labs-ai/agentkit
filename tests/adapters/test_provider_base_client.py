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
