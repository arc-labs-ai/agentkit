"""Pluggable token counters — ApproxTokenCounter (default) + TiktokenCounter (opt-in).

These tests pin:

  1. ``ApproxTokenCounter`` on an empty list is exactly 0 — a zero-cost
     starting point that won't surprise budget pre-checks.
  2. ``ApproxTokenCounter`` matches chars/4 (rounded down) for English
     content — the rest of agentkit's _approx_tokens uses the same
     heuristic, so a RequestBuilder pre-check stays consistent.
  3. ``TiktokenCounter`` soft-falls-back to the approximation when
     ``tiktoken`` is absent — no ImportError surfaces to the caller.
     The fallback path is forced via ``monkeypatch`` so the test
     runs whether or not the optional dep happens to be installed.
"""

from __future__ import annotations

import asyncio
import builtins

import pytest

from agentkit.context import ApproxTokenCounter, TiktokenCounter
from agentkit.kernel.types import Message


def _run(coro):
    return asyncio.run(coro)


def test_approx_counter_empty_list_is_zero():
    counter = ApproxTokenCounter()
    assert _run(counter.estimate([])) == 0


def test_approx_counter_matches_chars_over_four():
    counter = ApproxTokenCounter(chars_per_token=4.0)
    msgs = [Message("system", "abcd"), Message("user", "efghij")]  # 4 + 6 = 10 chars
    assert _run(counter.estimate(msgs)) == 2


def test_approx_counter_invalid_chars_per_token_raises():
    counter = ApproxTokenCounter(chars_per_token=0.0)
    with pytest.raises(ValueError):
        _run(counter.estimate([Message("user", "anything")]))


def test_tiktoken_counter_soft_falls_back_when_tiktoken_absent(monkeypatch):
    """Force the import-failure branch by blocking ``import tiktoken``.

    The counter must NOT raise; it should silently degrade to the
    chars/4 approximation so a borderline budget check stays sane.
    """
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    counter = TiktokenCounter()
    msgs = [Message("user", "abcd")]
    assert _run(counter.estimate(msgs)) == 1
