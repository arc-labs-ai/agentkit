"""Token estimates belong to WorkingContext and count complete message payloads."""

import asyncio
import builtins

import pytest

from agentkit.core.contracts.types import Message, ToolCall
from agentkit.state.context import ApproxTokenCounter, TiktokenCounter


def _estimate(messages, counter=None):
    return asyncio.run((counter or ApproxTokenCounter()).estimate(messages))


def test_empty_context_has_zero_estimated_tokens():
    assert _estimate([]) == 0


def test_messages_include_structural_overhead():
    messages = [Message("system", "abcd"), Message("user", "efghij")]
    assert _estimate(messages) == 10 // 4 + 2 * 4


def test_tool_calls_and_results_are_counted():
    messages = [
        Message("assistant", "", tool_calls=(ToolCall("call-1", "search", {"query": "x" * 4000}),)),
        Message("tool", "matched", name="search", tool_call_id="call-1"),
    ]
    assert _estimate(messages) > 1000


def test_tiktoken_counter_falls_back_when_optional_dependency_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    messages = [Message("user", "abcd")]
    assert _estimate(messages, TiktokenCounter()) == _estimate(messages)


@pytest.mark.parametrize("ratio", [0, float("nan"), float("inf"), -float("inf")])
def test_invalid_token_ratios_are_refused(ratio):
    with pytest.raises(ValueError):
        _estimate([], ApproxTokenCounter(ratio))
