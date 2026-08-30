"""Compactors — built-in strategies for bounding a transcript.

Each `Compactor` is a self-contained policy: the threshold check + the
transformation. Calling `compact()` on a transcript that's already under budget is a
no-op, so a caller can call it unconditionally without a guard."""

import pytest

from agentkit.capabilities.compaction import (
    Compactor,
    ImportanceFilteringCompactor,
    Message,
    SlidingWindowCompactor,
    SummarizationCompactor,
    TruncationCompactor,
)
from agentkit.kernel.types import ToolCall
from agentkit.testing import FakeCtx, FakeLLM

# ---- Protocol membership ----------------------------------------------------------------------


def test_concrete_types_satisfy_compactor_protocol():
    """All four built-in compactors should structurally satisfy the `Compactor` Protocol
    — no inheritance needed. This guards the rename: a wrapper class would have failed
    here since the old wrapper exposed `maybe_compact`, not `compact`."""
    assert isinstance(SlidingWindowCompactor(), Compactor)
    assert isinstance(TruncationCompactor(), Compactor)
    assert isinstance(SummarizationCompactor(summarizer=FakeLLM(responses="s")), Compactor)
    assert isinstance(ImportanceFilteringCompactor(filterer=FakeLLM(responses="s")), Compactor)


# ---- SlidingWindowCompactor -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_sliding_window():
    strategy = SlidingWindowCompactor(keep_recent=2)
    messages = [
        Message("system", "sys"),
        Message("user", "u1"),
        Message("assistant", "a1"),
        Message("user", "u2"),
        Message("assistant", "a2"),
    ]
    # Should keep system + last 2 (u2, a2)
    compacted = await strategy.compact(messages, FakeCtx())
    assert len(compacted) == 3
    assert compacted[0].content == "sys"
    assert compacted[1].content == "u2"
    assert compacted[2].content == "a2"


@pytest.mark.asyncio
async def test_sliding_window_no_system():
    strategy = SlidingWindowCompactor(keep_recent=2)
    messages = [
        Message("user", "u1"),
        Message("assistant", "a1"),
        Message("user", "u2"),
        Message("assistant", "a2"),
    ]
    compacted = await strategy.compact(messages, FakeCtx())
    assert len(compacted) == 2
    assert compacted[0].content == "u2"
    assert compacted[1].content == "a2"


@pytest.mark.asyncio
async def test_truncation():
    # estimate is len // 4. 4 messages of 20 chars each = 80 chars -> 20 tokens.
    strategy = TruncationCompactor(max_tokens=10, keep_recent=1)
    messages = [
        Message("system", "sys"),
        Message("user", "u" * 20),
        Message("assistant", "a" * 20),
        Message("user", "u" * 20),
    ]
    compacted = await strategy.compact(messages, FakeCtx())
    assert len(compacted) < 4
    assert compacted[0].role == "system"


@pytest.mark.asyncio
async def test_summarization():
    llm = FakeLLM(responses="Summarized text")
    strategy = SummarizationCompactor(summarizer=llm, max_tokens=5, keep_recent=1)
    messages = [
        Message("system", "sys"),
        Message("user", "user turn 1"),
        Message("assistant", "assistant turn 1"),
        Message("user", "user turn 2"),
    ]
    compacted = await strategy.compact(messages, FakeCtx())
    assert len(compacted) == 3  # system + summary + user turn 2
    assert compacted[0].content == "sys"
    assert "Summary" in compacted[1].content
    assert "Summarized text" in compacted[1].content
    assert compacted[2].content == "user turn 2"


@pytest.mark.asyncio
async def test_importance_filtering():
    llm = FakeLLM(responses="Important facts")
    strategy = ImportanceFilteringCompactor(filterer=llm, max_tokens=5, keep_recent=1)
    messages = [
        Message("system", "sys"),
        Message("user", "user turn 1"),
        Message("assistant", "assistant turn 1"),
        Message("user", "user turn 2"),
    ]
    compacted = await strategy.compact(messages, FakeCtx())
    assert len(compacted) == 3
    assert compacted[0].content == "sys"
    assert "Key points" in compacted[1].content
    assert "Important facts" in compacted[1].content


@pytest.mark.asyncio
async def test_sliding_window_orphaned_tool():
    strategy = SlidingWindowCompactor(keep_recent=1)
    messages = [
        Message("system", "sys"),
        Message("assistant", "call", tool_calls=[{"id": "1", "name": "f"}]),
        Message("tool", "result", tool_call_id="1"),
    ]
    # keep_recent=1 would normally keep only the tool message.
    # But it should walk back to avoid orphaning.
    compacted = await strategy.compact(messages, FakeCtx())
    assert len(compacted) == 3
    assert compacted[1].role == "assistant"
    assert compacted[2].role == "tool"


@pytest.mark.parametrize("keep_recent", [0, -1, -5])
@pytest.mark.asyncio
async def test_sliding_window_non_positive_keep_recent_drops_history(keep_recent: int):
    """``keep_recent=0`` means "keep the system prompt, drop every turn" — the
    natural way to express a full reset, and a value a config file reaches by
    default far more often than by intent.

    It used to raise ``IndexError``. ``start`` is computed as
    ``len(messages) - keep_recent``, which for a non-positive ``keep_recent``
    lands AT or PAST the end of the list — fine as a slice bound, fatal as the
    subscript in the orphaned-tool walk-back (``messages[start].role``). Any
    transcript of two or more messages hit it.

    ``ImportanceFilteringCompactor`` runs the identical walk-back and never had
    the bug, because its loop is guarded ``while keep and start > head``. This
    is that same guard, plus a clamp so a negative value means the same thing as
    zero rather than reaching further past the end."""
    messages = [Message("system", "sys")] + [Message("user", f"u{i}") for i in range(5)]

    compacted = await SlidingWindowCompactor(keep_recent=keep_recent).compact(
        messages, FakeCtx()
    )

    assert [m.content for m in compacted] == ["sys"]


@pytest.mark.asyncio
async def test_sliding_window_zero_keep_recent_without_system_prompt():
    """The same clamp with no system message to fall back on: everything goes,
    and the caller gets an empty transcript rather than an ``IndexError``."""
    messages = [Message("user", "u1"), Message("assistant", "a1")]

    compacted = await SlidingWindowCompactor(keep_recent=0).compact(messages, FakeCtx())

    assert compacted == []


# ---- under-ceiling is a no-op ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_under_ceiling_is_noop():
    """Each Compactor handles its own threshold check internally: callers can call
    `compact()` unconditionally without guarding."""
    msgs = [Message("system", "sys"), Message("user", "hi")]
    assert await TruncationCompactor(max_tokens=100_000).compact(msgs, FakeCtx()) == msgs
    assert (
        await SummarizationCompactor(summarizer=FakeLLM(responses="s"), max_tokens=100_000).compact(
            msgs, FakeCtx()
        )
        == msgs
    )
    assert await SlidingWindowCompactor(keep_recent=100).compact(msgs, FakeCtx()) == msgs


# ── Compaction middleware fallback on empty result ──────────────────────────
#
# A compactor is an optimization; the middleware must never install
# a rewrite that would send the LLM a zero-message request (400 or
# hallucination depending on the vendor). A buggy or over-aggressive
# custom compactor that returns ``[]`` falls back to the original
# messages, keeping the run alive.


@pytest.mark.asyncio
async def test_middleware_falls_back_when_compactor_returns_empty() -> None:
    """A compactor that returns ``[]`` must not blank the request.
    The middleware detects the empty result and reinstates the
    caller's original messages, preserving cache stability and
    saving the run from a mysterious provider 400."""
    from agentkit.kernel.middleware import Call
    from agentkit.kernel.types import ChatRequest, Message, Operation, Scope
    from agentkit.middlewares.compaction import Compaction
    from agentkit.runtime.context import RunContext

    class _EmptyingCompactor:
        """Buggy: always returns []. Exercises the middleware guard."""

        name = "empty"

        async def compact(self, messages, ctx):
            return []

    ctx = RunContext("run-x", Scope(1, 1))
    original = [
        Message("system", "you are a diligent assistant"),
        Message("user", "hello"),
        Message("assistant", "hi"),
        Message("user", "how are you"),
    ]
    call = Call(
        kind="chat",
        request=ChatRequest(messages=list(original), model="m"),
        ctx=ctx,
    )
    mctx = type("Ctx", (), {})()
    mctx.operation = Operation.MODEL_CALL
    mctx.request = call.request
    mctx.run = ctx
    # A tiny MiddlewareContext facade: real one is more elaborate but
    # ``on_request`` only reads ``operation``, ``request``, ``run``,
    # and writes ``request``.
    from agentkit.kernel.middleware import MiddlewareContext

    real_mctx = MiddlewareContext(call)

    mw = Compaction(_EmptyingCompactor())
    await mw.on_request(real_mctx)

    # Fallback fired: the request still carries the original messages.
    assert list(real_mctx.request.messages) == original


@pytest.mark.asyncio
async def test_middleware_honors_valid_compaction_result() -> None:
    """Sanity: a compactor that returns SOMETHING (even one message)
    is trusted. The empty-result guard is a last resort, not a
    filter on "compaction that shrinks too much"."""
    from agentkit.kernel.middleware import Call, MiddlewareContext
    from agentkit.kernel.types import ChatRequest, Message, Scope
    from agentkit.middlewares.compaction import Compaction
    from agentkit.runtime.context import RunContext

    class _AggressiveCompactor:
        """Keeps only the system prompt. Extreme but non-empty."""

        name = "aggressive"

        async def compact(self, messages, ctx):
            return [m for m in messages if m.role == "system"]

    ctx = RunContext("run-y", Scope(1, 1))
    original = [Message("system", "sys"), Message("user", "u"), Message("assistant", "a")]
    call = Call(
        kind="chat",
        request=ChatRequest(messages=list(original), model="m"),
        ctx=ctx,
    )
    mctx = MiddlewareContext(call)

    await Compaction(_AggressiveCompactor()).on_request(mctx)

    # The aggressive result stuck — the middleware only rescues empty results.
    assert len(mctx.request.messages) == 1
    assert mctx.request.messages[0].role == "system"


# ---- tool-heavy transcripts (compaction never fired) ------------------------------------------
#
# The default ``estimate`` summed ``len(m.content)`` only. A tool-heavy
# transcript — the NORMAL agentic shape — carries its bulk in
# ``Message.tool_calls``, so it scored ~0 and every compactor no-op'd.
# Measured before the fix: 80 messages / 324,420 chars of tool arguments
# (~81k tokens) estimated as 20, and ``TruncationCompactor(max_tokens=1000)``
# kept 80/80 messages. The provider rejected the request with a 400, far
# from the cause.


def _tool_heavy(turns: int, arg_chars: int) -> list[Message]:
    msgs: list[Message] = [Message("system", "sys")]
    for i in range(turns):
        msgs.append(
            Message(
                "assistant",
                "",
                tool_calls=(
                    ToolCall(id=f"call-{i}", name="search", arguments={"q": "x" * arg_chars}),
                ),
            )
        )
        msgs.append(Message("tool", "ok", name="search", tool_call_id=f"call-{i}"))
    return msgs


@pytest.mark.asyncio
async def test_truncation_fires_on_a_tool_heavy_transcript():
    """REGRESSION: 80/80 messages survived a 1000-token ceiling."""
    messages = _tool_heavy(turns=40, arg_chars=4000)
    compacted = await TruncationCompactor(max_tokens=1000, keep_recent=4).compact(
        messages, FakeCtx()
    )
    assert len(compacted) < len(messages)
    assert compacted[0].role == "system"  # cache-stable head survives


@pytest.mark.asyncio
async def test_summarization_fires_on_a_tool_heavy_transcript():
    """The same blindness hit every compactor that used the default estimate."""
    messages = _tool_heavy(turns=10, arg_chars=4000)
    strategy = SummarizationCompactor(
        summarizer=FakeLLM(responses="Summarized"), max_tokens=1000, keep_recent=2
    )
    compacted = await strategy.compact(messages, FakeCtx())
    assert len(compacted) < len(messages)
    assert "Summarized" in compacted[1].content


@pytest.mark.asyncio
async def test_tool_heavy_estimate_matches_the_context_counter():
    """The compactor's threshold check and a caller's ``ApproxTokenCounter``
    pre-check must see the SAME number — disagreeing copies are how the
    tool-blind estimate survived."""
    from agentkit.capabilities.compaction.base import _approx_tokens
    from agentkit.context import ApproxTokenCounter

    messages = _tool_heavy(turns=5, arg_chars=800)
    assert _approx_tokens(messages) == await ApproxTokenCounter().estimate(messages)


@pytest.mark.asyncio
async def test_short_tool_transcript_is_still_a_noop():
    """POSITIVE CONTROL — counting tool calls must not make a small
    tool-using transcript compact needlessly."""
    messages = _tool_heavy(turns=2, arg_chars=20)
    assert await TruncationCompactor(max_tokens=12_000).compact(messages, FakeCtx()) == messages
    assert (
        await SummarizationCompactor(
            summarizer=FakeLLM(responses="s"), max_tokens=12_000
        ).compact(messages, FakeCtx())
        == messages
    )
