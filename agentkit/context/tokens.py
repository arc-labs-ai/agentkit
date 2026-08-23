"""Pluggable token-count estimators for a message list.

The counter is the seam that lets ``WorkingContext.tokens()`` answer
"how big is this transcript right now?" without forcing a dependency
on ``tiktoken`` or a provider-side SDK. The default
``ApproxTokenCounter`` is the chars/4 heuristic the rest of agentkit
already uses; ``TiktokenCounter`` opt-in upgrades to provider-accurate
counts when ``tiktoken`` is installed (extras: ``arc-agentkit[fast]``).

Both count the SAME things via :func:`estimate_message_tokens`: message
content, the tool-result ``name``/``tool_call_id``, the serialised
``tool_calls`` of an assistant turn, and a per-wire-block structural
overhead. Tool calls used to be ignored entirely, which made every
compactor a no-op on exactly the transcripts that needed compacting.

The counter is intentionally async — a real counter might call a remote
counting endpoint (provider-side) or warm up an encoder lazily. The
in-process counters are ``async def`` purely to keep the seam uniform
(every counter await is a single ``await`` at the call site).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentkit.kernel.types import Message

_PER_BLOCK_OVERHEAD_TOKENS = 4
"""Structural tokens every provider charges for a wire block on top of its text.

A message is not just its text: role markers, block delimiters and the
start/end-of-turn sentinels cost ~3-4 tokens EACH on every provider we
target. Without this term 500 empty messages estimated as 0 tokens — a
transcript that is genuinely ~2k tokens of scaffolding. Counted once per
message and once per tool call (each tool call is its own wire block with
``id``/``type``/``function`` keys)."""


def _tool_call_texts(tc: Any) -> tuple[str, str, str]:
    """The billable text of one ToolCall: ``(id, name, serialised arguments)``.

    The provider bills the SERIALISED arguments, so that is what we
    measure. The ``dict()`` unwrap is no longer what makes that work:
    ``ToolCall.arguments`` is a ``FrozenDict``, not the
    ``MappingProxyType`` this docstring used to name, and ``json.dumps``
    encodes a ``dict`` subclass natively — verified by passing a
    ``default=`` that raises, which is never called, and the bytes match
    the plain dict's exactly.

    ``tc`` is ``Any`` and read entirely through ``getattr``, and that is
    what the unwrap is actually for. This estimator is deliberately
    duck-typed end to end — ``estimate_message_tokens`` reads
    ``getattr(m, "tool_calls", ())``, and ``middlewares.compaction``
    hands it a ``list[Any]`` it has to filter with ``isinstance(m,
    Message)`` first — because the objects reaching it are not all
    agentkit's. ``Message`` has no ``__post_init__``, so nothing coerces
    ``tool_calls`` either: a provider SDK object or a test double sits
    there unchanged, carrying whatever mapping its author gave it, having
    never passed through ``deep_freeze``. It is NOT for agentkit's own
    ``ToolCall``, whose ``deep_freeze`` now normalises even a
    ``MappingProxyType`` (nested ones too) into a ``FrozenDict``.

    What it buys is ACCURACY, not raise-safety — ``default=str`` already
    catches the exception. Measured on a foreign tool call whose
    ``arguments`` is a ``MappingProxyType({"q": "hi"})``: with the unwrap
    the text is ``{"q": "hi"}`` (11 chars); without it ``json.dumps``
    hands the proxy to ``default=str`` and bills the repr ``"{'q':
    'hi'}"`` (13). A ``ChainMap`` is worse — ``"ChainMap({'q': 'hi'})"``,
    a string about the container rather than a count of the payload. Both
    still produce a number; both produce the WRONG one, and this number
    decides whether a compactor fires.

    The unwrap is SHALLOW, so it is no defence one level down: a mapping
    nested INSIDE the arguments still reaches ``default=str`` and is
    billed as its repr. That is the deliberate floor — ``default=str``
    plus the ``repr`` fallback below keep the count approximate rather
    than fatal, and an approximate number beats an exception thrown from
    inside a budget pre-check.

    Shared by both counters so the approximate and the tiktoken path
    count the same THINGS and differ only in the chars→tokens step."""
    args = getattr(tc, "arguments", None) or {}
    try:
        args_text = json.dumps(dict(args), default=str)
    except Exception:
        args_text = repr(args)
    return (
        str(getattr(tc, "id", "") or ""),
        str(getattr(tc, "name", "") or ""),
        args_text,
    )


def estimate_message_tokens(messages: list[Message], *, chars_per_token: float = 4.0) -> int:
    """THE chars/N transcript estimate — the one every in-process estimator delegates to.

    Why it exists as a shared function: ``compaction.base._approx_tokens``
    and ``ApproxTokenCounter.estimate`` were independent copies of
    ``sum(len(m.content or "")) // 4``, and BOTH ignored ``tool_calls``
    entirely. Measured on an 80-message tool-heavy transcript carrying
    324,420 chars of tool arguments (~81k real tokens): both estimators
    reported **0**, and ``TruncationCompactor(max_tokens=1000)`` kept
    80/80 messages — the whole transcript went to the provider and died
    there as a 400, far from the cause. Tool-heavy IS the normal agentic
    shape, so the "compaction never fires" failure was the default path.

    What is counted, and why each term:

    * ``content`` — text of the turn. A tool RESULT arrives as
      ``Message.content`` on a ``role="tool"`` message, so results were
      already covered and are NOT double-counted here.
    * ``name`` / ``tool_call_id`` — echoed on the wire for every tool
      result message; small, but they are real tokens.
    * ``tool_calls`` — the requested calls on an assistant turn, measured
      in their serialised wire form (see :func:`_tool_call_texts`). This
      is the term whose absence caused the bug.
    * ``_PER_BLOCK_OVERHEAD_TOKENS`` per message and per tool call — the
      structural cost no chars/N ratio can see.

    Deliberately NOT inflated beyond that: a short transcript must not
    now over-count into a needless compaction. The overhead adds 4 tokens
    per block, i.e. ~40 tokens on a 10-turn chat against a 12,000-token
    default budget. An EMPTY transcript is exactly 0 — a zero-cost
    starting point budget pre-checks rely on."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    total_chars = 0
    blocks = 0
    for m in messages:
        total_chars += len(getattr(m, "content", "") or "")
        total_chars += len(getattr(m, "name", "") or "")
        total_chars += len(getattr(m, "tool_call_id", "") or "")
        blocks += 1
        for tc in getattr(m, "tool_calls", ()) or ():
            total_chars += sum(len(t) for t in _tool_call_texts(tc))
            blocks += 1
    return int(total_chars // chars_per_token) + blocks * _PER_BLOCK_OVERHEAD_TOKENS


@runtime_checkable
class TokenCounter(Protocol):
    """Estimates token usage of a message list.

    Pluggable so a cheap-but-fuzzy ``ApproxTokenCounter`` (chars/4) can
    be swapped for a provider-accurate ``TiktokenCounter`` without
    changing the ``WorkingContext`` API. Implementations MUST be
    pure-ish — same input, same output — so a caller can pre-check a
    budget against the same number the invoker will produce.
    """

    async def estimate(self, messages: list[Message]) -> int: ...


@dataclass(frozen=True)
class ApproxTokenCounter:
    """Default — chars/4 approximation over the whole wire payload.

    Free, no deps, ~ok within 30% of actual for English. Use
    ``TiktokenCounter`` when accuracy matters (right before a borderline
    context-limit decision).

    Shares :func:`estimate_message_tokens` with
    ``capabilities.compaction.base._approx_tokens`` — the two were
    separate copies of chars/4 and both silently ignored ``tool_calls``,
    which is how a pre-check here could pass while the compactor over
    there refused to fire on the same transcript.
    ``RequestBuilder.approx_tokens`` now delegates to the same function
    too, so a pre-check, a built request and a compaction decision all
    report the SAME number on a tool-heavy transcript.
    """

    chars_per_token: float = 4.0

    async def estimate(self, messages: list[Message]) -> int:
        # Delegates to the shared :func:`estimate_message_tokens` rather than
        # re-implementing chars/4: the previous private copy here and the one in
        # ``compaction.base`` had drifted into the SAME tool_calls-blind bug, and a
        # caller that pre-checks a budget with this counter then compacts with the
        # compactor MUST see the same number.
        return estimate_message_tokens(messages, chars_per_token=self.chars_per_token)


@dataclass(frozen=True)
class TiktokenCounter:
    """Provider-accurate when ``tiktoken`` is installed.

    Opt-in dep under ``arc-agentkit[fast]``. Falls back to
    ``ApproxTokenCounter`` when ``tiktoken`` is absent or fails to load
    (no import error — just less accurate). The encoding name keys
    into tiktoken's registry; ``cl100k_base`` matches GPT-4/3.5 and
    is a reasonable default for cross-provider estimates.
    """

    encoding: str = "cl100k_base"

    async def estimate(self, messages: list[Message]) -> int:
        try:
            import tiktoken  # type: ignore[import-not-found]
        except Exception:
            return await ApproxTokenCounter().estimate(messages)

        try:
            enc = tiktoken.get_encoding(self.encoding)
        except Exception:
            return await ApproxTokenCounter().estimate(messages)

        total = 0
        for m in messages:
            # Same accounting shape as the approximation (content + tool-call
            # wire form + per-block overhead) — only the chars→tokens step is
            # upgraded. An accurate counter that still ignored tool_calls would
            # reproduce the exact bug this module was fixed for, just with
            # better-looking numbers.
            for text in (m.content or "", m.name or "", m.tool_call_id or ""):
                if text:
                    total += len(enc.encode(text))
            total += _PER_BLOCK_OVERHEAD_TOKENS
            for tc in m.tool_calls or ():
                for text in _tool_call_texts(tc):
                    if text:
                        total += len(enc.encode(text))
                total += _PER_BLOCK_OVERHEAD_TOKENS
        return total


__all__ = [
    "ApproxTokenCounter",
    "TiktokenCounter",
    "TokenCounter",
    "estimate_message_tokens",
]
