"""Tolerant JSON parser for streaming structured-output.

The model emits JSON token-by-token; we want to surface a best-effort
partial Python object after each delta arrives, not wait for the
full stream to assemble. Strict ``json.loads`` rejects anything
incomplete — ``{`` alone, an unterminated string, a value with no
comma yet. This parser closes such structures opportunistically so
the caller sees the partial state.

Strategy: walk the input once, track open brackets / quotes /
escape state, then close trailing structures in reverse order. The
result is fed back through ``json.loads``. The pad makes incomplete
inputs syntactically valid; the trailing value the model hadn't
finished writing is dropped (e.g. a key whose value hasn't arrived,
a trailing comma, a truncated literal like ``tru``).

Edge cases handled:
    - Unclosed string: close with ``"``
    - Unterminated escape (``\\`` at end): drop the trailing backslash
    - Unclosed object: close with ``}``; if last key has no value,
      drop the key
    - Unclosed array: close with ``]``; drop trailing comma
    - Truncated literal (``tru``, ``nul``, ``2.3e``): replace with
      ``null`` / drop entirely if context allows
    - Empty input or pure whitespace: return ``None`` (no partial yet)
    - A single open bracket with no content: still surface ``{}`` /
      ``[]`` so the consumer sees the structure starting (not ``None``)

The parser is dep-free and never raises on malformed input. For
inputs that ARE valid JSON, behaves identically to ``json.loads``.
"""

from __future__ import annotations

import json
from typing import Any


def parse_partial(raw: str) -> Any | None:
    """Best-effort parse of an incomplete JSON string.

    Returns ``None`` when the input has zero useful structure (empty
    or whitespace). Returns the most-complete partial Python object
    otherwise. Never raises on malformed input; structure-completion
    is silently applied. For inputs that ARE valid JSON, behaves
    identically to ``json.loads``.
    """
    if not raw or not raw.strip():
        return None

    # Fast-path: already-valid JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fixed = _complete(raw)
    if fixed is None:
        return None
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Internal completion logic
# ---------------------------------------------------------------------------

# Container stack frame markers.
_OBJ = "{"
_ARR = "["


def _complete(raw: str) -> str | None:
    """Scan ``raw``, tracking open containers/strings, and emit a syntactically
    valid JSON string by truncating the tail to a clean cut-point and closing
    each open container.

    Returns ``None`` if the input can't be salvaged at all (only ever for inputs
    so degenerate that there's no useful partial — empty / pure whitespace,
    handled above).
    """
    n = len(raw)
    # Per-position cut points: indices where, if we truncate at this position,
    # the prefix is in a "clean" state (between values, between key-value pairs,
    # or just past a complete value). We track the deepest cut-point we've seen
    # along with the container stack at that point.
    #
    # Strategy:
    #   1. Walk forward, tracking the container stack and string state.
    #   2. Remember the LAST position at which we were "between values" —
    #      i.e. right after a complete value, a comma, ``[``/``{`` (start of
    #      container), or a colon (waiting for value).
    #   3. If the input ends inside a string, close the string at the end.
    #   4. If the input ends in the middle of a literal/number, truncate to
    #      the last clean cut-point.
    #   5. Close any still-open containers in reverse.
    stack: list[str] = []
    # Position tracking: best truncation point at each "key position" / "value
    # position". `cut` is (index, stack-depth-at-that-point, stack-snapshot).
    # We don't need a history — just the latest clean cut-point.
    last_clean_cut = 0  # index up to which the prefix is "clean"
    last_clean_stack: list[str] = []
    # Whether at last_clean_cut we'd be waiting for a value (just past ``:``
    # or just past ``,`` inside an object), or whether we'd be in a fresh
    # state where the next token is a key or a comma. We track this as a
    # boolean: "trailing_comma_or_colon" — if True, the truncation must DROP
    # the trailing ``,`` or ``:`` because closing the container would otherwise
    # leave dangling syntax.
    trailing_sep = False  # True if last_clean_cut sits just past ``,`` or ``:``
    sep_drop_len = 0  # how many chars to drop to remove the trailing sep

    i = 0
    in_string = False
    escape = False
    string_start = -1

    def _record_clean(pos: int, *, after_sep: bool, sep_len: int) -> None:
        nonlocal last_clean_cut, last_clean_stack, trailing_sep, sep_drop_len
        last_clean_cut = pos
        last_clean_stack = list(stack)
        trailing_sep = after_sep
        sep_drop_len = sep_len

    # The state right at the start of input is "clean": empty stack.
    _record_clean(0, after_sep=False, sep_len=0)

    while i < n:
        ch = raw[i]

        if in_string:
            if escape:
                # Whatever character follows ``\\`` is consumed as part of the
                # escape. JSON allows ``\\"`` ``\\\\`` ``\\n`` etc. — we don't
                # need to validate the specific escape, just skip it.
                escape = False
                i += 1
                continue
            if ch == "\\":
                escape = True
                i += 1
                continue
            if ch == '"':
                # End of string.
                in_string = False
                # After a string completes we treat the position AFTER the
                # closing quote as a clean cut-point ONLY when the string was a
                # value (not a key — keys must be followed by ``:`` to be
                # "complete"). The next non-whitespace token tells us; we set
                # the clean cut tentatively here and let the structural-token
                # branches refine it.
                i += 1
                # If the immediately-enclosing container is an array, this
                # string IS a value: clean cut just past the quote.
                # If it's an object, the string is either a key (followed by
                # ``:``) or a value (whatever came after a ``:``). We can't
                # tell without lookahead, so we DO NOT mark clean here for
                # object context; we wait for the structural token.
                if stack and stack[-1] == _ARR:
                    _record_clean(i, after_sep=False, sep_len=0)
                # For object context, the next ``:`` or ``,``/``}`` will set
                # the cut-point appropriately.
                continue
            i += 1
            continue

        if ch == '"':
            in_string = True
            escape = False
            string_start = i
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        if ch == "{":
            stack.append(_OBJ)
            i += 1
            # An empty object is a clean cut (the next ``}`` would close it
            # immediately): record so a truncation here yields ``{}``.
            _record_clean(i, after_sep=False, sep_len=0)
            continue

        if ch == "[":
            stack.append(_ARR)
            i += 1
            _record_clean(i, after_sep=False, sep_len=0)
            continue

        if ch == "}":
            if stack and stack[-1] == _OBJ:
                stack.pop()
            i += 1
            _record_clean(i, after_sep=False, sep_len=0)
            continue

        if ch == "]":
            if stack and stack[-1] == _ARR:
                stack.pop()
            i += 1
            _record_clean(i, after_sep=False, sep_len=0)
            continue

        if ch == ",":
            # A comma marks the end of a value. The position JUST PAST the
            # comma is a candidate cut-point — but closing the container
            # there would require dropping the comma. We mark with
            # ``after_sep=True`` so the closing pass knows to truncate at the
            # PRE-comma position.
            i += 1
            _record_clean(i, after_sep=True, sep_len=1)
            continue

        if ch == ":":
            # A colon is between a key and a value. Truncating at this
            # position would leave a dangling ``"key":`` — the closing pass
            # must drop the key entirely. We model this the same way: mark
            # after_sep=True, and the drop will go back to before the key.
            # Since dropping a key requires knowing where the key started,
            # we do NOT mark a clean cut here. Instead we rely on the
            # PREVIOUS clean cut (which was set when the enclosing ``{`` or
            # the preceding ``,`` was seen). The colon itself is not a
            # clean truncation point.
            i += 1
            continue

        # A literal / number / true / false / null starts here. Scan to its
        # end (whitespace, comma, or closing bracket).
        start = i
        while i < n and raw[i] not in ',:{}[]"\t\n\r ':
            i += 1
        literal = raw[start:i]
        if i >= n:
            # The literal runs to end-of-input — it may be incomplete
            # (e.g. ``tru``, ``2.3e``). If complete (a fully-formed
            # ``true``/``false``/``null``/number), close containers with the
            # literal in place; otherwise substitute ``null`` so the
            # surrounding structure still carries a value at that slot.
            if _is_complete_literal(literal):
                return _finalize(raw, i, list(stack), False, 0)
            # Incomplete literal at value position: replace with null so the
            # surrounding key still has a value when we close. We splice
            # the prefix before the literal + ``null`` + the closers.
            spliced = raw[:start] + "null"
            return _finalize(spliced, len(spliced), list(stack), False, 0)
        # The literal completed (a delimiter follows). Verify it parses; if
        # it's a recognised value-ish token, mark a clean cut just past it.
        if _is_complete_literal(literal):
            if stack and stack[-1] == _ARR:
                _record_clean(i, after_sep=False, sep_len=0)
            else:
                # In object context, the literal is a VALUE (a key must be a
                # string, which would have been handled in the in_string
                # branch). After a value in an object, we're between
                # key-value pairs.
                _record_clean(i, after_sep=False, sep_len=0)
        continue

    # End-of-input fall-through. Use the latest clean cut and close everything.
    if in_string:
        # The model is mid-string. Close the string AT END-OF-INPUT (drop a
        # trailing backslash so we don't leave a dangling escape).
        end = n
        # Drop trailing backslash to avoid emitting ``\"`` which would
        # actually be an escaped quote.
        if end > string_start + 1 and raw[end - 1] == "\\":
            # But only if the backslash isn't itself escaped. Walk back to
            # count consecutive backslashes; an odd count means the last
            # backslash is unescaped (a dangling escape).
            bs = 0
            k = end - 1
            while k > string_start and raw[k] == "\\":
                bs += 1
                k -= 1
            if bs % 2 == 1:
                end -= 1
        # Was this string a KEY in an object (no ``:`` after it) or a VALUE?
        # If at the time we entered the string we were waiting for a key (the
        # stack-top was an object and no clean cut had been recorded past a
        # ``:``), it's a key — and a key without a value isn't valid; we
        # truncate back to the previous clean cut instead of closing.
        in_object_key_position = _is_in_object_key_position(raw, string_start, stack)
        if in_object_key_position:
            # Dangling key — drop it: truncate to last_clean_cut.
            return _finalize(raw, last_clean_cut, last_clean_stack, trailing_sep, sep_drop_len)
        # Otherwise treat the partial string as a value: close at end.
        closed = raw[:end] + '"'
        return _finalize(closed, len(closed), stack, False, 0)

    return _finalize(raw, last_clean_cut, last_clean_stack, trailing_sep, sep_drop_len)


def _is_complete_literal(literal: str) -> bool:
    """Does ``literal`` look like a complete JSON literal/number?

    Used to decide whether to mark a clean cut-point. A truncated ``tru`` /
    ``nul`` / ``2.3e`` returns False so the cut-point stays at the previous
    clean position and the truncated literal is dropped.
    """
    if not literal:
        return False
    if literal in ("true", "false", "null"):
        return True
    # Try to JSON-parse as a number.
    try:
        json.loads(literal)
        return True
    except json.JSONDecodeError:
        return False


def _is_in_object_key_position(raw: str, string_start: int, stack: list[str]) -> bool:
    """At the position ``string_start`` (the opening ``"`` of a string),
    are we starting a KEY in an object (i.e. the enclosing container is an
    object and no ``:`` has appeared since the last ``{`` or ``,``)?
    """
    if not stack or stack[-1] != _OBJ:
        return False
    # Walk backwards from string_start, skipping whitespace, and check whether
    # the previous non-whitespace char is ``{`` or ``,`` (key position) vs
    # ``:`` (value position).
    k = string_start - 1
    while k >= 0 and raw[k].isspace():
        k -= 1
    if k < 0:
        return False
    return raw[k] in "{,"


def _finalize(
    prefix: str, cut: int, stack: list[str], trailing_sep: bool, sep_drop_len: int
) -> str:
    """Build a syntactically valid JSON string by truncating ``prefix`` to
    ``cut`` (less ``sep_drop_len`` chars if ``trailing_sep``) and appending
    the closer for each container in ``stack`` (in reverse).
    """
    end = cut
    if trailing_sep:
        end = max(0, end - sep_drop_len)
    body = prefix[:end].rstrip()
    closers = []
    for frame in reversed(stack):
        closers.append("}" if frame == _OBJ else "]")
    return body + "".join(closers)


__all__ = ["parse_partial"]
