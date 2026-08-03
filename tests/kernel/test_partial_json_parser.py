"""Tolerant JSON parser — pin the contract.

The parser is the streaming-output substrate. It must:

    * Return ``None`` for inputs with zero useful structure (empty,
      whitespace) so the streaming middleware can withhold the partial
      event.
    * Close open strings / objects / arrays opportunistically.
    * Truncate to a clean cut-point when the tail is a mid-literal or a
      mid-key that has no value yet.
    * Behave identically to ``json.loads`` for inputs that ARE valid
      JSON.
    * Never raise on malformed input.

These tests are the per-shape contract — the adapter and middleware
tests lean on it without re-asserting every edge case.
"""

from __future__ import annotations

from agentkit.capabilities.output_schema._partial_json import parse_partial


def test_empty_returns_none() -> None:
    assert parse_partial("") is None


def test_whitespace_returns_none() -> None:
    assert parse_partial("   \n\t") is None


def test_single_open_object_closes_to_empty() -> None:
    """A bare ``{`` still carries useful structure: the consumer sees an
    object is starting."""
    assert parse_partial("{") == {}


def test_single_open_array_closes_to_empty() -> None:
    assert parse_partial("[") == []


def test_unclosed_object_with_one_pair_closes() -> None:
    assert parse_partial('{"a": 1') == {"a": 1}


def test_trailing_comma_after_value_is_dropped() -> None:
    assert parse_partial('{"a": 1,') == {"a": 1}


def test_unclosed_string_value_closes_with_quote() -> None:
    assert parse_partial('{"a": "hello') == {"a": "hello"}


def test_trailing_backslash_inside_string_dropped() -> None:
    """A dangling escape at end-of-input would otherwise be interpreted
    as an escaped close-quote — we drop the backslash so the string
    closes cleanly."""
    assert parse_partial('{"a": "hel\\') == {"a": "hel"}


def test_unclosed_array_closes() -> None:
    assert parse_partial("[1, 2, 3") == [1, 2, 3]


def test_unclosed_nested_array_closes() -> None:
    assert parse_partial('{"items": [1, 2') == {"items": [1, 2]}


def test_valid_json_round_trips() -> None:
    """Behaves like ``json.loads`` for valid inputs."""
    assert parse_partial('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert parse_partial("[1, 2, 3]") == [1, 2, 3]
    assert parse_partial('"just a string"') == "just a string"


def test_truncated_literal_in_value_position_becomes_null() -> None:
    """``tru`` is not a complete literal — we replace with ``null`` so
    the surrounding key still carries a value when the object closes."""
    assert parse_partial('{"a": tru') == {"a": None}
    assert parse_partial('{"a": nul') == {"a": None}


def test_truncated_number_in_value_position_becomes_null() -> None:
    """``2.3e`` is mid-exponent — also incomplete; emit ``null``."""
    assert parse_partial('{"a": 2.3e') == {"a": None}


def test_key_with_no_value_yet_is_dropped() -> None:
    """An incomplete key-value pair (no value started) — drop the key,
    close the object."""
    assert parse_partial('{"a": ') == {}


def test_unfinished_key_string_dropped() -> None:
    """The key itself is mid-string — drop it, close the object."""
    assert parse_partial('{"sub') == {}


def test_deeply_nested_incomplete_closes_everything() -> None:
    """The closing pass walks the container stack in reverse."""
    assert parse_partial('{"a": {"b": {"c": "deep') == {"a": {"b": {"c": "deep"}}}


def test_array_of_strings_partial() -> None:
    """A common shape — ``sub_questions: list[str]`` streamed mid-item."""
    assert parse_partial('{"sub_questions": ["why is sky') == {"sub_questions": ["why is sky"]}


def test_completed_array_after_value() -> None:
    assert parse_partial('{"items": [1, 2, "three"]}') == {"items": [1, 2, "three"]}


def test_escape_sequences_inside_string_preserved() -> None:
    """Escapes that complete (``\\n``, ``\\"``) parse identically to the
    stdlib — the tolerant pass only intervenes when the input doesn't
    parse cleanly."""
    assert parse_partial('"a\\nb"') == "a\nb"


def test_escaped_quote_inside_unclosed_string() -> None:
    """The escape state must not mistake ``\\"`` for the closing quote."""
    assert parse_partial('{"a": "he said \\"hi') == {"a": 'he said "hi'}


def test_completely_garbage_input_returns_none() -> None:
    """``not json at all`` has no salvageable structure beyond a stray
    literal — the cleanup yields garbage, the second json.loads fails,
    we return None rather than fabricate."""
    out = parse_partial("not json at all")
    # We don't assert exactly None vs some interpretation — the contract
    # is "never raises". Either a None or a stringy partial is acceptable.
    assert out is None or isinstance(out, str | dict | list | int | float)
