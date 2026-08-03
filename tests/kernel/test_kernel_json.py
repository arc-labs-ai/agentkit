"""Tests for the agentkit.kernel._json fast-JSON shim.

Pins behaviour parity between the orjson and stdlib backends so a
swap doesn't silently break the SSE decode path. The shim is
intentionally narrow — these tests pin exactly the shapes the
adapter hot paths actually produce.
"""

import pytest

from agentkit.kernel._json import BACKEND, JSONDecodeError, dumps, loads


def test_backend_reports_one_of_two_values():
    assert BACKEND in {"orjson", "stdlib"}


def test_dumps_loads_round_trip_for_typical_sse_frame():
    payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    }
    assert loads(dumps(payload)) == payload


def test_dumps_uses_compact_separators_regardless_of_backend():
    # orjson always compacts; stdlib should too via separators=(",", ":")
    assert dumps({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_loads_accepts_bytes_and_str():
    raw = b'{"x": 1}'
    assert loads(raw) == {"x": 1}
    assert loads(raw.decode()) == {"x": 1}


def test_invalid_json_raises_unified_error():
    with pytest.raises(JSONDecodeError):
        loads("{not json")
