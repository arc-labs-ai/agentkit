"""Ports — `FakeSearch` (list + dict + k= + where filter), `FakeFetch` (fixture hit + unknown URL),
`FakeClock` (deterministic now + non-blocking sleep), and `isinstance` against each Protocol."""

import asyncio

import pytest

from agentkit.kernel.ports import (
    ClockPort,
    FetchPort,
    FetchResponse,
    SearchHit,
    SearchPort,
)
from agentkit.testing import FakeClock, FakeFetch, FakeSearch


def _run(coro):
    return asyncio.run(coro)


# ---------- SearchPort / FakeSearch ----------


def test_fake_search_list_returns_configured_hits_and_respects_k():
    hits = [
        SearchHit(url=f"https://x/{i}", title=f"t{i}", snippet=f"s{i}", score=1.0 - i * 0.1)
        for i in range(5)
    ]
    s = FakeSearch(hits)
    got = _run(s.search("anything", k=3))
    assert [h.url for h in got] == ["https://x/0", "https://x/1", "https://x/2"]
    assert all(isinstance(h, SearchHit) for h in got)


def test_fake_search_dict_filters_by_query_substring():
    bucket_a = [SearchHit(url="https://a", title="A", snippet="sa", score=None, metadata={})]
    bucket_b = [SearchHit(url="https://b", title="B", snippet="sb", score=None, metadata={})]
    s = FakeSearch({"alpha": bucket_a, "beta": bucket_b})
    assert _run(s.search("tell me about alpha particles")) == bucket_a
    assert _run(s.search("the beta release notes")) == bucket_b
    assert _run(s.search("nothing matches")) == []  # unknown query → empty, not error


def test_fake_search_applies_where_metadata_filter():
    hits = [
        SearchHit(url="https://x/1", title="t", snippet="s", score=None, metadata={"domain": "ok"}),
        SearchHit(
            url="https://x/2", title="t", snippet="s", score=None, metadata={"domain": "blocked"}
        ),
    ]
    s = FakeSearch(hits)
    got = _run(s.search("q", where={"domain": "ok"}))
    assert [h.url for h in got] == ["https://x/1"]


def test_fake_search_satisfies_search_port_protocol():
    assert isinstance(FakeSearch([]), SearchPort)


# ---------- FetchPort / FakeFetch ----------


def test_fake_fetch_returns_configured_response():
    resp = FetchResponse(
        url="https://example.com",
        status=200,
        headers={"content-type": "text/html"},
        body="<html>hi</html>",
        content_type="text/html",
        fetched_at=1_700_000_000.0,
    )
    f = FakeFetch({"https://example.com": resp})
    got = _run(f.fetch("https://example.com"))
    assert got is resp
    assert got.body == "<html>hi</html>" and got.status == 200
    assert f.calls == ["https://example.com"]


def test_fake_fetch_raises_on_unknown_url():
    f = FakeFetch({})
    with pytest.raises(KeyError) as ei:
        _run(f.fetch("https://nope"))
    assert "https://nope" in str(ei.value)


def test_fake_fetch_satisfies_fetch_port_protocol():
    assert isinstance(FakeFetch({}), FetchPort)


# ---------- ClockPort / FakeClock ----------


def test_fake_clock_now_is_deterministic_and_sleep_advances_without_blocking():
    c = FakeClock(start=1000.0)
    assert c.now() == 1000.0
    assert c.now() == 1000.0  # repeated calls do not drift

    import time as _time

    t0 = _time.monotonic()
    _run(c.sleep(60.0))  # 60 virtual seconds
    _run(c.sleep(5.0))
    wall_elapsed = _time.monotonic() - t0
    assert wall_elapsed < 0.5  # no real sleeping
    assert c.now() == 1065.0
    assert c.sleeps == [60.0, 5.0]


def test_fake_clock_advance_moves_time_without_recording_sleep():
    c = FakeClock(start=0.0)
    c.advance(10.0)
    assert c.now() == 10.0 and c.sleeps == []  # advance ≠ sleep


def test_fake_clock_satisfies_clock_port_protocol():
    assert isinstance(FakeClock(), ClockPort)
