"""`FakeFetch` — deterministic offline FetchPort for tests. Constructed with a `{url: FetchResponse}`
fixture map; raises a clear `KeyError` on any URL not in the map (no silent 404s in tests)."""

from __future__ import annotations

from agentkit.kernel.ports import FetchResponse


class FakeFetch:
    """Deterministic offline FetchPort. `fixtures` is `{url: FetchResponse}`; unknown URLs raise
    `KeyError(url)` so tests fail loud instead of silently falling back to a stub response."""

    def __init__(self, fixtures: dict[str, FetchResponse]):
        self._fixtures = dict(fixtures)
        self.calls: list[str] = []

    async def fetch(self, url: str, *, timeout_s: float = 10.0) -> FetchResponse:
        self.calls.append(url)
        if url not in self._fixtures:
            raise KeyError(f"FakeFetch: no fixture configured for url {url!r}")
        return self._fixtures[url]
