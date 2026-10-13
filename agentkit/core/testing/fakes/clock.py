"""`FakeClock` — deterministic offline ClockPort. `now()` is the current virtual time; `sleep(s)`
advances that virtual time by `s` without actually blocking. Same ergonomics as `FakeLLM`."""

from __future__ import annotations


class FakeClock:
    """Deterministic offline ClockPort. `start` is the initial unix timestamp; `now()` returns the
    current virtual time; `sleep(s)` advances it by `s` (no real wall-clock wait)."""

    def __init__(self, start: float = 0.0):
        self._now = float(start)
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += seconds

    def advance(self, seconds: float) -> None:
        """Test-only escape hatch: move the clock forward without recording a `sleep` call
        (useful for simulating external time passing between agent steps)."""
        self._now += seconds
