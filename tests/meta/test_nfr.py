"""Non-functional invariants (ch29), asserted deterministically — NOT wall-clock timing (CI-stable):
bounded fan-out never exceeds the semaphore, and Budget accounting is exact under concurrent charges."""

import asyncio

from agentkit.kernel.concurrency import gather_bounded
from agentkit.kernel.types import Usage
from agentkit.runtime import Budget


def _run(coro):
    return asyncio.run(coro)


def test_gather_bounded_never_exceeds_the_semaphore():
    state = {"cur": 0, "peak": 0}

    async def work(i):
        state["cur"] += 1
        state["peak"] = max(state["peak"], state["cur"])
        await asyncio.sleep(0)            # yield so siblings interleave (forces peak concurrency)
        state["cur"] -= 1
        return i

    async def go():
        sem = asyncio.Semaphore(4)
        out = await gather_bounded([work(i) for i in range(50)], sem=sem)
        return out, state["peak"]

    out, peak = _run(go())
    assert out == list(range(50))         # order preserved
    assert peak <= 4                      # concurrency strictly bounded by the semaphore


def test_budget_charges_are_exact_under_concurrency():
    async def go():
        b = Budget()                      # no ceilings → just accumulate
        sem = asyncio.Semaphore(8)
        await gather_bounded([b.charge(None, Usage(1, 1, 0.001)) for _ in range(200)], sem=sem)
        return b

    b = _run(go())
    assert b.calls == 200                              # no lost increments
    assert round(b.spent_usd, 6) == round(200 * 0.001, 6)   # exact cost accrual
