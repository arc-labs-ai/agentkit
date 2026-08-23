"""`Audit` must record the side effects that FAILED, not just the ones that returned.

`Audit` wrote only in `on_response`, i.e. only after a success. That is exactly
backwards for a side-effecting tool: the dangerous case is the one that fired
the side effect and *then* failed — the payment gateway that charges the card
and times out on the response.

Measured before the fix, with the documented tool chain
(`[… idempotent(), audit(), retry(breaker=…)]`): a charging tool that raised on
every attempt executed 3 times and left `audit records: []`. The money moved
and the audit trail was empty.

The attempt COUNT is an ordering property, not something `Audit` can detect
from outside `retry()` — both orderings are exercised below.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from agentkit.adapters.store.memory import InMemoryStore
from agentkit.kernel.types import Scope, ToolRequest
from agentkit.middlewares import audit, idempotent, retry
from agentkit.runtime import Budget, Invoker, RunContext, Services

_NO_SLEEP = lambda _s: asyncio.sleep(0)  # noqa: E731


class _ChargingTool:
    """Charges the card, then optionally fails on the way back — the shape the
    audit trail exists for."""

    name = "charge_card"

    def __init__(self, *, fail_times: int = 0, exc: BaseException | None = None) -> None:
        self.executions = 0
        self._fail_times = fail_times
        self._exc = exc or TimeoutError("gateway timed out AFTER charging the card")

    async def run(self, arguments: dict[str, Any], ctx: Any) -> Any:
        self.executions += 1
        if self.executions <= self._fail_times:
            raise self._exc
        return {"receipt": f"r{self.executions}"}


def _wire(store: InMemoryStore, *middleware: Any) -> tuple[Invoker, RunContext]:
    inv = Invoker(llm=None, tool_middleware=list(middleware))
    ctx = RunContext("run", Scope(org_id=7), Budget(), Services(invoker=inv, store=store))
    return inv, ctx


def _log(store: InMemoryStore) -> list[dict[str, Any]]:
    return asyncio.run(store.list("audit:log"))


def _req(tool: _ChargingTool) -> ToolRequest:
    return ToolRequest("charge_card", {"amount": 100}, tool, side_effecting=True)


def test_a_failed_side_effect_is_recorded() -> None:
    """The load-bearing test: the tool ran, the call raised, and the trail must
    not be empty."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=1)
    inv, ctx = _wire(store, audit(store=store))

    with pytest.raises(TimeoutError):
        asyncio.run(inv.invoke_tool(_req(tool), ctx))

    (record,) = _log(store)
    assert tool.executions == 1
    assert record["decision"] == "failed", "a failed side effect left no audit record"
    assert record["error_type"] == "TimeoutError"
    assert "charging the card" in record["error"]
    assert record["tool"] == "charge_card"
    assert record["scope"] == ctx.scope.key()


def test_the_failure_still_reaches_the_caller() -> None:
    """`Audit` records; it never recovers. `on_error` returning a value would
    silently turn a failed charge into a successful tool result."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=1)
    inv, ctx = _wire(store, audit(store=store))

    with pytest.raises(TimeoutError, match="gateway timed out"):
        asyncio.run(inv.invoke_tool(_req(tool), ctx))


def test_a_broken_audit_store_does_not_mask_the_tools_failure() -> None:
    """If the audit write itself fails on the error path, the caller must still
    see the TOOL's exception — not the store's."""

    class _BrokenStore(InMemoryStore):
        async def append(self, key: str, value: Any) -> None:
            raise OSError("audit disk full")

    store, tool = _BrokenStore(), _ChargingTool(fail_times=1)
    inv, ctx = _wire(store, audit(store=store))

    with pytest.raises(TimeoutError, match="gateway timed out"):
        asyncio.run(inv.invoke_tool(_req(tool), ctx))


def test_audit_OUTSIDE_retry_folds_the_attempts_into_one_record() -> None:
    """The documented tool chain. `retry()` is INSIDE `audit()`, so its attempts
    are invisible from here: three executions produce ONE `failed` record. This
    is the ordering trade-off, pinned so it stays a decision rather than a
    surprise."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=99)
    inv, ctx = _wire(store, audit(store=store), retry(max_attempts=3, sleep=_NO_SLEEP))

    with pytest.raises(TimeoutError):
        asyncio.run(inv.invoke_tool(_req(tool), ctx))

    assert tool.executions == 3
    assert [r["decision"] for r in _log(store)] == ["failed"]


def test_audit_INSIDE_retry_records_every_attempt() -> None:
    """The other documented ordering, for operators who need "how many times did
    the side effect actually fire". Two failed attempts then a success — three
    records, in order."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=2)
    inv, ctx = _wire(store, retry(max_attempts=3, sleep=_NO_SLEEP), audit(store=store))

    result = asyncio.run(inv.invoke_tool(_req(tool), ctx))

    assert result == {"receipt": "r3"}
    assert tool.executions == 3
    assert [r["decision"] for r in _log(store)] == ["failed", "failed", "executed"]


def test_a_successful_call_is_still_recorded_as_executed() -> None:
    """POSITIVE CONTROL. A "fix" that recorded only failures — or that turned
    the whole middleware off — would pass every test above and fail here."""
    store, tool = InMemoryStore(), _ChargingTool()
    inv, ctx = _wire(store, audit(store=store))

    result = asyncio.run(inv.invoke_tool(_req(tool), ctx))

    (record,) = _log(store)
    assert result == {"receipt": "r1"}
    assert record["decision"] == "executed"
    assert record["result_hash"] is not None
    assert "error_type" not in record


def test_an_idempotent_replay_is_still_recorded_as_deduped() -> None:
    """The other half of the decision vocabulary, unchanged: with `audit()`
    outside `idempotent()`, a replay is recorded — and marked — rather than
    counted as a second execution."""
    store, tool = InMemoryStore(), _ChargingTool()
    inv, ctx = _wire(store, audit(store=store), idempotent(store=store))

    async def go() -> None:
        await inv.invoke_tool(_req(tool), ctx)
        await inv.invoke_tool(_req(tool), ctx)

    asyncio.run(go())

    assert tool.executions == 1
    assert [r["decision"] for r in _log(store)] == ["executed", "deduped"]


def test_a_failure_is_not_cached_by_idempotent_and_is_audited_each_time() -> None:
    """A failure is never stored (`get_or_set` does not cache a raised
    producer), so the second call really re-fires — and must be audited again."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=99)
    inv, ctx = _wire(store, audit(store=store), idempotent(store=store))

    async def go() -> None:
        for _ in range(2):
            with contextlib.suppress(TimeoutError):
                await inv.invoke_tool(_req(tool), ctx)

    asyncio.run(go())

    assert tool.executions == 2
    assert [r["decision"] for r in _log(store)] == ["failed", "failed"]


# ── the record must not imply ONE execution when N happened ────────────────
#
# `Audit` is a `BaseMiddleware`: one phase pair per chain invocation. With
# `retry()` inside it — the documented tool-chain order — three executions
# folded into a single `executed` record indistinguishable from a clean single
# call. That is the difference between "we charged the card" and "we charged
# the card three times".
#
# It still cannot write three records from one invocation. It can stop lying
# about the count: `retry()` publishes the attempt number on `call.meta` and
# the record carries it.


def test_a_retried_call_records_how_many_times_it_ran() -> None:
    """THE regression. One record, but an honest one."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=2)
    inv, ctx = _wire(store, audit(store=store), retry(max_attempts=3))

    asyncio.run(inv.invoke_tool(_req(tool), ctx))

    (record,) = _log(store)
    assert tool.executions == 3
    assert record["decision"] == "executed"
    assert record["attempts"] == 3, "one record covering three charges must say so"


def test_a_call_that_fails_every_attempt_records_the_count() -> None:
    """The worst case for an audit trail: N side effects and no success."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=99)
    inv, ctx = _wire(store, audit(store=store), retry(max_attempts=3))

    with pytest.raises(TimeoutError):
        asyncio.run(inv.invoke_tool(_req(tool), ctx))

    (record,) = _log(store)
    assert tool.executions == 3
    assert record["decision"] == "failed"
    assert record["attempts"] == 3


def test_a_call_with_no_retry_wired_reports_one_attempt() -> None:
    """POSITIVE CONTROL. Without `retry()` nothing stamps the count, and the
    default must be the truth (one run) rather than a missing key a reader has
    to interpret."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=0)
    inv, ctx = _wire(store, audit(store=store))

    asyncio.run(inv.invoke_tool(_req(tool), ctx))

    (record,) = _log(store)
    assert tool.executions == 1
    assert record["attempts"] == 1


def test_a_first_attempt_success_is_not_inflated() -> None:
    """POSITIVE CONTROL for the counter itself: `retry()` stamps on every pass
    through the loop, so a call that succeeds immediately must still say 1 —
    a stamp placed after the attempt would read 2."""
    store, tool = InMemoryStore(), _ChargingTool(fail_times=0)
    inv, ctx = _wire(store, audit(store=store), retry(max_attempts=5))

    asyncio.run(inv.invoke_tool(_req(tool), ctx))

    (record,) = _log(store)
    assert record["attempts"] == 1
