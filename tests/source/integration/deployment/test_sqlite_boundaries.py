"""Real restart and process-boundary acceptance, without a model dependency."""

import asyncio
import multiprocessing
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import asdict, replace

import pytest

from agentkit.core.events import EventDeliveryError, EventHub
from agentkit.models.contracts.types import CallEvent
from agentkit.state.events import SQLiteEventJournal
from agentkit.state.leases import SQLiteLeaseStore


def _lease_worker(path, owner, ready, release, output):
    async def work():
        await asyncio.to_thread(ready.wait, 5)
        lease = await SQLiteLeaseStore(path).acquire("workers", owner, ttl_s=10, capacity=2)
        output.put(asdict(lease) if lease else None)
        await asyncio.to_thread(release.wait, 5)
        if lease:
            await SQLiteLeaseStore(path).release(lease)

    asyncio.run(work())


@pytest.mark.asyncio
async def test_event_journal_survives_process_restart_and_preserves_cursor_order(tmp_path):
    path = tmp_path / "events.db"
    script = """
import asyncio,sys
from agentkit.state.events import SQLiteEventJournal
from agentkit.models.contracts.types import CallEvent
async def work():
    journal=SQLiteEventJournal(sys.argv[1])
    await journal.append(CallEvent('tool_result','run',{'nested': {'values':[1,2]}},sequence=1,timestamp='first'))
    await journal.append(CallEvent('call_finished','run',{'outcome':'answered'},sequence=3,timestamp='second'))
asyncio.run(work())
"""
    result = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    journal = SQLiteEventJournal(path)
    events = await journal.read(call_id="run")
    assert [event.sequence for event in events] == [1, 3]
    assert events[0].fields["nested"]["values"] == [1, 2]
    assert [event.event_id for event in await journal.read(call_id="run", after_sequence=1)] == ["run:3"]
    await journal.append(events[0])  # an exact retry remains one stored event
    with pytest.raises(ValueError, match="different content"):
        await journal.append(replace(events[0], fields={"corrupted": True}))
    with pytest.raises(ValueError, match="increase"):
        await journal.append(CallEvent("late", "run", sequence=2, timestamp="different"))
    assert len(await journal.read(call_id="run")) == 2


@pytest.mark.asyncio
async def test_journal_is_durable_before_handler_failure_and_restart_replay_is_host_owned(tmp_path):
    path = tmp_path / "events.db"
    journal = SQLiteEventJournal(path)
    hub = EventHub(journal=journal)

    async def fail(event):
        # The durable write must already be visible while dispatch executes.
        assert [entry.event_id for entry in await SQLiteEventJournal(path).read(call_id="run")] == ["run:1"]
        raise RuntimeError("consumer unavailable")

    hub.subscribe("consumer", fail)
    with pytest.raises(EventDeliveryError):
        await hub.publish(CallEvent("started", "run", sequence=1))
    with pytest.raises(ExceptionGroup) as failures:
        await hub.aclose()
    assert all(isinstance(error, EventDeliveryError) for error in failures.value.exceptions)
    delivered = []

    async def record(event):
        delivered.append(event.event_id)

    async with EventHub(journal=SQLiteEventJournal(path)) as restarted:
        restarted.subscribe("consumer", record)
        for event in await journal.read(call_id="run"):
            await restarted.publish(event)
        await restarted.publish(CallEvent("finished", "run", sequence=2))
    assert delivered == ["run:1", "run:2"]
    assert [event.sequence for event in await journal.read(call_id="run")] == [1, 2]


def test_lease_capacity_is_atomic_across_four_real_processes(tmp_path):
    path = str(tmp_path / "leases.db")
    SQLiteLeaseStore(path)
    context = multiprocessing.get_context("spawn")
    ready, release, output = context.Event(), context.Event(), context.Queue()
    workers = [
        context.Process(target=_lease_worker, args=(path, f"worker-{index}", ready, release, output))
        for index in range(4)
    ]
    try:
        for worker in workers:
            worker.start()
        ready.set()
        results = [output.get(timeout=10) for _ in workers]
        leases = [entry for entry in results if entry]
        assert len(leases) == 2 and len({entry["fencing_token"] for entry in leases}) == 2
        with closing(sqlite3.connect(path)) as db:
            assert db.execute("SELECT COUNT(*) FROM admission_leases").fetchone()[0] == 2
    finally:
        release.set()
        for worker in workers:
            worker.join(10)
            if worker.is_alive():
                worker.kill()
                worker.join()
        output.close()
        output.join_thread()
    assert all(worker.exitcode == 0 for worker in workers)
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT COUNT(*) FROM admission_leases").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_expired_lease_cannot_renew_release_replacement_or_pass_host_fencing(tmp_path):
    path = tmp_path / "leases.db"
    old = await SQLiteLeaseStore(path).acquire("payments", "old-process", ttl_s=0.08)
    assert old is not None
    await asyncio.sleep(0.1)
    current = await SQLiteLeaseStore(path).acquire("payments", "replacement", ttl_s=2)
    assert current is not None and current.fencing_token > old.fencing_token
    assert await SQLiteLeaseStore(path).renew(old, ttl_s=2) is None
    assert await SQLiteLeaseStore(path).release(old) is False  # expired row removed by acquire
    assert await SQLiteLeaseStore(path).acquire("payments", "third", ttl_s=2) is None
    renewed = await SQLiteLeaseStore(path).renew(current, ttl_s=2)
    assert renewed is not None and renewed.lease_id == current.lease_id
    assert renewed.fencing_token == current.fencing_token
    # Example host effect boundary: atomically reject a stale fencing token.
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE protected_effect (id INTEGER PRIMARY KEY, fence INTEGER, value TEXT)")
        db.execute("INSERT INTO protected_effect VALUES(1, ?, 'replacement')", (current.fencing_token,))
        changed = db.execute(
            "UPDATE protected_effect SET value='stale' WHERE id=1 AND fence<?", (old.fencing_token,)
        ).rowcount
        assert changed == 0
        assert db.execute("SELECT value FROM protected_effect").fetchone()[0] == "replacement"
    assert await SQLiteLeaseStore(path).release(current)
    assert not await SQLiteLeaseStore(path).release(current)
    with pytest.raises(ValueError, match="capacity differs"):
        await SQLiteLeaseStore(path).acquire("payments", "third", ttl_s=2, capacity=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -1, float("nan"), float("inf")])
async def test_lease_rejects_unbounded_or_nonpositive_ttl_before_acquiring(tmp_path, ttl):
    store = SQLiteLeaseStore(tmp_path / "leases.db")
    with pytest.raises(ValueError, match="finite and positive"):
        await store.acquire("resource", "owner", ttl_s=ttl)


@pytest.mark.asyncio
async def test_invalid_journal_cursor_sequence_and_in_memory_adapters_are_rejected(tmp_path):
    journal = SQLiteEventJournal(tmp_path / "events.db")
    with pytest.raises(ValueError, match="positive"):
        await journal.append(CallEvent("started", "run"))
    with pytest.raises(ValueError, match="negative"):
        await journal.read(call_id="run", after_sequence=-1)
    for adapter in (SQLiteEventJournal, SQLiteLeaseStore):
        with pytest.raises(ValueError, match="filesystem"):
            adapter(":memory:")
