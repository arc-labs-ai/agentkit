"""Durable event errors must fail closed without corrupting later publication."""

import json
import sqlite3
from contextlib import closing

import pytest

from agentkit.core.events import EventHub
from agentkit.models.contracts.types import CallEvent
from agentkit.state.events import SQLiteEventJournal


@pytest.mark.asyncio
async def test_failed_insert_rolls_back_and_hub_can_retry_the_same_sequence(tmp_path):
    path = tmp_path / "journal.db"
    journal = SQLiteEventJournal(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TRIGGER reject_write BEFORE INSERT ON agent_events BEGIN SELECT RAISE(ABORT,'disk gate'); END")
    delivered = []

    async def handler(event):
        delivered.append(event.event_id)

    async with EventHub(journal=journal) as hub:
        hub.subscribe("record", handler)
        with pytest.raises(sqlite3.IntegrityError, match="disk gate"):
            await hub.publish(CallEvent("started", "run", sequence=1))
        assert delivered == [] and await journal.read(call_id="run") == ()
        with closing(sqlite3.connect(path)) as db:
            db.execute("DROP TRIGGER reject_write")
        await hub.publish(CallEvent("started", "run", sequence=1))
    assert delivered == ["run:1"]
    assert [event.sequence for event in await journal.read(call_id="run")] == [1]


@pytest.mark.asyncio
async def test_corrupt_persisted_payload_is_reported_instead_of_empty_recovery(tmp_path):
    path = tmp_path / "journal.db"
    journal = SQLiteEventJournal(path)
    await journal.append(CallEvent("started", "run", sequence=1))
    with closing(sqlite3.connect(path)) as db:
        with db:
            db.execute("UPDATE agent_events SET payload='broken-json' WHERE call_id='run'")
    with pytest.raises(json.JSONDecodeError):
        await SQLiteEventJournal(path).read(call_id="run")


@pytest.mark.asyncio
async def test_equal_sequences_are_independent_across_call_and_child_streams(tmp_path):
    journal = SQLiteEventJournal(tmp_path / "journal.db")
    parent = CallEvent("started", "parent", {"nested": {"values": [1]}}, sequence=1)
    child = CallEvent("reaction", "child", sequence=1, parent_call_id="parent")
    await journal.append(parent)
    await journal.append(child)
    await journal.append(parent)
    loaded = (await journal.read(call_id="child"))[0]
    assert loaded.parent_call_id == "parent" and loaded.event_id == "child:1"
    assert len(await journal.read(call_id="parent")) == 1
    assert await journal.read(call_id="missing") == ()
    with pytest.raises(TypeError):
        (await journal.read(call_id="parent"))[0].fields["nested"]["values"].append(2)


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [True, 0, -1, 1.5, "1"])
async def test_invalid_sequence_never_enters_journal(tmp_path, sequence):
    journal = SQLiteEventJournal(tmp_path / "journal.db")
    with pytest.raises(ValueError, match="positive explicit"):
        await journal.append(CallEvent("started", "run", sequence=sequence))
    assert await journal.read(call_id="run") == ()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_busy_timeout_never_creates_the_journal_database(tmp_path, timeout):
    path = tmp_path / "journal.db"
    with pytest.raises(ValueError, match="positive"):
        SQLiteEventJournal(path, busy_timeout_s=timeout)
    assert not path.exists()
