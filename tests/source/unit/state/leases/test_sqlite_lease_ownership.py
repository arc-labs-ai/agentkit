"""Lease transitions use holder identity and fencing, not owner-name shortcuts."""

import asyncio
from dataclasses import replace

import pytest

from agentkit.state.leases import SQLiteLeaseStore


@pytest.mark.asyncio
async def test_renewal_changes_expiry_without_replacing_holder_and_expiry_cannot_resurrect(tmp_path, monkeypatch):
    import agentkit.state.leases.sqlite as adapter

    now = [100.0]
    monkeypatch.setattr(adapter.time, "time", lambda: now[0])
    first, second = SQLiteLeaseStore(tmp_path / "leases.db"), SQLiteLeaseStore(tmp_path / "leases.db")
    original = await first.acquire("work", "worker", ttl_s=10)
    now[0] = 105
    renewed = await second.renew(original, ttl_s=10)
    assert renewed.lease_id == original.lease_id and renewed.fencing_token == original.fencing_token
    now[0] = 111
    assert await first.acquire("work", "other", ttl_s=10) is None
    now[0] = 115
    assert await second.renew(renewed, ttl_s=10) is None
    replacement = await second.acquire("work", "worker", ttl_s=10)
    assert replacement.lease_id != original.lease_id and replacement.fencing_token > original.fencing_token
    assert not await first.release(original)
    assert await first.renew(original, ttl_s=10) is None
    assert await first.acquire("work", "third", ttl_s=10) is None
    assert await first.release(replacement)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("owner", "impostor"), ("resource", "other"),
                                        ("lease_id", "forged-id"), ("fencing_token", 999)])
async def test_forged_holder_identity_cannot_renew_or_release_another_live_lease(tmp_path, field, value):
    store = SQLiteLeaseStore(tmp_path / "leases.db")
    lease = await store.acquire("work", "worker", ttl_s=30)
    forged = replace(lease, **{field: value})
    assert await store.renew(forged, ttl_s=30) is None
    assert not await store.release(forged)
    assert await store.acquire("work", "other", ttl_s=30) is None
    assert await store.release(lease)


@pytest.mark.asyncio
async def test_same_owner_parallel_acquisition_cannot_bypass_resource_capacity(tmp_path):
    path = tmp_path / "leases.db"
    stores = [SQLiteLeaseStore(path) for _ in range(8)]
    results = await asyncio.gather(*(store.acquire("work", "same-owner", ttl_s=30, capacity=3) for store in stores))
    held = [lease for lease in results if lease is not None]
    assert len(held) == 3 and len({lease.lease_id for lease in held}) == 3
    assert sorted(lease.fencing_token for lease in held) == [1, 2, 3]
    await asyncio.gather(*(stores[0].release(lease) for lease in held))
    next_lease = await stores[0].acquire("work", "next", ttl_s=30, capacity=3)
    assert next_lease.fencing_token == 4


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_busy_timeout_is_refused_before_creating_a_database(tmp_path, timeout):
    path = tmp_path / "leases.db"
    with pytest.raises(ValueError, match="positive"):
        SQLiteLeaseStore(path, busy_timeout_s=timeout)
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource,owner,capacity", [
    ("", "worker", 1), ("work", "", 1), ("work", "worker", 0),
    ("work", "worker", -1), ("work", "worker", True), ("work", "worker", 1.5),
])
async def test_invalid_admission_identity_or_capacity_does_not_establish_resource_policy(
    tmp_path, resource, owner, capacity,
):
    store = SQLiteLeaseStore(tmp_path / "leases.db")
    with pytest.raises(ValueError, match="nonempty.*positive integer"):
        await store.acquire(resource, owner, ttl_s=30, capacity=capacity)
    # The malformed attempt must not pin an incompatible resource policy.
    admitted = await store.acquire("work", "valid-worker", ttl_s=30, capacity=2)
    assert admitted is not None and admitted.fencing_token == 1
