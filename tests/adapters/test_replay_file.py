"""FileReplayStore — round-trip, concurrency, failure REPORTING, env migration.

Two defect classes are pinned here.

1. Silent failure. The store's contract is that a write must not raise into
   the run, and it used to honour that by swallowing every exception with a
   bare ``except Exception: return``. A store whose root was unwritable, whose
   disk was full, or whose file on disk was corrupt behaved identically to one
   that was working perfectly — nothing logged, nothing counted. The tests
   below assert on captured log records and on the public counters, so "it
   reports the failure" is proven rather than assumed. They also pin the
   inverse: a genuine cache MISS is not a failure and must stay quiet.

2. ``RIO_*`` branding in a package called agentkit. The env var and default
   directory were renamed; the tests pin the compatibility rules (new name
   wins, old name keeps working with a once-only ``DeprecationWarning``, an
   existing legacy directory is adopted rather than orphaned).

Note for readers: this suite runs with ``filterwarnings = ["error"]``, so any
``DeprecationWarning`` that fires where a test does not explicitly expect one
fails that test. That is what makes the "emitted exactly once" assertions
below real — a second emission raises.
"""

import asyncio
import errno
import logging
import os
from pathlib import Path

import pytest

from agentkit.adapters.replay import FileReplayStore
from agentkit.adapters.replay import file as file_mod
from agentkit.kernel.replay import ReplayRecord

LOGGER_NAME = "agentkit.adapters.replay.file"


@pytest.fixture(autouse=True)
def _reset_migration_latches(monkeypatch):
    """Re-arm the process-global one-shot notices for every test.

    They are global on purpose (the condition they report is a property of
    the environment, not of a store instance), which means test order would
    otherwise decide whether a given test sees the notice at all.
    """
    # ``raising=False``: the latches do not exist in the pre-fix module, and
    # this fixture is autouse — without it every test in the file would error
    # in setup instead of failing on its own assertion, which would make the
    # "these tests catch the bug" check meaningless.
    monkeypatch.setattr(file_mod, "_LEGACY_ENV_USED_WARNED", False, raising=False)
    monkeypatch.setattr(file_mod, "_LEGACY_ENV_IGNORED_WARNED", False, raising=False)
    monkeypatch.setattr(file_mod, "_LEGACY_ROOT_WARNED", False, raising=False)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """No test may inherit a real operator's replay configuration."""
    monkeypatch.delenv("AGENTKIT_REPLAY_DIR", raising=False)
    monkeypatch.delenv("RIO_REPLAY_DIR", raising=False)


def _rec(span: str, **kw):
    kw.setdefault("operation", "chat")
    kw.setdefault("request", {})
    kw.setdefault("response", None)
    return ReplayRecord(span_id=span, **kw)


# ── positive controls: these pass BEFORE and AFTER the fix ───────────────────


@pytest.mark.asyncio
async def test_put_then_get_round_trips(tmp_path):
    """A record written with put() is readable with get()."""
    store = FileReplayStore(tmp_path)
    rec = ReplayRecord(
        span_id="a" * 16,
        operation="chat",
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"content": "hello"},
    )
    await store.put(rec)
    got = await store.get(rec.span_id)
    assert got is not None
    assert got.span_id == rec.span_id
    assert got.operation == "chat"
    assert got.request == {"messages": [{"role": "user", "content": "hi"}]}
    assert got.response == {"content": "hello"}


@pytest.mark.asyncio
async def test_miss_is_quiet_and_not_counted_as_a_failure(tmp_path, caplog):
    """A genuine miss returns None WITHOUT logging noise.

    This is the counterweight to every "it must log" test in this file: if
    the fix made misses warn, the first real failure would be buried under
    one line per lookup for spans that were simply never recorded.
    """
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert await store.get("b" * 16) is None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_directory_auto_created(tmp_path):
    """The store creates its root directory on first write."""
    nested = tmp_path / "does" / "not" / "exist"
    store = FileReplayStore(nested)
    rec = _rec("c" * 16)
    await store.put(rec)
    assert nested.exists()
    assert (nested / f"{rec.span_id}.json").exists()


@pytest.mark.asyncio
async def test_write_is_tmp_file_then_os_replace(tmp_path, monkeypatch):
    """The atomic-write idiom itself is pinned, not just its outcome.

    A refactor that wrote the final path directly would still pass the
    round-trip test while reintroducing torn reads for concurrent readers.
    """
    seen: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    store = FileReplayStore(tmp_path)
    rec = _rec("1" * 16)
    await store.put(rec)

    assert len(seen) == 1
    src, dst = seen[0]
    assert src.endswith(".tmp")
    assert os.path.dirname(src) == str(tmp_path)  # same dir → rename is atomic
    assert dst == str(tmp_path / f"{rec.span_id}.json")
    assert [p.name for p in tmp_path.iterdir()] == [f"{rec.span_id}.json"]


@pytest.mark.asyncio
async def test_concurrent_writes_atomic(tmp_path):
    """Two concurrent writers to the same span_id don't corrupt
    the file — the last write wins, no torn reads."""
    store = FileReplayStore(tmp_path)
    span = "d" * 16

    async def write(content: str):
        await store.put(_rec(span, request={"c": content}))

    await asyncio.gather(*(write(f"v{i}") for i in range(50)))
    got = await store.get(span)
    assert got is not None
    # The content is ONE of the writes — no merge, no corruption
    assert got.request["c"].startswith("v")
    # Every writer's tmp file was renamed away; none leaked.
    assert [p.name for p in tmp_path.iterdir()] == [f"{span}.json"]


@pytest.mark.asyncio
async def test_concurrent_readers_never_see_a_torn_file(tmp_path):
    """Readers interleaved with writers get a whole record or a miss."""
    store = FileReplayStore(tmp_path)
    span = "9" * 16
    payload = "x" * 50_000  # big enough that a non-atomic write would tear

    async def write(i: int):
        await store.put(_rec(span, request={"c": f"{i:02d}{payload}"}))

    async def read():
        got = await store.get(span)
        if got is not None:
            # Fixed-width prefix: any length but this one means a partial read.
            assert len(got.request["c"]) == len(payload) + 2

    await asyncio.gather(*(write(i) for i in range(20)), *(read() for _ in range(20)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",
        "../" * 8 + "escaped",
        "/etc/passwd",
        "..",
        "",
        "g" * 16,  # not hex
        "a" * 15,  # too short
        "a" * 33,  # too long
        "a" * 8 + "/" + "b" * 8,
        "a" * 16 + "\x00",
    ],
)
async def test_malicious_span_id_cannot_write_outside_root(tmp_path, bad):
    """The ``_VALID_SPAN_ID`` guard is the only thing standing between a
    caller-supplied string and an arbitrary file path. Pin it directly."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = FileReplayStore(root)

    await store.put(_rec(bad))  # must not raise

    assert list(outside.iterdir()) == []
    assert not root.exists() or list(root.iterdir()) == []
    assert await store.get(bad) is None


@pytest.mark.asyncio
async def test_dataclass_request_serializes_via_to_jsonable(tmp_path):
    """Dataclasses in request/response serialize via dataclasses.asdict."""
    from dataclasses import dataclass

    @dataclass
    class Req:
        prompt: str
        model: str

    store = FileReplayStore(tmp_path)
    rec = _rec("f" * 16, request=Req(prompt="hi", model="claude"))
    await store.put(rec)
    got = await store.get(rec.span_id)
    assert got is not None
    assert got.request == {"prompt": "hi", "model": "claude"}


@pytest.mark.asyncio
async def test_unserialisable_record_falls_back_to_repr_not_failure(tmp_path, caplog):
    """The documented ``default=str``-style fallback still holds.

    An object the JSON backend cannot encode must become its repr and the
    write must SUCCEED — this is not one of the failure paths, so it must
    not log a warning or bump the drop counter.
    """

    class Weird:
        def __repr__(self) -> str:
            return "<Weird obj>"

    class BadModel:
        def model_dump(self):  # Pydantic-shaped but broken
            raise RuntimeError("boom")

        def __repr__(self) -> str:
            return "<BadModel>"

    store = FileReplayStore(tmp_path)
    rec = _rec("2" * 16, request={"o": Weird(), "s": {1, 2}}, response=BadModel())
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        await store.put(rec)
    got = await store.get(rec.span_id)
    assert got is not None
    assert got.request["o"] == "<Weird obj>"
    assert got.response == "<BadModel>"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# ── defect 1: failures must be observable ────────────────────────────────────


@pytest.mark.asyncio
async def test_counters_are_zero_when_nothing_fails(tmp_path):
    """The counters must be trustworthy in the NEGATIVE direction too.

    A `dropped_writes` that ticked on a successful write, or a
    `failed_reads` that counted ordinary misses, would be as useless as the
    silence it replaces.
    """
    store = FileReplayStore(tmp_path)
    await store.put(_rec("0" * 16, request={"ok": True}))
    assert await store.get("0" * 16) is not None
    assert await store.get("c" * 16) is None  # a miss is not a failed read
    assert store.dropped_writes == 0
    assert store.failed_reads == 0


@pytest.mark.asyncio
async def test_write_failure_is_logged_and_counted(tmp_path, caplog, monkeypatch):
    """Disk-full-shaped failure: ENOSPC out of the rename step.

    Pre-fix this returned None with zero trace of the loss anywhere.
    """

    def full_disk(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        await store.put(_rec("3" * 16))  # must not raise

    warnings_seen = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings_seen) == 1
    msg = warnings_seen[0].getMessage()
    assert "No space left on device" in msg
    assert f"{'3' * 16}.json" in msg
    assert store.dropped_writes == 1
    # The tmp file from the failed write must not be left behind as litter.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
async def test_unwritable_root_is_logged_and_counted(tmp_path, caplog):
    """An operator-level misconfiguration (read-only data dir)."""
    root = tmp_path / "ro"
    root.mkdir()
    root.chmod(0o500)
    try:
        store = FileReplayStore(root)
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            await store.put(_rec("4" * 16))  # must not raise
        warnings_seen = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings_seen) == 1
        assert "PermissionError" in warnings_seen[0].getMessage()
        assert store.dropped_writes == 1
    finally:
        root.chmod(0o700)


@pytest.mark.asyncio
async def test_repeated_write_failures_log_once_but_count_every_time(tmp_path, caplog, monkeypatch):
    """The anti-spam rule is itself a requirement.

    A full disk fails EVERY put for the rest of the run; one line per call
    would drown whatever the operator was reading. One line, accurate count.
    """

    def full_disk(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        for i in range(25):
            await store.put(_rec(f"{i:016x}"))

    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert store.dropped_writes == 25


@pytest.mark.asyncio
async def test_invalid_span_id_on_put_is_reported_as_a_caller_bug(tmp_path, caplog):
    """A rejected span_id silently drops data; it must not stay silent.

    Reported at ERROR rather than WARNING because, unlike a full disk, no
    machine state produces this — an OTel span id always matches the guard,
    so reaching here means the CALLER built the record wrong.
    """
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        await store.put(_rec("../../etc/passwd"))

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "span_id" in errors[0].getMessage()
    assert store.dropped_writes == 1
    assert not tmp_path.joinpath("etc").exists()


@pytest.mark.asyncio
async def test_corrupt_record_on_disk_is_logged_and_counted(tmp_path, caplog):
    """A file that exists but isn't valid JSON is corruption, not a miss."""
    span = "5" * 16
    (tmp_path / f"{span}.json").write_text("{not json at all")
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert await store.get(span) is None

    warnings_seen = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings_seen) == 1
    assert f"{span}.json" in warnings_seen[0].getMessage()
    assert store.failed_reads == 1


@pytest.mark.asyncio
async def test_wrong_shape_json_returns_none_instead_of_raising(tmp_path, caplog):
    """Valid JSON of the wrong SHAPE used to raise KeyError out of get().

    ``_record_from_dict`` sat outside the decode guard, so the one call in
    this module that could still blow up at a caller was a read of a foreign
    file that happened to be named like a span id.
    """
    span = "6" * 16
    (tmp_path / f"{span}.json").write_text('["not", "a", "record"]')
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert await store.get(span) is None  # must not raise

    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert store.failed_reads == 1


@pytest.mark.asyncio
async def test_unreadable_existing_record_is_distinguished_from_a_miss(tmp_path, caplog):
    """A directory where a record should be: exists, cannot be read."""
    span = "7" * 16
    (tmp_path / f"{span}.json").mkdir()
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert await store.get(span) is None

    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert store.failed_reads == 1


@pytest.mark.asyncio
async def test_write_and_read_failures_are_reported_independently(tmp_path, caplog, monkeypatch):
    """Warn-once is keyed per failure CLASS.

    A single global latch would let an early write failure permanently mask
    every later read failure — the same silence, one indirection down.
    """
    span = "8" * 16
    (tmp_path / f"{span}.json").write_text("{corrupt")

    def full_disk(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    store = FileReplayStore(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        await store.put(_rec("a" * 16))
        assert await store.get(span) is None

    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 2
    assert store.dropped_writes == 1
    assert store.failed_reads == 1


# ── defect 2: RIO_* → AGENTKIT_* migration ───────────────────────────────────


@pytest.mark.asyncio
async def test_from_env_returns_none_when_unset():
    assert FileReplayStore.from_env() is None


@pytest.mark.asyncio
async def test_from_env_reads_the_agentkit_var(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTKIT_REPLAY_DIR", str(tmp_path))
    store = FileReplayStore.from_env()
    assert store is not None
    rec = _rec("e" * 16, request={"x": 1})
    await store.put(rec)
    assert (tmp_path / f"{rec.span_id}.json").exists()


@pytest.mark.asyncio
async def test_legacy_env_var_still_works_and_warns(monkeypatch, tmp_path):
    """Backwards compatibility is not optional — but it is announced."""
    monkeypatch.setenv("RIO_REPLAY_DIR", str(tmp_path))
    with pytest.warns(DeprecationWarning, match="AGENTKIT_REPLAY_DIR"):
        store = FileReplayStore.from_env()
    assert store is not None
    assert store.root == tmp_path.resolve()


def test_legacy_env_deprecation_is_emitted_exactly_once(monkeypatch, tmp_path):
    """Once per process, not once per call.

    Python's own ``__warningregistry__`` de-dup is disabled under
    ``-W error`` / ``-W always`` — precisely the configurations a warning
    matters in — so the module must latch it explicitly. The suite runs with
    ``filterwarnings = ["error"]``, so a second emission raises here.
    """
    monkeypatch.setenv("RIO_REPLAY_DIR", str(tmp_path))
    with pytest.warns(DeprecationWarning):
        FileReplayStore.from_env()
    for _ in range(5):
        assert FileReplayStore.from_env() is not None  # no second warning → no error


def test_new_env_var_wins_over_legacy(monkeypatch, tmp_path, caplog):
    """Both set: the new name takes effect, the stale one is reported.

    Not a DeprecationWarning — nothing deprecated is being USED — but an
    ignored config value is exactly what an operator needs told once.
    """
    new = tmp_path / "new"
    old = tmp_path / "old"
    monkeypatch.setenv("AGENTKIT_REPLAY_DIR", str(new))
    monkeypatch.setenv("RIO_REPLAY_DIR", str(old))
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        store = FileReplayStore.from_env()
    assert store is not None
    assert store.root == new.resolve()
    warnings_seen = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings_seen) == 1
    assert "RIO_REPLAY_DIR" in warnings_seen[0].getMessage()


@pytest.mark.parametrize(
    ("new", "old", "expect"),
    [
        ("", "", None),
        ("", None, None),
        (None, "", None),
        ("", "legacy", "legacy"),
    ],
)
def test_empty_env_values_count_as_unset(monkeypatch, tmp_path, new, old, expect):
    """``FOO=`` is how a shell profile spells "not configured".

    Treating it as a path would resolve to the process CWD and quietly
    scatter replay JSON through whatever directory the app was started in.
    """
    for name, value in (("AGENTKIT_REPLAY_DIR", new), ("RIO_REPLAY_DIR", old)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(tmp_path / value) if value else "")

    if expect is None:
        assert FileReplayStore.from_env() is None
    else:
        with pytest.warns(DeprecationWarning):
            store = FileReplayStore.from_env()
        assert store is not None
        assert store.root == (tmp_path / expect).resolve()


def test_default_root_is_agentkit_branded(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert FileReplayStore.default().root == (tmp_path / "agentkit" / "replays").resolve()


def test_default_root_without_xdg_uses_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(file_mod.Path, "home", classmethod(lambda cls: tmp_path))
    expected = tmp_path / ".local" / "share" / "agentkit" / "replays"
    assert FileReplayStore.default().root == expected.resolve()
    assert "rio" not in str(FileReplayStore.default().root)


def test_default_adopts_existing_legacy_dir_rather_than_orphaning_it(monkeypatch, tmp_path, caplog):
    """An upgrading user's already-written replays must stay reachable.

    Silently switching to the new path would leave the store looking healthy
    while every historical lookup missed — the same class of "reports success
    while broken" bug this module is being fixed for. The library does not
    move the files itself; it keeps reading where they are and tells the
    operator how to migrate.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    legacy = tmp_path / "rio" / "replays"
    legacy.mkdir(parents=True)
    (legacy / f"{'a' * 16}.json").write_text("{}")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        store = FileReplayStore.default()

    assert store.root == legacy.resolve()
    warnings_seen = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings_seen) == 1
    assert "agentkit" in warnings_seen[0].getMessage()


def test_legacy_root_notice_is_emitted_once(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    (tmp_path / "rio" / "replays").mkdir(parents=True)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        for _ in range(5):
            FileReplayStore.default()
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


def test_default_prefers_new_root_once_it_exists(monkeypatch, tmp_path, caplog):
    """Self-healing: after the operator migrates, the legacy path is dead."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    (tmp_path / "rio" / "replays").mkdir(parents=True)
    new = tmp_path / "agentkit" / "replays"
    new.mkdir(parents=True)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        store = FileReplayStore.default()

    assert store.root == new.resolve()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_default_uses_new_root_on_a_clean_machine(monkeypatch, tmp_path, caplog):
    """No legacy directory → no notice, no legacy path."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        store = FileReplayStore.default()
    assert store.root == (tmp_path / "agentkit" / "replays").resolve()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_adopted_legacy_root_actually_round_trips(monkeypatch, tmp_path):
    """The whole point of adoption: old records are readable, new ones land
    next to them instead of in a directory nobody is looking at."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    legacy = tmp_path / "rio" / "replays"
    legacy.mkdir(parents=True)
    seeded = FileReplayStore(legacy)
    await seeded.put(_rec("a" * 16, request={"old": True}))

    store = FileReplayStore.default()
    got = await store.get("a" * 16)
    assert got is not None and got.request == {"old": True}

    await store.put(_rec("b" * 16, request={"new": True}))
    assert (legacy / f"{'b' * 16}.json").exists()
    assert not (tmp_path / "agentkit").exists()


def test_the_deprecation_and_the_ignored_notice_do_not_silence_each_other(caplog, monkeypatch):
    """Two conditions, two latches — not one latch for the module.

    These warnings answer different questions for different audiences: "you are
    using a deprecated name" is about the CALLER's code, while "your leftover
    stale name is being ignored" is about the OPERATOR's environment. A single
    shared latch means whichever fires first permanently silences the other.

    Measured with the shared latch that this test was written against: a
    process that starts with only the legacy var set (firing the
    DeprecationWarning), then later sees both set, is NEVER told that its stale
    var is being ignored — `step 2 (both set) -> SILENT`. That is the same
    swallow-the-second-class mistake `_warn_once` exists to avoid.
    """
    # Step 1: only the legacy var — the deprecation fires and trips its latch.
    monkeypatch.setenv(file_mod.LEGACY_ENV_REPLAY_DIR, "/tmp/legacy-replays")
    with pytest.deprecated_call():
        file_mod.FileReplayStore.from_env()

    # Step 2: now BOTH are set. The operator must still learn that the stale
    # one is being ignored, even though the deprecation already fired.
    monkeypatch.setenv(file_mod.ENV_REPLAY_DIR, "/tmp/new-replays")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=file_mod.logger.name):
        store = file_mod.FileReplayStore.from_env()

    assert any("IGNORING" in r.getMessage() for r in caplog.records), (
        "the deprecation latch swallowed the ignored-variable notice; the "
        "operator is never told their stale RIO_REPLAY_DIR does nothing"
    )
    assert store is not None and store._root == Path("/tmp/new-replays").resolve()


def test_each_legacy_notice_still_fires_only_once(caplog, monkeypatch):
    """POSITIVE CONTROL for the test above. Splitting one latch into two must
    not turn either warning into a per-call flood — the whole reason the latch
    exists is that a store is often constructed per run."""
    monkeypatch.setenv(file_mod.ENV_REPLAY_DIR, "/tmp/new-replays")
    monkeypatch.setenv(file_mod.LEGACY_ENV_REPLAY_DIR, "/tmp/legacy-replays")
    with caplog.at_level(logging.WARNING, logger=file_mod.logger.name):
        for _ in range(5):
            file_mod.FileReplayStore.from_env()
    ignored = [r for r in caplog.records if "IGNORING" in r.getMessage()]
    assert len(ignored) == 1, f"expected exactly one notice across 5 calls, got {len(ignored)}"
