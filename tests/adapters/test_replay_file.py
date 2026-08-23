"""FileReplayStore — round-trip, concurrency, failure REPORTING, configuration.

The defect class pinned here is silent failure. The store's contract is that a
write must not raise into the run, and it used to honour that by swallowing
every exception with a bare ``except Exception: return``. A store whose root
was unwritable, whose disk was full, or whose file on disk was corrupt behaved
identically to one that was working perfectly — nothing logged, nothing
counted. The tests below assert on captured log records and on the public
counters, so "it reports the failure" is proven rather than assumed. They also
pin the inverse: a genuine cache MISS is not a failure and must stay quiet.

Configuration is deliberately small and is pinned as such: one env var
(``AGENTKIT_REPLAY_DIR``, empty meaning unset) and one default path derived
from ``XDG_DATA_HOME``. There is no fallback env var and no directory probing,
so ``default()`` is a pure function of the environment — the tests at the
bottom pin that a leftover ``RIO_*`` name or a pre-rename directory changes
nothing.

Note for readers: this suite runs with ``filterwarnings = ["error"]``, so any
warning that fires where a test does not explicitly expect one fails that
test. That is what makes "the legacy name is silently ignored" a real
assertion rather than a hopeful one.
"""

import asyncio
import errno
import logging
import os

import pytest

from agentkit.adapters.replay import FileReplayStore
from agentkit.adapters.replay import file as file_mod
from agentkit.kernel.replay import ReplayRecord

LOGGER_NAME = "agentkit.adapters.replay.file"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """No test may inherit a real operator's replay configuration."""
    monkeypatch.delenv("AGENTKIT_REPLAY_DIR", raising=False)


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


# ── configuration: one env var, one default path ─────────────────────────────


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


def test_pre_rename_env_var_is_ignored_entirely(monkeypatch, tmp_path, caplog):
    """``RIO_REPLAY_DIR`` is not a name this package knows any more.

    It used to be read as a deprecated fallback. Now it is just another
    variable in the environment: ``from_env()`` reports "not configured" and
    the caller falls back to ``NoopReplayStore``, exactly as it would on a
    machine where nothing was set at all. Pinned because "silently ignored"
    is only correct if it is also QUIET — no ``DeprecationWarning`` (which
    ``filterwarnings = ["error"]`` would turn into a failure here) and no log
    line for a variable this package has no opinion about.
    """
    monkeypatch.setenv("RIO_REPLAY_DIR", str(tmp_path))
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert FileReplayStore.from_env() is None
    assert caplog.records == []


def test_agentkit_var_is_used_even_with_the_pre_rename_one_set(monkeypatch, tmp_path, caplog):
    """A leftover ``RIO_REPLAY_DIR`` export cannot influence the outcome.

    There is no precedence rule left to get wrong — the new name is not
    "winning" a contest, it is the only name read — so this must be
    indistinguishable from the case where the stale var was never exported.
    """
    new = tmp_path / "new"
    monkeypatch.setenv("AGENTKIT_REPLAY_DIR", str(new))
    monkeypatch.setenv("RIO_REPLAY_DIR", str(tmp_path / "old"))
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        store = FileReplayStore.from_env()
    assert store is not None
    assert store.root == new.resolve()
    assert caplog.records == []


@pytest.mark.parametrize("value", ["", None])
def test_empty_env_value_counts_as_unset(monkeypatch, value):
    """``FOO=`` is how a shell profile spells "not configured".

    Treating it as a path would resolve to the process CWD and quietly
    scatter replay JSON through whatever directory the app was started in.
    ``None`` here is the unset control: both spellings must behave the same.
    """
    if value is None:
        monkeypatch.delenv("AGENTKIT_REPLAY_DIR", raising=False)
    else:
        monkeypatch.setenv("AGENTKIT_REPLAY_DIR", value)
    assert FileReplayStore.from_env() is None


def test_default_root_is_agentkit_branded(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert FileReplayStore.default().root == (tmp_path / "agentkit" / "replays").resolve()


def test_default_root_without_xdg_uses_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(file_mod.Path, "home", classmethod(lambda cls: tmp_path))
    expected = tmp_path / ".local" / "share" / "agentkit" / "replays"
    assert FileReplayStore.default().root == expected.resolve()
    assert "rio" not in str(FileReplayStore.default().root)


def test_default_ignores_a_pre_rename_directory(monkeypatch, tmp_path, caplog):
    """``default()`` probes nothing on disk.

    An existing ``$XDG_DATA_HOME/rio/replays`` used to be ADOPTED as the root
    when the agentkit directory did not exist yet. That branch is gone, so the
    returned path is a pure function of ``XDG_DATA_HOME`` — the same answer
    whether or not a pre-rename directory happens to be sitting there, and
    with nothing logged about it.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    (tmp_path / "rio" / "replays").mkdir(parents=True)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        store = FileReplayStore.default()
    assert store.root == (tmp_path / "agentkit" / "replays").resolve()
    assert caplog.records == []


def test_default_uses_new_root_on_a_clean_machine(monkeypatch, tmp_path, caplog):
    """Nothing on disk, nothing in the env → the conventional path, quietly."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        store = FileReplayStore.default()
    assert store.root == (tmp_path / "agentkit" / "replays").resolve()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
