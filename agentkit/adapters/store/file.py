"""`FileStore` — the zero-dependency durable `StorePort` (JSON files; survives a process restart)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from agentkit.adapters.store._keylock import KeyLock, key_lock
from agentkit.adapters.store._primitives import (
    check_by,
    check_limit,
    is_counter,
    not_a_counter,
)
from agentkit.kernel.errors import StoreUnavailable


class _NeverJSONError(Exception):
    """Never raised. Exists so the mutation catalogue can neutralise an
    ``except json.JSONDecodeError`` clause by retargeting it, producing a real
    behavioural change rather than a NameError any test would "catch" for the
    wrong reason."""


class FileStore:
    """Durable `StorePort` backed by JSON files under `base_dir` — survives a process restart, so a
    human-gate suspend or a crashed run resumes from disk. Zero-dependency (stdlib `json`/`pathlib`) and
    async-first: the blocking file I/O is bridged via `asyncio.to_thread` so it never stalls the loop.

    The reference durable adapter; a Postgres/Redis `StorePort` is the same shape behind an extra. Values
    must be JSON-serializable (which the loop/workflow checkpoints are). Single-flight is in-process (an
    `asyncio.Lock` per key); cross-process single-flight needs the real DB's transaction — documented.
    `ttl` is accepted but not enforced (no expiry sweeper); use a TTL-native backend for that.
    """

    def __init__(self, base_dir: str) -> None:
        self._root = Path(base_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        # Reference-counted, like the other three backends. A plain
        # ``setdefault`` table is write-only: it keeps one lock per key the
        # process has ever touched, which is the exact regression
        # ``_keylock`` was extracted to fix. FileStore was the one backend
        # left out of that fix, and it was invisible because it was also the
        # one backend missing from the reclaim test.
        self._locks: dict[str, KeyLock] = {}
        # One-shot latches: a TTL that silently does nothing, and a log line
        # that could not be parsed, each deserve exactly one warning per store
        # rather than one per call.
        self._warned_ttl = False
        self._warned_corrupt_log = False

    def _safe(self, key: str) -> str:
        # percent-encode every non-alphanumeric byte → reversible and collision-free (so `a/b`, `a\b`,
        # `a:b` map to distinct filenames). `safe=""` encodes `/` and `:` too.
        return quote(key, safe="")

    def _kv_path(self, key: str) -> Path:
        return self._root / f"{self._safe(key)}.json"

    def _log_path(self, key: str) -> Path:
        return self._root / f"{self._safe(key)}.log"

    async def get(self, key: str) -> Any | None:
        path = self._kv_path(key)

        def _read() -> Any:
            if not path.exists():
                return None
            raw = path.read_text()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                # Writes go through `_write_atomic`, so this is not a torn
                # write from this process — it is external corruption (a disk
                # fault, a hand-edit, an older build that wrote
                # non-atomically). Re-raise rather than returning None, because
                # silently reporting "no checkpoint" would restart a run that
                # actually has durable state. But name the FILE: a bare
                # JSONDecodeError from a `to_thread` frame gives an operator
                # nothing to act on.
                raise StoreUnavailable(
                    f"FileStore entry for key {key!r} is not valid JSON: {path} "
                    f"({exc}). Inspect or delete the file; it will not repair itself."
                ) from exc

        return await asyncio.to_thread(_read)

    def _warn_ttl_ignored(self, ttl: int | None) -> None:
        """One warning per store, shared by every method that takes a ``ttl``.

        Extracted when `compare_and_set` and `increment` arrived: three copies
        of the latch would have meant three warnings from one store, and the
        one-shot behaviour is itself tested (a per-call warning gets filtered
        and stops being read).

        ``stacklevel=3`` — one for this helper, one for the public method,
        landing on the caller. The inlined version used 2 for the same reason;
        pointing the warning inside the adapter tells a reader nothing.

        The condition is kept in its original positive form rather than flipped
        into an early return: the mutation catalogue neutralises exactly this
        line to check the warning is still asserted somewhere, and a rewritten
        guard would make that mutant silently inapplicable — a mutant that no
        longer applies reads as "invariant enforced" while enforcing nothing.
        """
        if ttl is not None and not self._warned_ttl:
            self._warned_ttl = True
            warnings.warn(
                "FileStore ignores ttl — there is no expiry sweeper, so this entry is PERMANENT. "
                "That matters most for idempotency: an `idempotent()` key that never expires "
                "dedupes a legitimate retry of the same operation forever. Use a TTL-native "
                "backend (RedisStore) when expiry is load-bearing.",
                UserWarning,
                stacklevel=3,
            )

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        self._warn_ttl_ignored(ttl)
        path = self._kv_path(key)
        await asyncio.to_thread(self._write_atomic, path, json.dumps(value))

    @staticmethod
    def _write_atomic(path: Path, payload: str) -> None:
        """Write via a temp file in the SAME directory, then ``os.replace``.

        A plain ``path.write_text`` is not atomic, and this adapter exists to
        "survive a process restart, so a human-gate suspend or a crashed run
        resumes from disk". A crash DURING the write left a truncated file, and
        every later ``get`` raised ``JSONDecodeError`` — the checkpoint became
        permanently unreadable and the run could never resume. The failure mode
        the adapter is for was the one that broke it.

        ``os.replace`` is atomic on POSIX and Windows, so a reader sees either
        the whole old file or the whole new one, never a torn one. Same
        directory because a rename across filesystems is not atomic.

        Note the remaining caveat, which is not addressed here: without
        ``fsync`` the bytes may still be in the OS page cache, so this
        guarantees survival of a PROCESS crash, not of a power loss. That is
        the claim the docstring makes, and paying an fsync per write for a
        stronger one is a decision for whoever needs it.
        """
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp, path)
        except BaseException:
            # Never leave the scratch file behind — a directory slowly filling
            # with `.tmp` debris is its own operational problem.
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    async def delete(self, key: str) -> None:
        path = self._kv_path(key)
        await asyncio.to_thread(lambda: path.unlink(missing_ok=True))  # idempotent

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        """Single-flight, keyed on PRESENCE rather than on the value.

        This tested ``existing is not None``, which conflates "nothing stored"
        with "``None`` stored" — so a producer that legitimately returns
        ``None`` re-ran on every call and single-flight silently did not hold.
        ``InMemoryStore``, the declared reference contract, keys on ``in``, so
        the two backends disagreed on the same input: 1 call versus 3.
        ``_exists`` restores the contract without needing the value at all.
        """
        if await self._exists(key):
            return await self.get(key)
        async with key_lock(self._locks, key):
            if await self._exists(key):
                return await self.get(key)
            result = await fn()  # a raised fn propagates here, unwritten (never cache failure)
            await self.set(key, result, ttl=ttl)
            return result

    async def _exists(self, key: str) -> bool:
        path = self._kv_path(key)
        return await asyncio.to_thread(path.exists)

    async def append(self, key: str, value: Any) -> None:
        path = self._log_path(key)

        def _append() -> None:
            with path.open("a") as f:
                f.write(json.dumps(value) + "\n")

        await asyncio.to_thread(_append)

    async def list(self, key: str) -> list[Any]:
        path = self._log_path(key)

        def _read() -> tuple[list[Any], int]:
            if not path.exists():
                return [], 0
            records: list[Any] = []
            skipped = 0
            for line in path.read_text().splitlines():
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn append (a crash mid-write) leaves ONE unparseable
                    # line. Raising took the whole log with it — every earlier
                    # audit record became unreadable because of the last one.
                    # An append-only log degrades to "the records that
                    # survived", which is what an audit trail is for.
                    skipped += 1
            return records, skipped

        records, skipped = await asyncio.to_thread(_read)
        if skipped and not self._warned_corrupt_log:
            self._warned_corrupt_log = True
            warnings.warn(
                f"FileStore log {key!r} has {skipped} unparseable line(s) — most likely a crash "
                "during an append. They are skipped so the surviving records remain readable; "
                "the skipped ones are lost.",
                UserWarning,
                stacklevel=2,
            )
        return records

    async def compare_and_set(
        self, key: str, expected: Any, value: Any, *, ttl: int | None = None
    ) -> bool:
        """Atomic within the process, via the SAME lock table `get_or_set` uses.

        Sharing the table is the load-bearing part: two lock tables keyed on
        the same string give mutual exclusion between compare-and-sets and
        between single-flights, and none at all BETWEEN them — a producer
        writing its result while a CAS reads would be exactly the lost update
        the primitive exists to prevent.

        Across processes this is NOT atomic, and no file-based store can make
        it so without a lock file protocol; the class docstring already says
        cross-process single-flight needs the real DB's transaction, and this
        inherits that caveat. `PostgresStore` and `RedisStore` do it properly.
        """
        self._warn_ttl_ignored(ttl)
        async with key_lock(self._locks, key):
            # ``get`` collapses absent and stored-null onto None, which is the
            # documented CAS semantics — ``expected=None`` matches both.
            if await self.get(key) != expected:
                return False
            await self.set(key, value)
            return True

    async def increment(self, key: str, by: int = 1, *, ttl: int | None = None) -> int:
        """Read-add-write under the shared per-key lock — see `compare_and_set`.

        Presence, not ``get() is None``: a key holding a stored ``null`` is a
        value the durable backends refuse to add to, so it raises here too
        rather than restarting the counter at ``by``.
        """
        check_by(key, by)
        self._warn_ttl_ignored(ttl)
        async with key_lock(self._locks, key):
            current: Any = 0
            if await self._exists(key):
                current = await self.get(key)
                if not is_counter(current):
                    raise not_a_counter(key, current)
            total = int(current) + by
            await self.set(key, total)
            return total

    async def scan(self, prefix: str, *, limit: int | None = None) -> AsyncIterator[str]:
        """List the directory ONCE, in a thread, then yield from that list.

        Two reasons it is not lazy. A directory handle held open across an
        ``await`` would see entries appear and vanish mid-iteration — the same
        "must not raise while writers land" promise the in-memory backend needs
        a snapshot for. And ``iterdir`` is blocking: streaming it would put a
        stat syscall per key on the event loop, which is what
        ``asyncio.to_thread`` exists here to avoid.

        Filtering on the ``.json`` suffix is what excludes both namespaces that
        must not appear: ``.log`` files (`list` reads those) and the
        ``<name><random>.tmp`` scratch files `_write_atomic` leaves in flight,
        which would otherwise surface a half-written key as a real one.
        """
        check_limit(limit)
        if limit == 0:
            return

        def _keys() -> list[str]:
            suffix = ".json"
            return [
                # Reverse of ``_safe``: the filename is percent-encoded, so the
                # prefix has to be compared against the DECODED key. Comparing
                # encoded forms would mean `scan("a/b")` matched nothing, since
                # the stored name is ``a%2Fb``.
                unquote(entry.name[: -len(suffix)])
                for entry in self._root.iterdir()
                if entry.name.endswith(suffix)
            ]

        sent = 0
        for key in await asyncio.to_thread(_keys):
            if not key.startswith(prefix):
                continue
            yield key
            sent += 1
            if limit is not None and sent >= limit:
                return
