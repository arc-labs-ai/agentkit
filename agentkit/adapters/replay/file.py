"""FileReplayStore — disk-backed ReplayStore.

Writes each ``ReplayRecord`` as a single JSON file under
``<root>/<span_id>.json``. Reads look up by span_id. Idempotent —
re-writing the same span_id overwrites; reads after a write see
the new content.

Concurrency model: writes are atomic via the standard "write to a
``.tmp`` file then ``os.replace``" idiom — even under concurrent
writers no reader sees a torn file. The file system's per-inode
atomicity is the only correctness primitive we depend on.

Default location: ``$XDG_DATA_HOME/agentkit/replays`` or
``~/.local/share/agentkit/replays`` (the XDG Base Directory
convention). Override by passing ``root=Path(...)``, or set
``AGENTKIT_REPLAY_DIR`` and construct via ``from_env()``. That env
var is the ONLY one this module reads — there is no fallback
spelling and no legacy directory to inherit.

Failure reporting: this store is best-effort by contract —
``ReplayStore.put`` must not raise into the run — but "best-effort"
never meant "invisible". Every failure path logs (see ``_warn_once``
for the anti-spam rule) and bumps a counter, so a store that is
silently dropping every write is distinguishable from one that is
working. Read the counters via ``dropped_writes`` / ``failed_reads``.

Serialization: ``orjson`` when available (via the existing
``agentkit.kernel._json`` shim), stdlib ``json`` otherwise.
Records that aren't JSON-serializable (e.g. nested Pydantic
models, dataclasses, custom objects) are best-effort —
``ReplayRecord.request`` and ``.response`` are typed ``Any``
specifically to support any in-process shape; we serialize via a
``default=str`` fallback so unknown types become their ``repr``
rather than crashing the store. Operators using FileReplayStore
should keep records shallow + simple OR ship their own adapter
for richer types.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Final

from agentkit.kernel._json import dumps, loads
from agentkit.kernel.replay import ReplayRecord

logger = logging.getLogger(__name__)

# Bounded character class so a span_id can never write outside
# the configured root (defense against malformed input). Span IDs
# are hex strings per the OTel spec; reject anything else.
_VALID_SPAN_ID = re.compile(r"^[0-9a-fA-F]{16,32}$")

#: Env var an operator sets to point the store at a directory. The only one.
ENV_REPLAY_DIR: Final = "AGENTKIT_REPLAY_DIR"


def _current_default_root() -> Path:
    """The default root for this package.

    ``$XDG_DATA_HOME/agentkit/replays`` when the variable is set, otherwise
    ``~/.local/share/agentkit/replays`` — the XDG-specified fallback for
    ``XDG_DATA_HOME``, rather than a dotdir in ``$HOME`` (which is not what
    XDG specifies).
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "agentkit" / "replays"


class FileReplayStore:
    """ReplayStore that writes each record to ``<root>/<span_id>.json``.

    Constructed with an explicit root directory (created on first
    write if missing). Use ``FileReplayStore.from_env()`` to read
    the ``AGENTKIT_REPLAY_DIR`` env var, or ``FileReplayStore.default()``
    for the conventional location.

    Best-effort, but never silent: a failed write logs and returns
    ``None`` (the contract of ReplayStore.put says writes must not
    raise into the run) while incrementing ``dropped_writes``. A
    failed read returns ``None`` and increments ``failed_reads``. A
    genuine miss also returns ``None`` but is NOT a failure — it logs
    at DEBUG and touches no counter, so "the record isn't there" stays
    distinguishable from "the store is broken".
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._dropped_writes = 0
        self._failed_reads = 0
        # Keys of failure classes already reported. See ``_warn_once``.
        self._warned: set[str] = set()

    @property
    def root(self) -> Path:
        """The resolved directory this store reads and writes."""
        return self._root

    @property
    def dropped_writes(self) -> int:
        """Records ``put()`` accepted but could not persist.

        Non-zero means replay data is missing for this run. Exposed as a
        number rather than only a log line because the log is warned-once
        (see ``_warn_once``) — the counter is what stays accurate under a
        sustained failure such as a full disk.
        """
        return self._dropped_writes

    @property
    def failed_reads(self) -> int:
        """Reads that failed for a reason OTHER than a genuine miss.

        A missing file is a cache miss and is deliberately excluded: mixing
        misses into this counter would make it useless, since a miss is the
        expected outcome for any span that was never recorded.
        """
        return self._failed_reads

    @classmethod
    def from_env(cls) -> FileReplayStore | None:
        """Read ``AGENTKIT_REPLAY_DIR`` and construct; return ``None`` when
        the env var isn't set so callers can fall back to
        ``NoopReplayStore``.

        An empty value counts as unset. ``FOO=`` in a shell profile or a
        compose file is how people spell "not configured", and ``Path("")``
        resolves to the process CWD — so treating it as a path would scatter
        replay JSON through whatever directory the app happened to start in,
        while looking configured.
        """
        root = os.environ.get(ENV_REPLAY_DIR) or ""
        return cls(root) if root else None

    @classmethod
    def default(cls) -> FileReplayStore:
        """Conventional location: ``$XDG_DATA_HOME/agentkit/replays`` or
        ``~/.local/share/agentkit/replays``.

        No directory probing and no fallbacks: the path is a pure function of
        ``XDG_DATA_HOME``, so what this returns is predictable from the
        environment alone. An explicit ``root=`` or ``AGENTKIT_REPLAY_DIR``
        is how you point the store anywhere else.
        """
        return cls(_current_default_root())

    async def put(self, record: ReplayRecord) -> None:
        if not _VALID_SPAN_ID.match(record.span_id):
            # A rejected span_id is a CALLER bug, not an environmental one:
            # every OTel span id is 16 hex chars, so a value that fails the
            # guard means something upstream synthesised an id or passed a
            # path fragment. Logged at ERROR to separate it from the
            # operational (disk/permission) failures below — those say
            # "fix your machine", this one says "fix your code". Still
            # dropped rather than raised, per the put() contract.
            self._dropped_writes += 1
            self._warn_once(
                "invalid-span-id",
                logging.ERROR,
                "FileReplayStore: dropped a replay record whose span_id %r is not a valid "
                "OTel span id (expected %s). This is a caller bug — the record was NOT "
                "persisted. Further occurrences are counted on `.dropped_writes` and not logged.",
                record.span_id,
                _VALID_SPAN_ID.pattern,
            )
            return
        path = self._root / f"{record.span_id}.json"
        try:
            # asyncio.to_thread keeps the event loop free during the
            # FS write — modest payloads (a chat input) are <100KB
            # but a large tool result is unbounded.
            await asyncio.to_thread(self._write_atomic, path, record)
        except Exception as exc:
            # Swallowed per contract, but NOT silently. Everything that lands
            # here is an operator-actionable condition — unwritable root, full
            # disk, a payload the JSON backend refused even with the
            # ``default=`` fallback. Broad on purpose: the value of this
            # handler is that no failure mode escapes it, so narrowing it to
            # OSError would re-create the silent hole for encoder errors.
            self._dropped_writes += 1
            self._warn_once(
                "write-failed",
                logging.WARNING,
                "FileReplayStore: write to %s failed (%s: %s); the record was dropped and the "
                "run continued. Further failures are counted on `.dropped_writes` and not "
                "logged again.",
                path,
                type(exc).__name__,
                exc,
            )
            return

    async def get(self, span_id: str) -> ReplayRecord | None:
        if not _VALID_SPAN_ID.match(span_id):
            # Same caller-bug reasoning as put(), one level quieter: a bad
            # id on the read side loses no data, it just cannot match
            # anything. WARNING rather than ERROR, warned-once, because a
            # tool sweeping ids from an external source can hit this in a
            # loop and the counter carries the volume.
            self._failed_reads += 1
            self._warn_once(
                "invalid-span-id-read",
                logging.WARNING,
                "FileReplayStore: refusing to read span_id %r — not a valid OTel span id "
                "(expected %s). Returning a miss. Further occurrences are counted on "
                "`.failed_reads` and not logged again.",
                span_id,
                _VALID_SPAN_ID.pattern,
            )
            return None
        path = self._root / f"{span_id}.json"
        try:
            raw = await asyncio.to_thread(path.read_text)
        except FileNotFoundError:
            # NOT a failure. Most spans were never recorded (sampling, a
            # store attached mid-run, a purged directory), so a miss is the
            # ordinary outcome of a lookup. Logging it above DEBUG would
            # bury the real failures below in per-lookup noise, and it
            # deliberately does not touch ``failed_reads``.
            logger.debug("FileReplayStore: miss for span_id %s (no file at %s)", span_id, path)
            return None
        except Exception as exc:
            # A file that exists but cannot be read is a real problem —
            # wrong ownership after a container user change, a directory
            # where a record should be, bad encoding. Distinct from a miss
            # and reported as such.
            self._failed_reads += 1
            self._warn_once(
                "read-failed",
                logging.WARNING,
                "FileReplayStore: reading %s failed (%s: %s); treating as a miss. Further "
                "failures are counted on `.failed_reads` and not logged again.",
                path,
                type(exc).__name__,
                exc,
            )
            return None
        try:
            data = loads(raw)
            record = self._record_from_dict(data)
        except Exception as exc:
            # Corruption, not absence. Atomic writes mean this store cannot
            # produce a half-written file itself, so a record that fails to
            # decode means something else touched the directory (hand-edited
            # file, a truncating backup tool, a foreign file named like a
            # span id) — worth surfacing, never worth crashing a run over.
            # ``_record_from_dict`` is inside the same guard because a JSON
            # document of the wrong SHAPE raises KeyError/TypeError there,
            # which previously escaped get() entirely: the one path in this
            # module that could still raise at a caller.
            self._failed_reads += 1
            self._warn_once(
                "decode-failed",
                logging.WARNING,
                "FileReplayStore: %s is not a readable replay record (%s: %s); treating as a "
                "miss. Further failures are counted on `.failed_reads` and not logged again.",
                path,
                type(exc).__name__,
                exc,
            )
            return None
        return record

    # ── internals ────────────────────────────────────────────────

    def _warn_once(self, key: str, level: int, msg: str, *args: Any) -> None:
        """Log the first occurrence of a failure class, then stay quiet.

        A replay store fails in sustained bursts, not one-offs: a full disk
        or an unwritable root fails EVERY put for the rest of the run. One
        log line per failure would put thousands of identical warnings in
        front of whatever the operator was actually reading, which is its
        own outage. Same trade-off (and same shape) as
        ``RollupObserver`` in ``adapters/observer/cadence.py``: warn once
        with the cause, count the rest on a public counter.

        Keyed per failure class and per store instance, so a write failure
        never masks a later read failure, and each key's first line always
        carries the real exception rather than the first one ever seen.

        No lock: every caller mutates this from the event-loop thread (only
        the file I/O itself runs in ``to_thread``), so the check-then-add is
        not actually interleaved. A duplicate line under some exotic
        multi-loop sharing would be harmless anyway.
        """
        if key in self._warned:
            return
        self._warned.add(key)
        logger.log(level, msg, *args)

    def _write_atomic(self, path: Path, record: ReplayRecord) -> None:
        """Write JSON to a tmp file in the same directory, then
        ``os.replace`` to the final path. ``os.replace`` is atomic
        on POSIX + Windows (NTFS) — no torn reads even under
        concurrent writers."""
        self._root.mkdir(parents=True, exist_ok=True)
        data = self._record_to_dict(record)
        body = dumps(data, default=_repr_default)
        # NamedTemporaryFile in the same dir → atomic rename works.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self._root, delete=False, suffix=".tmp"
        ) as fh:
            fh.write(body if isinstance(body, str) else body.decode("utf-8"))
            tmp_path = Path(fh.name)
        try:
            os.replace(tmp_path, path)
        except Exception:
            # The failure itself is reported by put(); this only stops the
            # partial write from becoming litter. Without it a sustained
            # failure at the rename step (full disk, read-only remount
            # between create and rename) leaves one orphaned ``.tmp`` per
            # call in the operator's data directory forever, and nothing
            # ever collects them.
            tmp_path.unlink(missing_ok=True)
            raise

    def _record_to_dict(self, record: ReplayRecord) -> dict[str, Any]:
        # ``request`` and ``response`` are ``Any`` — best-effort
        # serialize via dataclasses.asdict first (handles nested
        # dataclasses), then the json backend's default=str
        # fallback handles whatever's left.
        return {
            "span_id": record.span_id,
            "operation": record.operation,
            "request": _to_jsonable(record.request),
            "response": _to_jsonable(record.response),
        }

    def _record_from_dict(self, data: dict[str, Any]) -> ReplayRecord:
        return ReplayRecord(
            span_id=str(data["span_id"]),
            operation=str(data["operation"]),
            request=data.get("request"),
            response=data.get("response"),
        )


def _repr_default(value: Any) -> Any:
    """Last-chance default for the JSON backend: coerce unknown
    types to their ``repr``. Reached only when ``_to_jsonable``
    let something through (which shouldn't happen for the shapes
    above, but the safety net keeps writes from crashing)."""
    return repr(value)


def _to_jsonable(value: Any) -> Any:
    """Best-effort coercion of an arbitrary value to a JSON-serializable
    shape. Dataclasses → dicts; lists/tuples recurse; Pydantic
    models with ``model_dump`` use it; anything else falls through
    to its ``repr``."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):  # Pydantic v2
        try:
            return value.model_dump()
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return repr(value)


__all__ = ["FileReplayStore", "ENV_REPLAY_DIR"]
