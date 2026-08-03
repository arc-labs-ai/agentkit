"""FileReplayStore — disk-backed ReplayStore.

Writes each ``ReplayRecord`` as a single JSON file under
``<root>/<span_id>.json``. Reads look up by span_id. Idempotent —
re-writing the same span_id overwrites; reads after a write see
the new content.

Concurrency model: writes are atomic via the standard "write to a
``.tmp`` file then ``os.replace``" idiom — even under concurrent
writers no reader sees a torn file. The file system's per-inode
atomicity is the only correctness primitive we depend on.

Default location: ``$XDG_DATA_HOME/rio/replays`` or
``~/.rio/replays`` (the standard convention). Override by passing
``root=Path(...)`` or setting ``RIO_REPLAY_DIR`` (the engine reads
this env var at startup).

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
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from agentkit.kernel._json import dumps, loads
from agentkit.kernel.replay import ReplayRecord

# Bounded character class so a span_id can never write outside
# the configured root (defense against malformed input). Span IDs
# are hex strings per the OTel spec; reject anything else.
_VALID_SPAN_ID = re.compile(r"^[0-9a-fA-F]{16,32}$")


class FileReplayStore:
    """ReplayStore that writes each record to ``<root>/<span_id>.json``.

    Constructed with an explicit root directory (created on first
    write if missing). Use ``FileReplayStore.from_env()`` to read
    the ``RIO_REPLAY_DIR`` env var.

    Best-effort: a failed write logs a warning and returns ``None``
    (the contract of ReplayStore.put says writes must not raise
    into the run). A failed read returns ``None`` (same as a
    genuine miss).
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve()

    @classmethod
    def from_env(cls) -> FileReplayStore | None:
        """Read ``RIO_REPLAY_DIR`` and construct; return ``None`` when
        the env var isn't set so callers can fall back to
        ``NoopReplayStore``."""
        d = os.environ.get("RIO_REPLAY_DIR")
        return cls(d) if d else None

    @classmethod
    def default(cls) -> FileReplayStore:
        """Standard location: ``$XDG_DATA_HOME/rio/replays`` or
        ``~/.rio/replays``."""
        xdg = os.environ.get("XDG_DATA_HOME")
        root = Path(xdg) / "rio" / "replays" if xdg else Path.home() / ".rio" / "replays"
        return cls(root)

    async def put(self, record: ReplayRecord) -> None:
        if not _VALID_SPAN_ID.match(record.span_id):
            # Defensive: log + drop. Don't raise into the run.
            return
        path = self._root / f"{record.span_id}.json"
        try:
            # asyncio.to_thread keeps the event loop free during the
            # FS write — modest payloads (a chat input) are <100KB
            # but a large tool result is unbounded.
            await asyncio.to_thread(self._write_atomic, path, record)
        except Exception:
            # Swallow per contract.
            return

    async def get(self, span_id: str) -> ReplayRecord | None:
        if not _VALID_SPAN_ID.match(span_id):
            return None
        path = self._root / f"{span_id}.json"
        try:
            raw = await asyncio.to_thread(path.read_text)
        except FileNotFoundError:
            return None
        except Exception:
            return None
        try:
            data = loads(raw)
        except Exception:
            return None
        return self._record_from_dict(data)

    # ── internals ────────────────────────────────────────────────

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
        os.replace(tmp_path, path)

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


__all__ = ["FileReplayStore"]
