"""Redacted, atomic evaluation reports and per-case evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentkit.models.contracts.constants import EventKind


def source_digest() -> str:
    """Identify the exact local implementation without recording private files."""
    root = Path(__file__).resolve().parents[3]
    paths = [
        p
        for directory in (root / "agentkit", root / "tests/evals")
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".md", ".csv"}
        and not any(part in {"reports", "__pycache__"} for part in p.relative_to(root).parts)
    ]
    checksum = hashlib.sha256()
    for path in sorted(paths):
        checksum.update(path.relative_to(root).as_posix().encode())
        checksum.update(b"\0")
        checksum.update(path.read_bytes())
        checksum.update(b"\0")
    return checksum.hexdigest()


def case_spend(events: list[Any], envelope: Decimal, *, dispatched: bool) -> Decimal:
    """Sum every call in a case; incomplete/unknown calls reserve the envelope."""
    starts = {e.call_id for e in events if e.kind == EventKind.CALL_STARTED}
    costs: dict[str, Decimal] = {}
    for event in events:
        if event.kind != EventKind.CALL_FINISHED:
            continue
        try:
            cost = Decimal(str(event.fields.get("cost_usd")))
        except Exception:
            return envelope
        if not cost.is_finite() or cost < 0:
            return envelope
        costs[event.call_id] = cost
    if starts - costs.keys() or (dispatched and not costs):
        return envelope
    return sum(costs.values(), Decimal(0))


def redact(value: Any, secrets: tuple[str, ...]) -> Any:
    """Redact exact resolver values from every report string, including failures."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    return value


def write_report(target: Path, report: dict[str, Any], secrets: tuple[str, ...]) -> None:
    """Atomically checkpoint progress so interruption leaves reviewable results."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(redact(report, secrets), indent=2, default=str) + "\n")
    temporary.chmod(0o600)
    temporary.replace(target)


def write_events(target: Path, events: list[Any], secrets: tuple[str, ...]) -> None:
    """Persist full normalized trace records, with credential redaction."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for event in events:
            stream.write(json.dumps(redact(asdict(event), secrets), default=str) + "\n")
    temporary.chmod(0o600)
    temporary.replace(target)
