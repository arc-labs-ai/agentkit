"""Narrow formatting normalization; factual uncertainty never becomes a pass."""

from __future__ import annotations

import re
from collections.abc import Collection
from datetime import date
from typing import Any

_VERSION = re.compile(r"(?:SQLite\s+|[Vv]ersion\s+)?[vV]?(\d+\.\d+\.\d+)(?:\s+\((\d{4}-\d{2}-\d{2})\))?")
_TRANSACTION = re.compile(
    r"(?:(?P<worker>billing|export)-worker\s*(?:/\s*|transaction\s+))?"
    r"(?P<identifier>(?:billing|export)-\d+)(?:;\s*(.+))?"
)
_NEGATION = re.compile(r"\b(?:not|no|never|unknown|uncertain|maybe|possibly|either|or|instead)\b|n['’]t\b", re.I)


def normalize_sqlite_version(value: Any) -> str | None:
    """Accept one bare version or vendor/date wrapper, with verified release date."""
    if not isinstance(value, str):
        return None
    match = _VERSION.fullmatch(value.strip())
    if match is None:
        return None
    version, released = match.groups()
    if released is not None:
        try:
            date.fromisoformat(released)
        except ValueError:
            return None
        if version == "3.11.0" and released != "2016-02-15":
            return None
    return version


def normalize_transaction_id(value: Any) -> str | None:
    """Accept a single observed namespace ID, optionally with worker/log context."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = _TRANSACTION.fullmatch(text)
    if match is None or _NEGATION.search(text):
        return None
    identifier = match.group("identifier")
    if match.group("worker") is not None and match.group("worker") != identifier.split("-", 1)[0]:
        return None
    identifiers = set(re.findall(r"\b(?:billing|export)-\d+\b", text))
    return identifier if identifiers == {identifier} else None


def citation_ids(value: Any) -> set[str] | None:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
        return None
    return set(value)


def citations_are_known(value: Any, known: Collection[str]) -> bool:
    identifiers = citation_ids(value)
    return identifiers is not None and identifiers <= set(known)


__all__ = ["normalize_sqlite_version", "normalize_transaction_id", "citation_ids", "citations_are_known"]
