"""Canonical image blocks hold immutable raw bytes; JSON boundaries encode once.

This codec validates representation, MIME and size, not image file integrity.
It has no provider selection, execution policy or credential responsibilities.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentkit.core.serialization._immutable import thaw

_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class InvalidContent(ValueError):
    """Safe representation failure; messages never contain tool payloads."""


def encode_content(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return independent JSON blocks, refusing malformed canonical images.

    The 20 MiB bound is on raw bytes. Total output accounting uses the UTF-8
    JSON envelope *after* this encoding, including base64 expansion.
    """
    encoded = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise InvalidContent("tool returned an invalid content block")
        item: dict[str, Any] = thaw(block)
        if item.get("type") == "image":
            data, mime = item.get("data"), item.get("mimeType")
            if not isinstance(data, bytes) or not data or not isinstance(mime, str) or mime not in _IMAGE_MIMES:
                raise InvalidContent("tool returned an invalid image block; supply raw bytes and a supported MIME type")
            if len(data) > _MAX_IMAGE_BYTES:
                raise InvalidContent("tool image exceeds the 20 MiB raw image limit")
            item["data"] = base64.b64encode(data).decode("ascii")
        elif item.get("type") == "text" and not isinstance(item.get("text", ""), str):
            raise InvalidContent("tool returned an invalid text block")
        encoded.append(item)
    try:
        json.dumps(encoded, allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidContent("tool returned invalid content; non-image values must be JSON serializable") from None
    return encoded


def decode_content(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Restore canonical images from JSON transcripts, including older base64 records."""
    decoded = []
    for block in blocks:
        item: dict[str, Any] = thaw(block)
        if item.get("type") == "image":
            data = item.get("data")
            if not isinstance(data, str) or len(data) > 4 * ((_MAX_IMAGE_BYTES + 2) // 3):
                raise InvalidContent("transcript contains an invalid image block")
            try:
                item["data"] = base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error):
                raise InvalidContent("transcript contains an invalid image block") from None
            # The same representation guards apply to durable and live content.
            encode_content([item])
        decoded.append(item)
    return decoded
