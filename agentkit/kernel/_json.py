"""Fast JSON shim — orjson if installed, stdlib json otherwise.

orjson is gated behind the ``agentkit[fast]`` extra. The two
functions here have the EXACT signature the call sites use:

- ``loads(s: str | bytes) -> Any``
- ``dumps(obj: Any) -> str``

orjson natively returns bytes from ``dumps``; we decode to ``str``
here so call sites stay uniform. orjson accepts both ``str`` and
``bytes`` for ``loads`` — same as stdlib. No behavioural difference
for the agentkit shapes (dicts of str / int / float / list / None);
orjson refuses non-string dict keys, but agentkit never produces
those on the hot paths we're touching.

``JSONDecodeError`` is re-exported here pointing at whichever
backend is active — orjson's decode error is a ``ValueError``
subclass but NOT a ``json.JSONDecodeError`` subclass, so callers
should catch THIS alias rather than ``json.JSONDecodeError``.
"""

from __future__ import annotations

from typing import Any

try:
    import orjson as _orjson  # type: ignore[import-not-found]

    def loads(s: str | bytes) -> Any:
        return _orjson.loads(s)

    def dumps(obj: Any, *, default: Any = None) -> str:
        if default is None:
            encoded: bytes = _orjson.dumps(obj)
        else:
            encoded = _orjson.dumps(obj, default=default)
        return encoded.decode("utf-8")

    JSONDecodeError = _orjson.JSONDecodeError
    BACKEND = "orjson"
except ImportError:
    import json as _json

    def loads(s: str | bytes) -> Any:
        return _json.loads(s)

    def dumps(obj: Any, *, default: Any = None) -> str:
        # Match orjson's compact output so bytes are identical across backends.
        return _json.dumps(obj, separators=(",", ":"), default=default)

    JSONDecodeError = _json.JSONDecodeError
    BACKEND = "stdlib"


__all__ = ["loads", "dumps", "JSONDecodeError", "BACKEND"]
