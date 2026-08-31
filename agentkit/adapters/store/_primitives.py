"""The parts of `compare_and_set` / `increment` / `scan` that must be IDENTICAL across backends.

Four adapters implement these three primitives over four completely different
substrates — a dict, a directory, a WATCH/MULTI transaction, and one SQL
statement. What a caller can rely on is the part that does not vary, and the
only way that part stays uniform is if it is written once.

The two things here are the ones that silently drift when copied:

* **The error a bad `increment` raises.** Left to each backend, this is
  ``TypeError`` on the reference store, ``redis.ResponseError`` over Redis and
  ``asyncpg.InvalidTextRepresentationError`` over Postgres — the exact
  backend-type leak ``agentkit.kernel.errors`` exists to close. One builder
  means one type AND one message.

* **What counts as an integer.** ``isinstance(True, int)`` is True in Python,
  so a dict-backed store would happily increment a stored ``True`` while Redis
  and Postgres reject ``true`` outright — the reference implementation would be
  the one that was wrong, and it is the one everybody tests against offline.
"""

from __future__ import annotations

from typing import Any

from agentkit.kernel.errors import StoreValueError


def check_limit(limit: int | None) -> None:
    """Reject a negative `scan` limit.

    ``None`` is "no cap" and ``0`` is a real cap of zero — collapsing the two
    would make ``limit=remaining_budget`` return the entire key space at
    exactly the moment the budget ran out. A negative limit is a caller bug (an
    underflowed budget), and every backend has a different way of ignoring it:
    a Python slice with a negative stop drops items from the END, Postgres's
    ``LIMIT -1`` raises, and Redis's COUNT would just be a hint. Rejecting up
    front is the only outcome all four can agree on.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"scan limit must be >= 0 or None, got {limit}")


def is_counter(value: Any) -> bool:
    """Whether ``value`` is something `increment` may add to.

    ``bool`` is excluded even though it is an ``int`` subclass: the durable
    backends see JSON, where ``true`` is not a number, and Redis's INCRBY
    refuses the string ``true``. Allowing it in memory only would make the
    offline reference store the single backend that accepted a value the
    others reject.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def not_a_counter(key: str, value: Any) -> StoreValueError:
    """The one error every backend raises when `increment` hits a non-integer.

    Names the offending type, because the caller's next question is always
    "what is in there?" and the answer is not visible from the traceback of a
    backend that failed inside a SQL cast.
    """
    return StoreValueError(
        f"increment on key {key!r} requires an integer value; the key holds "
        f"{type(value).__name__}. Counters and JSON documents cannot share a key."
    )


def check_by(key: str, by: Any) -> None:
    """Reject a non-integer ``by`` BEFORE it reaches any backend.

    `is_counter` was applied to the value already in the key and not to the
    amount being added, so the four backends each improvised, and all four
    improvised differently. Measured, on ``increment(k, 1.5)``:

    * `InMemoryStore` and `FileStore` returned ``1.5`` — a ``float`` out of a
      method annotated ``-> int`` — and left ``1.5`` in the key, which the NEXT
      increment then rejects as a non-counter. The counter is poisoned by a
      call that reported success.
    * `RedisStore` returned ``1`` while the key held ``1.5``: the ``int()``
      around INCRBY's reply truncates, so the number the caller acts on and the
      number the store holds disagree. Against a real Redis the same call
      raises instead, and the error names the type of the *stored* value —
      ``NoneType`` for a fresh key — pointing at everything except the argument
      that was wrong.
    * `PostgresStore` failed inside the driver with a bare ``ValueError``.

    ``bool`` is excluded for the same reason it is in `is_counter`: ``True``
    silently counts as ``1`` on three backends and is rejected by the fourth.
    Validating here makes the answer one type and one message everywhere, which
    is the whole reason this module exists.
    """
    if not is_counter(by):
        raise StoreValueError(
            f"increment on key {key!r} requires an integer `by`; got "
            f"{type(by).__name__}. A fractional or non-numeric amount cannot be "
            "applied atomically by any backend, and silently truncating it would "
            "make the returned total disagree with the stored one."
        )


__all__ = ["check_by", "check_limit", "is_counter", "not_a_counter"]
