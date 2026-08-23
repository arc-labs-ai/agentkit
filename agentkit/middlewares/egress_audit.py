"""security middlewares — `egress` (default-deny, fail-fast) and `audit` (one record per tool call).

Both are tool-chain `BaseMiddleware`s. `Egress` is a guard in `on_request` (a blocked URL never executes
a side effect); `Audit` records in `on_response` AND `on_error`, so a side effect that fired and then
failed still leaves a record.
"""

from __future__ import annotations

from typing import Any

from agentkit.kernel.middleware import BaseMiddleware, MiddlewareContext
from agentkit.kernel.resilience import stable_hash as _hash

_AUDIT_LOG = "audit:log"


class Egress(BaseMiddleware):
    """A tool with `request.url_arg` set has that URL checked (SSRF + allowlist) before it can run."""

    def __init__(self, guardrail: Any) -> None:
        # A security control that can be constructed inert is worse than one
        # that is absent: the chain LOOKS guarded. `egress(config.guardrail)`
        # with an unset config silently disabled every SSRF and allowlist
        # check, with no signal anywhere. Fail at wiring time instead.
        if guardrail is None:
            raise ValueError(
                "egress() requires a Guardrail — it is the thing that performs the SSRF and "
                "allowlist checks. Passing None would leave the middleware in the chain while "
                "checking nothing. Drop egress() entirely if that is genuinely what you want."
            )
        if not callable(getattr(guardrail, "check_url", None)):
            raise TypeError(
                f"egress() needs an object with a check_url(url) method; "
                f"{type(guardrail).__name__} has none, so no URL would ever be checked."
            )
        self._guardrail = guardrail

    async def on_request(self, ctx: MiddlewareContext) -> None:
        r = ctx.request
        url_arg = getattr(r, "url_arg", None)
        if url_arg:
            url = r.arguments.get(url_arg)
            if url is not None:
                self._guardrail.check_url(url)  # raises on a blocked URL — before any side effect


class Audit(BaseMiddleware):
    """One record per tool call — including the calls that FAILED.

    ``Audit`` used to write only in ``on_response``, i.e. only after a
    success. That is exactly backwards for a side-effecting tool: the
    dangerous case is the one that fired the side effect and *then*
    failed — the payment gateway that charged the card and timed out on
    the response. Measured with the documented tool chain
    (``[… idempotent(), audit(), retry(breaker=…)]``): a charging tool
    that raised after every attempt executed 3 times and left
    ``audit records: []``. The money moved and the audit trail was
    empty, which is the one thing an audit trail may never be.

    ``on_error`` now writes a ``"failed"`` record carrying the error
    type and message, then re-raises so the failure still propagates —
    the middleware records, it never recovers.

    **One record = one invocation of THIS middleware.** ``retry()`` sits
    INSIDE ``audit()`` in the documented chain, so its attempts are
    invisible from here and three executions fold into one record. That
    is an ordering choice, not something ``Audit`` can detect: placing
    ``audit()`` inside ``retry()`` yields one record per attempt, at the
    cost of the ``"deduped"`` record (``idempotent()`` short-circuits
    before ``audit()`` is ever reached). Since neither ordering gives
    both, the record now carries an ``attempts`` count that ``retry()``
    publishes on ``call.meta`` — so one record covering three charges
    says three, instead of reading exactly like a record covering one.
    Both orderings are supported and pinned by tests; see the chain note
    in ``middlewares/__init__``.
    """

    def __init__(self, *, store: Any = None, key: str = _AUDIT_LOG) -> None:
        self._store = store
        self._key = key

    def _record(self, ctx: MiddlewareContext, **extra: Any) -> dict[str, Any]:
        return {
            "run_id": ctx.run.correlation_id,
            "scope": ctx.run.scope.key(),
            "tool": ctx.request.name,
            "arg_hash": _hash(ctx.request.arguments),
            **extra,
        }

    async def on_response(self, ctx: MiddlewareContext, result: Any) -> Any:
        s = self._store or ctx.run.store
        if s is not None:
            # distinguish a real side effect from an idempotent replay (the inner memoize signals a hit),
            # so the audit trail can't be misread as N executions when only one ran.
            deduped = bool(ctx.call.meta.get("cache_hit"))
            # How many times the inner chain actually ran. ``retry()`` stamps
            # this; absent it (no retry wired) the call ran once. One record
            # covering three executions used to read exactly like a record
            # covering one, which is the difference between "we charged the
            # card" and "we charged the card three times".
            attempts = int(ctx.call.meta.get("attempts", 1))
            await s.append(
                self._key,
                self._record(
                    ctx,
                    result_hash=_hash(result),
                    decision="deduped" if deduped else "executed",
                    attempts=attempts,
                ),
            )
        return result

    async def on_error(self, ctx: MiddlewareContext, exc: Exception) -> Any:
        s = self._store or ctx.run.store
        if s is not None:
            # A failed write must not mask the tool's own failure: the
            # exception below is what the caller needs to see. The record is
            # best-effort; the raise is not.
            try:
                await s.append(
                    self._key,
                    self._record(
                        ctx,
                        result_hash=None,  # there is no result — do not hash the exception
                        decision="failed",
                        attempts=int(ctx.call.meta.get("attempts", 1)),
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                    ),
                )
            except Exception:  # noqa: BLE001 — recording must never replace the real failure
                pass
        raise exc


def egress(guardrail: Any) -> Egress:
    return Egress(guardrail)


def audit(*, store: Any = None, key: str = _AUDIT_LOG) -> Audit:
    return Audit(store=store, key=key)
