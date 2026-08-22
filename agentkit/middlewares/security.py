"""security middlewares — `egress` (default-deny, fail-fast) and `audit` (one record per tool call).

Both are tool-chain `BaseMiddleware`s. `Egress` is a guard in `on_request` (a blocked URL never executes
a side effect); `Audit` records in `on_response` (after a successful call).
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
    def __init__(self, *, store: Any = None, key: str = _AUDIT_LOG) -> None:
        self._store = store
        self._key = key

    async def on_response(self, ctx: MiddlewareContext, result: Any) -> Any:
        s = self._store or ctx.run.store
        if s is not None:
            # distinguish a real side effect from an idempotent replay (the inner memoize signals a hit),
            # so the audit trail can't be misread as N executions when only one ran.
            deduped = bool(ctx.call.meta.get("cache_hit"))
            await s.append(
                self._key,
                {
                    "run_id": ctx.run.correlation_id,
                    "scope": ctx.run.scope.key(),
                    "tool": ctx.request.name,
                    "arg_hash": _hash(ctx.request.arguments),
                    "result_hash": _hash(result),
                    "decision": "deduped" if deduped else "executed",
                },
            )
        return result


def egress(guardrail: Any) -> Egress:
    return Egress(guardrail)


def audit(*, store: Any = None, key: str = _AUDIT_LOG) -> Audit:
    return Audit(store=store, key=key)
