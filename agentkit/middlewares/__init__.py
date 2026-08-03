"""Middlewares — the batteries. The app assembles two explicit, ordered chains (chat + tool) and hands
them to the `Invoker`. There are two kinds:

- **`BaseMiddleware` (transform / guard / observe)** — phase classes that don't control invoking `next`:
  `MeterMiddleware`, `Compaction`, `Egress`, `Audit`, `SecurityMiddleware`. Lowercase factories
  (`meter()`, `compaction()`, …) return instances. ``MeterMiddleware`` (the recording middleware) is
  named to disambiguate from ``runtime.Meter`` (the Protocol it records into).
- **Raw `(call, next)` (resilience / caching / instrumentation)** — must re-invoke, skip, or wrap-with-a-
  context-manager `next`, which phase methods can't: `retry`/`fallback` (re-invoke), `memoize`/`idempotent`/
  `semantic_memoize` (skip on hit), `tracing` (hold a span open across the call).

Typical chat chain:  [tracing(), compaction(…), meter(), fallback([...]), retry(breaker=…)]
Typical tool chain:  [tracing(), meter(), egress(guardrail), idempotent(), audit(), retry(breaker=…)]

(`compaction` sits ahead of `meter` so the meter estimates tokens on the already-compacted transcript.)
"""

from agentkit.middlewares.compaction import Compaction, compaction
from agentkit.middlewares.guard import SecurityMiddleware, security
from agentkit.middlewares.memoize import idempotent, memoize, semantic_memoize
from agentkit.middlewares.meter import MeterMiddleware, meter
from agentkit.middlewares.output_coerce import output_coerce
from agentkit.middlewares.resilience import fallback, retry
from agentkit.middlewares.security import Audit, Egress, audit, egress
from agentkit.middlewares.tracing import tracing

__all__ = [
    # raw (call, next) — resilience / caching / instrumentation
    "tracing",
    "retry",
    "fallback",
    "memoize",
    "idempotent",
    "semantic_memoize",
    "output_coerce",
    # BaseMiddleware classes (+ lowercase factories) — transform / guard / observe
    "MeterMiddleware",
    "meter",
    "Compaction",
    "compaction",
    "Egress",
    "egress",
    "Audit",
    "audit",
    "SecurityMiddleware",
    "security",
]
