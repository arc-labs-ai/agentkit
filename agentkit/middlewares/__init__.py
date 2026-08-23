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

`audit()` vs `retry()` — pick the trail you want. In the tool chain above `retry()` is INSIDE `audit()`,
so `Audit` sees one outcome per logical tool call: three retried executions of a side-effecting tool fold
into ONE record. That is the right shape when the question is "what did this run ask the tool to do", and
it is the only ordering in which an `idempotent()` replay can be recorded as `"deduped"` (a hit
short-circuits everything inner). Swap to `[… idempotent(), retry(breaker=…), audit()]` when the question
is "how many times did the side effect actually fire" — then every attempt gets its own record, and a
deduped replay produces none at all because `audit()` is never reached. `Audit` records failures either
way (`decision: "failed"`); it cannot see retry attempts it sits outside of.

.. note::
   ``Egress`` / ``Audit`` live in ``egress_audit.py``, not ``security.py``. A
   submodule named ``security`` shadowed the ``security()`` factory re-exported
   here — importing a submodule binds its name onto the parent package, and
   that binding happened AFTER the ``from .guard import security`` line, so
   ``middlewares.egress_audit`` was the MODULE and ``security()`` raised
   ``TypeError: 'module' object is not callable``. The module was renamed
   rather than the binding patched, because reordering the imports is undone by
   the formatter and an explicit re-bind is one more thing to forget.
   ``tests/meta/test_public_surface.py`` now fails on any recurrence.
"""

from agentkit.middlewares.compaction import Compaction, compaction
from agentkit.middlewares.egress_audit import Audit, Egress, audit, egress
from agentkit.middlewares.guard import SecurityMiddleware, security
from agentkit.middlewares.memoize import idempotent, memoize, semantic_memoize
from agentkit.middlewares.meter import MeterMiddleware, meter
from agentkit.middlewares.output_coerce import output_coerce
from agentkit.middlewares.resilience import fallback, retry
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
