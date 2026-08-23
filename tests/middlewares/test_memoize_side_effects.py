"""`memoize()` must never cache a SIDE-EFFECTING tool call.

`default_key` gained tool support (name + arguments) and became the default,
but nothing in the module consulted `side_effecting` — `grep side_effecting
agentkit/middlewares/*.py` matched only `idempotent()`'s own predicate — and
`when` defaulted to `None`. So wiring `memoize(store=…)` into a tool chain
deduped mutations.

Measured before the fix: `send_email(to="a@b.c")` invoked twice through an
`Invoker` executed ONCE (`len(sent) == 1`) and the second caller was handed the
first call's stored `{'sent': True, 'n': 1}`. Both shapes reproduced — with the
flag mirrored onto the `ToolRequest` AND with only the tool object declaring it
— which is why `_side_effecting()` reads both.

Reporting success for an email that was never sent is the worst class of cache
bug: a missed hit costs one re-execution, a false hit costs the action itself.
The deliberate "safe to replay" case keeps its home in `idempotent()`, which
opts in via `allow_side_effects=True` and pays for it with a run-scoped key.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentkit.adapters.store.memory import InMemoryStore
from agentkit.kernel.types import Scope, ToolRequest
from agentkit.middlewares import idempotent, memoize
from agentkit.runtime import Budget, Invoker, RunContext, Services


class _CountingTool:
    """Records every execution so a suppressed call is visible in the log, not
    just in the returned value — a cache hit returns a *plausible* result, so
    the execution list is the only honest witness."""

    def __init__(self, name: str, *, side_effecting: bool) -> None:
        self.name = name
        self.side_effecting = side_effecting
        self.executions: list[dict[str, Any]] = []

    async def run(self, arguments: dict[str, Any], ctx: Any) -> dict[str, Any]:
        self.executions.append(dict(arguments))
        return {"tool": self.name, "n": len(self.executions), **arguments}


def _wire(*middleware: Any) -> tuple[Invoker, RunContext]:
    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=list(middleware) or [memoize(store=store)])
    ctx = RunContext("run", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
    return inv, ctx


def _twice(inv: Invoker, ctx: RunContext, req: Any) -> tuple[Any, Any]:
    async def go() -> tuple[Any, Any]:
        return await inv.invoke_tool(req(), ctx), await inv.invoke_tool(req(), ctx)

    return asyncio.run(go())


def test_a_side_effecting_tool_is_never_served_from_cache() -> None:
    """The load-bearing case: two `send_email` calls sent ONE email before the
    fix, and the caller was told the second one succeeded."""
    inv, ctx = _wire()
    mailer = _CountingTool("send_email", side_effecting=True)
    args = {"to": "a@b.c", "body": "hi"}

    a, b = _twice(inv, ctx, lambda: ToolRequest("send_email", dict(args), mailer, side_effecting=True))

    assert mailer.executions == [args, args], "a side effect was suppressed by the cache"
    assert a["n"] == 1 and b["n"] == 2, "the second caller was handed a stale result"


def test_side_effecting_is_read_off_the_TOOL_when_the_request_omits_it() -> None:
    """`ToolRequest.side_effecting` defaults to False and most call sites build
    the request positionally (`ToolRequest(name, args, tool)`) — only ReAct
    copies the flag across. The tool object is the declaration of record
    (`side_effecting` is a REQUIRED field on `FunctionTool`), so reading the
    request alone would have fixed only half of the reproduction."""
    inv, ctx = _wire()
    mailer = _CountingTool("send_email", side_effecting=True)

    _twice(inv, ctx, lambda: ToolRequest("send_email", {"to": "a@b.c"}, mailer))

    assert len(mailer.executions) == 2, "the tool's own declaration was ignored"


def test_a_read_only_tool_IS_still_memoized() -> None:
    """POSITIVE CONTROL. A "fix" that simply stops caching tool calls passes
    every assertion above and fails right here: an identical read-only call must
    still execute exactly ONCE. `memoize` is advertised for read-only tool reuse
    and that has to keep working."""
    inv, ctx = _wire()
    lookup = _CountingTool("lookup", side_effecting=False)

    a, b = _twice(inv, ctx, lambda: ToolRequest("lookup", {"q": "revenue"}, lookup))

    assert lookup.executions == [{"q": "revenue"}], "a read-only tool call was not memoized"
    assert a == b


def test_an_undeclared_tool_stays_cacheable() -> None:
    """POSITIVE CONTROL, second shape. "Unknown ⇒ dangerous" would disable the
    cache for every duck-typed tool that never opted in (test doubles, ad-hoc
    objects with no `side_effecting` attribute at all). Silence means read-only,
    matching `ToolRequest`'s own default."""
    inv, ctx = _wire()
    plain = type("P", (), {"name": "plain", "executions": [], "run": None})()
    calls: list[dict[str, Any]] = []

    async def run(arguments: dict[str, Any], ctx_: Any) -> dict[str, Any]:
        calls.append(dict(arguments))
        return {"ok": True}

    plain.run = run  # type: ignore[assignment]
    assert not hasattr(plain, "side_effecting")

    _twice(inv, ctx, lambda: ToolRequest("plain", {"x": 1}, plain))

    assert calls == [{"x": 1}], "an undeclared tool stopped being cacheable"


def test_the_same_side_effecting_tool_with_different_args_also_runs_twice() -> None:
    """Not a key-collision fix in disguise: distinct arguments already produce
    distinct keys, so this would pass even with the bug present. It pins that
    the guard is about the ACT, not about key precision."""
    inv, ctx = _wire()
    mailer = _CountingTool("send_email", side_effecting=True)

    async def go() -> None:
        await inv.invoke_tool(ToolRequest("send_email", {"to": "a@b.c"}, mailer), ctx)
        await inv.invoke_tool(ToolRequest("send_email", {"to": "d@e.f"}, mailer), ctx)

    asyncio.run(go())
    assert mailer.executions == [{"to": "a@b.c"}, {"to": "d@e.f"}]


def test_a_when_predicate_cannot_re_enable_side_effect_caching() -> None:
    """`when=` is ANDed with the side-effect guard, never a substitute for it.
    A caller filtering on something unrelated (say "only cache in this scope")
    must not silently opt back into caching mutations."""
    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=[memoize(store=store, when=lambda c: True)])
    ctx = RunContext("run", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
    mailer = _CountingTool("send_email", side_effecting=True)

    _twice(inv, ctx, lambda: ToolRequest("send_email", {"to": "a@b.c"}, mailer))

    assert len(mailer.executions) == 2, "when=lambda: True re-enabled side-effect caching"


def test_idempotent_still_dedupes_side_effecting_calls_within_a_run() -> None:
    """The deliberate case stays reachable. `idempotent()` is the ONE caller of
    `allow_side_effects=True`; it is safe because its key is pinned to
    `correlation_id`, so replay is confined to a retry of *this* call."""
    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=[idempotent(store=store)])
    ctx = RunContext("run-1", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
    mailer = _CountingTool("send_email", side_effecting=True)

    _twice(inv, ctx, lambda: ToolRequest("send_email", {"to": "a@b.c"}, mailer, side_effecting=True))

    assert len(mailer.executions) == 1, "idempotent() lost its dedupe"


def test_idempotent_reads_the_tool_declaration_too() -> None:
    """`idempotent`'s `when` used `getattr(request, "side_effecting", False)`
    alone, so a positionally-built request got NO idempotency for a tool that
    declares itself side-effecting — the exact call sites the guard above
    protects were also the ones idempotency skipped."""
    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=[idempotent(store=store)])
    ctx = RunContext("run-1", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
    mailer = _CountingTool("send_email", side_effecting=True)

    _twice(inv, ctx, lambda: ToolRequest("send_email", {"to": "a@b.c"}, mailer))

    assert len(mailer.executions) == 1, "idempotent() ignored the tool's own declaration"


def test_idempotent_leaves_read_only_calls_alone() -> None:
    """The complement: `idempotent()` is for mutations. A read-only call passes
    straight through it (`memoize()` is the middleware that caches those), so a
    run-scoped entry is never spent on a call that didn't need one."""
    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=[idempotent(store=store)])
    ctx = RunContext("run-1", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
    lookup = _CountingTool("lookup", side_effecting=False)

    _twice(inv, ctx, lambda: ToolRequest("lookup", {"q": "x"}, lookup))

    assert len(lookup.executions) == 2


def test_a_side_effecting_tool_with_unhashable_mutable_arguments_still_runs_twice() -> None:
    """Edge case: the guard must fire BEFORE the key is computed, so arguments
    that a naive `hash()` would reject (nested lists, sets, dicts) can't turn a
    suppressed side effect into a suppressed-and-crashed one. The same mutable
    dict object is reused across both calls, so an in-place mutation between
    them cannot be laundered into a hit either."""
    inv, ctx = _wire()
    mailer = _CountingTool("notify", side_effecting=True)
    shared: dict[str, Any] = {"to": ["a@b.c"], "tags": {"urgent", "ops"}, "meta": {"x": [1, 2]}}

    async def go() -> None:
        await inv.invoke_tool(ToolRequest("notify", shared, mailer), ctx)
        shared["to"].append("d@e.f")
        await inv.invoke_tool(ToolRequest("notify", shared, mailer), ctx)

    asyncio.run(go())
    assert len(mailer.executions) == 2


def test_a_read_only_tool_with_unhashable_arguments_is_still_memoized() -> None:
    """POSITIVE CONTROL for the edge case above: unhashable arguments must not
    quietly disable the read-only cache either. `stable_hash` JSON-encodes them
    (sets sorted, keys sorted), so the two calls share one entry."""
    inv, ctx = _wire()
    lookup = _CountingTool("search", side_effecting=False)

    async def go() -> None:
        for args in ({"q": ["a", "b"], "f": {"z": 1, "a": 2}}, {"f": {"a": 2, "z": 1}, "q": ["a", "b"]}):
            await inv.invoke_tool(ToolRequest("search", args, lookup), ctx)

    asyncio.run(go())
    assert len(lookup.executions) == 1, "unhashable-but-equal arguments produced two entries"


def test_semantic_memoize_never_short_circuits_a_side_effecting_tool() -> None:
    """The same hole, one function over: `semantic_memoize`'s docstring said
    "guard with `when=` (defaults to chat calls / explicitly read-only)" while
    `when` defaulted to `None` — no guard at all, and worse than the exact
    cache because a NEAR-duplicate is enough to suppress the action. The guard
    is enforced now rather than requested."""
    from agentkit.middlewares import semantic_memoize

    class _AlwaysHits:
        """A VectorPort stub whose every search is a perfect match."""

        async def search(self, scope: Any, query: str, k: int = 1) -> list[Any]:
            hit = type("C", (), {"id": "c1", "metadata": {"content": "already sent", "model": "m"}})()
            return [(1.0, hit)]

        async def upsert(self, scope: Any, chunks: Any) -> None:
            return None

    store = InMemoryStore()
    inv = Invoker(llm=None, tool_middleware=[semantic_memoize(vector=_AlwaysHits())])
    ctx = RunContext("run", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))
    mailer = _CountingTool("send_email", side_effecting=True)

    _twice(inv, ctx, lambda: ToolRequest("send_email", {"to": "a@b.c"}, mailer))

    assert len(mailer.executions) == 2, "a semantic hit masked a real side effect"
