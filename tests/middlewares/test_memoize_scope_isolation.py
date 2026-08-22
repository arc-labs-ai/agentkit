"""`memoize` must not leak a cached answer across a tenant boundary.

`Scope`'s own docstring calls itself the key "threaded through every memory
recall / cache key / meter / callback". `memoize` took an arbitrary `key`
callable and added nothing to it, so isolation depended on every caller
remembering — and the key the cheatsheet and the LangChain migration guide
taught was `lambda c: c.request.messages[-1].content`, which ignores the model,
the tools, the temperature AND the tenant.

Measured before the fix: two tenants asking the same question, one provider
call, and tenant 999 receiving tenant 1's answer.

A tenant-isolation boundary that relies on a caller-supplied key is not a
boundary, so the partitioning now happens inside the middleware.
"""

from __future__ import annotations

import asyncio

from agentkit.adapters.store import InMemoryStore
from agentkit.kernel.types import ChatRequest, Delta, Message, Scope, Usage
from agentkit.middlewares import memoize
from agentkit.middlewares.memoize import default_key
from agentkit.runtime import Invoker, RunContext, Services


class _PerCallLLM:
    """Returns a DISTINCT body per call, so a cache hit is unmistakable: if two
    tenants see the same string, they shared an entry."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, **_kw):
        self.calls += 1
        yield Delta(text=f"answer-{self.calls}", model="m", provider="f")
        yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")


def _ask(inv: Invoker, scope: Scope, text: str = "what is our revenue?"):
    ctx = RunContext("run", scope, services=Services(invoker=inv))
    req = ChatRequest(messages=[Message("user", text)], model="m")
    return asyncio.run(inv.chat(req, ctx))


def test_two_tenants_never_share_a_cache_entry() -> None:
    """The load-bearing test, using the exact key the docs used to teach."""
    store = InMemoryStore()
    llm = _PerCallLLM()
    inv = Invoker(
        llm=llm,
        chat_middleware=[
            memoize(key=lambda c: c.request.messages[-1].content or "", store=store)
        ],
    )

    a = _ask(inv, Scope(org_id=1))
    b = _ask(inv, Scope(org_id=999))

    assert a.content != b.content, "a cached answer crossed a tenant boundary"
    assert llm.calls == 2, "the second tenant was served from the first tenant's entry"


def test_the_same_tenant_still_gets_a_cache_hit() -> None:
    """Partitioning must not defeat caching — within one tenant the entry is
    reused, which is the whole point of the middleware."""
    store = InMemoryStore()
    llm = _PerCallLLM()
    inv = Invoker(llm=llm, chat_middleware=[memoize(store=store)])

    a = _ask(inv, Scope(org_id=1))
    b = _ask(inv, Scope(org_id=1))

    assert a.content == b.content
    assert llm.calls == 1, "an identical call within one tenant was not cached"


def test_domain_is_part_of_the_partition_not_just_org() -> None:
    """`Scope.key()` is `org:domain`; both halves must partition, or a
    per-domain tenant boundary silently collapses."""
    store = InMemoryStore()
    llm = _PerCallLLM()
    inv = Invoker(llm=llm, chat_middleware=[memoize(store=store)])

    a = _ask(inv, Scope(org_id=1, domain_id=1))
    b = _ask(inv, Scope(org_id=1, domain_id=2))

    assert a.content != b.content
    assert llm.calls == 2


def test_memoize_takes_no_required_arguments() -> None:
    """Two docs pages showed a bare `memoize()`; it raised
    `TypeError: missing 1 required keyword-only argument: 'key'`. Requiring a
    key also pushed the most dangerous decision in a cache — what counts as
    "the same call" — onto every caller."""
    memoize()  # must not raise


def test_the_default_key_covers_every_field_that_changes_the_answer() -> None:
    """The key the docs taught ignored the model, the tools, the temperature
    and the tenant. A cache that ignores the model serves a haiku answer to a
    sonnet request."""
    base = dict(messages=[Message("user", "hi")], model="m")

    def key_for(**over):
        req = ChatRequest(**{**base, **over})
        call = type("C", (), {"request": req})()
        return default_key(call)

    reference = key_for()
    assert key_for(model="other") != reference, "model must change the key"
    assert key_for(temperature=0.9) != reference, "temperature must change the key"
    assert key_for(max_tokens=10) != reference, "max_tokens must change the key"
    assert key_for(response_format={"type": "json_object"}) != reference, (
        "response_format must change the key"
    )
    assert key_for(messages=[Message("user", "different")]) != reference
    # ...and an identical request is stable across calls, or nothing ever hits.
    assert key_for() == reference


def test_the_default_key_ignores_tool_description_churn() -> None:
    """Tools reduce to their NAMES: two registries advertising the same tools
    should share an entry, and editing a schema's prose does not change the
    answer."""
    from agentkit.kernel.types import ToolSchema

    def key_with(tools):
        req = ChatRequest(messages=[Message("user", "hi")], model="m", tools=tools)
        return default_key(type("C", (), {"request": req})())

    a = key_with([ToolSchema(name="search", description="find things")])
    b = key_with([ToolSchema(name="search", description="COMPLETELY REWRITTEN")])
    c = key_with([ToolSchema(name="other", description="find things")])
    assert a == b, "a description edit should not invalidate the cache"
    assert a != c, "a different tool set must change the key"


# ── egress must not be constructible inert ──────────────────────────────────


def test_egress_refuses_a_missing_guardrail() -> None:
    """`egress(None)` constructed a middleware that sat in the chain and
    checked nothing — every SSRF and allowlist check silently off, with no
    signal. `egress(config.guardrail)` with an unset config is exactly how that
    happens. A security control that can be built inert is worse than an absent
    one, because the chain looks guarded.
    """
    import pytest

    from agentkit.middlewares import egress

    with pytest.raises(ValueError, match="requires a Guardrail"):
        egress(None)


def test_egress_refuses_an_object_that_cannot_check_urls() -> None:
    """Duck-typing is fine until the duck has no beak: an object without
    `check_url` would never check anything."""
    import pytest

    from agentkit.middlewares import egress

    with pytest.raises(TypeError, match="check_url"):
        egress(object())


def test_egress_still_blocks_a_blocked_url() -> None:
    """The control itself, end to end, so the constructor guard cannot be
    mistaken for the whole feature."""
    import pytest

    from agentkit.capabilities.guardrails.base import Guardrail
    from agentkit.middlewares import egress

    mw = egress(Guardrail(egress_allow=("example.com",)))

    class _Req:
        url_arg = "url"
        arguments = {"url": "http://169.254.169.254/latest/meta-data/"}

    class _Ctx:
        request = _Req()

    with pytest.raises(PermissionError):
        asyncio.run(mw.on_request(_Ctx()))
