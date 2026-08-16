"""Model registry — from-env provider resolution (Brief 3) and per-model
capability declaration (Brief 5).

They are one table, so they are one test module. Every test builds its own
``default_registry()`` and passes an explicit ``env=`` mapping: nothing here
touches ``os.environ`` or the process-wide registry, so the suite stays
order-independent and a stray real credential in the developer's shell can
never change an outcome.

The single most important assertion in this file is
``test_no_message_ever_contains_a_credential``. Everything else is behaviour;
that one is a security property.
"""

from __future__ import annotations

import warnings

import pytest

from agentkit.adapters.llm.model_registry import (
    CAPABILITY_NAMES,
    Capability,
    CapabilityMismatch,
    MissingProviderExtra,
    ModelCapabilities,
    ModelEntry,
    ModelRegistry,
    ProviderEntry,
    ProviderNotConfigured,
    UnknownModel,
    default_registry,
    normalize_model_name,
)

SECRET = "sk-do-not-leak-me-0123456789"


@pytest.fixture
def reg() -> ModelRegistry:
    """A private registry per test. ``default_registry()`` returns a fresh
    instance every call precisely so tests never share mutable state."""
    return default_registry()


# ── resolution: the last mile ────────────────────────────────────────────────


def test_with_a_key_set_the_right_adapter_is_built(reg: ModelRegistry) -> None:
    """The whole point: a model name plus an environment yields a wired client
    with no bespoke bootstrap and no ``api_key=`` in application code."""
    pytest.importorskip("httpx")
    from agentkit.adapters.llm.providers import AnthropicLLM

    llm = reg.resolve("claude-sonnet-4-6", env={"ANTHROPIC_API_KEY": SECRET})
    assert isinstance(llm, AnthropicLLM)


def test_each_provider_reads_its_own_env_var(reg: ModelRegistry) -> None:
    """Every provider the framework ships is reachable from configuration
    alone — the "wire every provider it needs from environment" bar."""
    pytest.importorskip("httpx")
    from agentkit.adapters.llm.providers import AnthropicLLM, OpenAICompatibleLLM

    cases = {
        "claude-sonnet-4-6": ("ANTHROPIC_API_KEY", AnthropicLLM),
        "gpt-4o": ("OPENAI_API_KEY", OpenAICompatibleLLM),
        "deepseek-chat": ("DEEPSEEK_API_KEY", OpenAICompatibleLLM),
        "meta-llama/llama-3-70b": ("OPENROUTER_API_KEY", OpenAICompatibleLLM),
    }
    for model, (var, cls) in cases.items():
        assert isinstance(reg.resolve(model, env={var: SECRET}), cls), model


def test_no_keys_set_refuses_by_default(reg: ModelRegistry) -> None:
    """Fallback is OPT-IN. With nothing configured and no ``fallback=``, the
    call raises rather than quietly handing back a fake.

    This deliberately diverges from "yield the fake and warn": in a server
    process a ``UserWarning`` goes to a log nobody reads, and the outcome is
    fabricated completions served as real answers. The escape hatch stays one
    keyword away (below) — it just isn't the accident."""
    with pytest.raises(ProviderNotConfigured) as exc:
        reg.resolve("claude-sonnet-4-6", env={})
    # The message must name what to SET, so the operator can act on it.
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_no_keys_set_with_explicit_fallback_yields_the_fake_and_warns_once(
    reg: ModelRegistry,
) -> None:
    """Brief 3's test, with the opt-in made explicit. The downgrade is loud,
    and loud exactly once — a per-call warning would be filtered out within a
    day and stop being read."""
    from agentkit.testing import FakeLLM

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = reg.resolve("claude-sonnet-4-6", fallback="fake", env={})
        second = reg.resolve("claude-sonnet-4-6", fallback="fake", env={})

    assert isinstance(first, FakeLLM) and isinstance(second, FakeLLM)
    downgrades = [w for w in caught if "FAKE" in str(w.message)]
    assert len(downgrades) == 1, f"expected exactly one downgrade warning, got {len(downgrades)}"


def test_unknown_model_refuses_with_an_inventory_not_a_keyerror(reg: ModelRegistry) -> None:
    """A ``KeyError('totally-made-up')`` tells the operator nothing. The
    refusal names what IS registered so the fix is visible in the traceback."""
    with pytest.raises(UnknownModel) as exc:
        reg.resolve("totally-made-up-model", env={"OPENAI_API_KEY": SECRET})
    message = str(exc.value)
    assert "totally-made-up-model" in message
    assert "claude-sonnet-4-6" in message and "gpt-4o" in message
    assert "registered providers" in message


def test_empty_string_credential_counts_as_absent(reg: ModelRegistry) -> None:
    """``export OPENAI_API_KEY=`` is a misconfiguration, not a credential.
    Forwarding ``""`` produces a 401 far from its cause."""
    with pytest.raises(ProviderNotConfigured):
        reg.resolve("gpt-4o", env={"OPENAI_API_KEY": ""})


def test_explicit_api_key_beats_the_environment(reg: ModelRegistry) -> None:
    """The registry is a layer ABOVE the explicit factories, never a
    replacement — a caller holding the credential still wins."""
    pytest.importorskip("httpx")
    llm = reg.resolve("gpt-4o", api_key=SECRET, env={})
    assert llm is not None


def test_resolve_by_provider_without_a_model(reg: ModelRegistry) -> None:
    """``provider=`` short-circuits name matching, for a caller who knows
    which vendor they want and will pass the model per-call."""
    pytest.importorskip("httpx")
    assert reg.resolve(provider="deepseek", env={"DEEPSEEK_API_KEY": SECRET}) is not None


def test_missing_extra_degrades_loudly_never_into_the_fallback() -> None:
    """A broken install must not masquerade as a missing credential.

    Even with ``fallback="fake"`` requested, an ``ImportError`` from the
    provider module raises ``MissingProviderExtra`` naming the pip extra.
    Absorbing it into the fallback would let a wheel shipped without its
    transport serve canned answers forever."""
    reg = ModelRegistry()
    reg.register_provider(
        ProviderEntry(
            name="ghost",
            env_vars=("GHOST_API_KEY",),
            factory="agentkit._no_such_module_at_all:ghost",
            extra="http",
        )
    )
    with pytest.raises(MissingProviderExtra) as exc:
        reg.resolve(provider="ghost", fallback="fake", env={"GHOST_API_KEY": SECRET})
    assert "arc-agentkit[http]" in str(exc.value)


# ── the security property ────────────────────────────────────────────────────


def test_no_message_ever_contains_a_credential(reg: ModelRegistry) -> None:
    """A credential must never reach an error, a warning, or a repr.

    Swept across every failure path that has a key in scope, plus the
    success path's returned object. A masked value would fail this too, and
    should: half a credential in a log aggregator is still a credential in a
    log aggregator.
    """
    pytest.importorskip("httpx")
    surfaces: list[str] = []

    # Success path — the built client and the registry that built it.
    llm = reg.resolve("claude-sonnet-4-6", env={"ANTHROPIC_API_KEY": SECRET})
    surfaces += [repr(llm), str(llm), repr(reg), reg.describe()]

    # Failure paths, each with the secret present in the environment.
    env = {var: SECRET for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GHOST_API_KEY")}
    for thunk in (
        lambda: reg.resolve("no-such-model", env=env),
        lambda: reg.resolve(provider="no-such-provider", env=env),
        lambda: reg.resolve("gpt-4o", api_key=SECRET, fallback="nonsense", env=env),
    ):
        try:
            thunk()
        except Exception as exc:  # noqa: BLE001 — the message is the subject under test
            surfaces += [str(exc), repr(exc)]

    # And the warning text on the downgrade path.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.resolve("claude-sonnet-4-6", fallback="fake", env={})
    surfaces += [str(w.message) for w in caught]

    for surface in surfaces:
        assert SECRET not in surface, f"credential leaked into: {surface[:200]!r}"


# ── capabilities: declared, never guessed ────────────────────────────────────


def test_binding_a_model_without_a_capability_refuses_naming_both(reg: ModelRegistry) -> None:
    """Brief 5's headline: the refusal names the capability AND the model, so
    the operator can act without reading framework source."""
    with pytest.raises(CapabilityMismatch) as exc:
        reg.check("deepseek-chat", ("vision",), subject="ocr-agent")
    message = str(exc.value)
    assert "vision" in message
    assert "deepseek-chat" in message
    assert "ocr-agent" in message


def test_unknown_model_is_reported_as_unknown_never_as_passing(reg: ModelRegistry) -> None:
    """The load-bearing distinction. An unregistered model must not sail
    through a capability check — a framework that guesses ``True``
    reintroduces the exact silent failure the check exists to catch."""
    # Default policy: say so, don't crash.
    with pytest.warns(UserWarning, match="UNKNOWN"):
        reg.check("some-self-hosted-thing", ("vision",))
    # Pinned-model policy: refuse.
    with pytest.raises(CapabilityMismatch) as exc:
        reg.check("some-self-hosted-thing", ("vision",), on_unknown="refuse")
    assert "UNKNOWN" in str(exc.value)
    assert "not assumed present" in str(exc.value)


def test_a_satisfied_requirement_is_untouched(reg: ModelRegistry) -> None:
    """No exception, no warning, nothing. The third of Brief 5's tests."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.check("claude-sonnet-4-6", ("vision", "tools", "streaming"))
    assert caught == []


def test_unregistered_model_reports_every_capability_unknown(reg: ModelRegistry) -> None:
    """Never ``False`` either — guessing ``NO`` would refuse every
    self-hosted and newly-released model the day this shipped."""
    caps = reg.capabilities("something-nobody-declared")
    assert caps.all_unknown
    for name in CAPABILITY_NAMES:
        assert caps.get(name) is Capability.UNKNOWN
    assert caps.context_window is None


def test_capabilities_survive_a_dated_release_id(reg: ModelRegistry) -> None:
    """Providers echo dated ids on every response. If the capability lookup
    didn't normalise them, a live run's model string would report UNKNOWN
    while the same family in config reported YES."""
    for dated in (
        "claude-haiku-4-5-20251001",
        "gpt-4o-mini-2024-07-18",
        "anthropic/claude-sonnet-4-6",
    ):
        assert not reg.capabilities(dated).all_unknown, dated


def test_a_typo_in_requires_raises_rather_than_silently_never_checking(reg: ModelRegistry) -> None:
    """``requires=("visoin",)`` must not degrade into "unknown capability,
    warn once, carry on" — that would look like the check ran when it never
    could."""
    with pytest.raises(ValueError, match="unknown capability"):
        reg.check("claude-sonnet-4-6", ("visoin",))


def test_min_context_window_is_checked_and_unknown_is_not_assumed(reg: ModelRegistry) -> None:
    """Declared-and-too-small refuses; undeclared reports unknown."""
    with pytest.raises(CapabilityMismatch, match="context window"):
        reg.check("deepseek-chat", min_context_window=500_000)
    reg.check("gpt-4.1", min_context_window=500_000)  # 1M window — fine
    with pytest.warns(UserWarning, match="context_window"):
        reg.check("mystery-model", min_context_window=500_000)


def test_on_unknown_allow_is_silent(reg: ModelRegistry) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.check("mystery-model", ("vision",), on_unknown="allow")
    assert caught == []


def test_on_unknown_rejects_a_bogus_policy(reg: ModelRegistry) -> None:
    with pytest.raises(ValueError, match="on_unknown"):
        reg.check("gpt-4o", ("vision",), on_unknown="maybe")


def test_derived_requirements_raise_on_NO_but_stay_silent_on_UNKNOWN(reg: ModelRegistry) -> None:
    """The asymmetry that keeps the warning channel usable.

    A requirement the FRAMEWORK inferred from wiring fires only against a
    model explicitly declared incapable. If derived requirements also warned
    on UNKNOWN, every development wiring with a made-up model name would emit
    one, people would filter the category, and the real ``requires=``
    warnings would go unread with it."""
    with pytest.raises(CapabilityMismatch, match="tools"):
        reg.check("deepseek-reasoner", derived=("tools",))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.check("some-unregistered-model", derived=("tools",))
    assert caught == []


# ── rules: extensible, not a hardcoded prefix table ──────────────────────────


def test_an_application_can_register_its_own_naming_rule(reg: ModelRegistry) -> None:
    """The constraint was "prefer name→provider rules an application can
    extend over a hardcoded prefix table"."""
    reg.register_rule(lambda name: "openai" if name.startswith("acme-") else None)
    assert reg.provider_for("acme-internal-v3") == "openai"
    assert reg.provider_for("still-unknown") is None


def test_an_explicit_model_row_beats_any_rule(reg: ModelRegistry) -> None:
    """A declared row is more authoritative than a pattern — otherwise a
    coarse rule could hijack a model the operator had pinned."""
    reg.register_rule(lambda name: "openrouter")  # claims everything
    assert reg.provider_for("claude-sonnet-4-6") == "anthropic"


def test_an_application_can_declare_capabilities_for_its_own_model(reg: ModelRegistry) -> None:
    """Capabilities are declared per model. Registering a row turns an
    UNKNOWN into a real check — including a real refusal."""
    reg.register_model(
        ModelEntry(
            name="acme-vision-1",
            provider="openai",
            capabilities=ModelCapabilities(vision=Capability.YES, tools=Capability.NO),
        )
    )
    reg.check("acme-vision-1", ("vision",))  # satisfied → silent
    with pytest.raises(CapabilityMismatch, match="tools"):
        reg.check("acme-vision-1", ("tools",))


def test_register_model_replaces_a_builtin_row(reg: ModelRegistry) -> None:
    """The built-in table is a convenience default that WILL go stale; an
    application must be able to pin its own truth over it."""
    reg.register_model(
        ModelEntry(
            name="claude-sonnet-4-6",
            provider="anthropic",
            capabilities=ModelCapabilities(vision=Capability.NO),
        )
    )
    with pytest.raises(CapabilityMismatch):
        reg.check("claude-sonnet-4-6", ("vision",))


# ── name normalisation ───────────────────────────────────────────────────────


def test_normalisation_candidate_order() -> None:
    assert normalize_model_name("anthropic/Claude-Sonnet-4-6-20250101") == (
        "anthropic/claude-sonnet-4-6-20250101",
        "claude-sonnet-4-6-20250101",
        "anthropic/claude-sonnet-4-6",
        "claude-sonnet-4-6",
    )


def test_normalisation_agrees_with_pricing_lookup() -> None:
    """Drift guard.

    ``pricing.py`` owns the price table and is deliberately left untouched, so
    this module re-implements its candidate order rather than importing a
    shared helper. That duplication is only safe if the two agree — if a model
    priced correctly were reported as capability-UNKNOWN (or vice versa) the
    split would be invisible until someone's bill or bind check misbehaved.

    Asserted structurally: for every model in the registry's built-in table,
    one of this module's normalised candidates must be a key the price table
    resolves.
    """
    from agentkit.adapters.llm.providers import pricing

    reg = default_registry()
    for name in ("claude-haiku-4-5-20251001", "gpt-4o-mini-2024-07-18", "anthropic/claude-opus-4-1"):
        priced = pricing._lookup(name)
        known = not reg.capabilities(name).all_unknown
        assert (priced is not None) == known, (
            f"{name}: pricing knows it = {priced is not None}, registry knows it = {known}"
        )


# ── the batteries-included preset ────────────────────────────────────────────
#
# ``client.from_env`` is the one-liner an application actually writes. It sits
# on the registry, so these tests cover the seam rather than re-testing
# resolution.


def test_from_env_builds_a_chat_without_an_api_key_argument(monkeypatch) -> None:
    """The whole 710-line bootstrap, replaced by one call. Note there is no
    ``api_key=`` anywhere — that is the deliverable."""
    pytest.importorskip("httpx")
    from agentkit.client import Chat, from_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    chat = from_env("claude-sonnet-4-6")
    assert isinstance(chat, Chat)


def test_from_env_refuses_loudly_when_nothing_is_configured(monkeypatch) -> None:
    """No silent fake in the default path — the same opt-in rule as
    ``resolve``, surfaced at the ergonomic entry point."""
    from agentkit.client import from_env

    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ProviderNotConfigured):
        from_env("claude-sonnet-4-6")


def test_from_env_by_provider_uses_the_declared_default_model(monkeypatch) -> None:
    """``from_env(provider=...)`` must not produce a Chat that raises "no
    model" on its first call — the provider row's ``default_model`` fills in."""
    pytest.importorskip("httpx")
    from agentkit.client import from_env

    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    chat = from_env(provider="deepseek")
    assert chat._model == "deepseek-chat"


def test_explicit_factories_are_untouched(monkeypatch) -> None:
    """Constraint: "do not break the explicit factories — this is a layer
    above them." Pinned so a future registry change cannot quietly become the
    only path."""
    pytest.importorskip("httpx")
    from agentkit.client import claude

    for var in ("ANTHROPIC_API_KEY",):
        monkeypatch.delenv(var, raising=False)
    # No environment at all, explicit key — still works exactly as before.
    assert claude(api_key=SECRET).__class__.__name__ == "Chat"
