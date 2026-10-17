"""The model registry: normalisation, capability checking and resolution.

This module had no tests at all (223 statements, 31% covered), and four
catalogue entries were parked pointing at this exact file after the
reorganisation deleted it.

The themes worth naming, because each is a decision the module argues for in
prose and only a test can hold:

- an unregistered model is UNKNOWN, never a guess in either direction
- a declared NO is fatal; an UNKNOWN is a policy question
- falling back to a fake LLM is opt-in, and a BROKEN INSTALL must never
  masquerade as a missing credential
- an explicit model row always beats a pattern rule
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from agentkit.models.catalog.registry import (
    CAPABILITY_NAMES,
    Capability,
    CapabilityMismatch,
    MissingProviderExtra,
    ModelCapabilities,
    ModelEntry,
    ModelRegistry,
    ProviderEntry,
    ProviderNotConfigured,
    RegistryError,
    UnknownModel,
    normalize_model_name,
)

# `dict` is a convenient real factory: it is importable by dotted path and its
# return value records exactly which kwargs the registry passed.
_DICT_FACTORY = "builtins:dict"


def _registry(**provider_kwargs: Any) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register_provider(
        ProviderEntry(
            name="acme",
            env_vars=("ACME_API_KEY", "ACME_TOKEN"),
            factory=_DICT_FACTORY,
            **provider_kwargs,
        )
    )
    return registry


# ── normalisation ────────────────────────────────────────────────────────────


def test_the_documented_candidate_order_is_exact() -> None:
    """Most specific first: exact, then without the OpenRouter-style prefix,
    then without the trailing date, then without both. Order is the contract —
    a caller takes the first hit, so reordering changes which row wins."""
    assert normalize_model_name("anthropic/claude-sonnet-4-6-20250101") == (
        "anthropic/claude-sonnet-4-6-20250101",
        "claude-sonnet-4-6-20250101",
        "anthropic/claude-sonnet-4-6",
        "claude-sonnet-4-6",
    )


@pytest.mark.parametrize(
    ("dated", "undated"),
    [
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),  # Anthropic: undashed
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),  # OpenAI: dashed
    ],
)
def test_both_provider_date_shapes_are_stripped(dated: str, undated: str) -> None:
    """Providers echo a dated release id on every response. If only one shape
    were handled, half of all real responses would miss the registry row for
    the model that produced them."""
    assert undated in normalize_model_name(dated)


def test_candidates_are_deduplicated_and_never_empty() -> None:
    """An undated, unprefixed name collapses to one candidate rather than four
    copies, and no candidate is the empty string — an empty key would match a
    dict entry nobody registered."""
    assert normalize_model_name("gpt-4o") == ("gpt-4o",)
    assert all(normalize_model_name("  GPT-4O  "))
    assert normalize_model_name("  GPT-4O  ") == ("gpt-4o",)


# ── capabilities ─────────────────────────────────────────────────────────────


def test_everything_defaults_to_unknown_so_a_partial_row_cannot_over_claim() -> None:
    caps = ModelCapabilities()
    assert caps.all_unknown is True
    assert all(caps.get(name) is Capability.UNKNOWN for name in CAPABILITY_NAMES)
    assert caps.context_window is None


def test_one_declared_capability_means_the_row_is_no_longer_a_placeholder() -> None:
    assert ModelCapabilities(tools=Capability.NO).all_unknown is False


def test_an_unrecognised_capability_name_raises_instead_of_reading_unknown() -> None:
    """A typo in `requires=` is a programming error. Degrading it to UNKNOWN
    would mean the check silently never ran — the caller asked for a guarantee
    and got nothing, with no indication."""
    with pytest.raises(ValueError, match="unknown capability"):
        ModelCapabilities().get("visoin")


def test_context_window_is_not_a_tri_state_capability() -> None:
    """It is an int, requested through `min_context_window=`, so it is
    deliberately absent from CAPABILITY_NAMES."""
    assert "context_window" not in CAPABILITY_NAMES
    with pytest.raises(ValueError, match="unknown capability"):
        ModelCapabilities().get("context_window")


# ── registration and lookup ──────────────────────────────────────────────────


def test_registration_is_chainable() -> None:
    registry = ModelRegistry()
    returned = registry.register_provider(
        ProviderEntry(name="acme", env_vars=("K",), factory=_DICT_FACTORY)
    ).register_model(ModelEntry(name="acme-1", provider="acme")).register_rule(lambda n: None)
    assert returned is registry


def test_a_dated_response_name_finds_the_undated_row() -> None:
    registry = _registry()
    registry.register_model(
        ModelEntry("acme-1", "acme", ModelCapabilities(tools=Capability.YES))
    )
    assert registry.entry_for("acme-1-20250101") is not None
    assert registry.capabilities("acme-1-20250101").tools is Capability.YES


def test_an_unregistered_model_yields_an_all_unknown_placeholder() -> None:
    assert ModelRegistry().capabilities("something-nobody-registered").all_unknown is True
    assert ModelRegistry().entry_for("nope") is None


def test_an_explicit_model_row_beats_a_pattern_rule() -> None:
    registry = _registry()
    registry.register_rule(lambda name: "from-rule" if name.startswith("acme") else None)
    registry.register_model(ModelEntry("acme-1", "acme"))
    assert registry.provider_for("acme-1") == "acme"
    assert registry.provider_for("acme-2") == "from-rule"


def test_rules_are_the_outer_loop_so_a_specific_rule_is_not_shadowed() -> None:
    """The measured bug this ordering fixes: iterating candidates outermost let
    the coarsest rule that matched the RAW name decide, so a later, more
    specific rule was never offered the bare name.

    `provider_for('vendor/acme-9')` returned the prefix rule's answer because
    it matched the slash before the family rule saw `acme-9`. Rules outermost,
    most-recently-registered first, is the short-circuiting form of "last
    non-None wins", which is what register_rule promises.
    """
    registry = ModelRegistry()
    registry.register_rule(lambda n: "by-prefix" if "/" in n else None)
    registry.register_rule(lambda n: "by-family" if n.startswith("acme-") else None)

    assert registry.provider_for("vendor/acme-9") == "by-family"


def test_a_rule_matching_nothing_leaves_the_model_unclaimed() -> None:
    registry = ModelRegistry()
    registry.register_rule(lambda n: None)
    assert registry.provider_for("whatever") is None


def test_provider_entry_exposes_a_row_without_reaching_into_internals() -> None:
    registry = _registry(default_model="acme-1")
    entry = registry.provider_entry("acme")
    assert entry is not None and entry.default_model == "acme-1"
    assert registry.provider_entry("nobody") is None


def test_describe_lists_what_the_caller_could_have_asked_for() -> None:
    assert "(none)" in ModelRegistry().describe()
    registry = _registry()
    registry.register_model(ModelEntry("acme-1", "acme"))
    described = registry.describe()
    assert "acme" in described and "acme-1" in described


# ── capability checking ──────────────────────────────────────────────────────


def test_an_invalid_on_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_unknown must be"):
        ModelRegistry().check("m", ("tools",), on_unknown="maybe")


@pytest.mark.parametrize("kind", ["requires", "derived"])
def test_a_declared_no_is_fatal_however_the_requirement_arose(kind: str) -> None:
    """No policy makes a declared NO acceptable, so `requires` and `derived`
    are equally fatal here. The asymmetry between them is only about UNKNOWN."""
    registry = _registry()
    registry.register_model(
        ModelEntry("acme-1", "acme", ModelCapabilities(tools=Capability.NO))
    )
    with pytest.raises(CapabilityMismatch, match="unsupported"):
        registry.check("acme-1", **{kind: ("tools",)})


def test_an_unknown_derived_requirement_stays_silent() -> None:
    """The asymmetry, and the reason for it: a derived requirement is the
    framework's own inference, so warning about it for every unregistered model
    would put a warning on essentially every development wiring and teach
    people to filter the category out — at which point the real `requires=`
    warnings go unread too."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModelRegistry().check("unregistered", derived=("tools",))


def test_an_unknown_declared_requirement_warns_once_by_default() -> None:
    """Warn rather than refuse, because refusing on UNKNOWN would break every
    self-hosted, fine-tuned or newly-released model name the moment this
    shipped. Once per registry, because the same wiring is checked repeatedly
    and a warning repeated per call is a warning nobody reads."""
    registry = ModelRegistry()
    with pytest.warns(UserWarning, match="not in the model registry") as caught:
        registry.check("unregistered", ("tools",))
        registry.check("unregistered", ("tools",))
    assert len(caught) == 1


def test_refuse_makes_not_knowing_a_deployment_time_stop() -> None:
    with pytest.raises(CapabilityMismatch, match="UNKNOWN"):
        ModelRegistry().check("unregistered", ("tools",), on_unknown="refuse")


def test_allow_says_nothing_for_a_caller_who_verified_out_of_band() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModelRegistry().check("unregistered", ("tools",), on_unknown="allow")


def test_a_declared_context_window_below_the_minimum_is_refused() -> None:
    registry = _registry()
    registry.register_model(
        ModelEntry("acme-1", "acme", ModelCapabilities(context_window=8192))
    )
    with pytest.raises(CapabilityMismatch, match="context window"):
        registry.check("acme-1", min_context_window=200_000)
    registry.check("acme-1", min_context_window=4096)


def test_an_undeclared_context_window_joins_the_unknown_list() -> None:
    """Not knowing the window is the same category of ignorance as not knowing
    a capability, so it follows the same policy rather than passing silently."""
    with pytest.raises(CapabilityMismatch, match="context_window"):
        ModelRegistry().check("unregistered", min_context_window=1000, on_unknown="refuse")


def test_the_subject_points_the_message_at_the_wiring() -> None:
    """An error that names the agent is actionable; one that names the
    framework sends the reader into agentkit's source."""
    registry = _registry()
    registry.register_model(
        ModelEntry("acme-1", "acme", ModelCapabilities(tools=Capability.NO))
    )
    with pytest.raises(CapabilityMismatch, match="'researcher'"):
        registry.check("acme-1", ("tools",), subject="researcher")


# ── resolution ───────────────────────────────────────────────────────────────


def test_resolve_needs_a_model_or_a_provider() -> None:
    with pytest.raises(UnknownModel, match="needs a model or a provider"):
        ModelRegistry().resolve()


def test_an_unclaimed_model_raises_rather_than_guessing_a_provider() -> None:
    with pytest.raises(UnknownModel, match="no registered provider serves"):
        ModelRegistry().resolve("mystery-model", env={})


def test_an_unregistered_provider_name_raises_even_with_a_fallback() -> None:
    """A typo in `provider=` is not a degradable condition — there is nothing
    to degrade FROM, and silently returning a fake would hide the typo."""
    with pytest.raises(UnknownModel, match="no provider named"):
        ModelRegistry().resolve(provider="openai-typo", fallback="fake", env={})


def test_a_missing_credential_raises_by_default() -> None:
    """Opt-in fallback. In a server process a UserWarning goes to a log nobody
    reads, and the result is fabricated completions served as real ones."""
    registry = _registry()
    with pytest.raises(ProviderNotConfigured, match="ACME_API_KEY"):
        registry.resolve(provider="acme", env={})


def test_a_missing_credential_degrades_only_when_asked_and_warns_once() -> None:
    from agentkit.core.testing import FakeLLM

    registry = _registry()
    with pytest.warns(UserWarning, match="FAKE LLM") as caught:
        first = registry.resolve(provider="acme", fallback="fake", env={})
        second = registry.resolve(provider="acme", fallback="fake", env={})
    assert isinstance(first, FakeLLM) and isinstance(second, FakeLLM)
    assert len(caught) == 1


def test_the_fallback_reason_never_contains_a_credential() -> None:
    """It names the variables consulted, not their values — the message ends up
    in logs."""
    registry = _registry()
    with pytest.warns(UserWarning) as caught:
        registry.resolve(provider="acme", fallback="fake", env={"IRRELEVANT": "s3cret"})
    assert "s3cret" not in str(caught[0].message)
    assert "ACME_API_KEY" in str(caught[0].message)


def test_an_unsupported_fallback_value_is_refused() -> None:
    registry = _registry()
    with pytest.raises(RegistryError, match="unknown fallback"):
        registry.resolve(provider="acme", fallback="silently-do-nothing", env={})


def test_env_vars_are_tried_in_order_and_an_empty_string_is_absent() -> None:
    """A shell exporting `ACME_API_KEY=` is a misconfiguration, and passing ""
    through produces a confusing 401 far from the cause."""
    registry = _registry()
    built = registry.resolve(provider="acme", env={"ACME_API_KEY": "", "ACME_TOKEN": "second"})
    assert built["api_key"] == "second"


def test_an_explicit_api_key_beats_the_environment() -> None:
    registry = _registry()
    built = registry.resolve(provider="acme", api_key="explicit", env={"ACME_API_KEY": "from-env"})
    assert built["api_key"] == "explicit"


def test_the_model_reaches_the_factory_and_falls_back_to_the_provider_default() -> None:
    registry = _registry(default_model="acme-default")
    registry.register_model(ModelEntry("acme-1", "acme"))

    chosen = registry.resolve("acme-1", env={"ACME_API_KEY": "k"})
    assert chosen == {"api_key": "k", "model": "acme-1"}

    defaulted = registry.resolve(provider="acme", env={"ACME_API_KEY": "k"})
    assert defaulted == {"api_key": "k", "model": "acme-default"}


def test_a_factory_that_takes_no_model_kwarg_is_called_with_the_key_alone() -> None:
    registry = _registry(accepts_model_kwarg=False, default_model="acme-default")
    assert registry.resolve(provider="acme", env={"ACME_API_KEY": "k"}) == {"api_key": "k"}


def test_a_missing_extra_names_the_extra_to_install() -> None:
    registry = ModelRegistry()
    registry.register_provider(
        ProviderEntry(
            name="needy",
            env_vars=("NEEDY_KEY",),
            factory="agentkit_this_module_does_not_exist:build",
            extra="some-extra",
        )
    )
    with pytest.raises(MissingProviderExtra, match="some-extra"):
        registry.resolve(provider="needy", env={"NEEDY_KEY": "k"})


def test_a_broken_install_is_never_absorbed_into_the_fallback() -> None:
    """The invariant the module states explicitly. A missing optional extra and
    a missing credential are different problems with different fixes, and
    degrading the former to a fake would hide a broken install behind a
    warning about configuration."""
    registry = ModelRegistry()
    registry.register_provider(
        ProviderEntry(
            name="needy",
            env_vars=("NEEDY_KEY",),
            factory="agentkit_this_module_does_not_exist:build",
            extra="some-extra",
        )
    )
    with pytest.raises(MissingProviderExtra):
        registry.resolve(provider="needy", fallback="fake", env={"NEEDY_KEY": "k"})
