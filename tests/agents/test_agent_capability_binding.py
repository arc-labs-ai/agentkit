"""Capability refusal at BIND TIME — ``Agent(...)`` construction, before spend.

Brief 5's constraint was "refuse at construction or first bind, not on the
call — the value is catching it before money is spent." For this framework
construction *is* the bind: ``model`` is a field on the ``Agent``, so
``__post_init__`` is the earliest moment both the model and the wiring
(cognition, tools, output schema) are known.

Every test here asserts on ``Agent(...)`` itself. None of them run an agent,
because if a check needed a run to fire it would have failed at its job.
"""

from __future__ import annotations

import warnings

import pytest

from agentkit.adapters.llm.model_registry import (
    Capability,
    CapabilityMismatch,
    ModelCapabilities,
    ModelEntry,
    register_model,
    registry,
)
from agentkit.agents import Agent
from agentkit.tools import tool


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch):
    """Agent construction reads the PROCESS-WIDE registry (that is the point —
    an application registers its models at startup and every later Agent sees
    them). Swap in a fresh one per test so registrations here can't leak into
    another module's expectations."""
    import agentkit.adapters.llm.model_registry as mod

    monkeypatch.setattr(mod, "_DEFAULT", mod.default_registry())
    return registry()


@tool(side_effecting=False)
def lookup(q: str) -> str:
    """A tool, so a ReAct cognition has a non-empty registry."""
    return q


# ── the refusal ──────────────────────────────────────────────────────────────


def test_binding_a_vision_agent_to_a_blind_model_refuses(_isolated_registry) -> None:
    """The brief's motivating failure, made impossible.

    Left unchecked this is silent AND well-formed: the agent returns a
    structurally valid result citing evidence it never read, and nothing
    downstream can tell. Here it is a construction error with the capability
    and the model both named."""
    with pytest.raises(CapabilityMismatch) as exc:
        Agent("ocr", "deepseek-chat", requires=("vision",))
    assert "vision" in str(exc.value) and "deepseek-chat" in str(exc.value)
    assert "'ocr'" in str(exc.value), "the message should name the agent, not just the framework"


def test_no_llm_call_is_needed_for_the_refusal(_isolated_registry) -> None:
    """Pinned explicitly: there is no ctx, no invoker, no event loop — the
    refusal happens with nothing wired but the Agent itself. That is what
    "before any spend" means operationally."""
    with pytest.raises(CapabilityMismatch):
        Agent("ocr", "deepseek-chat", requires=("vision",))


def test_a_satisfied_requirement_constructs_silently(_isolated_registry) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = Agent("reader", "claude-sonnet-4-6", requires=("vision", "tools"))
    assert agent.model == "claude-sonnet-4-6"
    assert caught == []


def test_unknown_model_warns_by_default_and_still_constructs(_isolated_registry) -> None:
    """Additive: existing wiring must keep working. A self-hosted or
    brand-new model name is UNKNOWN, not absent — the agent is built, and the
    operator is told the check could not actually run."""
    with pytest.warns(UserWarning, match="UNKNOWN"):
        agent = Agent("reader", "our-own-finetune-v2", requires=("vision",))
    assert agent.name == "reader"


def test_unknown_model_can_be_made_a_hard_stop(_isolated_registry) -> None:
    """The setting a production service that pins its models wants: "we don't
    know what this model can do" becomes a deployment-time refusal."""
    with pytest.raises(CapabilityMismatch):
        Agent(
            "reader",
            "our-own-finetune-v2",
            requires=("vision",),
            on_unknown_capability="refuse",
        )


# ── derived from wiring, not declared ────────────────────────────────────────


def test_a_tool_loop_on_a_toolless_model_refuses_without_being_declared(
    _isolated_registry,
) -> None:
    """The requirement nobody remembers to write down.

    ``deepseek-reasoner`` does not accept tools. Bound to a ``ReActCognition``
    the tools are simply ignored at request time and the loop degrades into a
    single call — a quiet wrong answer. The Agent derives ``tools`` from the
    non-empty registry and refuses.
    """
    from agentkit.agents.cognition import ReActCognition

    with pytest.raises(CapabilityMismatch, match="tools"):
        Agent("searcher", "deepseek-reasoner", cognition=ReActCognition(tools=[lookup]))


def test_a_toolless_cognition_on_the_same_model_is_fine(_isolated_registry) -> None:
    """The derivation is from WIRING, not from the model. Same model, no
    tools, no complaint."""
    Agent("summariser", "deepseek-reasoner")


def test_an_empty_tool_registry_derives_nothing(_isolated_registry) -> None:
    """A ReAct cognition with no tools in it doesn't need tool support."""
    from agentkit.agents.cognition import ReActCognition

    Agent("empty", "deepseek-reasoner", cognition=ReActCognition(tools=[]))


def test_derived_requirement_is_silent_on_an_unregistered_model(_isolated_registry) -> None:
    """The noise guard, asserted at the Agent boundary: a tool-using agent on
    a made-up model name — i.e. most tests and most local development — must
    not emit a warning. If it did, the category would get filtered and the
    real ``requires=`` warnings would be lost with it."""
    from agentkit.agents.cognition import ReActCognition

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Agent("dev", "my-local-model", cognition=ReActCognition(tools=[lookup]))
    assert caught == []


# ── context window ───────────────────────────────────────────────────────────


def test_min_context_window_refuses_a_short_model(_isolated_registry) -> None:
    with pytest.raises(CapabilityMismatch, match="context window"):
        Agent("longform", "deepseek-chat", min_context_window=400_000)


def test_min_context_window_accepts_a_long_model(_isolated_registry) -> None:
    Agent("longform", "gpt-4.1", min_context_window=400_000)


# ── re-checking after mutation ───────────────────────────────────────────────


def test_check_capabilities_can_be_reasserted_after_swapping_the_model(
    _isolated_registry,
) -> None:
    """``Agent`` is a mutable dataclass, so a caller CAN swap the model after
    construction and slip past ``__post_init__``. The check is therefore also
    a public method they can re-run — the framework cannot intercept the
    assignment, so it hands over the tool instead of pretending to."""
    agent = Agent("reader", "claude-sonnet-4-6", requires=("vision",))
    agent.model = "deepseek-chat"
    with pytest.raises(CapabilityMismatch, match="vision"):
        agent.check_capabilities()


def test_an_agent_with_no_model_skips_the_check(_isolated_registry) -> None:
    """A model injected later has nothing to check against yet — and must not
    raise at construction just for being incomplete."""
    Agent("later", requires=("vision",)).check_capabilities()


# ── the registry is the source, so declaring changes the outcome ─────────────


def test_declaring_a_model_turns_a_warning_into_a_real_check(_isolated_registry) -> None:
    """End-to-end on the merge of Briefs 3 and 5: the same table that routes a
    name to a provider is the one a bind check reads. Registering a row
    upgrades "we don't know" into a genuine refusal."""
    with pytest.warns(UserWarning, match="UNKNOWN"):
        Agent("a", "acme-tiny", requires=("vision",))

    register_model(
        ModelEntry(
            name="acme-tiny",
            provider="openai",
            capabilities=ModelCapabilities(vision=Capability.NO, tools=Capability.YES),
        )
    )
    with pytest.raises(CapabilityMismatch, match="vision"):
        Agent("a", "acme-tiny", requires=("vision",))
    # ...and the capability it DOES have is untouched.
    Agent("b", "acme-tiny", requires=("tools",))


# ── response_format inference now reads the registry, not a name prefix ──────


def test_native_json_schema_is_declared_not_guessed_from_the_name(_isolated_registry) -> None:
    """``_infer_response_format`` used to be ``model.startswith("gpt-")`` — the
    very name-prefix guessing Brief 5 exists to remove. It now reads the
    declared ``native_json_schema`` capability."""
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        x: int

    # Declared YES → native strict mode wired.
    rf = Agent("a", "gpt-4o", output=Out).response_format
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Out" and rf["json_schema"]["strict"] is True
    # Declared NO → prompt-injection fallback, which always works.
    assert Agent("b", "claude-sonnet-4-6", output=Out).response_format is None
    # UNKNOWN and not a gpt- name → the safe direction.
    assert Agent("c", "mystery-model", output=Out).response_format is None


def test_unknown_gpt_prefixed_model_keeps_the_last_resort_guess(_isolated_registry) -> None:
    """A ``gpt-`` name the registry has never seen still gets native mode.

    Deliberate, and the safe direction: an unregistered ``gpt-`` name is
    overwhelmingly an OpenAI model, and the cost of being wrong is a provider
    400 at wiring time — loud and immediate — not a plausible empty answer."""
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        x: int

    rf = Agent("d", "gpt-6-turbo-unreleased", output=Out).response_format
    assert rf is not None and rf["type"] == "json_schema"
