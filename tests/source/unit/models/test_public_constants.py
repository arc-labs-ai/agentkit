import pytest

from agentkit.agents import AgentDefinition
from agentkit.core.contracts import Mode
from agentkit.models import (
    ApiEndpoint,
    ApiPath,
    Effort,
    MindKind,
    ModelSettings,
    PriceTable,
    ReasoningSummary,
    ServiceTier,
    Verbosity,
)
from agentkit.models.contracts import Mode as ContractMode
from agentkit.models.providers import ApiEndpoint as ProviderApiEndpoint


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("effort", "high", Effort.HIGH),
        ("reasoning_summary", "concise", ReasoningSummary.CONCISE),
        ("verbosity", "medium", Verbosity.MEDIUM),
        ("tier", "priority", ServiceTier.PRIORITY),
    ],
)
def test_model_settings_normalize_supported_strings_to_public_enums(field, value, expected):
    assert getattr(ModelSettings(**{field: value}), field) is expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effort", "superhuman"),
        ("reasoning_summary", "verbose"),
        ("verbosity", "verbose"),
        ("tier", "fast"),
    ],
)
def test_model_settings_reject_unknown_values(field, value):
    with pytest.raises(ValueError, match=field):
        ModelSettings(**{field: value})


def test_agent_definition_rejects_unknown_mind_kind():
    with pytest.raises(ValueError, match="mind"):
        AgentDefinition("agent", "1", "description", mind="unknown_mind")


def test_public_mode_and_provider_endpoint_constants_are_importable():
    assert Mode.OPENAI_API.value == "openai_api"
    assert ContractMode is Mode
    assert ProviderApiEndpoint is ApiEndpoint
    assert ApiEndpoint.OPENAI.value.startswith("https://api.openai.com")
    assert ApiPath.RESPONSES.value == "/responses"
    assert PriceTable.default()
    assert AgentDefinition("agent", "1", "description", mind="react").mind is MindKind.REACT
