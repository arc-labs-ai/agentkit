import pytest

from agentkit.models.contracts.failures import classify


@pytest.mark.parametrize(
    "code",
    [
        "insufficient_quota",
        "billing_not_active",
        "enforced_spend_limit_reached",
        "usage_limit_reached",
        "spend_limit_reached",
    ],
)
def test_all_structured_billing_codes_override_rate_limit_status(code: str) -> None:
    failure = classify({"status": 429, "error": {"details": {"error_code": code}, "type": "rate_limit_error"}})
    assert failure is not None
    assert failure.classification == "quota_exhausted" and not failure.retryable


def test_genuine_rate_limit_stays_retryable_and_prose_is_not_classification_input() -> None:
    limited = classify({"status": 429, "headers": {"Retry-After": "3"}, "error": {"message": "billing quota maybe"}})
    assert limited is not None
    assert limited.classification == "rate_limited" and limited.retryable and limited.retry_after_s == 3
    overloaded = classify({"status": 529, "error": {"type": "overloaded_error"}})
    assert overloaded is not None
    assert overloaded.classification == "overloaded" and overloaded.retryable
    hostile = classify({"status": 500, "error": {"details": ["bad"], "code": []}})
    assert hostile is not None
    assert hostile.classification == "transient_network"


@pytest.mark.parametrize("status", [429, 503, 529])
def test_billing_precedence_applies_to_provider_overload_statuses(status: int) -> None:
    failure = classify({"status": status, "error": {"code": "spend_limit_reached", "type": "overloaded_error"}})
    assert failure is not None
    assert failure.classification == "quota_exhausted" and not failure.retryable


def test_provider_exception_body_is_classified_without_reading_prose() -> None:
    class ProviderFailure(RuntimeError):
        status_code = 429
        body = '{"error":{"code":"insufficient_quota","type":"rate_limit_error"}}'
        headers = {"Retry-After": "2"}

    failure = classify(ProviderFailure("temporarily overloaded; try again"))
    assert failure is not None
    assert failure.classification == "quota_exhausted" and not failure.retryable
    auth = classify({"status": 429, "error": {"code": "invalid_api_key"}})
    assert auth is not None
    assert auth.classification == "auth" and not auth.retryable
