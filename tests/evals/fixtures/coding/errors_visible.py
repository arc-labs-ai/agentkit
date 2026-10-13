from agentkit.models.contracts.failures import classify


def test_billing_quota_exhaustion_is_terminal_even_when_http_status_is_429() -> None:
    failure = classify({"status": 429, "error": {"code": "insufficient_quota", "type": "rate_limit_error"}})
    assert failure is not None
    assert failure.classification == "quota_exhausted"
    assert not failure.retryable
