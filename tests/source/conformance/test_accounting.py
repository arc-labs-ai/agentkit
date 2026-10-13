from datetime import UTC, datetime, timedelta
from decimal import Decimal
from email.utils import format_datetime

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.models.catalog.pricing import PriceTable, normalize_usage, price, usage_delta
from agentkit.models.contracts.constants import FailureClass
from agentkit.models.contracts.failures import classify, retry_after
from agentkit.models.contracts.types import Usage


@pytest.mark.parametrize(
    "facts,expected",
    [
        ({"status": 429, "error": {"code": "insufficient_quota"}}, FailureClass.QUOTA_EXHAUSTED),
        (
            {"status": 400, "error": {"details": {"error_code": "context_length_exceeded"}}},
            FailureClass.CONTEXT_EXCEEDED,
        ),
        ({"status": 401}, FailureClass.AUTH),
        ({"status": 429}, FailureClass.RATE_LIMITED),
        ({"status": 529}, FailureClass.OVERLOADED),
        ({"status": 504}, FailureClass.TIMEOUT),
        ({"status": 502}, FailureClass.TRANSIENT_NETWORK),
        ({"status": 422}, FailureClass.INVALID_REQUEST),
        ({"error": {"type": "refusal"}}, FailureClass.REFUSED),
        ({"error": {"type": "cancelled"}}, FailureClass.CANCELLED),
        ({"error": {"details": "malformed"}, "headers": "malformed"}, FailureClass.UNKNOWN),
        (TimeoutError(), FailureClass.TIMEOUT),
        (ConnectionError(), FailureClass.TRANSIENT_NETWORK),
    ],
)
def test_failures_classified_from_transport_facts_with_quota_priority(facts, expected):
    failure = classify(facts)
    assert failure.classification == expected
    assert failure.retryable == (
        expected
        in (FailureClass.RATE_LIMITED, FailureClass.OVERLOADED, FailureClass.TIMEOUT, FailureClass.TRANSIENT_NETWORK)
    )


def test_http_error_body_shape_cannot_break_classifier_or_leak_secret():
    class Error(Exception):
        status_code = 401
        headers = {"Retry-After": "2"}
        body = '["secret-token"]'

    assert classify(Error()).classification == FailureClass.AUTH

    Error.body = '{"error":{"type":"authentication_error","message":"secret-token"}}'
    failure = classify(Error())
    assert "secret-token" not in repr(failure)
    assert failure.retry_after_s == 2
    Error.body = "invalid-json"
    assert classify(Error()).classification == FailureClass.AUTH


def test_malformed_structured_error_fields_are_ignored_without_crashing():
    failure = classify(
        {"status": {"bad": True}, "error": {"code": [], "type": {}}, "headers": {"retry-after": ["bad"]}}
    )
    assert failure.classification == FailureClass.UNKNOWN
    assert failure.retry_after_s is None


@pytest.mark.parametrize(
    "value,expected", [(None, None), ("invalid", None), ("NaN", None), ("Infinity", None), ("-1", 0), ("0.5", 0.5)]
)
def test_retry_after_numeric_values_are_bounded_and_finite(value, expected):
    assert retry_after(value) == expected


def test_retry_after_http_date():
    value = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 28 <= retry_after(value) <= 30


def test_reasoning_tokens_are_preserved_and_never_charged_twice():
    usage = normalize_usage(
        Mode.CLAUDE_CODE,
        {
            "input_tokens": 10,
            "output_tokens": 30,
            "thinking_tokens": 20,
            "cache_creation": {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 7},
            "cache_read_input_tokens": 100,
        },
    )
    assert usage == Usage(10, 100, 5, 7, 30, 20)
    table = PriceTable(
        "test",
        "2026-10-13",
        {
            "model": {
                "input_uncached": "1",
                "input_cache_read": "0.1",
                "input_cache_write_5m": "1.25",
                "input_cache_write_1h": "2",
                "output": "5",
            }
        },
    )
    amount, source = price("model", usage, table)
    assert amount == Decimal("0.00019025")
    assert source == "price_table"
    assert usage_delta(Usage(1, 2, 3, 4, 5, 2), Usage(10, 20, 30, 40, 50, 20)) == Usage(1, 2, 3, 4, 5, 2)


def test_unpublished_cache_write_rate_is_unpriced_rather_than_guessed():
    table = PriceTable.default()
    assert price("gpt-5.4-mini", Usage(input_cache_write_5m=1), table)[0] is None
    assert price("unknown", Usage(input_uncached=10), table)[0] is None
    assert price("gpt-5.4-mini", Usage(output=10), table, modifiers=("unknown",))[0] is None


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity"])
def test_invalid_price_modifiers_are_refused(value):
    with pytest.raises(ValueError):
        PriceTable("test", "2026-10-13", {}, {"batch": value})
