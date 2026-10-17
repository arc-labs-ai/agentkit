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


def test_partial_prices_preserve_known_amount_and_toml_host_tables(tmp_path):
    table = PriceTable("partial", "2026-10-13", {"model": {"input_uncached": "2"}})
    amount, source = price("model", Usage(input_uncached=5, output=3), table, allow_partial=True)
    assert amount == Decimal("0.00001")
    assert source == "mixed"

    path = tmp_path / "prices.toml"
    path.write_text('version = "v1"\ndate = "2026-10-13"\n[rates.model]\ninput_uncached = "2"\noutput = "4"\n')
    assert PriceTable.load(path).row(Mode.OPENAI_API, "model")["output"] == "4"
    assert PriceTable.derive(input="2", output="8", cache_read_ratio="0.1")["input_cache_read"] == "0.2"


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity"])
def test_invalid_price_modifiers_are_refused(value):
    with pytest.raises(ValueError):
        PriceTable("test", "2026-10-13", {}, {"batch": value})


def test_a_spend_cap_stated_only_in_prose_is_quota_and_not_a_bad_request() -> None:
    """Found by running against a real account that had hit its cap.

    Anthropic reports a reached spend limit with NO machine-readable code:

        status 400, type "invalid_request_error", code None, message
        "You have reached your specified API usage limits. You will regain
         access on 2026-11-01 at 00:00 UTC."

    The quota branch keyed only on `code`, so this fell through to the 400
    branch and was filed INVALID_REQUEST — a client bug. An operator then went
    looking for a malformed request when the fix was billing. The information
    was never missing; it sat in a field the classifier did not read.

    Three directions are asserted, because the fix is only correct if it does
    not overreach: the prose case must classify, a provider that DOES send a
    code must keep working, and a genuine bad request whose message happens to
    be about limits must stay INVALID_REQUEST.
    """
    anthropic_spend_cap = {
        "status": 400,
        "error": {
            "type": "invalid_request_error",
            "message": (
                "You have reached your specified API usage limits. "
                "You will regain access on 2026-11-01 at 00:00 UTC."
            ),
        },
    }
    assert classify(anthropic_spend_cap).classification == FailureClass.QUOTA_EXHAUSTED

    # The structured path still wins and still works.
    openai_quota = {
        "status": 429,
        "error": {"code": "insufficient_quota", "message": "You exceeded your current quota."},
    }
    assert classify(openai_quota).classification == FailureClass.QUOTA_EXHAUSTED

    # And a real invalid request is not dragged into the quota class.
    genuine_bad_request = {
        "status": 400,
        "error": {"type": "invalid_request_error", "message": "max_tokens: must be greater than 0"},
    }
    assert classify(genuine_bad_request).classification == FailureClass.INVALID_REQUEST


def test_quota_prose_matching_survives_rewording() -> None:
    """The wording belongs to the provider and changes without notice, so these
    are patterns rather than exact substrings — and deliberately specific ones:
    a bare "quota" or "billing" would misfile a genuine invalid request whose
    message merely mentions either."""
    for message in (
        "You have reached your specified API usage limits.",
        "Monthly spend limit reached for this organization",
        "insufficient quota for this request",
        "Your quota was exceeded",
        "You exceeded your current quota, please check your plan",
        "Your credit balance is too low to access the API",
        "billing not active for this account",
    ):
        facts = {"status": 400, "error": {"type": "invalid_request_error", "message": message}}
        assert classify(facts).classification == FailureClass.QUOTA_EXHAUSTED, message

    for message in (
        "the `quota` parameter is not a valid field",
        "billing_address must be a string",
    ):
        facts = {"status": 400, "error": {"type": "invalid_request_error", "message": message}}
        assert classify(facts).classification == FailureClass.INVALID_REQUEST, message


# The provider taxonomy, pinned. Introspected from the official SDKs
# (anthropic 1.15.0, openai 3.33.0) rather than transcribed from prose, because
# agentkit speaks HTTP directly and has no SDK dependency — nothing else keeps
# this mapping in step with the providers.
#
# Each TYPE case deliberately carries NO status. An in-band SSE error frame
# arrives that way, so a condition recognised only by status is unclassifiable
# mid-stream, which is where the gaps were.
@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        # ── types, with no status ──
        ({"error": {"type": "billing_error"}}, FailureClass.QUOTA_EXHAUSTED),
        ({"error": {"type": "timeout_error"}}, FailureClass.TIMEOUT),
        ({"error": {"type": "permission_error"}}, FailureClass.AUTH),
        ({"error": {"type": "not_found_error"}}, FailureClass.INVALID_REQUEST),
        ({"error": {"type": "authentication_error"}}, FailureClass.AUTH),
        ({"error": {"type": "rate_limit_error"}}, FailureClass.RATE_LIMITED),
        ({"error": {"type": "overloaded_error"}}, FailureClass.OVERLOADED),
        ({"error": {"type": "invalid_request_error"}}, FailureClass.INVALID_REQUEST),
        ({"error": {"type": "api_error"}}, FailureClass.TRANSIENT_NETWORK),
        # ── statuses ──
        ({"status": 401}, FailureClass.AUTH),
        ({"status": 402}, FailureClass.QUOTA_EXHAUSTED),
        ({"status": 403}, FailureClass.AUTH),
        ({"status": 404}, FailureClass.INVALID_REQUEST),
        ({"status": 408}, FailureClass.TIMEOUT),
        ({"status": 409}, FailureClass.INVALID_REQUEST),
        ({"status": 413}, FailureClass.INVALID_REQUEST),
        ({"status": 422}, FailureClass.INVALID_REQUEST),
        ({"status": 429}, FailureClass.RATE_LIMITED),
        ({"status": 500}, FailureClass.TRANSIENT_NETWORK),
        ({"status": 502}, FailureClass.TRANSIENT_NETWORK),
        ({"status": 503}, FailureClass.OVERLOADED),
        ({"status": 504}, FailureClass.TIMEOUT),
        ({"status": 529}, FailureClass.OVERLOADED),
        # ── codes ──
        ({"error": {"code": "insufficient_quota"}}, FailureClass.QUOTA_EXHAUSTED),
        ({"error": {"code": "context_length_exceeded"}}, FailureClass.CONTEXT_EXCEEDED),
        ({"error": {"code": "invalid_api_key"}}, FailureClass.AUTH),
        # ── precedence: a structured client status outranks a generic type ──
        ({"status": 400, "error": {"type": "api_error"}}, FailureClass.INVALID_REQUEST),
    ],
)
def test_the_provider_error_taxonomy_is_mapped_completely(
    facts: dict[str, object], expected: FailureClass
) -> None:
    assert classify(facts).classification == expected


def test_the_five_conditions_that_used_to_land_as_unknown_or_the_wrong_class() -> None:
    """The gaps, called out individually so a regression names what it broke.

    `billing_error` and `timeout_error` are canonical Anthropic types that the
    `kind` checks did not know. `permission_error` and `not_found_error` were
    reachable only through their statuses, so an in-band frame carrying the
    type alone fell to UNKNOWN. 409 is defined by BOTH SDKs (`ConflictError`)
    and was mapped nowhere. And 402 Payment Required sat in the
    INVALID_REQUEST set, which told an operator to debug a request that was
    perfectly well formed when the fix was billing.
    """
    assert classify({"error": {"type": "billing_error"}}).classification is not FailureClass.UNKNOWN
    assert classify({"error": {"type": "timeout_error"}}).classification is FailureClass.TIMEOUT
    assert classify({"error": {"type": "permission_error"}}).classification is FailureClass.AUTH
    assert (
        classify({"error": {"type": "not_found_error"}}).classification
        is FailureClass.INVALID_REQUEST
    )
    assert classify({"status": 409}).classification is not FailureClass.UNKNOWN
    assert classify({"status": 402}).classification is FailureClass.QUOTA_EXHAUSTED
