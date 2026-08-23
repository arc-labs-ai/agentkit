"""4xx/5xx → exception mapping, read back through `kernel.resilience.classify`.

`classify` is what actually decides retry-or-fail-fast, and it is substring-based with TRANSIENT
checked FIRST — so asserting the exception TYPE proves nothing. Every test here asserts the
classification (and, where it matters, how many HTTP requests the retry loop actually made).
"""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import OpenAICompatibleLLM, ProviderError
from agentkit.adapters.llm.providers.base import ProviderAuthError, _classify_4xx
from agentkit.kernel.resilience import ErrorClass, classify, run_with_resilience
from agentkit.kernel.types import Message


def _run(coro):
    return asyncio.run(coro)


async def _nosleep(_seconds):  # deterministic retries: no real backoff
    return None


def _counting_client(status, body):
    """A MockTransport client plus the list of requests it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=body.encode())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def _llm(status, body):
    client, seen = _counting_client(status, body)
    return OpenAICompatibleLLM(api_key="k", base_url="http://x", client=client), seen


async def _chat_with_retries(llm, attempts=5):
    return await run_with_resilience(
        lambda: llm.chat(messages=[Message("user", "q")], model="m"),
        max_attempts=attempts,
        sleep=_nosleep,
    )


_BILLING = json.dumps({"error": {"type": "billing_not_active", "message": "account inactive"}})
_RATE_LIMIT = json.dumps({"error": {"type": "rate_limit_exceeded", "message": "slow down"}})


# ── the permanent-429 upgrade, which used to be a no-op ─────────────────────


@pytest.mark.parametrize(
    "error_type", ["billing_not_active", "insufficient_quota", "invalid_api_key", "account_deactivated"]
)
def test_a_429_naming_an_account_condition_classifies_permanent(error_type) -> None:
    """Regression: the upgrade to `ProviderAuthError` was a complete no-op. The message it built
    embedded "status 429" and the raw body, and `classify` checks its TRANSIENT list — which
    contains "429" — FIRST. Measured before the fix:
    `429/billing_not_active -> ProviderAuthError classify=transient`."""
    exc = _classify_4xx(429, json.dumps({"error": {"type": error_type, "message": "no"}}))
    assert isinstance(exc, ProviderAuthError)
    assert classify(exc) is ErrorClass.PERMANENT


def test_a_permanent_429_is_not_retried_at_all() -> None:
    """The behaviour the classification exists for. Measured before the fix:
    `HTTP attempts for a permanent billing error: 5` — the whole retry budget spent on a failure
    only a human topping up the account can clear."""
    llm, seen = _llm(429, _BILLING)
    with pytest.raises(ProviderAuthError):
        _run(_chat_with_retries(llm))
    assert len(seen) == 1


def test_a_genuine_429_rate_limit_stays_transient_and_is_retried() -> None:
    """POSITIVE CONTROL. A "fix" that simply made every 429 permanent — or that stopped
    mentioning the status at all — fails here: a real rate limit is the single most retryable
    provider failure and must still burn its retries."""
    llm, seen = _llm(429, _RATE_LIMIT)
    with pytest.raises(ProviderError) as exc:
        _run(_chat_with_retries(llm, attempts=3))
    assert not isinstance(exc.value, ProviderAuthError)
    assert classify(exc.value) is ErrorClass.TRANSIENT
    assert len(seen) == 3


def test_a_429_with_an_unparseable_body_stays_transient() -> None:
    """Edge: no JSON to read `error.type` from. Unknown means "might be a real rate limit", and
    the conservative choice on a 429 is to retry."""
    exc = _classify_4xx(429, "<html>429 Too Many Requests</html>")
    assert classify(exc) is ErrorClass.TRANSIENT


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (500, ErrorClass.TRANSIENT),
        (503, ErrorClass.TRANSIENT),
        (400, ErrorClass.PERMANENT),
        (401, ErrorClass.PERMANENT),
        (403, ErrorClass.PERMANENT),
        (404, ErrorClass.PERMANENT),
    ],
)
def test_the_rest_of_the_status_map_is_unchanged(status, expected) -> None:
    """Positive control for the whole table: reshaping the permanent-429 branch must not move
    any other status."""
    assert classify(_classify_4xx(status, "{}")) is expected


# ── the body: parsed whole, displayed short, never allowed to lie ───────────


def test_an_error_body_longer_than_the_message_excerpt_is_still_parsed() -> None:
    """Regression: the body was truncated to 500 chars BEFORE the JSON parse, so a longer error
    body was no longer valid JSON and `error.type` vanished with it. Measured:
    `truncated-body classify: ProviderError` vs `full-body: ProviderAuthError` — the same
    response retried five times or none, depending only on how chatty the provider was."""
    body = json.dumps({"error": {"message": "z" * 900, "type": "insufficient_quota"}})
    assert len(body) > 500

    llm, seen = _llm(429, body)
    with pytest.raises(ProviderAuthError) as exc:
        _run(_chat_with_retries(llm))
    assert classify(exc.value) is ErrorClass.PERMANENT
    assert len(seen) == 1
    assert exc.value.body == body  # the FULL body survives on the exception
    assert len(str(exc.value)) < len(body)  # ... while the message stays short


def test_the_streaming_path_parses_the_full_error_body_too() -> None:
    """The same truncate-before-parse bug sat on `_stream_events`. A pre-stream error is exactly
    where the retry decision matters, since nothing has been yielded yet."""
    body = json.dumps({"error": {"message": "z" * 900, "type": "billing_not_active"}})
    llm, seen = _llm(429, body)

    async def drain():
        return [d async for d in llm.stream(messages=[Message("user", "q")], model="m")]

    with pytest.raises(ProviderAuthError) as exc:
        _run(drain())
    assert classify(exc.value) is ErrorClass.PERMANENT
    assert exc.value.body == body


def test_a_permanent_error_whose_body_reads_transient_is_still_permanent() -> None:
    """Edge, and the reason the message is built rather than concatenated: OpenAI's own
    `insufficient_quota` body says "You exceeded your current quota ... rate limit", and
    "rate limit" is a TRANSIENT substring. The body is withheld from the message (and kept on
    `exc.body`) rather than allowed to flip the classification."""
    body = json.dumps(
        {"error": {"type": "insufficient_quota", "message": "You exceeded your rate limit"}}
    )
    exc = _classify_4xx(429, body)
    assert classify(exc) is ErrorClass.PERMANENT
    assert "rate limit" not in str(exc).lower()
    assert exc.body == body  # nothing is lost — it moved off the message


def test_a_401_body_mentioning_a_timeout_stays_permanent() -> None:
    """Edge: same failure mode on the plain auth branch. A stale key must not be retried because
    the provider's prose happened to contain "timeout"."""
    exc = _classify_4xx(401, '{"error": {"message": "session timeout; re-authenticate"}}')
    assert classify(exc) is ErrorClass.PERMANENT


def test_a_permanent_error_still_says_what_went_wrong() -> None:
    """Positive control for the withholding above: a "fix" that blanked every message would pass
    the classification tests and leave operators with nothing to read."""
    exc = _classify_4xx(429, _BILLING)
    assert "billing_not_active" in str(exc)
    assert "account inactive" in str(exc)  # this body carries no transient substring
    assert exc.status_code == 429
