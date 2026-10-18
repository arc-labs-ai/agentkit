"""One response, two readers, and they must not disagree.

WHY THIS FILE EXISTS
--------------------
Every provider failure is read TWICE, by two classifiers with different inputs:

* `core.execution.resilience.classify` matches SUBSTRINGS of the exception
  message and decides whether `run_with_resilience` retries at all;
* `models.contracts.failures.classify` reads STRUCTURED facts off the exception
  (`status_code`, `body`, `headers`) and becomes the `Turn.failure` a caller
  sees, including its `retryable` flag.

Both are deliberate. The split is not: a response that one reader calls
transient and the other calls permanent produces a run whose retry behaviour
and whose reported failure contradict each other, and whichever one a reviewer
reads, the other is the one that fires. Three such disagreements shipped, and
each was invisible from the side you happened to be looking at.

The composed message is what makes them hard to spot. `_permanent_message`
WITHHOLDS the body and the error type precisely because those strings read
transient to a substring matcher — so a misfiled response does not merely
disagree, it actively hides the evidence that it disagreed.

Each test below asserts BOTH readers on one exception, and the controls matter
as much as the fixes: a genuine rate limit must stay retryable and a malformed
request must stay permanent, or "make them agree" becomes "retry everything".
"""

from __future__ import annotations

import pytest

from agentkit.core.execution.resilience import classify as retry_verdict
from agentkit.models.contracts.constants import FailureClass
from agentkit.models.contracts.failures import classify as structured

httpx = pytest.importorskip("httpx")

from agentkit.models.providers.frontiers import AnthropicLLM  # noqa: E402
from agentkit.models.providers.frontiers.base import _classify_4xx  # noqa: E402

_RETRYABLE = (
    FailureClass.RATE_LIMITED,
    FailureClass.OVERLOADED,
    FailureClass.TIMEOUT,
    FailureClass.TRANSIENT_NETWORK,
)


def _both(status: int, body: str) -> tuple[str, FailureClass, bool]:
    """Both verdicts for one response, so a test cannot assert only the half it likes."""
    exc = _classify_4xx(status, body)
    failure = structured(exc)
    assert failure is not None
    assert failure.retryable == (failure.classification in _RETRYABLE), (
        "retryable must be derived from the class, never set independently"
    )
    return str(retry_verdict(exc)), failure.classification, failure.retryable


def _binding() -> object:
    from agentkit.models.contracts.types import Binding, CredentialRef

    return Binding("anthropic_api", "claude-haiku-4-5", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "k"}))


def _settings() -> object:
    from agentkit.models.contracts.types import ModelSettings

    return ModelSettings()


def _limits() -> object:
    from agentkit.models.contracts.types import CallLimits

    return CallLimits(max_turns=1, wall_clock_s=30)


def _prompt() -> object:
    from agentkit.agents.prompts.composition import inline_prompt

    return inline_prompt("question")


def test_a_quota_condition_under_type_is_permanent_to_both_readers() -> None:
    """Providers send the same quota conditions under `error.code` on some
    endpoints and `error.type` on others. `_PERMANENT_429_TYPES` reads `type`;
    `failures.classify` matched those names against `code` only, so a 429
    carrying `{"type": "insufficient_quota"}` and no `code` fell through to
    RATE_LIMITED — retryable — which is the exact budget-burning retry on a
    permanently broken account that the upstream upgrade exists to prevent.

    Two modules claiming one rule, disagreeing on which field it reads.
    """
    verdict, failure_class, retryable = _both(
        429, '{"error":{"type":"insufficient_quota","message":"no credit remaining"}}'
    )
    assert verdict == "permanent"
    assert failure_class is FailureClass.QUOTA_EXHAUSTED
    assert retryable is False


def test_a_request_timeout_is_retryable_to_both_readers() -> None:
    """408 is the textbook retryable failure and this module's own structured
    taxonomy already filed it as one (`_TIMEOUT_STATUSES` is {408, 504}). It
    nonetheless fell into the generic 4xx branch, so `resilience` read the
    `_permanent_message` composed for it and failed the call fast while
    `failures.classify` called the same response TIMEOUT and retryable. 504 was
    handled only incidentally, by being >= 500."""
    verdict, failure_class, retryable = _both(
        408, '{"error":{"type":"timeout_error","message":"request took too long"}}'
    )
    assert verdict == "transient"
    assert failure_class is FailureClass.TIMEOUT
    assert retryable is True


def test_a_genuine_rate_limit_is_still_retryable() -> None:
    """The control for the quota test. Upgrading account conditions to permanent
    must not catch an ordinary 429 — that one is the failure retry exists for,
    and filing it permanent would turn a momentary slowdown into a failed run."""
    verdict, failure_class, retryable = _both(
        429, '{"error":{"type":"rate_limit_error","message":"slow down"}}'
    )
    assert verdict == "transient"
    assert failure_class is FailureClass.RATE_LIMITED
    assert retryable is True


def test_a_malformed_request_is_still_permanent() -> None:
    """The control for the 408 test. Widening the transient set must not reach
    400 — a request that cannot succeed must never be retried, and "make the two
    readers agree" is only a fix if it does not collapse into "retry everything"."""
    verdict, failure_class, retryable = _both(
        400, '{"error":{"type":"invalid_request_error","message":"bad field"}}'
    )
    assert verdict == "permanent"
    assert failure_class is FailureClass.INVALID_REQUEST
    assert retryable is False


@pytest.mark.asyncio
async def test_a_refused_connection_stays_retryable_through_the_wrapper() -> None:
    """The one that shipped, and the one no status code can describe.

    `failures.classify` identifies transport failures by `type(exc).__name__`
    (`ConnectError`, `ReadTimeout`, …), which only works while the httpx error IS
    the exception. `HttpLLM._post` re-raises a plain `ProviderError` with no
    status and no body, keeping the original on `__cause__` alone — so that whole
    branch was dead for everything going through the HTTP transport, which is
    every API-mode call. Measured, before: UNKNOWN and `retryable=False`, i.e. a
    refused connection or a mid-stream socket death was never retried, while the
    same error handed to the classifier directly gave TRANSIENT_NETWORK.

    `resilience` never noticed, because the wrapper's message says
    "network/timeout" and it matches on substrings. One reader was right the
    whole time, which is why nothing failed loudly.
    """

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    llm = AnthropicLLM(
        api_key="k",
        base_url="http://x",
        client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
    )
    try:
        with pytest.raises(Exception) as excinfo:
            await llm.chat(messages=[], model="claude-haiku-4-5")
    finally:
        await llm.aclose()

    exc = excinfo.value
    assert isinstance(exc.__cause__, httpx.ConnectError), "the cause must be preserved to be followed"
    assert getattr(exc, "status_code", None) is None, "a transport failure has no HTTP status"

    failure = structured(exc)
    assert failure is not None
    assert failure.classification is FailureClass.TRANSIENT_NETWORK
    assert failure.retryable is True
    assert str(retry_verdict(exc)) == "transient", "the two readers must still agree"


def test_following_a_cause_never_overrides_a_verdict_the_exception_itself_gave() -> None:
    """The guard on the fix. Chasing `__cause__` is only consulted when THIS
    exception produced UNKNOWN, so a wrapper that deliberately reinterprets its
    cause keeps the last word. Without that, a permanent failure raised while
    handling a transient one would inherit the transient verdict and be retried
    forever."""
    from agentkit.models.providers.frontiers.base import ProviderError

    try:
        try:
            raise httpx.ConnectError("connection refused")
        except httpx.ConnectError as cause:
            raise ProviderError(
                "provider invalid request (status 400)",
                status_code=400,
                body='{"error":{"type":"invalid_request_error"}}',
            ) from cause
    except ProviderError as exc:
        failure = structured(exc)

    assert failure is not None
    assert failure.classification is FailureClass.INVALID_REQUEST, (
        "the wrapper's own 400 must win over the transient cause it was raised from"
    )
    assert failure.retryable is False


@pytest.mark.asyncio
async def test_an_unclassifiable_error_frame_never_reads_as_a_successful_turn() -> None:
    """`classify` returns None for an error frame whose `error` is not a mapping,
    and `Turn(failure=None)` is a SUCCESSFUL turn. So a non-conformant in-band
    error frame came back as `messages=()`, `failure=None`, `usage_complete=True`
    — an error reported as an empty success, with the classification lost rather
    than degraded. It was contained only by a downstream empty-message guard.

    An unclassifiable error is still an error; it degrades to UNKNOWN.
    """
    import json

    from agentkit.models.providers.frontiers.calls import ApiAdapter

    frames = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "error", "error": "rate limited"},  # `error` as a string, not an object
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(f) + "\n\n" for f in frames),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = ApiAdapter(client=client, base_url="http://x")
    try:
        turn = await adapter.turn(
            _binding(),
            _prompt(),
            settings=_settings(),
            limits=_limits(),
            schema=None,
            tools=None,
            session=None,
            cancel=None,
            emit=lambda _event: None,
        )
    finally:
        await client.aclose()

    assert turn.failure is not None, "an error frame was reported as a successful turn"
    assert turn.failure.classification is FailureClass.UNKNOWN
