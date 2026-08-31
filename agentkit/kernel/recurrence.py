"""Recurrence — bounding a loop by whether it is MOVING, not by how many turns it took.

``run_with_resilience`` (next door in ``resilience.py``) is the right shape for a *flaky* call
and the wrong shape for a *semantic* one: an attempt that completed, produced an answer, and did
not achieve the goal. Nothing raised, so there is nothing to classify and nothing to back off
from — and yet trying again is exactly right, provided the run is still getting somewhere.

The bound that tells those apart is recurrence, not a count. **Three attempts producing three
different failures is progress; two producing the same one is not.** A count cannot distinguish
them, so any count is simultaneously too tight for the first case and too loose for the second:
``max_attempts=3`` abandons a run that was one distinct step from the answer, and also pays for
three identical round trips at a dead end.

``LedgerPolicy`` (``agentkit.agents.policies.ledger``) already had the missing half — a progress
ledger re-derived each round asking *satisfied? looping?* — welded into a multi-agent supervisor.
What is genuinely shared with it turned out to be three lines ("have I seen this outcome
before?"), so it is inlined below rather than extracted: everything else in that policy is bound
to children, transcripts and ``AgentResult``. Worth recording, because the ledger's default
assessor derives ``in_a_loop`` from ``replies[-1] == replies[-2]`` — the adjacent compare this
module deliberately does not use, for the reason spelled out in ``attempt_until_stuck``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from agentkit.kernel.errors import AgentkitError, Failure
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.resilience import ErrorClass, stable_hash

R = TypeVar("R")


def _same_signature(a: Any, b: Any) -> bool:
    """Confirm that two signatures sharing a ``stable_hash`` key are actually the same.

    ``stable_hash`` is built for *cache* keys, where its lossiest branch is fail-safe: an
    object it cannot describe structurally degrades to the bare type name
    (``_stable_default``'s last resort, ``"<Foo>"``), two such objects collide, and the cost is
    a cache hit that should have been a miss. Here the same collision is fail-**dangerous** —
    it is a PERMANENT, non-retriable "you are going in circles" verdict on a run that produced
    two genuinely different answers. Measured before this guard existed: three distinct
    ``__slots__`` signatures all hashed to ``b5e4d8ee180594ea`` and the loop reported stuck on
    attempt 2, having proven nothing. ``("a",)`` and ``["a"]`` collide the same way, since JSON
    has one sequence type.

    So the hash indexes, and equality decides. The docstring's promise that a repeat is
    *demonstrated* rather than guessed is only true if something actually compares the two
    signatures.

    A comparison that raises or does not yield a bool (``numpy`` arrays return an elementwise
    array, and ``bool()`` of that raises) counts as "not the same". That direction matters:
    an unconfirmable repeat falls through to the ``max_attempts`` backstop, which reports
    UNKNOWN/retriable "out of budget" — honest and recoverable — whereas trusting the hash
    reports PERMANENT/non-retriable and kills a healthy run. Given a signature type this
    module cannot reason about, running out of budget is the safe way to be wrong.
    """
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


#: How a detected repeat is delivered. Not what it MEANS — the two modes produce the identical
#: ``Failure``; only one of them raises it. Keeping this a delivery switch rather than a
#: semantics switch is what lets a caller flip it without re-reading the finding.
OnRepeat = Literal["escalate", "stop"]

#: Named so ``Failure.source`` is greppable and stable across both exits (repeat and
#: exhaustion), the way ``gather_best_effort[i]`` is.
SOURCE = "attempt_until_stuck"


class Stuck(AgentkitError):
    """A loop bounded by recurrence hit a repeat: the same outcome signature twice.

    An ``AgentkitError`` and not a new root, so the framework-wide
    ``except AgentkitError:`` still catches it — this is the taxonomy in
    ``kernel.errors``, not a parallel one.

    It carries the ``Failure`` rather than only a message because the escalating caller and the
    quiet caller must be looking at the *same* finding. ``on_repeat`` chooses delivery; a handler
    that catches ``Stuck`` reads ``exc.failure`` and gets byte-identical data to what
    ``on_repeat="stop"`` would have returned — including ``partial_output``, which is the last
    answer the loop actually produced and the only artifact worth salvaging from a stall.
    """

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.message)
        self.failure = failure


async def attempt_until_stuck(
    fn: Callable[[], Awaitable[R]],
    *,
    fingerprint: Callable[[R], Any],
    on_repeat: OnRepeat = "escalate",
    max_attempts: int = 4,
    ctx: Ctx | None = None,
) -> R | Failure:
    """Attempt · fingerprint the outcome · attempt again on a *different* fingerprint · stop on a repeat.

    ::

        answer = await attempt_until_stuck(
            lambda: run_one(),
            fingerprint=lambda outcome: outcome.failure_signature,
            on_repeat="escalate",
            max_attempts=4,          # a BACKSTOP, not the bound
        )

    **``fingerprint`` returning ``None`` IS the success signal.** There is no second "did it
    work?" predicate, and the omission is deliberate: the natural fingerprint of a successful
    outcome is "no failure signature", which the motivating expression
    (``outcome.failure_signature``) already spells ``None``. A separate ``is_done=`` would let
    the two disagree — an outcome reported done while still carrying a signature, or the reverse
    — and there is no defensible behaviour for that state. One function, one answer: a signature
    means *not yet*, ``None`` means *done*, and ``fn``'s return value is handed straight back.

    **Signatures are compared against EVERY signature seen, not the previous one.** This is the
    case worth being careful about, and the reason the adjacent compare is wrong is
    A, B, A, B, …: a run oscillating between two dead ends never repeats *consecutively*, so a
    ``previous == current`` check never fires and the loop silently degrades to being bounded by
    ``max_attempts`` after all. Measured against the naive version on ``["A","B","A","B","A","B"]``
    with ``max_attempts=6``: 6 attempts and no repeat detected, versus 3 attempts here — the
    alternation is a cycle, and re-treading ground you have already covered is not progress no
    matter how the visits are interleaved.

    Comparison is by ``stable_hash`` of the signature rather than by ``hash()`` or ``==``, so a
    signature may be **unhashable** — a ``dict`` or ``list`` is the shape an LLM-authored or
    schema-validation signature naturally has (``{"missing": ["a", "b"]}``). Requiring hashability
    would push every caller into hand-serialising, and ``str()`` would reintroduce the memory
    addresses ``stable_hash`` exists to keep out of fingerprints. It also means the key is the
    same basis as cache and idempotency keys, so two processes agree on what "the same signature"
    means. The hash only **indexes**, though: a matching key is then confirmed with ``==`` before
    a repeat is declared, because ``stable_hash`` is deliberately lossy for values it cannot
    describe structurally and a collision here is a PERMANENT verdict rather than a cache miss.
    See ``_same_signature``. The practical consequence for a caller: a signature type with no
    value equality (a plain object without ``__eq__``) can no longer be *falsely* called stuck,
    but its genuine repeats will not be detected either — that run is bounded by ``max_attempts``
    and reported as exhaustion. Signatures should be values: strings, dicts, lists, tuples,
    enums, frozen dataclasses.

    **Raising attempts are not fingerprinted, ever — that is the seam with
    ``run_with_resilience``.** An exception is a *fault*; it is classified, backed off and
    breaker-counted by that function. An attempt that returned is a *result*, and only results
    have outcomes to fingerprint. Nothing is caught here, so an exception propagates untouched.
    The two therefore compose by nesting rather than overlapping::

        await attempt_until_stuck(
            lambda: run_with_resilience(call_model, max_attempts=3),
            fingerprint=score,
        )

    Collapsing that into one loop was tried and is wrong twice over: a transient timeout would
    burn a semantic attempt (and get a fresh "signature" each time, so the stall detector could
    never fire), and the semantic retries would multiply the transient ones — 4 × 3 round trips
    at a provider that is already rate-limiting.

    A raising ``fingerprint`` propagates for the same reason, stated in reverse: swallowing it and
    substituting a placeholder gives every attempt a distinct signature, which is a loop that can
    never detect a repeat. A broken fingerprint must fail loudly, not quietly turn this back into
    a counted retry.

    **What comes back.** House style, following ``gather_best_effort``: the failure is *data*.

    * success → ``fn``'s outcome, unwrapped;
    * repeat → a ``Failure`` categorised ``PERMANENT`` / ``retriable=False``. Not a guess, which
      is what ``classify``'s substring matching is doing when it says PERMANENT: the run has
      *demonstrated* that another identical attempt produces an identical result. Delivered by
      ``raise Stuck(failure)`` under ``escalate``, returned under ``stop``;
    * exhaustion → a ``Failure`` categorised ``UNKNOWN`` / ``retriable=True``.

    Exhaustion returns in **both** modes, and that asymmetry is the point of the whole helper:
    ``on_repeat`` names the repeat. Hitting the backstop while every signature still differs means
    the run was moving and ran out of a budget the caller chose — the fix is a larger backstop,
    not an escalation. Escalating it would put "you are going in circles" and "you were making
    progress" through the same handler, which is precisely the confusion a count causes.

    ``max_attempts < 1`` raises ``ValueError`` instead of returning an exhausted ``Failure``. A
    zero-attempt ``Failure`` is indistinguishable from a real one, so a caller would log and page
    on an upstream failure for work that never ran. ``max_attempts=1`` is legal and degenerate: a
    single shot, where no recurrence is possible and any signature exhausts. An ``on_repeat``
    outside ``{"escalate", "stop"}`` raises ``ValueError`` from the same block, for a sharper
    version of the reason: an unrecognised mode must not quietly resolve to the non-raising one.

    ``ctx`` is optional and, when given, ``check_cancelled()`` runs at the top of every attempt —
    the same safe point ``ReActCognition`` and ``LedgerPolicy`` use. Including the *first*
    attempt: a run cancelled before this was reached must not pay for one more model call to find
    out. Mid-attempt cancellation needs nothing here, because ``Cancelled`` raised inside ``fn``
    is an exception and exceptions are not caught.
    """
    if max_attempts < 1:
        # Before the first ``ctx.check_cancelled()`` and before any call to ``fn``: an argument
        # error is about the call site, and reporting it as a cancellation or a failed attempt
        # would send the reader looking in the wrong place.
        raise ValueError(f"attempt_until_stuck: max_attempts must be >= 1, got {max_attempts!r}")
    if on_repeat not in ("escalate", "stop"):
        # Same argument as ``max_attempts``, and checked in the same breath because the failure
        # mode is worse. An unrecognised mode used to fall through the ``== "escalate"`` test and
        # behave as ``"stop"``: a caller who wrote ``on_repeat="raise"`` got the stall handed back
        # as an ordinary return value, and code that only handles the raise treats a PERMANENT
        # Failure as the answer. Silently choosing the NON-raising branch on a typo is the one
        # direction a delivery switch must never fail in, and ``Literal`` only catches it at call
        # sites a type-checker actually sees — not a mode read from config.
        raise ValueError(f"attempt_until_stuck: on_repeat must be 'escalate' or 'stop', got {on_repeat!r}")

    # key → every (attempt, signature) that hashed to it. A ``set`` would do for the decision;
    # the attempt number is kept so the ``Failure`` can name BOTH ends of the cycle. On an A,B,A
    # stall "attempt 3 repeats attempt 1" is the sentence that tells an operator it is an
    # oscillation rather than a wedge, and that distinction changes what they go and look at.
    # The signature itself is kept so ``_same_signature`` can confirm the repeat instead of
    # trusting the hash — see that function for the collision this exists to survive.
    seen: dict[str, list[tuple[int, Any]]] = {}
    distinct = 0
    last: Any = None

    for attempt in range(1, max_attempts + 1):
        if ctx is not None:
            ctx.check_cancelled()
        outcome = await fn()
        # Held outside the loop body's scope so the exhaustion ``Failure`` below can still carry
        # the final answer as ``partial_output``. An exhausted loop that returns nothing throws
        # away every round trip it just paid for.
        last = outcome
        signature = fingerprint(outcome)
        if inspect.isawaitable(signature):
            # The signature is frequently itself a model call ("does this answer the goal?"), so
            # it has to be allowed to be async. Same duck-typed treatment ``LedgerPolicy`` gives
            # its assessor and planner — a sync lambda stays a sync lambda.
            signature = await signature
        if signature is None:
            return outcome
        key = stable_hash(signature)
        bucket = seen.setdefault(key, [])
        first_seen = next((earlier for earlier, prior in bucket if _same_signature(prior, signature)), None)
        if first_seen is not None:
            failure = Failure(
                category=ErrorClass.PERMANENT,
                source=SOURCE,
                message=(
                    f"stuck after {attempt} attempts: outcome signature {key} recurred "
                    f"(attempt {attempt} repeats attempt {first_seen}); "
                    f"{distinct} distinct signatures seen"
                ),
                retriable=False,
                partial_output=outcome,
            )
            if on_repeat == "escalate":
                raise Stuck(failure)
            return failure
        bucket.append((attempt, signature))
        distinct += 1

    return Failure(
        category=ErrorClass.UNKNOWN,
        source=SOURCE,
        message=(
            f"exhausted the {max_attempts}-attempt backstop with {distinct} distinct "
            f"signatures — still making progress, out of budget"
        ),
        retriable=True,
        partial_output=last,
    )


__all__ = ["OnRepeat", "Stuck", "attempt_until_stuck"]
