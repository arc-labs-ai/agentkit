"""The elicitation contract either side of the human: what goes out, what counts as yes.

WHY THIS FILE EXISTS
--------------------
``test_approvals.py`` next door tests ``ApprovalServer`` as a GATE — every way
it could fail open. This file tests the ``Elicitation``/``Decision`` pair as a
PROTOCOL, which is the half an application actually implements: an ``Asker`` is
written by the caller, against whatever the ``Elicitation`` carries, and its
answer is read back through one predicate.

Two things about that protocol are untested and both are reachable by an
ordinary, well-meaning ``Asker``:

* **What the request carries.** A UI cannot render a countdown it was never
  given, and a reviewer answering an ``approval`` request with free text is
  answering a different question than the one asked. ``kind``, ``choices`` and
  ``deadline_s`` are the fields that make the request self-describing, and a
  request that dropped them would still produce a working allow/deny for every
  test that only reads the verdict.

* **What an expiry that carries a value means.** ``Decision.approved`` is the
  single predicate the gate turns on, and it is shared with agentkit's own HITL
  path — so a kind that leaks into it starts allowing on BOTH paths at once. The
  sharp case is an ``expired`` decision that still carries a ``value``: a
  transport that fills in a default after its deadline, or a queue worker that
  returns the last thing it saw. A kind test that happened to be positional, or
  a predicate widened "so a timeout can carry the reviewer's draft", turns the
  passage of time into consent on the most dangerous call in the run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentkit.integrations.mcp.approvals import (
    ApprovalServer,
    Decision,
    Elicitation,
)


def _read(answer: str) -> dict[str, Any]:
    """The CLI's view of an answer: parse it as the CLI's own parser would."""
    parsed = json.loads(answer)
    assert parsed["behavior"] in ("allow", "deny"), parsed
    return dict(parsed)


class _Recording:
    """An ``Asker`` that keeps the request and answers with a fixed decision."""

    def __init__(self, decision: Decision) -> None:
        self._decision = decision
        self.requests: list[Elicitation] = []

    async def ask(self, request: Elicitation) -> Decision:
        self.requests.append(request)
        return self._decision


@pytest.mark.asyncio
async def test_an_expiry_that_arrives_carrying_a_value_is_still_not_consent() -> None:
    """An ``expired`` decision with a ``value`` on it is what a real transport
    produces: a form that submitted its draft when the countdown ran out, a
    queue worker handing back the last thing it saw. Every field on it says
    "somebody was going to approve this", and none of them says they did.

    The danger is specific. ``_render`` asks ``Decision.approved`` and nothing
    else before allowing, so a predicate that counts ``expired`` sends the
    CLI ``behavior: "allow"`` — and because the decision carries a value, it
    can allow with ARGUMENTS the reviewer never confirmed. The trail then
    records an allowed call with a human's name on the clock that refused it.
    """
    asker = _Recording(
        Decision(kind="expired", value={"command": "rm -rf /"}, note="no answer within 30s")
    )
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Bash", {"command": "ls"}))

    assert Decision(kind="expired", value={"command": "ls"}).approved is False, (
        "the predicate is what the gate turns on, and a value does not make an "
        "expiry into an approval"
    )
    assert answer["behavior"] == "deny"
    assert "updatedInput" not in answer, (
        "an expiry that allowed would run the UNCONFIRMED arguments it carried"
    )
    assert server.decisions[0].allowed is False
    assert server.decisions[0].source == "timeout", (
        "the clock refused, not a person and not a broken transport — the three "
        "send whoever reads the trail to three different places"
    )


@pytest.mark.asyncio
async def test_the_request_tells_the_reviewer_what_question_it_is_and_how_long_they_have() -> None:
    """An ``Asker`` is caller-written code, and everything it can show a person
    has to be on the ``Elicitation``. ``kind="approval"`` with
    ``choices=("approve", "deny")`` is what says "a verdict, not free text" —
    without it an implementation renders a text box, the person types "yes", and
    the answer comes back as ``kind="value"``, which the gate correctly refuses
    as not-consent. So a request that failed to declare itself produces a run
    that denies everything while every component behaves as written.

    ``deadline_s`` is the same argument about the clock: the gate enforces
    ``timeout_s`` on its own side regardless, so a UI that was never told the
    deadline shows a prompt that silently stops mattering."""
    asker = _Recording(Decision(kind="approve"))
    server = ApprovalServer(
        asker=asker, autonomy="manual", timeout_s=30.0, run_id="run-7", agent="reviewer"
    )

    await server._decide("Write", {"file_path": "/etc/hosts", "content": "x"})

    (request,) = asker.requests
    assert request.kind == "approval", "free text is not an answer to this question"
    assert request.choices == ("approve", "deny"), "the closed set of answers is part of the ask"
    assert request.deadline_s == 30.0, "a UI cannot show a countdown it was never given"
    assert request.run_id == "run-7"
    assert request.agent == "reviewer"
    assert request.tool_call == {
        "name": "Write",
        "arguments": {"file_path": "/etc/hosts", "content": "x"},
    }, "the arguments ARE the thing being approved, so they travel structured"
    assert request.id, "a prompt with no id cannot be correlated with its answer"


@pytest.mark.asyncio
async def test_a_reviewer_who_edits_the_call_is_consenting_to_what_they_edited() -> None:
    """The positive control for the predicate, and the reason it cannot simply
    be ``kind == "approve"``. ``modify`` IS consent — it is how a reviewer
    redirects a write to a sandbox path instead of derailing the run — and the
    tool must then run with their arguments rather than the ones they refused.
    Paired with the expiry test above because the predicate has to separate
    exactly these two: one kind that carries a value and approves, one kind that
    carries a value and does not."""
    asker = _Recording(Decision(kind="modify", value={"file_path": "/tmp/safe"}))
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Write", {"file_path": "/etc/hosts"}))

    assert Decision(kind="modify", value={}).approved is True
    assert answer["behavior"] == "allow"
    assert answer["updatedInput"] == {"file_path": "/tmp/safe"}
    assert server.decisions[0].arguments == {"file_path": "/tmp/safe"}, (
        "the record names the call that RAN; the CLI does not tell the model it changed"
    )
