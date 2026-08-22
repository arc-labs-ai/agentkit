"""Human-in-the-loop as value elicitation — parkable, deadlined, typed.

The four properties Brief 4 asks for, one section each:

(a) elicitation, not only approval — a run names what it needs and a person
    supplies a VALUE, from any cognition, nowhere near a tool call;
(b) parkable in place — a caller holding live, unserialisable state WAITS
    rather than unwinds, while return-and-resume stays available;
(c) deadlined — expiry is an ordinary recorded outcome, not a hang;
(d) typed — the decision carries who answered and when.

Plus the two cross-cutting requirements: a suspended run must be
distinguishable from a failed one, and the elicited value must never reach a
log, a payload, an error message, or a checkpoint.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition, SingleCallCognition
from agentkit.agents.control.elicitation import (
    Asker,
    Decision,
    Elicitation,
    ElicitationExpired,
    SecretValue,
    ask_human_tool,
    coerce_decision,
    elicit,
    is_context_tainted,
)
from agentkit.capabilities import Checkpointer
from agentkit.context import WorkingContext
from agentkit.kernel.types import Delta, ToolCall, Usage
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import tool

SECRET_CODE = "OTP-847263-DO-NOT-LEAK"


# ── test doubles ─────────────────────────────────────────────────────────────


class ScriptedAsker:
    """An ``Asker`` that answers from a script. The whole HITL integration is
    this one method — the runtime never learns whether it is a terminal, an
    HTTP round trip, or a queue."""

    def __init__(self, *answers: Decision) -> None:
        self._answers = list(answers)
        self.seen: list[Elicitation] = []

    async def ask(self, request: Elicitation) -> Decision:
        self.seen.append(request)
        if not self._answers:
            return Decision(kind="deny", actor="script", note="no scripted answer left")
        return self._answers.pop(0)


# Long enough that a working deadline (tests use 10-50ms) always wins the
# race, short enough that a BROKEN deadline costs seconds rather than hanging.
# The earlier value was 3600s, which meant any regression in the timeout path
# stopped being a test failure and became an hour-long CI hang.
_SILENT_ASKER_SLEEP_S = 5.0


class SilentAsker:
    """Never answers in time — the abandoned-tab case. Sleeps well past the
    deadlines these tests set, so the timeout path is what completes."""

    def __init__(self) -> None:
        self.asked = 0

    async def ask(self, request: Elicitation) -> Decision:
        self.asked += 1
        await asyncio.sleep(_SILENT_ASKER_SLEEP_S)
        raise AssertionError("unreachable: the deadline should have fired")  # pragma: no cover


@tool(side_effecting=True)
def transfer(amount: str) -> str:
    """Move money between accounts. Side-effecting, so the gate fires on it."""
    return f"transferred {amount}"


class ToolThenDone:
    """Requests ``transfer`` on the first turn, then answers normally — enough
    to drive one gate and then finish."""

    def __init__(self) -> None:
        self.n = 0

    async def stream(self, **_kw):
        self.n += 1
        if self.n == 1:
            yield Delta(text="I'll transfer it", model="m", provider="f")
            yield Delta(
                tool_calls=(ToolCall("t1", "transfer", {"amount": "100"}),),
                usage=Usage(1, 1, 0.0),
                finish_reason="tool_calls",
                model="m",
                provider="f",
            )
        else:
            yield Delta(text="done", model="m", provider="f")
            yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")


def _gated_ctx(**kw):
    """``autonomy="manual"`` gates everything, so the HITL path always fires."""
    return make_test_ctx(autonomy="manual", correlation_id="hitl-run", **kw)


def _run(agent: Agent, ctx):
    return asyncio.run(agent.run("go", ctx))


def _fresh_checkpointer() -> Checkpointer:
    from agentkit.adapters.checkpoint.in_memory import InMemoryCheckpointStore

    return Checkpointer(port=InMemoryCheckpointStore())


# ── (a) elicitation, not only approval ───────────────────────────────────────


def test_a_run_can_ask_a_person_for_a_value_from_any_cognition() -> None:
    """``elicit`` takes a ``Ctx``, not an ``Agent``, so it is reachable from a
    single-call cognition, a coordinator, or a hand-written one — none of
    which have a tool call to hang an approve/deny on."""
    asker = ScriptedAsker(Decision(kind="value", value="blue", actor="alice"))
    ctx = make_test_ctx(asker=asker)

    answer = asyncio.run(
        elicit(ctx, Elicitation(id="q1", prompt="favourite colour?", kind="value"))
    )
    assert answer.kind == "value" and answer.value == "blue"
    assert asker.seen[0].kind == "value"


def test_the_model_itself_can_ask_via_the_ask_human_tool() -> None:
    """The third route in: no framework gate fired and no cognition called
    ``elicit`` — the MODEL decided it needed something only a person has. That
    is structurally inexpressible as approve/deny, because the ask IS the
    action."""
    asker = ScriptedAsker(Decision(kind="value", value="1234", actor="bob"))
    ctx = make_test_ctx(asker=asker)
    tool_impl = ask_human_tool(deadline_s=5)

    result = asyncio.run(tool_impl.run({"question": "what was the code?"}, ctx))
    assert result == "1234"
    assert asker.seen[0].prompt == "what was the code?"


def test_a_missing_asker_denies_rather_than_silently_passing() -> None:
    """A gate with no human attached must not become a no-op. Direct
    ``elicit`` with nothing wired denies."""
    answer = asyncio.run(elicit(make_test_ctx(), Elicitation(id="q", prompt="ok?")))
    assert answer.kind == "deny"
    assert "no Asker" in answer.note


# ── (b) parkable in place ────────────────────────────────────────────────────


def test_a_parked_run_continues_holding_its_original_state() -> None:
    """The requirement ``resume()`` structurally cannot meet.

    With an ``Asker`` wired, the loop AWAITS the person from inside its own
    coroutine. Nothing unwinds — so the drive-local state (the working
    context, the accumulated usage, the termination clone) is the SAME
    objects afterwards, not a snapshot rehydrated from a store. A caller
    holding live, unserialisable state survives the pause.
    """
    asker = ScriptedAsker(Decision(kind="approve", actor="alice"))
    ctx = _gated_ctx(llm=ToolThenDone(), asker=asker)
    context = WorkingContext()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer]))

    async def go():
        return [ev async for ev in agent.stream("send it", ctx, context)]

    events = asyncio.run(go())
    final = next(ev for ev in events if ev.type == "final")

    # The run completed in ONE call — it never returned a suspended result and
    # was never resumed.
    assert final.result.stop_reason == "complete"
    assert final.result.output == "done"
    # The tool actually ran, with the human's approval, mid-loop.
    assert any(ev.type == "tool_result" and "transferred 100" in str(ev.tool_result) for ev in events)
    # And the caller's OWN context object carries the whole transcript — the
    # identity proof that nothing was rebuilt from a snapshot.
    assert any(m.role == "tool" and "transferred" in m.content for m in context.messages)


def test_the_interrupt_event_still_fires_on_the_park_path() -> None:
    """A consumer's event handling must not change depending on which HITL
    path ran — the operator UI still sees ``interrupt`` then ``tool_result``."""
    ctx = _gated_ctx(llm=ToolThenDone(), asker=ScriptedAsker(Decision(kind="approve")))
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer]))

    async def go():
        return [ev.type async for ev in agent.stream("send it", ctx)]

    types = asyncio.run(go())
    assert "interrupt" in types
    assert types.index("interrupt") < types.index("tool_result")


def test_without_an_asker_the_classic_suspend_and_resume_path_is_unchanged() -> None:
    """Additive, not a replacement. Callers that CAN serialise keep the
    return-and-resume behaviour exactly as it was."""
    cp = _fresh_checkpointer()
    # ONE llm across both halves: its turn counter must advance, or the
    # resumed loop re-requests the same tool and suspends again.
    llm = ToolThenDone()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer], checkpointer=cp))

    result = _run(agent, _gated_ctx(llm=llm, checkpointer=cp))  # no asker
    assert result.stop_reason == "suspended"
    assert result.evals["suspended"].run_id == "hitl-run"

    # ...and it resumes, with the legacy string decision shape.
    resumed = asyncio.run(
        agent.resume("hitl-run", {"t1": "approve"}, _gated_ctx(llm=llm, checkpointer=cp))
    )
    assert resumed.stop_reason == "complete"


def test_a_denial_on_the_park_path_reaches_the_model() -> None:
    ctx = _gated_ctx(llm=ToolThenDone(), asker=ScriptedAsker(Decision(kind="deny", actor="carol")))
    context = WorkingContext()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer]))
    asyncio.run(agent.run("send it", ctx, context=context))

    denial = next(m for m in context.messages if m.role == "tool")
    assert "DENIED" in denial.content
    assert "carol" in denial.content, "the model should be able to reason about WHO refused"


def test_modify_replaces_the_arguments_on_the_park_path() -> None:
    """The third useful answer: not yes, not no, but "do it differently"."""
    ctx = _gated_ctx(
        llm=ToolThenDone(),
        asker=ScriptedAsker(Decision(kind="modify", value='{"amount": "1"}', actor="dave")),
    )
    context = WorkingContext()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer]))
    asyncio.run(agent.run("send it", ctx, context=context))

    result_msg = next(m for m in context.messages if m.role == "tool")
    assert "transferred 1" in result_msg.content


# ── (c) deadlined ────────────────────────────────────────────────────────────


def test_a_deadline_expires_and_the_run_degrades_rather_than_dying() -> None:
    """The abandoned-tab case — one team's "silent-stuck finding #1".

    Nobody answers. The run does NOT hang, and does NOT raise: it records an
    expiry, tells the model, and carries on to a normal terminal result.
    """
    silent = SilentAsker()
    ctx = _gated_ctx(llm=ToolThenDone(), asker=silent)
    context = WorkingContext()
    agent = Agent(
        "banker",
        "m",
        cognition=ReActCognition(tools=[transfer], approval_deadline_s=0.05),
    )

    result = asyncio.run(agent.run("send it", ctx, context=context))

    assert silent.asked == 1
    assert result.stop_reason == "complete", "the run finished; it neither hung nor raised"
    expired_msg = next(m for m in context.messages if m.role == "tool")
    assert "EXPIRED" in expired_msg.content
    # Expiry is distinguishable from a refusal — "nobody was there" and
    # "someone said no" call for different operator responses.
    assert "DENIED" not in expired_msg.content


def test_expiry_is_a_typed_decision_not_an_exception() -> None:
    answer = asyncio.run(
        elicit(
            make_test_ctx(asker=SilentAsker()),
            Elicitation(id="q", prompt="?", deadline_s=0.05),
        )
    )
    assert answer.kind == "expired"
    assert answer.at > 0, "even an expiry is stamped, so the audit trail records when"


def test_a_caller_can_opt_into_raising_on_expiry() -> None:
    """For a caller that genuinely cannot continue without the answer."""
    from agentkit.agents.control.elicitation import elicit_or_raise

    with pytest.raises(ElicitationExpired):
        asyncio.run(
            elicit_or_raise(
                make_test_ctx(asker=SilentAsker()),
                Elicitation(id="q", prompt="?", deadline_s=0.05),
            )
        )


def test_no_deadline_means_wait_indefinitely_as_before() -> None:
    """The default is unchanged behaviour — a deadline is opt-in."""
    asker = ScriptedAsker(Decision(kind="approve"))
    asyncio.run(elicit(make_test_ctx(asker=asker), Elicitation(id="q", prompt="?")))
    assert asker.seen[0].deadline_s is None


def test_a_suspend_records_its_absolute_deadline_for_an_operator_ui() -> None:
    """On the return-and-resume path there is no coroutine to time out, so the
    deadline is carried as wall-clock expiry on ``Suspended`` — enough for a
    UI to render a countdown and for a late decision to be refused."""
    import time as _time

    cp = _fresh_checkpointer()
    ctx = _gated_ctx(llm=ToolThenDone(), checkpointer=cp)
    agent = Agent(
        "banker", "m", cognition=ReActCognition(tools=[transfer], checkpointer=cp, approval_deadline_s=60)
    )
    result = _run(agent, ctx)
    suspended = result.evals["suspended"]
    assert suspended.deadline_at is not None
    assert _time.time() < suspended.deadline_at <= _time.time() + 61


# ── (d) typed: who answered, and when ────────────────────────────────────────


def test_a_decision_carries_the_actor_and_the_timestamp() -> None:
    answer = asyncio.run(
        elicit(
            make_test_ctx(asker=ScriptedAsker(Decision(kind="approve", actor="alice@corp"))),
            Elicitation(id="q", prompt="?"),
        )
    )
    assert answer.actor == "alice@corp"
    assert answer.at > 0


def test_an_asker_that_forgets_provenance_still_gets_a_timestamp() -> None:
    """The floor. A lazy transport must not be able to produce an audit
    record with no time on it."""
    answer = asyncio.run(
        elicit(make_test_ctx(asker=ScriptedAsker(Decision(kind="approve"))), Elicitation(id="q", prompt="?"))
    )
    assert answer.at > 0 and answer.actor == "unknown"


def test_the_legacy_string_decision_map_still_works() -> None:
    """``dict[str, str]`` is coerced, so every existing ``resume`` caller keeps
    working from the same call site."""
    assert coerce_decision("approve").kind == "approve"
    assert coerce_decision("reject").kind == "deny"
    assert coerce_decision("deny").kind == "deny"
    modify = coerce_decision('{"amount": "5"}')
    assert modify.kind == "modify" and modify.value == '{"amount": "5"}'
    assert coerce_decision("approve").at > 0  # stamped even from the legacy shape


def test_resume_accepts_typed_decisions_too() -> None:
    """New callers get the audit trail without a second entry point."""
    cp = _fresh_checkpointer()
    llm = ToolThenDone()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer], checkpointer=cp))
    _run(agent, _gated_ctx(llm=llm, checkpointer=cp))

    resumed = asyncio.run(
        agent.resume(
            "hitl-run",
            {"t1": Decision(kind="approve", actor="alice@corp", at=1.0)},
            _gated_ctx(llm=llm, checkpointer=cp),
        )
    )
    assert resumed.stop_reason == "complete"


def test_an_unanswered_gate_defaults_to_denial() -> None:
    """An operator who answered three of four gates has not implicitly
    approved the fourth."""
    assert coerce_decision("reject").approved is False
    assert Decision(kind="expired").approved is False
    assert Decision(kind="approve").approved is True
    assert Decision(kind="modify").approved is True


# ── suspended is not failed ──────────────────────────────────────────────────


def test_a_suspended_run_reports_a_state_distinct_from_failure() -> None:
    """Waiting-for-you and it-fell-over are different states and a reader acts
    differently on each.

    A FAILED run raises and produces no ``AgentResult`` at all. A suspended
    run produces one whose ``stop_reason`` is typed ``"suspended"`` — readable
    without reaching into the ``evals`` bag, and type-checkable.
    """
    cp = _fresh_checkpointer()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer], checkpointer=cp))
    result = _run(agent, _gated_ctx(llm=ToolThenDone(), checkpointer=cp))

    assert result.stop_reason == "suspended"
    assert result.is_suspended is True
    assert result.is_resumable is True

    # Distinct from every other terminal state the framework can produce.
    completed = _run(Agent("plain", "m"), make_test_ctx(llm=FakeLLM("hi")))
    assert completed.stop_reason == "complete"
    assert completed.is_suspended is False and completed.is_resumable is False


def test_a_failing_run_raises_rather_than_reporting_suspended() -> None:
    """The other half of the distinction — a crash must not be mistakable for
    a park."""

    class Broken:
        async def stream(self, **_kw):
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover — makes this an async generator

    with pytest.raises(RuntimeError, match="provider exploded"):
        _run(Agent("a", "m"), make_test_ctx(llm=Broken()))


def test_stop_reason_is_typed_on_every_terminal_path() -> None:
    """A closed taxonomy is only useful if it is actually populated."""
    assert _run(Agent("a", "m"), make_test_ctx(llm=FakeLLM("hi"))).stop_reason == "complete"

    looping = Agent(
        "a", "m", cognition=ReActCognition(tools=[transfer], max_iterations=1)
    )
    out = _run(looping, make_test_ctx(llm=ToolThenDone(), autonomy="auto"))
    assert out.stop_reason == "max_iterations"


# ── the secret never escapes ─────────────────────────────────────────────────


def test_a_secret_value_refuses_to_render_itself() -> None:
    """It survives an f-string in a log line, an exception message, and a
    debugger pane — the three places a value ends up by accident."""
    secret = SecretValue(SECRET_CODE)
    assert SECRET_CODE not in repr(secret)
    assert SECRET_CODE not in str(secret)
    assert SECRET_CODE not in f"code was {secret}"
    assert SECRET_CODE not in str(RuntimeError(f"failed with {secret}"))
    assert secret.reveal() == SECRET_CODE  # the one explicit, greppable way out


def test_a_secret_value_does_not_compare_equal_to_a_bare_string() -> None:
    """So ``decision.value == "1234"`` never quietly works and becomes the
    idiom that bypasses the wrapper."""
    assert SecretValue("1234") != "1234"
    assert SecretValue("1234") == SecretValue("1234")


def test_the_value_never_reaches_a_log_a_payload_or_an_error_message() -> None:
    """Brief 4's security test, swept across every surface the value passes.

    The observer is the dangerous one: it fans out to Redis / Kafka / a UI
    socket, and it is exactly where an unredacted payload would escape the
    process.
    """
    emitted: list[object] = []

    class RecordingObserver:
        async def emit(self, observation):
            emitted.append(observation)

    asker = ScriptedAsker(Decision(kind="value", value=SECRET_CODE, actor="alice"))
    ctx = make_test_ctx(asker=asker, observer=RecordingObserver())

    answer = asyncio.run(
        elicit(ctx, Elicitation(id="otp", prompt="enter the code", kind="value", secret=True))
    )

    # Wrapped centrally, so protection doesn't depend on the Asker remembering.
    assert isinstance(answer.value, SecretValue)
    assert answer.value.reveal() == SECRET_CODE

    surfaces = [repr(answer), str(answer), repr(answer.redacted())]
    surfaces += [repr(o) for o in emitted]
    surfaces += [str(getattr(o, "payload", "")) for o in emitted]
    for surface in surfaces:
        assert SECRET_CODE not in surface, f"secret leaked into: {surface[:200]!r}"


def test_a_secret_prompt_is_redacted_too() -> None:
    """"enter the code we texted to +44…" is itself revealing."""
    request = Elicitation(id="q", prompt="code sent to +44 7700 900123", secret=True)
    assert "7700" not in request.redacted().prompt


def test_a_run_that_handled_a_secret_stops_checkpointing() -> None:
    """The load-bearing containment.

    A one-time code that entered the transcript would otherwise be serialised
    into Postgres, where it outlives by weeks the ten minutes it was valid
    for. The run loses durability — a real cost, and the right trade: an
    un-resumable run can be re-run, a leaked credential cannot be un-leaked.
    """
    cp = _fresh_checkpointer()
    context = WorkingContext()
    asker = ScriptedAsker(Decision(kind="value", value=SECRET_CODE, actor="alice"))
    ctx = make_test_ctx(asker=asker, checkpointer=cp, correlation_id="secret-run")

    asyncio.run(
        elicit(
            ctx,
            Elicitation(id="otp", prompt="code?", kind="value", secret=True),
            context=context,
        )
    )
    assert is_context_tainted(context)

    cognition = ReActCognition(tools=[transfer], checkpointer=cp)
    holder = Agent("holder", "m", cognition=cognition)
    asyncio.run(cognition._save(ctx, "secret-run", holder, context, Usage(), 1, False))

    slot = ReActCognition.checkpoint_slot("secret-run", holder.name)
    assert asyncio.run(cp.resume(slot)) is None, (
        "a context holding an elicited secret must never be snapshotted"
    )


def test_an_untainted_run_still_checkpoints_normally() -> None:
    """The containment is scoped to runs that actually handled a secret —
    every other run keeps its durability."""
    cp = _fresh_checkpointer()
    ctx = make_test_ctx(checkpointer=cp, correlation_id="normal-run")
    cognition = ReActCognition(tools=[transfer], checkpointer=cp)
    holder = Agent("holder", "m", cognition=cognition)
    asyncio.run(cognition._save(ctx, "normal-run", holder, WorkingContext(), Usage(), 1, False))
    slot = ReActCognition.checkpoint_slot("normal-run", holder.name)
    assert asyncio.run(cp.resume(slot)) is not None


def test_a_non_secret_elicitation_does_not_taint() -> None:
    context = WorkingContext()
    asyncio.run(
        elicit(
            make_test_ctx(asker=ScriptedAsker(Decision(kind="value", value="blue"))),
            Elicitation(id="q", prompt="colour?", kind="value"),
            context=context,
        )
    )
    assert not is_context_tainted(context)


# ── the transport really is the application's ────────────────────────────────


def test_the_runtime_never_branches_on_transport() -> None:
    """Structural assertion: ``Asker`` is a one-method Protocol, so anything
    with ``async def ask`` satisfies it — no registration, no subclassing, no
    ``if transport == "http"`` anywhere in the runtime."""

    class QueueBackedAsker:
        async def ask(self, request: Elicitation) -> Decision:
            return Decision(kind="approve", actor="queue-worker")

    assert isinstance(QueueBackedAsker(), Asker)
    ctx = _gated_ctx(llm=ToolThenDone(), asker=QueueBackedAsker())
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer]))
    assert _run(agent, ctx).stop_reason == "complete"


def test_elicit_works_from_a_non_react_cognition() -> None:
    """"Must work for cognitions other than ReAct." ``elicit`` takes a ``Ctx``,
    so a single-call cognition's own code can pause for a person without any
    tool-loop machinery."""

    class AskingCognition(SingleCallCognition):
        name: str = "asking"

        async def drive(self, agent, task, ctx, context):
            answer = await elicit(ctx, Elicitation(id="q", prompt="proceed?", kind="value"))
            async for ev in super().drive(f"{task} (human said {answer.value})", ctx, context):
                yield ev

    asker = ScriptedAsker(Decision(kind="value", value="yes", actor="alice"))
    ctx = make_test_ctx(llm=FakeLLM("ok"), asker=asker)
    answer = asyncio.run(elicit(ctx, Elicitation(id="q", prompt="proceed?", kind="value")))
    assert answer.value == "yes"
    # And the same ctx drives an ordinary single-call agent unchanged.
    assert _run(Agent("a", "m"), ctx).stop_reason == "complete"


# ── regressions found during review ──────────────────────────────────────────


def test_the_taint_constant_is_the_same_on_both_sides_of_the_layer_boundary() -> None:
    """``capabilities`` sits BELOW ``agents``, so the checkpointer cannot
    import the taint key and duplicates it instead. Pin the two equal — a
    silent drift would turn the containment into a no-op that still looks
    wired."""
    from agentkit.agents.control.elicitation import SECRET_TAINT_KEY
    from agentkit.capabilities.checkpointer.base import _SECRET_TAINT_KEY

    assert SECRET_TAINT_KEY == _SECRET_TAINT_KEY


def test_the_containment_covers_every_producer_not_just_the_tool_loop() -> None:
    """Regression: the guard first lived in ``ReActCognition._save``, which is
    one of SEVEN ``snapshot`` call sites — the coordinator policies persist a
    blackboard scratchpad through their own. Enforced at the ``Checkpointer``
    seam it holds for all of them, including the eighth nobody has written."""
    from agentkit.agents.control.elicitation import SECRET_TAINT_KEY

    cp = _fresh_checkpointer()
    # The shape a coordinator policy persists: scratchpad nested in the blob.
    coordinator_state = {
        "turn": 1,
        "transcript": [],
        "scratchpad": {SECRET_TAINT_KEY: True, "notes": "x"},
    }
    saved = asyncio.run(cp.snapshot("coord-run", coordinator_state))
    assert saved.version == 0 and saved.metadata == {"skipped": "secret_taint"}
    assert asyncio.run(cp.resume("coord-run")) is None

    # A flat blob (a caller passing a bare scratchpad as the whole state).
    asyncio.run(cp.snapshot("flat-run", {SECRET_TAINT_KEY: True}))
    assert asyncio.run(cp.resume("flat-run")) is None

    # And an untainted coordinator state is unaffected.
    asyncio.run(cp.snapshot("clean-run", {"turn": 1, "scratchpad": {"notes": "x"}}))
    assert asyncio.run(cp.resume("clean-run")) is not None


def test_the_park_path_emits_tool_call_like_the_ungated_path() -> None:
    """Regression: the park path yielded ``interrupt`` then ``tool_result``
    with no ``tool_call`` in between, so a consumer counting ``tool_call``
    events to render "running X…" saw nothing on approved gates."""
    ctx = _gated_ctx(llm=ToolThenDone(), asker=ScriptedAsker(Decision(kind="approve")))
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer]))

    async def go():
        return [ev.type async for ev in agent.stream("send it", ctx)]

    types = asyncio.run(go())
    assert types.index("interrupt") < types.index("tool_call") < types.index("tool_result")


def test_a_denied_gate_emits_no_tool_call() -> None:
    """The corollary: nothing ran, so nothing should claim to have run."""
    ctx = _gated_ctx(llm=ToolThenDone(), asker=ScriptedAsker(Decision(kind="deny")))
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer]))

    async def go():
        return [ev.type async for ev in agent.stream("send it", ctx)]

    types = asyncio.run(go())
    assert "interrupt" in types and "tool_call" not in types


def test_ask_human_tool_ids_are_stable_across_processes() -> None:
    """``hash(str)`` is randomised per interpreter (PYTHONHASHSEED), so an id
    built from it differs between the process that asked and any process later
    reading the audit trail. Pinned against a literal digest so a switch back
    to ``hash()`` fails here rather than in someone's log correlation."""
    asker = ScriptedAsker(Decision(kind="value", value="x"), Decision(kind="value", value="x"))
    ctx = make_test_ctx(asker=asker)
    impl = ask_human_tool()

    asyncio.run(impl.run({"question": "what was the code?"}, ctx))
    asyncio.run(impl.run({"question": "what was the code?"}, ctx))
    first, second = asker.seen[0].id, asker.seen[1].id

    assert first == second
    import hashlib

    expected = hashlib.sha256(b"what was the code?").hexdigest()[:12]
    assert first == f"ask_human:{expected}"


def test_a_blocking_asker_cannot_be_deadlined_and_says_so() -> None:
    """Documents an inherent limit rather than a fixable bug.

    Scheduling is cooperative: a synchronous wait inside ``ask`` never yields,
    so the timeout coroutine never runs and ``deadline_s`` silently becomes an
    unbounded hang. The `Asker` Protocol carries an explicit warning; this
    test pins the behaviour so the warning cannot quietly become untrue.
    """
    import time as _time

    class BlockingAsker:
        async def ask(self, request: Elicitation) -> Decision:
            # The anti-pattern under test. ruff ASYNC251 is exactly right to
            # flag it — that is the point: this is what a naive Asker does, and
            # it silently disables every deadline.
            _time.sleep(0.2)  # noqa: ASYNC251 — deliberate; see above
            return Decision(kind="value", value="late", actor="slow")

    answer = asyncio.run(
        elicit(
            make_test_ctx(asker=BlockingAsker()),
            Elicitation(id="q", prompt="?", kind="value", deadline_s=0.01),
        )
    )
    # NOT "expired" — the deadline could not fire.
    assert answer.kind == "value" and answer.value == "late"

    # The Protocol must keep telling people. Asserted on the docstring so
    # deleting the warning fails a test rather than only a review.
    from agentkit.agents.control.elicitation import Asker as AskerProtocol

    doc = AskerProtocol.__doc__ or ""
    assert "must not block the event loop" in doc and "to_thread" in doc


def test_an_awaiting_asker_is_deadlined_correctly() -> None:
    """The contrast: the same wait done properly IS bounded."""

    class SleepingAsker:
        async def ask(self, request: Elicitation) -> Decision:
            await asyncio.sleep(0.2)
            return Decision(kind="value", value="late")

    answer = asyncio.run(
        elicit(
            make_test_ctx(asker=SleepingAsker()),
            Elicitation(id="q", prompt="?", kind="value", deadline_s=0.01),
        )
    )
    assert answer.kind == "expired"


def test_the_runpolicy_gate_fires_on_the_resume_path_too() -> None:
    """Regression, security-relevant: ``Agent.resume`` used to skip the
    lethal-trifecta gate entirely.

    The gate lived inline in ``stream()``, so an agent whose tool set combines
    private-data access, untrusted-content ingestion, and egress was denied on
    ``run()`` and then executed that exact tool on ``resume()``. Resume is the
    worst possible place for the gate to be missing: it is the path a human
    has just approved something on, and approving one tool CALL is not
    approval of the capability COMBINATION.
    """
    from agentkit.agents.control.safety import TRIFECTA, RunPolicy

    @tool(side_effecting=True, caps=tuple(TRIFECTA))
    def exfil(url: str) -> str:
        """Read private data, ingest untrusted content, and send it onward."""
        return "sent"

    class ExfilThenDone:
        def __init__(self) -> None:
            self.n = 0

        async def stream(self, **_kw):
            self.n += 1
            if self.n == 1:
                yield Delta(text="x", model="m", provider="f")
                yield Delta(
                    tool_calls=(ToolCall("t1", "exfil", {"url": "http://x"}),),
                    usage=Usage(1, 1, 0.0),
                    finish_reason="tool_calls",
                    model="m",
                    provider="f",
                )
            else:
                yield Delta(text="done", model="m", provider="f")
                yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")

    cp = _fresh_checkpointer()
    llm = ExfilThenDone()

    # Suspend it first, with no policy attached, so a checkpoint exists.
    ungated = Agent("a", "m", cognition=ReActCognition(tools=[exfil], checkpointer=cp))
    assert _run(ungated, _gated_ctx(llm=llm, checkpointer=cp)).stop_reason == "suspended"

    # An operator now attaches a deny-mode policy. BOTH entry points refuse.
    guarded = Agent(
        "a",
        "m",
        policy=RunPolicy(mode="deny"),
        cognition=ReActCognition(tools=[exfil], checkpointer=cp),
    )
    with pytest.raises(PermissionError):
        _run(guarded, _gated_ctx(llm=llm, checkpointer=cp))
    with pytest.raises(PermissionError):
        asyncio.run(
            guarded.resume("hitl-run", {"t1": "approve"}, _gated_ctx(llm=llm, checkpointer=cp))
        )


def test_a_benign_tool_set_resumes_normally_under_a_deny_policy() -> None:
    """The gate must not become a blanket block on resume — only a real
    trifecta refuses."""
    from agentkit.agents.control.safety import RunPolicy

    cp = _fresh_checkpointer()
    llm = ToolThenDone()
    agent = Agent(
        "banker",
        "m",
        policy=RunPolicy(mode="deny"),
        cognition=ReActCognition(tools=[transfer], checkpointer=cp),
    )
    assert _run(agent, _gated_ctx(llm=llm, checkpointer=cp)).stop_reason == "suspended"
    resumed = asyncio.run(
        agent.resume("hitl-run", {"t1": "approve"}, _gated_ctx(llm=llm, checkpointer=cp))
    )
    assert resumed.stop_reason == "complete"


def test_a_resume_arriving_after_the_deadline_expires_rather_than_acting() -> None:
    """The deadline on the SUSPEND path had to become real, not decorative.

    It was stamped onto ``Suspended.deadline_at`` and then never checked — and
    never even persisted, so the process that resumes (usually not the process
    that suspended) had no way to check it. An operator answering an hour late
    silently got the tool executed.

    Now the deadline is persisted with the checkpoint and honoured on resume:
    the run DEGRADES — every pending call becomes ``expired`` and the loop
    continues — rather than acting on a decision nobody was still entitled to
    make. ``EXPIRED`` is worded distinctly from ``DENIED`` so the transcript
    records which of the two happened.
    """
    fired: list[str] = []

    @tool(side_effecting=True)
    def spy_transfer(amount: str) -> str:
        """Move money between accounts, recording that it actually happened."""
        fired.append(amount)
        return f"transferred {amount}"

    class SpyThenDone(ToolThenDone):
        async def stream(self, **_kw):
            self.n += 1
            if self.n == 1:
                yield Delta(text="I'll transfer it", model="m", provider="f")
                yield Delta(
                    tool_calls=(ToolCall("t1", "spy_transfer", {"amount": "100"}),),
                    usage=Usage(1, 1, 0.0),
                    finish_reason="tool_calls",
                    model="m",
                    provider="f",
                )
            else:
                yield Delta(text="done", model="m", provider="f")
                yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")

    cp = _fresh_checkpointer()
    llm = SpyThenDone()
    agent = Agent(
        "banker",
        "m",
        cognition=ReActCognition(
            tools=[spy_transfer], checkpointer=cp, approval_deadline_s=-1  # already past
        ),
    )
    suspended = _run(agent, _gated_ctx(llm=llm, checkpointer=cp))
    assert suspended.stop_reason == "suspended"
    assert suspended.evals["suspended"].deadline_at is not None

    resumed = asyncio.run(
        agent.resume("hitl-run", {"t1": "approve"}, _gated_ctx(llm=llm, checkpointer=cp))
    )
    # Degraded and finished — it neither hung nor raised...
    assert resumed.stop_reason == "complete"
    # ...and, the load-bearing part, the late approval did NOT move the money.
    assert fired == [], "a decision arriving after the deadline still executed the tool"


def test_a_resume_inside_the_deadline_still_acts_normally() -> None:
    """The deadline must not become a blanket refusal."""
    cp = _fresh_checkpointer()
    llm = ToolThenDone()
    agent = Agent(
        "banker",
        "m",
        cognition=ReActCognition(tools=[transfer], checkpointer=cp, approval_deadline_s=3600),
    )
    _run(agent, _gated_ctx(llm=llm, checkpointer=cp))
    resumed = asyncio.run(
        agent.resume("hitl-run", {"t1": "approve"}, _gated_ctx(llm=llm, checkpointer=cp))
    )
    assert resumed.stop_reason == "complete"
    saved = asyncio.run(cp.resume("hitl-run"))
    assert saved is None or all(
        "EXPIRED" not in m["content"] for m in saved.state["messages"] if m["role"] == "tool"
    )


def test_no_deadline_configured_means_a_resume_is_never_too_late() -> None:
    """Unbounded waiting stays the default — a deadline is opt-in."""
    cp = _fresh_checkpointer()
    llm = ToolThenDone()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[transfer], checkpointer=cp))
    suspended = _run(agent, _gated_ctx(llm=llm, checkpointer=cp))
    assert suspended.evals["suspended"].deadline_at is None
    resumed = asyncio.run(
        agent.resume("hitl-run", {"t1": "approve"}, _gated_ctx(llm=llm, checkpointer=cp))
    )
    assert resumed.stop_reason == "complete"


def test_resume_with_no_decision_for_a_pending_call_does_not_run_the_tool() -> None:
    """Kills: the resume default flipping from deny to approve.

    ``test_an_unanswered_gate_defaults_to_denial`` looked like it covered this
    and did not — it asserted on ``coerce_decision`` in isolation and never
    drove ``resume()`` with a gap in the map. Mutation testing found the hole:
    changing the ``decisions.get(tc.id, "reject")`` default to ``"approve"``
    left the whole HITL suite green.

    The real property is about consequences, not about a helper's return
    value: an operator who answers three of four gates has not implicitly
    approved the fourth, so the tool must not fire. Asserted by observing
    whether the side effect happened.
    """
    fired: list[str] = []

    @tool(side_effecting=True)
    def spy(amount: str) -> str:
        """Move money between accounts, recording that it actually happened."""
        fired.append(amount)
        return f"transferred {amount}"

    class SpyThenDone:
        def __init__(self) -> None:
            self.n = 0

        async def stream(self, **_kw):
            self.n += 1
            if self.n == 1:
                yield Delta(text="x", model="m", provider="f")
                yield Delta(
                    tool_calls=(ToolCall("t1", "spy", {"amount": "100"}),),
                    usage=Usage(1, 1, 0.0),
                    finish_reason="tool_calls",
                    model="m",
                    provider="f",
                )
            else:
                yield Delta(text="done", model="m", provider="f")
                yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")

    cp = _fresh_checkpointer()
    llm = SpyThenDone()
    agent = Agent("banker", "m", cognition=ReActCognition(tools=[spy], checkpointer=cp))
    assert _run(agent, _gated_ctx(llm=llm, checkpointer=cp)).stop_reason == "suspended"

    # An EMPTY decision map — the operator closed the tab, or answered a
    # different run's gate, or the payload lost a key in transit.
    resumed = asyncio.run(agent.resume("hitl-run", {}, _gated_ctx(llm=llm, checkpointer=cp)))

    assert resumed.stop_reason == "complete"  # the run still finishes
    assert fired == [], "an unanswered gate executed the tool"


def test_resume_answering_only_some_gates_runs_only_those() -> None:
    """The partial-answer case, which is the realistic one: a UI that renders
    three approvals and submits after two."""
    fired: list[str] = []

    @tool(side_effecting=True)
    def spy2(which: str) -> str:
        """Perform a side-effecting action, recording which one ran."""
        fired.append(which)
        return f"did {which}"

    class TwoGatesThenDone:
        def __init__(self) -> None:
            self.n = 0

        async def stream(self, **_kw):
            self.n += 1
            if self.n == 1:
                yield Delta(text="x", model="m", provider="f")
                yield Delta(
                    tool_calls=(
                        ToolCall("a", "spy2", {"which": "alpha"}),
                        ToolCall("b", "spy2", {"which": "beta"}),
                    ),
                    usage=Usage(1, 1, 0.0),
                    finish_reason="tool_calls",
                    model="m",
                    provider="f",
                )
            else:
                yield Delta(text="done", model="m", provider="f")
                yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")

    cp = _fresh_checkpointer()
    llm = TwoGatesThenDone()
    agent = Agent("op", "m", cognition=ReActCognition(tools=[spy2], checkpointer=cp))
    _run(agent, _gated_ctx(llm=llm, checkpointer=cp))

    asyncio.run(agent.resume("hitl-run", {"a": "approve"}, _gated_ctx(llm=llm, checkpointer=cp)))

    assert fired == ["alpha"], f"expected only the approved gate to fire, got {fired}"
