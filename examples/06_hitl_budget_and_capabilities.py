"""Example 06 — human-in-the-loop, recoverable budgets, and capability checks.

Three things an application used to have to build itself, demonstrated
together because they compose:

1. **Capability refusal at bind time.** `Agent(requires=("vision",))` is
   checked in `__post_init__`, against the model registry, BEFORE any spend.
   A model declared incapable raises `CapabilityMismatch`; a model the
   registry has never heard of is `UNKNOWN` — never assumed present, because
   guessing `True` is exactly the silent, well-formed wrong answer the check
   exists to catch.

2. **HITL as elicitation, parkable in place.** An injected `Asker` on
   `Services` turns a gated tool call into a PARK: the cognition awaits the
   person from inside its own coroutine, so live unserialisable state
   survives. With no `Asker` the classic checkpoint-and-resume path runs
   unchanged. The decision is typed (who answered, and when) and deadlined
   (an expiry degrades the run instead of hanging it).

3. **Recoverable budget exhaustion.** `Budget(on_exceeded="stop")` returns a
   verdict instead of raising, so the tool loop writes a CURRENT `suspended`
   checkpoint before it stops. Under the default `"raise"` the exception
   unwinds past every checkpoint write and the spend is unrecoverable.

Runs against `FakeLLM` — no API key needed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from agentkit import Agent, Budget, tool
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.adapters.llm import Capability, CapabilityMismatch, model_capabilities
from agentkit.agents.cognition import ReActCognition
from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.capabilities import Checkpointer
from agentkit.kernel.types import Delta, ToolCall, Usage
from agentkit.middlewares import meter
from agentkit.testing import make_test_ctx


@tool(side_effecting=True)
async def transfer(amount: str) -> str:
    """Move money between accounts. Side-effecting, so it gates for a human."""
    return f"transferred {amount}"


class AutoApprover:
    """An `Asker`. The entire HITL integration is this one method — the
    runtime never learns whether it's a terminal, an HTTP round trip, a
    queue, or a Slack bot. A real one would await a person here; awaiting
    is exactly what makes the run park rather than unwind."""

    def __init__(self, actor: str) -> None:
        self.actor = actor
        self.asked: list[Elicitation] = []

    async def ask(self, request: Elicitation) -> Decision:
        self.asked.append(request)
        return Decision(kind="approve", actor=self.actor, note="policy: under $1000")


class ExpensiveLLM:
    """Requests the gated tool each turn, at $0.60 a turn."""

    def __init__(self) -> None:
        self.n = 0

    async def stream(self, **_kw):
        self.n += 1
        yield Delta(text=f"turn {self.n}: transferring", model="m", provider="fake")
        yield Delta(
            tool_calls=(ToolCall(f"c{self.n}", "transfer", {"amount": "100"}),),
            usage=Usage(input_tokens=1000, output_tokens=100, cost_usd=0.60),
            finish_reason="tool_calls",
            model="m",
            provider="fake",
        )


def demo_capabilities() -> None:
    print("── 1. capability refusal, before any spend ─────────────")

    # Declared per model, never inferred from the name.
    print(f"claude-sonnet-4-6 vision : {model_capabilities('claude-sonnet-4-6').vision}")
    print(f"deepseek-chat     vision : {model_capabilities('deepseek-chat').vision}")
    print(f"our-own-finetune  vision : {model_capabilities('our-own-finetune').vision}")

    try:
        Agent("ocr", "deepseek-chat", requires=("vision",))
    except CapabilityMismatch as exc:
        print(f"\nrefused at construction: {str(exc).splitlines()[0]}")

    # UNKNOWN is never treated as present. Default policy warns and continues
    # so existing wiring with custom model names keeps working; "refuse" is
    # the setting for a service that pins its models.
    try:
        Agent("ocr", "our-own-finetune", requires=("vision",), on_unknown_capability="refuse")
    except CapabilityMismatch as exc:
        print(f"refused on UNKNOWN     : {str(exc).splitlines()[0][:96]}…")

    # A satisfied requirement is untouched.
    Agent("reader", "claude-sonnet-4-6", requires=("vision", "tools"))
    print("satisfied requirement  : constructed silently")
    assert model_capabilities("claude-haiku-4-5-20251001").vision is Capability.YES
    print("dated release ids normalise to their family row")


async def demo_park_and_budget() -> None:
    print("\n── 2 + 3. park for a human, then stop on the ceiling ───")

    asker = AutoApprover("alice@corp")
    cp = Checkpointer(port=InMemoryCheckpointStore())
    budget = Budget(max_cost_usd="1.00", on_exceeded="stop")
    ctx = make_test_ctx(
        llm=ExpensiveLLM(),
        budget=budget,
        checkpointer=cp,
        asker=asker,  # <- the one seam that turns a suspend into a park
        chat_middleware=[meter()],
        autonomy="manual",  # gate everything
        correlation_id="run-1",
    )
    agent = Agent(
        "banker",
        "claude-sonnet-4-6",
        prompt="Transfer the money.",
        cognition=ReActCognition(
            tools=[transfer],
            checkpointer=cp,
            approval_deadline_s=30,  # an abandoned tab degrades, never hangs
        ),
    )

    result = await agent.run("Send $100.", ctx)

    print(f"human asked {len(asker.asked)}x, all inline — the loop never unwound")
    for request in asker.asked:
        print(f"  gate {request.id}: {request.prompt} (deadline {request.deadline_s}s)")

    print(f"\nstop_reason  : {result.stop_reason}")
    print(f"is_suspended : {result.is_suspended}   (parked on a person? no)")
    print(f"is_resumable : {result.is_resumable}   (raise the ceiling and retry)")

    # Money is exact, and the token totals survived the whole run.
    print(f"\nspent (exact): {budget.spent()}  == Decimal('1.20') -> {budget.spent() == Decimal('1.20')}")
    print(f"spent (cents): {budget.spent_cents()}")
    print(f"spent_usd    : {budget.spent_usd}   (float mirror, for display only)")
    print(f"tokens       : in={budget.usage.input_tokens} out={budget.usage.output_tokens}")

    # The load-bearing bit: a CURRENT checkpoint exists, written before the
    # stop. Under on_exceeded="raise" the exception would have unwound past
    # every _save and left a stale snapshot recording half the spend.
    # The slot is namespaced PER PRODUCER, so it is not the bare run id: a
    # coordinator writing at ``run-1`` and its children writing at
    # ``run-1:agent:<name>`` would otherwise collide, and a child finishing
    # normally would delete its parent's in-progress state.
    # ``agent.resume(run_id, ...)`` re-derives this for you; only direct port
    # introspection like this needs to know.
    slot = ReActCognition.checkpoint_slot("run-1", agent.name)
    saved = await cp.resume(slot)
    assert saved is not None, f"no checkpoint at {slot!r}"
    print(f"\ncheckpoint   : status={saved.status} iteration={saved.state['iteration']}")
    print(f"               records ${saved.state['usage']['cost']:.2f} of the ${budget.spent()} spent")
    print("               -> nothing lost; raise the ceiling and resume")


async def main() -> None:
    demo_capabilities()
    await demo_park_and_budget()


if __name__ == "__main__":
    asyncio.run(main())
