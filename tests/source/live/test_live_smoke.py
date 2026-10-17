"""End-to-end smoke against REAL providers and REAL CLIs.

Why this file exists
--------------------
Every other test in this repository fakes the far side. Provider HTTP goes
through mock transports, and the CLI tests either drive a double or ask the real
binary only to PARSE argv. That is deliberate and it is also a hole: a green
suite says "a regression would be caught", not "a real call works". Those are
different claims, and only this file makes the second one.

What it covers is chosen to be the things reality can disagree with:

- a real completion, including real streaming, on both API modes
- a real TURN through `claude` and through `codex` — an actual answer, not flags
- structured output against a real model, where schema handling actually bites
- usage and cost against REAL receipts, which is the one way to catch a ledger
  that silently reports $0.00
- the approval gate driven by a real CLI permission prompt, which no other test
  reaches: everything else exercises `_decide` directly, and this is the path
  where failing open would be worst

Running it
----------
It is opt-in twice over, because it spends real money::

    export AGENTKIT_LIVE=1
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
    uv run pytest tests/source/live -q

Without ``AGENTKIT_LIVE`` every test here skips, so CI stays key-free and no
run bills anybody by accident. Each test also skips individually when its own
credential or binary is absent, so a host with one of the two still gets the
half it can actually check.

Models are the cheapest PRICED ones, deliberately. The runtime's
``unpriced_policy`` defaults to REFUSE, so an unpriced model fails before it
reaches the network — and asserting on cost is half the point of being here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.constants import CostSource
from agentkit.models.contracts.schema import OutputSpec
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef
from agentkit.models.providers.cli.calls import CliAdapter
from agentkit.models.providers.frontiers.calls import ApiAdapter

# Live I/O is allowed to warn. The suite turns warnings into errors, which is
# right for our own code and wrong for a third-party client talking to a real
# endpoint — a deprecation inside an SDK is not this repository's regression.
pytestmark = [
    pytest.mark.live,
    pytest.mark.filterwarnings("default"),
]

_OPTED_IN = os.environ.get("AGENTKIT_LIVE") == "1"
_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_CLAUDE_CLI = shutil.which("claude") is not None
_CODEX_CLI = shutil.which("codex") is not None

# The cheapest models the bundled table prices.
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-4o-mini"

needs_optin = pytest.mark.skipif(
    not _OPTED_IN, reason="live tests spend real money; set AGENTKIT_LIVE=1 to opt in"
)
needs_anthropic = pytest.mark.skipif(not _ANTHROPIC, reason="needs ANTHROPIC_API_KEY")
needs_openai = pytest.mark.skipif(not _OPENAI, reason="needs OPENAI_API_KEY")
needs_claude_cli = pytest.mark.skipif(not _CLAUDE_CLI, reason="needs the `claude` binary on PATH")
needs_codex_cli = pytest.mark.skipif(not _CODEX_CLI, reason="needs the `codex` binary on PATH")


def _env(name: str) -> CredentialRef:
    """Read the key from the environment at call time and never anywhere else.

    The value is not logged, not echoed into argv and not stored on the Binding
    — ``CredentialRef`` exists so the credential is resolved at the moment of
    use by the adapter that needs it.
    """
    return CredentialRef(lambda: {name: os.environ[name]})


def _runtime(**kwargs: Any) -> CallRuntime:
    return CallRuntime(pricing=PriceTable.default(), **kwargs)


def _assert_really_billed(answer: Any) -> None:
    """The receipt has to come from the provider, not from our own arithmetic.

    This is the assertion the whole file is worth writing for. A ledger that
    reports 0 tokens and $0.00 while the answer is correct looks like success
    everywhere except the invoice, and no mocked test can tell the difference —
    the fake supplies whatever usage the fixture author typed.
    """
    total_in = (
        answer.usage.input_uncached
        + answer.usage.input_cache_read
        + answer.usage.input_cache_write_5m
        + answer.usage.input_cache_write_1h
    )
    assert total_in > 0, f"no input tokens reported by the provider: {answer.usage}"
    assert answer.usage.output > 0, f"no output tokens reported: {answer.usage}"
    assert answer.raw_usage, "the provider's own usage payload was not retained"
    assert answer.cost_usd is not None and answer.cost_usd > 0, (
        f"a real call cost nothing: {answer.cost_usd!r} from {answer.cost_source!r}"
    )
    assert answer.cost_source in (CostSource.PRICE_TABLE, CostSource.MIXED, CostSource.PROVIDER)


# ── the API modes ────────────────────────────────────────────────────────────


@needs_optin
@needs_anthropic
@pytest.mark.asyncio
async def test_a_real_anthropic_completion_answers_and_is_billed() -> None:
    async with _runtime(adapters={Mode.ANTHROPIC_API: ApiAdapter()}) as runtime:
        answer = await runtime.ask(
            Binding(Mode.ANTHROPIC_API, ANTHROPIC_MODEL, _env("ANTHROPIC_API_KEY")),
            "Reply with exactly the word: pong",
            limits=CallLimits(max_turns=1),
            call_id="live-anthropic",
        )
    assert answer.outcome == "answered", answer.failure
    assert "pong" in (answer.text or "").lower(), answer.text
    _assert_really_billed(answer)


@needs_optin
@needs_openai
@pytest.mark.asyncio
async def test_a_real_openai_completion_answers_and_is_billed() -> None:
    async with _runtime(adapters={Mode.OPENAI_API: ApiAdapter()}) as runtime:
        answer = await runtime.ask(
            Binding(Mode.OPENAI_API, OPENAI_MODEL, _env("OPENAI_API_KEY")),
            "Reply with exactly the word: pong",
            limits=CallLimits(max_turns=1),
            call_id="live-openai",
        )
    assert answer.outcome == "answered", answer.failure
    assert "pong" in (answer.text or "").lower(), answer.text
    _assert_really_billed(answer)


@needs_optin
@needs_anthropic
@pytest.mark.asyncio
async def test_a_real_model_fills_a_schema_we_can_validate() -> None:
    """Structured output is where a real model disagrees with a fake one.

    A double returns whatever the fixture says, so every schema path passes. A
    real model has to be steered into the shape, and the adapter has to coerce
    and validate what comes back — including refusing what does not fit.
    """
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population_millions": {"type": "number"},
        },
        "required": ["city", "population_millions"],
        "additionalProperties": False,
    }
    async with _runtime(adapters={Mode.ANTHROPIC_API: ApiAdapter()}) as runtime:
        answer = await runtime.ask(
            Binding(Mode.ANTHROPIC_API, ANTHROPIC_MODEL, _env("ANTHROPIC_API_KEY")),
            "Give the population of Tokyo, in millions.",
            OutputSpec(schema),
            limits=CallLimits(max_turns=2),
            call_id="live-schema",
        )
    assert answer.outcome == "answered", answer.failure
    assert isinstance(answer.value, dict), answer.value
    assert set(answer.value) == {"city", "population_millions"}, answer.value
    assert isinstance(answer.value["population_millions"], (int, float))
    _assert_really_billed(answer)


# ── the CLI modes: a real turn, not argv ─────────────────────────────────────


@needs_optin
@needs_claude_cli
@pytest.mark.asyncio
async def test_a_real_claude_cli_turn_answers(tmp_path: Path) -> None:
    async with _runtime(
        adapters={Mode.CLAUDE_CODE: CliAdapter(working_dir=tmp_path)}
    ) as runtime:
        answer = await runtime.ask(
            Binding(Mode.CLAUDE_CODE, ANTHROPIC_MODEL),
            "Reply with exactly the word: pong. Do not use any tools.",
            limits=CallLimits(max_turns=1, wall_clock_s=180),
            call_id="live-claude-cli",
        )
    assert answer.outcome == "answered", answer.failure
    assert "pong" in (answer.text or "").lower(), answer.text
    assert answer.session, "a CLI turn must report the session it ran in"


@needs_optin
@needs_codex_cli
@pytest.mark.asyncio
async def test_a_real_codex_cli_turn_answers(tmp_path: Path) -> None:
    async with _runtime(adapters={Mode.CODEX: CliAdapter(working_dir=tmp_path)}) as runtime:
        answer = await runtime.ask(
            Binding(Mode.CODEX, OPENAI_MODEL),
            "Reply with exactly the word: pong. Do not use any tools.",
            limits=CallLimits(max_turns=1, wall_clock_s=180),
            call_id="live-codex-cli",
        )
    assert answer.outcome == "answered", answer.failure
    assert "pong" in (answer.text or "").lower(), answer.text


# ── the approval gate, driven by a real CLI ──────────────────────────────────


@needs_optin
@needs_claude_cli
@pytest.mark.asyncio
async def test_a_real_cli_permission_prompt_is_refused_and_recorded(tmp_path: Path) -> None:
    """The one path nothing else reaches.

    Every other approval test calls ``_decide`` directly. This one makes the
    real binary ask: it wires `ApprovalServer` in as the CLI's
    ``--permission-prompt-tool``, gives it a reviewer that refuses everything,
    and asks the agent to write a file.

    Two things are asserted, and the second is the one that matters. The
    reviewer must be consulted — otherwise the gate is not in the path at all —
    and the file must NOT exist, because a gate that records a refusal while
    the tool runs anyway is worse than no gate.
    """
    pytest.importorskip("mcp", reason="needs the `mcp` extra")
    from agentkit.integrations.mcp.approvals import ApprovalServer, Decision

    target = tmp_path / "must-not-exist.txt"
    asked: list[str] = []

    class RefuseEverything:
        async def ask(self, request: Any) -> Decision:
            call = request.tool_call or {}
            asked.append(str(call.get("name", "")))
            return Decision(kind="deny", note="refused by the live smoke test")

    async with ApprovalServer(
        asker=RefuseEverything(), autonomy="manual", timeout_s=120
    ) as approvals:
        async with _runtime(
            adapters={Mode.CLAUDE_CODE: CliAdapter(working_dir=tmp_path)}
        ) as runtime:
            await runtime.ask(
                Binding(Mode.CLAUDE_CODE, ANTHROPIC_MODEL),
                f"Create a file called {target.name} containing the word hello. "
                "Use your file-writing tool.",
                limits=CallLimits(max_turns=3, wall_clock_s=240),
                workspace=approvals.workspace(),
                call_id="live-approval-gate",
            )

        assert asked, (
            "the real CLI never asked the approval server, so the gate was not in "
            "the path — check --permission-prompt-tool wiring before trusting any "
            "other assertion here"
        )
        assert approvals.decisions, "a prompt was answered but never recorded"
        assert all(not d.allowed for d in approvals.decisions), [
            (d.tool, d.allowed, d.reason) for d in approvals.decisions
        ]

    assert not target.exists(), (
        "the reviewer refused and the tool ran anyway — the gate recorded a "
        "denial it did not enforce"
    )
