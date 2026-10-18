"""Execution assessment for the two SDK modes: what the API path really enforces.

WHY THIS FILE EXISTS
--------------------
`ExecutionRequirements` is a security contract that fails closed, and the only thing standing
between a host's stated requirement and a dispatch is an assessment matrix. For the two CLI modes
that matrix is produced by `evaluate_cli_execution` and is honest. For `openai_api` and
`anthropic_api` it was, measured on `main`:

    anthropic_api | default reqs      | compatible=True  backend=unspecified assessed=0 unmet=0
    anthropic_api | HOST_TOOLS/SCHEMA | compatible=False backend=unspecified assessed=4 unmet=4

Both lines are wrong, and in opposite directions. The first asserts NOTHING while reading as a
pass: a host that calls `evaluate_execution` to decide whether a binding is usable gets
`compatible=True` with an empty matrix, which is indistinguishable from a verified clean bill of
health. The second is the dangerous one: `host_tools`, `schema` and `usage` were reported
UNSUPPORTED, so `CallRuntime.run` REFUSED both SDK modes outright for any host that states
`HOST_TOOLS` — even though SDK mode is where host tools have the most authority, because the
runtime invokes the host's own functions through its own `ToolSet` with validation, approval
gating and timeouts, and no provider-native tool exists at all. The verdict was inverted on
exactly the capability the abstraction is for.

So the tests here are not "does the function return a matrix". They are:

* every requirement is answered for every mode, with transport identity, and without resolving a
  credential (an assessment that spends a secret cannot be used to CHOOSE a binding);
* a host that needs only host tools, structured output and usage metering can actually RUN on both
  SDK modes — asserted through `CallRuntime.run` against a mock wire, because an assessment that
  says SUPPORTED while `run` refuses is worse than no assessment;
* the in-process boundaries (`filesystem_*`, `network`, `processes`, `credentials`,
  `configuration`) are still refused with ZERO provider attempts, since a host callback runs in
  this process with full host authority and nothing sandboxes it;
* a transport that DECLARES a guarantee cannot promote itself to SUPPORTED — the invariant
  `BackendReport.guarantees` already carries on the CLI side, where a declaration may only
  downgrade a verdict to UNVERIFIED;
* the same matrix governs the assessment and the dispatch: requirements are re-checked inside
  `ApiAdapter.turn`, so they bind through retry, continuation and correction turns rather than
  only at the runtime's door.

HARNESS
-------
`httpx.MockTransport` (as the neighbouring `test_api_adapter_turn.py` and
`test_anthropic_response_format.py` do), because "no provider attempt" and "the tool declaration
reached the wire" are both claims about requests, and a dict-level fake cannot make either.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict
from decimal import Decimal
from typing import Any

import pytest

from agentkit.agents.prompts.composition import inline_prompt
from agentkit.core.contracts.types import Mode
from agentkit.core.execution.concurrency import CancellationToken
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.constants import FailureClass, NativeToolPolicy
from agentkit.models.contracts.execution import (
    CapabilityStatus,
    ExecutionRequirement,
    ExecutionRequirements,
    NativeExecution,
)
from agentkit.models.contracts.schema import OutputSpec
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef, ModelSettings
from agentkit.models.providers.cli.calls import CliAdapter
from agentkit.tools import ToolRegistry, tool
from agentkit.tools.execution.policy import ToolSet

httpx = pytest.importorskip("httpx")

from agentkit.models.providers.frontiers.calls import ApiAdapter
from agentkit.models.providers.frontiers.capabilities import TransportReport, evaluate_api_execution

_SDK_MODES = (Mode.ANTHROPIC_API, Mode.OPENAI_API)
_ALL_MODES = (Mode.CLAUDE_CODE, Mode.CODEX, Mode.ANTHROPIC_API, Mode.OPENAI_API)
_SECRET = "planted-private-api-key"

_PRICES = PriceTable(
    "test",
    "2026-10-13",
    {
        "test": {
            "input_uncached": "1",
            "input_cache_read": "0.1",
            "input_cache_write_5m": "1.25",
            "input_cache_write_1h": "2",
            "output": "5",
        }
    },
)

_ANSWER = OutputSpec(
    {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    "answer",
)

# What the host asks for when it wants its OWN functions called with a validated schema and a
# priced token count — the ordinary, non-containment request that must not be refused.
_HOST_PROTOCOL = ExecutionRequirements(
    required=frozenset(
        {ExecutionRequirement.HOST_TOOLS, ExecutionRequirement.SCHEMA, ExecutionRequirement.USAGE}
    )
)


def _credential(mode: Mode) -> CredentialRef:
    key = "ANTHROPIC_API_KEY" if mode == Mode.ANTHROPIC_API else "OPENAI_API_KEY"
    return CredentialRef(lambda: {key: _SECRET})


def _forbidden_credential() -> CredentialRef:
    def resolve() -> dict[str, str]:
        raise AssertionError("an assessment must not resolve a credential")

    return CredentialRef(resolve)


def _tool_call_frames(mode: Mode) -> tuple[dict[str, Any], ...]:
    """One provider turn that asks for the host tool `lookup_fact`."""
    if mode == Mode.ANTHROPIC_API:
        return (
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call-1", "name": "lookup_fact", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"subject": "ledger"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        )
    return (
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "lookup_fact",
                        "arguments": '{"subject": "ledger"}',
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        },
    )


def _final_frames(mode: Mode, text: str) -> tuple[dict[str, Any], ...]:
    """One provider turn that answers with `text`."""
    if mode == Mode.ANTHROPIC_API:
        return (
            {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        )
    return (
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        },
    )


class _Wire:
    """A `MockTransport` client that serves one scripted SSE body per request, in order."""

    def __init__(self, bodies: Iterable[tuple[dict[str, Any], ...]]) -> None:
        self._bodies = list(bodies)
        self.payloads: list[dict[str, Any]] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._respond))

    def _respond(self, request: Any) -> Any:
        self.payloads.append(json.loads(request.content))
        if not self._bodies:
            raise AssertionError(f"the adapter made {len(self.payloads)} requests; the script had fewer")
        events = self._bodies.pop(0)
        body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


@asynccontextmanager
async def _wire(*bodies: tuple[dict[str, Any], ...]) -> AsyncIterator[_Wire]:
    wire = _Wire(bodies)
    try:
        yield wire
    finally:
        await wire.client.aclose()


def _host_tools() -> tuple[ToolSet, list[str]]:
    """A real `ToolSet`, because HOST_TOOLS claims the host's validation path, not a stub."""
    executed: list[str] = []

    @tool(side_effecting=False)
    def lookup_fact(subject: str) -> str:
        """Look up a fact only the host process can read."""
        executed.append(subject)
        return f"the {subject} fact"

    return ToolSet(ToolRegistry.from_tools([lookup_fact])), executed


def _status(report: Any, requirement: ExecutionRequirement) -> Any:
    return next(fact for fact in report.assessments if fact.requirement == requirement)


# ── the matrix itself ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _ALL_MODES)
async def test_every_mode_answers_every_requirement_and_names_its_transport(mode: Mode) -> None:
    """An assessment is how a host CHOOSES a binding, so a mode that answers only the subset it was
    asked about is useless for discovery, and a mode that answers nothing at all while reporting
    `compatible=True` is actively misleading — that was the SDK behaviour: zero assessments,
    `backend="unspecified"`, `backend_version="unknown"`, read by a caller as a clean bill of
    health. Identity matters for the same reason a CLI binary version does: an incompatibility
    report that cannot say WHICH transport produced it cannot be acted on. And none of it may cost
    a credential, because the whole point of a pre-dispatch assessment is that it is free and
    secret-free — so the resolver here raises if it is touched, and the matrix is scanned for the
    planted secret."""
    adapters: dict[Any, Any] = {} if mode.is_api else {mode: CliAdapter(spawn=lambda *a, **k: None)}
    async with CallRuntime(adapters=adapters) as runtime:
        report = await runtime.evaluate_execution(Binding(mode, "test", _forbidden_credential()))

    assert {fact.requirement for fact in report.assessments} == set(ExecutionRequirement)
    assert len(report.assessments) == len(ExecutionRequirement), "one verdict per requirement, no duplicates"
    assert report.mode == str(mode)
    assert report.backend not in ("", "unspecified"), "an unidentified transport cannot be acted on"
    assert report.backend_version not in ("", "unknown")
    for fact in report.assessments:
        assert re.fullmatch(r"[a-z0-9_]+", fact.code), f"{fact.requirement} needs a stable machine-readable code"
        assert len(fact.explanation.strip()) > 20, f"{fact.requirement} needs a readable explanation"
        assert _SECRET not in fact.explanation
    assert _SECRET not in repr(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_the_sdk_matrix_marks_the_protocol_supported_and_the_boundaries_not(mode: Mode) -> None:
    """The specific verdicts, pinned one by one, because every one of them is a security statement
    that a later refactor could flip without a visible failure.

    `host_tools`/`schema`/`usage` are SUPPORTED: the adapter declares the host's tools on the wire,
    parses the provider's calls back, puts the compiled output schema in the request, and reports
    token usage from the stream. `native_tools_unavailable` is SUPPORTED because an SDK request
    carries no provider-native or provider-hosted tool at all — this is the one mode where strict
    native-tool absence is honestly satisfiable. Everything that claims to CONTAIN what a tool may
    do is UNSUPPORTED: those callbacks run in this process with the host's own authority.
    `cancellation` is UNVERIFIED rather than SUPPORTED, and `hard_spending_cap` stays UNSUPPORTED
    even though `observed_spending_threshold` is implemented — collapsing those two is how a
    "cap" that is really a post-hoc threshold gets sold as a hard limit."""
    async with CallRuntime() as runtime:
        report = await runtime.evaluate_execution(Binding(mode, "test"))

    expected = {
        ExecutionRequirement.HOST_TOOLS: CapabilityStatus.SUPPORTED,
        ExecutionRequirement.SCHEMA: CapabilityStatus.SUPPORTED,
        ExecutionRequirement.USAGE: CapabilityStatus.SUPPORTED,
        ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE: CapabilityStatus.SUPPORTED,
        ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD: CapabilityStatus.SUPPORTED,
        ExecutionRequirement.CANCELLATION: CapabilityStatus.UNVERIFIED,
        ExecutionRequirement.MCP_ALLOWLIST: CapabilityStatus.UNVERIFIED,
        ExecutionRequirement.FILESYSTEM_READ: CapabilityStatus.UNSUPPORTED,
        ExecutionRequirement.FILESYSTEM_WRITE: CapabilityStatus.UNSUPPORTED,
        ExecutionRequirement.NETWORK: CapabilityStatus.UNSUPPORTED,
        ExecutionRequirement.PROCESSES: CapabilityStatus.UNSUPPORTED,
        ExecutionRequirement.CREDENTIALS: CapabilityStatus.UNSUPPORTED,
        ExecutionRequirement.CONFIGURATION: CapabilityStatus.UNSUPPORTED,
        ExecutionRequirement.HARD_SPENDING_CAP: CapabilityStatus.UNSUPPORTED,
    }
    assert {fact.requirement: fact.status for fact in report.assessments} == expected
    assert report.compatible, "a request that states no requirement has nothing unmet"
    assert "host authority" in _status(report, ExecutionRequirement.CREDENTIALS).explanation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_a_default_assessment_is_evidence_rather_than_an_empty_pass(mode: Mode) -> None:
    """`compatible=True` with an empty matrix was the original defect, and it is worth its own test
    because it is the one failure mode that looks like success from every angle: the property is
    True, no exception is raised, and nothing in the report contradicts it. A host rendering this
    report shows the operator a verified-looking summary built from zero facts."""
    async with CallRuntime() as runtime:
        report = await runtime.evaluate_execution(Binding(mode, "test"))

    assert report.compatible
    assert report.incompatibilities == ()
    assert len(report.assessments) == len(ExecutionRequirement) > 0
    supported = {fact.requirement for fact in report.assessments if fact.status == CapabilityStatus.SUPPORTED}
    assert supported, "a pass with no SUPPORTED verdict asserts nothing"
    assert supported != set(ExecutionRequirement), "and a pass on everything would be a lie"


# ── the regression that matters: a host-tool run must be possible ───────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_a_host_needing_only_tools_schema_and_usage_can_run_on_an_sdk_mode(mode: Mode) -> None:
    """THE REGRESSION. With `host_tools`/`schema`/`usage` reported UNSUPPORTED, this exact call —
    a host that states the protocol it depends on and then runs its own function — failed with
    `execution_incompatible` before a socket was opened, on the two modes where the host has the
    MOST control over tool execution. The assessment is only worth having if it governs dispatch,
    so this asserts through `CallRuntime.run`: the tool declaration reaches the wire, the host
    function actually executes, the compiled schema is in the request, usage is priced, and the
    strictest native-tool policy does not block it (SDK requests carry no native tools, so DENY
    has nothing to refuse)."""
    tools, executed = _host_tools()
    async with _wire(_tool_call_frames(mode), _final_frames(mode, '{"answer": "ok"}')) as wire:
        async with CallRuntime(
            adapters={mode: ApiAdapter(client=wire.client)},
            pricing=_PRICES,
            native_tool_policy=NativeToolPolicy.DENY,
        ) as runtime:
            answer = await runtime.run(
                Binding(mode, "test", _credential(mode)),
                "look the ledger fact up",
                tools,
                output=_ANSWER,
                execution=_HOST_PROTOCOL,
                limits=CallLimits(max_turns=4),
                call_id="sdk-host-protocol",
            )

        assert answer.failure is None, answer.failure
        assert answer.outcome == "answered"
        assert answer.value == {"answer": "ok"}
        assert executed == ["ledger"], "the host function itself must have run"
        assert answer.attempts == 2 and len(wire.payloads) == 2
        assert answer.usage.output > 0 and answer.cost_usd is not None
        declared = wire.payloads[0]["tools"]
        assert [entry["name"] for entry in declared] == ["lookup_fact"]
        assert json.dumps(wire.payloads[0]).count("answer") >= 1, "the compiled output schema must be on the wire"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_requirements_stay_attached_to_every_turn_of_an_sdk_invocation(mode: Mode) -> None:
    """`ExecutionRequirements` used to be checked at the runtime's door and then dropped: the SDK
    adapter had no `execution` parameter at all. A gate-only check is satisfied by the first turn
    and silently absent from every continuation, retry and correction turn after it — which is
    precisely the window in which the requirement matters, because that is where tool calls and
    repaired prompts happen. Recording the kwarg on each turn is the only way to see this; a
    passing answer looks identical either way."""
    seen: list[Any] = []

    class Recording(ApiAdapter):
        async def turn(self, binding: Any, prompt: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get("execution"))
            return await super().turn(binding, prompt, **kwargs)

    tools, _ = _host_tools()
    async with _wire(_tool_call_frames(mode), _final_frames(mode, "done")) as wire:
        async with CallRuntime(adapters={mode: Recording(client=wire.client)}, pricing=_PRICES) as runtime:
            answer = await runtime.run(
                Binding(mode, "test", _credential(mode)),
                "look the ledger fact up",
                tools,
                execution=_HOST_PROTOCOL,
                limits=CallLimits(max_turns=4),
                call_id="sdk-binding-requirements",
            )

    assert answer.outcome == "answered"
    assert len(seen) == 2, "a tool round trip is two turns"
    assert seen == [_HOST_PROTOCOL, _HOST_PROTOCOL]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_an_observed_threshold_runs_and_bites_while_a_hard_cap_is_refused(mode: Mode) -> None:
    """`observed_spending_threshold` and `hard_spending_cap` are separate requirements for a reason,
    and a matrix that answered them together would be wrong in one direction or the other. The
    threshold is real: the runtime prices the usage it observed and stops the invocation before the
    next dispatch, which this asserts by letting a first turn ask for a tool and showing the second
    request never happens. A hard cap is not real: nothing in this transport refuses a request that
    would exceed a dollar figure mid-stream, so a host that genuinely needs one must be refused
    rather than handed a threshold wearing a cap's name."""
    tools, _ = _host_tools()
    threshold = ExecutionRequirements(required=frozenset({ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD}))
    async with _wire(_tool_call_frames(mode)) as wire:
        async with CallRuntime(adapters={mode: ApiAdapter(client=wire.client)}, pricing=_PRICES) as runtime:
            observed = await runtime.run(
                Binding(mode, "test", _credential(mode)),
                "look the ledger fact up",
                tools,
                execution=threshold,
                limits=CallLimits(max_cost_usd=Decimal("0.000001"), max_turns=4),
                call_id="sdk-observed-threshold",
            )
            assert observed.outcome == "limit_reached", observed.failure
            assert len(wire.payloads) == 1, "the threshold must stop the invocation before the next dispatch"

            capped = await runtime.run(
                Binding(mode, "test", _forbidden_credential()),
                "look the ledger fact up",
                tools,
                execution=ExecutionRequirements(required=frozenset({ExecutionRequirement.HARD_SPENDING_CAP})),
                limits=CallLimits(max_cost_usd=Decimal("0.000001"), max_turns=4),
                call_id="sdk-hard-cap",
            )
            assert len(wire.payloads) == 1, "the refused call must not add a request"

    assert capped.failure is not None
    assert capped.failure.classification == FailureClass.EXECUTION_INCOMPATIBLE
    assert [item["code"] for item in capped.failure.mode_detail["incompatibilities"]] == [
        "hard_spending_cap_unavailable"
    ]


# ── what must still be refused ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_contained_execution_is_refused_on_an_sdk_mode_with_no_provider_attempt(mode: Mode) -> None:
    """Widening the SDK matrix must not widen it into the boundaries. A contained request asks for
    filesystem roots, egress limits, process separation and credential secrecy; on the SDK path the
    model's tools ARE host callbacks running in this process with this process's authority, and the
    provider key is held in the same memory. Dispatching anyway would answer the question while
    ignoring the sandbox the caller scoped it to, so the refusal has to be local: zero attempts,
    zero requests, and no credential resolved."""
    async with _wire() as wire:
        async with CallRuntime(adapters={mode: ApiAdapter(client=wire.client)}, pricing=_PRICES) as runtime:
            answer = await runtime.run(
                Binding(mode, "test", _forbidden_credential()),
                "task",
                execution=ExecutionRequirements(native_execution=NativeExecution.CONTAINED),
                limits=CallLimits(max_turns=1),
                call_id="sdk-contained",
            )

        assert wire.payloads == [], "a refused call must not spend a round trip"

    assert answer.failure is not None
    assert answer.failure.classification == FailureClass.EXECUTION_INCOMPATIBLE
    assert answer.failure.mode_detail["dispatch_state"] == "not_started"
    assert answer.attempts == 0 and answer.cost_usd is None and not answer.usage_complete
    codes = {item["code"] for item in answer.failure.mode_detail["incompatibilities"]}
    assert "credential_boundary_unavailable" in codes
    assert answer.failure.mode_detail["backend"] not in ("", "unspecified")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("filesystem_read", "unsupported"),
        ("filesystem_write", "unsupported"),
        ("network", "unsupported"),
        ("processes", "unsupported"),
        ("credentials", "unsupported"),
        ("configuration", "unsupported"),
        ("hard_spending_cap", "unsupported"),
        ("mcp_allowlist", "unverified"),
        ("cancellation", "unverified"),
    ],
)
async def test_every_unproved_sdk_boundary_independently_refuses(mode: Mode, boundary: str, expected: str) -> None:
    """Requirements are independent: satisfying the host-tool protocol says nothing about egress,
    and a matrix that refused them as a BLOCK would hide which one is missing — and would make the
    whole set pass the moment the block was broken up. One requirement in, exactly one
    incompatibility out, with its status preserved. The two UNVERIFIED entries are here
    deliberately: UNVERIFIED is still a refusal when required, and it must not quietly satisfy a
    requirement just because it is not UNSUPPORTED."""
    report = await ApiAdapter().evaluate_execution(
        Binding(mode, "test"),
        execution=ExecutionRequirements(required=frozenset({ExecutionRequirement(boundary)})),
    )

    assert not report.compatible
    assert len(report.incompatibilities) == 1
    assert report.incompatibilities[0].requirement == boundary
    assert report.incompatibilities[0].status == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_a_cancelled_token_alone_does_not_stop_an_sdk_request(mode: Mode) -> None:
    """WHY `cancellation` is UNVERIFIED and not SUPPORTED, demonstrated rather than asserted. The
    adapter accepts a `cancel` token and never reads it: cancellation on this path works only
    because the runtime cancels the awaiting task, which closes the stream locally and proves
    nothing about the provider stopping — or stopping billing for — work already in flight. A
    pre-cancelled token still dispatches, which is exactly the gap a host requiring a cancellation
    boundary must be refused on. If someone wires the token into the adapter and this starts
    failing, the verdict should be revisited, not the test deleted."""
    token = CancellationToken()
    token.cancel()
    async with _wire(_final_frames(mode, "answered anyway")) as wire:
        turn = await ApiAdapter(client=wire.client).turn(
            Binding(mode, "test", _credential(mode)),
            inline_prompt("q"),
            settings=ModelSettings(),
            limits=CallLimits(max_output_tokens=64),
            schema=None,
            tools=None,
            session=None,
            cancel=token,
            emit=lambda _event: None,
        )
        assert len(wire.payloads) == 1, "the cooperative token did not prevent the request"

    assert turn.failure is None

    report = await ApiAdapter().evaluate_execution(Binding(mode, "test"))
    cancellation = _status(report, ExecutionRequirement.CANCELLATION)
    assert cancellation.status == CapabilityStatus.UNVERIFIED
    assert "in flight" in cancellation.explanation or "in-flight" in cancellation.explanation


# ── nothing may promote a verdict ───────────────────────────────────────────


@pytest.mark.parametrize("mode", _SDK_MODES)
def test_a_self_declared_transport_guarantee_cannot_reach_supported(mode: Mode) -> None:
    """The invariant the CLI side already holds: a report's `guarantees` may only DOWNGRADE a
    verdict to UNVERIFIED, never raise one to SUPPORTED. Otherwise containment becomes
    self-attesting — whoever constructs the transport report decides whether the boundary exists,
    which is the one authority this design refuses to delegate. A declared guarantee for a
    boundary agentkit cannot verify changes the explanation, not the permission."""
    honest = evaluate_api_execution(mode, TransportReport("https-sse", "1"), ExecutionRequirements(),
                                    NativeToolPolicy.DENY)
    declared = evaluate_api_execution(
        mode,
        TransportReport("vendor-sdk", "9", frozenset(ExecutionRequirement)),
        ExecutionRequirements(native_execution=NativeExecution.CONTAINED),
        NativeToolPolicy.WARN,
    )

    def supported(report: Any) -> set[Any]:
        return {fact.requirement for fact in report.assessments if fact.status == CapabilityStatus.SUPPORTED}

    assert supported(declared) <= supported(honest), "a declaration must not add a single SUPPORTED verdict"
    assert not declared.compatible
    assert all(fact.status != CapabilityStatus.SUPPORTED for fact in declared.incompatibilities)
    assert any(fact.status == CapabilityStatus.UNVERIFIED for fact in declared.incompatibilities)


@pytest.mark.parametrize("mode", _SDK_MODES)
def test_transport_identity_metadata_cannot_change_a_verdict(mode: Mode) -> None:
    """Version strings are diagnostics. The CLI side learned this explicitly — a binary version is
    "metadata, not proof of isolation" — and the same trap exists here the moment identity is
    threaded into the report: an assessment keyed on a version string becomes a capability that
    appears and disappears with a dependency bump, silently, in either direction."""
    first = evaluate_api_execution(mode, TransportReport("https-sse", "1/httpx-0.0.1"),
                                   ExecutionRequirements(), NativeToolPolicy.WARN)
    second = evaluate_api_execution(mode, TransportReport("other-transport", "9/experimental"),
                                    ExecutionRequirements(), NativeToolPolicy.WARN)

    assert first.backend_version != second.backend_version
    assert first.assessments == second.assessments


def test_the_two_sdk_modes_are_assessed_identically_but_identified_apart() -> None:
    """Neither SDK mode has a native tool surface, a sandbox or a hard cap, so a per-mode
    DIFFERENCE in status would mean one of the two verdicts is guesswork rather than a reading of
    the shared code path. The wire contract they target is not shared, though, and an
    incompatibility report that cannot name which interface produced it cannot be acted on."""
    anthropic = evaluate_api_execution(Mode.ANTHROPIC_API, TransportReport("https-sse", "1"),
                                       ExecutionRequirements(), NativeToolPolicy.DENY)
    openai = evaluate_api_execution(Mode.OPENAI_API, TransportReport("https-sse", "1"),
                                    ExecutionRequirements(), NativeToolPolicy.DENY)

    assert {fact.requirement: fact.status for fact in anthropic.assessments} == {
        fact.requirement: fact.status for fact in openai.assessments
    }
    assert anthropic.binary_version != openai.binary_version
    assert anthropic.binary_version and openai.binary_version


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_a_host_payload_rewriter_withdraws_the_native_tool_absence_claim(mode: Mode) -> None:
    """`native_tools_unavailable` is SUPPORTED on the SDK path because the adapter declares nothing
    but host functions — no provider-hosted or server-side tool can be requested through any public
    parameter. `request_hook` is the one exception: it replaces the request body on its way out, so
    with a hook installed agentkit can no longer READ the payload it claimed. That is missing
    evidence, not an impossibility, so the verdict degrades to UNVERIFIED — and under a DENY policy
    UNVERIFIED still refuses, which is what stops a rewritten payload from quietly reintroducing a
    native tool under the strictest policy."""
    report = await ApiAdapter(request_hook=lambda payload: payload).evaluate_execution(
        Binding(mode, "test"), native_tool_policy=NativeToolPolicy.DENY
    )

    native = _status(report, ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE)
    assert native.status == CapabilityStatus.UNVERIFIED
    assert not report.compatible, "DENY requires native-tool absence, and UNVERIFIED is a refusal"
    assert [fact.requirement for fact in report.incompatibilities] == [
        ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE
    ]

    without = await ApiAdapter().evaluate_execution(Binding(mode, "test"), native_tool_policy=NativeToolPolicy.DENY)
    assert _status(without, ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE).status == CapabilityStatus.SUPPORTED
    assert without.compatible


# ── one matrix for assessment and dispatch ──────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _SDK_MODES)
async def test_the_adapter_refuses_with_the_same_matrix_that_assessed_the_call(mode: Mode) -> None:
    """Two code paths answering the same question is how a gate gets bypassed. `ApiAdapter.turn` is
    reachable directly — hosts hold adapters, and the runtime's own continuation turns call it
    repeatedly — so the requirements have to be enforced there too, and by the SAME evaluation that
    `evaluate_execution` reports. Byte-comparing the structured incompatibilities is what makes a
    second, divergent copy of the rules impossible to add quietly."""
    requirements = ExecutionRequirements(required=frozenset({ExecutionRequirement.NETWORK}))
    expected = await ApiAdapter().evaluate_execution(Binding(mode, "test"), execution=requirements)

    async with _wire() as wire:
        turn = await ApiAdapter(client=wire.client).turn(
            Binding(mode, "test", _forbidden_credential()),
            inline_prompt("q"),
            settings=ModelSettings(),
            limits=CallLimits(max_output_tokens=64),
            schema=None,
            tools=None,
            session=None,
            cancel=None,
            emit=lambda _event: None,
            execution=requirements,
        )
        assert wire.payloads == []

    assert turn.failure is not None
    assert turn.failure.classification == FailureClass.EXECUTION_INCOMPATIBLE
    assert turn.failure.mode_detail["dispatch_state"] == "not_started"
    assert turn.failure.mode_detail["incompatibilities"] == tuple(
        asdict(item) for item in expected.incompatibilities
    )
    assert not turn.usage_complete
