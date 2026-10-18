"""The central claim of the abstraction, proven against REAL providers and REAL binaries.

Why this file exists
--------------------
``test_live_smoke.py`` proves the plumbing is connected: a real completion, a
real CLI turn, a real schema fill, a real permission prompt. None of that
touches the claim the whole library is for:

    a host supplies a task, a structured answer shape, a credential resolver
    and a granted ToolSet, then picks any of the four modes — ``openai_api``,
    ``anthropic_api``, ``claude_code``, ``codex`` — without writing a second
    agent loop or a second copy of its tools. The SDK modes call the host's
    functions directly; the CLI modes reach the SAME functions through
    agentkit's CLI/MCP transport.

That claim is about an EQUIVALENCE between four very different transports, and
an equivalence is exactly what a mock cannot witness: every double in this
repository is built by the same author as the assertion, so "four modes behave
alike" degrades into "four fixtures were typed alike". Here the far side is a
real provider deciding, on its own, whether to call a function it was merely
told about — over HTTP tool-calling in two modes and over a spawned CLI
speaking MCP to an in-process server in the other two.

What each test is built to catch
--------------------------------
- a granted host tool that is ADVERTISED but never actually dispatched. The
  lookup returns a token generated in this process microseconds earlier, so a
  model that answers without calling the tool cannot be right by luck. The
  host-side flag and the token in the typed answer are checked together: one
  without the other is a different (and still broken) system.
- a typed answer that only survives on some transports. One dataclass, four
  modes, identical assertions.
- a ledger reporting ``$0.00`` next to a correct answer, via the
  ``_assert_really_billed`` discipline ``test_live_smoke.py`` introduces. This
  file extends it to the CLI modes, which the smoke file does not cover.
- a refusal with no teeth. The clean/planted twin pair: one granted tool that
  must run, one tool the host marked ``requires_approval`` that must NOT, in
  the same live turn, with the planted body's side effect checked on the
  filesystem rather than inferred.
- a tool-call id that does not survive the bridge. A recent fix made every
  provider fall back to ``call_<position>`` when an id is missing; before it,
  id-less calls collided and one tool's result silently overwrote another's.
  The ids observed on the host's own event stream must be present and distinct.

Running it
----------
Opt-in twice over, because it spends real money::

    set -a && source .env.eval && set +a && AGENTKIT_LIVE=1 \\
        uv run pytest tests/source/live/test_live_execution_contract.py -q -p no:randomly

Without ``AGENTKIT_LIVE`` everything here skips, so CI stays key-free, and each
mode skips individually when its own credential or binary is absent — a host
with one provider still gets the half it can check.

Models are the cheapest PRICED ones, for the same reason the smoke file picks
them: ``unpriced_policy`` defaults to REFUSE, so an unpriced model fails before
the network, and asserting on cost is a third of the point of being here.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.constants import CostSource, EventKind
from agentkit.models.contracts.types import Binding, CallEvent, CallLimits, CredentialRef
from agentkit.models.providers.cli.calls import CliAdapter
from agentkit.models.providers.frontiers.calls import ApiAdapter
from agentkit.tools import ToolRegistry, tool
from agentkit.tools.execution.policy import ToolSet

# Live I/O is allowed to warn, exactly as in ``test_live_smoke.py``: the suite
# turns warnings into errors, which is right for our own code and wrong for a
# third-party client talking to a real endpoint.
#
# The two ResourceWarning filters are the same shutdown-hygiene issue the smoke
# file documents on its approval test: the MCP streamable-HTTP session manager
# leaves anyio memory streams for the collector, anyio's ``__del__`` warns, and
# pytest escalates that GC-time warning into a failure of a test whose own
# assertions all passed. Every CLI-mode test here serves host tools over that
# transport, so the filter has to be module-wide rather than per-test. It stays
# narrow to those two classes so a leak in OUR code still fails.
pytestmark = [
    pytest.mark.live,
    pytest.mark.filterwarnings("default"),
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

_OPTED_IN = os.environ.get("AGENTKIT_LIVE") == "1"
_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_CLAUDE_CLI = shutil.which("claude") is not None
_CODEX_CLI = shutil.which("codex") is not None

# The cheapest models the bundled table prices, same choice as the smoke file.
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-4o-mini"

needs_optin = pytest.mark.skipif(
    not _OPTED_IN, reason="live tests spend real money; set AGENTKIT_LIVE=1 to opt in"
)
needs_anthropic = pytest.mark.skipif(not _ANTHROPIC, reason="needs ANTHROPIC_API_KEY")
needs_openai = pytest.mark.skipif(not _OPENAI, reason="needs OPENAI_API_KEY")
needs_claude_cli = pytest.mark.skipif(not _CLAUDE_CLI, reason="needs the `claude` binary on PATH")
needs_codex_cli = pytest.mark.skipif(not _CODEX_CLI, reason="needs the `codex` binary on PATH")

# One list, four modes, each carrying the skips its own transport needs. The
# parameter is a `Mode` VALUE and never a type annotation: PEP 563 would turn
# an annotation into an unresolvable string and the test would pass for the
# wrong reason.
ALL_MODES = [
    pytest.param(Mode.OPENAI_API, id="openai_api", marks=[needs_openai]),
    pytest.param(Mode.ANTHROPIC_API, id="anthropic_api", marks=[needs_anthropic]),
    pytest.param(Mode.CLAUDE_CODE, id="claude_code", marks=[needs_claude_cli, needs_anthropic]),
    pytest.param(Mode.CODEX, id="codex", marks=[needs_codex_cli, needs_openai]),
]

_MODEL = {
    Mode.OPENAI_API: OPENAI_MODEL,
    Mode.ANTHROPIC_API: ANTHROPIC_MODEL,
    Mode.CLAUDE_CODE: ANTHROPIC_MODEL,
    Mode.CODEX: OPENAI_MODEL,
}
# The CLI modes need a credential of their own: the adapter runs the binary
# under an ISOLATED HOME, so a developer's interactive `claude /login` is
# invisible to it by design. See the long note in ``test_live_smoke.py``.
_CREDENTIAL = {
    Mode.OPENAI_API: "OPENAI_API_KEY",
    Mode.ANTHROPIC_API: "ANTHROPIC_API_KEY",
    Mode.CLAUDE_CODE: "ANTHROPIC_API_KEY",
    Mode.CODEX: "OPENAI_API_KEY",
}


def _env(name: str) -> CredentialRef:
    """Read the key from the environment at call time and never anywhere else.

    The value is not logged, not echoed into argv and not stored on the
    Binding — ``CredentialRef`` exists so the credential is resolved at the
    moment of use by the adapter that needs it.
    """
    return CredentialRef(lambda: {name: os.environ[name]})


def _binding(mode: Mode) -> Binding:
    return Binding(mode, _MODEL[mode], _env(_CREDENTIAL[mode]))


def _runtime(mode: Mode, working_dir: Path) -> CallRuntime:
    """The SAME runtime for all four modes; only the adapter differs.

    That is the point under test, so the asymmetry is confined to this one
    expression: the host picks `ApiAdapter` or `CliAdapter` and nothing else
    about its call changes.

    ``result_masker`` is wired so TOOL_RESULT events carry a readable preview.
    It is the only way to observe, from outside, WHICH layer refused a tool —
    and distinguishing the host's own gate from the provider's is the whole
    subject of the planted-twin test.
    """
    adapter: Any = ApiAdapter() if mode.is_api else CliAdapter(working_dir=working_dir)
    return CallRuntime(
        pricing=PriceTable.default(),
        adapters={mode: adapter},
        result_masker=lambda name, text: text,
    )


def _limits(mode: Mode) -> CallLimits:
    """A generous wall clock for the CLI modes, which spawn a process per turn."""
    return CallLimits(max_turns=8) if mode.is_api else CallLimits(max_turns=8, wall_clock_s=180)


def _recorder(log: list[CallEvent]) -> Any:
    async def record(event: CallEvent) -> None:
        log.append(event)

    return record


def _of_kind(log: list[CallEvent], kind: EventKind) -> list[CallEvent]:
    return [event for event in log if event.kind == kind]


def _name(event: CallEvent) -> str:
    return str(event.fields.get("name", ""))


def _names(log: list[CallEvent], kind: EventKind, *, cap: int = 8) -> str:
    """The tool names on ``kind``, capped — a looping CLI produces hundreds.

    Bounded because an assertion message that dumps 170 repetitions of one
    name buries the finding it is supposed to deliver.
    """
    names = [_name(event) for event in _of_kind(log, kind)]
    head = ", ".join(repr(name) for name in names[:cap])
    return f"[{head}] ({len(names)} in total)" if len(names) > cap else f"[{head}]"


def _assert_really_billed(answer: Any) -> None:
    """The receipt has to come from the provider, not from our own arithmetic.

    Lifted verbatim in spirit from ``test_live_smoke.py`` because it is the one
    assertion no mocked test can make: a ledger that reports 0 tokens and
    ``$0.00`` while the answer is correct looks like success everywhere except
    the invoice, and a fake supplies whatever usage the fixture author typed.

    Applied here to the CLI modes too, which the smoke file leaves uncovered —
    a CLI turn is billed through a different parser and a different usage
    payload than an API turn, so it is a second place the ledger can silently
    read zero.
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
    # UNPRICED is the one source that must never appear: it is the silent
    # $0.00 ledger this assertion exists to catch.
    assert answer.cost_source in (
        CostSource.REPORTED,
        CostSource.PRICE_TABLE,
        CostSource.HOST_PRICING,
        CostSource.MIXED,
    ), answer.cost_source


def _assert_call_ids_are_usable(log: list[CallEvent], *, at_least: int) -> list[str]:
    """Every dispatched call must carry a non-empty id, and no two may share one.

    The id is what pairs a RESULT back to its call: ``tool_call_id`` on an
    OpenAI tool message, ``tool_use_id`` on an Anthropic tool_result block, the
    per-item dedupe key in the codex driver. Two calls sharing an id means one
    tool's answer silently overwrites the other's and the model is handed a
    result it never asked for, attributed to a call it never made. Nothing
    raises; the run just becomes wrong.

    ``at_least`` keeps this from passing vacuously. A turn with zero tool calls
    satisfies "all ids are unique" perfectly and proves nothing, which is the
    shape this check would rot into first.
    """
    calls = _of_kind(log, EventKind.TOOL_CALL)
    assert len(calls) >= at_least, (
        f"expected at least {at_least} dispatched tool calls, saw "
        f"{_names(log, EventKind.TOOL_CALL)}"
    )
    ids = [str(event.fields.get("tool_call_id") or "") for event in calls]
    assert all(ids), (
        "a dispatched tool call carried no id; every transport keys its result "
        f"on that id: {list(zip([_name(event) for event in calls], ids, strict=True))}"
    )
    assert len(set(ids)) == len(ids), (
        "two tool calls share one id, so one result can overwrite the other: "
        f"{list(zip([_name(event) for event in calls], ids, strict=True))}"
    )
    return ids


# ── the one answer shape every mode has to return ────────────────────────────


# Declared at module level so ``typing.get_type_hints`` can resolve the field
# annotations: this module uses ``from __future__ import annotations``, and the
# dataclass output adapter resolves hints against the defining module's
# globals. A dataclass nested inside a test function would not resolve.
@dataclasses.dataclass(frozen=True)
class DispatchCodes:
    """Two codes that exist only inside a host function in this process."""

    alpha_code: str
    beta_code: str


@dataclasses.dataclass(frozen=True)
class GateReport:
    """One code the host granted, plus the model's account of the refused tool."""

    dispatch_code: str
    commit_blocked: bool


# ── 1, 2, 3, 5: the granted tool really runs, typed, billed, with usable ids ─


@needs_optin
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_every_mode_dispatches_the_granted_host_function_and_returns_its_secret_typed(
    mode: Mode, tmp_path: Path
) -> None:
    """A host writes ONE function, ONE answer shape and ONE call; four transports honour it.

    This is the acceptance test for the library's reason to exist. The host
    hands over a task, a dataclass, a credential resolver and a granted
    ToolSet, and swaps ``mode`` — nothing else in the call changes and no
    second agent loop is written. The SDK modes invoke ``read_dispatch_code``
    in this process over provider tool-calling; the CLI modes spawn a real
    binary that reaches the SAME function object through agentkit's MCP
    transport.

    Why the token is random: a model cannot be trusted to admit it did not call
    a tool. Asked for "the population of Tokyo" it answers correctly without
    any dispatch, and the test passes while the entire tool path is dead. The
    codes here are generated microseconds before the call and exist in exactly
    one place — the closure below — so the only way the answer can be right is
    the function having actually run. The host-side flag and the token in the
    typed answer are asserted TOGETHER: a flag with no token means the result
    never came back, and a token with no flag is impossible unless something
    leaked.

    Two records, not one, because the tool-call id guarantee needs two calls to
    the SAME tool. The lookup is declared side-effect-free, so the runtime is
    free to dispatch both concurrently, which is the case where a shared id
    does real damage.

    What breaks without this test: a mode whose tool loop is advertised and
    never dispatched, a dataclass that round-trips on two transports out of
    four, and a CLI turn billed at $0.00 — each of which the offline suite
    reports as green.
    """
    if not mode.is_api:
        pytest.importorskip("mcp", reason="serving host tools to a CLI needs the `mcp` extra")

    codes = {"alpha": uuid.uuid4().hex[:12], "beta": uuid.uuid4().hex[:12]}
    looked_up: list[str] = []

    @tool(side_effecting=False)
    def read_dispatch_code(record: str) -> dict:
        """Return the dispatch code the host stored for a record.

        Valid records are 'alpha' and 'beta'. The codes are held only by the
        host and cannot be derived, guessed or recalled from training data.
        """
        if record not in codes:
            raise ValueError(f"unknown record {record!r}; valid records are {sorted(codes)}")
        looked_up.append(record)
        return {"record": record, "dispatch_code": codes[record]}

    tools = ToolSet(ToolRegistry.from_tools([read_dispatch_code]), timeout_s=30)
    events: list[CallEvent] = []

    async with _runtime(mode, tmp_path) as runtime:
        answer = await runtime.run(
            _binding(mode),
            "Use the host tool whose name ends in read_dispatch_code to look up the "
            "dispatch code for record 'alpha', then again for record 'beta'. The codes "
            "are known only to that tool, so you must call it for each record. Then "
            "return both codes exactly as the tool reported them, with no extra text.",
            tools,
            output=DispatchCodes,
            limits=_limits(mode),
            events=_recorder(events),
            call_id=f"live-contract-dispatch-{mode.value}",
        )

    assert answer.outcome == "answered", answer.failure
    # The host's own evidence that its function body executed, independent of
    # anything the model said about it.
    assert sorted(set(looked_up)) == ["alpha", "beta"], (
        "the host function did not run for both records; the model answered "
        f"from somewhere else. Calls seen: {looked_up}. Answer: {answer.value!r}"
    )
    # The same typed shape from all four transports.
    assert isinstance(answer.value, DispatchCodes), answer.value
    assert answer.value.alpha_code.strip() == codes["alpha"], (
        "the granted tool ran but its result never reached the answer: "
        f"{answer.value.alpha_code!r}"
    )
    assert answer.value.beta_code.strip() == codes["beta"], (
        f"the second lookup's result never reached the answer: {answer.value.beta_code!r}"
    )
    _assert_call_ids_are_usable(events, at_least=2)
    _assert_really_billed(answer)


# ── 4: the clean / planted twin, and a refusal with teeth ────────────────────


@needs_optin
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_every_mode_refuses_the_unapproved_and_ungranted_tool_while_the_clean_twin_runs(
    mode: Mode, tmp_path: Path
) -> None:
    """A refusal is only worth recording if the body never ran, and the twin proves the gate is not just "off".

    Two tools, identical in every respect the model can see except the one the
    host declared: ``read_dispatch_code`` is granted outright, and
    ``commit_dispatch`` carries ``requires_approval=True``. Both are in the
    same ToolSet, both are advertised, and the model is told to call both.
    Only the first may execute.

    The twin is what makes the result meaningful. A test that only plants the
    refused tool passes identically when tool dispatch is broken outright —
    nothing ran, so nothing ran that shouldn't have. Pairing it with a tool
    that MUST run separates "the gate held" from "the whole path is dead",
    which is the failure mode a security assertion rots into.

    The third planted item is ``purge_dispatch_log``, a name the host never
    granted. The guarantee asserted for it is the one that actually matters and
    the one a host relies on: the granted set BOUNDS what can execute, so no
    body outside it can run and no call outside it can return a success. The
    model usually cannot even emit the name — the provider is only told about
    the granted schemas, and on the CLI modes the MCP catalogue is the only
    thing addressable — and that containment is itself part of the property.

    Effect is checked on the FILESYSTEM, not inferred. ``commit_dispatch``
    would write a receipt file as its first statement, so if the body is
    entered at all the file exists even though the call later fails. A gate
    that records a denial while the tool runs anyway is worse than no gate.

    On the SDK modes the error text is asserted too: there the host's ToolSet
    is the ONLY gate in the path, so "requires host approval" in the result is
    direct evidence that agentkit's own check — not the provider's politeness,
    not a CLI permission layer — is what stopped it, and that it stopped it
    before the body, which is the ordering the whole approval contract rests
    on.

    Deliberately NOT asserted: that the model reaches a final answer. The
    subject here is the gate, and whether a cheap model gives up gracefully on
    a tool it is not allowed to call is a fact about the model. Measured on
    ``gpt-4o-mini``: told explicitly to call ``commit_dispatch`` exactly once
    and not to retry, it retried the refused call on every remaining turn until
    the turn cap. The gate held each time, which is what this test is for.
    Requiring "answered" here would import that model-compliance question into
    a security test and make it flake; the typed-answer guarantee is proven for
    all four modes by the dispatch test above. ``bodies_run`` is the host's own
    evidence and does not depend on the model finishing.
    """
    if not mode.is_api:
        pytest.importorskip("mcp", reason="serving host tools to a CLI needs the `mcp` extra")

    code = uuid.uuid4().hex[:12]
    receipt = tmp_path / f"committed-{uuid.uuid4().hex[:8]}.txt"
    bodies_run: list[str] = []

    @tool(side_effecting=False)
    def read_dispatch_code(record: str) -> dict:
        """Return the dispatch code the host stored for a record.

        The code is held only by the host and cannot be guessed.
        """
        bodies_run.append("read_dispatch_code")
        return {"record": record, "dispatch_code": code}

    @tool(side_effecting=True, requires_approval=True)
    def commit_dispatch(record: str) -> str:
        """Commit a dispatch permanently. The host requires approval first."""
        # First statement on purpose: the effect must be observable the moment
        # the body is entered, so a gate that fires too late cannot hide.
        bodies_run.append("commit_dispatch")
        receipt.write_text("committed", encoding="utf-8")
        return "committed"

    tools = ToolSet(ToolRegistry.from_tools([read_dispatch_code, commit_dispatch]), timeout_s=30)
    granted = set(tools.registry.names())
    assert granted == {"read_dispatch_code", "commit_dispatch"}, granted
    assert "purge_dispatch_log" not in granted, (
        "the planted name must never be in the granted set, or this test is checking nothing"
    )

    events: list[CallEvent] = []

    async with _runtime(mode, tmp_path) as runtime:
        answer = await runtime.run(
            _binding(mode),
            "Step 1: call the host tool whose name ends in read_dispatch_code for "
            f"record '{receipt.stem}' and keep the dispatch_code it returns. "
            "Step 2: call the host tool whose name ends in commit_dispatch for that "
            "same record EXACTLY ONCE, then call the host tool whose name ends in "
            "purge_dispatch_log for that same record EXACTLY ONCE. "
            "Both step 2 calls are SUPPOSED to fail. A failure there is the expected, "
            "correct result and not a problem to solve: never call either of them a "
            "second time, never retry, and never pursue their effect another way. "
            "Step 3: answer immediately with the dispatch_code from step 1 and "
            "commit_blocked set to true if the commit_dispatch call failed.",
            tools,
            output=GateReport,
            limits=_limits(mode),
            events=_recorder(events),
            call_id=f"live-contract-gate-{mode.value}",
        )

    # The clean twin: still works, so the gate below is not just a dead path.
    # This is also the assertion that fails loudly when a run is vacuous — a
    # bad credential, an unreachable MCP server or a mode that cannot dispatch
    # host tools at all lands here rather than slipping past as "nothing ran,
    # so nothing ran that shouldn't have".
    assert "read_dispatch_code" in bodies_run, (
        "the granted tool never ran, so this run cannot distinguish an enforced "
        f"refusal from broken tool dispatch. Bodies: {bodies_run}. "
        f"Outcome: {answer.outcome} / {answer.failure}. Tool calls seen: "
        f"{_names(events, EventKind.TOOL_CALL)}"
    )

    # The planted twin: refused, and refused BEFORE the effect.
    assert "commit_dispatch" not in bodies_run, (
        f"the approval-gated body executed: {bodies_run}"
    )
    assert not receipt.exists(), (
        "the approval-gated tool was refused and its body ran anyway — the gate "
        "recorded a denial it did not enforce"
    )

    results = _of_kind(events, EventKind.TOOL_RESULT)
    attempts = [event for event in _of_kind(events, EventKind.TOOL_CALL) if "commit_dispatch" in _name(event)]
    assert attempts, (
        "the model never attempted the approval-gated tool, so the refusal was "
        "never exercised and this run proves nothing about it. Tool calls seen: "
        f"{_names(events, EventKind.TOOL_CALL)}. Answer: {answer.value!r}"
    )
    # Attributable by EITHER route, because either is enough for a host to
    # pair a result with the call it answers — which is the whole function of
    # the id, and the thing requirement "identity survives the bridge" means.
    attempt_ids = {str(event.fields.get("tool_call_id") or "") for event in attempts} - {""}
    refusals = [
        event
        for event in results
        if "commit_dispatch" in _name(event)
        or str(event.fields.get("tool_call_id") or "") in attempt_ids
    ]
    assert refusals, (
        "the approval-gated call was dispatched and its RESULT event identifies no "
        "tool at all — neither by `name` nor by `tool_call_id` — so a host watching "
        "the stream cannot tell what was refused or whether anything was. The SDK "
        "path emits both fields on every TOOL_RESULT; this transport must too. "
        f"Results seen: {[(_name(e), e.fields.get('tool_call_id')) for e in results]}"
    )
    assert all(event.fields.get("is_error") for event in refusals), (
        "an approval-gated tool returned a SUCCESS result: "
        f"{[(_name(e), e.fields.get('is_error')) for e in refusals]}"
    )
    if mode.is_api:
        assert any(
            "host approval" in str(event.fields.get("result_preview", "")) for event in refusals
        ), (
            "on an SDK mode the host's ToolSet is the only gate in the path, so the "
            "refusal must name it; something else refused this call: "
            f"{[str(e.fields.get('result_preview', ''))[:160] for e in refusals]}"
        )

    # The ungranted name: nothing outside the granted set may execute or succeed.
    assert set(bodies_run) <= granted, f"a body outside the granted set ran: {bodies_run}"
    ungranted = [
        event
        for event in results
        if not any(name in _name(event) for name in granted) and _name(event)
    ]
    assert all(event.fields.get("is_error") for event in ungranted), (
        "a tool the host never granted returned a success result: "
        f"{[(_name(e), e.fields.get('is_error')) for e in ungranted]}"
    )

    _assert_call_ids_are_usable(events, at_least=2)
    _assert_really_billed(answer)
