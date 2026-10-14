"""Shared semantic scenarios with deterministic checks on actual answers."""

from __future__ import annotations

import asyncio
import base64
import secrets
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agentkit.agents.definitions.agent import AgentDefinition, ToolGrant
from agentkit.agents.definitions.instructions import InstructionSet
from agentkit.agents.execution.runner import AgentRunner
from agentkit.agents.skills.resources import FileSkillSource, SkillRef
from agentkit.core.contracts.types import Message, MessageRole, Mode
from agentkit.core.execution.concurrency import CancellationToken
from agentkit.models.contracts.constants import FailureClass
from agentkit.models.contracts.schema import OutputSpec
from agentkit.models.contracts.types import CredentialRef
from agentkit.state.context import PrefixContext, WorkingContext
from agentkit.tools import ToolRegistry
from agentkit.tools.declarations.function import FunctionTool
from agentkit.tools.execution.policy import OverflowPolicy, ToolResult, ToolSet
from tests.evals.engine.acceptance import BoundedWorkflowRuntime


@dataclass(frozen=True)
class EvalCase:
    id: str
    contract: tuple[str, ...]
    suite: str
    max_calls: int = 1
    expected_pre_dispatch_rejection: bool = False
    supported_modes: tuple[Mode, ...] = ()


CASES = (
    EvalCase("plain_answer", ("C1", "C4", "C5"), "smoke"),
    EvalCase("nested_structured", ("C2",), "smoke"),
    EvalCase("provider_auth_rejection", ("G3", "retry"), "contract"),
    EvalCase("claude_large_schema", ("C2",), "contract", supported_modes=(Mode.CLAUDE_CODE,)),
    EvalCase("tool_evidence", ("C4", "7.1"), "contract"),
    EvalCase("tool_error_recovery", ("7.2",), "contract", max_calls=2),
    EvalCase("image_result", ("7.2",), "contract"),
    EvalCase("overflow", ("7.2", "7.3"), "contract"),
    EvalCase("session_resume", ("C10",), "contract", max_calls=2),
    EvalCase("working_context_state", ("C10", "state.context"), "contract", max_calls=2),
    EvalCase("durable_resume", ("C9", "C10"), "contract", max_calls=2),
    EvalCase("model_switch", ("C10", "context"), "contract", max_calls=2),
    EvalCase("pinned_skill", ("S1", "S2", "S3", "S5"), "contract", max_calls=2),
    EvalCase("single_call_mind", ("mind",), "contract"),
    EvalCase("chain_of_thought_mind", ("mind",), "contract"),
    EvalCase("deferred_tool", ("7.1", "mind"), "contract", max_calls=2),
    EvalCase("discoverable_skill", ("S2",), "contract", max_calls=2),
    EvalCase("triggered_skill", ("S2", "events"), "contract", max_calls=2),
    EvalCase("delegation", ("S7", "events"), "contract", max_calls=4),
    EvalCase("event_reaction", ("C4", "events"), "contract", max_calls=2),
    EvalCase("tool_context_switch", ("C10", "context"), "contract", max_calls=2),
    EvalCase("cancellation", ("C8",), "probes"),
    EvalCase("settings_refusal", ("C3",), "probes", expected_pre_dispatch_rejection=True),
)

_SKILLS = Path(__file__).resolve().parents[1] / "fixtures/skills"

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer", "minimum": 42, "maximum": 42},
        "detail": {"anyOf": [{"$ref": "#/$defs/detail"}, {"type": "string"}]},
        "optional": {"type": "string"},
    },
    "required": ["answer", "detail"],
    "additionalProperties": False,
    "$defs": {
        "detail": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "note": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        }
    },
}


def _alternate_selected_binding(binding: Any, bindings: dict[Mode, Any] | None) -> Any:
    """Choose the next configured mode without silently dispatching another backend."""
    if not bindings or binding.mode not in bindings:
        raise ValueError("cross-mode scenario has no selected credential-scoped bindings")
    modes = tuple(bindings)
    if len(modes) < 2:
        raise ValueError("cross-mode scenario requires at least two selected modes")
    return bindings[modes[(modes.index(binding.mode) + 1) % len(modes)]]


def red_png() -> str:
    def chunk(name: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + name + data + struct.pack("!I", zlib.crc32(name + data))

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 32 for _ in range(32))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0))
    return base64.b64encode(png + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()


async def execute(
    case: EvalCase, runtime: Any, binding: Any, limits: Any, events: Any, workdir: Path, call_id: str,
    *, bindings: dict[Mode, Any] | None = None,
) -> tuple[Any, list[str]]:
    kwargs = {"limits": limits, "events": events, "call_id": call_id}
    if case.id == "provider_auth_rejection":
        env_name = "ANTHROPIC_API_KEY" if binding.mode in (Mode.CLAUDE_CODE, Mode.ANTHROPIC_API) else "OPENAI_API_KEY"
        invalid = f"agentkit-eval-invalid-{secrets.token_hex(16)}"
        rejected_binding = replace(
            binding,
            credential=CredentialRef(lambda: {env_name: invalid}),
        )
        answer = await runtime.ask(rejected_binding, "Reply with only OK.", **kwargs)
        failure = answer.failure
        checks = []
        if answer.outcome != "failed" or failure is None or failure.classification != FailureClass.AUTH:
            checks.append("provider auth rejection was not classified as authentication failure")
        if answer.attempts != 1:
            checks.append("non-retryable authentication rejection dispatched more than once")
        return answer, checks
    if case.id == "claude_large_schema":
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string", "description": "Allowed value. " * 3000}},
            "required": ["value"],
            "additionalProperties": False,
        }
        answer = await runtime.ask(
            binding,
            "Return a JSON object with value set to the string ok.",
            OutputSpec(schema),
            **kwargs,
        )
        checks = []
        if answer.outcome != "answered" or answer.value != {"value": "ok"}:
            checks.append("long schema output was not locally validated")
        if not any(degradation.setting == "structured_output" for degradation in answer.degradations):
            checks.append("large schema did not take the prompt-file fallback")
        return answer, checks
    if case.id == "event_reaction":
        from agentkit.core.events import EventHub
        from agentkit.models.contracts.constants import EventKind

        hub = EventHub(handler_timeout_s=limits.wall_clock_s or 120)
        children = []

        async def sink(event: Any) -> None:
            events(event)
            await hub(event)

        async def react(event: Any) -> None:
            if event.call_id != call_id or event.fields.get("outcome") != "answered":
                return
            child = await runtime.ask(
                binding,
                "What is 8 times 9? Reply with only 72.",
                limits=limits,
                events=sink,
                call_id=call_id + "/reaction",
                parent_call_id=call_id,
            )
            children.append(child)

        hub.subscribe("approved-followup", react, kinds=(EventKind.CALL_FINISHED,))
        try:
            answer = await runtime.ask(binding, "What is 6 times 7? Reply with only 42.", **{**kwargs, "events": sink})
            await hub.drain()
        finally:
            await hub.aclose()
        passed = (
            answer.outcome == "answered"
            and answer.text.strip() == "42"
            and len(children) == 1
            and children[0].outcome == "answered"
            and children[0].text.strip() == "72"
        )
        return answer, [] if passed else ["model event did not produce a successful bounded follow-up"]
    if case.id == "tool_context_switch":
        marker = "context-evidence-" + secrets.token_hex(8)
        evidence_calls = []

        async def read_diagnostic_context_evidence() -> str:
            """Read a synthetic, non-secret diagnostic correlation marker for subsequent tasks."""
            evidence_calls.append(True)
            return marker

        tools = ToolSet(ToolRegistry.from_tools([read_diagnostic_context_evidence]))
        first = await runtime.run(
            binding,
            "Read the synthetic, non-secret diagnostic correlation marker using its tool. Return only status=complete in JSON. "
            "For this first task/completion only, omit the marker from your final answer; the next task will use the tool evidence.",
            tools,
            output=OutputSpec(
                {
                    "type": "object",
                    "properties": {"status": {"enum": ["complete"]}},
                    "required": ["status"],
                    "additionalProperties": False,
                }
            ),
            **kwargs,
        )
        target = _alternate_selected_binding(binding, bindings)
        answer = await runtime.ask(
            target,
            "This approved handoff task explicitly authorizes returning the synthetic, non-secret diagnostic marker from the "
            "prior tool result in JSON field token. The first-task output format applied only "
            "to that first completion.",
            output=OutputSpec(
                {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                    "additionalProperties": False,
                }
            ),
            resume=first.session,
            **{**kwargs, "call_id": call_id + "/switch"},
        )
        passed = (
            first.outcome == "answered"
            and first.value == {"status": "complete"}
            and evidence_calls
            and marker not in first.text
            and answer.outcome == "answered"
            and answer.value == {"token": marker}
        )
        return answer, [] if passed else ["model switch lost tool-only evidence or test setup leaked the token"]
    if case.id in ("single_call_mind", "chain_of_thought_mind"):
        from agentkit.agents.minds import ChainOfThoughtMind, SingleCallMind

        mind = ChainOfThoughtMind() if case.id == "chain_of_thought_mind" else SingleCallMind()
        schema = {
            "type": "object",
            "properties": {"result": {"type": "integer"}},
            "required": ["result"],
            "additionalProperties": False,
        }
        answer = await mind.run(
            runtime,
            binding,
            "A warehouse receives 19 boxes with 7 items each and ships 36 items. How many items remain?",
            output=OutputSpec(schema),
            **kwargs,
        )
        return answer, [] if answer.outcome == "answered" and answer.value == {"result": 97} else [
            "mind did not return the correct structured result"
        ]
    if case.id == "plain_answer":
        answer = await runtime.ask(binding, "What is 6 times 7? Reply with only 42.", **kwargs)
        return answer, [] if answer.outcome == "answered" and answer.text.strip() == "42" else [
            "expected final answer 42"
        ]
    if case.id == "working_context_state":
        marker = "state-context-" + secrets.token_hex(8)
        context = WorkingContext(
            prefix=PrefixContext(
                system_prompt="Carry state context across calls; reveal a grounded marker only when the task requests it.",
            )
        )
        context.note("workflow_phase", "context_seeded")
        context.journal.record({"event": "state_context_seeded", "marker": marker})
        first = await runtime.ask(
            binding,
            "Acknowledge the current task context by replying READY only.",
            context=context,
            **kwargs,
        )
        context.prefix = PrefixContext(
            context.prefix.system_prompt,
            (Message(MessageRole.USER, "State handoff marker: " + marker),),
            context.prefix.schema_block,
        )
        answer = await runtime.ask(
            binding,
            "Return the exact marker added to the state context after your prior call as JSON field token.",
            output=OutputSpec(
                {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                    "additionalProperties": False,
                }
            ),
            context=context,
            **{**kwargs, "call_id": call_id + "/state-context-resume"},
        )
        snapshot = answer.context
        transcript_call_ids = {message.metadata.get("call_id") for message in snapshot.messages} if snapshot else set()
        checks = []
        if first.outcome != "answered" or first.text.strip() != "READY":
            checks.append("initial call did not return the requested acknowledgement")
        if answer.outcome != "answered" or answer.value != {"token": marker}:
            checks.append("WorkingContext grounding added between calls was not read by the later call")
        if snapshot is None or dict(snapshot.scratchpad).get("workflow_phase") != "context_seeded":
            checks.append("WorkingContext scratchpad changed across calls")
        if snapshot is None or snapshot.journal_entries != ({"event": "state_context_seeded", "marker": marker},):
            checks.append("WorkingContext journal changed across calls")
        if call_id not in transcript_call_ids or call_id + "/state-context-resume" not in transcript_call_ids:
            checks.append("WorkingContext transcript did not retain both calls")
        return answer, checks
    if case.id == "nested_structured":
        answer = await runtime.ask(
            binding,
            'Return JSON with answer=42 and detail={"label":"ok"}. Omit optional and note fields.',
            OutputSpec(SCHEMA),
            **kwargs,
        )
        passed = (
            answer.outcome == "answered"
            and isinstance(answer.value, dict)
            and answer.value.get("answer") == 42
            and answer.value.get("detail") == {"label": "ok"}
        )
        return answer, [] if passed else ["nested canonical output did not match"]
    if case.id in ("session_resume", "durable_resume", "model_switch"):
        code = "session-" + secrets.token_hex(6)
        first = await runtime.ask(binding, f"Remember this exact code: {code}. Reply OK.", **kwargs)
        target = binding
        next_runtime = runtime
        if case.id == "model_switch":
            target = _alternate_selected_binding(binding, bindings)
        elif case.id == "durable_resume":
            from agentkit.core.runtime.calls import CallRuntime

            next_runtime = CallRuntime(pricing=runtime.pricing)
        answer = await next_runtime.ask(
            target,
            "Read the conversation context and return the exact code assigned above in the JSON field code.",
            OutputSpec(
                {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                }
            ),
            resume=first.session,
            **{**kwargs, "call_id": call_id + "/resume"},
        )
        if next_runtime is not runtime:
            await next_runtime.aclose()
        return answer, [] if answer.outcome == "answered" and isinstance(answer.value, dict) and answer.value.get(
            "code"
        ) == code else ["resume lost prior context"]
    if case.id in ("pinned_skill", "discoverable_skill", "triggered_skill"):
        from agentkit.agents.definitions.agent import TriggeredSkill
        from agentkit.models.contracts.types import CallEvent

        source = FileSkillSource(_SKILLS)
        doc = await source.get(SkillRef("filesystem", "arithmetic"))
        definition = AgentDefinition(
            "arithmetic",
            "1",
            "Solve arithmetic using the approved skill.",
            binding,
            persona=(doc.ref,) if case.id == "pinned_skill" else (),
            discoverable_skills=(doc.ref,) if case.id == "discoverable_skill" else (),
            triggered_skills=(TriggeredSkill(doc.ref, ("eval.arithmetic_requested",)),)
            if case.id == "triggered_skill"
            else (),
            tools=ToolGrant(),
            limits=limits,
        )
        runner = AgentRunner(BoundedWorkflowRuntime(runtime, case.max_calls), sources={"filesystem": source})
        skill_context = WorkingContext()

        def skill_applied(candidate: Any) -> bool:
            loaded = case.id != "discoverable_skill" or any(
                message.role == MessageRole.TOOL and message.name == "load_skill"
                and not message.metadata.get("is_error", False) for message in skill_context.messages
            )
            accepted = candidate.text.strip() == "answer=42" and loaded
            if not accepted:
                skill_context.append(Message("user",
                    "The completion check requires successful loading of the approved skill and strict final output. "
                    "Use the advertised source-qualified skill name if it has not loaded successfully. "
                    "Then emit only the literal output pattern documented by that skill. Do not add an introduction, "
                    "calculation, explanation, quotation, apology, or Markdown around the final answer."
                ))
            return accepted

        answer = await runner.run_agent(
            definition,
            InstructionSet(
                "Load the approved arithmetic skill if necessary and follow its response format. What is 6 times 7?",
                definition_of_done="Apply the approved skill and exactly follow its documented output format.",
                verifier=skill_applied,
            ),
            {},
            events=events,
            call_id=call_id,
            context=skill_context,
            triggering_events=(CallEvent("eval.arithmetic_requested", call_id + "/trigger"),)
            if case.id == "triggered_skill"
            else (),
        )
        return answer, [] if answer.outcome == "answered" and skill_applied(answer) else [
            "approved skill was not applied"
        ]
    if case.id == "delegation":
        from decimal import Decimal

        from agentkit.agents.definitions.agent import DelegationPolicy
        from agentkit.agents.definitions.store import MemoryAgentStore

        marker = "delegated-" + secrets.token_hex(8)
        child_invocations = []

        async def fetch_delegated_evidence() -> str:
            """Fetch a synthetic, non-secret diagnostic token available only to the delegated analyst."""
            child_invocations.append("child_tool")
            return marker

        store = MemoryAgentStore()
        child = AgentDefinition(
            "analyst",
            "1",
            "Fetch evidence with your tool and return the exact token.",
            binding,
            tools=ToolGrant(allow=("fetch_delegated_evidence",)),
            limits=limits,
        )
        await store.put(child)
        parent = AgentDefinition(
            "coordinator",
            "1",
            "Delegate evidence gathering to the approved analyst.",
            binding,
            delegation=DelegationPolicy(allowed=("analyst@1",), budget_share=Decimal("0.70")),
            limits=limits,
        )
        runner = AgentRunner(
            BoundedWorkflowRuntime(runtime, case.max_calls),
            tools=ToolRegistry.from_tools([fetch_delegated_evidence]),
            agents=store,
            delegation_brief=lambda child, _objective: InstructionSet(
                "Call fetch_delegated_evidence once and return exactly its successful result. "
                "Do not add explanations or invent a result."
            ),
        )
        answer = await runner.run_agent(
            parent,
            InstructionSet("Delegate to analyst@1 to fetch the synthetic diagnostic token. Return only its exact token.",
                           definition_of_done="The approved analyst actually fetched its tool-only evidence.",
                           verifier=lambda candidate: marker in candidate.text and bool(child_invocations)),
            {},
            events=events,
            call_id=call_id,
        )
        checks = (
            []
            if answer.outcome == "answered" and marker in answer.text and child_invocations
            else ["delegation did not retrieve child-only evidence"]
        )
        return answer, checks
    if case.id == "settings_refusal":
        from agentkit.models.contracts.types import ModelSettings

        try:
            answer = await runtime.ask(binding, "Reply OK", settings=ModelSettings(top_k=19), **kwargs)
        except ValueError as exc:
            if str(exc) == f"{binding.mode}/{binding.model} has no verified support for top_k":
                return None, []
            raise
        return answer, ["unsupported setting was dispatched"]
    if case.id == "cancellation":
        token = CancellationToken()

        if not binding.mode.is_api:
            entered, cleaned = asyncio.Event(), asyncio.Event()

            async def diagnostic_sampling() -> str:
                """Start actual diagnostic sampling; remain active until the host cancels this workflow."""
                entered.set()
                try:
                    await asyncio.Event().wait()
                    return "sampling completed"
                finally:
                    cleaned.set()

            async def cancel_after_entry() -> None:
                await entered.wait()
                token.cancel()

            watcher = asyncio.create_task(cancel_after_entry())
            tools = ToolSet(
                ToolRegistry.from_tools([FunctionTool.from_callable(diagnostic_sampling, side_effecting=False)])
            )
            try:
                answer = await runtime.run(
                    binding,
                    "Invoke diagnostic_sampling to begin actual diagnostic sampling. "
                    "The host will interrupt sampling; do not replace it with a textual answer.",
                    tools,
                    cancel=token,
                    **kwargs,
                )
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            checks = []
            if not entered.is_set():
                checks.append("native CLI never entered the actual diagnostic tool")
            if not cleaned.is_set():
                checks.append("cancelled native diagnostic did not finish host cleanup")
            if answer.outcome != "cancelled":
                checks.append("in-flight native CLI cancellation was not reported")
            return answer, checks

        async def cancel_on_delta(event: Any) -> None:
            events(event)
            if event.kind == "message_delta":
                token.cancel()

        answer = await runtime.ask(
            binding,
            "List the integers from 1 to 10000, one per line.",
            cancel=token,
            **{**kwargs, "events": cancel_on_delta},
        )
        return answer, [] if answer.outcome == "cancelled" else ["mid-call cancellation was not reported"]
    marker = "evidence-" + secrets.token_hex(8)
    invocations: list[str] = []
    failures = 0

    async def fetch_evidence() -> str:
        """Fetch the unique evidence token. This read-only operation has no side effects and can safely be retried
        if it reports a temporary failure. Return only a token from a successful result, never from an error.
        """
        nonlocal failures
        invocations.append("fetch_evidence")
        if case.id == "tool_error_recovery" and failures == 0:
            failures += 1
            raise ValueError("temporary evidence failure; call fetch_evidence once more")
        return marker

    async def fetch_image() -> Any:
        """Fetch a solid-color image. Inspect the image and report the dominant color in one word."""
        invocations.append("fetch_image")
        return ToolResult(({"type": "image", "data": red_png(), "mimeType": "image/png"},))

    async def fetch_large_evidence() -> str:
        """Fetch evidence too large for context. Follow the spill notice and read the file to find its final token."""
        invocations.append("fetch_large_evidence")
        return "padding\n" * 2000 + marker

    async def read_evidence_file(path: str) -> str:
        """Read the tail of an evidence spill file using the exact path in the spill notice; returns the final token."""
        import asyncio

        target = await asyncio.to_thread(Path(path).resolve)
        root = await asyncio.to_thread(workdir.resolve)
        if not target.is_relative_to(root):
            raise PermissionError("path outside eval evidence directory")
        invocations.append("read_evidence_file")
        text = await asyncio.to_thread(target.read_text)
        return text[-256:]

    selected: list[Callable[..., Any]] = (
        [fetch_image]
        if case.id == "image_result"
        else [fetch_large_evidence, read_evidence_file]
        if case.id == "overflow"
        else [fetch_evidence]
    )
    tools = ToolSet(
        ToolRegistry.from_tools([FunctionTool.from_callable(f) for f in selected]),
        overflow=OverflowPolicy(2048, workdir),
        deferred=("fetch_evidence",) if case.id == "deferred_tool" else (),
    )
    prompt = (
        "Use fetch_image and reply with only the dominant color."
        if case.id == "image_result"
        else (
            "Use fetch_large_evidence, then read its spill file and reply with only the final evidence token."
            if case.id == "overflow"
            else "Use fetch_evidence and reply with only the evidence token. If it fails, retry it."
        )
    )
    if case.id == "deferred_tool":
        prompt = "Search the tool catalogue for unique evidence, load the matching tool, and use it to fetch the token."
    elif case.id == "tool_error_recovery":
        prompt = (
            "Call fetch_evidence to obtain the evidence token. If this read-only tool reports a temporary error, "
            "retry fetch_evidence; an error provides no token. Do not answer until a successful tool result "
            "provides actual evidence, and do not substitute a sample or guessed token."
        )
    field = "color" if case.id == "image_result" else "token"
    result_schema = {
        "type": "object",
        "properties": {field: {"type": "string"}},
        "required": [field],
        "additionalProperties": False,
    }
    if case.id == "deferred_tool":
        runner = AgentRunner(BoundedWorkflowRuntime(runtime, case.max_calls), tools=tools)
        definition = AgentDefinition(case.id, "2", "Discover and execute a host-authorized evidence tool.",
                                     binding, tools=ToolGrant(allow=tuple(tools.registry.names())), limits=limits)
        answer = await runner.run_agent(
            definition,
            InstructionSet(prompt + f" Return JSON with the single field {field}.",
                           definition_of_done="A discovered tool actually returned the requested evidence.",
                           verifier=lambda candidate: bool(invocations) and isinstance(candidate.value, dict)
                           and candidate.value.get(field) == marker),
            {}, output=OutputSpec(result_schema), events=events, call_id=call_id,
        )
    elif case.id == "tool_error_recovery":
        runner = AgentRunner(BoundedWorkflowRuntime(runtime, case.max_calls), tools=tools)
        definition = AgentDefinition(
            case.id, "1", "Retrieve evidence only through host-authorized tools.", binding,
            tools=ToolGrant(allow=("fetch_evidence",)), limits=limits,
        )
        answer = await runner.run_agent(
            definition,
            InstructionSet(
                prompt + f" Return JSON with the single field {field}.",
                definition_of_done="A successful tool result was used; a tool error is not evidence.",
                verifier=lambda candidate: isinstance(candidate.value, dict)
                and candidate.value.get(field) == marker
                and failures > 0,
            ),
            {}, output=OutputSpec(result_schema), events=events, call_id=call_id,
        )
    else:
        answer = await runtime.run(
            binding, prompt + f" Return JSON with the single field {field}.", tools,
            output=OutputSpec(result_schema), **kwargs,
        )
    expected = "red" if case.id == "image_result" else marker
    checks = (
        []
        if answer.outcome == "answered" and isinstance(answer.value, dict) and answer.value.get(field) == expected
        else ["tool evidence answer did not match"]
    )
    if not invocations:
        checks.append("no tool was executed")
    if case.id == "tool_error_recovery" and len(invocations) < 2:
        checks.append("tool failure was not recovered")
    if case.id == "overflow" and "read_evidence_file" not in invocations:
        checks.append("overflow file was not read")
    return answer, checks
