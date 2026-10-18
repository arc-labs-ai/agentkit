"""One policy loop for all four modes: ask, run and bounded continuation."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import json
import random
import secrets
import time
from collections.abc import Callable, Coroutine, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

from agentkit.agents.prompts.composition import ComposedPrompt, digest, inline_prompt
from agentkit.core.contracts.types import Message, MessageRole, Mode, Scope, ToolCall
from agentkit.core.execution.concurrency import CancellationToken, Cancelled
from agentkit.models.adapters.messages import canonical_turn, close_pending_tools, history_for, tool_content
from agentkit.models.catalog.pricing import PriceTable, normalize_usage, price, usage_delta
from agentkit.models.contracts.constants import (
    CostSource,
    EventKind,
    FailureClass,
    NativeToolPolicy,
    Outcome,
    UnpricedPolicy,
)
from agentkit.models.contracts.execution import (
    CapabilityStatus,
    ExecutionAssessment,
    ExecutionCompatibility,
    ExecutionRequirement,
    ExecutionRequirements,
)
from agentkit.models.contracts.failures import classify
from agentkit.models.contracts.schema import (
    OutputSpec,
    SchemaError,
    compile_schema,
    tolerant_json_object,
    validate_output,
)
from agentkit.models.contracts.settings import DEFAULT_CAPABILITIES, CapabilityTable
from agentkit.models.contracts.types import (
    Answer,
    Binding,
    CallEvent,
    CallLimits,
    Degradation,
    EventSink,
    Failure,
    ModeAdapter,
    ModelSettings,
    SessionHandle,
    Turn,
    Usage,
    Workspace,
)
from agentkit.state.context import PrefixContext, WorkingContext


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_s: float = 0.5
    cap_s: float = 30
    max_elapsed_s: float = 120

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or min(self.base_s, self.cap_s, self.max_elapsed_s) < 0:
            raise ValueError("invalid retry bounds")


@dataclass(frozen=True)
class _ToolContext:
    """Invocation scope and cancellation boundary shared with every tool call."""

    scope: Scope
    cancel: Any
    correlation_id: str
    autonomy: str = "auto"
    attempt_id: str | None = None

    def check_cancelled(self) -> None:
        if self.cancel is not None:
            self.cancel.raise_if_cancelled()


_DEFAULT_RETRY = RetryPolicy()
_DEFAULT_SETTINGS = ModelSettings()
_DEFAULT_LIMITS = CallLimits()


class CallRuntime:
    """Owns adapter resources and the common contract.

    Use as an async context manager when CLI sessions must survive multiple
    calls. Inject adapters for offline tests and prices/capabilities for hosts.
    """

    def __init__(
        self,
        *,
        adapters: Mapping[str | Mode, ModeAdapter] | None = None,
        pricing: PriceTable | None = None,
        capabilities: CapabilityTable = DEFAULT_CAPABILITIES,
        retry: RetryPolicy = _DEFAULT_RETRY,
        capture_text: bool = False,
        native_tool_policy: NativeToolPolicy = NativeToolPolicy.WARN,
        unpriced_policy: UnpricedPolicy = UnpricedPolicy.REFUSE,
        max_concurrent_cli: int = 8,
        max_concurrent_api: int = 32,
        acquire: Callable[[Mode], AbstractAsyncContextManager[Any]] | None = None,
        argument_masker: Callable[[str, Any], Any] | None = None,
        result_masker: Callable[[str, str], str] | None = None,
    ) -> None:
        self.adapters = {Mode(mode): adapter for mode, adapter in (adapters or {}).items()}
        self.pricing = pricing or PriceTable.default()
        self.capabilities = capabilities
        self.retry = retry
        self.capture_text = capture_text
        self.native_tool_policy = NativeToolPolicy(native_tool_policy)
        self.unpriced_policy = UnpricedPolicy(unpriced_policy)
        if max_concurrent_cli < 1:
            raise ValueError("max_concurrent_cli must be positive")
        if max_concurrent_api < 1:
            raise ValueError("max_concurrent_api must be positive")
        self.max_concurrent_cli = max_concurrent_cli
        self.max_concurrent_api = max_concurrent_api
        self.acquire = acquire
        self._event_digest_key = secrets.token_bytes(32)
        self.argument_masker = argument_masker or (
            lambda name, value: _mask_tool_arguments(name, value, key=self._event_digest_key)
        )
        self.result_masker = result_masker
        self._cli_semaphores: dict[Mode, asyncio.Semaphore] = {}
        self._api_semaphores: dict[Mode, asyncio.Semaphore] = {}

    async def __aenter__(self) -> CallRuntime:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        closed = set()
        for adapter in self.adapters.values():
            if id(adapter) in closed:
                continue
            closed.add(id(adapter))
            if hasattr(adapter, "aclose"):
                await adapter.aclose()

    def _adapter(self, binding: Binding) -> ModeAdapter:
        if binding.mode not in self.adapters:
            if binding.mode.is_api:
                from agentkit.models.providers.frontiers.calls import ApiAdapter

                self.adapters[binding.mode] = ApiAdapter()
            else:
                from agentkit.models.providers.cli.calls import CliAdapter

                self.adapters[binding.mode] = CliAdapter(native_tool_policy=self.native_tool_policy)
        return self.adapters[binding.mode]

    async def ask(
        self, binding: Binding, prompt: ComposedPrompt | str, output: OutputSpec | type | None = None, **kwargs: Any
    ) -> Answer[Any]:
        """A one-shot call, with one shared bounded correction turn."""
        return await self.run(binding, prompt, output=output, **kwargs)

    async def evaluate_execution(
        self,
        binding: Binding,
        *,
        execution: ExecutionRequirements | None = None,
        native_tool_policy: NativeToolPolicy | None = None,
    ) -> ExecutionCompatibility:
        """Assess execution without credentials, prompts or provider dispatch."""
        requested = NativeToolPolicy(native_tool_policy or self.native_tool_policy)
        rank = {NativeToolPolicy.DENY: 0, NativeToolPolicy.WARN: 1, NativeToolPolicy.ALLOW: 2}
        if rank[requested] > rank[self.native_tool_policy]:
            raise ValueError("per-call native_tool_policy cannot exceed the CallRuntime policy ceiling")
        adapter = self._adapter(binding)
        # Every shipped adapter assesses its own mode — the CLI pair through
        # `evaluate_cli_execution`, the SDK pair through `evaluate_api_execution`
        # — so there is no per-mode special case here. An adapter injected by a
        # host may implement none, and that is the only path below.
        evaluate = getattr(adapter, "evaluate_execution", None)
        if evaluate is not None:
            return cast(ExecutionCompatibility, await evaluate(
                binding, execution=execution, native_tool_policy=requested,
            ))
        # An adapter that declares no assessment proves nothing, so it answers
        # the FULL matrix as UNSUPPORTED rather than an empty report that reads
        # as a pass. A DENY ceiling still contributes its own requirement, which
        # is what keeps the strictest policy a constraint rather than a label.
        required = set((execution or ExecutionRequirements()).required)
        if requested == NativeToolPolicy.DENY:
            required.add(ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE)
        facts = tuple(ExecutionAssessment(
            requirement, CapabilityStatus.UNSUPPORTED, "execution_mode_unsupported",
            "This adapter does not implement enforcement for the requested execution requirement.",
        ) for requirement in ExecutionRequirement)
        unmet = tuple(fact for fact in facts if fact.requirement in required)
        return ExecutionCompatibility(str(binding.mode), "unspecified", "unknown", None, facts, unmet)

    async def run(
        self,
        binding: Binding,
        prompt: ComposedPrompt | str,
        tools: Any = None,
        *,
        output: OutputSpec | type | None = None,
        settings: ModelSettings = _DEFAULT_SETTINGS,
        limits: CallLimits = _DEFAULT_LIMITS,
        cancel: CancellationToken | None = None,
        events: EventSink | None = None,
        pricing: PriceTable | None = None,
        call_id: str,
        resume: SessionHandle | None = None,
        turns_per_slice: int | None = None,
        stable_prefix: bool = False,
        capture_text: bool | None = None,
        fork: bool = False,
        context: WorkingContext | None = None,
        parent_call_id: str | None = None,
        native_agents: Mapping[str, Any] | None = None,
        workspace: Workspace | None = None,
        scope: Scope | None = None,
        context_policy: Any = None,
        native_tool_policy: NativeToolPolicy | None = None,
        execution: ExecutionRequirements | None = None,
    ) -> Answer[Any]:
        if not call_id:
            raise ValueError("call_id is required")
        capture_text_for_call = self.capture_text if capture_text is None else capture_text
        if capture_text_for_call and self.result_masker is None:
            raise ValueError("capture_text requires a host result_masker so narration cannot disclose secrets")
        requested_native_policy = self.native_tool_policy if native_tool_policy is None else NativeToolPolicy(
            native_tool_policy
        )
        policy_rank = {NativeToolPolicy.DENY: 0, NativeToolPolicy.WARN: 1, NativeToolPolicy.ALLOW: 2}
        if resume is not None:
            if resume.execution is not None:
                execution = (execution or ExecutionRequirements()).constrain(resume.execution)
            if resume.native_tool_policy is not None:
                requested_native_policy = min(
                    (requested_native_policy, resume.native_tool_policy), key=policy_rank.__getitem__,
                )
        if policy_rank[requested_native_policy] > policy_rank[self.native_tool_policy]:
            raise ValueError("per-call native_tool_policy cannot exceed the CallRuntime policy ceiling")
        adapter_policy = getattr(self.adapters.get(binding.mode), "native_tool_policy", requested_native_policy)
        effective_native_policy = min(
            (self.native_tool_policy, requested_native_policy, NativeToolPolicy(adapter_policy)),
            key=policy_rank.__getitem__,
        )
        scope = scope or Scope()
        if turns_per_slice is not None and turns_per_slice < 1:
            raise ValueError("turns_per_slice must be positive")
        if turns_per_slice is not None and binding.mode == Mode.CODEX:
            raise ValueError("Codex CLI does not expose a verifiable agentic-turn slice limit")
        if stable_prefix and binding.mode != Mode.CLAUDE_CODE:
            raise ValueError("stable_prefix is currently supported only by Claude Code CLI")
        if fork and (binding.mode.is_api or resume is None or not resume.cli_session_id):
            raise ValueError("fork requires a resumable CLI session")
        model_changed = resume is not None and (
            resume.mode != binding.mode or (resume.model and resume.model != binding.model)
        )
        prompt = inline_prompt(prompt) if isinstance(prompt, str) else prompt
        working = context if context is not None else WorkingContext()
        if not working.messages and resume is not None:
            working.extend(resume.messages)
            if context is None:
                working.prefix = resume.prefix
        if prompt.system:
            working.prefix = PrefixContext(prompt.system, working.prefix.grounding, working.prefix.schema_block)
        prefix_messages = working.prefix.as_messages()
        system = "\n\n".join(
            m.content for m in prefix_messages if m.role in (MessageRole.SYSTEM, MessageRole.DEVELOPER)
        )
        if system != prompt.system:
            prompt = replace(prompt, system=system, prefix_digest=digest([prompt.prefix_digest, system]))
        context_changed = resume is not None and (
            tuple(working.messages) != resume.messages or working.prefix != resume.prefix
        )
        working.append(
            Message(
                MessageRole.USER,
                prompt.user,
                content_blocks=tuple(
                    {"type": "image", "mimeType": image.mime_type, "data": base64.b64encode(image.data).decode("ascii")}
                    for image in prompt.images
                ),
                metadata={"call_id": call_id, "agent_id": working.get("agent_id")},
            )
        )
        effective = self.capabilities.check(binding, settings)
        schema_adapter = None
        if output is not None and not isinstance(output, OutputSpec):
            from agentkit.models.output import adapt

            schema_adapter = adapt(output)
            output = OutputSpec(schema_adapter.json_schema(), schema_adapter.name)
        schema_failure = None
        try:
            schema = compile_schema(output.schema, binding.mode, binding.model) if output else None
        except SchemaError as exc:
            schema = None
            explanations = {
                "schema_budget_exceeded": "Reduce optional fields or union parameters in the compiled output schema.",
                "invalid_schema": "Validate the canonical JSON Schema locally before retrying.",
                "recursive_schema": "This transport requires an acyclic output schema; use a finite structure.",
                "unresolved_reference": "Use resolvable local references in the output schema.",
                "unsupported_ref_siblings": "Inline and combine reference sibling constraints explicitly.",
                "unsupported_reference_location": "Move references into schema locations this compiler supports.",
            }
            schema_failure = Failure(
                FailureClass.INVALID_REQUEST, False,
                mode_detail={
                    "reason": "schema_compilation_failed", "origin": "local", "dispatch_state": "not_started",
                    "code": exc.code, **exc.details,
                    "explanation": explanations.get(exc.code, "Use an output schema supported by this mode's dialect."),
                },
            )
        table = pricing or self.pricing
        adapter = self._adapter(binding)
        started = time.monotonic()
        last_event = started
        saw_event = False

        sequence = 0

        event_sink_failure: str | None = None
        event_sink_failure_streak = 0
        event_sink_disabled = False

        async def emit(kind: str, **fields: Any) -> None:
            nonlocal sequence, event_sink_failure, event_sink_failure_streak, event_sink_disabled
            sequence += 1
            if events is not None and not event_sink_disabled:
                event = CallEvent(kind, call_id, fields, sequence, datetime.now(UTC).isoformat(), parent_call_id)
                try:
                    result = events(event)
                    if inspect.isawaitable(result):
                        await result
                    event_sink_failure_streak = 0
                except Exception as exc:
                    # Event delivery is observational. Preserve the answer,
                    # report the first failure immediately, and keep trying;
                    # only a sustained failure disables the sink.
                    event_sink_failure = type(exc).__name__
                    event_sink_failure_streak += 1
                    if kind != EventKind.DEGRADED and event_sink_failure_streak < 3:
                        sequence += 1
                        degraded = CallEvent(
                            EventKind.DEGRADED,
                            call_id,
                            {
                                "setting": "event_sink",
                                "requested": "enabled",
                                "applied": "retrying delivery",
                                "reason": f"subscriber raised {event_sink_failure}; answer is preserved",
                            },
                            sequence,
                            datetime.now(UTC).isoformat(),
                            parent_call_id,
                        )
                        try:
                            result = events(degraded)
                            if inspect.isawaitable(result):
                                await result
                        except Exception:
                            event_sink_failure_streak += 1
                    if event_sink_failure_streak >= 3:
                        event_sink_disabled = True

        async def mode_event(event: CallEvent) -> None:
            nonlocal last_event, saw_event, effective
            last_event, saw_event = time.monotonic(), True
            if event.kind == EventKind.TRANSPORT:
                return
            fields = dict(event.fields)
            if event.kind == EventKind.DEGRADED:
                effective = replace(
                    effective,
                    degradations=(
                        *effective.degradations,
                        Degradation(
                            str(fields.get("setting", "transport")),
                            fields.get("requested"),
                            fields.get("applied"),
                            str(fields.get("reason", "transport capability unavailable")),
                        ),
                    ),
                )
            if event.kind == EventKind.MESSAGE_DELTA and not capture_text_for_call:
                fields = {"length": len(str(fields.get("text", "")))}
            if event.kind == EventKind.TOOL_CALL and "arguments" in fields:
                fields["arguments"] = self.argument_masker(str(fields.get("name", "")), fields["arguments"])
            if event.kind == EventKind.TOOL_RESULT and "result_preview" in fields:
                if self.result_masker is None:
                    fields.pop("result_preview", None)
                else:
                    preview = self.result_masker(str(fields.get("name", "")), str(fields["result_preview"]))
                    fields["result_preview"] = preview.encode("utf-8")[:512].decode("utf-8", errors="ignore")
            await emit(event.kind, **fields)

        async def apply_context_policy() -> None:
            nonlocal working
            if context_policy is None:
                return
            revised = context_policy(working)
            if inspect.isawaitable(revised):
                revised = await revised
            from agentkit.state.context import WorkingContext

            if not isinstance(revised, WorkingContext):
                raise TypeError("context policy must return WorkingContext")
            working = revised

        await emit(
            EventKind.CALL_STARTED,
            mode=binding.mode,
            model=binding.model,
            agent_id=working.get("agent_id"),
            context_digest=digest([asdict(m) for m in working.messages]),
            settings=asdict(effective.settings),
            inputs_digest=prompt.manifest.inputs_digest,
            prompt_manifest_digest=prompt.manifest.digest,
            attempt=0 if schema_failure is not None else 1,
        )
        if model_changed:
            assert resume is not None
            degradation = Degradation(
                "continuation",
                f"{resume.mode}/{resume.model}",
                f"{binding.mode}/{binding.model}",
                "portable conversation replayed; incompatible provider reasoning state omitted",
            )
            effective = replace(effective, degradations=(*effective.degradations, degradation))
        if context_changed and not model_changed:
            effective = replace(
                effective,
                degradations=(
                    *effective.degradations,
                    Degradation(
                        "context_replay",
                        "opaque provider continuation",
                        "canonical caller context",
                        "caller context changed; portable conversation replayed; opaque reasoning/native state omitted",
                    ),
                ),
            )
        for degradation in effective.degradations:
            await emit(EventKind.DEGRADED, **asdict(degradation))
        prefix = digest(
            [
                prompt.prefix_digest,
                binding.mode,
                binding.model,
                schema,
                asdict(effective.settings),
                [asdict(s) for s in tools.schemas()] if tools else [],
            ]
        )
        if resume and resume.prefix_digest != prefix:
            await emit(EventKind.CACHE_INVALIDATED, reason="prompt, tools, schema or settings changed")
        usage = Usage(reasoning=0)
        usage_complete = True
        raw: dict[str, Any] = {}
        total_cost: Decimal | None = Decimal(0)
        budget_spent = Decimal(0)
        budget_unknown = False
        cost_source: CostSource = CostSource.PRICE_TABLE
        outcome: Outcome = Outcome.FAILED
        failure: Failure | None = None
        messages: list[str] = []
        text = ""
        value = None
        attempts = 0
        retry_streak = 0
        corrections = 0
        turns = 0
        current_session = resume
        model_in_flight = False
        started_tools: set[str] = set()
        active_host_tools = 0
        last_tool_activity_end: float | None = None

        def tool_activity(active: bool | None = None) -> bool:
            nonlocal active_host_tools, last_tool_activity_end
            if active is None:
                return active_host_tools > 0
            previous = active_host_tools
            active_host_tools = max(0, active_host_tools + (1 if active else -1))
            if previous > 0 and active_host_tools == 0:
                last_tool_activity_end = time.monotonic()
            return active_host_tools > 0

        def tool_activity_ended_at() -> float | None:
            return last_tool_activity_end
        if (
            model_changed
            or context_changed
            or (resume is None and (len(working.messages) > 1 or working.prefix.grounding))
        ):
            keep_native_cli_session = (
                resume is not None
                and not model_changed
                and not binding.mode.is_api
                and resume.mode == binding.mode
                and bool(getattr(adapter, "preserves_native_session", False))
            )
            current_session = SessionHandle(
                binding.mode,
                call_id,
                cli_session_id=resume.cli_session_id if keep_native_cli_session and resume is not None else None,
                history=history_for((*working.prefix.grounding, *working.messages), binding.mode),
                messages=tuple(working.messages),
                model=binding.model,
                prefix=working.prefix,
                home_bundle_ref=resume.home_bundle_ref if keep_native_cli_session and resume is not None else None,
                execution=execution,
                native_tool_policy=effective_native_policy,
            )
        elif resume and resume.history:
            current_session = replace(resume, history=(*resume.history, {"role": "user", "content": prompt.user}))

        async def monitored(coro: Coroutine[Any, Any, Turn], *, transport: bool = True) -> Turn:
            task = asyncio.create_task(coro)
            try:
                while not task.done():
                    now = time.monotonic()
                    cancelled = cancel is not None and cancel.cancelled
                    wall = limits.wall_clock_s is not None and now - started >= limits.wall_clock_s
                    idle_reference = max(last_event, last_tool_activity_end or 0.0)
                    idle = (
                        transport and active_host_tools == 0 and saw_event and limits.idle_s is not None
                        and now - idle_reference >= limits.idle_s
                    )
                    first = (
                        transport
                        and not saw_event
                        and limits.first_event_s is not None
                        and now - last_event >= limits.first_event_s
                    )
                    if cancelled or wall or idle or first:
                        task.cancel()
                        # CLI drive delivers a terminal result while unwinding.
                        try:
                            result = await task
                        except (asyncio.CancelledError, Cancelled):
                            result = Turn()
                        result.failure = Failure(
                            FailureClass.CANCELLED if cancelled else FailureClass.TIMEOUT,
                            False,
                            mode_detail={
                                "limit": "wall" if wall else "idle" if idle else "first_event",
                                "dispatch_state": "unknown",
                            },
                        )
                        return result
                    await asyncio.wait({task}, timeout=0.05)
                return await task
            except Cancelled:
                # A hosted tool can observe the run token before our monitor
                # poll. Its joined cooperative abort has the same terminal
                # meaning as token cancellation observed in this loop.
                return Turn(failure=Failure(FailureClass.CANCELLED, False))
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Cancelled):
                        pass

        try:
            # Two local refusals, both with zero provider attempts, so the
            # order is about cost rather than safety. `schema_failure` is
            # already computed and certain; evaluating execution compatibility
            # inspects the backend, which can spawn the binary to read its
            # version. A request whose schema cannot compile will never run, so
            # answering that first avoids asking a question that no longer
            # matters — and the compatibility evaluation is skipped entirely
            # rather than performed and discarded.
            compatibility = None
            if schema_failure is None and (
                execution is not None or hasattr(adapter, "evaluate_execution")
                or effective_native_policy == NativeToolPolicy.DENY
            ):
                compatibility = await self.evaluate_execution(
                    binding, execution=execution, native_tool_policy=effective_native_policy,
                )
            if schema_failure is not None:
                failure = schema_failure
                total_cost, cost_source, usage_complete = None, CostSource.UNPRICED, False
            elif compatibility is not None and not compatibility.compatible:
                from agentkit.models.providers.cli.calls import execution_failure

                failure = execution_failure(compatibility)
                total_cost, cost_source, usage_complete = None, CostSource.UNPRICED, False
            elif prompt.images and binding.mode == Mode.CLAUDE_CODE:
                degradation = Degradation(
                    "images", "image prompt parts", "refused", "Claude Code CLI transport has no image input contract"
                )
                await emit(EventKind.DEGRADED, **asdict(degradation))
                effective = replace(effective, degradations=(*effective.degradations, degradation))
                failure = Failure(FailureClass.INVALID_REQUEST, mode_detail={"reason": degradation.reason})
            elif native_agents and binding.mode != Mode.CLAUDE_CODE:
                failure = Failure(
                    FailureClass.INVALID_REQUEST,
                    mode_detail={"reason": "native subagents are supported only by Claude Code"},
                )
            elif workspace is not None and binding.mode.is_api:
                failure = Failure(
                    FailureClass.INVALID_REQUEST,
                    mode_detail={"reason": "workspace authority is available only to CLI modes"},
                )
            elif binding.mode == Mode.CODEX and limits.max_cost_usd is not None:
                failure = Failure(
                    FailureClass.INVALID_REQUEST,
                    mode_detail={
                        "reason": (
                            "Codex CLI cannot enforce a hard mid-call cost cap; "
                            "remove max_cost_usd or use another mode"
                        )
                    },
                )
            elif binding.mode == Mode.CODEX and effective_native_policy == NativeToolPolicy.DENY:
                from agentkit.models.providers.cli.calls import execution_failure

                compatibility = await self.evaluate_execution(binding, native_tool_policy=effective_native_policy)
                failure = execution_failure(compatibility)
                total_cost, cost_source, usage_complete = None, CostSource.UNPRICED, False
            elif (
                binding.mode == Mode.CLAUDE_CODE
                and effective_native_policy == NativeToolPolicy.DENY
                and workspace is not None
                and workspace.native_tools
            ):
                failure = Failure(
                    FailureClass.INVALID_REQUEST,
                    mode_detail={"reason": "Claude native tools were explicitly granted while per-call policy is DENY"},
                )
            elif (table is None or not table.has(binding.mode, binding.model)) and (
                self.unpriced_policy == UnpricedPolicy.REFUSE
                or (limits.max_turns is None and limits.max_output_tokens is None)
            ):
                failure = Failure(
                    FailureClass.INVALID_REQUEST,
                    mode_detail={
                        "reason": "unpriced model; supply a PriceTable or enable proceed_unpriced with a turn/token cap"
                    },
                )
                total_cost, cost_source = None, CostSource.UNPRICED
            else:
                while True:
                    if cancel and cancel.cancelled:
                        outcome, failure = Outcome.CANCELLED, Failure(FailureClass.CANCELLED)
                        break
                    if limits.max_turns is not None and turns >= limits.max_turns:
                        outcome = Outcome.LIMIT_REACHED
                        break
                    if working.limit is not None and await working.tokens() > working.limit:
                        outcome = Outcome.LIMIT_REACHED
                        failure = Failure(FailureClass.INVALID_REQUEST, mode_detail={"limit": "context"})
                        break
                    attempts += 1
                    last_event, saw_event = time.monotonic(), False
                    turn_limits = limits
                    if limits.max_cost_usd is not None:
                        if budget_unknown:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        remaining_cost = limits.max_cost_usd - budget_spent
                        if remaining_cost <= 0:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        turn_limits = replace(turn_limits, max_cost_usd=remaining_cost)
                    if limits.max_output_tokens is not None:
                        remaining = limits.max_output_tokens - usage.output
                        if remaining <= 0:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        turn_limits = replace(turn_limits, max_output_tokens=remaining)
                    turn_settings = effective.settings
                    if turn_limits.max_output_tokens is not None:
                        cap = min(
                            turn_settings.max_output_tokens or turn_limits.max_output_tokens,
                            turn_limits.max_output_tokens,
                        )
                        # CLI transports have no output-token setting; their
                        # usage is independently checked as soon as it arrives.
                        if binding.mode.is_api:
                            turn_settings = replace(turn_settings, max_output_tokens=cap)
                    attempt_session = current_session
                    assessment_kwargs = (
                        {"assessment_id": compatibility.configuration_id,
                         **({"attempt_id": f"{call_id}:{attempts}"} if not binding.mode.is_api else {})}
                        if compatibility is not None and compatibility.configuration_id else {}
                    )
                    model_in_flight = True
                    await emit(EventKind.ATTEMPT_STARTED, attempt=attempts, mode=binding.mode, model=binding.model)

                    async def invoke_adapter(
                        prompt_arg: ComposedPrompt = prompt,
                        settings_arg: ModelSettings = turn_settings,
                        limits_arg: CallLimits = turn_limits,
                        session_arg: SessionHandle | None = current_session,
                        agents_arg: Mapping[str, Any] | None = native_agents,
                        workspace_arg: Workspace | None = workspace,
                        slice_arg: int | None = turns_per_slice,
                        stable_prefix_arg: bool = stable_prefix,
                        fork_arg: bool = fork,
                        assessed_kwargs: dict[str, Any] = assessment_kwargs,
                    ) -> Turn:
                        if slice_arg is not None and binding.mode == Mode.CLAUDE_CODE:
                            limits_arg = replace(
                                limits_arg,
                                max_turns=min(limits_arg.max_turns or slice_arg, slice_arg),
                            )
                        if binding.mode.is_api:
                            semaphore = self._api_semaphores.setdefault(
                                binding.mode, asyncio.Semaphore(self.max_concurrent_api)
                            )
                            async with semaphore:
                                return cast(
                                    Turn,
                                    await cast(Any, adapter).turn(
                                        binding,
                                        prompt_arg,
                                        settings=settings_arg,
                                        limits=limits_arg,
                                        schema=schema,
                                        tools=tools,
                                        session=session_arg,
                                        cancel=cancel,
                                        emit=mode_event,
                                        call_id=call_id,
                                        native_agents=agents_arg,
                                        workspace=workspace_arg,
                                        **assessed_kwargs,
                                        # Stated requirements travel with every
                                        # turn so they bind through retries and
                                        # corrections, not only the entry gate.
                                        # Omitted when unstated, so an adapter
                                        # predating the parameter still runs.
                                        **(
                                            {
                                                "execution": execution,
                                                "native_tool_policy": effective_native_policy,
                                            }
                                            if execution is not None
                                            else {}
                                        ),
                                    ),
                                )
                        semaphore = self._cli_semaphores.setdefault(
                            binding.mode, asyncio.Semaphore(self.max_concurrent_cli)
                        )
                        async with semaphore:
                            if self.acquire is None:
                                return cast(
                                    Turn,
                                    await cast(Any, adapter).turn(
                                        binding,
                                        prompt_arg,
                                        settings=settings_arg,
                                        limits=limits_arg,
                                        schema=schema,
                                        tools=tools,
                                        session=session_arg,
                                        cancel=cancel,
                                        emit=mode_event,
                                        call_id=call_id,
                                        native_agents=agents_arg,
                                        workspace=workspace_arg,
                                        scope=scope,
                                        stable_prefix=stable_prefix_arg,
                                        fork=fork_arg,
                                        **assessed_kwargs,
                                        **(
                                            {
                                                "native_tool_policy": effective_native_policy,
                                                **({"execution": execution} if execution is not None else {}),
                                                "tool_activity": tool_activity,
                                                "tool_activity_ended_at": tool_activity_ended_at,
                                                **(
                                                    {"context_policy": context_policy}
                                                    if context_policy is not None
                                                    else {}
                                                ),
                                            }
                                            if not binding.mode.is_api
                                            else {}
                                        ),
                                    ),
                                )
                            async with self.acquire(binding.mode):
                                return cast(
                                    Turn,
                                    await cast(Any, adapter).turn(
                                        binding,
                                        prompt_arg,
                                        settings=settings_arg,
                                        limits=limits_arg,
                                        schema=schema,
                                        tools=tools,
                                        session=session_arg,
                                        cancel=cancel,
                                        emit=mode_event,
                                        call_id=call_id,
                                        native_agents=agents_arg,
                                        workspace=workspace_arg,
                                        scope=scope,
                                        stable_prefix=stable_prefix_arg,
                                        fork=fork_arg,
                                        **assessed_kwargs,
                                        **(
                                            {
                                                "native_tool_policy": effective_native_policy,
                                                **({"execution": execution} if execution is not None else {}),
                                                "tool_activity": tool_activity,
                                                "tool_activity_ended_at": tool_activity_ended_at,
                                                **(
                                                    {"context_policy": context_policy}
                                                    if context_policy is not None
                                                    else {}
                                                ),
                                            }
                                            if not binding.mode.is_api
                                            else {}
                                        ),
                                    ),
                                )

                    turn = await monitored(invoke_adapter())
                    model_in_flight = False
                    usage_complete = usage_complete and turn.usage_complete
                    normalized = normalize_usage(binding.mode, turn.raw_usage)
                    if turn.usage_cumulative and current_session:
                        normalized = usage_delta(normalized, normalize_usage(binding.mode, current_session.raw_usage))
                    usage += normalized
                    raw = turn.raw_usage
                    amount, source = price(
                        binding.model,
                        normalized,
                        table,
                        mode=binding.mode,
                        allow_partial=self.unpriced_policy == UnpricedPolicy.PROCEED_UNPRICED,
                    )
                    if turn.reported_cost is not None:
                        reported = turn.reported_cost
                        previous = current_session.reported_cost if current_session else None
                        if turn.reported_cost_cumulative and previous is not None and reported >= previous:
                            reported -= previous
                        amount = reported
                        source = CostSource.MIXED if not turn.usage_complete else CostSource.REPORTED
                    incomplete_accounting = (not turn.usage_complete and turn.reported_cost is None) or (
                        not turn.raw_usage and turn.reported_cost is None
                    )
                    if incomplete_accounting:
                        amount, source = None, CostSource.UNPRICED
                    rejected_failure_without_usage_or_cost = (
                        turn.failure is not None
                        and turn.reported_cost is None
                        and not any(bool(value) for value in turn.raw_usage.values())
                        and not any(message.strip() for message in turn.messages)
                        and not turn.tool_calls
                        and turn.failure.classification in (FailureClass.RATE_LIMITED, FailureClass.AUTH)
                    )
                    if rejected_failure_without_usage_or_cost:
                        # A pre-output provider rejection has done no billable
                        # work. Keep its failure classification and monetary
                        # ledger intact; token completeness remains separate.
                        amount, source = Decimal(0), cost_source
                    if amount is None:
                        total_cost, cost_source = None, CostSource.UNPRICED
                        if limits.max_cost_usd is not None and not rejected_failure_without_usage_or_cost:
                            budget_unknown = True
                    elif total_cost is not None:
                        total_cost += amount
                        cost_source = CostSource.MIXED if CostSource.MIXED in (cost_source, source) else source
                    if amount is not None:
                        budget_spent += amount
                    await emit(EventKind.USAGE, raw_usage=raw, usage=asdict(usage))
                    await emit(
                        EventKind.ATTEMPT_FINISHED,
                        attempt=attempts,
                        mode=binding.mode,
                        model=binding.model,
                        outcome="failed" if turn.failure else "completed",
                        failure_class=turn.failure.classification if turn.failure else None,
                        usage=asdict(normalized),
                        cost_usd=str(amount) if amount is not None else None,
                        cost_source=source,
                    )
                    messages.extend(turn.messages)
                    for index, message in enumerate(turn.messages, len(messages) - len(turn.messages)):
                        fields: dict[str, Any] = {
                            "index": index,
                            "length": len(message),
                            "final": not turn.tool_calls,
                        }
                        if capture_text_for_call:
                            try:
                                assert self.result_masker is not None
                                masked_text = self.result_masker("assistant", message)
                                if not isinstance(masked_text, str):
                                    raise TypeError("result_masker must return text")
                                encoded = masked_text.encode("utf-8")
                                fields["text"] = encoded[:4096].decode("utf-8", errors="ignore")
                                if len(encoded) > 4096:
                                    fields["text_truncated"] = True
                            except Exception as exc:
                                fields["text_omitted"] = True
                                await emit(
                                    EventKind.DEGRADED,
                                    setting="capture_text",
                                    requested="masked narration",
                                    applied="length only",
                                    reason=f"host masker failed ({type(exc).__name__})",
                                )
                        await emit(EventKind.MESSAGE_DONE, **fields)
                    text = turn.messages[-1] if turn.messages else ""
                    if not binding.mode.is_api:
                        working.extend(turn.canonical_messages)
                    if not turn.failure:
                        portable = (
                            canonical_turn(turn.history, binding.mode)
                            if binding.mode.is_api
                            else tuple(
                                Message(MessageRole.ASSISTANT, message, metadata={"call_id": call_id})
                                for message in turn.messages
                            )
                        )
                        if binding.mode.is_api and turn.tool_calls and not any(m.tool_calls for m in portable):
                            calls = tuple(
                                ToolCall(c["id"], c["name"], c["arguments"])
                                for c in turn.tool_calls
                                if c["name"] != "__pause_turn__"
                            )
                            if calls:
                                portable = (*portable, Message(MessageRole.ASSISTANT, tool_calls=calls))
                        if not portable and turn.messages:
                            portable = tuple(Message(MessageRole.ASSISTANT, message) for message in turn.messages)
                        working.extend(portable)
                    current_session = SessionHandle(
                        binding.mode,
                        call_id,
                        turn.session_id or (current_session.cli_session_id if current_session else None),
                        turn.history
                        or (
                            current_session.history
                            if turn.failure and current_session
                            else history_for((*working.prefix.grounding, *working.messages), binding.mode)
                            if binding.mode.is_api
                            else ()
                        ),
                        prefix,
                        raw or (current_session.raw_usage if current_session else {}),
                        messages=tuple(working.messages),
                        model=binding.model,
                        home_bundle_ref=turn.home_bundle_ref
                        or (current_session.home_bundle_ref if current_session else None),
                        prefix=working.prefix,
                        reported_cost=(
                            turn.reported_cost
                            if turn.reported_cost is not None
                            else current_session.reported_cost
                            if current_session
                            else None
                        ),
                        execution=execution,
                        native_tool_policy=effective_native_policy,
                    )
                    if turn.failure:
                        retry_streak += 1
                        failure = turn.failure
                        if failure.classification == FailureClass.CANCELLED:
                            outcome = Outcome.CANCELLED
                            break
                        if failure.classification == FailureClass.TIMEOUT and not failure.retryable:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        if limits.max_cost_usd is not None and budget_unknown:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        if limits.max_cost_usd is not None and budget_spent >= limits.max_cost_usd:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        retry_allowed = (
                            failure.retryable
                            and retry_streak < self.retry.max_attempts
                            and time.monotonic() - started < self.retry.max_elapsed_s
                        )
                        if retry_allowed and failure.classification == FailureClass.AUTH:
                            prepare = getattr(adapter, "prepare_retry", None)
                            retry_allowed = bool(
                                prepare and retry_streak == 1 and await prepare(binding, failure, call_id=call_id)
                            )
                        if not retry_allowed:
                            break
                        wait = max(
                            failure.retry_after_s or 0,
                            random.uniform(0, min(self.retry.cap_s, self.retry.base_s * 2 ** (retry_streak - 1))),
                        )
                        if wait > self.retry.cap_s or time.monotonic() - started + wait > self.retry.max_elapsed_s:
                            break
                        await emit(
                            EventKind.RETRY,
                            failure_class=failure.classification,
                            wait_seconds=wait,
                            source=(
                                "transport"
                                if not binding.mode.is_api
                                else "retry-after"
                                if failure.retry_after_s is not None
                                else "backoff"
                            ),
                            attempt=attempts,
                        )
                        # Backoff is cancellable and shares the wall deadline.
                        sleep_result = await monitored(_sleep_turn(wait), transport=False)
                        if sleep_result.failure:
                            failure = sleep_result.failure
                            outcome = (
                                Outcome.CANCELLED
                                if failure.classification == FailureClass.CANCELLED
                                else Outcome.LIMIT_REACHED
                            )
                            break
                        if not binding.mode.is_api:
                            # Retry the same provider conversation after a
                            # failed process. The failed attempt may have
                            # emitted a different/empty handle; only a
                            # successful turn advances the resumable session.
                            # Keep cumulative accounting baselines from the
                            # failed attempt so resumed usage/cost is charged
                            # exactly once.
                            failed_session = current_session
                            current_session = (
                                replace(
                                    attempt_session,
                                    raw_usage=failed_session.raw_usage,
                                    reported_cost=failed_session.reported_cost,
                                )
                                if attempt_session is not None and failed_session is not None
                                else attempt_session
                            )
                        continue
                    turns += 1
                    failure = None
                    retry_streak = 0
                    if turn.limit_reached:
                        outcome = Outcome.LIMIT_REACHED
                        break
                    if turn.refused:
                        outcome, failure = Outcome.REFUSED, Failure(FailureClass.REFUSED)
                        break
                    if turn.truncated:
                        outcome = Outcome.TRUNCATED
                        break
                    if amount is None and not incomplete_accounting:
                        if self.unpriced_policy == UnpricedPolicy.REFUSE:
                            failure = Failure(
                                FailureClass.INVALID_REQUEST,
                                mode_detail={"reason": "usage contains unpriced token kinds"},
                            )
                            break
                        await emit(
                            EventKind.DEGRADED,
                            setting="accounting",
                            requested="fully priced usage",
                            applied="answer preserved with unknown total cost",
                            reason="provider reported a token category with no configured price",
                        )
                        if limits.max_cost_usd is not None and turn.tool_calls:
                            outcome = Outcome.LIMIT_REACHED
                            break
                    if (
                        limits.max_cost_usd is not None and (budget_unknown or budget_spent >= limits.max_cost_usd)
                    ) or (limits.max_output_tokens is not None and usage.output > limits.max_output_tokens):
                        outcome = Outcome.LIMIT_REACHED
                        break
                    if turn.tool_calls:
                        if tools is None:
                            failure = Failure(FailureClass.TOOL_ERROR, mode_detail={"reason": "tools are unavailable"})
                            break
                        tool_ctx = _ToolContext(scope=scope, correlation_id=call_id, cancel=cancel,
                                                attempt_id=f"{call_id}:{attempts}")
                        history = list(current_session.history)
                        anthropic_results = []
                        dispatchable = [call for call in turn.tool_calls if call["name"] != "__pause_turn__"]
                        for call in dispatchable:
                            await emit(
                                EventKind.TOOL_CALL,
                                name=call["name"],
                                tool_call_id=call["id"],
                                argument_digest=_keyed_digest(self._event_digest_key, call["arguments"]),
                                arguments=self.argument_masker(call["name"], call["arguments"]),
                            )

                        semaphore = asyncio.Semaphore(max(1, int(getattr(tools, "max_concurrency", 4))))

                        registry = getattr(tools, "registry", None)
                        def is_side_effect_free(
                            call: Mapping[str, Any], *, registry: Any = registry,
                        ) -> bool:
                            if registry is None:
                                return False
                            try:
                                tool = registry.get(str(call["name"]))
                            except (KeyError, TypeError):
                                # Deferred dispatchers and tools supplied by an adapter are
                                # conservatively serialized because their effects are unknown.
                                return False
                            return not bool(getattr(tool, "side_effecting", False))

                        async def execute_tool(
                            call: Mapping[str, Any], *, tool_semaphore: asyncio.Semaphore = semaphore,
                            run_ctx: _ToolContext = tool_ctx,
                        ) -> Turn:
                            async with tool_semaphore:
                                if cancel is not None and cancel.cancelled:
                                    return Turn(failure=Failure(
                                        FailureClass.CANCELLED,
                                        mode_detail={"reason": "tool was not started before cancellation"},
                                    ))
                                started_tools.add(call["id"])
                                from agentkit.tools.execution.context import bind_tool_invocation

                                with bind_tool_invocation(run_ctx, provider_id=call["id"]) as invocation_ctx:
                                    return await monitored(
                                        _tool_turn(tools, call["name"], call["arguments"], invocation_ctx),
                                        transport=False,
                                    )

                        if all(is_side_effect_free(call) for call in dispatchable):
                            tool_turns = await asyncio.gather(*(execute_tool(call) for call in dispatchable))
                        else:
                            tool_turns = []
                            for call in dispatchable:
                                tool_turns.append(await execute_tool(call))
                        tool_failed = False
                        for call, tool_turn in zip(dispatchable, tool_turns, strict=True):
                            if call["name"] == "__pause_turn__":
                                continue
                            if tool_turn.failure:
                                if failure is None:
                                    failure = tool_turn.failure
                                    outcome = (
                                        Outcome.CANCELLED
                                        if failure.classification == FailureClass.CANCELLED
                                        else Outcome.LIMIT_REACHED
                                    )
                                tool_failed = True
                                continue
                            result = tool_turn.value
                            preview_fields = {}
                            if self.result_masker is not None:
                                preview = self.result_masker(call["name"], result.text)
                                preview_fields["result_preview"] = preview.encode("utf-8")[:512].decode(
                                    "utf-8", errors="ignore"
                                )
                            await emit(
                                EventKind.TOOL_RESULT,
                                name=call["name"],
                                tool_call_id=call["id"],
                                result_size=len(result.text),
                                result_digest=digest(result.content),
                                overflowed=result.overflowed,
                                is_error=result.is_error,
                                **preview_fields,
                            )
                            working.append(
                                Message(
                                    MessageRole.TOOL,
                                    result.text,
                                    name=call["name"],
                                    tool_call_id=call["id"],
                                    content_blocks=tuple(dict(x) for x in result.content),
                                    metadata={"call_id": call_id, "is_error": result.is_error},
                                )
                            )
                            if binding.mode == Mode.ANTHROPIC_API:
                                anthropic_results.append(
                                    {
                                        "type": EventKind.TOOL_RESULT,
                                        "tool_use_id": call["id"],
                                        "content": tool_content(result.content, Mode.ANTHROPIC_API),
                                        "is_error": result.is_error,
                                    }
                                )
                            else:
                                history.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": call["id"],
                                        "output": tool_content(result.content, Mode.OPENAI_API),
                                    }
                                )
                        if tool_failed:
                            break
                        if failure:
                            break
                        if anthropic_results:
                            history.append({"role": "user", "content": anthropic_results})
                        await apply_context_policy()
                        current_session = replace(
                            current_session, history=tuple(history), messages=tuple(working.messages)
                        )
                        if turns_per_slice is not None and turns >= turns_per_slice:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        continue
                    if output:
                        try:
                            if turn.value is not None:
                                raw_value = turn.value
                            else:
                                try:
                                    raw_value = json.loads(text)
                                except json.JSONDecodeError:
                                    raw_value = tolerant_json_object(text)
                            value = validate_output(raw_value, output.schema)
                            if schema_adapter:
                                value = schema_adapter.validate(value)
                            validator_errors = [error for validator in output.validators for error in validator(value)]
                            if validator_errors:
                                raise ValueError("; ".join(str(error) for error in validator_errors)[:2000])
                        except Exception as exc:
                            # A single correction turn on every transport. Do
                            # not repair refusal/truncation or terminal failures.
                            if corrections >= output.max_corrections:
                                failure = Failure(
                                    FailureClass.INVALID_REQUEST,
                                    mode_detail={"reason": "canonical output validation failed"},
                                )
                                break
                            corrections += 1
                            correction = (
                                "Return the requested JSON object. Correct these validation errors: " + str(exc)[:2000]
                            )
                            await emit(
                                EventKind.RETRY,
                                failure_class="invalid_output",
                                wait_seconds=0,
                                source="correction",
                                attempt=attempts,
                            )
                            if current_session.history:
                                current_session = replace(
                                    current_session,
                                    history=(*current_session.history, {"role": "user", "content": correction}),
                                )
                            working.append(Message(MessageRole.USER, correction, metadata={"call_id": call_id}))
                            prompt = replace(prompt, user=correction)
                            continue
                    if not text.strip() and turn.value is None:
                        failure = Failure(
                            FailureClass.INVALID_REQUEST, mode_detail={"reason": "transport produced no answer"}
                        )
                        break
                    outcome = Outcome.ANSWERED
                    break
        except asyncio.CancelledError:
            if model_in_flight:
                total_cost, cost_source = None, CostSource.UNPRICED
            await emit(
                EventKind.CALL_FINISHED,
                outcome=Outcome.CANCELLED,
                usage=asdict(usage),
                cost_usd=str(total_cost) if total_cost is not None else None,
                cost_source=cost_source,
                duration_s=time.monotonic() - started,
            )
            raise
        except Exception as exc:
            failure = classify(exc)
        if current_session is not None:
            if binding.mode.is_api:
                repaired_history, interrupted_results = close_pending_tools(
                    current_session.history,
                    working.messages,
                    binding.mode,
                    started_tools,
                )
                working.extend(interrupted_results)
                current_session = replace(current_session, history=repaired_history)
                if interrupted_results:
                    degradation = Degradation(
                        "interrupted_tools",
                        "pending tool calls",
                        "paired error results",
                        "continuation retains explicit interrupted or unexecuted tool evidence",
                    )
                    effective = replace(effective, degradations=(*effective.degradations, degradation))
                    await emit(EventKind.DEGRADED, **asdict(degradation))
            current_session = replace(current_session, messages=tuple(working.messages), prefix=working.prefix)
        if outcome in (Outcome.FAILED, Outcome.CANCELLED, Outcome.REFUSED):
            text, value = "", None
        answer = Answer(
            value,
            text,
            tuple(messages),
            outcome,
            usage,
            raw,
            total_cost,
            cost_source,
            effective,
            attempts,
            failure,
            current_session,
            prompt.manifest,
            context=working.freeze(),
            usage_complete=usage_complete,
        )
        await emit(
            EventKind.CALL_FINISHED,
            outcome=outcome,
            usage=asdict(usage),
            cost_usd=str(total_cost) if total_cost is not None else None,
            cost_source=cost_source,
            duration_s=time.monotonic() - started,
        )
        if event_sink_failure is not None:
            degradation = Degradation(
                "event_sink",
                "enabled",
                "disabled for this call",
                f"subscriber raised {event_sink_failure}; answer was preserved",
            )
            answer = replace(answer, effective=replace(
                answer.effective, degradations=(*answer.effective.degradations, degradation)
            ))
        return answer


def _mask_tool_arguments(name: str, arguments: Any, *, key: bytes | None = None) -> Any:
    """Digest every leaf and key before tool arguments enter the event journal."""
    if isinstance(arguments, Mapping):
        return {
            f"key:{_masked_event_scalar(key_name, digest_key=key)['sha256']}": _mask_tool_arguments(
                name, value, key=key
            )
            for key_name, value in arguments.items()
        }
    if isinstance(arguments, (tuple, list)):
        return tuple(_mask_tool_arguments(name, item, key=key) for item in arguments)
    return _masked_event_scalar(arguments, digest_key=key)


def _keyed_digest(key: bytes | None, value: Any) -> str:
    encoded = repr(value).encode("utf-8", errors="replace")
    if key is None:
        return digest(value)
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _masked_event_scalar(value: Any, *, digest_key: bytes | None = None) -> dict[str, Any]:
    kind = type(value).__name__
    encoded = repr(value).encode("utf-8", errors="replace")
    return {
        "type": kind,
        "length": len(encoded),
        "sha256": _keyed_digest(digest_key, {"type": kind, "value": repr(value)}),
    }


async def _tool_turn(tools: Any, name: str, arguments: Any, ctx: Any) -> Turn:
    return Turn(value=await tools.execute(name, arguments, ctx))


async def _sleep_turn(seconds: float) -> Turn:
    await asyncio.sleep(seconds)
    return Turn()


async def ask(binding: Binding, prompt: ComposedPrompt | str, output: Any = None, **kwargs: Any) -> Answer[Any]:
    """Convenience one-shot call; use CallRuntime for long-lived sessions."""
    async with CallRuntime() as runtime:
        return await runtime.ask(binding, prompt, output, **kwargs)


async def run(binding: Binding, prompt: ComposedPrompt | str, tools: Any = None, **kwargs: Any) -> Answer[Any]:
    """Convenience agent loop with shared tool handling."""
    async with CallRuntime() as runtime:
        return await runtime.run(binding, prompt, tools, **kwargs)
