"""One policy loop for all four modes: ask, run and bounded continuation."""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import time
from collections.abc import Coroutine, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from agentkit.agents.prompts.composition import ComposedPrompt, digest, inline_prompt
from agentkit.core.contracts.types import Message, MessageRole, Mode, Scope, ToolCall
from agentkit.core.execution.concurrency import CancellationToken, Cancelled
from agentkit.models.adapters.messages import canonical_turn, close_pending_tools, history_for, tool_content
from agentkit.models.catalog.pricing import PriceTable, normalize_usage, price, usage_delta
from agentkit.models.contracts.constants import CostSource, EventKind, FailureClass, NativeToolPolicy, Outcome
from agentkit.models.contracts.failures import classify
from agentkit.models.contracts.schema import OutputSpec, compile_schema, validate_output
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
        adapters: Mapping[str, ModeAdapter] | None = None,
        pricing: PriceTable | None = None,
        capabilities: CapabilityTable = DEFAULT_CAPABILITIES,
        retry: RetryPolicy = _DEFAULT_RETRY,
        capture_text: bool = False,
        native_tool_policy: NativeToolPolicy = NativeToolPolicy.WARN,
    ) -> None:
        self.adapters = dict(adapters or {})
        self.pricing = pricing or PriceTable.default()
        self.capabilities = capabilities
        self.retry = retry
        self.capture_text = capture_text
        self.native_tool_policy = NativeToolPolicy(native_tool_policy)

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
        context: WorkingContext | None = None,
        parent_call_id: str | None = None,
    ) -> Answer[Any]:
        if not call_id:
            raise ValueError("call_id is required")
        if turns_per_slice is not None and turns_per_slice < 1:
            raise ValueError("turns_per_slice must be positive")
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
            Message(MessageRole.USER, prompt.user, metadata={"call_id": call_id, "agent_id": working.get("agent_id")})
        )
        effective = self.capabilities.check(binding, settings)
        schema_adapter = None
        if output is not None and not isinstance(output, OutputSpec):
            from agentkit.models.output import adapt

            schema_adapter = adapt(output)
            output = OutputSpec(schema_adapter.json_schema(), schema_adapter.name)
        schema = compile_schema(output.schema, binding.mode, binding.model) if output else None
        table = pricing or self.pricing
        adapter = self._adapter(binding)
        started = time.monotonic()
        last_event = started
        saw_event = False

        sequence = 0

        async def emit(kind: str, **fields: Any) -> None:
            nonlocal sequence
            sequence += 1
            if events is not None:
                result = events(
                    CallEvent(kind, call_id, fields, sequence, datetime.now(UTC).isoformat(), parent_call_id)
                )
                if inspect.isawaitable(result):
                    await result

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
            if event.kind == EventKind.MESSAGE_DELTA and not self.capture_text:
                fields = {"length": len(str(fields.get("text", "")))}
            await emit(event.kind, **fields)

        await emit(
            EventKind.CALL_STARTED,
            mode=binding.mode,
            model=binding.model,
            agent_id=working.get("agent_id"),
            context_digest=digest([asdict(m) for m in working.messages]),
            settings=asdict(effective.settings),
            inputs_digest=prompt.manifest.inputs_digest,
            prompt_manifest_digest=prompt.manifest.digest,
            attempt=1,
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
        raw: dict[str, Any] = {}
        total_cost: Decimal | None = Decimal(0)
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
        if (
            model_changed
            or context_changed
            or (resume is None and (len(working.messages) > 1 or working.prefix.grounding))
        ):
            current_session = SessionHandle(
                binding.mode,
                call_id,
                history=history_for((*working.prefix.grounding, *working.messages), binding.mode),
                messages=tuple(working.messages),
                model=binding.model,
                prefix=working.prefix,
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
                    idle = transport and saw_event and limits.idle_s is not None and now - last_event >= limits.idle_s
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
                            mode_detail={"limit": "wall" if wall else "idle" if idle else "first_event"},
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
            if binding.mode == Mode.CODEX and self.native_tool_policy == NativeToolPolicy.DENY:
                failure = Failure(
                    FailureClass.INVALID_REQUEST,
                    mode_detail={
                        "reason": "Codex cannot prove complete native-tool isolation; DENY policy refuses dispatch"
                    },
                )
            elif table is None or not table.has(binding.mode, binding.model):
                failure = Failure(
                    FailureClass.INVALID_REQUEST, mode_detail={"reason": "unpriced model; supply a PriceTable"}
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
                    if limits.max_output_tokens is not None:
                        remaining = limits.max_output_tokens - usage.output
                        if remaining <= 0:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        turn_limits = replace(limits, max_output_tokens=remaining)
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
                    model_in_flight = True
                    turn = await monitored(
                        adapter.turn(
                            binding,
                            prompt,
                            settings=turn_settings,
                            limits=turn_limits,
                            schema=schema,
                            tools=tools,
                            session=current_session,
                            cancel=cancel,
                            emit=mode_event,
                        )
                    )
                    model_in_flight = False
                    turns += 1
                    normalized = normalize_usage(binding.mode, turn.raw_usage)
                    if turn.usage_cumulative and current_session:
                        normalized = usage_delta(normalized, normalize_usage(binding.mode, current_session.raw_usage))
                    usage += normalized
                    raw = turn.raw_usage
                    amount, source = price(binding.model, normalized, table, mode=binding.mode)
                    if turn.reported_cost is not None:
                        reported = turn.reported_cost
                        previous = current_session.reported_cost if current_session else None
                        if turn.reported_cost_cumulative and previous is not None and reported >= previous:
                            reported -= previous
                        amount, source = reported, CostSource.REPORTED
                    incomplete_accounting = not turn.usage_complete or (
                        not turn.raw_usage and turn.reported_cost is None
                    )
                    if incomplete_accounting:
                        amount, source = None, CostSource.UNPRICED
                    if amount is None:
                        total_cost, cost_source = None, CostSource.UNPRICED
                    elif total_cost is not None:
                        total_cost += amount
                        cost_source = source
                    await emit(EventKind.USAGE, raw_usage=raw, usage=asdict(usage))
                    messages.extend(turn.messages)
                    for index, message in enumerate(turn.messages, len(messages) - len(turn.messages)):
                        await emit(EventKind.MESSAGE_DONE, index=index, length=len(message), final=not turn.tool_calls)
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
                        home_bundle=turn.home_bundle or (current_session.home_bundle if current_session else {}),
                        prefix=working.prefix,
                        reported_cost=(
                            turn.reported_cost
                            if turn.reported_cost is not None
                            else current_session.reported_cost
                            if current_session
                            else None
                        ),
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
                        if (
                            not failure.retryable
                            or retry_streak >= self.retry.max_attempts
                            or time.monotonic() - started >= self.retry.max_elapsed_s
                        ):
                            break
                        # CLI internal retries are opaque; never multiply them.
                        if not binding.mode.is_api:
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
                            source="retry-after" if failure.retry_after_s is not None else "backoff",
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
                        continue
                    failure = None
                    retry_streak = 0
                    if turn.refused:
                        outcome, failure = Outcome.REFUSED, Failure(FailureClass.REFUSED)
                        break
                    if turn.truncated:
                        outcome = Outcome.TRUNCATED
                        break
                    if amount is None and not incomplete_accounting:
                        failure = Failure(
                            FailureClass.INVALID_REQUEST, mode_detail={"reason": "usage contains unpriced token kinds"}
                        )
                        break
                    if (
                        limits.max_cost_usd is not None and (total_cost is None or total_cost >= limits.max_cost_usd)
                    ) or (limits.max_output_tokens is not None and usage.output > limits.max_output_tokens):
                        outcome = Outcome.LIMIT_REACHED
                        break
                    if turn.tool_calls:
                        if tools is None:
                            failure = Failure(FailureClass.TOOL_ERROR, mode_detail={"reason": "tools are unavailable"})
                            break
                        from agentkit.core.runtime import RunContext

                        tool_ctx = RunContext(scope=Scope(), correlation_id=call_id, cancel=cancel)
                        history = list(current_session.history)
                        anthropic_results = []
                        for call in turn.tool_calls:
                            if call["name"] == "__pause_turn__":
                                continue
                            await emit(
                                EventKind.TOOL_CALL,
                                name=call["name"],
                                tool_call_id=call["id"],
                                argument_digest=digest(call["arguments"]),
                            )
                            started_tools.add(call["id"])
                            tool_turn = await monitored(
                                _tool_turn(tools, call["name"], call["arguments"], tool_ctx), transport=False
                            )
                            if tool_turn.failure:
                                failure = tool_turn.failure
                                outcome = (
                                    Outcome.CANCELLED
                                    if failure.classification == FailureClass.CANCELLED
                                    else Outcome.LIMIT_REACHED
                                )
                                break
                            result = tool_turn.value
                            await emit(
                                EventKind.TOOL_RESULT,
                                name=call["name"],
                                tool_call_id=call["id"],
                                result_size=len(result.text),
                                overflowed=result.overflowed,
                                is_error=result.is_error,
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
                        if failure:
                            break
                        if anthropic_results:
                            history.append({"role": "user", "content": anthropic_results})
                        current_session = replace(
                            current_session, history=tuple(history), messages=tuple(working.messages)
                        )
                        if turns_per_slice is not None and turns >= turns_per_slice:
                            outcome = Outcome.LIMIT_REACHED
                            break
                        continue
                    if output:
                        try:
                            raw_value = turn.value if turn.value is not None else json.loads(text)
                            value = validate_output(raw_value, output.schema)
                            if schema_adapter:
                                value = schema_adapter.validate(value)
                        except Exception as exc:
                            # A single correction turn on every transport. Do
                            # not repair refusal/truncation or terminal failures.
                            if corrections >= 1:
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
        )
        await emit(
            EventKind.CALL_FINISHED,
            outcome=outcome,
            usage=asdict(usage),
            cost_usd=str(total_cost) if total_cost is not None else None,
            cost_source=cost_source,
            duration_s=time.monotonic() - started,
        )
        return answer


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
