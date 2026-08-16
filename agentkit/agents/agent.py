"""Agent — identity + chat-call config + one Cognition Strategy.

The Agent class is the **Context** in the Strategy pattern: it holds
identity (name) and the chat-call configuration every LLM-using
cognition needs to share (model / prompt / temperature / etc.). The
**Cognition** plug-in holds the variation: tool registry, iteration
ceiling, children roster, policy, termination, checkpointer. Adding
a new turn-taking regime is implementing the ``Cognition`` Protocol;
no Agent subclassing.

Three first-party cognitions cover today's regimes:

- ``SingleCallCognition``  — one chat call + parse-and-repair (default).
- ``ReActCognition(tools=…)`` — tool-loop with HITL suspend + durable resume.
- ``CoordinatorCognition(children=…, policy=…)`` — multi-agent loop.

Prompt wiring still goes through a ``RequestBuilder`` (the seam where
prompt/grounding/compaction meet). A plain ``prompt: str`` is wrapped
into a one-off ``Prompt`` for ergonomics; an explicit
``request_builder=`` overrides everything. Optional typed output via
``parse``: any ``Callable[[str], T]`` that returns the validated
object or raises on invalid; ``max_repairs`` controls the
reflect-and-retry budget.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentkit.agents.control.safety import RunPolicy

from agentkit.agents._agent_helpers import (
    _chain_has_output_coerce,
    _derived_capabilities,
    _infer_response_format,
    _stamp_terminal_span_attrs,
    _warn_missing_output_coerce,
)
from agentkit.agents.cognition import (
    Cognition,
    ReActCognition,
    SingleCallCognition,
)
from agentkit.agents.control.elicitation import Decision
from agentkit.agents.result import AgentResult
from agentkit.capabilities.output_schema import SchemaAdapter, adapt
from agentkit.capabilities.request_builder import RequestBuilder
from agentkit.context import WorkingContext
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import StreamEvent
from agentkit.prompts.prompt import Prompt


@dataclass
class Agent:
    """An agent with one Cognition Strategy.

    Identity + chat-call configuration live here. The cognition owns
    the turn-taking regime + its own configuration (tools, children,
    iteration ceiling, termination, checkpointer, …).

    Required:
        name

    Required (any chat-using cognition):
        model

    Prompt wiring (one of):
        prompt: a ``Prompt`` (versioned) OR a plain ``str`` (wrapped into a one-off
            ``Prompt`` with version="inline" so traces still attribute clearly).
            If neither ``prompt`` nor ``request_builder`` is given, an empty inline
            ``Prompt`` is used — this keeps the ergonomic ``Agent("a", "m")`` form
            working in tests and small examples without silently producing an
            un-traced run.
        request_builder: a fully wired ``RequestBuilder`` (prompt + grounding +
            compaction). When set, this overrides ``prompt`` entirely.

    Output shaping:
        response_format, max_tokens, temperature: passed through on each ChatRequest.
        output: type | dict — Pydantic BaseModel, attrs class, dataclass, or raw
            JSON Schema dict. The adapter is built once in ``__post_init__``;
            its JSON Schema lands in the cache-stable prefix via the RequestBuilder.
        parse: ``Callable[[str], T]`` that returns the typed object or raises on
            invalid. ``AgentResult.parsed`` carries the typed object. On invalid
            output, the error is reflected back to the model up to
            ``max_repairs`` times.
        max_repairs: reflect-and-retry budget for parse failures.

    Cognition (the Strategy plug-in):
        cognition: the turn-taking strategy. Defaults to ``SingleCallCognition()``.
            Pass ``ReActCognition(tools=…)`` for a tool loop, or
            ``CoordinatorCognition(children=…, policy=…)`` for multi-agent
            orchestration. Any ``Cognition`` Protocol impl works.

    Memory (the external-reach seam):
        memory: a ``MemorySource`` the cognition queries to ground reasoning.
            Composable: pass a ``CompositeMemory`` to fan out across vector
            + journal + tool-wrapped sources, or a single ``VectorMemory``
            for plain RAG. ``None`` (default) disables the memory hook.
            Results are stamped onto ``working_context.scratchpad["memory"]``
            and the cognition decides how to fold them into the prompt
            (typically via the ``RequestBuilder``'s grounding hook).
    """

    name: str
    model: str | None = None
    prompt: Prompt | str | None = None
    request_builder: RequestBuilder | None = None
    response_format: dict[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    output: type | dict[str, Any] | None = None
    parse: Callable[[str], Any] | None = None
    max_repairs: int = 1
    cognition: Cognition = field(default_factory=SingleCallCognition)
    memory: Any = None  # MemorySource | None — declared Any to avoid import gravity
    # Pre-run trifecta gate. When set, ``RunPolicy.check`` fires ONCE before the
    # first cognition drive so a deny-mode policy short-circuits the run before
    # any LLM call. ``None`` (default) opts out entirely.
    policy: RunPolicy | None = None
    # Per-agent SignalChannel — the coordination seam a child uses to fan
    # ``DataSignal``s (progress/done/escalate) upward to a coordinator's merge
    # inbox. Opt-in: when unset the coordinator dispatch path skips
    # ``attach_parent`` and behavior is exactly as before. When both this
    # agent's ``channel`` AND its parent ctx's ``signal_channel`` are set,
    # ``run_agents`` wires the two by calling
    # ``channel.attach_parent(parent_ctx.signal_channel.merge_inbox)``.
    # Declared ``Any`` to avoid an import-time dependency on
    # ``agents.control.channel`` (which would drag the observation/signal
    # tree into every ``Agent`` import).
    channel: Any = field(default=None)
    # ---- model capability contract (Brief 5) ------------------------------------------
    # Capabilities this agent's job REQUIRES of whatever model it is bound to —
    # names from ``adapters.llm.model_registry.CAPABILITY_NAMES``, e.g.
    # ``requires=("vision", "tools")``. Checked in ``__post_init__``, i.e. at
    # construction, BEFORE any spend: the whole value of the check is catching
    # the mismatch before money moves, so doing it on the call would be
    # pointless. A capability the model declares as unsupported raises
    # ``CapabilityMismatch``; one it doesn't declare at all is governed by
    # ``on_unknown`` below. A tool-using cognition implies ``"tools"``
    # automatically — see ``_derived_capabilities``.
    requires: tuple[str, ...] = ()
    # Minimum context window in tokens, when the job needs one. ``None`` skips
    # the check entirely.
    min_context_window: int | None = None
    # What to do when the registry has never heard of this model, so its
    # capabilities are UNKNOWN rather than declared:
    #   "warn"   (default) — say so, once, and continue. Existing wiring with
    #                        self-hosted / fine-tuned / brand-new model names
    #                        must keep working; refusing by default would
    #                        break all of it on upgrade.
    #   "refuse" — raise. Correct for a production service that pins models:
    #              turns "we don't know" into a deployment-time stop.
    #   "allow"  — silence, for a caller who verified out of band.
    # UNKNOWN is NEVER treated as present under any setting.
    on_unknown_capability: str = "warn"
    # Set in __post_init__ from `output=`; threaded through the chat-using cognitions'
    # parse-and-repair loops. None when no output schema was declared — every
    # output-related code path then short-circuits to the unstructured behaviour.
    _output_adapter: SchemaAdapter[Any] | None = field(default=None, init=False, repr=False)
    # One-shot latch for the "output schema declared but output_coerce() is
    # missing from the chain" warning. Per-instance rather than module-global
    # so two differently-wired agents in one process each get told once, and
    # so a test can construct a fresh Agent to observe the warning again.
    _warned_no_coerce: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Structured output: build the adapter once. adapt() raises TypeError /
        # NotImplementedError at build-time for unsupported shapes, so the
        # caller sees the error here rather than mid-run.
        self._output_adapter = adapt(self.output) if self.output is not None else None
        # When both `output=` and `parse=` are declared the explicit parse wins —
        # honour the caller's intent. Otherwise derive the parse callable from
        # the adapter so the parse-and-repair loop fires unchanged.
        if self._output_adapter is not None and self.parse is None:
            self.parse = self._output_adapter.parse
        # Provider-native structured-output wiring. We set `response_format`
        # opportunistically when the model name signals a provider that knows
        # how to consume the dict; for everything else the prompt-injection
        # path via the schema_block is the single source of truth. Honour an
        # explicit `response_format=` over our inference — the caller knows
        # their provider better than the heuristic does.
        if self._output_adapter is not None and self.response_format is None:
            self.response_format = _infer_response_format(self.model, self._output_adapter)
        # Capability contract. Runs LAST in __post_init__ so the cognition and
        # output adapter are already settled and the derived requirements
        # (``tools`` from a tool-holding cognition) are accurate. Refusing here
        # means a mismatch is a startup error, not a plausible empty answer
        # discovered after a bill.
        self.check_capabilities()

    # ---- public surfaces -------------------------------------------------------------------

    def check_capabilities(self) -> None:
        """Refuse this agent's model if it lacks a capability the agent needs.

        Called automatically at the end of ``__post_init__`` — construction IS
        the bind, because the model is a field on the Agent. Exposed publicly
        so a caller who mutates ``agent.model`` after construction (or swaps a
        cognition in) can re-assert the contract:

            agent.model = "deepseek-reasoner"
            agent.check_capabilities()        # raises: reasoner has tools=NO

        A no-op when the agent has no model yet (a chat-less agent, or one
        whose model is injected later) — there is nothing to check against.
        Requirements come from ``requires=`` plus anything the wiring implies;
        ``on_unknown_capability`` governs the unregistered-model case.
        """
        if not self.model:
            return
        from agentkit.adapters.llm.model_registry import require_capabilities

        require_capabilities(
            self.model,
            tuple(self.requires),
            derived=_derived_capabilities(self),
            min_context_window=self.min_context_window,
            on_unknown=self.on_unknown_capability,
            subject=self.name,
        )

    async def run(
        self, task: str, ctx: Ctx, *, context: WorkingContext | None = None
    ) -> AgentResult:
        """Run the agent on ``task``. Collects the stream into the final ``AgentResult``."""
        final: AgentResult | None = None
        async for ev in self.stream(task, ctx, context):
            if ev.type == "final":
                final = ev.result
        if final is None:  # pragma: no cover
            raise RuntimeError(f"agent {self.name!r} produced no final result")
        return final

    async def stream(
        self,
        task: str,
        ctx: Ctx,
        context: WorkingContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the agent's run — ``message_delta`` tokens, tool events, and a
        terminal ``final`` event.

        The actual loop is owned by ``self.cognition``; the Agent opens the
        ``invoke_agent`` span, threads the working context, and stamps terminal
        attributes on close. ``agentkit.agent.queue_wait_ms`` measures the gap
        between ``stream()`` being called and the span opening — meaningful for
        coordinator paths where a parent dispatched us under a bounded semaphore.
        """
        dispatch_started = time.perf_counter()
        # One-shot per Agent instance: a declared output schema with no
        # output_coerce() in the chain streams no partials. Guarded by the
        # instance flag so a hot loop over ``stream()`` warns once, not
        # once per call.
        if (
            self._output_adapter is not None
            and not self._warned_no_coerce
            and not _chain_has_output_coerce(ctx)
        ):
            self._warned_no_coerce = True
            _warn_missing_output_coerce(self.name)
        with ctx.trace.span("invoke_agent", "client", **self._span_attrs(ctx)) as span:
            span.set(
                "agentkit.agent.queue_wait_ms",
                (time.perf_counter() - dispatch_started) * 1000,
            )
            await self._run_policy_gate(ctx)
            async for ev in self.cognition.drive(self, task, ctx, context or WorkingContext()):
                if ev.type == "final" and ev.result is not None:
                    _stamp_terminal_span_attrs(span, ev.result)
                yield ev

    async def _run_policy_gate(self, ctx: Ctx) -> None:
        """Fire the pre-run lethal-trifecta gate, if one is configured.

        Deny mode raises ``PermissionError`` before any LLM/tool call. Flag
        mode stamps a ``policy.flagged`` observation for the operator and the
        run continues.

        Called from BOTH ``stream()`` and ``resume()``. It used to live inline
        in ``stream()`` only, which meant a resumed run skipped the gate
        entirely: an agent whose tool set combines private-data access,
        untrusted-content ingestion, and egress would be denied on
        ``run()`` and then execute that exact tool on ``resume()``. That is
        the worst possible place for the gate to be missing — resume is the
        path a human has just approved something on, and approving one tool
        call is not approval of the capability combination.
        """
        if self.policy is None:
            return
        with ctx.trace.span("policy.check", "internal") as policy_span:
            tool_list: list[Any] = []
            cog = self.cognition
            if isinstance(cog, ReActCognition) and cog.tools is not None:
                tool_list = cog.tools.tools()
            verdict = self.policy.check(tool_list)
            policy_span.set("agentkit.policy.mode", self.policy.mode)
            policy_span.set("agentkit.policy.capabilities", ",".join(verdict.capabilities))
            policy_span.set("agentkit.policy.allowed", verdict.allowed)
            if not verdict.allowed:
                await ctx.emit(
                    "policy.flagged",
                    render="RunPolicy flagged the lethal trifecta on this tool set",
                    payload={
                        "capabilities": list(verdict.capabilities),
                        "reason": verdict.reason,
                        "mode": self.policy.mode,
                    },
                    agent=self.name,
                )

    async def resume(
        self, run_id: str, decisions: Mapping[str, str | Decision], ctx: Ctx
    ) -> AgentResult:
        """Resume a suspended tool-loop run with human decisions per pending tool call.

        Accepts a typed ``Decision`` (carrying actor + timestamp for the audit
        trail) or the legacy ``str`` form (``"approve"`` / ``"reject"`` /
        ``"deny"`` / a JSON arguments override), which is coerced. Both work
        from the same call site, so no existing caller has to change.

        Declared as ``Mapping``, not ``dict``: ``dict`` is invariant in its
        value type, so a caller holding a plain ``dict[str, str]`` — i.e.
        everyone who wrote against the old signature — would fail type-checking
        against ``dict[str, str | Decision]``. ``Mapping`` is covariant and
        accepts both.

        Only supported when ``self.cognition`` is a ``ReActCognition`` — that's
        where suspend/resume + checkpoint state live. Calling ``resume`` on any
        other cognition is a contract violation; raised explicitly so the
        caller's bug surfaces immediately.

        For a run that PARKED rather than suspended (an ``Asker`` wired on the
        ctx) there is nothing to resume — the run never unwound. See
        ``agentkit.agents.control.elicitation``.
        """
        cognition = self.cognition
        if not isinstance(cognition, ReActCognition):
            raise RuntimeError(
                f"agent {self.name!r} cognition {cognition.name!r} does not support resume "
                "— only tool-loop (ReAct) agents can be resumed"
            )
        with ctx.trace.span("invoke_agent", "client", **self._span_attrs(ctx)) as span:
            # The gate fires here too. A resumed run reaches the same tools
            # through the same registry, so skipping it would make ``resume``
            # a way around the policy. See ``_run_policy_gate``.
            await self._run_policy_gate(ctx)
            final = await cognition.resume(self, run_id, decisions, ctx)
            _stamp_terminal_span_attrs(span, final)
            return final

    # ---- request builder resolution --------------------------------------------------------

    def _resolve_request_builder(self) -> RequestBuilder:
        """Return a RequestBuilder for this run.

        Precedence:
        1. An explicitly wired ``request_builder=`` wins — caller took full
           control of prompt/grounding/compaction.
        2. Otherwise build one from ``prompt`` (str OR ``Prompt``), falling
           back to an empty inline ``Prompt`` if neither is set so
           ``Agent("a", "m")`` still works in tests.
        3. When ``memory`` is set AND no explicit request_builder was
           provided, auto-wire ``memory`` as the grounder via
           ``as_grounder(memory)``. The default rendering is
           ``[source] content`` lines — callers wanting custom formatting
           construct their own RequestBuilder + grounder explicitly.
        """
        if self.request_builder is not None:
            return self.request_builder
        if isinstance(self.prompt, Prompt):
            seed = self.prompt
        else:
            template = self.prompt if isinstance(self.prompt, str) else ""
            seed = Prompt(id=f"{self.name}.inline", version="inline", template=template)
        grounder = None
        if self.memory is not None:
            from agentkit.memory import as_grounder

            grounder = as_grounder(self.memory)
        return RequestBuilder(prompt=seed, grounder=grounder)

    # ---- span attributes (OTel GenAI v1.41 alignment) --------------------------------------

    def _span_attrs(self, ctx: Ctx | None = None) -> dict[str, Any]:
        """Attributes for the ``invoke_agent`` span at the Agent boundary.

        Names follow the OTel GenAI v1.41 operation taxonomy. ``gen_ai.agent.*`` is the
        cross-vendor surface; ``agentkit.agent.*`` carries our additions.
        ``agentkit.agent.cognition`` carries the cognition's ``name`` so a trace shows
        which turn-taking regime ran. Coordinator-specific attributes (``is_coordinator``,
        ``policy``) light up when the cognition is a ``CoordinatorCognition``; tool-loop
        attributes (``tools_count``) when ``ReActCognition``.

        Multi-tenant filter: ``agentkit.scope.{org_id,domain_id}`` stamp from ``ctx.scope``
        so operators can filter traces by tenant. Only stamped when set — the dev-default
        zero-tenant case leaves the keys absent.
        """
        # Late imports avoid circulars; both modules import Agent forward refs.
        from agentkit.agents.cognition import CoordinatorCognition

        cognition = self.cognition
        tools_count = (
            len(cognition.tools.names())
            if isinstance(cognition, ReActCognition) and cognition.tools is not None
            else 0
        )
        attrs: dict[str, Any] = {
            "gen_ai.agent.name": self.name,
            "gen_ai.agent.tools_count": tools_count,
            "gen_ai.agent.has_output_schema": self._output_adapter is not None,
            "agentkit.agent.id": id(self),
            "agentkit.agent.cognition": cognition.name,
        }
        if isinstance(cognition, CoordinatorCognition):
            attrs["agentkit.agent.is_coordinator"] = True
            attrs["agentkit.agent.policy"] = type(cognition.policy).__name__
        if ctx is not None:
            scope = getattr(ctx, "scope", None)
            if scope is not None:
                if scope.org_id is not None:
                    attrs["agentkit.scope.org_id"] = scope.org_id
                if scope.domain_id is not None:
                    attrs["agentkit.scope.domain_id"] = scope.domain_id
        return attrs


__all__ = ["Agent"]
