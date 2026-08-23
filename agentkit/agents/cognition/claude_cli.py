"""ClaudeCliCognition — delegate the agent loop to a locally-installed ``claude`` CLI.

Zero pip dependency. Users install the ``claude`` CLI separately (from
Anthropic) and this cognition subprocesses it per ``agent.run(...)`` /
``agent.stream(...)`` with ``claude -p "<task>" --output-format stream-json
--verbose``.

Auth is entirely the CLI's problem: whatever ``CLAUDE_CODE_OAUTH_TOKEN``,
``ANTHROPIC_API_KEY``, or ``~/.claude/`` OAuth (from ``claude login``) the CLI
would find on its own is what this cognition uses. agentkit itself never
touches an API key here.

Wire it like any other cognition:

    from agentkit import Agent
    from agentkit.agents.cognition import ClaudeCliCognition

    agent = Agent(
        name="local",
        prompt="You are a concise assistant.",
        cognition=ClaudeCliCognition(
            model="claude-opus-4-5",
            permission_mode="acceptEdits",
            working_dir=Path("/tmp/sandbox"),
            tools=("Read", "Grep"),           # the session HAS only these
            allowed_tools=("Read", "Grep"),   # ...and runs them without prompting
        ),
    )
    result = await agent.run("Summarize README.md", ctx)

``tools`` and ``allowed_tools`` are different flags and mixing them up is the
easy mistake: ``--tools`` restricts what the session has, ``--allowed-tools``
auto-approves what it may run unprompted. Naming three tools in
``allowed_tools`` alone leaves every other tool — Bash included — available.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agentkit.agents.result import AgentResult, AgentStopReason, stop_reason_for
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import StreamEvent, ToolCall, Usage
from agentkit.prompts.prompt import Prompt

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.context import WorkingContext


PermissionMode = Literal[
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
    "auto",
    "manual",
    "dontAsk",
]


# ─────────────────────────────────────────────────────────────────────────────
# Class-level concurrency guard.
#
# The upstream Claude Code SDK has a known hang at ~200 parallel subprocesses
# (SDK issue #728) because the subprocess close path serializes on shared IPC.
# We bound spawn concurrency per (binary, config_dir) tuple so that two
# distinct CLI installations (or two isolated config dirs) don't share a
# single semaphore. The default (8) keeps a healthy margin under the observed
# hang threshold; callers who know their environment can bump it via
# ``max_concurrent``.
#
# ``weakref.WeakValueDictionary`` lets the semaphore get GC'd when no
# ``ClaudeCliCognition`` referencing that key remains — no permanent global
# state accumulated across long-running processes.
# ─────────────────────────────────────────────────────────────────────────────
_SEMAPHORES: weakref.WeakValueDictionary[tuple[str, str | None, int], asyncio.BoundedSemaphore] = (
    weakref.WeakValueDictionary()
)


def _get_semaphore(bin_: str, config_dir: str | None, max_concurrent: int) -> asyncio.BoundedSemaphore:
    """Return the shared BoundedSemaphore for this (bin, config_dir, max) triple.

    A ``WeakValueDictionary`` alone doesn't hold the semaphore alive — the
    caller must keep the returned reference for the duration of the acquire.
    That's the intended lifetime: as soon as no ``ClaudeCliCognition`` still
    has an in-flight ``drive`` referencing it, the entry drops.
    """
    key = (bin_, config_dir, max_concurrent)
    sem = _SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.BoundedSemaphore(max_concurrent)
        _SEMAPHORES[key] = sem
    return sem


# Reasons this cognition emits that mean "the run ERRORED", as opposed to
# "something stopped it deliberately". ``cli_exit_<n>`` is dynamic, which is
# why the mapping lives here rather than in the framework-wide table: only this
# module knows how it spells its own failures.
_CLI_FAILURE_REASONS = frozenset(
    {
        "spawn_failed",
        "parse_failed",
        "working_dir_missing",
        "cli_reported_error",
        # A turn sent into a dead session produced no answer. ``failed`` rather
        # than ``terminated``: nobody chose to stop this, the process was
        # already gone.
        "session_closed",
    }
)

# Structured-output failures are ``invalid_output`` in the closed taxonomy —
# the same category the tool loop uses when parse-and-repair is exhausted. They
# are NOT ``failed``: the run itself worked, the shape did not.
_CLI_INVALID_OUTPUT_REASONS = frozenset(
    {
        "error_max_structured_output_retries",
        "structured_output_missing",
        "structured_output_mismatch",
    }
)


def _coerce_structured(agent: Agent | None, value: Any) -> tuple[Any, str | None]:
    """Turn the CLI's validated JSON into the type the agent declared.

    The CLI validates against the schema and hands back a plain dict. An agent
    that declared ``output=Invoice`` wants an ``Invoice``, so the value goes
    back through the same ``SchemaAdapter`` that produced the schema — one
    round trip, one definition of the type.

    Returns ``(parsed, error)``. A coercion failure is reported, never raised:
    the run happened and its text is real, so the caller gets a terminal event
    with ``partial=True`` and an explanation rather than an exception thrown
    from inside a generator.

    With no adapter (an explicit ``json_schema=`` on an agent that declares no
    ``output=``) the validated dict IS the parsed value — there is no Python
    type to build.
    """
    adapter = getattr(agent, "_output_adapter", None)
    if adapter is None:
        return value, None
    try:
        return adapter.validate(value), None
    except Exception as exc:  # noqa: BLE001 — reported as data, see docstring
        # ``OutputCoercionError.__str__`` summarises ("1 error(s)"); the
        # per-field diagnostics live on ``.errors`` and are the only part a
        # caller can act on, so they go into the message.
        detail = "; ".join(str(e) for e in getattr(exc, "errors", ()) or ())
        return None, f"{type(exc).__name__}: {exc}" + (f" — {detail}" if detail else "")


def _cli_stop_reason(reason: str | None) -> AgentStopReason:
    """Map this cognition's free-form terminal reason onto the closed taxonomy.

    ``None`` and ``"success"`` are completion; the failure set above and any
    ``cli_exit_<n>`` are ``"failed"``; everything else (``"cancelled"``) defers
    to the shared table.
    """
    if reason in _CLI_INVALID_OUTPUT_REASONS:
        return "invalid_output"
    if reason in _CLI_FAILURE_REASONS or (reason is not None and reason.startswith("cli_exit_")):
        return "failed"
    return stop_reason_for(reason)


@dataclass(slots=True)
class ClaudeCliCognition:
    """Delegates the agent loop to a locally-installed ``claude`` CLI.

    Subprocesses ``claude -p "<task>" --output-format stream-json --verbose``
    per ``agent.run(...)`` / ``agent.stream(...)``. Uses whatever auth the
    CLI resolves — no API key handling on agentkit's side.

    Read from the agent: ``prompt`` (APPENDED to the CLI's own system prompt
    via ``--append-system-prompt``; set ``system_prompt_mode="replace"`` to
    override the CLI's prompt entirely instead), ``model`` (via ``--model`` when set on the
    cognition — the agent's ``model`` field is NOT consulted; the cognition's
    own ``model`` wins so a caller can point the CLI at a different model
    than the rest of the agentkit chain).

    Emits: ``message_delta`` per completed assistant text block — or per TOKEN
    with ``partial_messages=True``, which turns on the CLI's
    ``--include-partial-messages`` and streams the provider's own deltas;
    ``tool_call`` / ``tool_result`` for each CLI tool event; ``step`` for a
    provider retry; exactly one terminal ``final`` event carrying
    ``AgentResult(output=<accumulated assistant text>, usage=<Usage>,
    evals={"session_id": ..., "cli_duration_ms": ...})``.

    ``Usage.cost_usd`` is populated from the CLI's ``result.total_cost_usd``.
    Surface this as **a CLI-side estimate**, not a billed number — the CLI
    computes it from published per-token prices and can drift from provider
    invoices.
    """

    name: str = "claude_cli"
    claude_bin: str = "claude"
    model: str | None = None
    # How ``agent.prompt`` reaches the CLI. ``--system-prompt`` REPLACES the
    # entire Claude Code system prompt — tool guidance, environment info, the
    # lot — so wiring an agentkit prompt straight into it turned a capable
    # coding agent into a bare chat model that still had tools it no longer
    # knew how to use. ``--append-system-prompt`` is what the CLI docs
    # recommend for "add instructions while keeping Claude Code's default
    # behavior", and is the default here. Choose ``"replace"`` deliberately.
    system_prompt_mode: Literal["append", "replace"] = "append"
    working_dir: Path | None = None
    config_dir: Path | None = None  # → CLAUDE_CONFIG_DIR
    # ``--allowed-tools``: tools that run WITHOUT a permission prompt. This is
    # an auto-approve list, NOT a restriction — naming three tools here leaves
    # every other tool available and merely prompting. To restrict what exists
    # at all, use ``tools`` below. The CLI reference is explicit about the
    # split ("To restrict which tools are available, use --tools instead") and
    # conflating them is how a supposedly sandboxed agent keeps Bash.
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    # ``--tools``: the built-in tools the session HAS. ``()`` (the default)
    # passes the flag at all, leaving the CLI's own default set; ``("",)`` is
    # the CLI's spelling of "disable all tools"; a tuple of names restricts to
    # exactly those.
    tools: tuple[str, ...] | None = None
    permission_mode: PermissionMode = "default"
    permission_prompt_tool: str | None = None  # → --permission-prompt-tool (an MCP tool)
    max_turns: int | None = None

    # ── structured output ───────────────────────────────────────────────────
    # ``--json-schema``: the CLI validates its own final answer against this
    # schema and returns it in the result payload's ``structured_output``
    # field, re-prompting itself on a mismatch. Leave ``None`` and the schema
    # is taken from ``agent.output`` when one is declared, so the same
    # ``output=`` that types a normal agentkit run types a CLI-delegated one.
    # Set explicitly to override, or to ``{}``-free JSON Schema for an agent
    # that declares no ``output=``.
    json_schema: dict[str, Any] | None = None

    # ``--include-partial-messages``: token-level streaming. Without it the CLI
    # emits one ``assistant`` message per completed block, so a
    # ``message_delta`` arrives per PARAGRAPH, not per token — fine for a
    # backend, wrong for a UI with a cursor in it. With it on, the deltas come
    # from ``stream_event`` payloads and the completed ``assistant`` message is
    # used only to accumulate the authoritative text (never re-emitted, or the
    # consumer would see every sentence twice).
    partial_messages: bool = False

    # ── environment: what the session can reach ─────────────────────────────
    # ``--add-dir``: directories outside ``working_dir`` the session may read
    # and edit. Existence is checked at construction, matching what the CLI
    # itself validates, because a typo'd path is otherwise a subprocess that
    # dies three seconds in.
    add_dirs: tuple[Path | str, ...] = ()
    # ``--mcp-config``: MCP servers, as file paths or inline JSON strings.
    # ``--strict-mcp-config`` ignores every other MCP configuration, which is
    # what a service wants: the servers it declared, not the ones a developer
    # happened to have in ``~/.claude``.
    mcp_config: tuple[str | Path, ...] = ()
    strict_mcp_config: bool = False
    # ``--settings``: a settings file path or an inline JSON string, overriding
    # the same keys in the user's settings.json for this session only.
    settings: str | Path | None = None
    # ``--agents``: subagent definitions as JSON, serialised for you.
    agents: dict[str, Any] | None = None
    # ``--bare``: skip auto-discovery of hooks, skills, commands, subagents,
    # plugins, MCP servers, auto memory and CLAUDE.md. The CLI docs call this
    # "the recommended mode for scripted and SDK calls" and say it will become
    # the default for ``-p``. It is what makes a run reproducible across
    # machines: without it, a hook in a teammate's ``~/.claude`` or an MCP
    # server in the checked-out repo's ``.mcp.json`` executes inside your
    # service. Left False here because turning it on changes what a run can
    # see, which is not a default this cognition should flip under anyone.
    #
    # In bare mode the CLI never reads OAuth credentials or the keychain — set
    # ``ANTHROPIC_API_KEY`` (or an ``apiKeyHelper`` in ``settings``).
    bare: bool = False
    # ``--exclude-dynamic-system-prompt-sections``: move per-machine sections
    # (cwd, environment info, memory paths) out of the system prompt and into
    # the first user message, so the cache-stable prefix is identical across
    # users and machines running the same task. Documented for exactly this
    # workload: "Use with -p for scripted, multi-user workloads".
    stable_prompt_prefix: bool = False
    fallback_model: str | tuple[str, ...] | None = None  # → --fallback-model (tried in order)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # ``--no-session-persistence``: sessions are not written to disk. For a
    # multi-tenant service this is a containment control, not an optimisation.
    no_session_persistence: bool = False

    # ── spend ───────────────────────────────────────────────────────────────
    # Two halves of one problem. ``--max-budget-usd`` hands the run's remaining
    # headroom to the CLI so IT stops itself mid-flight; charging the meters
    # afterwards puts what it actually spent on the framework's books. Before
    # this the cognition was invisible to both: a $50 CLI run against a $1
    # Budget completed happily and the ledger read $0.00, which the class
    # docstring admitted ("callers who need a hard ceiling on CLI spend must
    # impose it externally").
    #
    # Turn it off for a run that should not draw on the shared envelope at all
    # — a warm-up call, an eval harness with its own accounting.
    meter_spend: bool = True

    # ── session identity ────────────────────────────────────────────────────
    # Three DIFFERENT things the CLI keeps separate, and so must we:
    #
    #   session_id         --session-id <uuid>   NAME a fresh session
    #   resume_session_id  --resume <id>         CONTINUE a specific session
    #   continue_session   --continue            CONTINUE the latest one here
    #
    # ``session_id`` used to be documented as "resume an existing session",
    # which it has never been: the CLI reference says "Use a specific session
    # ID for the conversation (must be a valid UUID)", i.e. it names a NEW
    # session. Passing a finished run's id back through it does not resume it.
    session_id: str | None = None
    resume_session_id: str | None = None
    continue_session: bool = False
    fork_session: bool = False  # → --fork-session (only with resume/continue)
    extra_args: tuple[str, ...] = ()  # escape hatch for future CLI flags
    terminate_grace_s: float = 5.0
    max_concurrent: int = 8  # class-level semaphore (see module doc)

    def __post_init__(self) -> None:
        """Refuse combinations the CLI itself refuses, at construction.

        Every one of these is cheaper to catch here than as a subprocess that
        exits non-zero three seconds later with a message the caller has to
        parse out of stderr — and ``session_id`` in particular fails in the
        least helpful way, because an invalid UUID is only rejected once the
        binary starts.
        """
        if self.session_id is not None and self.resume_session_id is not None:
            raise ValueError(
                "ClaudeCliCognition: session_id names a NEW session and "
                "resume_session_id continues an existing one — pass one, not both"
            )
        if self.continue_session and self.resume_session_id is not None:
            raise ValueError(
                "ClaudeCliCognition: continue_session resumes the latest session in "
                "working_dir and resume_session_id resumes a specific one — pass one, not both"
            )
        if self.fork_session and not (self.continue_session or self.resume_session_id):
            raise ValueError(
                "ClaudeCliCognition: fork_session only applies when resuming "
                "(set resume_session_id= or continue_session=True)"
            )
        if self.strict_mcp_config and not self.mcp_config:
            raise ValueError(
                "ClaudeCliCognition: strict_mcp_config only means something alongside "
                "mcp_config= — on its own it would leave the session with no MCP servers "
                "at all, which disallowed_tools=('mcp__*',) says more clearly"
            )
        if self.stable_prompt_prefix and self.system_prompt_mode == "replace":
            raise ValueError(
                "ClaudeCliCognition: stable_prompt_prefix has no effect with "
                "system_prompt_mode='replace' — the CLI only moves the dynamic sections out "
                "of ITS OWN default prompt, which a replacement discards anyway"
            )
        missing_dirs = [str(d) for d in self.add_dirs if not Path(d).is_dir()]
        if missing_dirs:
            raise ValueError(
                f"ClaudeCliCognition: add_dirs entries are not directories: {missing_dirs}"
            )
        if self.tools == ():
            # ``--tools`` is variadic and needs at least one value, so an empty
            # tuple would emit a bare flag the CLI rejects. It is also
            # genuinely ambiguous between the two things a reader might mean.
            raise ValueError(
                "ClaudeCliCognition: tools=() is ambiguous — pass tools=None to leave the "
                "CLI's default tool set alone, or tools=('',) to disable every tool"
            )
        if self.session_id is not None:
            try:
                uuid.UUID(str(self.session_id))
            except ValueError:
                raise ValueError(
                    f"ClaudeCliCognition: session_id must be a valid UUID, got "
                    f"{self.session_id!r}. To CONTINUE a previous run, use "
                    "resume_session_id= instead — that is the flag that resumes."
                ) from None

    # ---- public surface --------------------------------------------------------------------

    async def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]:
        """Run the CLI once, mapping its stream-json output to agentkit
        ``StreamEvent``s.

        **Terminal event guarantee.** Exactly one ``final`` event is yielded
        on every exit path — success, cancellation, non-zero CLI exit,
        CLI-signalled semantic error (``is_error: true`` in the result
        payload, e.g., ``error_max_turns``), and any exception raised
        before or during the spawn (e.g., ``FileNotFoundError`` if the
        ``claude`` binary isn't on PATH). Callers may drive the loop
        with ``async for ev in agent.stream(...)`` and rely on seeing
        one and only one ``StreamEvent(type='final')`` regardless of
        outcome. On failure paths, ``AgentResult.partial=True`` and
        ``evals["stop_reason"]`` names the failure mode.

        **`asyncio.CancelledError` semantics.** When the caller wraps this
        in ``asyncio.wait_for(...)`` or a ``TaskGroup`` and cancels,
        ``CancelledError`` is delivered mid-await. We terminate the
        subprocess, yield the terminal ``final(stop_reason="cancelled")``,
        AND re-raise ``CancelledError`` so the caller's cancel /
        timeout mechanism sees the signal. Suppressing it (as
        ``except BaseException`` would) breaks ``wait_for`` timeouts.

        **What this cognition IGNORES from ``ctx`` and ``agent``.** By
        design, this cognition delegates the whole loop to the CLI, so
        several agentkit contracts do NOT apply:

        - ``ctx.autonomy`` — the CLI's own ``permission_mode`` owns
          permissions; agentkit's autonomy tier is not translated.
        - ``agent.memory`` — the CLI manages its own context; the agent's
          ``MemorySource`` (if any) is never queried.
        - ``Agent.resume()`` — not supported (ReAct-only). For a CLI-native
          resume, pass the ``evals["session_id"]`` value back as
          ``ClaudeCliCognition(resume_session_id=...)`` — NOT ``session_id=``,
          which names a new session rather than continuing an old one.
        - ``ctx.budget`` — CHARGED, as of the spend integration. The run's
          remaining headroom goes out as ``--max-budget-usd`` so the CLI
          stops itself mid-flight, and what it actually spent is charged
          to every meter on the context plus the per-actor envelope once
          the run ends. An already-exhausted budget refuses to spawn at
          all, with the resumable ``budget_exhausted`` stop reason. Set
          ``meter_spend=False`` to opt a run out of both ends.
        """
        del context  # unused — the CLI owns its own transcript

        # Bounded-concurrency spawn guard. Held for the whole CLI lifetime so
        # concurrent drives don't race the subprocess table (SDK issue #728).
        # The local ``sem`` reference keeps the WeakValueDictionary entry
        # alive for the ``async with sem`` scope — no instance-field
        # bookkeeping needed.
        cfg_dir = str(self.config_dir) if self.config_dir is not None else None
        sem = _get_semaphore(self.claude_bin, cfg_dir, self.max_concurrent)

        state = _TurnState()
        cancelled = False
        # Set once the schema is resolved. Initialised False so a spawn that
        # dies before resolution still reaches the terminal event.
        schema_requested = False
        stderr_bytes: bytes = b""
        proc: asyncio.subprocess.Process | None = None
        # ``fatal_exc`` records any exception raised before/during spawn or
        # while parsing the CLI's output. We can't ``raise`` past the final
        # yield without breaking the terminal-event guarantee, so we stash
        # the message and fold it into the final event's ``stop_reason`` +
        # ``evals``. The generator still terminates normally.
        fatal_exc: BaseException | None = None
        # Set when a caller-injected cancel (asyncio.CancelledError from
        # wait_for / TaskGroup) arrives. We yield the terminal event first,
        # then re-raise so the caller's cancel mechanism honours the signal.
        # Suppressing would make wait_for timeouts silently return a partial
        # result instead of raising TimeoutError.
        should_reraise_cancel = False

        # Pre-flight: if working_dir is set and doesn't exist, surface a
        # distinct stop_reason so operators don't confuse it with a missing
        # binary. `create_subprocess_exec` would raise FileNotFoundError
        # either way but the reader loses the distinction.
        if self.working_dir is not None and not self.working_dir.exists():
            wd_missing = f"working_dir does not exist: {self.working_dir}"
        else:
            wd_missing = None

        try:
            # argv / env resolution can throw (e.g., ``Prompt.render()`` on a
            # malformed template, ``os.environ.copy()`` under bizarre OS
            # states). Inside the outer try so we always yield a final.
            system_prompt = self._resolve_system_prompt(agent.prompt)
            schema = self._resolve_json_schema(agent)
            schema_requested = schema is not None
            budget_cap = self._budget_cap(ctx)
            argv = self._build_argv(
                task, system_prompt=system_prompt, json_schema=schema, max_budget_usd=budget_cap
            )
            env = self._build_env(ctx=ctx)

            if wd_missing is not None:
                raise FileNotFoundError(wd_missing)

            async with sem:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.working_dir) if self.working_dir is not None else None,
                    env=env,
                )
                assert proc.stdout is not None  # PIPE
                try:
                    async for line in proc.stdout:
                        try:
                            ctx.check_cancelled()
                        except Exception:
                            cancelled = True
                            break
                        payload = _parse_line(line)
                        if payload is None:
                            continue
                        async for ev, delta in _events_from_payload(
                            payload, partial=self.partial_messages
                        ):
                            if ev is not None:
                                yield ev
                            state.fold(delta)
                finally:
                    if cancelled and proc.returncode is None:
                        await _terminate(proc, self.terminate_grace_s)
                    # Collect stderr — best-effort; the pipe may already be closed.
                    if proc.stderr is not None:
                        with contextlib.suppress(Exception):
                            stderr_bytes = await proc.stderr.read()
                    with contextlib.suppress(Exception):
                        await proc.wait()
        except asyncio.CancelledError:
            cancelled = True
            should_reraise_cancel = True
            if proc is not None and proc.returncode is None:
                await _terminate(proc, self.terminate_grace_s)
        except BaseException as exc:  # noqa: BLE001 — see terminal-event guarantee
            # Anything else that escapes to here (FileNotFoundError on
            # missing ``claude`` binary, PermissionError, parse-time bug
            # in _events_from_payload) becomes a final event with a
            # ``spawn_failed`` / ``parse_failed`` stop_reason. We
            # deliberately widen to ``BaseException`` so ``KeyboardInterrupt``
            # and ``SystemExit`` also produce a terminal event before
            # propagating. (``CancelledError`` is handled separately above
            # so timeouts propagate correctly.)
            fatal_exc = exc

        result = await self._finalise(
            agent=agent,
            ctx=ctx,
            state=state,
            cancelled=cancelled,
            fatal_exc=fatal_exc,
            spawned=proc is not None,
            return_code=proc.returncode if proc is not None else -1,
            stderr_bytes=stderr_bytes,
            schema_requested=schema_requested,
        )
        yield StreamEvent("final", usage=state.usage, result=result)
        if should_reraise_cancel:
            # Terminal event delivered; now propagate the cancel so
            # ``asyncio.wait_for(..., timeout=X)`` raises ``TimeoutError``
            # and TaskGroup cancels propagate to siblings.
            raise asyncio.CancelledError()


    async def _finalise(
        self,
        *,
        agent: Agent | None,
        ctx: Ctx | None,
        state: _TurnState,
        cancelled: bool,
        fatal_exc: BaseException | None,
        spawned: bool,
        return_code: int | None,
        stderr_bytes: bytes,
        schema_requested: bool,
    ) -> AgentResult:
        """Turn one completed turn's state into its terminal ``AgentResult``.

        Shared by the one-shot ``drive`` and by a persistent
        :class:`ClaudeCliSession` turn, which differ only in process
        lifecycle. The stop-reason priority, the structured-output
        decision, the ``evals`` shape and the metering are identical
        between them, and a second copy of that logic is precisely how the
        two paths would drift apart.

        ``return_code`` is ``None`` when the process is still ALIVE — a session
        turn ends at its ``result`` payload, not at process exit — and ``-1``
        when no process was ever spawned. Both mean "the exit code says
        nothing about this turn", so neither is treated as a failure here.
        """

        # Decide terminal stop_reason + partial flag.
        # Priority (highest first): cancellation → fatal exception → CLI
        # non-zero exit → CLI semantic error (is_error) → success.
        # Cancellation is above fatal_exc because a cancel that races with
        # a fatal error should still surface as ``cancelled``.
        final_stop_reason: str | None
        final_partial: bool
        if cancelled:
            final_stop_reason = "cancelled"
            final_partial = True
        elif fatal_exc is not None:
            # Distinguish working_dir_missing from spawn_failed for
            # operator clarity (the CLI would raise FileNotFoundError for
            # both, but the fix path differs).
            if isinstance(fatal_exc, _SessionClosed):
                # The process behind a persistent session is gone (or the turn
                # asked for something only a fresh process can do). NOT
                # ``spawn_failed``: nothing failed to start — a conversation
                # ended. The caller's fix is a new session, not a retry.
                final_stop_reason = "session_closed"
            elif type(fatal_exc).__name__ == "MeterExceeded":
                # The pre-flight refusal from ``_budget_cap``: no subprocess
                # was spawned. ``budget_exhausted`` is a RESUMABLE stop reason,
                # which is the honest one — raise the ceiling and run again.
                # Matched by name so this module keeps its import of
                # ``runtime.meter`` lazy.
                final_stop_reason = "budget_exhausted"
            elif not spawned:
                if isinstance(fatal_exc, FileNotFoundError) and str(fatal_exc).startswith(
                    "working_dir does not exist:"
                ):
                    final_stop_reason = "working_dir_missing"
                else:
                    final_stop_reason = "spawn_failed"
            else:
                final_stop_reason = "parse_failed"
            final_partial = True
        elif return_code not in (0, None):
            final_stop_reason = f"cli_exit_{return_code}"
            final_partial = True
        elif state.is_error:
            # When the CLI signals ``is_error: true`` while exiting 0, the
            # useful stop_reason is often on ``terminal_reason`` (e.g.,
            # ``"api_error"``) rather than ``subtype`` (which can still
            # read ``"success"``). We already prefer subtype in
            # ``_events_from_payload`` — but when subtype is the useless
            # ``"success"`` string, fall through to a generic marker.
            if state.stop_reason == "success" or state.stop_reason is None:
                final_stop_reason = "cli_reported_error"
            else:
                final_stop_reason = state.stop_reason
            final_partial = True
        else:
            final_stop_reason = state.stop_reason  # may still be None on clean success
            final_partial = False

        # ── structured output ───────────────────────────────────────────────
        # A schema was requested. Three outcomes, and only the first is a
        # success:
        #
        #   value present   → coerce to the declared type; ``parsed`` is typed
        #   retries burnt   → subtype ``error_max_structured_output_retries``
        #   absent, exit 0  → the docs are explicit that this is a failure too
        #
        # The third is the one worth spelling out: without this branch the run
        # returns ``partial=False`` and ``parsed=None``, and a caller that
        # declared ``output=Invoice`` reads the prose as if the object simply
        # had not been wired. Treating it as a failure makes it visible.
        parsed: Any = None
        if schema_requested:
            if state.structured_output is not None:
                parsed, coercion_error = _coerce_structured(agent, state.structured_output)
                if coercion_error is not None:
                    final_partial = True
                    final_stop_reason = "structured_output_mismatch"
                    evals_structured_error = coercion_error
                else:
                    evals_structured_error = None
            else:
                final_partial = True
                if final_stop_reason in (None, "success"):
                    final_stop_reason = "structured_output_missing"
                evals_structured_error = (
                    "the CLI returned no state.structured_output despite --json-schema"
                )
        else:
            evals_structured_error = None

        # Charge the framework's meters with what the CLI actually spent. After
        # the stop-reason decision (so a charge cannot change it) and before the
        # terminal event (so the books are straight by the time the caller sees
        # the result).
        charge_error = await self._charge_meters(ctx, state.usage)

        evals: dict[str, Any] = {
            "session_id": state.session_id or "",
            "cli_duration_ms": state.duration_ms or 0,
            "cli_return_code": return_code if return_code is not None else 0,
        }
        # Bridge agentkit's correlation_id into the final result so downstream
        # observability can join on it (matching the env var we set on the
        # subprocess via `CLAUDE_TRACE_EXTERNAL_ID`).
        external_id = getattr(ctx, "correlation_id", None)
        if external_id:
            evals["external_run_id"] = str(external_id)
        if final_stop_reason is not None:
            evals["stop_reason"] = final_stop_reason
        if state.init:
            # Startup facts an operator needs when a run behaves oddly: which
            # model actually ran, which MCP servers connected — and which were
            # SKIPPED. ``mcp_server_errors`` / ``plugin_errors`` appear only
            # when non-empty, so their presence is the CI gate the CLI docs
            # recommend.
            evals["cli_init"] = state.init
        if state.api_retries:
            # The CLI retried the provider. A run that took 40s with one API
            # call is explained by this and nothing else in the result.
            evals["api_retries"] = state.api_retries
        if state.structured_output is not None:
            # The RAW validated dict, alongside the typed ``parsed`` object. A
            # caller that declared no Python type still wants the data.
            evals["structured_output"] = state.structured_output
        if evals_structured_error is not None:
            evals["structured_output_error"] = evals_structured_error
        if state.thinking:
            # Reasoning chain from ``thinking`` blocks — separate from
            # ``output`` (the final response). Live consumers already saw
            # each chunk as a ``message_delta``; this is the folded copy for
            # AgentResult callers.
            evals["thinking"] = state.thinking
        if fatal_exc is not None:
            evals["error"] = f"{type(fatal_exc).__name__}: {fatal_exc}"
        if charge_error is not None:
            # A meter refused the charge — almost always a ceiling crossed by
            # this very run. The spend is on the books either way; this records
            # that the ceiling is now behind us.
            evals["meter_error"] = charge_error
        if final_partial and stderr_bytes:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if stderr_text:
                evals["stderr"] = stderr_text

        return AgentResult(
            output=state.text,
            usage=state.usage,
            partial=final_partial,
            evals=evals,
            parsed=parsed,
            # This cognition reports failures as DATA (a terminal event is
            # guaranteed even when the subprocess never starts), so it is
            # the one producer that can legitimately stamp ``"failed"``.
            stop_reason=_cli_stop_reason(final_stop_reason),
        )

    def session(self, *, agent: Agent | None = None) -> ClaudeCliSession:
        """Open a persistent CLI session — one process, many turns.

        ``drive()`` spawns a subprocess per turn, which costs two to five
        seconds of CLI warm-up every time. A session pays that once and keeps
        the CLI's own conversation context alive between turns::

            async with cognition.session() as chat:
                async for ev in chat.turn("Summarise README.md"):
                    ...
                async for ev in chat.turn("Now list the risks you skipped"):
                    ...

        See :class:`ClaudeCliSession` for what a shared process implies —
        serialised turns, and a session that ends when its process does.
        """
        return ClaudeCliSession(self, agent=agent)

    # ---- helpers ---------------------------------------------------------------------------

    def _resolve_system_prompt(self, prompt: Prompt | str | None) -> str:
        """Extract a rendered system-prompt string from ``agent.prompt``.

        The agent's ``prompt`` field accepts three shapes: ``None``,
        ``str``, or ``Prompt``. Match all three; render the ``Prompt``
        via ``Prompt.render()`` for the versioned path.
        """
        if prompt is None:
            return ""
        if isinstance(prompt, Prompt):
            return prompt.render()
        return prompt

    def _warn_if_bare_mode_has_no_credential(self, env: dict[str, str]) -> None:
        """Bare mode ignores OAuth and the keychain — say so before the CLI does.

        Warned rather than refused: the credential may arrive through an
        ``apiKeyHelper`` in ``settings``, or through a provider mechanism this
        list does not know about, and refusing a run over an env-var heuristic
        would be worse than the confusing message it replaces.
        """
        if self.settings is not None:
            return
        if any(env.get(name) for name in _BARE_CREDENTIAL_ENV):
            return
        import warnings

        warnings.warn(
            "ClaudeCliCognition(bare=True) but no API credential is in the environment "
            f"({', '.join(_BARE_CREDENTIAL_ENV)}). Bare mode never reads OAuth credentials "
            "or the system keychain, so a `claude` that works in your terminal will fail "
            "here with an auth error ('Not logged in · Please run /login', or 'Invalid API "
            "key' depending on state) that points at exactly the wrong fix. Set "
            "ANTHROPIC_API_KEY, or supply an apiKeyHelper via settings=.",
            UserWarning,
            stacklevel=3,
        )

    def _resolve_json_schema(self, agent: Agent) -> dict[str, Any] | None:
        """The JSON Schema to hand the CLI, or ``None``.

        An explicit ``json_schema=`` wins. Otherwise the agent's own
        ``output=`` schema is used, through the same ``SchemaAdapter`` the rest
        of the framework uses — so declaring ``output=Invoice`` types a
        CLI-delegated run exactly like it types a normal one, instead of being
        silently ignored the way it was before this existed.

        Reading the adapter defensively (``getattr``) keeps this working for
        agent-likes in tests that don't build one.
        """
        if self.json_schema is not None:
            return self.json_schema
        adapter = getattr(agent, "_output_adapter", None)
        if adapter is None:
            return None
        try:
            schema = adapter.json_schema()
        except Exception:  # noqa: BLE001 — a schema we cannot render is not a run-ender
            return None
        return schema if isinstance(schema, dict) and schema else None

    def _budget_cap(self, ctx: Ctx | None) -> str | None:
        """The run's remaining headroom, as the CLI's ``--max-budget-usd`` wants it.

        ``None`` when metering is off, no budget is wired, or the budget has no
        ceiling — in each case there is no number to enforce and the flag must
        not appear, because passing one would invent a limit the caller never
        set.

        Raises :class:`MeterExceeded` when the headroom is already gone. That is
        a pre-flight refusal on purpose: spawning a subprocess to be told what
        we already know costs two to five seconds of CLI warm-up, and the
        terminal event this produces (``budget_exhausted``) is the resumable
        one — raise the ceiling and run again.
        """
        if not self.meter_spend or ctx is None:
            return None
        budget = getattr(ctx, "budget", None)
        remaining = getattr(budget, "remaining", None)
        if remaining is None:
            return None
        headroom = remaining()
        if headroom is None:  # no ceiling configured
            return None
        if headroom <= 0:
            from agentkit.runtime.meter import MeterExceeded

            raise MeterExceeded(
                f"claude CLI not spawned: the run budget has {headroom} USD left"
            )
        return f"{headroom:f}"

    async def _charge_meters(self, ctx: Ctx | None, usage: Usage) -> str | None:
        """Put the CLI's spend on the framework's books. Returns an error note.

        The CLI bypasses the ``Invoker``, so the ``meter()`` middleware never
        sees this usage and every meter on the context stays at zero. That is
        how a documented safety mechanism ends up doing nothing — the same
        failure ``ActorBudget`` had — so the charge happens here instead.

        Nothing raises out of this. The spend already happened, the run already
        produced an answer, and the terminal-event guarantee says the caller
        gets that answer; a ceiling crossed on the LAST call is recorded and
        reported, not converted into a lost result. A custom meter that
        misbehaves is contained for the same reason.
        """
        if not self.meter_spend or ctx is None:
            return None
        call = _CliCall(ctx=ctx)
        note: str | None = None
        for meter in getattr(ctx, "all_meters", None) or []:
            try:
                await meter.charge(call, usage)
            except Exception as exc:  # noqa: BLE001 — see docstring
                note = f"{type(exc).__name__}: {exc}"
        actor = getattr(ctx, "actor_budget", None)
        if actor is not None:
            with contextlib.suppress(Exception):
                actor.charge(tokens=usage.total_tokens, cost_usd=usage.cost_usd, steps=1)
        return note

    def _build_argv(
        self,
        task: str,
        *,
        system_prompt: str,
        json_schema: dict[str, Any] | None = None,
        max_budget_usd: str | None = None,
        stream_input: bool = False,
    ) -> list[str]:
        """Assemble the CLI argv. Order-independent; kept grouped by role
        (identity → format → model → prompt → tools → permissions → resume →
        extras) so a diff is legible."""
        argv: list[str] = [self.claude_bin, "-p"]
        if stream_input:
            # A session feeds turns over stdin as newline-delimited JSON, so
            # there is no prompt ARGUMENT — passing one alongside
            # ``--input-format stream-json`` would make the CLI run it as a
            # first turn nobody asked for.
            argv += ["--input-format", "stream-json"]
        else:
            argv += [task]
        argv += ["--output-format", "stream-json", "--verbose"]
        if self.model is not None:
            argv += ["--model", self.model]
        if system_prompt:
            flag = "--system-prompt" if self.system_prompt_mode == "replace" else (
                "--append-system-prompt"
            )
            argv += [flag, system_prompt]
        if self.allowed_tools:
            argv += ["--allowed-tools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            argv += ["--disallowed-tools", ",".join(self.disallowed_tools)]
        if self.tools is not None:
            # ``--tools`` is variadic on the CLI; each name is its own argv
            # entry. ``("",)`` — the documented "disable all tools" spelling —
            # therefore survives as a single empty argument.
            argv += ["--tools", *self.tools]
        if self.partial_messages:
            argv += ["--include-partial-messages"]
        if self.bare:
            argv += ["--bare"]
        if self.stable_prompt_prefix:
            argv += ["--exclude-dynamic-system-prompt-sections"]
        if self.fallback_model is not None:
            models = (
                self.fallback_model
                if isinstance(self.fallback_model, str)
                else ",".join(self.fallback_model)
            )
            argv += ["--fallback-model", models]
        if self.effort is not None:
            argv += ["--effort", self.effort]
        for d in self.add_dirs:
            argv += ["--add-dir", str(d)]
        if self.mcp_config:
            # Variadic: every entry after the flag, path or inline JSON alike.
            argv += ["--mcp-config", *(str(c) for c in self.mcp_config)]
        if self.strict_mcp_config:
            argv += ["--strict-mcp-config"]
        if self.settings is not None:
            argv += ["--settings", str(self.settings)]
        if self.agents is not None:
            argv += ["--agents", json.dumps(self.agents)]
        if self.no_session_persistence:
            argv += ["--no-session-persistence"]
        if self.permission_mode != "default":
            argv += ["--permission-mode", self.permission_mode]
        if self.permission_prompt_tool is not None:
            argv += ["--permission-prompt-tool", self.permission_prompt_tool]
        if self.max_turns is not None:
            argv += ["--max-turns", str(self.max_turns)]
        if max_budget_usd is not None:
            argv += ["--max-budget-usd", max_budget_usd]
        if json_schema is not None:
            # The CLI parses this argument as JSON and rejects an invalid
            # schema at startup ("Error: --json-schema is not a valid JSON
            # Schema"), so a malformed one fails the run rather than silently
            # returning prose — which is what it did before CLI v2.1.205.
            argv += ["--json-schema", json.dumps(json_schema)]
        if self.session_id is not None:
            argv += ["--session-id", self.session_id]
        if self.resume_session_id is not None:
            argv += ["--resume", self.resume_session_id]
        if self.continue_session:
            argv += ["--continue"]
        if self.fork_session:
            argv += ["--fork-session"]
        argv += list(self.extra_args)
        return argv

    def _build_env(self, *, ctx: Ctx | None = None) -> dict[str, str]:
        """Copy the process env; layer ``CLAUDE_CONFIG_DIR`` on top when the
        cognition was constructed with a ``config_dir`` (for isolated auth /
        settings, e.g. per-tenant server-side wrapper). Also default
        ``CLAUDE_ENABLE_STREAM_WATCHDOG=1`` to mitigate long-tail SSE hangs
        (SDK issue #33949) unless the caller already set it explicitly.

        When ``ctx`` carries a ``correlation_id``, bridge it into the child
        as ``CLAUDE_TRACE_EXTERNAL_ID`` so operators can join agentkit and
        CLI traces on a single id. Idempotent under nested drives — a
        caller-set value wins.
        """
        env = os.environ.copy()
        if self.bare:
            self._warn_if_bare_mode_has_no_credential(env)
        if self.config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = str(self.config_dir)
        env.setdefault("CLAUDE_ENABLE_STREAM_WATCHDOG", "1")
        if ctx is not None:
            correlation_id = getattr(ctx, "correlation_id", None)
            if correlation_id:
                env.setdefault("CLAUDE_TRACE_EXTERNAL_ID", str(correlation_id))
        return env


class ClaudeCliSession:
    """One `claude` process, many turns.

    ``ClaudeCliCognition.drive`` spawns a subprocess per turn, which costs two
    to five seconds of CLI warm-up EVERY time — measured on a two-turn
    conversation, 4.2s for the first turn and 1.1s for the second when they
    share a process. For a chat UI or an agent that iterates with a person,
    that difference is the whole interaction.

    A session holds the process open and feeds turns over stdin as
    newline-delimited JSON (``--input-format stream-json``), so the CLI keeps
    its own conversation context in memory. Verified against the binary: the
    model recalls a number from turn 1 in turn 2, one session id spans both,
    and closing stdin exits 0.

    ::

        async with cognition.session() as chat:
            async for ev in chat.turn("Summarise README.md"):
                ...
            async for ev in chat.turn("Now list the risks you skipped"):
                ...

    Every per-turn contract of ``drive`` holds unchanged, because both go
    through the same ``_TurnState`` and ``_finalise``: exactly one terminal
    ``final`` event, the same stop-reason taxonomy, the same structured-output
    handling, the same metering. What differs is only what a shared process
    implies, and each of those is a real trade:

    * **Turns are serialised.** One stdin, one transcript, so a second
      concurrent ``turn()`` on the same session would interleave two
      conversations into one context. A lock makes the second caller wait.
    * **A dead process stays dead.** The CLI exiting mid-session (a crash, an
      OOM kill, ``--max-turns`` reached) ends the session; the turn that
      noticed reports it and every later ``turn()`` refuses rather than
      silently starting a fresh conversation with no history.
    * **Cancelling a turn ends the session.** There is no way to tell the CLI
      "forget the turn you were mid-way through" over this protocol, so the
      process is terminated. That is the honest outcome: the alternative is a
      session whose context contains half an answer nobody saw.
    """

    def __init__(self, cognition: ClaudeCliCognition, *, agent: Agent | None = None) -> None:
        self._cog = cognition
        self._agent = agent
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._sem_holder: Any = None
        self.session_id: str | None = None  # populated from the first turn's init payload
        # Control requests are answered on the SAME stdout the turn reader is
        # consuming, so the reader routes each ``control_response`` to the
        # future waiting on it. Keyed by request id.
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._control_seq = 0
        # Set while a turn is streaming. ``interrupt()`` needs to know whether
        # anyone is reading — a control request with no reader never gets an
        # answer, so it would hang instead of failing.
        self._turn_active = False
        self._interrupted = False
        self._capabilities: frozenset[str] = frozenset()

    # ---- lifecycle -------------------------------------------------------------------------

    async def __aenter__(self) -> ClaudeCliSession:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Spawn the CLI. Idempotent; ``__aenter__`` calls it for you."""
        if self._proc is not None:
            return
        cog = self._cog
        # The spawn semaphore is held for the WHOLE session, not per turn: the
        # bound exists because the SDK hangs at ~200 live subprocesses, and a
        # session's subprocess is live the entire time.
        sem = _get_semaphore(
            cog.claude_bin,
            str(cog.config_dir) if cog.config_dir is not None else None,
            cog.max_concurrent,
        )
        self._sem_holder = sem
        await sem.acquire()
        try:
            argv = cog._build_argv("", system_prompt="", stream_input=True)
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cog.working_dir) if cog.working_dir is not None else None,
                env=cog._build_env(),
            )
        except BaseException:
            sem.release()
            self._sem_holder = None
            raise

    async def close(self) -> None:
        """Close stdin and wait for the CLI to exit, then release the permit.

        Closing stdin is the protocol's own end-of-conversation signal and the
        CLI exits 0 on it, so this is a clean shutdown rather than a kill. A
        process that ignores it is terminated after the usual grace.
        """
        self._closed = True
        proc, self._proc = self._proc, None
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None and not proc.stdin.is_closing():
                    proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=self._cog.terminate_grace_s)
            except TimeoutError:
                await _terminate(proc, self._cog.terminate_grace_s)
        if self._sem_holder is not None:
            self._sem_holder.release()
            self._sem_holder = None

    # ---- control protocol ------------------------------------------------------------------

    async def interrupt(
        self, *, cancel_queued: bool = False, ack_timeout_s: float = 5.0
    ) -> InterruptReceipt:
        """Stop the turn that is currently streaming, keeping the session alive.

        This is the piece a chat UI needs and cancellation cannot give you.
        Cancelling a turn terminates the process — no protocol message retracts
        a half-finished turn, so the conversation ends with it. An interrupt is
        the CLI's own verb for the same intent: the in-flight turn stops, the
        process stays up, and the next ``turn()`` continues the SAME
        conversation. Verified against the binary — an interrupted turn is
        followed by a normal one in the same process, which then exits 0.

        The interrupted turn still yields exactly one terminal ``final`` event,
        with ``stop_reason="interrupted"`` (``terminated`` in the closed
        taxonomy — somebody stopped this deliberately; nothing failed).

        ``cancel_queued`` also drops messages the CLI has queued but not yet
        dispatched. Requires the ``interrupt_cancel_queued_v1`` capability,
        which is checked before the request is sent rather than after it is
        ignored.

        Returns an :class:`InterruptReceipt`. ``still_queued`` names the queued
        messages that will run anyway, which is the difference between "the
        agent stopped" and "the agent stopped and has three more things to do".

        Safe to call when no turn is running: there is nothing to interrupt, so
        it returns an empty receipt rather than sending a request nobody would
        answer — the reader that resolves it only exists during a turn.

        **The interrupt is delivered synchronously; the RECEIPT is best-effort.**
        The CLI answers on the same stdout the turn reader consumes, so the
        acknowledgement only arrives while somebody is draining the stream. Call
        this from a separate task — a UI's stop button, which is the natural
        shape — and it returns promptly with the full receipt::

            stop = asyncio.create_task(chat.interrupt())

        Call it inline from the loop consuming ``turn()`` and that loop is
        suspended at a yield, so nothing is reading and the acknowledgement
        cannot arrive. ``ack_timeout_s`` bounds that into a wait rather than a
        deadlock: the receipt comes back ``delivered=True`` with an ``error``
        noting the missing acknowledgement, and the interrupt still takes effect
        because the WRITE already happened. Getting this wrong should cost a
        field on a receipt, not a hung application.
        """
        proc = self._proc
        if proc is None or proc.returncode is not None or not self._turn_active:
            return InterruptReceipt(delivered=False)

        subtype = "interrupt"
        if cancel_queued:
            if not self.supports("interrupt_cancel_queued_v1"):
                raise ValueError(
                    "this claude CLI does not advertise 'interrupt_cancel_queued_v1'; "
                    f"it reports {sorted(self.capabilities) or '<none>'}. Call interrupt() "
                    "without cancel_queued, or upgrade the CLI."
                )
            subtype = "interrupt_cancel_queued"

        self._control_seq += 1
        request_id = f"agentkit-{self._control_seq}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future

        # Set BEFORE the write. The turn's terminal event is assembled by
        # whichever coroutine is reading, and it may reach that point before
        # this one resumes from the await below.
        self._interrupted = True
        try:
            assert proc.stdin is not None  # PIPE
            proc.stdin.write(
                (
                    json.dumps(
                        {
                            "type": "control_request",
                            "request_id": request_id,
                            "request": {"subtype": subtype},
                        }
                    )
                    + "\n"
                ).encode()
            )
            await proc.stdin.drain()
            try:
                payload = await asyncio.wait_for(future, timeout=ack_timeout_s)
            except TimeoutError:
                return InterruptReceipt(
                    delivered=True,  # the request was written; the answer was not read
                    error=(
                        f"no acknowledgement within {ack_timeout_s}s — nothing was draining "
                        "the CLI's output. Call interrupt() from a separate task "
                        "(asyncio.create_task) rather than from inside the loop consuming "
                        "turn()."
                    ),
                )
        finally:
            self._pending.pop(request_id, None)

        response = payload.get("response") or {}
        inner = response.get("response") or {}
        return InterruptReceipt(
            delivered=response.get("subtype") == "success",
            still_queued=tuple(inner.get("still_queued") or ()),
            cancelled=tuple(inner.get("cancelled") or ()),
            error=str(response.get("error") or "") or None,
        )

    @property
    def capabilities(self) -> frozenset[str]:
        """Protocol behaviours this CLI advertises on ``system/init``.

        Feature-detect against this instead of comparing version strings — it
        is what the field exists for, and the set is open, so an unrecognised
        value is not an error.

        Empty until the first turn has produced its init payload.
        """
        return frozenset(self._capabilities)

    def supports(self, capability: str) -> bool:
        """Whether the CLI advertised ``capability``."""
        return capability in self._capabilities

    def _absorb_init(self, init: dict[str, Any]) -> None:
        """Record what the ``system/init`` payload said about this CLI."""
        advertised = init.get("capabilities")
        if advertised:
            self._capabilities = frozenset(advertised)

    def _resolve_control(self, payload: dict[str, Any]) -> None:
        """Hand a ``control_response`` to whoever is awaiting it.

        An unmatched id is dropped rather than raised: the CLI may answer
        something a previous turn abandoned, and a stray reply must not take
        down the turn currently streaming.
        """
        response = payload.get("response") or {}
        request_id = str(response.get("request_id") or payload.get("request_id") or "")
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(payload)

    def _fail_pending(self, reason: str) -> None:
        """Fail every outstanding control request.

        Called when a turn ends: the reader is gone, so nothing will ever
        resolve these, and an awaiting caller would hang forever otherwise.
        """
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(_SessionClosed(reason))
        self._pending.clear()

    # ---- turns -----------------------------------------------------------------------------

    async def turn(
        self,
        task: str,
        *,
        agent: Agent | None = None,
        ctx: Ctx | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send one user turn and stream its events, ending at exactly one ``final``.

        ``agent`` supplies the output schema (and is required for one, since the
        schema is a property of the AGENT, not the session); ``ctx`` supplies
        the meters and the cancel token. Both may be omitted for a bare
        conversational session.
        """
        async for ev in self._turn(task, agent=agent or self._agent, ctx=ctx):
            yield ev

    async def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]:
        """``Cognition``-shaped entry point, so a session can BE an agent's
        cognition and consecutive ``agent.run(...)`` calls share one process and
        one CLI-side conversation."""
        del context  # the CLI owns its own transcript
        async for ev in self._turn(task, agent=agent, ctx=ctx):
            yield ev

    async def _turn(
        self, task: str, *, agent: Agent | None, ctx: Ctx | None
    ) -> AsyncIterator[StreamEvent]:
        state = _TurnState()
        cancelled = False
        fatal_exc: BaseException | None = None
        should_reraise_cancel = False
        stderr_bytes = b""
        schema_requested = False
        # Serialise turns: one stdin and one transcript mean two concurrent
        # turns would interleave into a single conversation.
        async with self._lock:
            proc = self._proc
            try:
                if self._closed or proc is None:
                    raise _SessionClosed(
                        "the CLI session is closed — start a new one; its conversation "
                        "context is gone with the process"
                    )
                if proc.returncode is not None:
                    raise _SessionClosed(
                        f"the CLI process exited with code {proc.returncode}; its "
                        "conversation context is gone, so this turn was not sent"
                    )
                if agent is not None:
                    schema_requested = self._cog._resolve_json_schema(agent) is not None
                    if schema_requested:
                        # ``--json-schema`` is a process-level flag, fixed at
                        # spawn. Saying so beats silently returning prose for an
                        # ``output=`` the caller believes is wired.
                        raise _SessionClosed(
                            "structured output is a process-level flag on the CLI, so it "
                            "cannot be turned on per turn. Use ClaudeCliCognition.drive() "
                            "for a typed run, or pass json_schema= on the cognition before "
                            "opening the session."
                        )
                assert proc.stdin is not None  # PIPE
                self._interrupted = False
                self._turn_active = True
                proc.stdin.write(_user_turn(task).encode())
                await proc.stdin.drain()

                assert proc.stdout is not None  # PIPE
                async for line in proc.stdout:
                    if ctx is not None:
                        try:
                            ctx.check_cancelled()
                        except Exception:
                            cancelled = True
                            break
                    payload = _parse_line(line)
                    if payload is None:
                        continue
                    if payload.get("type") == "control_response":
                        # Not turn content: the CLI's answer to something we
                        # asked out-of-band. Routing it here is what makes
                        # ``interrupt()`` awaitable at all — nothing else reads
                        # this stream while a turn is in flight.
                        self._resolve_control(payload)
                        continue
                    async for ev, delta in _events_from_payload(
                        payload, partial=self._cog.partial_messages
                    ):
                        if ev is not None:
                            yield ev
                        state.fold(delta)
                        # Capabilities are folded LIVE, not after the turn:
                        # ``interrupt()`` is called mid-turn and feature-detects
                        # against them, so a capability that only lands at the
                        # terminal event is one nobody can act on.
                        if delta.init:
                            self._absorb_init(delta.init)
                    # A turn ends at its ``result`` payload — NOT at EOF, which
                    # is what a one-shot drive waits for. The process stays
                    # alive for the next turn.
                    if payload.get("type") == "result":
                        break
                else:
                    # stdout ended without a result: the CLI died mid-turn.
                    raise _SessionClosed(
                        "the CLI closed its output mid-turn; the session is over"
                    )
            except asyncio.CancelledError:
                cancelled = True
                should_reraise_cancel = True
            except BaseException as exc:  # noqa: BLE001 — terminal-event guarantee
                fatal_exc = exc
            finally:
                self._turn_active = False
                self._fail_pending("the turn ended before the CLI answered")
                if cancelled and proc is not None and proc.returncode is None:
                    # No protocol message retracts a half-finished turn, so the
                    # session ends with it.
                    await _terminate(proc, self._cog.terminate_grace_s)
                if (cancelled or fatal_exc is not None) and proc is not None:
                    self._closed = True
                    if proc.stderr is not None:
                        with contextlib.suppress(Exception):
                            stderr_bytes = await asyncio.wait_for(proc.stderr.read(), 0.5)

            if state.session_id:
                self.session_id = state.session_id

            if self._interrupted:
                # The CLI ends an interrupted turn as ``error_during_execution``
                # — the same subtype it uses for a genuine execution failure, so
                # the payload alone cannot tell them apart. WE know: we sent the
                # interrupt. Stamping it here beats inferring it from an
                # ambiguous field.
                #
                # ``is_error`` stays TRUE so the terminal event is marked
                # ``partial``: the turn stopped mid-answer and whatever text
                # arrived is a fragment. Only the reason changes, from "the CLI
                # hit a problem" to "we stopped it".
                state.stop_reason = "interrupted"

            result = await self._cog._finalise(
                agent=agent,
                ctx=ctx,
                state=state,
                cancelled=cancelled,
                fatal_exc=fatal_exc,
                spawned=proc is not None,
                # ``None`` = the process is still alive, which is the normal
                # end of a turn.
                return_code=proc.returncode if proc is not None else -1,
                stderr_bytes=stderr_bytes,
                schema_requested=schema_requested,
            )
            yield StreamEvent("final", usage=state.usage, result=result)
            if should_reraise_cancel:
                raise asyncio.CancelledError()


@dataclass(frozen=True, slots=True)
class InterruptReceipt:
    """What an interrupt actually achieved.

    ``delivered`` is False when there was nothing to interrupt — no live turn,
    or a process that has already exited. It is not an error: "stop" with
    nothing running is a no-op, and raising would make every UI wrap its stop
    button in a try block.

    ``still_queued`` names messages the CLI had queued and will still run, which
    is the difference between "the agent stopped" and "the agent stopped and
    has three more things to do". ``cancelled`` is populated only by an
    interrupt that asked to drop them.
    """

    delivered: bool
    still_queued: tuple[str, ...] = ()
    cancelled: tuple[str, ...] = ()
    error: str | None = None


class _SessionClosed(RuntimeError):
    """The session's process is gone (or was never usable for this turn).

    Surfaced through the normal terminal event as ``session_closed`` rather
    than raised at the caller, so a session turn keeps the same
    exactly-one-``final`` contract as a one-shot drive.
    """


def _user_turn(text: str) -> str:
    """One NDJSON user message for ``--input-format stream-json``.

    The shape is the SDK's own (``parent_tool_use_id`` marks a subagent turn;
    ``None`` is the main conversation) and was confirmed against the binary.
    """
    return (
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": text},
                "parent_tool_use_id": None,
            }
        )
        + "\n"
    )


# Credential env vars that make ``--bare`` viable. In bare mode the CLI never
# reads OAuth credentials or the system keychain, so a developer whose
# ``claude`` works perfectly in a terminal gets "Not logged in · Please run
# /login" the first time their service turns bare mode on — a message that
# points at exactly the wrong fix. The check below is deliberately
# conservative: any one of these, or a ``settings`` blob (which may carry an
# ``apiKeyHelper``), and we stay quiet.
_BARE_CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)


# ─────────────────────────────────────────────────────────────────────────────
# Stream-JSON parsing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _CliCall:
    """The minimum a ``Meter`` needs from a "call" it is charging.

    ``Budget.charge`` ignores its ``call`` entirely, but ``Quota`` reads
    ``call.ctx.scope.key()`` to partition per tenant — so a bare ``None`` would
    charge the run budget and crash the quota. ``request`` is present and
    ``None`` because a custom meter may look for it; there is no ``ChatRequest``
    here, and inventing one would be a lie.
    """

    ctx: Any
    request: Any = None


@dataclass(slots=True)
class _TurnState:
    """Everything one turn accumulates from the CLI's stream.

    Extracted so a persistent session and a one-shot drive fold the stream the
    same way. It is a plain accumulator: no decisions live here, only the
    "last value wins / text concatenates" rules the payload sequence implies.
    """

    text: str = ""
    thinking: str = ""
    usage: Usage = field(default_factory=Usage)
    session_id: str | None = None
    duration_ms: int | None = None
    is_error: bool = False
    stop_reason: str | None = None
    structured_output: Any = None
    init: dict[str, Any] = field(default_factory=dict)
    api_retries: list[dict[str, Any]] = field(default_factory=list)

    def fold(self, delta: _EventDelta) -> None:
        """Apply one payload's state delta. ``None`` means "this payload said
        nothing about that field", which is why every branch is a guard rather
        than an assignment."""
        if delta.text:
            self.text += delta.text
        if delta.thinking:
            self.thinking += delta.thinking
        if delta.usage is not None:
            self.usage = delta.usage
        if delta.session_id is not None:
            self.session_id = delta.session_id
        if delta.duration_ms is not None:
            self.duration_ms = delta.duration_ms
        if delta.is_error:
            self.is_error = True
        if delta.stop_reason is not None:
            self.stop_reason = delta.stop_reason
        if delta.structured_output is not None:
            self.structured_output = delta.structured_output
        if delta.init:
            self.init = delta.init
        if delta.api_retry is not None:
            self.api_retries.append(delta.api_retry)


@dataclass(slots=True)
class _EventDelta:
    """Small state-delta value returned alongside each yielded event so the
    drive loop can fold ``usage`` / ``session_id`` / ``text`` / ``thinking``
    / ``is_error`` without a second parsing pass."""

    text: str = ""
    thinking: str = ""
    usage: Usage | None = None
    session_id: str | None = None
    duration_ms: int | None = None
    # Populated by the ``result`` payload — the CLI can exit 0 with
    # ``is_error: true`` and a ``subtype`` like ``error_max_turns`` /
    # ``error_during_execution``. Surface both so ``drive()`` can flip the
    # ``AgentResult.partial`` flag and stamp a stop_reason.
    is_error: bool = False
    stop_reason: str | None = None
    # The CLI's validated structured answer, present on the ``result`` payload
    # only when ``--json-schema`` was passed AND the run produced a conforming
    # value. A ``success`` result WITHOUT it is a failure the docs call out
    # explicitly, so ``None`` here is meaningful and the drive loop needs to
    # distinguish "never asked for one" from "asked and did not get one".
    structured_output: Any = None
    init: dict[str, Any] | None = None  # ``system/init`` startup metadata
    api_retry: dict[str, Any] | None = None  # one ``system/api_retry`` payload


def _parse_line(line: bytes) -> dict[str, Any] | None:
    """Parse one stdout line as a JSON object; skip blank / non-JSON lines.

    The CLI occasionally emits a blank warmup line or a non-JSON diagnostic;
    those must NOT crash the loop — return ``None`` and let the caller move
    on.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


async def _events_from_payload(
    payload: dict[str, Any], *, partial: bool = False
) -> AsyncIterator[tuple[StreamEvent | None, _EventDelta]]:
    """Translate one stream-json payload into zero or more
    ``(StreamEvent | None, _EventDelta)`` tuples.

    Yields ``(None, delta)`` when the payload carries state (usage /
    session_id / duration) but no user-facing event — e.g., the ``system``
    init message.
    """
    ptype = payload.get("type")

    if ptype == "system":
        sid = payload.get("session_id")
        subtype = payload.get("subtype")
        if subtype == "api_retry":
            # The CLI retrying a failed API request. Surfaced as a ``step`` so
            # an operator watching a stream sees WHY a run went quiet for
            # thirty seconds instead of guessing.
            attempt = payload.get("attempt")
            retries = payload.get("max_retries")
            reason = payload.get("error") or "unknown"
            yield (
                StreamEvent("step", text=f"api_retry:{reason} ({attempt}/{retries})"),
                _EventDelta(session_id=str(sid) if sid else None, api_retry=dict(payload)),
            )
            return
        if subtype == "init":
            # Startup metadata: which model, which MCP servers connected, which
            # ones were SKIPPED. The error keys are omitted entirely when empty,
            # so their presence is the signal — that is what the CLI docs
            # recommend gating CI on.
            yield None, _EventDelta(
                session_id=str(sid) if sid else None,
                init={
                    k: payload[k]
                    for k in (
                        "model",
                        "capabilities",
                        "mcp_servers",
                        "mcp_server_errors",
                        "plugin_errors",
                        "claude_code_version",
                        "permissionMode",
                        "apiKeySource",
                    )
                    if k in payload
                },
            )
            return
        yield None, _EventDelta(session_id=str(sid) if sid else None)
        return

    if ptype == "stream_event":
        # Only present with ``--include-partial-messages``. One payload per
        # provider SSE event; the only ones a consumer can render are the
        # content deltas (``signature_delta`` / ``input_json_delta`` carry
        # cryptographic and tool-argument fragments, which are not text).
        event = payload.get("event") or {}
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            chunk = str(delta.get("text") or "")
            if chunk:
                # NOT accumulated: the completed ``assistant`` message is the
                # authoritative copy and folds the same text once.
                yield StreamEvent("message_delta", text=chunk), _EventDelta()
        elif dtype == "thinking_delta":
            chunk = str(delta.get("thinking") or "")
            if chunk:
                yield StreamEvent("message_delta", text=chunk), _EventDelta()
        return

    if ptype == "assistant":
        msg = payload.get("message") or {}
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text") or "")
                if text:
                    # Under partial streaming the consumer has already seen
                    # this text token by token; re-emitting the whole block
                    # would show every sentence twice. The state delta still
                    # flows, because this message is the authoritative copy.
                    yield (
                        None if partial else StreamEvent("message_delta", text=text),
                        _EventDelta(text=text),
                    )
            elif btype == "thinking":
                # Extended-thinking blocks (Opus / Sonnet with thinking on).
                # Surface as a ``message_delta`` so live consumers see the
                # reasoning chain in real time, but do NOT fold into
                # ``accumulated_text`` — that's the response, not the
                # reasoning. The drive loop tracks thinking separately and
                # hands it back via ``AgentResult.evals["thinking"]``.
                thinking = str(block.get("thinking") or "")
                if thinking:
                    yield (
                        None if partial else StreamEvent("message_delta", text=thinking),
                        _EventDelta(thinking=thinking),
                    )
            elif btype == "tool_use":
                tc = ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=dict(block.get("input") or {}),
                )
                yield StreamEvent("tool_call", tool_call=tc), _EventDelta()
        return

    if ptype == "user":
        # ``user`` messages in stream-json carry echoed tool_result blocks
        # (the CLI's post-execution reply to its own tool_use). Surface each
        # as a ``tool_result`` event so downstream observers see the pairing.
        msg = payload.get("message") or {}
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            content = block.get("content")
            # ``content`` may be a string or a list of {type,text} blocks —
            # collapse to a plain string for the StreamEvent payload.
            result_text = _flatten_tool_result_content(content)
            tc = ToolCall(id=str(tool_use_id or ""), name="", arguments={})
            yield (
                StreamEvent("tool_result", tool_call=tc, tool_result=result_text),
                _EventDelta(),
            )
        return

    if ptype == "result":
        usage_obj = payload.get("usage") or {}
        # Match Anthropic wire shape: `usage.input_tokens` /
        # `usage.output_tokens` / `usage.cache_read_input_tokens` /
        # `usage.cache_creation_input_tokens`. All optional; default to 0.
        usage = Usage(
            input_tokens=int(usage_obj.get("input_tokens") or 0),
            output_tokens=int(usage_obj.get("output_tokens") or 0),
            cost_usd=float(payload.get("total_cost_usd") or 0.0),
            cache_read_tokens=int(usage_obj.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage_obj.get("cache_creation_input_tokens") or 0),
        )
        duration = payload.get("duration_ms")
        session_id = payload.get("session_id")
        # The CLI can exit 0 with ``is_error: true`` and a ``subtype`` like
        # ``error_max_turns`` / ``error_during_execution``. Surface both so
        # ``drive()`` can honour the semantic outcome, not just the exit code.
        is_error = bool(payload.get("is_error"))
        subtype = payload.get("subtype")
        stop_reason = str(subtype) if is_error and isinstance(subtype, str) else None
        # ``error_max_structured_output_retries`` arrives as a subtype and is
        # NOT flagged ``is_error`` in every version, so key off the subtype
        # directly: the run produced no valid structured output and the caller
        # must not read the prose as if it were the object.
        if subtype == "error_max_structured_output_retries":
            stop_reason = str(subtype)
        yield (
            None,
            _EventDelta(
                usage=usage,
                session_id=str(session_id) if session_id else None,
                duration_ms=int(duration) if duration is not None else None,
                is_error=is_error,
                stop_reason=stop_reason,
                structured_output=payload.get("structured_output"),
            ),
        )
        return

    # Unknown type — ignore. The CLI can add new event types over time;
    # forward-compat is "don't crash".
    return


def _flatten_tool_result_content(content: Any) -> str:
    """Fold the CLI's tool_result content (string OR list of {type, text}
    blocks) into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


async def _terminate(proc: asyncio.subprocess.Process, grace_s: float) -> None:
    """Send SIGTERM, wait ``grace_s`` for a clean exit, then SIGKILL if the
    process is still alive. Never raises: this runs from a ``finally`` and
    must not mask the original ``Cancelled``.
    """
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()


__all__ = [
    "ClaudeCliCognition",
    "ClaudeCliSession",
    "InterruptReceipt",
    "PermissionMode",
]
