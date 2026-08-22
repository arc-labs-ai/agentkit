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
from dataclasses import dataclass
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
    {"spawn_failed", "parse_failed", "working_dir_missing", "cli_reported_error"}
)


def _cli_stop_reason(reason: str | None) -> AgentStopReason:
    """Map this cognition's free-form terminal reason onto the closed taxonomy.

    ``None`` and ``"success"`` are completion; the failure set above and any
    ``cli_exit_<n>`` are ``"failed"``; everything else (``"cancelled"``) defers
    to the shared table.
    """
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

    Emits: ``message_delta`` for token chunks (one per ``assistant`` message
    text block streamed by the CLI), ``tool_call`` / ``tool_result`` for each
    CLI tool event, exactly one terminal ``final`` event carrying
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
        - ``ctx.budget`` — the CLI's cost accounting is surfaced via
          ``Usage.cost_usd`` on the final event but is NOT charged
          against ``ctx.budget`` (the cognition bypasses the ``Invoker``
          and its ``meter()`` middleware). Callers who need a hard
          ceiling on CLI spend must impose it externally.
        """
        del context  # unused — the CLI owns its own transcript

        # Bounded-concurrency spawn guard. Held for the whole CLI lifetime so
        # concurrent drives don't race the subprocess table (SDK issue #728).
        # The local ``sem`` reference keeps the WeakValueDictionary entry
        # alive for the ``async with sem`` scope — no instance-field
        # bookkeeping needed.
        cfg_dir = str(self.config_dir) if self.config_dir is not None else None
        sem = _get_semaphore(self.claude_bin, cfg_dir, self.max_concurrent)

        accumulated_text = ""
        accumulated_thinking = ""
        usage = Usage()
        session_id: str | None = None
        duration_ms: int | None = None
        cancelled = False
        is_error = False
        stop_reason: str | None = None
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
            argv = self._build_argv(task, system_prompt=system_prompt)
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
                        async for ev, delta in _events_from_payload(payload):
                            if ev is not None:
                                yield ev
                            if delta.text:
                                accumulated_text += delta.text
                            if delta.thinking:
                                accumulated_thinking += delta.thinking
                            if delta.usage is not None:
                                usage = delta.usage
                            if delta.session_id is not None:
                                session_id = delta.session_id
                            if delta.duration_ms is not None:
                                duration_ms = delta.duration_ms
                            if delta.is_error:
                                is_error = True
                            if delta.stop_reason is not None:
                                stop_reason = delta.stop_reason
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

        return_code = proc.returncode if proc is not None else -1

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
            if proc is None:
                if isinstance(fatal_exc, FileNotFoundError) and str(fatal_exc).startswith(
                    "working_dir does not exist:"
                ):
                    final_stop_reason = "working_dir_missing"
                else:
                    final_stop_reason = "spawn_failed"
            else:
                final_stop_reason = "parse_failed"
            final_partial = True
        elif return_code != 0:
            final_stop_reason = f"cli_exit_{return_code}"
            final_partial = True
        elif is_error:
            # When the CLI signals ``is_error: true`` while exiting 0, the
            # useful stop_reason is often on ``terminal_reason`` (e.g.,
            # ``"api_error"``) rather than ``subtype`` (which can still
            # read ``"success"``). We already prefer subtype in
            # ``_events_from_payload`` — but when subtype is the useless
            # ``"success"`` string, fall through to a generic marker.
            if stop_reason == "success" or stop_reason is None:
                final_stop_reason = "cli_reported_error"
            else:
                final_stop_reason = stop_reason
            final_partial = True
        else:
            final_stop_reason = stop_reason  # may still be None on clean success
            final_partial = False

        evals: dict[str, Any] = {
            "session_id": session_id or "",
            "cli_duration_ms": duration_ms or 0,
            "cli_return_code": return_code,
        }
        # Bridge agentkit's correlation_id into the final result so downstream
        # observability can join on it (matching the env var we set on the
        # subprocess via `CLAUDE_TRACE_EXTERNAL_ID`).
        external_id = getattr(ctx, "correlation_id", None)
        if external_id:
            evals["external_run_id"] = str(external_id)
        if final_stop_reason is not None:
            evals["stop_reason"] = final_stop_reason
        if accumulated_thinking:
            # Reasoning chain from ``thinking`` blocks — separate from
            # ``output`` (the final response). Live consumers already saw
            # each chunk as a ``message_delta``; this is the folded copy for
            # AgentResult callers.
            evals["thinking"] = accumulated_thinking
        if fatal_exc is not None:
            evals["error"] = f"{type(fatal_exc).__name__}: {fatal_exc}"
        if final_partial and stderr_bytes:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if stderr_text:
                evals["stderr"] = stderr_text

        yield StreamEvent(
            "final",
            usage=usage,
            result=AgentResult(
                output=accumulated_text,
                usage=usage,
                partial=final_partial,
                evals=evals,
                # This cognition reports failures as DATA (a terminal event is
                # guaranteed even when the subprocess never starts), so it is
                # the one producer that can legitimately stamp ``"failed"``.
                # Leaving the field at its default made a spawn failure and a
                # clean success indistinguishable to a typed reader.
                stop_reason=_cli_stop_reason(final_stop_reason),
            ),
        )
        if should_reraise_cancel:
            # Terminal event delivered; now propagate the cancel so
            # ``asyncio.wait_for(..., timeout=X)`` raises ``TimeoutError``
            # and TaskGroup cancels propagate to siblings.
            raise asyncio.CancelledError()


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

    def _build_argv(self, task: str, *, system_prompt: str) -> list[str]:
        """Assemble the CLI argv. Order-independent; kept grouped by role
        (identity → format → model → prompt → tools → permissions → resume →
        extras) so a diff is legible."""
        argv: list[str] = [
            self.claude_bin,
            "-p",
            task,
            "--output-format",
            "stream-json",
            "--verbose",
        ]
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
        if self.permission_mode != "default":
            argv += ["--permission-mode", self.permission_mode]
        if self.permission_prompt_tool is not None:
            argv += ["--permission-prompt-tool", self.permission_prompt_tool]
        if self.max_turns is not None:
            argv += ["--max-turns", str(self.max_turns)]
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
        if self.config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = str(self.config_dir)
        env.setdefault("CLAUDE_ENABLE_STREAM_WATCHDOG", "1")
        if ctx is not None:
            correlation_id = getattr(ctx, "correlation_id", None)
            if correlation_id:
                env.setdefault("CLAUDE_TRACE_EXTERNAL_ID", str(correlation_id))
        return env


# ─────────────────────────────────────────────────────────────────────────────
# Stream-JSON parsing
# ─────────────────────────────────────────────────────────────────────────────


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
    payload: dict[str, Any],
) -> AsyncIterator[tuple[StreamEvent | None, _EventDelta]]:
    """Translate one stream-json payload into zero or more
    ``(StreamEvent | None, _EventDelta)`` tuples.

    Yields ``(None, delta)`` when the payload carries state (usage /
    session_id / duration) but no user-facing event — e.g., the ``system``
    init message.
    """
    ptype = payload.get("type")

    if ptype == "system":
        # init metadata — capture session_id, no user-facing event
        sid = payload.get("session_id")
        yield None, _EventDelta(session_id=str(sid) if sid else None)
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
                    yield StreamEvent("message_delta", text=text), _EventDelta(text=text)
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
                        StreamEvent("message_delta", text=thinking),
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
        yield (
            None,
            _EventDelta(
                usage=usage,
                session_id=str(session_id) if session_id else None,
                duration_ms=int(duration) if duration is not None else None,
                is_error=is_error,
                stop_reason=stop_reason,
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


__all__ = ["ClaudeCliCognition", "PermissionMode"]
