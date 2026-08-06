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
            allowed_tools=("Read", "Grep"),
        ),
    )
    result = await agent.run("Summarize README.md", ctx)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agentkit.agents.result import AgentResult
from agentkit.kernel.errors import AgentkitError
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import StreamEvent, ToolCall, Usage
from agentkit.prompts.prompt import Prompt

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.context import WorkingContext


PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


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


@dataclass(slots=True)
class ClaudeCliCognition:
    """Delegates the agent loop to a locally-installed ``claude`` CLI.

    Subprocesses ``claude -p "<task>" --output-format stream-json --verbose``
    per ``agent.run(...)`` / ``agent.stream(...)``. Uses whatever auth the
    CLI resolves — no API key handling on agentkit's side.

    Read from the agent: ``prompt`` (used as system prompt via
    ``--system-prompt``), ``model`` (via ``--model`` when set on the
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
    working_dir: Path | None = None
    config_dir: Path | None = None  # → CLAUDE_CONFIG_DIR
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    permission_mode: PermissionMode = "default"
    max_turns: int | None = None
    session_id: str | None = None  # resume an existing session
    extra_args: tuple[str, ...] = ()  # escape hatch for future CLI flags
    terminate_grace_s: float = 5.0
    max_concurrent: int = 8  # class-level semaphore (see module doc)
    _sem_ref: asyncio.BoundedSemaphore | None = field(default=None, init=False, repr=False)

    # ---- public surface --------------------------------------------------------------------

    async def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]:
        """Run the CLI once, mapping its stream-json output to agentkit
        ``StreamEvent``s. Guarantees one terminal ``final`` event on every
        exit path — success, non-zero exit, cancellation."""
        del context  # unused — the CLI owns its own transcript
        system_prompt = self._resolve_system_prompt(agent.prompt)
        argv = self._build_argv(task, system_prompt=system_prompt)
        env = self._build_env()

        # Bounded-concurrency spawn guard. Held for the whole CLI lifetime so
        # concurrent drives don't race the subprocess table (SDK issue #728).
        cfg_dir = str(self.config_dir) if self.config_dir is not None else None
        sem = _get_semaphore(self.claude_bin, cfg_dir, self.max_concurrent)
        self._sem_ref = sem  # keep alive for the WeakValueDictionary

        accumulated_text = ""
        usage = Usage()
        session_id: str | None = None
        duration_ms: int | None = None
        cancelled = False
        stderr_bytes: bytes = b""
        proc: asyncio.subprocess.Process | None = None

        async with sem:
            try:
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
                            if delta.usage is not None:
                                usage = delta.usage
                            if delta.session_id is not None:
                                session_id = delta.session_id
                            if delta.duration_ms is not None:
                                duration_ms = delta.duration_ms
                finally:
                    if cancelled and proc.returncode is None:
                        await _terminate(proc, self.terminate_grace_s)
                    # Collect stderr — best-effort; the pipe may already be closed.
                    if proc.stderr is not None:
                        with contextlib.suppress(Exception):
                            stderr_bytes = await proc.stderr.read()
                    with contextlib.suppress(Exception):
                        await proc.wait()
            finally:
                self._sem_ref = None

        return_code = proc.returncode if proc is not None else -1
        if cancelled:
            yield StreamEvent(
                "final",
                usage=usage,
                result=AgentResult(
                    output=accumulated_text,
                    usage=usage,
                    partial=True,
                    evals={
                        "stop_reason": "cancelled",
                        "session_id": session_id or "",
                        "cli_duration_ms": duration_ms or 0,
                        "cli_return_code": return_code,
                    },
                ),
            )
            return

        if return_code != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise AgentkitError(f"claude CLI exited with code {return_code}: {stderr_text or '<no stderr>'}")

        yield StreamEvent(
            "final",
            usage=usage,
            result=AgentResult(
                output=accumulated_text,
                usage=usage,
                evals={
                    "session_id": session_id or "",
                    "cli_duration_ms": duration_ms or 0,
                },
            ),
        )

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
            argv += ["--system-prompt", system_prompt]
        if self.allowed_tools:
            argv += ["--allowed-tools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            argv += ["--disallowed-tools", ",".join(self.disallowed_tools)]
        if self.permission_mode != "default":
            argv += ["--permission-mode", self.permission_mode]
        if self.max_turns is not None:
            argv += ["--max-turns", str(self.max_turns)]
        if self.session_id is not None:
            argv += ["--session-id", self.session_id]
        argv += list(self.extra_args)
        return argv

    def _build_env(self) -> dict[str, str]:
        """Copy the process env; layer ``CLAUDE_CONFIG_DIR`` on top when the
        cognition was constructed with a ``config_dir`` (for isolated auth /
        settings, e.g. per-tenant server-side wrapper). Also default
        ``CLAUDE_ENABLE_STREAM_WATCHDOG=1`` to mitigate long-tail SSE hangs
        (SDK issue #33949) unless the caller already set it explicitly."""
        env = os.environ.copy()
        if self.config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = str(self.config_dir)
        env.setdefault("CLAUDE_ENABLE_STREAM_WATCHDOG", "1")
        return env


# ─────────────────────────────────────────────────────────────────────────────
# Stream-JSON parsing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _EventDelta:
    """Small state-delta value returned alongside each yielded event so the
    drive loop can fold ``usage`` / ``session_id`` / ``text`` without a
    second parsing pass."""

    text: str = ""
    usage: Usage | None = None
    session_id: str | None = None
    duration_ms: int | None = None


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
        yield (
            None,
            _EventDelta(
                usage=usage,
                session_id=str(session_id) if session_id else None,
                duration_ms=int(duration) if duration is not None else None,
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
