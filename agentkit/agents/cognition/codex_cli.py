"""CodexCliCognition — delegate the agent loop to a locally-installed ``codex`` CLI.

Zero pip dependency. Users install OpenAI's ``codex`` CLI separately and this
cognition subprocesses it per ``agent.run(...)`` / ``agent.stream(...)`` with
``codex exec --json "<task>"``.

Auth is entirely the CLI's problem: whatever ``CODEX_API_KEY``,
``OPENAI_API_KEY``, or ``$CODEX_HOME`` login the CLI would find on its own is
what this cognition uses. agentkit itself never touches an API key here.

Wire it like any other cognition::

    from agentkit import Agent
    from agentkit.agents.cognition import CodexCliCognition

    agent = Agent(
        name="local",
        prompt="You are a concise assistant.",
        cognition=CodexCliCognition(
            model="gpt-5-codex",
            working_dir=Path("/tmp/sandbox"),
            sandbox="read-only",          # what the session may DO
            ask_for_approval="never",     # ...and whether it may pause to ask
        ),
    )
    result = await agent.run("Summarize README.md", ctx)


HOW THIS DIFFERS FROM ``ClaudeCliCognition``, AND WHY THE SURFACE IS NOT A MIRROR
--------------------------------------------------------------------------------
The two cognitions do the same job and are deliberately parallel wherever the
binaries are: same ``drive`` contract, same terminal-event guarantee, same
stop-reason taxonomy, same ``spawn=`` seam, same ``evals`` keys where the same
fact exists. Where they diverge, they diverge because the *programs* do, and
inventing a field to paper over that would be a field that silently does
nothing. The four that matter:

**Containment is a sandbox, not a tool list.** ``claude`` restricts what a
session HAS with ``--tools``; ``codex`` gives every session the same small
native toolbox (``shell``, ``apply_patch``, ``update_plan``, optionally
``web_search``) and restricts what those tools may DO with ``--sandbox`` and
``--ask-for-approval``. So there is no ``tools=``/``allowed_tools=`` here, and
:attr:`~CodexCliCognition.caps` — the Rule-of-Two tags ``RunPolicy`` reads —
are derived from the sandbox and search settings instead of from a tool table.
``sandbox="read-only"`` with ``web_search=True`` IS the lethal trifecta, and
``RunPolicy(mode="deny").check([cognition])`` refuses it.

**There is no system-prompt flag.** ``claude`` has ``--append-system-prompt``.
``codex`` has nothing equivalent, so ``agent.prompt`` reaches the model one of
two ways and both are stated rather than guessed at:
``system_prompt_mode="prepend"`` (the default) puts it at the top of the first
user message, and ``system_prompt_mode="replace"`` writes it to a file passed
as ``-c experimental_instructions_file=…``, which REPLACES Codex's own base
instructions the way ``--system-prompt`` replaces Claude Code's.

**The CLI does not report cost.** ``claude``'s ``result`` payload carries
``total_cost_usd``; ``codex``'s ``turn.completed`` carries only token counts.
``Usage.cost_usd`` is therefore *computed* from
:func:`agentkit.adapters.llm.providers.pricing.cost` and is ``0.0`` for a model
that table does not know — which is most Codex models. Inject ``pricing=`` for
a number you can bill against, and read ``evals["cost_source"]`` before you
treat the field as authoritative. There is also no ``--max-budget-usd``: the
budget is charged *after* the run and refused *before* it, but nothing hands
the CLI a ceiling it can stop itself against mid-flight.

**A session is a resumed thread, not a held process.** ``codex exec`` is
one-shot; its continuation seam is ``codex exec resume <thread-id>``. So
:class:`CodexCliSession` spawns per turn and threads the conversation through
the thread id, which costs a warm-up per turn but buys two things the Claude
session cannot have: a cancelled turn does NOT end the conversation, and
per-turn structured output works, because ``--output-schema`` is chosen at
spawn and every turn is its own spawn.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agentkit.agents.cognition._cli_common import (
    CliLineTooLong,
    CliSpawn,
    CliTimedOut,
    CliTimeouts,
    EnvPolicy,
    StructuredOutputFailure,
    _as_dict,
    _as_int,
    _as_list,
    _build_child_env,
    _charge_meters,
    _CliLaunch,
    _CliRunOutcome,
    _coerce_structured,
    _get_semaphore,
    _is_working_dir_missing,
    _looks_secret,
    _redact_secrets,
    _require_working_dir,
    _reraise_if_not_an_exception,
    _run_cli_process,
    _timeout_stop_reason,
    _tool_middleware_names,
    _validate_json_schema,
)
from agentkit.agents.result import AgentResult, AgentStopReason, stop_reason_for
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import StreamEvent, ToolCall, Usage
from agentkit.prompts.prompt import Prompt

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.context import WorkingContext


# ``--sandbox``: what the session's own tools may DO. This is Codex's whole
# containment story — there is no tool allow-list — so it is also where the
# Rule-of-Two tags come from.
SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]

# ``--ask-for-approval``: when the CLI pauses to ask a human. ``never`` is the
# right value for a service (nobody is at the terminal to answer), which makes
# ``sandbox`` the only thing standing between the model and the machine —
# stated here because the two flags are routinely set as if either alone were
# the control.
ApprovalMode = Literal["untrusted", "on-failure", "on-request", "never"]

# ``-c model_reasoning_effort=…``. Not a top-level flag on ``codex`` the way
# ``--effort`` is on ``claude``; it is a config key, and passing it as one is
# what keeps this working across CLI versions that have not promoted it.
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


# Reasons this cognition emits that mean "the run ERRORED", as opposed to
# "something stopped it deliberately". ``cli_exit_<n>`` is dynamic, which is
# why the mapping lives here rather than in the framework-wide table: only this
# module knows how it spells its own failures.
_CODEX_FAILURE_REASONS = frozenset(
    {
        "spawn_failed",
        "parse_failed",
        # A single NDJSON payload outgrew the reader's buffer. ``failed``, and
        # deliberately NOT ``parse_failed``: the line was valid JSON, just too
        # big to assemble, so the fix is a size rather than a corruption hunt.
        "output_line_too_long",
        "working_dir_missing",
        # The CLI wrote a top-level ``{"type":"error"}`` — a transport or
        # stream failure it could not attribute to the turn.
        "cli_reported_error",
        # ``{"type":"turn.failed"}`` — the turn itself failed. Kept distinct
        # from ``cli_reported_error`` because the message differs in kind: one
        # is "the model run went wrong", the other is "the pipe broke".
        "turn_failed",
        # A turn was asked of a session with no thread to resume. ``failed``
        # rather than ``terminated``: nobody chose to stop this.
        "session_closed",
        # ── liveness (see CliTimeouts) ──────────────────────────────────────
        # ``failed`` and not ``expired``: ``expired`` means a human-gate
        # deadline passed and the run degraded and CONTINUED. Nothing continues
        # here. And not ``terminated``: nobody chose this, a bound did.
        "startup_timeout",
        "first_event_timeout",
        "idle_timeout",
        "total_timeout",
        "authentication_failed",
        # ── stream integrity ────────────────────────────────────────────────
        # The stream ended without the CLI's own end-of-turn payload: a process
        # killed from outside, a full disk, a broken pipe, a truncated final
        # line. Measured before this reason existed — the run reported
        # ``stop_reason="complete"``, ``partial=False``, and handed back half an
        # answer as a finished one, which is the worst shape a failure can take.
        "malformed_output",
        # A death by signal rather than a chosen exit, and a schema the CLI
        # refused at startup. Both are refinements of ``cli_exit_<n>``.
        "process_crashed",
        "schema_rejected",
    }
)

# Phrases ``codex`` prints when authentication has FAILED. Lower-case,
# substring-matched against stderr, and used only to refine an already-failed
# ``startup_timeout`` — see ``_timeout_stop_reason`` for why the list must not
# contain anything the CLI prints while working.
_CODEX_AUTH_FAILURE_STDERR: tuple[str, ...] = (
    "not logged in",
    "run `codex login`",
    "codex login",
    "unauthorized",
    "invalid_api_key",
    "invalid api key",
    "401",
)


def _classify_timeout(exc: CliTimedOut, stderr_bytes: bytes) -> str:
    """This CLI's spelling of :func:`_timeout_stop_reason`."""
    return _timeout_stop_reason(exc, stderr_bytes, _CODEX_AUTH_FAILURE_STDERR)

# Structured-output failures are ``invalid_output`` in the closed taxonomy —
# the same category the tool loop uses when parse-and-repair is exhausted. They
# are NOT ``failed``: the run itself worked, the shape did not.
_CODEX_INVALID_OUTPUT_REASONS = frozenset(
    {
        "structured_output_missing",
        "structured_output_mismatch",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Capability tags for a Codex session, in ``RunPolicy``'s vocabulary.
#
# ``ClaudeCliCognition`` derives these per built-in tool, because that CLI's
# containment IS a tool list. Codex's is not: every session gets the same
# ``shell`` and the sandbox decides what it reaches. So the table is keyed on
# the sandbox, and this is the honest translation of it:
#
# * Every mode can READ the workspace — that is what the agent is for — so
#   ``private_data`` is in all three. A "read-only" sandbox is read-only about
#   WRITES; it is not blind.
# * ``egress`` belongs to ``danger-full-access`` alone, because the other two
#   modes disable network access by default. ``network_access=True`` adds it to
#   ``workspace-write``, which is exactly what that flag turns on.
# * ``untrusted_content`` is not a sandbox property at all. It arrives with
#   ``--search`` (see :attr:`CodexCliCognition.web_search`), and it brings
#   ``egress`` with it — a web search is a network call by definition.
#
# The consequence worth stating out loud, because it is the case people wire by
# accident: ``sandbox="read-only", web_search=True`` is private data plus
# untrusted content plus egress. That is the lethal trifecta in the most
# innocuous-looking configuration Codex has, and ``RunPolicy(mode="deny")``
# refuses it.
# ─────────────────────────────────────────────────────────────────────────────
CODEX_SANDBOX_CAPS: dict[str, tuple[str, ...]] = {
    "read-only": ("private_data",),
    "workspace-write": ("private_data",),
    "danger-full-access": ("private_data", "egress"),
}

# The CLI's own tools, as they appear in this cognition's ``tool_call`` events.
# Not a restriction knob — Codex has none — but the middleware-bypass warning
# has to be able to NAME what the chain is not applying to, and a warning that
# says "the CLI's tools" teaches nobody anything.
CODEX_NATIVE_TOOLS: tuple[str, ...] = ("shell", "apply_patch", "update_plan")

# Ambient variables that decide WHICH identity the CLI runs as. Removed by
# ``env_policy="profile"``, absent under ``"isolated"``.
#
# Shorter than the Claude list, and honestly so. Measured against codex
# 0.152.1: a bogus ``OPENAI_API_KEY`` alongside a valid ``CODEX_HOME`` login was
# ignored and the run succeeded — the opposite of the Claude CLI, which prefers
# the ambient key and says so on stderr. There is no session-linkage half here
# either, because nothing in a running Codex process exports handles to itself
# the way ``CLAUDE_CODE_MESSAGING_TOKEN`` and friends do.
#
# Kept anyway: precedence is a config key (``preferred_auth_method``) rather
# than a law, ``--ignore-user-config`` changes which config is consulted, and a
# service that has declared a policy should get the same guarantee from both
# cognitions rather than one that happens to be unnecessary today.
_CODEX_AMBIENT_AUTH_ENV: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "OPENAI_BASE_URL",
)

# What a Codex sandbox mode is called when nobody passed ``--sandbox``. The CLI
# documents ``codex exec`` as defaulting to read-only, and the caps property
# has to answer for the default configuration too — reporting no capabilities
# for "the flag was not passed" would make the trifecta gate silently blind on
# the most common wiring there is.
_DEFAULT_SANDBOX: SandboxMode = "read-only"

# Codex's ``turn.completed`` usage counts input tokens INCLUSIVE of the cached
# prefix, and reports the cached part separately. agentkit's convention is the
# other one: ``Usage.input_tokens`` is the FRESH (billed-at-full-rate) input
# and ``cache_read_tokens`` is the discount. Subtracting is therefore not a
# tidy-up, it is the difference between a cost estimate and a cost estimate
# that double-counts every cached token in a long session — and Codex sessions
# are mostly cache.
def _split_input_tokens(total_input: int, cached: int) -> tuple[int, int]:
    """``(fresh, cached)`` from Codex's inclusive input count.

    Clamped at zero rather than trusted: the two numbers come from different
    places in the CLI and a cached count larger than the total would otherwise
    produce a negative ``input_tokens``, which flows straight into a meter and
    makes a budget go UP.
    """
    cached = max(0, cached)
    return max(0, total_input - cached), cached



# Phrases each CLI prints when it has REJECTED THE SCHEMA we handed it, as
# opposed to failing for any other reason. Same discipline as the auth markers:
# only ever refines an already-failed run, and must not contain anything the
# CLI prints while working.
_SCHEMA_REJECTED_STDERR: tuple[str, ...] = (
    "is not a valid json schema",
    "invalid json schema",
    "--json-schema",
    "--output-schema",
    "output schema",
)


def _exit_stop_reason(return_code: int, stderr_bytes: bytes) -> str:
    """Name a non-zero exit more precisely than ``cli_exit_<n>`` where we can.

    Three outcomes, in decreasing confidence:

    ``process_crashed``
        A NEGATIVE return code is a death by signal, not a chosen exit — the
        OOM killer, a segfault, an operator's ``kill``. ``cli_exit_-9`` reads
        as an exit status that does not exist; ``process_crashed`` names what
        happened and the signal is kept alongside. Only reachable for a signal
        WE did not send: cancellation and timeouts are decided earlier and win.

    ``schema_rejected``
        The CLI refused the schema at startup. Pre-spawn validation catches the
        structural mistakes, but each binary has its own additional rules, and
        "exit 2" for a bad schema is a message an operator has to dig out of
        stderr to understand.

    ``cli_exit_<n>``
        Everything else, unchanged.
    """
    if return_code < 0:
        return "process_crashed"
    haystack = stderr_bytes.decode("utf-8", errors="replace").lower()
    if any(marker in haystack for marker in _SCHEMA_REJECTED_STDERR):
        return "schema_rejected"
    return f"cli_exit_{return_code}"


def _codex_stop_reason(reason: str | None) -> AgentStopReason:
    """Map this cognition's free-form terminal reason onto the closed taxonomy.

    ``None`` and ``"success"`` are completion; the failure set above and any
    ``cli_exit_<n>`` are ``"failed"``; everything else (``"cancelled"``) defers
    to the shared table.
    """
    if reason in _CODEX_INVALID_OUTPUT_REASONS:
        return "invalid_output"
    if reason in _CODEX_FAILURE_REASONS or (reason is not None and reason.startswith("cli_exit_")):
        return "failed"
    return stop_reason_for(reason)


def _default_pricing(model: str | None, usage: Usage) -> float:
    """The framework's own best-effort price table.

    Imported lazily so this module stays importable with nothing installed and
    so the table's staleness is a runtime concern a caller can override, not an
    import-time dependency they cannot.
    """
    from agentkit.adapters.llm.providers.pricing import cost

    return cost(model, usage)


@dataclass(slots=True)
class CodexCliCognition:
    """Delegates the agent loop to a locally-installed ``codex`` CLI.

    Subprocesses ``codex exec --json "<task>"`` per ``agent.run(...)`` /
    ``agent.stream(...)``. Uses whatever auth the CLI resolves — no API key
    handling on agentkit's side.

    Read from the agent: ``prompt`` (prepended to the task, or written to an
    instructions file with ``system_prompt_mode="replace"``; see the module
    docstring for why there is no flag for this), ``output`` (becomes
    ``--output-schema``). The agent's ``model`` field is NOT consulted — the
    cognition's own ``model`` wins, so a caller can point the CLI at a
    different model than the rest of the agentkit chain.

    Emits: ``message_delta`` per assistant text (and per reasoning summary);
    ``tool_call`` / ``tool_result`` for each shell command, patch, MCP call and
    web search; ``step`` for a plan update or a non-fatal item error; exactly
    one terminal ``final`` event carrying ``AgentResult(output=<assistant
    text>, usage=<Usage>, evals={"session_id": ..., "cli_duration_ms": ...})``.

    **The middleware chain does not apply to a native CLI tool call.** The CLI
    bypasses the ``Invoker``, so for every ``shell`` command, ``apply_patch``
    and ``web_search`` the CLI runs inside its own process there is no
    ``egress()`` URL check, no ``Guardrail.check_url`` SSRF or allowlist check,
    no ``security()`` input guard, no ``audit()`` record, and no ``memoize()``
    / ``idempotent()`` key. Not "reduced" — absent. This is the identical hole
    ``ClaudeCliCognition`` documents, and it is worse here in one specific way
    and better in another:

    * Worse: Codex has no ``PreToolUse`` hook, so the
      ``agentkit.integrations.claude_cli.hook_settings`` escape hatch has no
      counterpart. There is no way to make the chain reach ``shell``.
    * Better: Codex's containment is an OS sandbox rather than a tool
      allow-list, so ``sandbox="read-only"`` is enforced below the CLI by the
      operating system rather than by the model's cooperation.

    Two things are done about it here, neither of which is a fix:

    * The first ``drive`` on a context that carries tool middleware **warns and
      names the middlewares that will not apply**. The failure is invisible by
      construction, so a generic "middleware may not apply" would leave the
      reader no better off.
    * :attr:`caps` reports the session's Rule-of-Two tags, so ``RunPolicy`` can
      refuse a Codex session for the same lethal trifecta it refuses an
      agentkit tool set for: ``RunPolicy(mode="deny").check([cognition,
      *other_tools])``.

    Serving your own tools over MCP instead — ``mcp_servers=...`` — routes
    those calls back through agentkit's own tool path, where the chain does
    apply. The CLI's own ``shell`` is unreachable either way; the sandbox is
    what contains it.
    """

    name: str = "codex_cli"
    codex_bin: str = "codex"
    model: str | None = None

    # How ``agent.prompt`` reaches the model. Codex has no
    # ``--append-system-prompt``, so both options are constructions rather than
    # flags and the difference is worth choosing deliberately:
    #
    #   "prepend"  the prompt is the first thing in the first user message,
    #              above the task, separated by a rule. Codex's own base
    #              instructions (tool guidance, sandbox explanation, apply_patch
    #              format) stay in place. This is the default for the same
    #              reason ``append`` is Claude's: a replacement turns a capable
    #              coding agent into a chat model that still has tools it no
    #              longer knows how to drive.
    #   "replace"  the prompt is written to a temp file passed as
    #              ``-c experimental_instructions_file=…``, which REPLACES the
    #              base instructions entirely. The config key is marked
    #              experimental by the CLI, so this mode is the one that will
    #              break first on a CLI upgrade — deliberate, since the
    #              alternative is silently not replacing anything.
    system_prompt_mode: Literal["prepend", "replace"] = "prepend"

    # ── environment: where the session runs and what it can reach ───────────
    # ``--cd``: the workspace root. Also passed as the subprocess ``cwd`` so
    # relative paths in the task text mean what the caller thinks they mean.
    working_dir: Path | None = None
    # → ``CODEX_HOME``. Isolated auth + config + session history, which is what
    # a per-tenant server-side wrapper needs; the CLI keeps all three there.
    config_home: Path | None = None
    # ``--add-dir``: directories outside ``working_dir`` the session may also
    # write to. Existence is checked at construction, matching what the CLI
    # itself validates, because a typo'd path is otherwise a subprocess that
    # dies three seconds in.
    add_dirs: tuple[Path | str, ...] = ()

    # ── containment ─────────────────────────────────────────────────────────
    # ``--sandbox``. ``None`` passes no flag, leaving the CLI's own default
    # (read-only for ``codex exec``) and whatever the user's config.toml says.
    # Pass one explicitly for a service: "whatever the machine happens to be
    # configured for" is not a containment decision.
    sandbox: SandboxMode | None = None
    # ``--ask-for-approval``. In a service this should be ``"never"`` — there
    # is nobody at the terminal — which is precisely why ``sandbox`` has to be
    # set too. The CLI's interactive default assumes a human.
    ask_for_approval: ApprovalMode | None = None
    # ``--dangerously-bypass-approvals-and-sandbox``. Refused at construction
    # alongside an explicit ``sandbox``/``ask_for_approval``, because the two
    # would be a contradiction the CLI resolves silently.
    bypass_sandbox: bool = False
    # ``-c sandbox_workspace_write.network_access=true``. Only meaningful under
    # ``workspace-write``; it is what turns that sandbox's network back on, and
    # it is why ``caps`` grows an ``egress`` tag when it is set.
    network_access: bool = False
    # ``--search``: the CLI's own web search tool. Brings BOTH
    # ``untrusted_content`` and ``egress`` into :attr:`caps` — see the module
    # comment on ``CODEX_SANDBOX_CAPS`` for the configuration this makes
    # refusable.
    web_search: bool = False

    # ── structured output ───────────────────────────────────────────────────
    # ``--output-schema``: the CLI constrains its final message to this JSON
    # Schema. Leave ``None`` and the schema is taken from ``agent.output`` when
    # one is declared, so the same ``output=`` that types a normal agentkit run
    # types a CLI-delegated one.
    #
    # The flag takes a FILE PATH, not inline JSON (this is where it differs
    # from ``claude --json-schema``), so the schema is written to a temp file
    # under a 0700 directory for the life of the spawn and removed after. A
    # caller never sees the file and must not be asked to manage one.
    json_schema: dict[str, Any] | None = None
    # ``--output-last-message``: a path the CLI also writes the final message
    # to. Not needed by this cognition — the JSON stream carries it — and
    # exposed only because a caller may want the file for something else.
    output_last_message: Path | None = None

    # ── reproducibility ─────────────────────────────────────────────────────
    # ``--ignore-user-config``: do not read ``$CODEX_HOME/config.toml``. Auth
    # still comes from ``CODEX_HOME``, so unlike ``claude --bare`` this does
    # not strand the run without credentials.
    ignore_user_config: bool = False
    # ``--ignore-rules``: do not read user or project execpolicy ``.rules``
    # files. Together with the above this is what makes a run reproducible
    # across machines — without them, a rule file in a teammate's home
    # directory changes what your service is allowed to do.
    ignore_rules: bool = False
    # ``--strict-config``: fail on an unrecognised key in config.toml rather
    # than ignoring it. Worth turning on in CI: a silently-ignored key is a
    # setting the operator believes is in force.
    strict_config: bool = False
    # ``--skip-git-repo-check``: Codex refuses to run outside a git repo by
    # default, which is a safety feature (it is about to edit files) and an
    # obstacle for a service whose workspace is a scratch directory.
    skip_git_repo_check: bool = False
    # ``--ephemeral``: do not write session files to disk. For a multi-tenant
    # service this is a containment control, not an optimisation — and it is
    # incompatible with :meth:`session`, which resumes by thread id.
    ephemeral: bool = False
    # ``--oss`` / ``--profile``: a local open-source provider, and a named
    # config profile layered over the base config.
    oss: bool = False
    profile: str | None = None

    # ``-c key=value`` overrides, and the MCP servers that are the main reason
    # to reach for them. ``mcp_servers`` is a mapping of server name → the
    # config table Codex expects (``command``/``args``/``env`` for stdio,
    # ``url``/``bearer_token_env_var`` for streamable HTTP); it is flattened
    # into ``-c mcp_servers.<name>.<key>=<json>`` for you, because hand-writing
    # that is the step where a caller gets the TOML quoting wrong and finds out
    # three seconds into a run.
    mcp_servers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    # Extra environment for the child, layered over this process's ``os.environ``.
    # It exists for the one thing ``mcp_servers`` cannot express: Codex reads an
    # HTTP MCP server's bearer token from an env var it names
    # (``bearer_token_env_var``) rather than from the config, so the token has
    # to reach the child some other way. A CALLER-set value wins over the
    # parent's, because the point is to override.
    env: Mapping[str, str] = field(default_factory=dict)
    effort: ReasoningEffort | None = None
    images: tuple[Path | str, ...] = ()

    # ── session identity ────────────────────────────────────────────────────
    # Codex names its own threads, so there is no counterpart to
    # ``claude --session-id``: you cannot pick the id in advance. Only the two
    # continuations exist.
    #
    #   resume_session_id  codex exec resume <id>      CONTINUE a specific thread
    #   continue_session   codex exec resume --last    CONTINUE the latest one
    resume_session_id: str | None = None
    continue_session: bool = False

    extra_args: tuple[str, ...] = ()  # escape hatch for future CLI flags

    # ── liveness ────────────────────────────────────────────────────────────
    # The same four bounds as the Claude cognition, with the same defaults and
    # the same reasons — see :class:`CliTimeouts`. One Codex-specific note on
    # ``idle``: this CLI's ``shell`` tool is the whole point of it, so a run
    # that is silently compiling for four minutes is doing its job. An ``idle``
    # tuned to model latency will kill working runs here even more readily than
    # it will on the Claude side.
    timeouts: CliTimeouts = field(default_factory=CliTimeouts)
    terminate_grace_s: float = 5.0
    max_concurrent: int = 8  # class-level semaphore, shared per (bin, home, max)

    # ── environment and authentication ──────────────────────────────────────
    # The same knob as ``ClaudeCliCognition.env_policy``, and deliberately the
    # same three names — but the urgency is NOT the same, and saying so is the
    # point of this comment.
    #
    # Measured, codex 0.152.1: a bogus ``OPENAI_API_KEY`` in the environment was
    # IGNORED in favour of the ``CODEX_HOME`` login, and the run succeeded. The
    # Claude CLI does the opposite and says so on stderr. So there is no live
    # credential-override bug here to fix; what this buys is determinism (the
    # child gets what we chose, not what the operator's shell had), symmetry
    # for a caller wiring both cognitions behind one policy, and cover for the
    # fact that Codex's precedence is itself configurable and a CLI upgrade
    # away from flipping.
    #
    # ``None`` resolves to "profile" when ``config_home`` is set and "inherit"
    # otherwise, matching the Claude cognition's rule.
    env_policy: EnvPolicy | None = None

    # ── native tool policy ──────────────────────────────────────────────────
    # Same field as the Claude cognition, and the one place where the same
    # value means something genuinely different — because the programs differ.
    #
    # Claude can satisfy "deny": ``tools=("",)`` plus ``mcp_config=`` leaves the
    # session with no ungoverned native tool. **Codex cannot.** Every Codex
    # session has ``shell``, there is no tool allow-list, and there is no
    # PreToolUse hook to route the call back through agentkit. So
    # ``native_tool_policy="deny"`` here does not configure anything away — it
    # refuses to construct, which is the honest answer to "my policy is that no
    # ungovernable tool may run" and is exactly what a deny policy is for. A
    # service that sets it learns at startup that Codex cannot meet it, instead
    # of never learning.
    native_tool_policy: Literal["deny", "warn", "allow"] = "warn"

    # ── spend ───────────────────────────────────────────────────────────────
    # Half of what the Claude cognition gets. There is no ``--max-budget-usd``
    # on ``codex``, so the CLI cannot stop itself mid-flight against the run's
    # remaining headroom; all this can do is refuse to spawn when the budget is
    # already gone, and charge what the run cost once it ends. Stated because
    # "the budget is wired" reads as a hard ceiling and here it is not one.
    meter_spend: bool = True
    # ``(model, usage) -> usd``. The CLI reports no cost at all, so this is the
    # ONLY source of one. The default is agentkit's best-effort public price
    # table, which returns 0.0 for a model it does not know — most Codex models
    # today. A run whose cost matters passes its own contractual rates here.
    pricing: Callable[[str | None, Usage], float] | None = None

    # ── the transport seam ──────────────────────────────────────────────────
    # How the subprocess gets created. ``None`` — the default, and the only
    # value production ever uses — means ``asyncio.create_subprocess_exec``.
    # See ``agentkit.testing.fakes.FakeCodexCli``, which is the intended
    # occupant, and ``ClaudeCliCognition.spawn`` for the full argument.
    spawn: CliSpawn | None = None

    # Latch for the middleware-bypass warning. Per INSTANCE — not per drive (a
    # warning that repeats every iteration is noise, and noise is what teaches
    # people to add a ``filterwarnings`` line) and not per process (two
    # cognitions in one service are usually two configurations, and the second
    # one is the one nobody has audited).
    _bypass_warned: bool = field(default=False, init=False, repr=False, compare=False)
    # Same latch discipline for the env-strip notice: once per instance.
    _env_warned: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse combinations the CLI itself refuses, at construction.

        Every one of these is cheaper to catch here than as a subprocess that
        exits non-zero three seconds later with a message the caller has to
        parse out of stderr.
        """
        if self.continue_session and self.resume_session_id is not None:
            raise ValueError(
                "CodexCliCognition: continue_session resumes the most recent thread and "
                "resume_session_id resumes a specific one — pass one, not both"
            )
        if self.bypass_sandbox and (self.sandbox is not None or self.ask_for_approval is not None):
            raise ValueError(
                "CodexCliCognition: bypass_sandbox turns OFF both the sandbox and the "
                "approval prompt, so passing sandbox= or ask_for_approval= alongside it "
                "states a policy that will not be in force. Pass one or the other."
            )
        if self.network_access and self.sandbox not in (None, "workspace-write"):
            raise ValueError(
                "CodexCliCognition: network_access sets sandbox_workspace_write.network_access, "
                f"which only applies under sandbox='workspace-write' (got {self.sandbox!r}). "
                "Under 'read-only' it does nothing; under 'danger-full-access' the network is "
                "already open and the flag reads as a restriction that is not there."
            )
        if self.ephemeral and (self.continue_session or self.resume_session_id is not None):
            raise ValueError(
                "CodexCliCognition: ephemeral=True means the CLI writes no session file, so "
                "there is nothing for resume_session_id/continue_session to resume. Drop one."
            )
        missing_dirs = [str(d) for d in self.add_dirs if not Path(d).is_dir()]
        if missing_dirs:
            raise ValueError(f"CodexCliCognition: add_dirs entries are not directories: {missing_dirs}")
        # ``-c key=value`` is the ONLY way to set a Codex config override, and
        # it is an argument — ``codex exec`` has no config-file flag, so unlike
        # the Claude cognition there is nowhere safe to move a secret to.
        # Warned rather than refused: the value may be a placeholder, an
        # already-scoped short-lived token, or a key the caller genuinely
        # accepts on this host, and refusing would break a working
        # configuration over a name heuristic.
        exposed = sorted(k for k in self.config_overrides if _looks_secret(k, None))
        for server, table in self.mcp_servers.items():
            exposed += sorted(
                f"mcp_servers.{server}.{k}" for k in table if _looks_secret(k, None)
            )
        if exposed:
            import warnings

            warnings.warn(
                f"CodexCliCognition: {', '.join(exposed)} will be rendered into the "
                "process argument list as `-c key=value`, where any local account can "
                "read it with `ps`. `codex exec` has no config-file flag, so agentkit "
                "cannot move it out of argv the way it does for the Claude cognition's "
                "inline --mcp-config/--settings blobs. Codex's own answer is to keep the "
                "secret in the ENVIRONMENT and name it from the config: set "
                "mcp_servers[...]['bearer_token_env_var'] = 'MY_TOKEN' and pass "
                "env={'MY_TOKEN': ...}, which this cognition already supports.",
                UserWarning,
                stacklevel=3,
            )
        missing_images = [str(i) for i in self.images if not Path(i).is_file()]
        if missing_images:
            raise ValueError(f"CodexCliCognition: images entries are not files: {missing_images}")
        if self.native_tool_policy == "deny":
            raise ValueError(
                "CodexCliCognition(native_tool_policy='deny') cannot be satisfied. Every "
                "Codex session has the `shell` tool — there is no tool allow-list and no "
                "PreToolUse hook — so the CLI runs it inside its own process where "
                "agentkit's tool middleware (egress/SSRF checks, input guards, audit "
                "records, idempotency keys) cannot reach it. A deny policy means an "
                "ungovernable tool must not start, and here that is the whole cognition. "
                "Use ClaudeCliCognition(tools=('',), mcp_config=...) if the policy has to "
                "hold, or set native_tool_policy='warn' and rely on the OS sandbox "
                f"(sandbox={self.effective_sandbox!r}) as the containment that does apply."
            )
        if self.env_policy is not None and self.env_policy not in (
            "inherit",
            "profile",
            "isolated",
        ):
            raise ValueError(
                f"CodexCliCognition: env_policy must be 'inherit', 'profile' or 'isolated', "
                f"got {self.env_policy!r}"
            )
        if self.json_schema is not None:
            # Before the spawn. The Codex path writes the schema to a temp file
            # and the CLI reads it there, so a malformed one surfaces as a bare
            # non-zero exit with the complaint on stderr — even less legible
            # than Claude's, which at least names the flag.
            _validate_json_schema(self.json_schema, who="CodexCliCognition")
        for server, table in self.mcp_servers.items():
            if not isinstance(table, Mapping) or not table:
                raise ValueError(
                    f"CodexCliCognition: mcp_servers[{server!r}] must be a non-empty mapping of "
                    "config keys (command/args/env for stdio, url/bearer_token_env_var for HTTP)"
                )
            if "command" not in table and "url" not in table:
                raise ValueError(
                    f"CodexCliCognition: mcp_servers[{server!r}] declares neither 'command' "
                    "(a stdio server) nor 'url' (a streamable-HTTP one), so the CLI has no way "
                    f"to start or reach it. Got keys: {sorted(table)}"
                )

    # ---- public surface --------------------------------------------------------------------

    async def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]:
        """Run the CLI once, mapping its JSONL output to agentkit ``StreamEvent``s.

        **Terminal event guarantee.** Exactly one ``final`` event is yielded on
        every exit path — success, cancellation, non-zero CLI exit, a
        ``turn.failed`` payload, and any exception raised before or during the
        spawn (e.g. ``FileNotFoundError`` if the ``codex`` binary isn't on
        PATH). Callers may drive the loop with ``async for ev in
        agent.stream(...)`` and rely on seeing one and only one
        ``StreamEvent(type='final')`` regardless of outcome. On failure paths,
        ``AgentResult.partial=True`` and ``evals["stop_reason"]`` names the
        failure mode.

        The guarantee is about the ``final`` event, not about swallowing. A
        ``BaseException`` that is not an ``Exception`` — ``KeyboardInterrupt``,
        ``SystemExit``, a test double's contract violation — is delivered as a
        terminal event AND THEN re-raised, the same order ``CancelledError``
        uses.

        **`asyncio.CancelledError` semantics.** When the caller wraps this in
        ``asyncio.wait_for(...)`` or a ``TaskGroup`` and cancels, we terminate
        the subprocess, yield the terminal ``final(stop_reason="cancelled")``,
        AND re-raise so the caller's cancel / timeout mechanism sees the
        signal.

        **What this cognition IGNORES from ``ctx`` and ``agent``.** By design,
        this cognition delegates the whole loop to the CLI, so several agentkit
        contracts do NOT apply:

        - ``ctx.autonomy`` — ``sandbox`` / ``ask_for_approval`` own permissions;
          agentkit's autonomy tier is not translated.
        - ``ctx.invoker.tool_middleware`` — NOT RUN for a native CLI tool call.
          The first drive on a context that carries tool middleware warns and
          names them; see the class docstring for why that is a warning and not
          a fix, and why Codex has no hook escape hatch.
        - ``agent.memory`` — the CLI manages its own context; the agent's
          ``MemorySource`` (if any) is never queried.
        - ``Agent.resume()`` — not supported (ReAct-only). For a CLI-native
          resume, pass the ``evals["session_id"]`` value back as
          ``CodexCliCognition(resume_session_id=...)``.
        - ``ctx.budget`` — CHARGED, but only AFTER the fact. An already-exhausted
          budget refuses to spawn at all, with the resumable
          ``budget_exhausted`` stop reason, and what the run cost is charged to
          every meter once it ends. Unlike the Claude cognition there is no
          in-flight ceiling: ``codex`` has no ``--max-budget-usd``. Set
          ``meter_spend=False`` to opt a run out of both ends.
        """
        del context  # unused — the CLI owns its own transcript
        async for ev in self._run_once(
            agent=agent,
            ctx=ctx,
            prompt=self._compose_prompt(agent.prompt, task),
            resume=self._resume_target(),
        ):
            yield ev

    def session(self, *, agent: Agent | None = None) -> CodexCliSession:
        """Open a multi-turn conversation — one thread, many spawns.

        ``drive()`` starts a fresh conversation every time. A session keeps the
        CLI's thread id and resumes it, so turn two remembers turn one::

            async with cognition.session() as chat:
                async for ev in chat.turn("Summarise README.md"):
                    ...
                async for ev in chat.turn("Now list the risks you skipped"):
                    ...

        See :class:`CodexCliSession` for what resuming implies — a warm-up per
        turn, and a conversation that survives a cancelled turn.

        Raises:
            ValueError: when ``ephemeral=True`` (nothing is written to resume
                from) or the cognition already names a resume target (the
                session owns thread identity; two owners would fork the
                conversation on turn two).
        """
        if self.ephemeral:
            raise ValueError(
                "CodexCliCognition.session() needs the CLI to persist its thread so later "
                "turns can resume it, and ephemeral=True turns that off. Drop ephemeral, or "
                "use drive() for one-shot runs."
            )
        return CodexCliSession(self, agent=agent)

    # ---- capability reporting --------------------------------------------------------------

    def native_tools(self) -> tuple[str, ...]:
        """The CLI's OWN tools this session holds, sorted.

        Codex has no tool allow-list, so this is a fixed set plus ``web_search``
        when ``--search`` is on. It exists for the middleware-bypass warning,
        which has to name what the chain is not applying to.

        ``apply_patch`` is reported even under ``sandbox="read-only"``. The tool
        is still present and the model still calls it; the sandbox is what makes
        the write fail. Dropping it here would tell a reader the chain has
        nothing to miss, when what actually happens is an attempted edit the
        chain never saw.
        """
        tools = set(CODEX_NATIVE_TOOLS)
        if self.web_search:
            tools.add("web_search")
        return tuple(sorted(tools))

    @property
    def effective_sandbox(self) -> SandboxMode:
        """The sandbox that will actually be in force, including the defaults.

        ``bypass_sandbox`` is ``danger-full-access`` by another name — the flag
        turns the sandbox off — and an unset ``sandbox`` is the CLI's own
        ``read-only`` default for ``codex exec``. Both are resolved here rather
        than at each reader, because a caps property that answered "nothing" for
        an unset flag would leave the trifecta gate blind on the most common
        wiring there is.
        """
        if self.bypass_sandbox:
            return "danger-full-access"
        return self.sandbox or _DEFAULT_SANDBOX

    @property
    def caps(self) -> tuple[str, ...]:
        """Rule-of-Two tags for this session, in ``RunPolicy``'s vocabulary.

        Derived from :attr:`effective_sandbox`, ``network_access`` and
        ``web_search`` — see the comment on :data:`CODEX_SANDBOX_CAPS`. The
        shape is what ``RunPolicy.capabilities`` reads (``getattr(t, "caps",
        ())``), so the cognition can be handed to the gate exactly like a
        tool::

            RunPolicy(mode="deny").check([cognition, *agentkit_tools])

        Mixing the two lists is the case that matters and the one a per-path
        check misses: the CLI's shell supplies private data, ``--search``
        supplies untrusted content and egress, and only a check over BOTH sees
        an agentkit tool completing a set the CLI started.

        **Known under-approximation.** MCP servers wired through
        ``mcp_servers`` contribute nothing here, because their caps live on the
        server's own tool definitions and this cognition never sees them. Pass
        those tools' agentkit-side objects alongside the cognition (as above)
        rather than assuming this property covers them.
        """
        tags = set(CODEX_SANDBOX_CAPS[self.effective_sandbox])
        if self.network_access:
            tags.add("egress")
        if self.web_search:
            tags.update(("untrusted_content", "egress"))
        return tuple(sorted(tags))

    # ---- the run ---------------------------------------------------------------------------

    async def _run_once(
        self,
        *,
        agent: Agent | None,
        ctx: Ctx | None,
        prompt: str,
        resume: tuple[str, ...],
    ) -> AsyncIterator[StreamEvent]:
        """One spawn, start to terminal event. Shared by ``drive`` and by a
        :class:`CodexCliSession` turn, which differ only in what they resume
        and how they compose the prompt — everything below (argv, the stream
        fold, the stop-reason priority, the metering) must be identical between
        them, and a second copy is precisely how the two would drift apart."""
        self._warn_if_middleware_bypassed(ctx)

        cfg_home = str(self.config_home) if self.config_home is not None else None
        sem = _get_semaphore(self.codex_bin, cfg_home, self.max_concurrent)

        state = _TurnState()
        log = _ItemLog()
        outcome = _CliRunOutcome()
        schema_requested = False
        # Temp files the spawn needs and nobody else should have to manage: the
        # ``--output-schema`` document and, under ``system_prompt_mode="replace"``,
        # the instructions file. Removed in the ``finally`` regardless of how the
        # run ends — including a KeyboardInterrupt, which is why this is a
        # directory rather than two NamedTemporaryFiles whose cleanup order
        # would have to be tracked.
        scratch: str | None = None
        # Wall time, because the CLI does not report its own. ``claude``'s
        # ``result`` payload carries ``duration_ms`` and this cognition's
        # ``evals["cli_duration_ms"]`` has to mean the same thing in both, so
        # the number is measured here rather than left at 0 — a duration field
        # that is always zero is worse than absent, since a dashboard plots it.
        # Started before the semaphore so it includes time spent WAITING for a
        # spawn permit, which is the number an operator debugging a slow run
        # under concurrency actually needs.
        started_at = time.monotonic()

        def _prepare() -> _CliLaunch:
            """argv/env resolution and the scratch files, inside the driver's
            guarded block so a throw here still reaches the terminal event."""
            nonlocal schema_requested, scratch
            schema = self._resolve_json_schema(agent)
            schema_requested = schema is not None
            self._refuse_if_budget_exhausted(ctx)
            instructions = self._replacement_instructions(agent)
            if schema is not None or instructions is not None:
                scratch = tempfile.mkdtemp(prefix="agentkit-codex-")
                os.chmod(scratch, 0o700)
            schema_path = (
                _write(scratch, "output_schema.json", json.dumps(_strict_object_schema(schema)))
                if schema
                else None
            )
            instr_path = _write(scratch, "instructions.md", instructions) if instructions else None
            argv = self._build_argv(
                prompt,
                resume=resume,
                output_schema_path=schema_path,
                instructions_path=instr_path,
                stream_input=True,
            )
            _require_working_dir(self.working_dir)
            return _CliLaunch(
                argv=argv,
                env=self._build_env(),
                cwd=str(self.working_dir) if self.working_dir is not None else None,
                # Raw text — the ``-`` in argv is the CLI's own marker for
                # "instructions come from stdin". No envelope, unlike claude.
                stdin_payload=prompt.encode(),
            )

        async def _handle(payload: dict[str, Any]) -> AsyncIterator[StreamEvent]:
            """One payload → its events, folding this run's state on the way.

            ``_events_from_payload`` is a SYNC generator here and an async one
            on the Claude side; the driver takes an async iterator, so the
            adaptation happens in this closure rather than by giving the shared
            driver two shapes to support.
            """
            for ev, delta in _events_from_payload(payload, log):
                state.fold(delta)
                if ev is not None:
                    yield ev

        try:
            async for event in _run_cli_process(
                prepare=_prepare,
                handle=_handle,
                spawn=self.spawn,
                semaphore=sem,
                timeouts=self.timeouts,
                terminate_grace_s=self.terminate_grace_s,
                ctx=ctx,
                outcome=outcome,
            ):
                yield event
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)

        result = await self._finalise(
            agent=agent,
            ctx=ctx,
            state=state,
            cancelled=outcome.cancelled,
            fatal_exc=outcome.fatal_exc,
            spawned=outcome.spawned,
            return_code=outcome.return_code,
            stderr_bytes=outcome.stderr_bytes,
            schema_requested=schema_requested,
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
            timed_out=outcome.timed_out,
        )
        yield StreamEvent("final", usage=result.usage, result=result)
        if outcome.should_reraise_cancel:
            raise asyncio.CancelledError()
        _reraise_if_not_an_exception(outcome.fatal_exc)

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
        elapsed_ms: int | None = None,
        timed_out: CliTimedOut | None = None,
    ) -> AgentResult:
        """Turn one completed turn's state into its terminal ``AgentResult``.

        ``return_code`` is ``None`` when the process is still alive and ``-1``
        when none was ever spawned. Both mean "the exit code says nothing about
        this turn", so neither is treated as a failure here.
        """
        # Priority (highest first): cancellation → fatal exception → CLI
        # non-zero exit → CLI semantic error → success. Cancellation is above
        # fatal_exc because a cancel that races with a fatal error should still
        # surface as ``cancelled``.
        final_stop_reason: str | None
        final_partial: bool
        if cancelled:
            final_stop_reason = "cancelled"
            final_partial = True
        elif timed_out is not None:
            # Above ``fatal_exc``: a parse error on the half-written line we cut
            # off mid-flight must not outrank the bound that caused the cut.
            final_stop_reason = _classify_timeout(timed_out, stderr_bytes)
            final_partial = True
        elif fatal_exc is not None:
            if isinstance(fatal_exc, _SessionClosed):
                final_stop_reason = "session_closed"
            elif type(fatal_exc).__name__ == "MeterExceeded":
                # The pre-flight refusal: no subprocess was spawned.
                # ``budget_exhausted`` is a RESUMABLE stop reason, which is the
                # honest one — raise the ceiling and run again. Matched by name
                # so this module keeps its import of ``runtime.meter`` lazy.
                final_stop_reason = "budget_exhausted"
            elif not spawned:
                if _is_working_dir_missing(fatal_exc):
                    final_stop_reason = "working_dir_missing"
                else:
                    final_stop_reason = "spawn_failed"
            elif isinstance(fatal_exc, CliLineTooLong):
                # Its own reason, not ``parse_failed``. Nothing was malformed:
                # the CLI emitted well-formed JSON that is simply bigger than
                # the reader will assemble, and the fix is a size — read less
                # per tool call, or raise the limit — where ``parse_failed``
                # sends an operator looking for a corrupt payload.
                final_stop_reason = "output_line_too_long"
            else:
                final_stop_reason = "parse_failed"
            final_partial = True
        elif return_code not in (0, None):
            assert return_code is not None
            final_stop_reason = _exit_stop_reason(return_code, stderr_bytes)
            final_partial = True
        elif state.is_error:
            final_stop_reason = state.stop_reason or "cli_reported_error"
            final_partial = True
        elif spawned and not state.saw_terminal:
            # The process ended cleanly but never emitted its end-of-turn
            # payload, so whatever text arrived is a fragment the CLI had not
            # finished writing. Placed BELOW the ``is_error`` branch so a CLI
            # that reported its own failure keeps saying so, and above plain
            # success so a truncated stream can never be reported as complete.
            final_stop_reason = "malformed_output"
            final_partial = True
        else:
            final_stop_reason = state.stop_reason  # may still be None on clean success
            final_partial = False

        # ── cost ────────────────────────────────────────────────────────────
        # Before the structured-output decision and before metering, because
        # both the meters and the terminal event must see the same number. The
        # CLI supplies none, so this is where ``Usage.cost_usd`` comes from at
        # all — see the ``pricing`` field for what 0.0 means.
        usage = self._priced(state)

        # ── structured output ───────────────────────────────────────────────
        # A schema was requested. Codex constrains its FINAL MESSAGE to it
        # rather than returning a separate validated field the way ``claude``
        # does, so the object has to be parsed back out of the answer text.
        # Three outcomes, and only the first is a success:
        #
        #   parses, coerces  → ``parsed`` is the declared type
        #   parses, does not → structured_output_mismatch
        #   does not parse   → structured_output_missing
        parsed: Any = None
        structured_failure: StructuredOutputFailure | None = None
        structured_output: Any = None
        if schema_requested:
            structured_output, decode_error = _decode_structured(state.text)
            if decode_error is not None:
                final_partial = True
                if final_stop_reason in (None, "success"):
                    final_stop_reason = "structured_output_missing"
                # ``undecodable`` and not ``missing``: for Codex the structured
                # answer IS the final message, so "there was text and it was not
                # JSON" is a different fault from "there was nothing" — and it
                # is the one where the raw excerpt is worth keeping, because
                # what the model actually said is the whole diagnosis.
                structured_failure = StructuredOutputFailure(
                    kind="undecodable" if state.text.strip() else "missing",
                    detail=decode_error,
                    raw_excerpt=(state.text[:800] + "…") if len(state.text) > 800 else state.text,
                    truncated=len(state.text) > 800,
                )
            else:
                parsed, structured_failure = _coerce_structured(agent, structured_output)
                if structured_failure is not None:
                    final_partial = True
                    final_stop_reason = "structured_output_mismatch"

        charge_error = await _charge_meters(ctx, usage, enabled=self.meter_spend)

        evals: dict[str, Any] = {
            "session_id": state.session_id or "",
            # The CLI's own figure when a version of it starts reporting one,
            # and the measured wall time until then. Never 0 for a run that
            # actually happened.
            "cli_duration_ms": state.duration_ms if state.duration_ms is not None else (elapsed_ms or 0),
            "cli_return_code": return_code if return_code is not None else 0,
            # Never "reported". The CLI does not report one, and a caller who
            # reads ``usage.cost_usd`` without reading this will otherwise
            # treat a table lookup — or a 0.0 for an unknown model — as a
            # billed number.
            "cost_source": "estimated",
        }
        external_id = getattr(ctx, "correlation_id", None)
        if external_id:
            evals["external_run_id"] = str(external_id)
        if final_stop_reason is not None:
            evals["stop_reason"] = final_stop_reason
        if timed_out is not None:
            evals["timeout_s"] = timed_out.limit_s
            evals["timeout_kind"] = timed_out.reason
        if state.model:
            # Which model ACTUALLY ran. Codex resolves aliases and profiles on
            # its side, so this can differ from ``self.model`` — and it is the
            # value the cost above was computed against.
            evals["cli_model"] = state.model
        if state.thinking:
            # Reasoning summaries, separate from ``output`` (the final answer).
            # Live consumers already saw each as a ``message_delta``; this is
            # the folded copy for AgentResult callers.
            evals["thinking"] = state.thinking
        if structured_output is not None:
            evals["structured_output"] = structured_output
        if structured_failure is not None:
            # The string is the shape callers and tests have always read; the
            # dict is the one an application can branch on, via
            # ``StructuredOutputFailure.of(result.evals)``. A dict rather than
            # the dataclass because ``evals`` is deep-frozen and checkpointed.
            evals["structured_output_error"] = str(structured_failure)
            evals["structured_output_failure"] = structured_failure.to_dict()
        if state.errors:
            # Non-fatal item-level errors (a truncated command output, a
            # retried tool). A run that took 40s and looks fine is explained by
            # these and nothing else in the result.
            evals["cli_errors"] = list(state.errors)
        if fatal_exc is not None:
            # Redacted like stderr: an exception message can carry a URL with a
            # token in it, and this string is checkpointed and fanned out to
            # observers exactly like the rest of ``evals``.
            evals["error"] = _redact_secrets(f"{type(fatal_exc).__name__}: {fatal_exc}")
        if charge_error is not None:
            evals["meter_error"] = charge_error
        if final_partial and stderr_bytes:
            stderr_text = _redact_secrets(
                stderr_bytes.decode("utf-8", errors="replace").strip()
            )
            if stderr_text:
                # The CLI's own diagnostics, verbatim except for
                # credential-shaped runs. An ``AgentResult`` is checkpointed,
                # logged and fanned out to observers, so a CLI that prints a
                # failing request's Authorization header would otherwise have
                # it persisted. Defence in depth, not a boundary — see
                # ``_redact_secrets``.
                evals["stderr"] = stderr_text

        return AgentResult(
            output=state.text,
            usage=usage,
            partial=final_partial,
            evals=evals,
            parsed=parsed,
            # This cognition reports failures as DATA (a terminal event is
            # guaranteed even when the subprocess never starts), so it is
            # the one producer that can legitimately stamp ``"failed"``.
            stop_reason=_codex_stop_reason(final_stop_reason),
        )

    # ---- helpers ---------------------------------------------------------------------------

    def _resolve_system_prompt(self, prompt: Prompt | str | None) -> str:
        """Extract a rendered system-prompt string from ``agent.prompt``.

        The agent's ``prompt`` field accepts three shapes: ``None``, ``str``,
        or ``Prompt``. Match all three; render the ``Prompt`` via
        ``Prompt.render()`` for the versioned path.
        """
        if prompt is None:
            return ""
        if isinstance(prompt, Prompt):
            return prompt.render()
        return prompt

    def _compose_prompt(self, prompt: Prompt | str | None, task: str) -> str:
        """The single user message the CLI receives.

        Under ``system_prompt_mode="replace"`` the agent's prompt goes to the
        instructions file instead, so the task travels alone; under
        ``"prepend"`` the two are joined by a horizontal rule. The rule is not
        decoration — without a separator a one-line prompt and a one-line task
        read to the model as one run-on instruction, which is how a "be terse"
        prompt ends up being treated as part of the question.
        """
        if self.system_prompt_mode == "replace":
            return task
        system = self._resolve_system_prompt(prompt).strip()
        if not system:
            return task
        return f"{system}\n\n---\n\n{task}"

    def _replacement_instructions(self, agent: Agent | None) -> str | None:
        """The text for ``experimental_instructions_file``, or ``None``.

        Only under ``system_prompt_mode="replace"``, and only when there is a
        prompt to write: an empty instructions file would replace Codex's base
        instructions with nothing, which is a strictly worse agent than the one
        the caller started with and would look like the mode had no effect.
        """
        if self.system_prompt_mode != "replace" or agent is None:
            return None
        text = self._resolve_system_prompt(agent.prompt).strip()
        return text or None

    def _resume_target(self) -> tuple[str, ...]:
        """The ``resume`` subcommand's argv fragment, or ``()`` for a fresh run."""
        if self.resume_session_id is not None:
            return ("resume", self.resume_session_id)
        if self.continue_session:
            return ("resume", "--last")
        return ()

    def _resolve_json_schema(self, agent: Agent | None) -> dict[str, Any] | None:
        """The JSON Schema to hand the CLI, or ``None``.

        An explicit ``json_schema=`` wins. Otherwise the agent's own ``output=``
        schema is used, through the same ``SchemaAdapter`` the rest of the
        framework uses — so declaring ``output=Invoice`` types a CLI-delegated
        run exactly like it types a normal one.

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

    def _priced(self, state: _TurnState) -> Usage:
        """``state.usage`` with a ``cost_usd`` on it.

        Never raises. A caller-supplied ``pricing=`` that blows up must not
        cost them the run's answer — the tokens are real and the terminal-event
        guarantee is unconditional — so a failure leaves the cost at 0.0 and
        the rest of the result intact.
        """
        price = self.pricing or _default_pricing
        model = state.model or self.model
        try:
            cost_usd = float(price(model, state.usage))
        except Exception:  # noqa: BLE001 — see docstring
            return state.usage
        return Usage(
            input_tokens=state.usage.input_tokens,
            output_tokens=state.usage.output_tokens,
            cost_usd=round(cost_usd, 6),
            cache_read_tokens=state.usage.cache_read_tokens,
            cache_write_tokens=state.usage.cache_write_tokens,
        )

    def _refuse_if_budget_exhausted(self, ctx: Ctx | None) -> None:
        """Raise :class:`MeterExceeded` when the run has no headroom left.

        A pre-flight refusal on purpose: spawning a subprocess to be told what
        we already know costs seconds of CLI warm-up, and the terminal event it
        produces (``budget_exhausted``) is the resumable one.

        This is the WHOLE of the budget's in-flight power here. ``codex`` has
        no ``--max-budget-usd``, so a run that starts with headroom can spend
        past it and only be caught by the charge at the end. Stated plainly
        because the Claude cognition's identical-looking wiring does stop a run
        mid-flight, and a reader moving between them will assume this one does
        too.
        """
        if not self.meter_spend or ctx is None:
            return
        budget = getattr(ctx, "budget", None)
        remaining = getattr(budget, "remaining", None)
        if remaining is None:
            return
        headroom = remaining()
        if headroom is None:  # no ceiling configured
            return
        if headroom <= 0:
            from agentkit.runtime.meter import MeterExceeded

            raise MeterExceeded(f"codex CLI not spawned: the run budget has {headroom} USD left")


    def _warn_if_middleware_bypassed(self, ctx: Ctx | None) -> None:
        """Say which middlewares will not run, once, before the first spawn.

        ``warnings.warn`` and not a log line or an ``Observation``: this is a
        wiring mistake made once at build time, and the audience is the
        developer at the REPL or in CI, not the operator reading a run's
        telemetry. A log line needs a configured logger to be seen at all; an
        ``Observation`` arrives on the run's observer, i.e. inside the very
        machinery the caller has just been told is not applying. ``warnings``
        is also the only one of the three a caller can promote to an error
        (``-W error``) or silence per-message.

        Silent when the chain is empty or absent — there is nothing being
        skipped. Unlike the Claude cognition there is no "no native tools"
        escape: a Codex session always has ``shell``.
        """
        if self._bypass_warned:
            return
        names = _tool_middleware_names(ctx)
        if not names:
            return
        # Latch before warning, not after: ``warnings.warn`` can be promoted to
        # an exception by the caller's filters, and an escaping error must not
        # leave the latch unset so that the next drive raises again.
        self._bypass_warned = True
        import warnings

        warnings.warn(
            "CodexCliCognition bypasses the Invoker, so the tool middleware on this context "
            f"does NOT run for a native CLI tool call. Not applied to {', '.join(self.native_tools())}: "
            f"{', '.join(names)}. The CLI executes those inside its own process and agentkit never "
            "sees the call, so there is no egress/SSRF or allowlist check, no input guard, no audit "
            "record and no idempotency key for any of them — only the run's total cost is reconciled "
            "afterwards. Codex has no PreToolUse hook, so unlike the Claude CLI there is no way to "
            "make the chain reach `shell`; the containment that does apply is the OS sandbox "
            f"(sandbox={self.effective_sandbox!r}). Serve your own tools over MCP (mcp_servers=...) "
            "so those calls come back through agentkit, and gate the rest with "
            'RunPolicy(mode="deny").check([cognition]).',
            UserWarning,
            stacklevel=3,
        )

    def _build_argv(
        self,
        prompt: str,
        *,
        resume: tuple[str, ...] = (),
        output_schema_path: Path | None = None,
        instructions_path: Path | None = None,
        stream_input: bool = False,
    ) -> list[str]:
        """Assemble the CLI argv.

        **Every option goes before the subcommand.** ``codex exec`` marks some
        of its flags ``global`` (they parse on either side of ``resume``) and
        flattens others onto the parent only (they do not). Putting all of them
        ahead of ``resume`` is the one ordering that is correct for both, and
        getting it wrong is a clap parse error several seconds into a run whose
        message names the flag rather than the position.

        The prompt goes last. With ``stream_input`` it is the single character
        ``-``, which is the CLI's own documented spelling of "read the
        instructions from stdin" (``codex exec --help``: *"If not provided as an
        argument (or if `-` is used), instructions are read from stdin"*), and
        the real prompt is written to the pipe instead. Otherwise it is the text
        itself, behind a ``--`` separator when it begins with a dash — a task
        like ``"--force a rewrite"`` would otherwise be read as flags and the
        run would die on an unknown argument.

        ``-`` is passed WITHOUT the ``--`` separator even though it starts with
        a dash: the separator would turn it into a literal one-character prompt,
        which is the one way to get this exactly backwards.
        """
        # ``--search`` is a PARENT-level flag, not an ``exec`` one. It used to
        # be accepted after the subcommand; codex 0.152.1 answers that with
        # ``error: unexpected argument '--search' found`` and exits 2 before
        # starting a thread, so every run with ``web_search=True`` died on a
        # clap usage message. It still exists — ``codex --help`` documents it as
        # "Enable live web search" — it simply has to come before ``exec``.
        # Verified against the binary: ``codex --search exec …`` runs.
        argv: list[str] = [self.codex_bin]
        if self.web_search:
            argv += ["--search"]
        argv += ["exec"]
        # ── format ──────────────────────────────────────────────────────────
        # ``--color never`` unconditionally: the CLI writes human progress to
        # stderr, and this cognition surfaces that stderr verbatim in
        # ``evals["stderr"]`` on a failure. ANSI escapes in a stored diagnostic
        # are noise in every reader that is not a terminal.
        argv += ["--json", "--color", "never"]
        if self.model is not None:
            argv += ["--model", self.model]
        if self.working_dir is not None:
            argv += ["--cd", str(self.working_dir)]
        # ── containment ─────────────────────────────────────────────────────
        if self.bypass_sandbox:
            argv += ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            if self.sandbox is not None:
                argv += ["--sandbox", self.sandbox]
            # ``--ask-for-approval`` is NOT a flag on ``codex exec``. It used to
            # be; codex 0.152.1 rejects it outright —
            #
            #     error: unexpected argument '--ask-for-approval' found
            #
            # — and the CLI exits 2 before starting a thread, so every run of a
            # cognition carrying this field failed with a clap usage message
            # and no output. The approval policy is a config key now, and
            # ``-c approval_policy=<mode>`` was verified to run. It is emitted
            # here rather than through ``_resolved_overrides`` so that an
            # explicit ``config_overrides={"approval_policy": ...}`` still wins,
            # which is the precedence that function already documents.
            if self.ask_for_approval is not None:
                argv += ["-c", f"approval_policy={_toml_scalar(self.ask_for_approval)}"]
        for d in self.add_dirs:
            argv += ["--add-dir", str(d)]
        # ── reproducibility ─────────────────────────────────────────────────
        if self.ignore_user_config:
            argv += ["--ignore-user-config"]
        if self.ignore_rules:
            argv += ["--ignore-rules"]
        if self.strict_config:
            argv += ["--strict-config"]
        if self.skip_git_repo_check:
            argv += ["--skip-git-repo-check"]
        if self.ephemeral:
            argv += ["--ephemeral"]
        if self.oss:
            argv += ["--oss"]
        if self.profile is not None:
            argv += ["--profile", self.profile]
        for image in self.images:
            argv += ["--image", str(image)]
        # ── output ──────────────────────────────────────────────────────────
        if output_schema_path is not None:
            argv += ["--output-schema", str(output_schema_path)]
        if self.output_last_message is not None:
            argv += ["--output-last-message", str(self.output_last_message)]
        # ── config overrides ────────────────────────────────────────────────
        for key, value in self._resolved_overrides(instructions_path).items():
            argv += ["-c", f"{key}={_toml_scalar(value)}"]
        argv += list(self.extra_args)
        # ── the subcommand and its positionals, last ────────────────────────
        argv += list(resume)
        if stream_input:
            argv += ["-"]
        else:
            if prompt.startswith("-"):
                argv += ["--"]
            argv += [prompt]
        return argv

    def _resolved_overrides(self, instructions_path: Path | None) -> dict[str, Any]:
        """Every ``-c key=value`` this run needs, in a stable order.

        The caller's own ``config_overrides`` go LAST so an explicit override
        wins over one this cognition derived — someone who writes
        ``config_overrides={"model_reasoning_effort": "high"}`` alongside
        ``effort="low"`` has said two things, and the one they typed as a
        config key is the more specific.
        """
        out: dict[str, Any] = {}
        if self.effort is not None:
            out["model_reasoning_effort"] = self.effort
        if self.network_access:
            out["sandbox_workspace_write.network_access"] = True
        if instructions_path is not None:
            out["experimental_instructions_file"] = str(instructions_path)
        for server, table in self.mcp_servers.items():
            for key, value in table.items():
                out[f"mcp_servers.{server}.{key}"] = value
        out.update(self.config_overrides)
        return out

    @property
    def effective_env_policy(self) -> EnvPolicy:
        """The policy actually in force, with ``None`` resolved.

        ``config_home=`` implies ``"profile"``, matching the Claude cognition's
        rule so one wiring reads the same across both. Codex was measured NOT to
        prefer an ambient ``OPENAI_API_KEY`` over the ``CODEX_HOME`` login, so
        this is hygiene here rather than a fix — see the ``env_policy`` field.
        """
        if self.env_policy is not None:
            return self.env_policy
        return "profile" if self.config_home is not None else "inherit"

    def _build_env(self) -> dict[str, str]:
        """The child's environment, under :attr:`effective_env_policy`.

        Layers ``CODEX_HOME`` on top when the cognition was constructed with a
        ``config_home`` (isolated auth / settings / session history, e.g. a
        per-tenant server-side wrapper), then :attr:`env` last.

        Nothing else is injected. The Claude cognition bridges
        ``correlation_id`` into ``CLAUDE_TRACE_EXTERNAL_ID`` because that
        variable exists; ``codex`` has no counterpart, and setting an invented
        one would put a value in the child's environment that nothing reads
        while reading, from the outside, exactly like a working trace bridge.
        The id is still on the result as ``evals["external_run_id"]``. Anything
        else a run needs goes through :attr:`env`, explicitly.
        """
        env, removed = _build_child_env(
            policy=self.effective_env_policy,
            credential_vars=_CODEX_AMBIENT_AUTH_ENV,
            overrides=self.env,
        )
        if removed:
            self._warn_env_stripped(removed)
        if self.config_home is not None:
            env["CODEX_HOME"] = str(self.config_home)
        return env

    def _warn_env_stripped(self, removed: tuple[str, ...]) -> None:
        """Name what the policy removed, once. See the Claude counterpart —
        a silent strip is as invisible as the silent inherit it replaces."""
        if self._env_warned:
            return
        self._env_warned = True
        import warnings

        warnings.warn(
            f"CodexCliCognition(env_policy={self.effective_env_policy!r}) removed "
            f"{', '.join(removed)} from the CLI's environment"
            + (" (implied by config_home=)" if self.env_policy is None else "")
            + ". Pass env_policy='inherit' to keep the ambient values, or env={...} to "
            "supply the credential explicitly.",
            UserWarning,
            stacklevel=4,
        )


class CodexCliSession:
    """One Codex thread, many turns — a spawn each, resumed by thread id.

    ``codex exec`` is one-shot: it starts a thread, answers, and exits. Its
    continuation seam is ``codex exec resume <thread-id>``, so a session here
    is not a held process the way :class:`~agentkit.agents.cognition.ClaudeCliSession`
    is. Turn one runs plain ``codex exec`` and captures the thread id from the
    ``thread.started`` payload; every later turn resumes it::

        async with cognition.session() as chat:
            async for ev in chat.turn("Summarise README.md"):
                ...
            async for ev in chat.turn("Now list the risks you skipped"):
                ...

    Every per-turn contract of ``drive`` holds unchanged, because both go
    through the same ``_run_once``: exactly one terminal ``final`` event, the
    same stop-reason taxonomy, the same structured-output handling, the same
    metering. What differs is what a resumed thread implies, and each is a real
    trade against the held-process design:

    * **A warm-up per turn.** Resuming rehydrates the thread from disk, so
      there is no equivalent of the Claude session's one-time startup cost.
      This is the price of the two advantages below.
    * **A cancelled turn does NOT end the conversation.** Cancelling kills a
      subprocess, and the thread it was resuming is still on disk. The next
      ``turn()`` resumes the same conversation. (The Claude session has to
      declare itself over: its process WAS the conversation.)
    * **Per-turn structured output works.** ``--output-schema`` is chosen at
      spawn and every turn is its own spawn, so an ``output=``-carrying agent
      may be passed to any turn. The Claude session has to refuse this — there
      the flag is fixed for the life of one process.
    * **Turns are serialised.** One thread, one transcript: two concurrent
      ``turn()`` calls resuming the same thread would fork it. A lock makes the
      second caller wait.
    * **``ephemeral=True`` is refused** by :meth:`CodexCliCognition.session`,
      since a thread that was never written cannot be resumed.

    There is no ``interrupt()``. The Claude session has one because a held
    process needs a way to stop a turn without killing the conversation; here
    cancelling the turn already leaves the conversation intact, so the same
    intent is served by cancelling the task (or tripping
    ``ctx.check_cancelled()``) and calling ``turn()`` again.
    """

    def __init__(self, cognition: CodexCliCognition, *, agent: Agent | None = None) -> None:
        self._cog = cognition
        self._agent = agent
        self._lock = asyncio.Lock()
        self._closed = False
        self._turns = 0
        # The CLI's own id for this conversation, learned from turn one's
        # ``thread.started`` payload. ``None`` until then — which is also the
        # state a failed first turn leaves behind, and the reason turn two
        # starts a fresh thread rather than refusing: a conversation with no
        # history is exactly what a first turn that produced none left us.
        self.session_id: str | None = cognition.resume_session_id

    @property
    def name(self) -> str:
        """``"codex_cli_session"`` — the cognition's name, marked as a session.

        Required, not decorative. ``Agent._span_attrs`` reads ``cognition.name``
        for the ``agentkit.agent.cognition`` trace attribute on EVERY run, so a
        session without one raised ``AttributeError: 'CodexCliSession' object has no
        attribute 'name'`` straight out of ``agent.run(...)`` — which is
        precisely the usage :meth:`drive` exists for and this class's docstring
        advertises. Measured; the one-shot cognition has had the field since it
        was written and the session never did.

        Suffixed rather than delegated verbatim so a trace distinguishes the two
        regimes. They have genuinely different behaviour to debug — one
        conversation across many turns versus a fresh one each time — and
        ``agentkit.agent.cognition`` is the attribute that is supposed to say
        which turn-taking regime ran.
        """
        return f"{self._cog.name}_session"

    # ---- lifecycle -------------------------------------------------------------------------

    async def __aenter__(self) -> CodexCliSession:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Present for symmetry with :class:`ClaudeCliSession`, and a no-op.

        There is no process to spawn until the first turn asks for one. Kept so
        the two sessions are interchangeable in a caller's ``async with`` and
        so a future version that DOES pre-warm something does not change the
        calling convention.
        """
        self._closed = False

    async def close(self) -> None:
        """End the session. Idempotent.

        Nothing is killed — a turn owns its own subprocess for the whole of its
        life and there is never one running between turns. Closing marks the
        session so a later ``turn()`` refuses rather than silently starting a
        conversation the caller believes is finished, and it deliberately does
        NOT delete the CLI-side thread: the id is on every terminal event, and
        a caller may want to resume it tomorrow with
        ``CodexCliCognition(resume_session_id=...)``.
        """
        self._closed = True

    @property
    def caps(self) -> tuple[str, ...]:
        """Rule-of-Two tags for this session — the cognition's, unchanged.

        A session can BE an agent's cognition (see :meth:`drive`), so it can be
        handed to ``RunPolicy.check`` in exactly the place the cognition would
        be. Without this delegation ``getattr(session, "caps", ())`` returns the
        empty tuple and the gate sees a tool-less run.
        """
        return self._cog.caps

    @property
    def turns_taken(self) -> int:
        """How many turns this session has completed. A session reporting one
        turn where the caller expected three is the difference between "the
        model was unhelpful" and "turns two and three never ran"."""
        return self._turns

    # ---- turns -----------------------------------------------------------------------------

    async def turn(
        self,
        task: str,
        *,
        agent: Agent | None = None,
        ctx: Ctx | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send one user turn and stream its events, ending at exactly one ``final``.

        ``agent`` supplies the output schema and the system prompt; ``ctx``
        supplies the meters and the cancel token. Both may be omitted for a
        bare conversational session.
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
        cognition and consecutive ``agent.run(...)`` calls continue one
        CLI-side thread."""
        del context  # the CLI owns its own transcript
        async for ev in self._turn(task, agent=agent, ctx=ctx):
            yield ev

    async def _turn(
        self, task: str, *, agent: Agent | None, ctx: Ctx | None
    ) -> AsyncIterator[StreamEvent]:
        # Serialise turns: one thread and one transcript mean two concurrent
        # resumes of the same thread would fork the conversation.
        async with self._lock:
            if self._closed:
                # Reported as a terminal event, not raised, so a session turn
                # keeps the same exactly-one-``final`` contract as a one-shot
                # drive. ``_run_once`` is bypassed entirely — there is nothing
                # to spawn — so the result is assembled here.
                result = await self._cog._finalise(
                    agent=agent,
                    ctx=ctx,
                    state=_TurnState(),
                    cancelled=False,
                    fatal_exc=_SessionClosed(
                        "this Codex session is closed. Its CLI-side thread still exists — "
                        "open a new session, or pass resume_session_id="
                        f"{self.session_id!r} to a fresh CodexCliCognition."
                    ),
                    spawned=False,
                    return_code=-1,
                    stderr_bytes=b"",
                    schema_requested=False,
                )
                yield StreamEvent("final", usage=result.usage, result=result)
                return

            # Turn one starts a thread; every later turn resumes the one turn
            # one named. A session that never learned an id (a first turn that
            # died before ``thread.started``) starts fresh rather than resuming
            # nothing — see ``session_id``'s comment.
            resume = ("resume", self.session_id) if self.session_id else ()
            # The system prompt belongs to the CONVERSATION, not to each turn.
            # Prepending it again on turn three would put three copies in the
            # transcript and read to the model as escalating emphasis. Under
            # ``replace`` it is not in the transcript at all — it is the base
            # instructions file, rewritten for every spawn — so that mode is
            # unaffected and handled inside ``_run_once``.
            prompt = (
                self._cog._compose_prompt(agent.prompt if agent is not None else None, task)
                if not resume
                else task
            )
            async for ev in self._cog._run_once(agent=agent, ctx=ctx, prompt=prompt, resume=resume):
                if ev.type == "final" and ev.result is not None:
                    sid = ev.result.evals.get("session_id")
                    if sid:
                        self.session_id = str(sid)
                    self._turns += 1
                yield ev


class _SessionClosed(RuntimeError):
    """A turn was asked of a session that is over.

    Surfaced through the normal terminal event as ``session_closed`` rather
    than raised at the caller, so a session turn keeps the same
    exactly-one-``final`` contract as a one-shot drive.
    """


def _write(directory: str | None, name: str, text: str) -> Path:
    """Write ``text`` into the run's scratch directory and return its path."""
    assert directory is not None  # the caller creates it before asking
    path = Path(directory) / name
    path.write_text(text, encoding="utf-8")
    return path



def _strict_object_schema(node: Any) -> Any:
    """A copy of ``node`` with ``additionalProperties: false`` on every object.

    OpenAI's structured-output mode is STRICT, and strict mode refuses a schema
    whose objects do not close themselves:

        'additionalProperties' is required to be supplied and to be false.
        (param: text.format.schema, status 400)

    That is a provider constraint, not an agentkit one, which is why the
    normalisation lives here and not in the schema adapters — ``claude
    --json-schema`` accepts the open form happily, and rewriting a caller's
    schema framework-wide to satisfy one provider would be the wrong blast
    radius.

    It matters because the two adapters disagree. The dataclass adapter already
    emits ``additionalProperties: false``; Pydantic's ``model_json_schema()``
    does not. So ``output=SomeDataclass`` worked on Codex and
    ``output=SomePydanticModel`` failed every single time with a 400 the caller
    could only see by reading ``evals["stderr"]`` on a failed run — an arbitrary
    difference between two ways of declaring the same shape. Verified against
    the binary: identical schemas, exit 1 without the key and exit 0 with it.

    An explicitly-set ``additionalProperties`` is left ALONE, including a
    ``True`` the provider will reject. A caller who wrote it meant it, and
    silently inverting a stated instruction is worse than the error that names
    it.
    """
    if isinstance(node, list):
        return [_strict_object_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _strict_object_schema(value) for key, value in node.items()}
    is_object = out.get("type") == "object" or "properties" in out
    if is_object and "additionalProperties" not in out:
        out["additionalProperties"] = False
    return out


def _toml_scalar(value: Any) -> str:
    """Render a ``-c key=value`` right-hand side the way Codex parses it.

    The CLI parses the value as TOML, so a bare ``true`` is a boolean, a bare
    ``7`` is an integer, and anything else has to be quoted or bracketed.
    ``json.dumps`` produces exactly the right spelling for every shape this
    accepts — strings, numbers, booleans, lists and inline tables — because
    TOML's scalar and array syntax is JSON's here, and it is the one function
    that will not forget to escape a Windows path's backslashes or a prompt's
    embedded quote.

    ``None`` is refused rather than rendered. TOML has no null: ``json.dumps``
    would emit ``null``, which the CLI reads as the four-character string
    ``"null"`` or as a parse error depending on the key, and both are a setting
    that is not what the caller wrote.
    """
    if value is None:
        raise ValueError(
            "CodexCliCognition: a config override value cannot be None — TOML has no null, "
            "so this would reach the CLI as the string 'null' or as a parse error. Omit the "
            "key instead."
        )
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(value, separators=(",", ":"))


def _decode_structured(text: str) -> tuple[Any, str | None]:
    """Pull the JSON object out of a final message constrained by a schema.

    Returns ``(value, error)``. Codex constrains the answer TEXT rather than
    returning a separate validated field, so the object has to come back out of
    the prose — and the prose is usually just the object, but not always: a
    model told to answer in JSON will sometimes wrap it in a ``` fence, and a
    fence is a formatting habit rather than a failure to comply. Stripping one
    is the only accommodation made here.

    Everything else is reported as an error rather than repaired. Scanning for
    the first ``{`` and hoping was considered and rejected: it turns "the model
    ignored the schema and explained itself instead" — which the caller needs
    to see — into a confident parse of whatever JSON-shaped fragment happened
    to be in the explanation.
    """
    stripped = text.strip()
    if not stripped:
        return None, "the CLI returned no final message to validate against --output-schema"
    if stripped.startswith("```"):
        # ```json\n{...}\n```  →  {...}
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -len("```")]
        stripped = body.strip()
    try:
        return json.loads(stripped), None
    except ValueError as exc:
        return None, (
            f"the CLI's final message is not the JSON --output-schema asked for ({exc}). "
            f"First 200 chars: {text.strip()[:200]!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL parsing
#
# Codex has shipped TWO event vocabularies on ``--json`` and both are still in
# the wild, because the CLI is something users install themselves and a service
# does not get to pick their version:
#
#   thread events   {"type": "item.completed", "item": {...}}
#   legacy events   {"id": "0", "msg": {"type": "agent_message", ...}}
#
# Supporting one and ignoring the other is the worst outcome available: the
# unknown-payload rule below is "don't crash", so an older CLI would produce a
# run that exits 0, emits no events, and returns an empty answer with no
# stop_reason — a silent, plausible, wrong result. Both are parsed, into the
# same ``_EventDelta``, and the shape is detected per payload rather than
# sniffed once, so a CLI that mixed them mid-stream would still be read
# correctly.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _ItemLog:
    """Per-turn bookkeeping the payload parser needs but a payload cannot hold.

    Both vocabularies emit an item's text more than once — ``item.updated``
    then ``item.completed``, or a run of ``agent_message_delta`` then the whole
    ``agent_message``. A consumer must see each character exactly once, and the
    fold must count it exactly once, and neither is derivable from the payload
    in front of you.

    ``emitted`` is characters already yielded as ``message_delta`` per item id;
    ``called`` is the item ids a ``tool_call`` has already gone out for, so a
    ``begin``/``end`` pair produces one call and one result rather than two
    calls.
    """

    emitted: dict[str, int] = field(default_factory=dict)
    called: set[str] = field(default_factory=set)

    def suffix(self, key: str, text: str) -> str:
        """The part of ``text`` not yet emitted for ``key``, marking it emitted.

        Guards against a shrinking value — a CLI that re-sent a shorter text
        for the same id would otherwise slice with a stale, larger offset and
        yield nothing forever after.
        """
        already = self.emitted.get(key, 0)
        if already > len(text):
            already = 0
        self.emitted[key] = len(text)
        return text[already:]

    def first_call(self, key: str) -> bool:
        """Whether a ``tool_call`` still needs emitting for ``key``."""
        if key in self.called:
            return False
        self.called.add(key)
        return True


@dataclass(slots=True)
class _TurnState:
    """Everything one turn accumulates from the CLI's stream.

    Extracted so a session turn and a one-shot drive fold the stream the same
    way. It is a plain accumulator: no decisions live here, only the "last
    value wins / text concatenates" rules the payload sequence implies.
    """

    text: str = ""
    thinking: str = ""
    usage: Usage = field(default_factory=Usage)
    session_id: str | None = None
    model: str | None = None
    duration_ms: int | None = None
    is_error: bool = False
    stop_reason: str | None = None
    saw_terminal: bool = False
    errors: list[str] = field(default_factory=list)

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
        if delta.model is not None:
            self.model = delta.model
        if delta.duration_ms is not None:
            self.duration_ms = delta.duration_ms
        if delta.is_error:
            self.is_error = True
        if delta.saw_terminal:
            self.saw_terminal = True
        if delta.stop_reason is not None:
            self.stop_reason = delta.stop_reason
        if delta.error is not None:
            self.errors.append(delta.error)


@dataclass(slots=True)
class _EventDelta:
    """Small state-delta value returned alongside each yielded event so the
    drive loop can fold ``usage`` / ``session_id`` / ``text`` / ``thinking``
    without a second parsing pass."""

    text: str = ""
    thinking: str = ""
    usage: Usage | None = None
    session_id: str | None = None
    model: str | None = None
    # NOT emitted by any shipped ``codex`` version — its ``turn.completed``
    # carries usage and nothing else. Read anyway so a future version that adds
    # it is honoured rather than ignored; until then ``_run_once`` measures the
    # wall time and ``_finalise`` uses that.
    duration_ms: int | None = None
    is_error: bool = False
    stop_reason: str | None = None
    # Did the CLI reach its OWN end-of-turn payload? Tracked explicitly rather
    # than inferred, because a SUCCESSFUL terminal payload leaves
    # ``stop_reason`` at ``None`` — exactly the value a stream that stopped
    # early also has, so the two were indistinguishable. Measured before this
    # existed: stdout truncated mid-object reported ``stop_reason="complete"``,
    # ``partial=False``, and handed back half an answer as a finished one.
    saw_terminal: bool = False
    # A NON-fatal diagnostic the CLI attached to an item. Collected rather than
    # folded into ``is_error``: the run continues and usually succeeds, and
    # promoting "command output truncated" to a failed run would be wrong.
    error: str | None = None


def _events_from_payload(
    payload: dict[str, Any], log: _ItemLog
) -> list[tuple[StreamEvent | None, _EventDelta]]:
    """Translate one JSONL payload into zero or more
    ``(StreamEvent | None, _EventDelta)`` tuples.

    ``(None, delta)`` means the payload carries state (usage / session id /
    model) but nothing a consumer would render.

    A list rather than a generator, unlike the Claude parser: nothing here
    awaits, and an async generator that never suspends is a coroutine frame per
    payload for no benefit — measurable on a long session, and the reason the
    caller's loop is a plain ``for``.
    """
    if "msg" in payload and isinstance(payload.get("msg"), dict):
        return _legacy_events(payload, log)
    return _thread_events(payload, log)


def _thread_events(
    payload: dict[str, Any], log: _ItemLog
) -> list[tuple[StreamEvent | None, _EventDelta]]:
    """The current vocabulary: ``thread.*`` / ``turn.*`` / ``item.*``."""
    ptype = payload.get("type")

    if ptype == "thread.started":
        thread_id = payload.get("thread_id")
        return [(None, _EventDelta(session_id=str(thread_id) if thread_id else None))]

    if ptype == "turn.started":
        return []

    if ptype == "turn.completed":
        duration = payload.get("duration_ms")
        return [
            (
                None,
                _EventDelta(
                    usage=_usage_from(payload.get("usage")),
                    duration_ms=_as_int(duration) if duration is not None else None,
                    # The CLI's own end-of-turn marker. Reaching it is what
                    # separates "the run finished" from "the stream stopped".
                    saw_terminal=True,
                ),
            )
        ]

    if ptype == "turn.failed":
        error = _as_dict(payload.get("error"))
        message = str(error.get("message") or "the CLI reported a failed turn")
        return [
            (
                StreamEvent("step", text=f"turn_failed:{message}"),
                _EventDelta(
                    is_error=True,
                    stop_reason="turn_failed",
                    error=message,
                    # A FAILED turn is still a completed stream: the CLI said
                    # how it ended. Only a stream that stops without saying
                    # anything is malformed.
                    saw_terminal=True,
                ),
            )
        ]

    if ptype == "error":
        # Stream-level, not attributable to the turn: a broken pipe, a
        # transport fault. Distinct from ``turn.failed`` because the fix is.
        message = str(payload.get("message") or "the CLI reported a stream error")
        return [
            (
                StreamEvent("step", text=f"error:{message}"),
                _EventDelta(is_error=True, stop_reason="cli_reported_error", error=message),
            )
        ]

    if ptype in ("item.started", "item.updated", "item.completed"):
        item = payload.get("item")
        if not isinstance(item, dict):
            return []
        return _item_events(item, phase=str(ptype).split(".", 1)[1], log=log)

    # Unknown type — ignore. The CLI adds event types over time; forward-compat
    # is "don't crash".
    return []


def _item_events(
    item: dict[str, Any], *, phase: str, log: _ItemLog
) -> list[tuple[StreamEvent | None, _EventDelta]]:
    """One ``item.*`` payload → its events.

    ``phase`` is ``started`` / ``updated`` / ``completed``. Only ``completed``
    folds text into the turn's answer: an item's text is cumulative, so folding
    an update AND the completion would double the answer, while emitting only
    the not-yet-emitted suffix keeps a live consumer incremental. That split —
    emit suffixes, fold once — is the whole reason :class:`_ItemLog` exists.
    """
    item_id = str(item.get("id") or "")
    # Two spellings in the wild for the same field. ``item_type`` was the
    # earlier one and still appears in some builds; reading only ``type`` there
    # silently drops every item.
    itype = str(item.get("type") or item.get("item_type") or "")

    if itype == "agent_message":
        text = str(item.get("text") or "")
        chunk = log.suffix(f"msg:{item_id}", text)
        ev = StreamEvent("message_delta", text=chunk) if chunk else None
        return [(ev, _EventDelta(text=text if phase == "completed" else ""))]

    if itype == "reasoning":
        text = str(item.get("text") or "")
        chunk = log.suffix(f"reason:{item_id}", text)
        # Surfaced live as a ``message_delta`` so a consumer sees the reasoning
        # in real time, but folded into ``thinking`` and NOT into ``output``:
        # that is the response, this is how it got there.
        ev = StreamEvent("message_delta", text=chunk) if chunk else None
        return [(ev, _EventDelta(thinking=text if phase == "completed" else ""))]

    if itype == "command_execution":
        return _paired(
            item_id=item_id,
            phase=phase,
            log=log,
            name="shell",
            arguments={"command": item.get("command")},
            result=lambda: _command_result(item),
        )

    if itype == "file_change":
        changes = _as_list(item.get("changes"))
        return _paired(
            item_id=item_id,
            phase=phase,
            log=log,
            name="apply_patch",
            arguments={"changes": changes},
            result=lambda: _render_changes(changes),
        )

    if itype == "mcp_tool_call":
        server = str(item.get("server") or "")
        tool = str(item.get("tool") or "")
        return _paired(
            item_id=item_id,
            phase=phase,
            log=log,
            # The CLI's own naming convention for an MCP tool, so a name in an
            # agentkit audit record matches the name in a Codex transcript.
            name=f"mcp__{server}__{tool}" if server or tool else "mcp",
            arguments=dict(_as_dict(item.get("arguments"))),
            result=lambda: _flatten_content(item.get("result")),
        )

    if itype == "web_search":
        return _paired(
            item_id=item_id,
            phase=phase,
            log=log,
            name="web_search",
            arguments={"query": item.get("query")},
            result=lambda: "",
        )

    if itype == "todo_list":
        if phase == "started":
            return []
        entries = _as_list(item.get("items"))
        done = sum(1 for e in entries if isinstance(e, dict) and e.get("completed"))
        return [(StreamEvent("step", text=f"plan:{done}/{len(entries)}"), _EventDelta())]

    if itype == "error":
        # An item-level warning: the run continues. Recorded, never promoted to
        # a failed run — "command output truncated" is not a failure.
        message = str(item.get("message") or "")
        return [(StreamEvent("step", text=f"item_error:{message}"), _EventDelta(error=message))]

    return []


def _paired(
    *,
    item_id: str,
    phase: str,
    log: _ItemLog,
    name: str,
    arguments: dict[str, Any],
    result: Callable[[], str],
) -> list[tuple[StreamEvent | None, _EventDelta]]:
    """A tool item → at most one ``tool_call`` and, on completion, one ``tool_result``.

    Every tool-ish item type has the same two problems and this is the one
    place they are solved. Some arrive as a ``started``/``completed`` pair and
    some (``file_change``, ``web_search``) only ever complete, so the call
    event has to be emitted by whichever phase gets there first — and exactly
    once, or a consumer counting tool calls counts every paired item twice.

    ``result`` is a callable so the completion text is only rendered when there
    is a completion to render; several of them walk a list.
    """
    out: list[tuple[StreamEvent | None, _EventDelta]] = []
    call = ToolCall(id=item_id, name=name, arguments={k: v for k, v in arguments.items() if v is not None})
    if log.first_call(f"call:{item_id}"):
        out.append((StreamEvent("tool_call", tool_call=call), _EventDelta()))
    if phase == "completed":
        out.append(
            (StreamEvent("tool_result", tool_call=call, tool_result=result()), _EventDelta())
        )
    return out


def _command_result(item: dict[str, Any]) -> str:
    """A shell item's output, with its exit code when the run failed.

    The exit code is appended only on failure. On success it is noise on every
    single command; on failure it is the first thing a reader wants and is
    frequently the ONLY thing there is — a command that fails silently has an
    empty ``aggregated_output`` and a non-zero code, and without this the
    tool_result would be the empty string.
    """
    output = str(item.get("aggregated_output") or "")
    code = item.get("exit_code")
    if code not in (None, 0):
        suffix = f"[exit {code}]"
        return f"{output}\n{suffix}" if output else suffix
    return output


def _render_changes(changes: Any) -> str:
    """A patch item's changes as ``kind path`` lines."""
    if not isinstance(changes, list):
        return ""
    return "\n".join(
        f"{c.get('kind', '?')} {c.get('path', '?')}" for c in changes if isinstance(c, dict)
    )


def _flatten_content(result: Any) -> str:
    """Fold an MCP result (``{"content": [{"type": "text", "text": …}]}``, a
    bare string, or a list of blocks) into a single string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        return _flatten_content(content) if content is not None else json.dumps(result, default=str)
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(result)


def _usage_from(raw: Any) -> Usage:
    """Codex's token counts → agentkit's ``Usage``.

    The subtraction in :func:`_split_input_tokens` is the load-bearing line;
    see its docstring. ``cache_write_tokens`` stays 0 because Codex reports no
    cache-creation figure — inventing one from the difference would be a number
    with no source.
    """
    if not isinstance(raw, dict):
        return Usage()
    fresh, cached = _split_input_tokens(
        _as_int(raw.get("input_tokens")), _as_int(raw.get("cached_input_tokens"))
    )
    return Usage(
        input_tokens=fresh,
        output_tokens=_as_int(raw.get("output_tokens")),
        cache_read_tokens=cached,
    )


def _legacy_events(
    payload: dict[str, Any], log: _ItemLog
) -> list[tuple[StreamEvent | None, _EventDelta]]:
    """The older vocabulary: ``{"id": …, "msg": {"type": …}}``.

    Mapped onto the same ``_EventDelta`` as the thread events so everything
    downstream — the fold, the stop-reason priority, the cost, the terminal
    event — has exactly one implementation. Only the reading differs.
    """
    msg = _as_dict(payload.get("msg"))
    mtype = str(msg.get("type") or "")
    # The legacy stream keys tool events on ``call_id``; the outer ``id`` is a
    # per-event sequence number and would make every begin/end pair look like
    # two unrelated items.
    call_id = str(msg.get("call_id") or payload.get("id") or "")

    if mtype == "session_configured":
        sid = msg.get("session_id")
        model = msg.get("model")
        return [
            (
                None,
                _EventDelta(
                    session_id=str(sid) if sid else None,
                    model=str(model) if model else None,
                ),
            )
        ]

    if mtype in ("task_started", "turn_aborted"):
        return []

    if mtype == "agent_message_delta":
        chunk = str(msg.get("delta") or "")
        if not chunk:
            return []
        # Tracked so the whole ``agent_message`` that follows emits only what
        # the deltas did not — the same suffix rule the thread vocabulary uses,
        # reached from the other direction.
        key = "msg:legacy"
        log.emitted[key] = log.emitted.get(key, 0) + len(chunk)
        return [(StreamEvent("message_delta", text=chunk), _EventDelta())]

    if mtype == "agent_message":
        text = str(msg.get("message") or "")
        chunk = log.suffix("msg:legacy", text)
        # Reset: the legacy stream can carry several assistant messages in one
        # turn, and the next one starts its own delta run from zero.
        log.emitted.pop("msg:legacy", None)
        ev = StreamEvent("message_delta", text=chunk) if chunk else None
        return [(ev, _EventDelta(text=text))]

    if mtype == "agent_reasoning_delta":
        chunk = str(msg.get("delta") or "")
        if not chunk:
            return []
        key = "reason:legacy"
        log.emitted[key] = log.emitted.get(key, 0) + len(chunk)
        return [(StreamEvent("message_delta", text=chunk), _EventDelta())]

    if mtype == "agent_reasoning":
        text = str(msg.get("text") or "")
        chunk = log.suffix("reason:legacy", text)
        log.emitted.pop("reason:legacy", None)
        ev = StreamEvent("message_delta", text=chunk) if chunk else None
        return [(ev, _EventDelta(thinking=text))]

    if mtype == "exec_command_begin":
        command = msg.get("command")
        return _paired(
            item_id=call_id,
            phase="started",
            log=log,
            name="shell",
            arguments={"command": command, "cwd": msg.get("cwd")},
            result=lambda: "",
        )

    if mtype == "exec_command_end":
        return _paired(
            item_id=call_id,
            phase="completed",
            log=log,
            name="shell",
            arguments={},
            result=lambda: _command_result(
                {
                    "aggregated_output": (str(msg.get("stdout") or "") + str(msg.get("stderr") or "")),
                    "exit_code": msg.get("exit_code"),
                }
            ),
        )

    if mtype == "mcp_tool_call_begin":
        invocation = _as_dict(msg.get("invocation"))
        server = str(invocation.get("server") or msg.get("server") or "")
        tool = str(invocation.get("tool") or msg.get("tool") or "")
        return _paired(
            item_id=call_id,
            phase="started",
            log=log,
            name=f"mcp__{server}__{tool}" if server or tool else "mcp",
            arguments=dict(invocation.get("arguments") or _as_dict(msg.get("arguments"))),
            result=lambda: "",
        )

    if mtype == "mcp_tool_call_end":
        return _paired(
            item_id=call_id,
            phase="completed",
            log=log,
            name="mcp",
            arguments={},
            result=lambda: _flatten_content(msg.get("result")),
        )

    if mtype == "patch_apply_begin":
        return _paired(
            item_id=call_id,
            phase="started",
            log=log,
            name="apply_patch",
            arguments={"changes": sorted(_as_dict(msg.get("changes")))},
            result=lambda: "",
        )

    if mtype == "patch_apply_end":
        return _paired(
            item_id=call_id,
            phase="completed",
            log=log,
            name="apply_patch",
            arguments={},
            result=lambda: str(msg.get("stdout") or ""),
        )

    if mtype == "token_count":
        info = _as_dict(msg.get("info"))
        total = info.get("total_token_usage") if isinstance(info, dict) else None
        # Cumulative, not per-call: the LAST one is the turn's total, and
        # ``fold`` replaces rather than adds for exactly this reason.
        return [(None, _EventDelta(usage=_usage_from(total)))]

    if mtype == "error":
        message = str(msg.get("message") or "the CLI reported an error")
        return [
            (
                StreamEvent("step", text=f"error:{message}"),
                _EventDelta(
                    is_error=True,
                    stop_reason="cli_reported_error",
                    error=message,
                    # A reported error is a stream that ended by SAYING so,
                    # which is not the same as one that stopped mid-sentence.
                    saw_terminal=True,
                ),
            )
        ]

    if mtype == "task_complete":
        # The LEGACY vocabulary's end-of-turn marker — the counterpart of
        # ``turn.completed`` on the current one. It carries nothing this
        # cognition needs (the text arrived on ``agent_message``, the usage on
        # ``token_count``), which is why it used to return an empty list. It
        # still has to say the stream ENDED: without that, a complete run on an
        # older ``codex`` reads as ``malformed_output``, and a version-compat
        # feature would have become a version-compat bug.
        return [(None, _EventDelta(saw_terminal=True))]

    return []


__all__ = [
    "CODEX_NATIVE_TOOLS",
    "CODEX_SANDBOX_CAPS",
    "ApprovalMode",
    "CodexCliCognition",
    "CodexCliSession",
    "ReasoningEffort",
    "SandboxMode",
]
