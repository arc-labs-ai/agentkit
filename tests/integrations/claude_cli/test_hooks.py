"""``hook_settings`` — the middleware chain reaching the CLI's own tools.

``ClaudeCliCognition`` hands the whole loop to the CLI, and the CLI runs
``Write`` / ``Edit`` / ``Bash`` / ``WebFetch`` itself. Nothing in that path goes
through the ``Invoker``, so ``egress``, ``guard`` and every other tool-chain
middleware stop applying — silently, while the chain still LOOKS wired. These
tests pin the seam that closes that hole: a generated ``PreToolUse`` hook that
calls back into the very same ``chain()`` the ``Invoker`` would have run.

The two load-bearing tests here are the ones the spec named:

* a hook generated from a chain refuses a write outside an allowlist, and the
  CLI reports the refusal (deterministically by running the generated script
  exactly as the CLI runs it, and once against the real binary);
* a hook that itself fails does not take the session down — and, separately,
  does not let the call through either. Not-crashing and allowing are two
  different things, and a guard that fails open is not a guard.

Most tests drive ``_decide`` directly, because that is where the policy is.
The ones that shell out are the ones where the sharp edge is the wire format:
the CLI reads one JSON object off stdout and an exit code, and getting either
wrong turns a refusal into a shrug.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agentkit.capabilities.guardrails import Guardrail
from agentkit.integrations.claude_cli import HookSettings, hook_settings
from agentkit.kernel.middleware import BaseMiddleware, Blocked, MiddlewareContext
from agentkit.middlewares import audit, egress, memoize, tracing
from agentkit.middlewares.guard import SecurityMiddleware
from agentkit.testing import make_test_ctx

pytestmark = pytest.mark.asyncio

real_cli = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


# ── test middlewares ────────────────────────────────────────────────────────


class _PathAllowlist(BaseMiddleware):
    """Refuse a write whose ``file_path`` is outside ``prefix``."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    async def on_request(self, ctx: MiddlewareContext) -> None:
        path = str((ctx.data or {}).get("file_path", ""))
        if not path.startswith(self._prefix):
            raise Blocked(f"path {path!r} is outside {self._prefix}", {"path": path})


class _Explodes(BaseMiddleware):
    """A guard with a bug in it. The interesting case, not the tidy one."""

    async def on_request(self, ctx: MiddlewareContext) -> None:
        raise RuntimeError("guard is broken")


class _Hangs(BaseMiddleware):
    async def on_request(self, ctx: MiddlewareContext) -> None:
        await asyncio.sleep(30)


class _RewritesArguments(BaseMiddleware):
    """Approve-with-changes, which a ``PreToolUse`` hook cannot express."""

    async def on_request(self, ctx: MiddlewareContext) -> None:
        from dataclasses import replace

        ctx.request = replace(ctx.request, arguments={"file_path": "/tmp/redirected"})


class _Counts(BaseMiddleware):
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def on_request(self, ctx: MiddlewareContext) -> None:
        await asyncio.sleep(0.01)  # give the other in-flight calls a chance to interleave
        self.seen.append(str((ctx.data or {}).get("file_path", "")))


class _NeedsTheResult(BaseMiddleware):
    async def on_response(self, ctx: MiddlewareContext, result: Any) -> Any:
        return result


# ── helpers ─────────────────────────────────────────────────────────────────


def _settings(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _hook_command(path: Path) -> str:
    entry = _settings(path)["hooks"]["PreToolUse"][0]
    return str(entry["hooks"][0]["command"])


async def _run_hook(
    path: Path, tool: str, tool_input: dict[str, Any], *, command: str | None = None
) -> tuple[int, str, str]:
    """Run the generated hook exactly as the CLI runs it: shell command, the
    hook payload on stdin, one JSON object expected on stdout."""
    return await _run_hook_raw(
        path,
        json.dumps(
            {
                "session_id": "test-session",
                "transcript_path": "/dev/null",
                "cwd": "/workspace",
                "hook_event_name": "PreToolUse",
                "tool_name": tool,
                "tool_input": tool_input,
            }
        ),
        command=command,
    )


async def _run_hook_raw(
    path: Path, payload: str, *, command: str | None = None
) -> tuple[int, str, str]:
    """The same, with the payload verbatim — for the shapes a well-formed
    ``_run_hook`` call cannot produce."""
    proc = await asyncio.create_subprocess_shell(
        command or _hook_command(path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(payload.encode()), timeout=60)
    return proc.returncode or 0, out.decode(), err.decode()


def _decision(stdout: str) -> dict[str, Any]:
    body = json.loads(stdout or "{}")
    return dict(body.get("hookSpecificOutput") or {})


# ── 1. the spec's first test: a refusal that reaches the CLI ────────────────


async def test_a_generated_hook_refuses_a_write_outside_the_allowlist() -> None:
    """The whole feature in one test. The chain is the SAME object list an
    ``Invoker`` would take; the hook runs it and comes back with a deny in the
    CLI's own shape."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/workspace/")], ctx=ctx, tools=("Write", "Edit")
    ) as settings:
        code, out, _ = await _run_hook(settings.path, "Write", {"file_path": "/etc/passwd"})

        assert code == 0, "a refusal is an ordinary outcome, not a crashed hook"
        decision = _decision(out)
        assert decision["permissionDecision"] == "deny"
        assert "/etc/passwd" in decision["permissionDecisionReason"]


async def test_an_allowed_write_gets_no_decision_at_all() -> None:
    """The chain can only SUBTRACT. Returning ``permissionDecision: "allow"``
    would bypass the CLI's own permission system — the CLI's changelog carries
    a fix for exactly that ("PreToolUse auto-allow hooks bypassing tool
    restrictions"). A pass is silence, so the CLI's normal flow decides."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/workspace/")], ctx=ctx, tools=("Write",)
    ) as settings:
        code, out, _ = await _run_hook(settings.path, "Write", {"file_path": "/workspace/ok.txt"})

    assert code == 0
    assert "permissionDecision" not in _decision(out)


# ── 2. the spec's second test: a broken hook must not take the session down ──


async def test_a_hook_whose_guard_raises_denies_and_the_session_survives() -> None:
    """An ordinary exception is not a refusal, but it is not an allow either.
    A guard that cannot answer must fail CLOSED — and the session keeps going,
    because a deny is an in-band outcome the model is told about."""
    ctx = make_test_ctx()
    async with hook_settings(middleware=[_Explodes()], ctx=ctx, tools=("Bash",)) as settings:
        code, out, _ = await _run_hook(settings.path, "Bash", {"command": "ls"})
        assert code == 0
        assert _decision(out)["permissionDecision"] == "deny"
        assert "RuntimeError" in _decision(out)["permissionDecisionReason"]

        # ...and the listener is still there for the next tool call.
        code2, out2, _ = await _run_hook(settings.path, "Bash", {"command": "pwd"})
        assert code2 == 0
        assert _decision(out2)["permissionDecision"] == "deny"


async def test_a_hook_that_cannot_reach_the_chain_denies_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """The transport failure. The listener is gone (crashed parent, closed
    settings); the script must still exit 0 with a well-formed deny, because a
    non-zero exit the CLI does not recognise is treated as a NON-blocking error
    and the tool runs."""
    ctx = make_test_ctx()
    settings = hook_settings(middleware=[_Explodes()], ctx=ctx, tools=("Bash",))
    orphan = tmp_path / "orphaned_hook.py"
    orphan.write_text(settings.script_path.read_text())  # keeps the now-dead socket path
    await settings.aclose()

    code, out, _ = await _run_hook(
        settings.path, "Bash", {"command": "ls"}, command=f"{sys.executable} {orphan}"
    )

    assert code == 0
    assert _decision(out)["permissionDecision"] == "deny"
    assert "could not reach" in _decision(out)["permissionDecisionReason"]


async def test_a_hook_that_hangs_is_denied_by_the_server_deadline() -> None:
    """A hook that never answers is worse than one that refuses: the CLI's own
    hook timeout treats a killed hook as a non-blocking error and RUNS the
    tool. So the deadline is enforced innermost, where it can still produce a
    deny."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_Hangs()], ctx=ctx, tools=("Bash",), timeout_s=0.2
    ) as settings:
        code, out, _ = await _run_hook(settings.path, "Bash", {"command": "sleep 60"})

    assert code == 0
    assert _decision(out)["permissionDecision"] == "deny"
    assert "0.2" in _decision(out)["permissionDecisionReason"]


async def test_the_script_bounds_itself_even_when_nothing_ever_arrives() -> None:
    """The socket deadline does not cover reading stdin. A hook that hangs
    there would be killed by the CLI's own timeout — a NON-blocking error, so
    the tool runs. The script's alarm covers the whole of itself, so it refuses
    before anyone kills it."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Write",), timeout_s=0.1
    ) as settings:
        proc = await asyncio.create_subprocess_shell(
            _hook_command(settings.path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        # Never write, never close stdin: the CLI died mid-write, or the pipe
        # wedged. ``communicate()`` would close it for us, which is the one
        # thing that must not happen here.
        reading = asyncio.create_task(proc.stdout.read())
        await asyncio.wait_for(proc.wait(), timeout=20)
        out = await reading
        proc.stdin.close()

    assert proc.returncode == 0
    assert _decision(out.decode())["permissionDecision"] == "deny"


# ── 3. Blocked vs an ordinary exception ─────────────────────────────────────


async def test_blocked_and_an_ordinary_exception_both_deny_but_read_differently() -> None:
    """Both refuse; the REASON has to say which, because "policy said no" and
    "the policy code is broken" call for different fixes and the reason is what
    the model (and the operator) sees."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/workspace/")], ctx=ctx, tools=("Write",)
    ) as blocked_settings:
        refusal = await blocked_settings._decide("Write", {"file_path": "/etc/hosts"})

    async with hook_settings(middleware=[_Explodes()], ctx=ctx, tools=("Write",)) as broken:
        fault = await broken._decide("Write", {"file_path": "/etc/hosts"})

    assert refusal["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert fault["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "RuntimeError" not in refusal["hookSpecificOutput"]["permissionDecisionReason"]
    assert "RuntimeError" in fault["hookSpecificOutput"]["permissionDecisionReason"]


# ── 4. what cannot work must fail at generation time ────────────────────────


@pytest.mark.parametrize(
    "mw",
    [memoize(), tracing(), audit(), _NeedsTheResult()],
    ids=["memoize", "tracing", "audit", "on_response"],
)
async def test_a_middleware_that_needs_the_result_is_refused_at_generation_time(mw: Any) -> None:
    """A ``PreToolUse`` hook runs BEFORE the CLI's tool and never sees a
    result, so anything that acts on one cannot apply. Silently not applying is
    the bug this whole feature exists to fix, so it is a wiring-time error."""
    ctx = make_test_ctx()
    with pytest.raises(TypeError, match="on_request"):
        hook_settings(middleware=[mw], ctx=ctx, tools=("Write",))


async def test_an_on_request_only_middleware_is_accepted() -> None:
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[egress(Guardrail(egress_allow=("example.com",)))], ctx=ctx, tools=("WebFetch",)
    ) as settings:
        assert settings.path.exists()


async def test_a_middleware_that_rewrites_the_arguments_is_denied() -> None:
    """A ``PreToolUse`` hook can refuse; it cannot hand the CLI different
    arguments. A rewrite that is silently dropped would run the call the
    middleware tried to prevent, so the rewrite is a refusal instead."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_RewritesArguments()], ctx=ctx, tools=("Write",)
    ) as settings:
        body = await settings._decide("Write", {"file_path": "/etc/passwd"})

    assert body["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "rewrite" in body["hookSpecificOutput"]["permissionDecisionReason"]


# ── 5. wiring-time refusals ─────────────────────────────────────────────────


async def test_an_empty_tools_tuple_is_refused() -> None:
    ctx = make_test_ctx()
    with pytest.raises(ValueError, match="at least one tool"):
        hook_settings(middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=())


async def test_a_chain_with_no_middlewares_is_refused() -> None:
    ctx = make_test_ctx()
    with pytest.raises(ValueError, match="no middlewares"):
        hook_settings(middleware=[], ctx=ctx, tools=("Write",))


async def test_a_tool_the_cli_does_not_have_is_refused_unless_opted_in() -> None:
    """A typo'd tool name produces a matcher that never fires — a guard that
    looks wired and enforces nothing."""
    ctx = make_test_ctx()
    with pytest.raises(ValueError, match="Wrtie"):
        hook_settings(middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Wrtie",))

    async with hook_settings(
        middleware=[_PathAllowlist("/w/")],
        ctx=ctx,
        tools=("SomeNewTool",),
        allow_unknown_tools=True,
    ) as settings:
        assert "SomeNewTool" in _settings(settings.path)["hooks"]["PreToolUse"][0]["matcher"]


async def test_caller_settings_merge_only_when_they_carry_no_hooks() -> None:
    """Merging two ``PreToolUse`` arrays means either dropping the caller's
    hooks or running ours twice. Refusing is the honest answer; everything
    provably disjoint is carried through verbatim."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/w/")],
        ctx=ctx,
        tools=("Write",),
        base={"env": {"FOO": "bar"}, "model": "claude-haiku-4-5"},
    ) as settings:
        body = _settings(settings.path)
        assert body["env"] == {"FOO": "bar"}
        assert body["model"] == "claude-haiku-4-5"
        assert "PreToolUse" in body["hooks"]

    with pytest.raises(ValueError, match="hooks"):
        hook_settings(
            middleware=[_PathAllowlist("/w/")],
            ctx=ctx,
            tools=("Write",),
            base={"hooks": {"PostToolUse": []}},
        )


# ── 6. the pieces that only work because the chain runs in-process ──────────


async def test_egress_checks_the_url_the_cli_tool_actually_carries() -> None:
    """``Egress`` reads ``request.url_arg``; a native CLI tool has no
    ``ToolRequest`` to carry one. Without the tool table, ``egress()`` would sit
    in the chain checking nothing — inert in exactly the way ``Egress.__init__``
    refuses to be at construction."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[egress(Guardrail(egress_allow=("example.com",)))],
        ctx=ctx,
        tools=("WebFetch",),
    ) as settings:
        blocked = await settings._decide("WebFetch", {"url": "http://169.254.169.254/latest/meta-data"})
        allowed = await settings._decide("WebFetch", {"url": "https://example.com/docs"})

    assert blocked["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "permissionDecision" not in (allowed.get("hookSpecificOutput") or {})


async def test_concurrent_tool_calls_each_get_their_own_decision() -> None:
    """The CLI runs tools in parallel, so hooks arrive concurrently. Each call
    is its own ``Call`` over the shared chain; nothing is stashed on the ctx."""
    ctx = make_test_ctx()
    counter = _Counts()
    async with hook_settings(
        middleware=[counter, _PathAllowlist("/workspace/")], ctx=ctx, tools=("Write",)
    ) as settings:
        results = await asyncio.gather(
            *(
                _run_hook(settings.path, "Write", {"file_path": p})
                for p in ("/workspace/a", "/etc/b", "/workspace/c", "/etc/d")
            )
        )

    decisions = [_decision(out).get("permissionDecision") for _, out, _ in results]
    assert decisions == [None, "deny", None, "deny"]
    assert sorted(counter.seen) == ["/etc/b", "/etc/d", "/workspace/a", "/workspace/c"]
    assert len(settings.decisions) == 4


async def test_decisions_are_recorded_for_the_operator() -> None:
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/workspace/")], ctx=ctx, tools=("Write",)
    ) as settings:
        await settings._decide("Write", {"file_path": "/etc/passwd"})
        await settings._decide("Write", {"file_path": "/workspace/ok"})

    assert [(d.tool, d.allowed) for d in settings.decisions] == [
        ("Write", False),
        ("Write", True),
    ]


# ── 7. lifetime ─────────────────────────────────────────────────────────────


async def test_closing_removes_the_whole_directory_and_is_idempotent() -> None:
    """Script, socket and settings live in one 0700 directory this process
    owns, so a stale settings file cannot outlive the listener it points at."""
    ctx = make_test_ctx()
    settings = hook_settings(middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Write",))
    directory = settings.path.parent
    assert settings.path.exists()
    assert (directory.stat().st_mode & 0o777) == 0o700

    await settings.aclose()
    await settings.aclose()
    assert not directory.exists()


# ── 8. the real binary ──────────────────────────────────────────────────────


@real_cli
async def test_the_real_cli_accepts_the_settings_we_generate() -> None:
    """A mock cannot tell us the settings schema drifted. The CLI validates
    ``--settings`` at startup, so a bad shape is a non-zero exit here."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/workspace/")], ctx=ctx, tools=("Write", "Edit")
    ) as settings:
        proc = await asyncio.to_thread(
            subprocess.run,
            [
                "claude",
                "-p",
                "Reply with the single word: OK",
                "--model",
                "claude-haiku-4-5-20251001",
                "--settings",
                str(settings.path),
                "--tools",
                "",
                "--permission-mode",
                "dontAsk",
                "--max-turns",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            stdin=subprocess.DEVNULL,
        )

    assert "settings" not in proc.stderr.lower(), proc.stderr[:600]
    assert proc.returncode == 0, f"exit={proc.returncode} stderr={proc.stderr[:600]}"


@real_cli
async def test_the_real_cli_reports_the_refusal_and_the_write_never_happens(
    tmp_path: Path,
) -> None:
    """End to end: the CLI is told to write a file, the generated hook refuses
    it through the chain, and the CLI reports the refusal instead of writing.

    ``permission_mode`` is ``bypassPermissions`` deliberately — it is the
    setting under which the CLI would otherwise write without asking anyone,
    which is precisely the hole this feature fills."""
    target = tmp_path / "forbidden.txt"
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/nowhere-at-all/")], ctx=ctx, tools=("Write",)
    ) as settings:
        proc = await asyncio.to_thread(
            subprocess.run,
            [
                "claude",
                "-p",
                f"Use the Write tool to create {target} containing the word hello. "
                f"If the write is refused, reply with the refusal reason verbatim.",
                "--model",
                "claude-haiku-4-5-20251001",
                "--settings",
                str(settings.path),
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "bypassPermissions",
                "--max-turns",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
        )

    assert not target.exists(), "the hook refused the write and the CLI wrote it anyway"
    assert "nowhere-at-all" in proc.stdout, proc.stdout[-2000:]
    assert any(d.tool == "Write" and not d.allowed for d in settings.decisions)


# ── 9. the surface the spec asked for ───────────────────────────────────────


async def test_the_documented_call_shape_works() -> None:
    """The snippet in the spec, verbatim in shape: build the chain once, hand
    ``.path`` to the cognition."""
    from agentkit.agents.cognition import ClaudeCliCognition

    ctx = make_test_ctx()
    tool_chain = [egress(Guardrail(egress_allow=("example.com",))), _PathAllowlist("/workspace/")]
    settings = hook_settings(middleware=tool_chain, ctx=ctx, tools=("Write", "Edit", "Bash"))
    try:
        assert isinstance(settings, HookSettings)
        cognition = ClaudeCliCognition(model="claude-haiku-4-5", settings=settings.path)
        argv = cognition._build_argv("noop", system_prompt="")
        assert "--settings" in argv
        assert str(settings.path) in argv
    finally:
        await settings.aclose()


async def test_the_hook_script_never_imports_agentkit() -> None:
    """The script is a dumb pipe by design: read stdin, hand it to the parent,
    print the answer. Every line of policy stays in the one chain the
    ``Invoker`` runs, so there is no second implementation to drift."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Write",)
    ) as settings:
        source = settings.script_path.read_text()
        command = _hook_command(settings.path)

    assert "import agentkit" not in source
    assert source.count("import ") <= 5, "the pipe grew a dependency; policy belongs in the chain"
    assert sys.executable in command


# ── 10. the payload the CLI actually sends ──────────────────────────────────
# `_decide` is called directly by most tests above, which means the step that
# turns the CLI's JSON into a `Call` — the only place the two processes agree
# on anything — was never exercised. Everything in this section goes over the
# wire on purpose.


class _RefusesOneTool(BaseMiddleware):
    """A guard keyed on the TOOL NAME, which is how most real ones are written
    (``if ctx.request.name == "Bash"``). It is inert unless the name the CLI
    sent is the name the chain sees."""

    async def on_request(self, ctx: MiddlewareContext) -> None:
        if ctx.request.name == "Bash":
            raise Blocked("no shell in this run")


class _RefusesSideEffects(BaseMiddleware):
    """A guard keyed on ``side_effecting``, the other ``ToolRequest`` field
    ``_NATIVE_TOOLS`` fills in."""

    async def on_request(self, ctx: MiddlewareContext) -> None:
        if getattr(ctx.request, "side_effecting", False):
            raise Blocked(f"{ctx.request.name} changes the world")


async def test_the_tool_name_the_cli_sent_is_the_one_the_chain_sees() -> None:
    """A guard that keys on ``ctx.request.name`` is the common shape, and it is
    only a guard if the name survives the trip. Both directions, over the wire:
    the guarded name refuses and the unguarded one does not."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_RefusesOneTool()], ctx=ctx, tools=("Bash", "Read")
    ) as settings:
        _, refused, _ = await _run_hook(settings.path, "Bash", {"command": "ls"})
        _, passed, _ = await _run_hook(settings.path, "Read", {"file_path": "/etc/hosts"})

    assert _decision(refused)["permissionDecision"] == "deny"
    assert "no shell in this run" in _decision(refused)["permissionDecisionReason"]
    assert "permissionDecision" not in _decision(passed)
    assert [(d.tool, d.allowed) for d in settings.decisions] == [("Bash", False), ("Read", True)]


async def test_a_payload_with_no_tool_name_is_refused_rather_than_waved_through() -> None:
    """A payload with no ``tool_name`` used to become ``ToolRequest(name="")``:
    a well-formed call about nothing, which every name-keyed guard passes. A
    pass emits no ``permissionDecision``, so under ``bypassPermissions`` the
    CLI then ran the tool with the chain having checked nothing. Not being able
    to read the payload is a failure, and every failure here is a deny."""
    ctx = make_test_ctx()
    async with hook_settings(middleware=[_RefusesOneTool()], ctx=ctx, tools=("Bash",)) as settings:
        code, out, _ = await _run_hook_raw(
            settings.path, json.dumps({"tool_input": {"command": "rm -rf /"}})
        )

    assert code == 0
    assert _decision(out)["permissionDecision"] == "deny"
    assert "tool_name" in _decision(out)["permissionDecisionReason"]
    assert [(d.tool, d.allowed) for d in settings.decisions] == [("", False)]


async def test_a_tool_input_that_is_not_an_object_is_refused() -> None:
    """The same hole through the other field. ``tool_input`` that is not a dict
    used to be replaced by ``{}`` — and ``Egress`` reads
    ``arguments[url_arg]``, where a MISSING url is not a blocked url, so the
    fetch was allowed with the URL never looked at. The arguments a guard is
    meant to inspect being unreadable is a refusal, not an empty inspection."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[egress(Guardrail(egress_allow=("example.com",)))],
        ctx=ctx,
        tools=("WebFetch",),
    ) as settings:
        code, out, _ = await _run_hook_raw(
            settings.path,
            json.dumps(
                {"tool_name": "WebFetch", "tool_input": [{"url": "http://169.254.169.254/"}]}
            ),
        )

    assert code == 0
    assert _decision(out)["permissionDecision"] == "deny"
    assert "tool_input" in _decision(out)["permissionDecisionReason"]
    assert [(d.tool, d.allowed) for d in settings.decisions] == [("WebFetch", False)]


async def test_a_payload_that_is_not_json_at_all_fails_closed_and_is_recorded() -> None:
    """The transport's own worst case. It must not merely not-crash: a deny
    that leaves ``decisions`` empty looks exactly like a hook that never ran."""
    ctx = make_test_ctx()
    async with hook_settings(middleware=[_RefusesOneTool()], ctx=ctx, tools=("Bash",)) as settings:
        code, out, _ = await _run_hook_raw(settings.path, "this is not json")

    assert code == 0
    assert _decision(out)["permissionDecision"] == "deny"
    assert len(settings.decisions) == 1 and not settings.decisions[0].allowed


async def test_a_payload_too_large_for_the_wire_still_fails_closed() -> None:
    """``Write``'s ``content`` is in the payload, so a big write is a big
    payload. The stream limit bounds it; what matters is that overrunning it
    lands on a deny at exit 0 and not on a crashed hook, which the CLI reads as
    non-blocking and follows by running the tool."""
    ctx = make_test_ctx()
    async with hook_settings(middleware=[_RefusesOneTool()], ctx=ctx, tools=("Write",)) as settings:
        code, out, _ = await _run_hook(
            settings.path, "Write", {"file_path": "/w/big", "content": "z" * (6 * 1024 * 1024)}
        )

    assert code == 0
    assert _decision(out)["permissionDecision"] == "deny"


async def test_an_unknown_tool_is_treated_as_side_effecting() -> None:
    """``_NATIVE_TOOLS`` cannot know a tool the CLI grew after this table was
    written, and the default it falls back to is a policy decision: a guard
    keyed on ``side_effecting`` must treat an unrecognised tool as the
    dangerous kind, or ``allow_unknown_tools=True`` quietly widens what the
    chain permits."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_RefusesSideEffects()],
        ctx=ctx,
        tools=("SomeNewTool", "Read"),
        allow_unknown_tools=True,
    ) as settings:
        unknown = await settings._decide("SomeNewTool", {})
        known_readonly = await settings._decide("Read", {"file_path": "/etc/hosts"})

    assert unknown["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "permissionDecision" not in (known_readonly.get("hookSpecificOutput") or {})


# ── 11. the generated artefact, read as the CLI reads it ────────────────────
# This item's whole output is a file a DIFFERENT process consumes. These pin
# the parts of it that no in-process test can see.


async def test_the_matcher_is_anchored_so_it_cannot_fire_for_a_longer_name() -> None:
    """The matcher is a regex the CLI evaluates. Unanchored, a ``Write``
    matcher also fires for a hypothetical ``WriteAll`` — a guard applying to a
    tool nobody chose — and, more to the point, an unanchored ``Read`` matcher
    fires for ``ReadWrite``. Asserting the name appears in the string does not
    notice the anchors going missing; compiling it and testing it does."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Write", "Edit")
    ) as settings:
        matcher = re.compile(_settings(settings.path)["hooks"]["PreToolUse"][0]["matcher"])

    assert matcher.search("Write") and matcher.search("Edit")
    for other in ("WriteAll", "PreWrite", "Editor", "NotebookEdit"):
        assert not matcher.search(other), f"the matcher also fires for {other}"


async def test_the_clis_hook_timeout_is_the_loosest_of_the_three() -> None:
    """Three deadlines, and their ORDER is the fail-closed property: a hook the
    CLI times out is a non-blocking error, which the CLI prints and then runs
    the tool anyway. So the CLI's must fire last, after the script's alarm and
    after the chain's. Nothing in-process can check this — it is two numbers in
    a generated file and one in a generated script."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Write",), timeout_s=3.0
    ) as settings:
        cli_timeout = _settings(settings.path)["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"]
        script_timeout = float(
            next(
                line.split("=", 1)[1]
                for line in settings.script_path.read_text().splitlines()
                if line.startswith("TIMEOUT_S")
            )
        )

    assert settings.timeout_s < script_timeout < cli_timeout, (
        f"chain={settings.timeout_s} script={script_timeout} cli={cli_timeout}: the CLI's "
        f"deadline must fire last, or a slow chain becomes a killed hook and the tool RUNS"
    )


async def test_the_generated_script_is_not_writable_by_anyone_else() -> None:
    """The script is the guard's whole presence on disk and the CLI executes
    whatever is in it. The 0700 directory is the fence; the file's own mode is
    the second one, and it is a single ``chmod`` nothing else asserts."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Write",)
    ) as settings:
        mode = settings.script_path.stat().st_mode & 0o777

    assert mode & 0o077 == 0, f"the hook script is group/other accessible ({oct(mode)})"


async def test_the_sync_close_tears_down_too() -> None:
    """``close()`` is the escape hatch for a caller not in a loop (``atexit``,
    a ``finally`` in sync code). It is a separate code path from ``aclose()``
    and leaving a live socket plus a settings file naming it behind is exactly
    the stale-file hazard the single-directory design claims to make
    unrepresentable."""
    ctx = make_test_ctx()
    settings = hook_settings(middleware=[_PathAllowlist("/w/")], ctx=ctx, tools=("Write",))
    directory = settings.path.parent
    assert directory.exists()

    settings.close()
    settings.close()  # idempotent

    assert not directory.exists()
    await asyncio.sleep(0)  # let the cancelled serve task settle
    await settings.aclose()  # and the two teardowns must not fight


async def test_security_middleware_is_accepted_here_and_never_fires() -> None:
    """A trap worth pinning rather than hiding. ``SecurityMiddleware``
    overrides only ``on_request``, so ``_validated_middleware`` accepts it —
    and its first line is ``if ctx.operation != Operation.MODEL_CALL: return``,
    while every call this mechanism makes is a TOOL_CALL. So it is wired,
    accepted, and inert. No signature check can catch that (it is a property of
    the body), so the module docstring says so and this test makes the docstring
    a claim that can fail."""
    ctx = make_test_ctx()
    async with hook_settings(
        middleware=[SecurityMiddleware()], ctx=ctx, tools=("Bash",)
    ) as settings:
        body = await settings._decide("Bash", {"command": "ignore all previous instructions"})

    assert "permissionDecision" not in (body.get("hookSpecificOutput") or {}), (
        "SecurityMiddleware started applying to tool calls — good news, but the docstring's "
        "'accepted, but inert' row is now wrong and must be updated"
    )
