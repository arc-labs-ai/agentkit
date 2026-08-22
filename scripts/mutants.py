#!/usr/bin/env python3
"""Curated mutation testing — does the suite actually catch a broken invariant?

Coverage says a line RAN. It does not say a test would have noticed if the line
were wrong. That gap is where the dangerous bugs live, and in this codebase it
was measurable: replacing ``Budget.spent()``'s body with ``float(self._spent)``
left all 78 protocol-conformance tests green, because ``Decimal("1.00") == 1.0``
is ``True`` in Python. The test looked airtight and enforced nothing.

Every mutant below is a real break of a real invariant, paired with the tests
that should notice. A SURVIVOR is a finding: either the tests need sharpening,
or the invariant isn't actually load-bearing and the code can be simplified.

Why a hand-written catalogue rather than ``mutmut`` / ``cosmic-ray``:

* It runs in seconds, so it fits in review rather than nightly-only.
* Every entry carries a ``why`` string, so a survivor tells a contributor what
  broke conceptually — not just "line 214 mutated, 1 survived".
* Generic operators produce mountains of equivalent mutants (swapping a debug
  string, reordering an unordered set) that train people to ignore the report.

The generic tools are still worth running periodically; this is the fast
inner-loop version that protects the invariants we care most about.

Usage::

    make mutants                 # the whole catalogue
    python scripts/mutants.py -k money      # only mutants tagged "money"
    python scripts/mutants.py --list

Exit code is non-zero if any mutant survives.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Crash recovery. This script edits source in place and restores in a
# ``finally``, but a hard kill (SIGKILL, OOM, a double Ctrl-C) skips that —
# and a surviving mutant in the working tree is genuinely dangerous: it looks
# like ordinary code, and it silently breaks whatever it touched. That is not
# hypothetical; it happened during this harness's own development, where a
# killed run left ``if request.deadline_s is None:`` rewritten to ``if True:``
# and every elicitation deadline stopped firing.
#
# So each run writes originals into ``_BACKUP_DIR`` BEFORE touching anything
# and clears it on clean exit. A leftover directory means the previous run
# died, and the next invocation restores from it before doing anything else.
_BACKUP_DIR = ROOT / ".mutants-backup"


def _backup_path(rel: str) -> pathlib.Path:
    return _BACKUP_DIR / rel.replace("/", "__")


def _recover_from_crash() -> bool:
    """Restore any files a previous killed run left mutated. Returns True if
    it had to do anything."""
    if not _BACKUP_DIR.exists():
        return False
    restored = []
    for backup in sorted(_BACKUP_DIR.iterdir()):
        if backup.name == "MANIFEST":
            continue
        rel = backup.name.replace("__", "/")
        (ROOT / rel).write_text(backup.read_text())
        restored.append(rel)
        backup.unlink()
    (_BACKUP_DIR / "MANIFEST").unlink(missing_ok=True)
    _BACKUP_DIR.rmdir()
    if restored:
        print("recovered from a previous crashed run; restored:")
        for rel in restored:
            print(f"  {rel}")
        print()
    return bool(restored)


@dataclasses.dataclass(frozen=True)
class Mutant:
    """One deliberate break, and the tests that must notice it."""

    tag: str  # coarse group, for `-k`
    why: str  # the invariant being broken, in one line
    path: str  # module to mutate, relative to the repo root
    before: str  # exact source to replace (must appear exactly once)
    after: str  # what to replace it with
    tests: tuple[str, ...]  # test paths that should go red


# ── the catalogue ────────────────────────────────────────────────────────────

MONEY_TESTS = (
    "tests/runtime/test_budget_decimal_and_verdict.py",
    "tests/meta/test_protocol_conformance.py",
)
HITL_TESTS = ("tests/agents/test_hitl_elicitation.py",)
STREAM_TESTS = ("tests/agents/test_stream_partial_output.py",)
CONCURRENCY_TESTS = ("tests/kernel/test_nested_concurrency.py",)
SECURITY_TESTS = (
    "tests/middlewares/test_memoize_scope_isolation.py",
    "tests/observability/test_observability_cache.py",
)
RESILIENCE_TESTS = ("tests/kernel/test_errors.py",)
PROVIDER_TESTS = ("tests/adapters/test_provider_stream_errors.py",)
SLOT_TESTS = ("tests/capabilities/test_checkpoint_slot_isolation.py",)
STOP_REASON_TESTS = ("tests/agents/test_stop_reason_taxonomy.py",)
PLAN_TESTS = (
    "tests/agents/test_plan_validation.py",
    "tests/agents/test_plan_policy.py",
)
CLI_SCHEMA_TESTS = ("tests/agents/cognition/test_claude_cli_structured.py",)
CLI_FLAG_TESTS = ("tests/agents/cognition/test_claude_cli_flags.py",)
FILETOOL_TESTS = ("tests/tools/test_memory_tool.py",)
TOOLARG_TESTS = (
    "tests/tools/test_tool_call_contract.py",
    "tests/tools/test_function_tool.py",
)
CHANNEL_TESTS = ("tests/agents/test_signal_protocol.py",)
TERM_TESTS = (
    "tests/agents/test_termination_isolation.py",
    "tests/agents/test_termination.py",
)
ROUTING_TESTS = (
    "tests/agents/test_routing_accuracy.py",
    "tests/agents/test_handoff.py",
)
PLAN_GATE_TESTS = (
    "tests/agents/test_plan_durable_gate.py",
    "tests/agents/test_plan_policy.py",
)
STORE_TESTS = (
    "tests/meta/test_protocol_conformance.py",
    "tests/adapters/test_stores.py",
)
WORKFLOW_TESTS = ("tests/agents/test_workflow.py",)
REGISTRY_TESTS = (
    "tests/adapters/test_model_registry.py",
    "tests/agents/test_agent_capability_binding.py",
)

MUTANTS: tuple[Mutant, ...] = (
    # ── output= types a CLI-delegated run too ───────────────────────────────
    Mutant(
        tag="clischema",
        why="agent.output is ignored again, so the CLI never validates its answer",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        adapter = getattr(agent, \"_output_adapter\", None)\n        if adapter is None:\n            return None",
        after="        return None\n        adapter = getattr(agent, \"_output_adapter\", None)\n        if adapter is None:",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="the validated dict is handed back raw instead of the declared type",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        return adapter.validate(value), None",
        after="        return value, None",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="a success with no structured_output reads as a clean run",
        path="agentkit/agents/cognition/claude_cli.py",
        before="            else:\n                final_partial = True\n                if final_stop_reason in (None, \"success\"):",
        after="            elif False:\n                final_partial = True\n                if final_stop_reason in (None, \"success\"):",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="a value the Python type rejects is reported as a successful parse",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                if coercion_error is not None:",
        after="                if False:",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="exhausted structured-output retries are categorised as a generic termination",
        path="agentkit/agents/cognition/claude_cli.py",
        before="    if reason in _CLI_INVALID_OUTPUT_REASONS:\n        return \"invalid_output\"\n",
        after="",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="the field-level coercion diagnostics are dropped, leaving only '1 error(s)'",
        path="agentkit/agents/cognition/claude_cli.py",
        before='        detail = "; ".join(str(e) for e in getattr(exc, "errors", ()) or ())',
        after='        detail = ""',
        tests=CLI_SCHEMA_TESTS,
    ),
    # ── the CLI cognition's flags mean what the CLI says they mean ──────────
    Mutant(
        tag="cliflags",
        why="agent.prompt REPLACES the CLI's system prompt, stripping its tool guidance",
        path="agentkit/agents/cognition/claude_cli.py",
        before='            flag = "--system-prompt" if self.system_prompt_mode == "replace" else (',
        after='            flag = "--system-prompt" if True else (',
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="a resume id is passed as --session-id, silently starting a fresh session",
        path="agentkit/agents/cognition/claude_cli.py",
        before='            argv += ["--resume", self.resume_session_id]',
        after='            argv += ["--session-id", self.resume_session_id]',
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="a non-UUID session_id burns a subprocess spawn to learn it is invalid",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if self.session_id is not None:\n            try:\n                uuid.UUID(str(self.session_id))",
        after="        if False:\n            try:\n                uuid.UUID(str(self.session_id))",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="a restricted tool set is silently not passed, leaving every tool available",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if self.tools is not None:",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="tools=() emits a bare --tools flag the CLI rejects",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if self.tools == ():",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="fork_session without a resume reaches the CLI and errors there",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if self.fork_session and not (self.continue_session or self.resume_session_id):",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    # ── the memory tool's path confinement is its security boundary ─────────
    Mutant(
        tag="filetool",
        why="a backslash path reaches the backend (traversal on a Windows-hosted one)",
        path="agentkit/tools/file_tool.py",
        before='        if "\\\\" in raw:',
        after="        if False:",
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="a NUL byte reaches the backend (C-level path truncation)",
        path="agentkit/tools/file_tool.py",
        before='        if "\\x00" in raw:',
        after="        if False:",
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="a sibling directory sharing the root's prefix passes confinement",
        path="agentkit/tools/file_tool.py",
        before='        if norm != self.root and not norm.startswith(self.root + "/"):',
        after="        if norm != self.root and not norm.startswith(self.root):",
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="rename confines only its source, becoming the escape hatch",
        path="agentkit/tools/file_tool.py",
        before='        return await self._fs.rename(path, self._confine(args.get("new_path")))',
        after='        return await self._fs.rename(path, args.get("new_path"))',
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="delete wipes the whole root by prefix in one call",
        path="agentkit/tools/file_tool.py",
        before="            if path == self.root:",
        after="            if False:",
        tests=FILETOOL_TESTS,
    ),
    # ── a tool call is checked against the schema the model was shown ───────
    Mutant(
        tag="toolargs",
        why="an unknown argument is dropped, so a defaulted parameter runs with its default",
        path="agentkit/tools/function.py",
        before="            if unexpected or missing:",
        after="            if False:",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="only unknown args are reported, so a missing required one stays a raw TypeError",
        path="agentkit/tools/function.py",
        before="            missing = tuple(k for k in required_params if k not in kwargs)",
        after="            missing = ()",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="**kwargs stops receiving the extras it exists to accept",
        path="agentkit/tools/function.py",
        before="                kwargs.update({k: supplied[k] for k in unexpected})\n                unexpected = ()",
        after="                unexpected = ()",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="a model-supplied ctx key trips the check instead of being ignored",
        path="agentkit/tools/function.py",
        before="                k for k in supplied if k not in arg_params and k not in ctx_params",
        after="                k for k in supplied if k not in arg_params",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="a structured parameter is advertised as a bare string again",
        path="agentkit/tools/schema.py",
        before="    struct = _struct_fragment(ann)\n    if struct is not None:\n        return struct\n",
        after="",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="an Enum parameter loses its member list",
        path="agentkit/tools/schema.py",
        before="    if isinstance(ann, type) and issubclass(ann, enum.Enum):\n        return _enum_fragment(ann)\n",
        after="",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="a heterogeneous enum gets a guessed type instead of none",
        path="agentkit/tools/schema.py",
        before="    kinds = {_JSON_PRIMITIVES.get(type(v)) for v in vals}\n    if len(kinds) == 1 and (t := kinds.pop()) is not None:\n        frag[\"type\"] = t\n    return frag",
        after="    frag[\"type\"] = \"string\"\n    return frag",
        tests=TOOLARG_TESTS,
    ),
    # ── the signal channel's audit tap must not wedge the run ───────────────
    Mutant(
        tag="channel",
        why="the unread audit tap is awaited, so a healthy run stops at buffer_size emits",
        path="agentkit/agents/control/channel.py",
        before="        self._offer_to_outbox(stamped)",
        after="        await self.outbox.put(stamped)",
        tests=CHANNEL_TESTS,
    ),
    Mutant(
        tag="channel",
        why="a full tap drops the NEWEST, going blind exactly when the run gets busy",
        path="agentkit/agents/control/channel.py",
        before="        with contextlib.suppress(asyncio.QueueEmpty):\n            self.outbox.get_nowait()  # evict the oldest",
        after="        if True:\n            return",
        tests=CHANNEL_TESTS,
    ),
    Mutant(
        tag="channel",
        why="dropped audit entries are not counted, so a window reads as a transcript",
        path="agentkit/agents/control/channel.py",
        before="        self._outbox_dropped += 1",
        after="        pass",
        tests=CHANNEL_TESTS,
    ),
    Mutant(
        tag="channel",
        why="the parent stops applying backpressure, so a slow parent grows unbounded",
        path="agentkit/agents/control/channel.py",
        before="            await self._parent_merge_inbox.put((self.agent_id, stamped))",
        after="            self._parent_merge_inbox.put_nowait((self.agent_id, stamped))",
        tests=CHANNEL_TESTS,
    ),
    # ── termination: run-local state, and a judge read literally ────────────
    Mutant(
        tag="termination",
        why="the coordinator shares one condition, so concurrent runs eat each other's turns",
        path="agentkit/agents/policies/roundrobin.py",
        before="    return copy.deepcopy(getattr(cognition, \"termination\", None) or fallback)",
        after="    return getattr(cognition, \"termination\", None) or fallback",
        tests=TERM_TESTS,
    ),
    Mutant(
        tag="termination",
        why="the selector policy keeps its own uncloned resolution",
        path="agentkit/agents/policies/selector_policy.py",
        before="        termination: TerminationCondition = _run_local_termination(\n            cognition, MaxTurns(self.max_turns)\n        )",
        after="        termination: TerminationCondition = getattr(cognition, \"termination\", None) or MaxTurns(self.max_turns)",
        tests=TERM_TESTS,
    ),
    Mutant(
        tag="termination",
        why="cloning the external switch, so set() cannot reach a running loop",
        path="agentkit/agents/control/termination.py",
        before="    def __deepcopy__(self, memo: dict[int, Any]) -> ExternalTermination:\n        memo[id(self)] = self\n        return self\n",
        after="",
        tests=TERM_TESTS,
    ),
    Mutant(
        tag="termination",
        why="the deepcopy carve-out leaks to every condition, undoing run-locality",
        path="agentkit/agents/control/termination.py",
        before="class ExternalTermination(TerminationCondition):",
        after="class ExternalTermination(TerminationCondition):\n    pass\n\n\nclass _Unused(TerminationCondition):",
        tests=TERM_TESTS,
    ),
    Mutant(
        tag="termination",
        why="the judge's YES is matched anywhere, so 'yesterday' stops the run",
        path="agentkit/agents/control/termination.py",
        before='        return re.match(rf"\\W*{re.escape(yes)}(?!\\w)", out, re.IGNORECASE) is not None',
        after="        return yes.upper() in out.upper()",
        tests=TERM_TESTS,
    ),
    Mutant(
        tag="termination",
        why="the judge's affirmative need not be a whole word ('yes/no' counts)",
        path="agentkit/agents/control/termination.py",
        before='        return re.match(rf"\\W*{re.escape(yes)}(?!\\w)", out, re.IGNORECASE) is not None',
        after='        return re.search(rf"(?<!\\w){re.escape(yes)}(?!\\w)", out, re.IGNORECASE) is not None',
        tests=TERM_TESTS,
    ),
    Mutant(
        tag="termination",
        why="a latched Stop is mutable, so a consumer can rewrite why the run stopped",
        path="agentkit/agents/control/termination.py",
        before="@dataclass(frozen=True)\nclass Stop:",
        after="@dataclass\nclass Stop:",
        tests=TERM_TESTS,
    ),
    # ── routing: who speaks next is who the model named ─────────────────────
    Mutant(
        tag="routing",
        why="the roster is scanned in its own order, so a negated name wins",
        path="agentkit/agents/policies/selector_policy.py",
        before="        if best is None or cand[:2] > best[:2]:",
        after="        if best is None:",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="substring matching, so 'bob' is found inside 'nobody' and 'bobby'",
        path="agentkit/agents/policies/selector_policy.py",
        before='        for m in re.finditer(rf"(?<!\\w){re.escape(n)}(?!\\w)", reply):',
        after="        for m in re.finditer(re.escape(n), reply):",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="the FIRST mention wins, so reasoning-aloud beats the conclusion",
        path="agentkit/agents/policies/selector_policy.py",
        before="            last = m.start()",
        after="            last = m.start() if last is None else last",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="an exact reply loses to a longer roster entry containing it",
        path="agentkit/agents/policies/selector_policy.py",
        before="    stripped = reply.strip()\n    for n in names:\n        if stripped == n:\n            return n\n",
        after="",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="an invented handoff target is returned verbatim, ignoring the default",
        path="agentkit/agents/control/handoff.py",
        before="        resolved = _resolve_target(ho.target, agents)",
        after="        resolved = ho.target",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="an invented target falls back in silence",
        path="agentkit/agents/control/handoff.py",
        before="        if ho.target not in warned:",
        after="        if False:",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="the same hallucination warns once per turn",
        path="agentkit/agents/control/handoff.py",
        before="            warned.add(ho.target)",
        after="            pass",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="an ambiguous case-fold is GUESSED instead of deferring to the default",
        path="agentkit/agents/control/handoff.py",
        before="    return folded[0] if len(folded) == 1 else None",
        after="    return folded[0] if folded else None",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="an empty roster rejects every target, breaking direct callers",
        path="agentkit/agents/control/handoff.py",
        before="    if not names:\n        return target",
        after="    if not names:\n        return None",
        tests=ROUTING_TESTS,
    ),
    Mutant(
        tag="routing",
        why="trailing punctuation stays on the target, matching no roster entry",
        path="agentkit/agents/control/handoff.py",
        before='    target = parts[0].rstrip(".,;:!?)\\"\'")',
        after="    target = parts[0]",
        tests=ROUTING_TESTS,
    ),
    # ── the plan's human gate must be durable on a REAL store ───────────────
    Mutant(
        tag="plangate",
        why="the checkpoint holds live dataclasses, so any serializing store raises",
        path="agentkit/agents/policies/plan.py",
        before="                        _encode_plan_state(",
        after="                        dict(",
        tests=PLAN_GATE_TESTS,
    ),
    Mutant(
        tag="plangate",
        why="a gate with no durable seam suspends in silence",
        path="agentkit/agents/policies/plan.py",
        before="                    _warn_unpersisted_gate(gate_name, run_id)",
        after="                    pass",
        tests=PLAN_GATE_TESTS,
    ),
    Mutant(
        tag="plangate",
        why="the plan writes at the bare run id, colliding with a nested coordinator",
        path="agentkit/agents/policies/plan.py",
        before='    return f"{run_id}:plan"',
        after="    return run_id",
        tests=PLAN_GATE_TESTS,
    ),
    Mutant(
        tag="plangate",
        why="the snapshot is RUNNING, so resume() reads it as engine-in-motion",
        path="agentkit/agents/policies/plan.py",
        before="                        status=CheckpointStatus.SUSPENDED,",
        after="                        status=CheckpointStatus.RUNNING,",
        tests=PLAN_GATE_TESTS,
    ),
    Mutant(
        tag="plangate",
        why="a pre-upgrade checkpoint is decoded as if it were encoded",
        path="agentkit/agents/policies/plan.py",
        before='    if state.get("v") is None:  # legacy: live objects, no encoding',
        after="    if False:  # legacy: live objects, no encoding",
        tests=PLAN_GATE_TESTS,
    ),
    Mutant(
        tag="plangate",
        why="an unserialisable child payload takes the whole run down at the gate",
        path="agentkit/agents/policies/plan.py",
        before="    if _serializable(payload):\n        return payload",
        after="    return payload\n    if _serializable(payload):",
        tests=PLAN_GATE_TESTS,
    ),
    Mutant(
        tag="plangate",
        why="results are stripped unconditionally, losing evals nobody asked to drop",
        path="agentkit/agents/policies/plan.py",
        before="    if _serializable(payload):\n        return payload\n",
        after="",
        tests=PLAN_GATE_TESTS,
    ),
    Mutant(
        tag="plangate",
        why="resume ignores the legacy key, stranding a plan suspended across the upgrade",
        path="agentkit/agents/policies/plan.py",
        before="        if not saved and ctx.store is not None:",
        after="        if False and ctx.store is not None:",
        tests=PLAN_GATE_TESTS,
    ),
    # ── plan shape: refuse before dispatch, never mid-flight ─────────────────
    Mutant(
        tag="plan",
        why="an unknown child is only discovered mid-dispatch, after earlier groups spent",
        path="agentkit/agents/policies/plan.py",
        before="        steps, dropped = _validate_plan(steps, children, best_effort=self.best_effort)\n\n        total = len(steps)",
        after="        dropped: list[tuple[str | None, Failure]] = []\n\n        total = len(steps)",
        tests=PLAN_TESTS,
    ),
    Mutant(
        tag="plan",
        why="best_effort refuses an unknown child instead of isolating it",
        path="agentkit/agents/policies/plan.py",
        before="            if not best_effort:",
        after="            if True:",
        tests=PLAN_TESTS,
    ),
    Mutant(
        tag="plan",
        why="a gate co-grouped with work is accepted, silently dropping those steps",
        path="agentkit/agents/policies/plan.py",
        before="        if gates and work:",
        after="        if gates and work and False:",
        tests=PLAN_TESTS,
    ),
    Mutant(
        tag="plan",
        why="resume trusts the pre-suspend roster, so a shrunken one KeyErrors mid-flight",
        path="agentkit/agents/policies/plan.py",
        before="        steps, dropped = _validate_plan(steps, children, best_effort=self.best_effort)\n        errors.extend(dropped)",
        after="        dropped = []",
        tests=PLAN_TESTS,
    ),
    Mutant(
        tag="plan",
        why="an unknown child is dropped SILENTLY instead of recorded as a failure",
        path="agentkit/agents/policies/plan.py",
        before="            results=[],\n            errors=list(dropped),",
        after="            results=[],\n            errors=[],",
        tests=PLAN_TESTS,
    ),
    Mutant(
        tag="plan",
        why="the validator sorts its output, silently rewriting the plan's order",
        path="agentkit/agents/policies/plan.py",
        before="    return keep, dropped",
        after="    return sorted(keep, key=lambda s: s.group), dropped",
        tests=PLAN_TESTS,
    ),
    Mutant(
        tag="plan",
        why="an unknown child is reported as retriable, so a caller re-dispatches forever",
        path="agentkit/agents/policies/plan.py",
        before="                    Failure(category=ErrorClass.PERMANENT, source=\"PlanPolicy\", message=msg),",
        after="                    Failure(category=ErrorClass.TRANSIENT, source=\"PlanPolicy\", message=msg),",
        tests=PLAN_TESTS,
    ),
    # ── stop reasons: the typed field must not outlive its meaning ───────────
    Mutant(
        tag="stopreason",
        why="a plan parked on a human gate reports itself complete (the original bug)",
        path="agentkit/agents/policies/plan.py",
        before='                    stop_reason=stop_reason_for("awaiting_decision"),\n',
        after="",
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="a gate suspend is categorised as terminal, so nothing ever resumes it",
        path="agentkit/agents/result.py",
        before='    "awaiting_decision": "suspended",',
        after='    "awaiting_decision": "terminated",',
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="a coordinator out of turns is indistinguishable from one that finished",
        path="agentkit/agents/policies/roundrobin.py",
        before="        stop_reason=stop_reason_for(stop_reason),\n",
        after="",
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="a ledger out of rounds claims the goal was met",
        path="agentkit/agents/policies/ledger.py",
        before="            stop_reason=stop_reason_for(stop_reason),\n",
        after="",
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="an unrecognised reason is GUESSED as completion instead of terminated",
        path="agentkit/agents/result.py",
        before='    return _REASON_TO_STOP.get(reason, "terminated")',
        after='    return _REASON_TO_STOP.get(reason, "complete")',
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="a CLI subprocess that exited non-zero reads as a deliberate stop",
        path="agentkit/agents/cognition/claude_cli.py",
        before='    if reason in _CLI_FAILURE_REASONS or (reason is not None and reason.startswith("cli_exit_")):',
        after="    if reason in _CLI_FAILURE_REASONS:",
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="the typed reason is dropped on the durable round trip",
        path="agentkit/capabilities/checkpointer/persistence.py",
        before='        "stop_reason": r.stop_reason,\n',
        after="",
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="a legacy record without the field reads back as a bare completion",
        path="agentkit/capabilities/checkpointer/persistence.py",
        before='        stop_reason=d.get("stop_reason") or stop_reason_for((d.get("evals") or {}).get("stop_reason")),',
        after='        stop_reason=d.get("stop_reason", "complete"),',
        tests=STOP_REASON_TESTS,
    ),
    # ── money: the ledger must stay exact ────────────────────────────────────
    Mutant(
        tag="money",
        why="the exact ledger accessor silently degrades to float",
        path="agentkit/runtime/meter.py",
        # Trailing newline matters: without it this is a PREFIX of
        # ``return self._spent.quantize(...)`` two methods down, and the
        # anchor silently matches twice. `--verify` catches exactly this.
        before="        return self._spent\n",
        after="        return float(self._spent)\n",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="accumulation round-trips through float, losing precision at scale",
        path="agentkit/runtime/meter.py",
        before="            self._spent += to_money(usage.cost_usd)",
        after="            self._spent = to_money(float(self._spent) + usage.cost_usd)",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="to_money reads the BINARY float instead of its decimal spelling",
        path="agentkit/runtime/meter.py",
        before="        amount = Decimal(str(value))",
        after="        amount = Decimal(value)",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="quantizing per charge instead of at read rounds sub-cent calls to zero",
        path="agentkit/runtime/meter.py",
        before="            self._spent += to_money(usage.cost_usd)",
        after="            self._spent += to_money(usage.cost_usd).quantize(_CENTS)",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="invoicing truncates instead of rounding half-up (systematic under-billing)",
        path="agentkit/runtime/meter.py",
        before="        return self._spent.quantize(_CENTS, rounding=ROUND_HALF_UP)",
        after="        return self._spent.quantize(_CENTS, rounding='ROUND_DOWN')",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="guard() stops enforcing, so an exhausted budget still buys a call",
        path="agentkit/runtime/meter.py",
        before="        async with self._get_lock():\n            verdict = self._verdict()",
        after="        async with self._get_lock():\n            verdict = Charge(ok=True)",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="token totals stop accumulating, so a run's usage under-reports",
        path="agentkit/runtime/meter.py",
        before="            self.usage = self.usage + usage",
        after="            pass",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="the ceiling stops re-deriving, so raising it post-construction is ignored",
        path="agentkit/runtime/meter.py",
        before="        if self.max_cost_usd != self._ceiling_src:",
        after="        if False:",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="the float mirror stops tracking the ledger and silently goes stale",
        path="agentkit/runtime/meter.py",
        before="        self.spent_usd = float(self._spent)",
        after="        pass",
        tests=MONEY_TESTS,
    ),
    # ── budget recovery: exhaustion must stay recoverable ────────────────────
    Mutant(
        tag="budget",
        why="the tool loop stops pre-flighting, so each retry burns another call",
        path="agentkit/agents/cognition/react.py",
        before="            ceiling = self._budget_exhausted(ctx)\n            if ceiling is not None:",
        after="            ceiling = self._budget_exhausted(ctx)\n            if False:",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="budget",
        why="the budget-exhausted checkpoint is written as running, not suspended",
        path="agentkit/agents/cognition/react.py",
        before='                        ctx, run_id, agent, context, usage, i + 1, repaired, status="suspended"',
        after="                        ctx, run_id, agent, context, usage, i + 1, repaired",
        tests=MONEY_TESTS,
    ),
    # ── concurrency: nested fan-out must not deadlock ────────────────────────
    Mutant(
        tag="concurrency",
        why="the permit pool goes back to one-per-tree, deadlocking nested fan-out",
        path="agentkit/runtime/context.py",
        before="        return self.budget.semaphore(self.depth)",
        after="        return self.budget.semaphore(0)",
        tests=CONCURRENCY_TESTS,
    ),
    Mutant(
        tag="concurrency",
        why="the per-level concurrency cap stops being enforced",
        path="agentkit/runtime/meter.py",
        before="            sem = asyncio.Semaphore(self.max_concurrency)",
        after="            sem = asyncio.Semaphore(1000)",
        tests=CONCURRENCY_TESTS,
    ),
    Mutant(
        tag="concurrency",
        why="cooperative cancellation is isolated into a Failure slot again",
        path="agentkit/kernel/concurrency.py",
        before="            except Cancelled:\n",
        after="            except _NeverRaised:\n",
        tests=CONCURRENCY_TESTS,
    ),
    Mutant(
        tag="concurrency",
        why="a starved slice floors to zero, producing silent no-op children",
        path="agentkit/kernel/concurrency.py",
        before="    return max(1, units)",
        after="    return units",
        tests=("tests/kernel/test_run_agents_actor_slicing.py",),
    ),
    Mutant(
        tag="concurrency",
        why="a fan-out from an already-spent envelope stops failing fast",
        path="agentkit/kernel/concurrency.py",
        before="        if parent_actor_budget.exhausted():",
        after="        if False:",
        tests=("tests/kernel/test_run_agents_actor_slicing.py",),
    ),
    Mutant(
        tag="concurrency",
        why="the money slice round-trips through the float mirror again",
        path="agentkit/kernel/concurrency.py",
        before="        base_cost = parent_actor_budget.remaining_cost()  # exact, not the float mirror",
        after="        base_cost = to_money(parent_actor_budget.remaining_cost_usd())",
        tests=("tests/kernel/test_run_agents_actor_slicing.py",),
    ),
    Mutant(
        tag="concurrency",
        why="a child is over-granted a step the parent never reserved",
        path="agentkit/kernel/concurrency.py",
        before="                    max_steps=slice_steps,",
        after="                    max_steps=max(slice_steps, 1) + 1,",
        tests=("tests/kernel/test_run_agents_actor_slicing.py",),
    ),
    Mutant(
        tag="concurrency",
        why="reservations leak when a fan-out fails (the finally settlement)",
        path="agentkit/kernel/concurrency.py",
        before="        if parent_actor_budget is not None:\n            for idx, slice_ in enumerate(reserved_slices):",
        after="        if False:\n            for idx, slice_ in enumerate(reserved_slices):",
        tests=("tests/kernel/test_run_agents_actor_slicing.py",),
    ),
    # ── budget: the sibling float axes ───────────────────────────────────────
    # NOTE: mutating ``remaining_cost() <= 0`` to ``remaining_cost_usd() == 0.0``
    # was tried and is an EQUIVALENT mutant — the exact Decimal ledger leaves no
    # residue, so both spellings agree for every input. That is the point of the
    # conversion, and a mutant that cannot fail is noise in the report. The
    # load-bearing invariant here is that the float MIRRORS track the ledger.
    Mutant(
        tag="budget",
        why="ActorBudget float mirrors stop tracking the Decimal ledger",
        path="agentkit/agents/control/budget.py",
        before="        self.used_cost_usd = float(self._used_cost)",
        after="        pass",
        tests=("tests/runtime/test_budget_decimal_and_verdict.py",),
    ),
    Mutant(
        tag="budget",
        why="Quota stops evicting expired tenants (unbounded key growth)",
        path="agentkit/runtime/meter.py",
        before="            self._sweep(now)  # evict long-dead tenants; at most once per window",
        after="            pass  # no sweep",
        tests=("tests/runtime/test_budget_decimal_and_verdict.py",),
    ),
    Mutant(
        tag="budget",
        why="the per-actor envelope stops being charged (an inert ActorBudget again)",
        path="agentkit/middlewares/meter.py",
        before="            actor = getattr(ctx.run, \"actor_budget\", None)",
        after="            actor = None",
        tests=("tests/agents/test_actor_budget.py",),
    ),
    Mutant(
        tag="budget",
        why="no loop consults the actor envelope, so exhausting it stops nothing",
        path="agentkit/agents/_agent_helpers.py",
        before="    actor = getattr(ctx, \"actor_budget\", None)",
        after="    actor = None",
        tests=("tests/agents/test_actor_budget.py",),
    ),
    Mutant(
        tag="budget",
        why="ActorBudget cost accumulates through float, losing exactness",
        path="agentkit/agents/control/budget.py",
        before="        self._used_cost += to_money(cost_usd)",
        after="        self._used_cost = to_money(float(self._used_cost) + cost_usd)",
        tests=("tests/runtime/test_budget_decimal_and_verdict.py",),
    ),
    # ── isolation + resilience ───────────────────────────────────────────────
    Mutant(
        tag="security",
        why="cache keys stop being tenant-partitioned (cross-tenant answer leak)",
        path="agentkit/middlewares/memoize.py",
        before="        cache_key = _scoped(call, key(call))",
        after="        cache_key = key(call)",
        tests=SECURITY_TESTS,
    ),
    Mutant(
        tag="security",
        why="the default cache key ignores the model (a cheap answer served as a good one)",
        path="agentkit/middlewares/memoize.py",
        before='            "model": getattr(r, "model", None),',
        after='            "model": None,',
        tests=SECURITY_TESTS,
    ),
    Mutant(
        tag="security",
        why="egress can be constructed inert, silently disabling every URL check",
        path="agentkit/middlewares/security.py",
        before="        if guardrail is None:",
        after="        if False:",
        tests=SECURITY_TESTS,
    ),
    Mutant(
        tag="resilience",
        why="a late success closes an OPEN breaker, bypassing the cooldown",
        path="agentkit/kernel/resilience.py",
        before='        if self.state == "open":\n            return',
        after='        if False:\n            return',
        tests=RESILIENCE_TESTS,
    ),
    Mutant(
        tag="provider",
        why="an in-band SSE error frame is swallowed, truncating the answer silently",
        path="agentkit/adapters/llm/providers/base.py",
        before="    err = event.get(\"error\")",
        after="    return\n    err = event.get(\"error\")",
        tests=PROVIDER_TESTS,
    ),
    Mutant(
        tag="provider",
        why="Anthropic stops checking for the error frame",
        path="agentkit/adapters/llm/providers/anthropic.py",
        before="            raise_if_error_frame(ev)",
        after="            pass",
        tests=PROVIDER_TESTS,
    ),
    Mutant(
        tag="provider",
        why="an overload frame stops classifying as retryable",
        path="agentkit/kernel/resilience.py",
        before='    "rate_limit",\n    "server_error",\n    "overloaded",',
        after='    "__never_matches_rate_limit",\n    "__never_matches_server_error",\n    "__never_matches_overloaded",',
        tests=PROVIDER_TESTS,
    ),
    # ── checkpoint slots: one producer per slot ──────────────────────────────
    Mutant(
        tag="checkpoint",
        why="producers share a slot again (a child's completion deletes the coordinator's state)",
        path="agentkit/agents/cognition/react.py",
        before='        return f"{run_id}:agent:{agent_name}"',
        after="        return run_id",
        tests=SLOT_TESTS,
    ),
    Mutant(
        tag="checkpoint",
        why="the legacy read accepts another producer's payload",
        path="agentkit/agents/cognition/react.py",
        before='        return legacy if "messages" in state else None',
        after="        return legacy",
        tests=SLOT_TESTS,
    ),
    Mutant(
        tag="checkpoint",
        why="coordinator policies drop back to a third resolution order (no store bridge)",
        path="agentkit/agents/policies/roundrobin.py",
        before='    return resolve_checkpointer(ctx, getattr(coordinator.cognition, "checkpointer", None))',
        after='    return getattr(coordinator.cognition, "checkpointer", None) or getattr(ctx, "checkpointer", None)',
        tests=SLOT_TESTS,
    ),
    # ── stores: the reference contract every backend must match ──────────────
    Mutant(
        tag="store",
        why="FileStore.get_or_set keys on truthiness again (single-flight breaks on None)",
        path="agentkit/adapters/store/file.py",
        # Targets `_exists` itself, not one of its two call sites: mutating
        # only the pre-lock check leaves the in-lock double-check correct, and
        # the mutant survives for the wrong reason.
        before="        return await asyncio.to_thread(path.exists)",
        after="        return (await self.get(key)) is not None",
        tests=STORE_TESTS,
    ),
    Mutant(
        tag="store",
        why="FileStore writes non-atomically again (a crash mid-write orphans the checkpoint)",
        path="agentkit/adapters/store/file.py",
        before="        await asyncio.to_thread(self._write_atomic, path, json.dumps(value))",
        after="        await asyncio.to_thread(lambda: path.write_text(json.dumps(value)))",
        tests=STORE_TESTS,
    ),
    Mutant(
        tag="store",
        why="one unparseable log line takes the whole audit trail down again",
        path="agentkit/adapters/store/file.py",
        before="                except json.JSONDecodeError:",
        after="                except _NeverJSONError:",
        tests=STORE_TESTS,
    ),
    Mutant(
        tag="store",
        why="a silently-ignored ttl stops warning (permanent idempotency entries)",
        path="agentkit/adapters/store/file.py",
        before="        if ttl is not None and not self._warned_ttl:",
        after="        if False:",
        tests=STORE_TESTS,
    ),
    # ── workflow ─────────────────────────────────────────────────────────────
    Mutant(
        tag="workflow",
        why="an unpersistable gate suspend goes silent again",
        path="agentkit/agents/workflow.py",
        before="                        _warn_unpersisted_gate(gate.name, run_id)",
        after="                        pass",
        tests=WORKFLOW_TESTS,
    ),
    Mutant(
        tag="workflow",
        why="Workflow goes back to store-only persistence, ignoring a Checkpointer",
        path="agentkit/agents/workflow.py",
        before="                    cp = resolve_checkpointer(ctx)",
        after="                    cp = None",
        tests=WORKFLOW_TESTS,
    ),
    # ── HITL: the containment and the deadline ───────────────────────────────
    Mutant(
        tag="hitl",
        why="a secret-tainted state is persisted after all",
        path="agentkit/capabilities/checkpointer/base.py",
        before="        if _is_tainted(state):",
        after="        if False:",
        tests=HITL_TESTS,
    ),
    Mutant(
        tag="hitl",
        why="SecretValue renders its contents in repr, leaking into every log line",
        path="agentkit/agents/control/elicitation.py",
        before='        return "SecretValue(\'***\')"',
        after="        return f\"SecretValue({self._value!r})\"",
        tests=HITL_TESTS,
    ),
    Mutant(
        tag="hitl",
        why="an expired elicitation is treated as an approval",
        path="agentkit/agents/control/elicitation.py",
        before='        return self.kind in ("approve", "modify")',
        after='        return self.kind in ("approve", "modify", "expired")',
        tests=HITL_TESTS,
    ),
    Mutant(
        tag="hitl",
        why="the deadline is dropped, so a wait becomes unbounded",
        path="agentkit/agents/control/elicitation.py",
        before="        if request.deadline_s is None:",
        after="        if True:",
        tests=HITL_TESTS,
    ),
    Mutant(
        tag="hitl",
        why="a late resume acts on the decision instead of expiring it",
        path="agentkit/agents/cognition/react.py",
        before="        expired = deadline_at is not None and time.time() > deadline_at",
        after="        expired = False",
        tests=HITL_TESTS,
    ),
    Mutant(
        tag="hitl",
        why="the RunPolicy trifecta gate is skipped on the resume path",
        path="agentkit/agents/agent.py",
        before="            await self._run_policy_gate(ctx)\n            final = await cognition.resume(",
        after="            final = await cognition.resume(",
        tests=HITL_TESTS,
    ),
    Mutant(
        tag="hitl",
        why="a missing decision defaults to approve instead of deny",
        path="agentkit/agents/cognition/react.py",
        before='coerce_decision(decisions.get(tc.id, "reject"))',
        after='coerce_decision(decisions.get(tc.id, "approve"))',
        tests=HITL_TESTS,
    ),
    # ── streaming: the partial contract ──────────────────────────────────────
    Mutant(
        tag="stream",
        why="the in-progress typed object is dropped again",
        path="agentkit/agents/cognition/single_call.py",
        before='yield StreamEvent("message_delta", text=d.text, partial_output=d.partial)',
        after='yield StreamEvent("message_delta", text=d.text)',
        tests=STREAM_TESTS,
    ),
    Mutant(
        tag="stream",
        why="a coercion failure escapes past reflect-and-repair again",
        path="agentkit/agents/cognition/single_call.py",
        before="                if agent.parse is None:\n                    raise",
        after="                raise",
        tests=STREAM_TESTS,
    ),
    # ── registry: nothing is ever guessed ────────────────────────────────────
    Mutant(
        tag="registry",
        why="an unknown model reports capabilities as present instead of UNKNOWN",
        path="agentkit/adapters/llm/model_registry.py",
        before="        return entry.capabilities if entry is not None else ModelCapabilities()",
        after="        return entry.capabilities if entry is not None else ModelCapabilities(\n"
        "            tools=_Y, structured_output=_Y, native_json_schema=_Y, streaming=_Y, vision=_Y\n"
        "        )",
        tests=REGISTRY_TESTS,
    ),
    Mutant(
        tag="registry",
        why="a missing credential silently returns the fake instead of raising",
        path="agentkit/adapters/llm/model_registry.py",
        before="        if fallback is None:\n            raise hard",
        after="        if False:\n            raise hard",
        tests=REGISTRY_TESTS,
    ),
    Mutant(
        tag="registry",
        why="a declared NO capability stops refusing at bind time",
        path="agentkit/adapters/llm/model_registry.py",
        before="        missing = [n for n in (*requires, *derived) if caps.get(n) is Capability.NO]",
        after="        missing = []",
        tests=REGISTRY_TESTS,
    ),
    Mutant(
        tag="registry",
        why="the downgrade warning fires every time instead of once",
        path="agentkit/adapters/llm/model_registry.py",
        before="        if key in self._warned:\n            return",
        after="        if False:\n            return",
        tests=REGISTRY_TESTS,
    ),
)


# ── runner ───────────────────────────────────────────────────────────────────


# A mutant can make the suite HANG rather than fail — removing a timeout is a
# perfectly good mutation, and the tests that exercise it then wait forever.
# A hang is a finding, not a reason to block the run.
_PER_MUTANT_TIMEOUT_S = 120


def _run_tests(tests: tuple[str, ...]) -> bool | None:
    """``False`` = tests failed (mutant killed), ``True`` = tests passed
    (survived), ``None`` = the run timed out."""
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", *tests,
                "-q", "-x", "--no-header", "-p", "no:cacheprovider",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=_PER_MUTANT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    return proc.returncode == 0


def verify_anchors() -> list[str]:
    """Every mutant's ``before`` anchor must still be present exactly once.

    This is the integrity check AND the staleness check in one. A missing
    anchor means either a refactor moved the code (the catalogue entry needs
    updating) or a mutant is still applied. Checking the ``after`` text
    instead would be unreliable: several replacements are substrings of
    legitimate code elsewhere in the same file.
    """
    problems = []
    for m in MUTANTS:
        target = ROOT / m.path
        if not target.exists():
            problems.append(f"{m.path}: file does not exist (renamed or removed?)")
            continue
        count = target.read_text().count(m.before)
        if count != 1:
            problems.append(f"{m.path}: anchor for {m.why!r} appears {count}x, expected 1")
    return problems


def _dirty(paths: set[str]) -> list[str]:
    """Files with uncommitted changes, so a crash can't destroy real work."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", *sorted(paths)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return [line[3:] for line in proc.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", "--tag", help="only mutants with this tag")
    parser.add_argument("--list", action="store_true", help="list the catalogue and exit")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every anchor resolves (and nothing is left mutated), then exit",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run even with uncommitted changes to the mutated files (unsafe)",
    )
    args = parser.parse_args()

    _recover_from_crash()

    if args.verify:
        problems = verify_anchors()
        for p in problems:
            print(f"  {p}")
        print("catalogue is stale" if problems else "  all anchors resolve; nothing left mutated")
        return 1 if problems else 0

    selected = [m for m in MUTANTS if not args.tag or m.tag == args.tag]
    if args.list:
        for m in selected:
            print(f"[{m.tag}] {m.why}\n      {m.path}")
        return 0
    if not selected:
        print(f"no mutants tagged {args.tag!r}; tags: {sorted({m.tag for m in MUTANTS})}")
        return 1

    # Refuse on a dirty tree. This script edits source in place and restores in
    # a `finally`, but a hard kill (Ctrl-C twice, OOM) can still leave a mutant
    # behind. Requiring a clean tree means `git checkout` always recovers.
    targets = {m.path for m in selected}
    if not args.allow_dirty and (dirty := _dirty(targets)):
        print("refusing to run: uncommitted changes in files this script rewrites:")
        for f in dirty:
            print(f"  {f}")
        print("\ncommit or stash first, or pass --allow-dirty if you accept the risk.")
        return 2

    # Snapshot every file we are about to touch, so a hard kill is recoverable.
    _BACKUP_DIR.mkdir(exist_ok=True)
    (_BACKUP_DIR / "MANIFEST").write_text("\n".join(sorted(targets)) + "\n")
    for rel in targets:
        source = ROOT / rel
        if source.exists():
            _backup_path(rel).write_text(source.read_text())

    print(f"running {len(selected)} mutants\n", flush=True)
    survivors: list[Mutant] = []
    try:
        for i, m in enumerate(selected, 1):
            target = ROOT / m.path
            if not target.exists():
                print(f"  {i:>2}. [{m.tag}] MISSING FILE — {m.path}", flush=True)
                survivors.append(m)
                continue
            original = target.read_text()
            if original.count(m.before) != 1:
                print(f"  {i:>2}. [{m.tag}] STALE ANCHOR — {m.why}", flush=True)
                print(
                    f"      {m.path}: anchor appears {original.count(m.before)}x, expected once",
                    flush=True,
                )
                survivors.append(m)
                continue
            try:
                target.write_text(original.replace(m.before, m.after, 1))
                survived = _run_tests(m.tests)
            finally:
                target.write_text(original)
            mark = {True: "SURVIVED", False: "killed  ", None: "TIMEOUT "}[survived]
            print(f"  {i:>2}. [{m.tag}] {mark}  {m.why}", flush=True)
            if survived is not False:
                survivors.append(m)
    finally:
        # Clean exit: drop the crash-recovery snapshot. Anything that skips
        # this leaves the snapshot behind for the next run to restore from.
        for rel in targets:
            _backup_path(rel).unlink(missing_ok=True)
        (_BACKUP_DIR / "MANIFEST").unlink(missing_ok=True)
        _BACKUP_DIR.rmdir()

    print()
    if survivors:
        print(f"{len(survivors)} of {len(selected)} mutants SURVIVED — the suite does not")
        print("enforce these invariants. Sharpen the tests, or drop the invariant:\n")
        for m in survivors:
            print(f"  [{m.tag}] {m.why}")
            print(f"        {m.path}  ->  tests: {', '.join(m.tests)}")
        return 1
    print(f"all {len(selected)} mutants killed — the suite enforces every catalogued invariant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
