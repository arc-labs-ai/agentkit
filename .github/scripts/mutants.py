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
import shutil
import subprocess
import sys

# The repo root, found by walking up to the directory holding pyproject.toml
# rather than counting parents. This file lives under `.github/scripts/`, and
# a hardcoded parent count silently resolves to `.github/` if it ever moves
# again — which would make every anchor "not found" and the catalogue look
# entirely stale rather than mislocated.
def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"no pyproject.toml above {here}")


ROOT = _repo_root()

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
    # The backup FILENAME is a LOSSY encoding of the path: `_backup_path` maps
    # "/" to "__", so `agentkit/core/policies/__init__.py` is stored as
    # `agentkit__middlewares____init__.py`, and decoding "__" back to "/" gives
    # `agentkit/core/policies//init/.py` — a directory that does not exist. Every
    # `__init__.py` and `__main__.py` hits it. Recovery then died on the write
    # with `FileNotFoundError`, which is the worst possible moment to fail:
    # every file after it in the sort order stayed MUTATED, and because the
    # crash happened before the backup directory was cleared, the NEXT
    # invocation crashed in the same place. The tool bricked itself and left
    # the working tree silently modified (observed: a neutralised guard left
    # behind in `claude_cli.py`).
    #
    # The MANIFEST already records the real paths, so decode through it rather
    # than trying to invert an encoding that has no inverse.
    manifest = _BACKUP_DIR / "MANIFEST"
    by_encoded = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            rel = line.strip()
            if rel:
                by_encoded[_backup_path(rel).name] = rel

    restored, orphans = [], []
    for backup in sorted(_BACKUP_DIR.iterdir()):
        if backup.name == "MANIFEST":
            continue
        rel = by_encoded.get(backup.name)
        if rel is None:
            # Never guess: writing a backup to a path decoded by hand is how
            # the original bug turned a crash into a corrupted tree.
            orphans.append(backup.name)
            continue
        (ROOT / rel).write_text(backup.read_text())
        restored.append(rel)
        backup.unlink()
    if orphans:
        print("could NOT map these backups to a path (restore them by hand):")
        for name in orphans:
            print(f"  {_BACKUP_DIR.name}/{name}")
        return True
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
    "tests/source/unit/core/runtime/test_budget_decimal_and_verdict.py",
    "tests/source/architecture/test_protocol_conformance.py",
)
HITL_TESTS = ("tests/source/unit/agents/test_hitl_elicitation.py",)
STREAM_TESTS = ("tests/source/unit/agents/test_stream_partial_output.py",)
CONCURRENCY_TESTS = ("tests/source/unit/core/test_nested_concurrency.py",)
SECURITY_TESTS = (
    "tests/source/unit/core/policies/test_memoize_scope_isolation.py",
    "tests/source/unit/integrations/observability/test_observability_cache.py",
)
RESILIENCE_TESTS = ("tests/source/unit/core/test_errors.py",)
PROVIDER_TESTS = ("tests/source/unit/models/adapters/test_provider_stream_errors.py",)
SLOT_TESTS = ("tests/source/unit/state/checkpoints/test_checkpoint_slot_isolation.py",)
STOP_REASON_TESTS = ("tests/source/unit/agents/test_stop_reason_taxonomy.py",)
PLAN_TESTS = (
    "tests/source/unit/agents/test_plan_validation.py",
    "tests/source/unit/agents/test_plan_policy.py",
)
AUDIT2_TESTS = ("tests/source/unit/core/policies/test_audit_records_failures.py",)
STORELOCK_TESTS = ("tests/source/unit/state/adapters/test_stores.py",)
EST_TESTS = (
    "tests/source/unit/state/context/test_context_tokens.py",
)
OBS_TESTS = ("tests/source/unit/integrations/adapters/test_observer_adapters.py",)
STORE2_TESTS = ("tests/source/unit/state/adapters/test_stores.py",)
VEC_TESTS = ("tests/source/unit/state/adapters/test_vector_adapters.py",)
PROV_TESTS = ("tests/source/unit/models/adapters/test_provider_error_classification.py",)
SSE_TESTS = ("tests/source/unit/models/adapters/test_provider_sse_and_tool_calls.py",)
MAP_TESTS = ("tests/source/unit/models/adapters/test_callable_llm_mapping.py",)
TOK_TESTS = (
    "tests/source/unit/state/context/test_context_tokens.py",
)
MCPHTTP_TESTS = ("tests/source/unit/integrations/mcp/test_http_transport.py",)
SCHEMA2_TESTS = (
    "tests/source/unit/models/output/test_output_schema_dataclass.py",
    "tests/source/unit/models/output/test_output_schema_attrs.py",
    "tests/source/unit/models/output/test_output_schema_pydantic.py",
)
MW_MEMO_TESTS = ("tests/source/unit/core/policies/test_memoize_tool_identity.py",)
MW_METER_TESTS = ("tests/source/unit/core/policies/test_meter_stream_and_cache.py",)
MW_SEM_TESTS = ("tests/source/unit/core/policies/test_semantic_memoize_tool_turns.py",)
MW_TRACE_TESTS = ("tests/source/unit/core/policies/test_tracing_records_failures.py",)
MW_COMPACT_TESTS = ("tests/source/unit/core/policies/test_compaction_broken_tracer.py",)
COMPACTION_TESTS = ("tests/source/unit/state/compaction/test_compaction_strategies.py",)
TOOLCOERCE_TESTS = ("tests/source/unit/tools/test_tool_argument_coercion.py",)
MW_AUDIT_TESTS = ("tests/source/unit/core/policies/test_audit_records_failures.py",)
STREAMS_TESTS = ("tests/source/unit/core/test_streams.py",)
BREAKER_TESTS = ("tests/source/unit/core/test_kernel.py",)
CONC_TESTS = ("tests/source/unit/core/test_concurrency.py",)
WF_TESTS = ("tests/source/unit/agents/test_workflow.py",)
CEILING_TESTS = ("tests/source/unit/agents/test_ceiling_propagation.py",)
SURFACE_TESTS = ("tests/source/architecture/test_public_surface.py",)
PROMPT_TESTS = ("tests/source/unit/agents/prompts/test_prompt_render.py",)
AGENT_PROMPT_TESTS = ("tests/source/unit/agents/test_agent_prompt_binding.py",)
APPROVAL_TESTS = ("tests/source/unit/integrations/mcp/test_approvals.py",)
CLI_SESSION_TESTS = ("tests/source/unit/models/providers/cli/claude/test_claude_cli_session.py",)
CLI_STREAM_TESTS = ("tests/source/unit/models/providers/cli/claude/test_claude_cli_streaming.py",)
CLI_BUDGET_TESTS = ("tests/source/unit/models/providers/cli/claude/test_claude_cli_budget.py",)
CLI_SCHEMA_TESTS = ("tests/source/unit/models/providers/cli/claude/test_claude_cli_structured.py",)
CLI_FLAG_TESTS = ("tests/source/unit/models/providers/cli/claude/test_claude_cli_flags.py",)
FAKE_CLI_TESTS = ("tests/source/unit/core/testing/test_fake_claude_cli.py",)
CODEX_TESTS = ("tests/source/unit/models/providers/cli/codex/test_codex_cli.py",)
CODEX_EVENT_TESTS = ("tests/source/unit/models/providers/cli/codex/test_codex_cli_events.py",)
CODEX_FLAG_TESTS = ("tests/source/unit/models/providers/cli/codex/test_codex_cli_flags.py",)
CODEX_CAPS_TESTS = ("tests/source/unit/models/providers/cli/codex/test_codex_cli_caps.py",)
CODEX_SESSION_TESTS = ("tests/source/unit/models/providers/cli/codex/test_codex_cli_session.py",)
CODEX_BUDGET_TESTS = ("tests/source/unit/models/providers/cli/codex/test_codex_cli_budget.py",)
CODEX_SCHEMA_TESTS = ("tests/source/unit/models/providers/cli/codex/test_codex_cli_structured.py",)
FAKE_CODEX_TESTS = ("tests/source/unit/core/testing/test_fake_codex_cli.py",)
MEM_DEDUPE_TESTS = (
    "tests/source/unit/state/memory/test_composite_memory_dedupe.py",
    "tests/source/unit/state/memory/test_memory_item.py",
)
MEM_VECTOR_TESTS = ("tests/source/unit/state/memory/test_vector_memory.py",)
WORKFLOW_MAP_TESTS = ("tests/source/unit/agents/test_workflow_map.py",)
RECURRENCE_TESTS = ("tests/source/unit/core/test_attempt_until_stuck.py",)
APPROVAL_AUTONOMY_TESTS = ("tests/source/unit/integrations/mcp/test_approvals.py",)
READONLY_TESTS = ("tests/source/unit/state/memory/test_read_only_memory.py", "tests/source/unit/state/memory/test_composite_memory_dedupe.py")
TOOLDESC_TESTS = (
    "tests/source/unit/tools/test_tool_description_resolution.py",
)
CLIAGENT_TESTS = ("tests/integrations/claude_cli/",)
HOOKS_TESTS = ("tests/source/unit/models/providers/cli/claude/helpers/test_hooks.py",)
APPROVAL_RECORD_TESTS = ("tests/source/unit/integrations/mcp/test_approvals.py",)
EXECUTION_TESTS = (
    "tests/source/conformance/test_execution_capabilities.py",
    "tests/source/conformance/test_execution_regressions.py",
)
MCP_AUTH_TESTS = (
    "tests/source/unit/integrations/mcp/test_transport_auth.py",
    "tests/source/unit/integrations/mcp/test_approvals.py",
)
GROUNDING_TESTS = ("tests/source/unit/agents/requests/test_request_builder_grounding.py",)
STORE_PRIM_TESTS = (
    "tests/source/unit/state/adapters/test_store_primitives.py",
    "tests/source/unit/state/adapters/test_stores.py",
    "tests/source/architecture/test_protocol_conformance.py",
)
FILETOOL_TESTS = ("tests/source/unit/tools/test_memory_tool.py",)
TOOLARG_TESTS = (
    "tests/source/unit/tools/test_tool_call_contract.py",
)
CHANNEL_TESTS = ("tests/source/unit/agents/test_signal_protocol.py",)
TERM_TESTS = (
    "tests/source/unit/agents/test_termination_isolation.py",
    "tests/source/unit/agents/test_termination.py",
)
ROUTING_TESTS = (
    "tests/source/unit/agents/test_routing_accuracy.py",
    "tests/source/unit/agents/test_handoff.py",
)
PLAN_GATE_TESTS = (
    "tests/source/unit/agents/test_plan_durable_gate.py",
    "tests/source/unit/agents/test_plan_policy.py",
)
STORE_TESTS = (
    "tests/source/architecture/test_protocol_conformance.py",
    "tests/source/unit/state/adapters/test_stores.py",
)
WORKFLOW_TESTS = ("tests/source/unit/agents/test_workflow.py",)
REGISTRY_TESTS = (
    "tests/source/unit/models/adapters/test_model_registry.py",
    "tests/source/unit/agents/test_agent_capability_binding.py",
)

MEMOIZE_TESTS = (
    "tests/source/unit/core/policies/test_memoize_side_effects.py",
    "tests/source/unit/core/policies/test_memoize_chat_key_tool_fields.py",
)
SIGNAL_TESTS = ("tests/source/unit/agents/test_signal_immutability.py",)
BUS_TESTS = ("tests/source/unit/core/runtime/test_event_bus_close_race.py",)
PARSED_TESTS = (
    "tests/source/unit/core/test_result_to_stream_parity.py",
    "tests/source/unit/core/policies/test_memoize_preserves_parsed.py",
)
MAPPING_TESTS = ("tests/source/unit/models/adapters/test_callable_llm_mapping.py",)
PORTS_TESTS = ("tests/source/unit/core/test_value_type_hashability.py",)
FROZEN_TESTS = ("tests/source/unit/core/test_frozen_payloads.py",)
FROZENVAL_TESTS = ("tests/source/unit/core/test_frozen_value_payloads.py",)
FROZENREQ_TESTS = ("tests/source/unit/core/test_frozen_request_payloads.py",)
KERNELTYPE_TESTS = ("tests/source/unit/core/test_kernel.py",)
RESULT_TESTS = ("tests/source/unit/agents/test_agents.py",)
MEMITEM_TESTS = ("tests/source/unit/state/memory/test_memory_item.py",)
SKILL_TESTS = ("tests/source/unit/agents/skills/test_skill.py",)
CTXDIFF_TESTS = ("tests/source/unit/state/context/test_context_working.py",)
VALUETYPE_TESTS = (
    "tests/source/unit/state/context/test_context_working.py",
)
REPLAY_TESTS = ("tests/source/unit/state/adapters/test_replay_file.py",)

MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        tag="approvals",
        why="auto_allow_when can BROADEN instead of only narrowing, becoming a second way in",
        path="agentkit/integrations/mcp/approvals.py",
        before="            tier != Autonomy.MANUAL\n            and tool_name in self.auto_allow\n            and self._arguments_are_auto_allowed(tool_name, arguments)",
        after="            tier != Autonomy.MANUAL\n            and tool_name in self.auto_allow\n            or self._arguments_are_auto_allowed(tool_name, arguments)",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="frozen",
        why="deep_freeze goes shallow, so cp.state['a']['b'] = 1 rewrites a durable record again",
        path="agentkit/core/serialization/_immutable.py",
        before="    if isinstance(value, dict):\n        return FrozenDict({k: deep_freeze(v) for k, v in value.items()})",
        after="    if isinstance(value, dict):\n        return FrozenDict(value)",
        tests=FROZEN_TESTS,
    ),
    Mutant(
        tag="frozen",
        why="FrozenDict loses __reduce__, so deepcopy and pickle break at the checkpoint seam",
        path="agentkit/core/serialization/_immutable.py",
        before="        return (FrozenDict, (dict(self),))",
        after="        return (dict, (dict(self),))",
        tests=FROZEN_TESTS,
    ),
    Mutant(
        tag="frozen",
        why="deep_freeze stops copying, so the caller keeps a live handle on what they handed over",
        path="agentkit/core/serialization/_immutable.py",
        before="    if isinstance(value, list):\n        return FrozenList([deep_freeze(v) for v in value])",
        after="    if isinstance(value, list):\n        return FrozenList(value)",
        tests=FROZEN_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ContextDiff keys on insertion ORDER, breaking the hash invariant for equal diffs",
        path="agentkit/state/context/context.py",
        before="                frozenset(self.scratchpad_changes),",
        after="                tuple(self.scratchpad_changes),",
        tests=CTXDIFF_TESTS,
    ),
    Mutant(
        tag="valuetype",
        why="ToolCall is unhashable again, so union merge dies on every tool-using agent",
        path="agentkit/core/contracts/types.py",
        before="        return hash((self.id, self.name))",
        after="        return hash(self.arguments)",
        tests=VALUETYPE_TESTS,
    ),
    Mutant(
        tag="valuetype",
        why="ToolCall.arguments stops being frozen, so a tool call's payload is editable again",
        path="agentkit/core/contracts/types.py",
        before=(
            '        object.__setattr__(self, "arguments", deep_freeze(self.arguments))\n\n'
            "    def __hash__(self) -> int:\n"
            '        """Hash on IDENTITY — ``(id, name)`` — never on ``arguments``.'
        ),
        after=(
            "        pass\n\n"
            "    def __hash__(self) -> int:\n"
            '        """Hash on IDENTITY — ``(id, name)`` — never on ``arguments``.'
        ),
        tests=VALUETYPE_TESTS,
    ),
    Mutant(
        tag="valuetype",
        why="scratchpad VALUES re-enter the FrozenContext hash, breaking its memoization-key promise",
        path="agentkit/state/context/context.py",
        before="        return hash((self.prefix, self.messages, tuple(k for k, _ in self.scratchpad)))",
        after="        return hash((self.prefix, self.messages, self.scratchpad))",
        tests=VALUETYPE_TESTS,
    ),
    Mutant(
        tag="parsed",
        why="CallableLLM.stream drops parsed, so chat() and stream() disagree on the same call",
        path="agentkit/models/adapters/_mapping.py",
        before="        parsed=res.parsed,\n",
        after="",
        tests=MAPPING_TESTS,
    ),
    Mutant(
        tag="schema",
        why="tuple containers fall through unremapped, so nested models keep the wrong alias",
        path="agentkit/models/output/pydantic_adapter.py",
        before='and isinstance(dumped, (list, tuple)):',
        after='and isinstance(dumped, (list,)):',
        tests=SCHEMA2_TESTS,
    ),
    Mutant(
        tag="schema",
        why="the walk stops recursing, fixing only the top level of a nested model",
        path="agentkit/models/output/pydantic_adapter.py",
        before='return {rename.get(k, k): _to_validation_keys(children.get(k), v) for k, v in dumped.items()}',
        after='return {rename.get(k, k): v for k, v in dumped.items()}',
        tests=SCHEMA2_TESTS,
    ),
    Mutant(
        tag="mcphttp",
        why="the client falls back to the deprecated transport even where the new one exists",
        path="agentkit/integrations/mcp/client.py",
        before="    _OWNS_HTTP_CLIENT = True",
        after="    raise ImportError('forced fallback')",
        tests=MCPHTTP_TESTS,
    ),
    Mutant(
        tag="mcphttp",
        why="headers are dropped when they move onto the client we now own",
        path="agentkit/integrations/mcp/client.py",
        before="                            headers=self.server.headers or {},",
        after="                            headers={},",
        tests=MCPHTTP_TESTS,
    ),
    Mutant(
        tag="prov",
        why="the permanent-429 message carries a transient marker again, so classify says retry",
        path="agentkit/models/providers/frontiers/base.py",
        before="def _permanent_message(",
        after="def _unused_permanent_message(",
        tests=PROV_TESTS,
    ),
    Mutant(
        tag="prov",
        why="the error body is truncated BEFORE parsing, losing error.type on a long body",
        path="agentkit/models/providers/frontiers/base.py",
        before="def _error_type(",
        after="def _unused_error_type(",
        tests=PROV_TESTS,
    ),
    Mutant(
        tag="prov",
        why="a dict-shaped provider response coerces to an empty result again",
        path="agentkit/models/adapters/_mapping.py",
        before="def _first_int(",
        after="def _unused_first_int(",
        tests=MAP_TESTS,
    ),
    Mutant(
        tag="prov",
        why="multi-line SSE data frames are dropped whole and silently",
        path="agentkit/models/providers/frontiers/base.py",
        before="def _sse_decode(",
        after="def _unused_sse_decode(",
        tests=SSE_TESTS,
    ),
    Mutant(
        tag="tokens",
        why="tool calls stop counting, so compaction never fires on an agentic transcript",
        path="agentkit/state/context/tokens.py",
        before="def estimate_message_tokens(",
        after="def _unused_estimate_message_tokens(",
        tests=TOK_TESTS,
    ),
    Mutant(
        tag="schema",
        why="an init=False dataclass field crashes with a raw TypeError again",
        path="agentkit/models/output/dataclass_adapter.py",
        before="def _coerce_dataclass(",
        after="def _unused_coerce_dataclass(",
        tests=SCHEMA2_TESTS,
    ),
    Mutant(
        tag="schema",
        why="serialize emits the SERIALIZATION alias, which parse cannot read",
        path="agentkit/models/output/pydantic_adapter.py",
        before="        return cast(dict[str, Any], _to_validation_keys(value, dumped))",
        after="        return dumped",
        tests=SCHEMA2_TESTS,
    ),
    Mutant(
        tag="merge",
        why="a raising source posts the DONE frame again, truncating the fan-in in silence",
        path="agentkit/core/execution/streams.py",
        before="            elif error is not None:",
        after="            elif False:",
        tests=STREAMS_TESTS,
    ),
    Mutant(
        tag="merge",
        why="the error is dropped instead of transported, so gather() discards it",
        path="agentkit/core/execution/streams.py",
        before="            error = exc",
        after="            error = None",
        tests=STREAMS_TESTS,
    ),
    Mutant(
        tag="conc",
        why="an abort leaves its siblings running detached, still spending",
        path="agentkit/core/execution/concurrency.py",
        before=(
            "        for t in tasks:\n            t.cancel()\n"
            "        if tasks:\n"
            "            _, pending = await asyncio.wait(tasks, timeout=SIBLING_CLEANUP_GRACE_S)"
        ),
        after="        pass",
        tests=CONC_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="the result carries structuredContent, which the CLI rejects outright",
        path="agentkit/integrations/mcp/approvals.py",
        before="        @mcp.tool(name=TOOL_NAME, structured_output=False)",
        after="        @mcp.tool(name=TOOL_NAME)",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="an approval gate that fails OPEN when the transport breaks",
        path="agentkit/integrations/mcp/approvals.py",
        before='            return _deny(f"the approval transport failed ({type(exc).__name__}: {exc})")',
        after="            return _allow(arguments)",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="an expired prompt allows, so a slow reviewer becomes a yes",
        path="agentkit/integrations/mcp/approvals.py",
        before="            return Decision(kind=\"expired\", note=f\"no answer within {timeout}s\")",
        after="            return Decision(kind=\"approve\", note=\"timed out\")",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="the timeout is not enforced, so a turn can park forever",
        path="agentkit/integrations/mcp/approvals.py",
        before="        if timeout is None:\n            return await asker.ask(request)",
        after="        if True:\n            return await asker.ask(request)",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="a modify decision is treated as a plain approval, discarding the reviewer's edits",
        path="agentkit/integrations/mcp/approvals.py",
        before="        if decision.kind == \"modify\":",
        after="        if False:",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="a denial reaches the model with no reason, so it cannot adapt",
        path="agentkit/integrations/mcp/approvals.py",
        before='        return _deny(decision.note or f"a reviewer declined the {tool_name} call")',
        after='        return _deny("")',
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="auto_allow is ignored, so every read prompts and habituates the reviewer",
        path="agentkit/integrations/mcp/approvals.py",
        before="            tier != Autonomy.MANUAL\n            and tool_name in self.auto_allow\n            and self._arguments_are_auto_allowed(tool_name, arguments)",
        after="            False",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="the arguments never reach the human, who then approves a path they cannot see",
        path="agentkit/integrations/mcp/approvals.py",
        before='            tool_call={"name": tool_name, "arguments": arguments},',
        after="            tool_call=None,",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="the CLI is left to load whatever MCP servers the working directory defines",
        path="agentkit/integrations/mcp/approvals.py",
        before='            "strict_mcp_config": True,',
        after='            "strict_mcp_config": False,',
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="an unknown argument is dropped, so a defaulted parameter runs with its default",
        path="agentkit/tools/declarations/function.py",
        before="            if unexpected or missing:",
        after="            if False:",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="only unknown args are reported, so a missing required one stays a raw TypeError",
        path="agentkit/tools/declarations/function.py",
        before="            missing = tuple(k for k in required_params if k not in kwargs)",
        after="            missing = ()",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="**kwargs stops receiving the extras it exists to accept",
        path="agentkit/tools/declarations/function.py",
        before="                kwargs.update({k: supplied[k] for k in unexpected})\n                unexpected = ()",
        after="                unexpected = ()",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="a model-supplied ctx key trips the check instead of being ignored",
        path="agentkit/tools/declarations/function.py",
        before="                k for k in supplied if k not in arg_params and k not in ctx_params",
        after="                k for k in supplied if k not in arg_params",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="a structured parameter is advertised as a bare string again",
        path="agentkit/tools/declarations/schema.py",
        before="    struct = _struct_fragment(ann)\n    if struct is not None:\n        return struct\n",
        after="",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="an Enum parameter loses its member list",
        path="agentkit/tools/declarations/schema.py",
        before="    if isinstance(ann, type) and issubclass(ann, enum.Enum):\n        return _enum_fragment(ann)\n",
        after="",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="toolargs",
        why="a heterogeneous enum gets a guessed type instead of none",
        path="agentkit/tools/declarations/schema.py",
        before=(
            "    kinds = {_JSON_PRIMITIVES.get(type(v)) for v in vals}\n"
            "    if len(kinds) == 1 and (t := kinds.pop()) is not None:\n"
            "        frag[\"type\"] = t\n"
            "    return frag"
        ),
        after="    frag[\"type\"] = \"string\"\n    return frag",
        tests=TOOLARG_TESTS,
    ),
    Mutant(
        tag="resilience",
        why="a late success closes an OPEN breaker, bypassing the cooldown",
        path="agentkit/core/execution/resilience.py",
        before='            if self.state == "open":\n                return',
        after='        if False:\n            return',
        tests=RESILIENCE_TESTS,
    ),
    Mutant(
        tag="provider",
        why="an in-band SSE error frame is swallowed, truncating the answer silently",
        path="agentkit/models/providers/frontiers/base.py",
        before="    err = event.get(\"error\")",
        after="    return\n    err = event.get(\"error\")",
        tests=PROVIDER_TESTS,
    ),
    Mutant(
        tag="provider",
        why="Anthropic stops checking for the error frame",
        path="agentkit/models/providers/frontiers/anthropic.py",
        before="            raise_if_error_frame(ev)",
        after="            pass",
        tests=PROVIDER_TESTS,
    ),
    Mutant(
        tag="provider",
        why="an overload frame stops classifying as retryable",
        path="agentkit/core/execution/resilience.py",
        before='    "rate_limit",\n    "server_error",\n    "overloaded",',
        after='    "__never_matches_rate_limit",\n    "__never_matches_server_error",\n    "__never_matches_overloaded",',
        tests=PROVIDER_TESTS,
    ),
    Mutant(
        tag="recurrence",
        why="The `is None` success check must be read AFTER the fingerprint is awaited. Moved before, it sees a coroutine \u2014 never None \u2014 so an async fingerprint can never report success: the run loops, repeats, and is declared PERMANENTLY stuck on a SUCCESSFUL outcome, leaving an un-awaited coroutine each turn. THIS MUTANT SURVIVED THE SHIPPED 21-TEST SUITE (their test_an_async_fingerprint_is_awaited only exercises a non-None async signature); the test below is new and kills it.",
        path="agentkit/core/execution/recurrence.py",
        before="        if inspect.isawaitable(signature):\n            # The signature is frequently itself a model call (\"does this answer the goal?\"), so\n            # it has to be allowed to be async. Same duck-typed treatment ``LedgerPolicy`` gives\n            # its assessor and planner \u2014 a sync lambda stays a sync lambda.\n            signature = await signature\n        if signature is None:\n            return outcome",
        after="        if signature is None:\n            return outcome\n        if inspect.isawaitable(signature):\n            signature = await signature",
        tests=RECURRENCE_TESTS,
    ),
    Mutant(
        tag="recurrence",
        why="A shared stable_hash key must be CONFIRMED by equality before a repeat is declared. Trusting the hash alone restores the shipped bug: stable_hash's last-resort branch describes an opaque value by bare type name, so distinct signatures collide and a run making real progress is killed with a PERMANENT/non-retriable verdict it never proved.",
        path="agentkit/core/execution/recurrence.py",
        before="        first_seen = next((earlier for earlier, prior in bucket if _same_signature(prior, signature)), None)",
        after="        first_seen = next((earlier for earlier, _prior in bucket), None)",
        tests=RECURRENCE_TESTS,
    ),
    Mutant(
        tag="recurrence",
        why="An unrecognised on_repeat must raise, not fall through to the non-raising branch. Widening the accepted set restores the silent-success path: a caller who asked for a raise gets a PERMANENT Failure returned as an ordinary value, and code that only handles Stuck treats the stall as the answer.",
        path="agentkit/core/execution/recurrence.py",
        before="    if on_repeat not in (\"escalate\", \"stop\"):",
        after="    if on_repeat not in (\"escalate\", \"stop\", \"raise\"):",
        tests=RECURRENCE_TESTS,
    ),
    Mutant(
        tag="recurrence",
        why="A signature comparison that raises (numpy-style elementwise `==`) must count as NOT the same, so the run falls through to the backstop and is reported UNKNOWN/retriable 'out of budget'. Flipping it to True makes an uncomparable signature type report PERMANENT/non-retriable stuck on attempt 2 \u2014 wrong in the terminal direction.",
        path="agentkit/core/execution/recurrence.py",
        before="    except Exception:\n        return False",
        after="    except Exception:\n        return True",
        tests=RECURRENCE_TESTS,
    ),
    Mutant(
        tag="recurrence",
        why="The recorded attempt number is the entire reason the history is a dict of attempts and not a bare set \u2014 'attempt 3 repeats attempt 1' is what tells an operator this is an oscillation rather than a wedge. The shipped suite never asserted the message, so the attempt number was free to rot into a placeholder undetected.",
        path="agentkit/core/execution/recurrence.py",
        before="        bucket.append((attempt, signature))",
        after="        bucket.append((0, signature))",
        tests=RECURRENCE_TESTS,
    ),
    Mutant(
        tag="recurrence",
        why="The history must be every signature seen, not a one-element window. Collapsing it makes an A,B,A,B oscillation read as progress and silently reverts the bound to max_attempts \u2014 the exact naive form the module exists to reject.",
        path="agentkit/core/execution/recurrence.py",
        before="        bucket = seen.setdefault(key, [])",
        after="        seen.clear()\n        bucket = seen.setdefault(key, [])",
        tests=RECURRENCE_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="A `value` decision counted as CONSENT. `_render` allowed on `kind in (\"approve\", \"value\")` while `Decision.approved` \u2014 the predicate `ReActCognition` gates on \u2014 is `kind in (\"approve\", \"modify\")`. Same Asker, same Decision: a refusal on agentkit's own tool loop, an approval here. Observed: `Decision(kind=\"value\", value=\"no, absolutely not\")` returned `{\"behavior\": \"allow\"}`. This is exactly the divergence class F2 routed `autonomy` through `should_gate` to prevent, and it was unpinned in either direction. FIXED by gating on `decision.approved`; the mutant reverts it.",
        path="agentkit/integrations/mcp/approvals.py",
        before="        if not decision.approved:",
        after="        if decision.kind not in (\"approve\", \"modify\", \"value\"):",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="`auto_allow=\"Read\"` \u2014 the missing comma \u2014 auto-allowed every SUBSTRING of itself, approving those calls with the reviewer never consulted. Observed against the shipped code: with `auto_allow=\"Read\"`, tool `\"R\"` -> allow and tool `\"ea\"` -> allow, asker never called. The constructor already refused three misconfigurations; this fourth one fails OPEN and was unguarded. FIXED with a construction-time refusal; the mutant removes it.",
        path="agentkit/integrations/mcp/approvals.py",
        before="        if isinstance(self.auto_allow, str):",
        after="        if False:",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="`_decide`'s stated never-raises contract was violated on all three ALLOW paths. `_allow()`'s `json.dumps` sat outside every `try`, so an unencodable argument escaped as `TypeError` \u2014 which the CLI reports as a BROKEN PERMISSION SYSTEM, not a denial, leaving the model to retry a call nobody approved. Observed: `RAISED TypeError` on the AUTO branch and the auto_allow branch. The catch-all on the reviewer path did fire but mislabelled it, emitting the self-contradictory `\"the reviewer's answer was not a Decision (Decision: TypeError: ...)\"`. FIXED by guarding serialisation inside `_allow` (also `allow_nan=False`, since bare `NaN` is not JSON and the CLI's parser rejects it).",
        path="agentkit/integrations/mcp/approvals.py",
        before="    except (TypeError, ValueError) as exc:",
        after="    except ValueError as exc:",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="uvicorn signals a failed bind with `sys.exit(3)`, and asyncio treats a Task's `SystemExit` as 'stop the loop' \u2014 so `start()` was cancelled at its `sleep` before ever reaching its own `_task.done()` branch. Observed against an occupied port: the caller got a bare `CancelledError`, a BaseException, so `except Exception` around the documented `async with ApprovalServer(...)` caught NOTHING and the process unwound (in the FastAPI recipe on this seam, the whole worker). FIXED by containing `SystemExit` inside `_serve` and converting it to a RuntimeError naming host:port; the mutant restores the raw `serve()`.",
        path="agentkit/integrations/mcp/_transport.py",
        before="        except OSError as exc:\n            sock.close()",
        after="        except OSError as exc:\n            pass",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="THE SILENT SUCCESS. A failed `start()` left `_task` and `_server` set, so a retry hit `if self._task is not None: return` and returned cleanly from a server that was NOT listening \u2014 observed `serving=False` with no error raised. Its `url` then went to the CLI, which fails ~30s later with a startup timeout: precisely the failure the `while not self._server.started` wait loop was written to prevent, reached through the path meant to report it. FIXED by clearing both fields before raising; the mutant restores the leak.",
        path="agentkit/integrations/mcp/_transport.py",
        before="                f\"{exc.strerror}: could not bind {self.host}:{self.port}\",",
        after="                exc.strerror,",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="`timeout_s` re-assigned to 0 after construction denied EVERY prompt with the Asker never scheduled \u2014 `dontAsk`, the exact failure the module exists to remove \u2014 and did so under the message 'no answer within 0s', which blames a reviewer who was never asked and sends the operator to debug the transport. Observed: deny, `asker.seen == []`. The implementer built belt-and-braces for the other mutable field (`autonomy` -> `_tier()` fails closed) but not for this one. FIXED with the symmetric re-check in `_ask`; it still denies (closed direction) but now names the misconfiguration.",
        path="agentkit/integrations/mcp/approvals.py",
        before="        if timeout <= 0:",
        after="        if False:",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="SURVIVED THE SHIPPED SUITE \u2014 a genuine coverage gap, not a bug. `prompts_seen` was never asserted under `autonomy=\"auto\"`, which is the one tier where there is no Asker, no reviewer and no elicitation, so the count is the ONLY evidence the server saw a tool call and allowed it. A regression stopping the AUTO increment was invisible. Test added, no source change needed.",
        path="agentkit/integrations/mcp/approvals.py",
        before="        self._seen += 1\n",
        after="        self._seen += 0 if self.autonomy == 'auto' else 1\n",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="SURVIVED THE SHIPPED SUITE \u2014 the second coverage gap. Nothing asserted that `deadline_s` reaches the `Elicitation`. `timeout_s` is enforced server-side either way, so losing it is invisible to the CLI and very visible to the human: their pending card shows no deadline, they answer at their own pace, and the answer is discarded by a timeout they were never shown. Test added (both the 30.0 case and that None stays None), no source change needed.",
        path="agentkit/integrations/mcp/approvals.py",
        before="            deadline_s=self.timeout_s,",
        after="            deadline_s=None,",
        tests=APPROVAL_AUTONOMY_TESTS,
    ),
    Mutant(
        tag="tooldesc",
        why="THE BUG. `inspect.getdoc` dedents by the COMMON leading whitespace, so a developer note written one level in keeps its extra indent through the dedent and a column-0 anchor never sees it. Reverting to column-0-only lets `Notes on behaviour:` / `  - TODO: rewrite before Q3` ship to the model verbatim \u2014 the exact failure mode the check exists to prevent, arriving by the route authors actually take. Verified: `before` occurs exactly once, mutated source parses (ast.parse OK).",
        path="agentkit/tools/declarations/schema.py",
        before="_DEV_NOTE_RE = re.compile(r\"^[ \\t]*(?:[-*+][ \\t]+)?(TODO|FIXME|XXX|HACK)\\b\", re.MULTILINE)",
        after="_DEV_NOTE_RE = re.compile(r\"^(TODO|FIXME|XXX|HACK)\\b\", re.MULTILINE)",
        tests=TOOLDESC_TESTS,
    ),
    Mutant(
        tag="tooldesc",
        why="The bullet allowance specifically. A note is most often written as one list item among real ones, where the `- ` sits between the marker and the line start. Dropping only this clause still catches a plainly-indented note but silently passes the commonest form. Verified: `before` occurs exactly once, mutated source parses.",
        path="agentkit/tools/declarations/schema.py",
        before="_DEV_NOTE_RE = re.compile(r\"^[ \\t]*(?:[-*+][ \\t]+)?(TODO|FIXME|XXX|HACK)\\b\", re.MULTILINE)",
        after="_DEV_NOTE_RE = re.compile(r\"^[ \\t]*(TODO|FIXME|XXX|HACK)\\b\", re.MULTILINE)",
        tests=TOOLDESC_TESTS,
    ),
    Mutant(
        tag="tooldesc",
        why="`\\b` is the only thing keeping the guard from firing on ordinary prose that merely BEGINS with a marker. Without it, a docstring line reading `XXXL is a size, not a note.` or `HACKS are documented below.` becomes a hard ToolDefinitionError at import in somebody else's application \u2014 a false positive on a decoration-time refusal, which is the most expensive kind. The shipped suite did not pin this: its mid-sentence test is saved by the `^` anchor, not by `\\b`, so it passes with `\\b` removed. Verified: `before` occurs exactly once, mutated source parses.",
        path="agentkit/tools/declarations/schema.py",
        before="_DEV_NOTE_RE = re.compile(r\"^[ \\t]*(?:[-*+][ \\t]+)?(TODO|FIXME|XXX|HACK)\\b\", re.MULTILINE)",
        after="_DEV_NOTE_RE = re.compile(r\"^[ \\t]*(?:[-*+][ \\t]+)?(TODO|FIXME|XXX|HACK)\", re.MULTILINE)",
        tests=TOOLDESC_TESTS,
    ),
    Mutant(
        tag="tooldesc",
        why="The second `.replace` handles a bare `\\r`, which is what a classic-Mac docstring uses throughout and what a `\\r\\r` paragraph break degrades to. The author's CRLF test only exercises the first replace, so this half was unpinned \u2014 and since the whole docstring now ships, an un-normalised `\\r` goes straight into the JSON handed to the provider. Observed with the mutant applied: the description came back as 'Do the documented thing properly.\\r\\r    And the second paragraph.' with the phantom indent intact. Verified: `before` occurs exactly once, mutated source parses.",
        path="agentkit/tools/declarations/schema.py",
        before="        doc = inspect.cleandoc(doc.replace(\"\\r\\n\", \"\\n\").replace(\"\\r\", \"\\n\"))",
        after="        doc = inspect.cleandoc(doc.replace(\"\\r\\n\", \"\\n\"))",
        tests=TOOLDESC_TESTS,
    ),
    Mutant(
        tag="tooldesc",
        why="The conditional clause the author added to the floor error so that a docstring emptied by its own `---` line reports something fixable instead of \"got 0 chars: ''\". The shipped suite exercises that scenario but asserts only on the generic half of the message, so the entire clause could be deleted with the suite still green \u2014 the explanatory feature was untested. Verified: `before` occurs exactly once, mutated source parses.",
        path="agentkit/tools/declarations/schema.py",
        before="            if description is None and tail_cut\n            else \"\"",
        after="            if False\n            else \"\"",
        tests=TOOLDESC_TESTS,
    ),
    Mutant(
        tag="tooldesc",
        why="The per-line `rstrip()` is load-bearing, not tidying, and nothing pinned it. Measured directly: `inspect.cleandoc`/`getdoc` dedent and trim blank lines at the ends but do NOT rstrip interior lines, so trailing whitespace an editor left behind survives into the description and now \u2014 with whole docstrings shipping \u2014 rides to the provider verbatim. Verified: `before` occurs exactly once, mutated source parses.",
        path="agentkit/tools/declarations/schema.py",
        before="    return \"\\n\".join(line.rstrip() for line in lines).strip(), cut",
        after="    return \"\\n\".join(lines).strip(), cut",
        tests=TOOLDESC_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="the gate.check event loses the agent it was gated FOR, so a run with a supervisor and three workers puts every permission decision on one stream with nothing saying whose call was refused \u2014 provenance is the entire point of the record",
        path="agentkit/integrations/mcp/approvals.py",
        before="                    agent=self.agent,",
        after="                    agent=\"\",",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="the emit fires BEFORE the append, so a UI that renders the trail when it hears gate.check paints a pane one row short of the run \u2014 this is _record's stated append-first contract, which the shipped suite could not observe because _emit swallows its own exceptions",
        path="agentkit/integrations/mcp/approvals.py",
        before="        self._decisions.append(record)\n        await self._emit(record)",
        after="        await self._emit(record)\n        self._decisions.append(record)",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="the broadcast claims nobody was interrupted for every prompt, so a consumer reading only the stream cannot tell a reviewer who was woken up from a tier that decided on its own \u2014 `asked` is the field the class docstring says is derivable from nothing else",
        path="agentkit/integrations/mcp/approvals.py",
        before="                        \"asked\": record.asked,",
        after="                        \"asked\": False,",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="the broadcast drops the run-clock timestamp, so the one field that lets an approval be lined up against the checkpoint beside it never leaves the process \u2014 the whole reason `clock` was injected instead of calling time.time()",
        path="agentkit/integrations/mcp/approvals.py",
        before="                        \"at\": record.at,",
        after="                        \"at\": \"\",",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="a `value` answer \u2014 the transport handing back what the person TYPED instead of a verdict \u2014 is filed as `source=\"asker\"`, i.e. as a human who declined. This is the shipped bug: identical to a real refusal in every field, sending the reader after a reviewer instead of after the Asker whose author is the only one who can tell yes from no. _render already refuses to make this mistake in the text the model reads.",
        path="agentkit/integrations/mcp/approvals.py",
        before="            elif decision.kind == \"value\":",
        after="            elif decision.kind == \"never\":",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="the observer emit becomes unbounded again, so a hung log shipper parks the CLI's permission answer forever and wedges the run \u2014 `dontAsk` by a slower route. contextlib.suppress covers an observer that raises and says nothing about one that hangs.",
        path="agentkit/integrations/mcp/approvals.py",
        before="            await asyncio.wait_for(",
        after="            await asyncio.shield(",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="small argument values are stored as their JSON RENDERING rather than as they arrived, so decision.arguments['file_path'] is '\"/tmp/out\"' with the quotes and every caller comparison against a real path silently fails",
        path="agentkit/integrations/mcp/approvals.py",
        before="            out[str(key)] = value",
        after="            out[str(key)] = rendered",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="approvalrec",
        why="a reviewer's `modify` records the REQUEST rather than the arguments the CLI actually ran, so the trail names a call that never happened while the one that did \u2014 which exists nowhere else in the run, because the CLI does not tell the model it changed \u2014 goes unrecorded",
        path="agentkit/integrations/mcp/approvals.py",
        before="            arguments=_bounded(updated if updated is not None else arguments),",
        after="            arguments=_bounded(arguments),",
        tests=APPROVAL_RECORD_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="THE FAIL-OPEN, and it is the shipped code. `if scope.get(\"type\") != \"http\"` made \"not an HTTP request\" mean \"not this fence's problem\", so every other ASGI scope reached the app with no credential. Not theoretical: uvicorn is configured with the default `ws=\"auto\"`, which enables websocket upgrades the moment `websockets` or `wsproto` is importable \u2014 neither is an agentkit dependency, so whether the fence held depended on an unrelated package in the caller's environment. Measured with `websockets` on the path: an unauthenticated upgrade against an AUTHENTICATED listener answered `HTTP/1.1 101 Switching Protocols` and the scope reached the served app. It was survivable only because the inner Starlette app happens to have no websocket route today, which puts the authorisation decision in the app's routing table rather than in the fence \u2014 and `LoopbackMcpTransport` is documented as where a third server inherits its fence from. This mutant restores it.",
        path="agentkit/integrations/mcp/_transport.py",
        before="        scope_type = scope.get(\"type\")\n        if scope_type == \"lifespan\":\n            await app(scope, receive, send)\n            return\n        if scope_type != \"http\":",
        after="        scope_type = scope.get(\"type\")\n        if scope_type != \"http\":\n            await app(scope, receive, send)\n            return\n        if False:",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="The refusal has to be SHAPED for the scope, not just withheld. Returning silently from a websocket scope leaves uvicorn holding a half-open handshake rather than denying it; a close before accept is the portable denial and uvicorn renders it as 403 Forbidden (verified over a real socket). The mutant refuses silently, which still keeps the app untouched \u2014 so only an assertion on the messages actually sent can see the difference between 'denied' and 'hung'.",
        path="agentkit/integrations/mcp/_transport.py",
        before="            if scope_type == \"websocket\":\n                await send({\"type\": \"websocket.close\", \"code\": 1008})",
        after="            if False:\n                await send({\"type\": \"websocket.close\", \"code\": 1008})",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="`secrets.compare_digest(b\"\", b\"\")` is TRUE, so a fence built on an empty token is not a closed door with a bad key \u2014 it is an open one that admits everybody sending `Authorization: Bearer ` and logs nothing. Observed before the fix: status 200, inner app entered. `start()` guards its own call site with `if self._token`, but `require_bearer` is in the module's `__all__` and the failure is completely silent, so the refusal belongs where the fence is built. The mutant drops the guard.",
        path="agentkit/integrations/mcp/_transport.py",
        before="    if not token:\n        raise ValueError(",
        after="    if False:\n        raise ValueError(",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="SURVIVED THE SHIPPED SUITE. The whole stated point of refusing a query-string credential on the NAME is to stop the PATTERN, and that is only true if the pattern is more than one spelling \u2014 but the suite only ever exercised `access_token`, so shrinking the set to that one name passed everything. Six of the seven names were dead code as far as any test could tell, and `?api_key=`/`?token=` are at least as likely to be what a caller reaches for.",
        path="agentkit/integrations/mcp/_transport.py",
        before="{\"access_token\", \"api_key\", \"apikey\", \"authorization\", \"bearer\", \"key\", \"token\"}",
        after="{\"access_token\"}",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="SURVIVED THE SHIPPED SUITE. Five lines of comment explain why the header loop `break`s and nothing tested it, so turning the `break` into a `continue` passed the entire suite. Two things go wrong without it: a caller sprays many candidate credentials in ONE request, so a rate limit counting requests counts one; and what the server accepts starts depending on header ORDER, which nothing in an HTTP path is obliged to preserve. Verified by hand that the shipped code is correct \u2014 [wrong, correct] gets 401 \u2014 it simply had no test.",
        path="agentkit/integrations/mcp/_transport.py",
        before="            # order \u2014 which no proxy in the path is obliged to preserve.\n            break",
        after="            # order \u2014 which no proxy in the path is obliged to preserve.\n            continue",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="SURVIVED THE SHIPPED SUITE, and it is the most damaging regression this feature could have. The 401 body exists to tell an operator WHERE the token already is; an f-string interpolating `token` into `detail` makes it tell them what the token IS \u2014 handed to the caller that just proved it has no credential, and into their logs. Every existing assertion on the body checks for a substring that is still present, so nothing noticed. The mutant restores the leak.",
        path="agentkit/integrations/mcp/_transport.py",
        before="                \"this agentkit MCP server requires the bearer token from its \"",
        after="                f\"this agentkit MCP server requires the bearer token {token} from its \"",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="THE TOCTOU, and it is the shipped code. `write_text` creates at `0o666 & ~umask` \u2014 0644 under the common umask \u2014 and only then does `chmod(0o600)` run, so the live bearer token sat on disk world-readable in between. Measured, not argued: a polling thread read the real credential out of the file at 0644 and it matched the running server's `auth_headers`. The default config_path hides inside a 0700 mkdtemp, but `config_path=` is a supported argument and `stop()`'s own docstring reasons about callers who point it at a directory that persists \u2014 exactly where the window is reachable \u2014 and it reopened on every `start()`. Asserting the FINAL mode, which is what the shipped test does, cannot see this at all.",
        path="agentkit/integrations/mcp/serve.py",
        before="        fd = os.open(\n            self.config_path,\n            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,\n            0o600,\n        )",
        after="        self.config_path.write_text(json.dumps(self._document, indent=2))\n        fd = os.open(self.config_path, os.O_RDONLY)",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="`O_CREAT`'s mode argument applies ONLY when the file is created, so the open-with-0600 alone does nothing for a path that already exists \u2014 a previous run's file, or one pre-created by whoever can write that directory. Without the fchmod such a file keeps whatever mode it had and is then filled with a fresh token anyway. The mutant drops the fchmod; the create-from-nothing path still passes, so only a test that pre-creates the file at a loose mode can see it.",
        path="agentkit/integrations/mcp/serve.py",
        before="            os.fchmod(fd, 0o600)",
        after="            pass",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="The silent regression that came in with the lazy, cached listener. Making the transport outlive `stop()` also made it built on first reference, freezing host/port at whatever they were when anything first touched `mcp_config`/`auth_headers`/`cli_kwargs()`. Before the change `start()` built it fresh, so a late `server.port = ...` was honoured. Measured after: asked for a specific port following an early `mcp_config` read, got an OS-chosen one \u2014 silently, while the caller believed they had pinned it. The mutant drops the re-sync. Note the fix is a re-sync rather than a rebuild on purpose: rebuilding would rotate the token out from under a config document the caller already holds, which is the thing the caching exists to prevent.",
        path="agentkit/integrations/mcp/approvals.py",
        before="        transport.host, transport.port = self.host, self.port",
        after="        pass",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="mcpauth",
        why="the config open follows a symlink again, so a link already sitting at config_path redirects the bearer token to whatever it points at, at that file's mode \u2014 the fchmod then tightens a descriptor that was never the risk",
        path="agentkit/integrations/mcp/serve.py",
        before="            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,",
        after="            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,",
        tests=MCP_AUTH_TESTS,
    ),
    Mutant(
        tag="schemabudget",
        why="expanded Anthropic references retain redundant definitions and reject reachable schemas",
        path="agentkit/models/contracts/schema.py",
        before='            out.pop("$defs", None)',
        after='            out["$defs"] = copy.deepcopy(node.get("$defs", {}))',
        tests=("tests/source/conformance/test_schema_reference_budgets.py",),
    ),
    Mutant(
        tag="schemabudget",
        why="one union-bearing parameter is counted twice when it has both anyOf and a type array",
        path="agentkit/models/contracts/schema.py",
        before='            union_count += int("anyOf" in node or isinstance(node.get("type"), list))',
        after='            union_count += int("anyOf" in node) + int(isinstance(node.get("type"), list))',
        tests=("tests/source/conformance/test_schema_reference_budgets.py",),
    ),
    Mutant(
        tag="schemabudget",
        why="real Anthropic optional and union budget overflows are allowed past the documented boundary",
        path="agentkit/models/contracts/schema.py",
        before='        if optional_count > 24 or union_count > 16:',
        after='        if optional_count > 25 or union_count > 17:',
        tests=("tests/source/conformance/test_schema_reference_budgets.py",),
    ),
    Mutant(
        tag="calls",
        why="optional strict placeholders erase required nulls from canonical answers",
        path="agentkit/models/contracts/schema.py",
        before='if item is None and key in props and key not in required and not valid(None, child):',
        after='if item is None and key in props:',
        tests=("tests/source/conformance/test_calls.py", "tests/source/conformance/test_schema_boundaries.py"),
    ),
    Mutant(
        tag="calls",
        why="cumulative usage charges the cached prefix repeatedly on resumed CLI calls",
        path="agentkit/models/catalog/pricing.py",
        before="        return a - b if a >= b else a",
        after="        return a",
        tests=("tests/source/conformance/test_calls.py", "tests/source/conformance/test_accounting.py"),
    ),
    Mutant(
        tag="calls",
        why="MCP deny rules cease to override matching allow grants",
        path="agentkit/integrations/mcp/gateway.py",
        before=(
            'return not any(fnmatchcase(key, x) for x in self.deny) '
            'and any(fnmatchcase(key, x) for x in self.allow)'
        ),
        after='return any(fnmatchcase(key, x) for x in self.allow)',
        tests=("tests/source/conformance/test_gateway.py",),
    ),
    # ── fail-closed execution contracts ─────────────────────────────────────
    #
    # `evaluate_cli_execution` decides, before any credential is resolved and
    # before the CLI is spawned, whether a containment request can be honoured.
    # Its entire value is that it answers NO unless there is evidence. Every
    # mutant below turns one of those refusals into a grant, which is the only
    # direction that matters: a false UNSUPPORTED costs a caller a refusal they
    # can read, and a false SUPPORTED hands them a boundary that does not exist.
    Mutant(
        tag="execfailclosed",
        why="every requirement starts SUPPORTED instead of UNSUPPORTED, so a caller asking for filesystem, network, process or credential boundaries is told all of them hold while nothing enforces any of them — the single most dangerous line in the module",
        path="agentkit/models/providers/cli/capabilities.py",
        before="        status = CapabilityStatus.UNSUPPORTED\n",
        after="        status = CapabilityStatus.SUPPORTED\n",
        tests=EXECUTION_TESTS,
    ),
    Mutant(
        tag="execfailclosed",
        why="a backend declaring a guarantee is promoted straight to SUPPORTED rather than clamped to UNVERIFIED, so containment becomes self-attesting: whoever writes the BackendReport decides whether the boundary exists, which is exactly the authority the design refuses to delegate",
        path="agentkit/models/providers/cli/capabilities.py",
        before="        if requirement in report.guarantees and status != CapabilityStatus.SUPPORTED:\n            status = CapabilityStatus.UNVERIFIED\n",
        after="        if requirement in report.guarantees and status != CapabilityStatus.SUPPORTED:\n            status = CapabilityStatus.SUPPORTED\n",
        tests=EXECUTION_TESTS,
    ),
    Mutant(
        tag="execfailclosed",
        why="only UNSUPPORTED counts as unmet, so UNVERIFIED silently satisfies a required boundary — the module's docstring says 'Unverified is a refusal when the capability is required' and this is the line that makes it true. Claude's native-tool denial is UNVERIFIED, so this is the mutation that would let it pass as proven",
        path="agentkit/models/providers/cli/capabilities.py",
        before="    unmet = [fact for fact in facts if fact.requirement in required and fact.status != CapabilityStatus.SUPPORTED]\n",
        after="    unmet = [fact for fact in facts if fact.requirement in required and fact.status == CapabilityStatus.UNSUPPORTED]\n",
        tests=EXECUTION_TESTS,
    ),
    Mutant(
        tag="execfailclosed",
        why="NativeToolPolicy.DENY stops adding NATIVE_TOOLS_UNAVAILABLE to the required set, so the strictest policy is evaluated as though it had asked for nothing — DENY becomes a label rather than a constraint",
        path="agentkit/models/providers/cli/capabilities.py",
        before="    if policy == NativeToolPolicy.DENY:\n        required.add(ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE)\n",
        after="    if policy == NativeToolPolicy.DENY:\n        pass\n",
        tests=EXECUTION_TESTS,
    ),
    Mutant(
        tag="execfailclosed",
        why="the contradiction between CONTAINED native execution and strict native-tool denial stops being reported, so a request that asks both to run native tools in a sandbox AND to have no native tools is answered as satisfiable",
        path="agentkit/models/providers/cli/capabilities.py",
        before="    if execution.native_execution == NativeExecution.CONTAINED and policy == NativeToolPolicy.DENY:\n",
        after="    if False:\n",
        tests=EXECUTION_TESTS,
    ),
    Mutant(
        tag="execfailclosed",
        why="Codex stops being treated as unable to prove native-tool isolation and is downgraded to the same UNVERIFIED as Claude, erasing the one documented difference between the two backends — Codex keeps file/patch/plan capabilities that an empty tool list does not remove",
        path="agentkit/models/providers/cli/capabilities.py",
        before="            if mode == Mode.CODEX:\n",
        after="            if False:\n",
        tests=EXECUTION_TESTS,
    ),
)


# ── invariants the rewrite left unenforced ───────────────────────────────────
#
# These entries' source anchors are LIVE — the code they mutate still exists
# and `--verify` still checks it, so none of this rots. What they lost is their
# verification: the reorganisation deleted the test files that caught them, and
# nothing has replaced those yet. Probed against the test directories that did
# survive, these are not caught.
#
# They are parked here rather than deleted because the analysis in each `why`
# is the map of what the rewrite stopped guaranteeing, and each one becomes a
# working catalogue entry again the moment a test covers it: move it back into
# MUTANTS and point `tests` at the file.
#
# They are OUT of the gate because a catalogue entry that cannot be run proves
# nothing, and leaving them in made the job fail without telling anyone which
# invariants were actually unguarded. `MUTANTS` is now exactly the set that is
# enforced, and the gate's green means that literally.
#
# Run `python .github/scripts/mutants.py --unenforced` to list them.
UNENFORCED: tuple[Mutant, ...] = (
    Mutant(
        tag="frozen",
        why="Checkpoint.state stops being frozen, so a persisted record is rewritable in memory",
        path="agentkit/core/contracts/ports.py",
        before='        object.__setattr__(self, "state", deep_freeze(self.state))',
        after="        pass",
        tests=FROZENVAL_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="Checkpoint hashes its mutable state, so a durable record is unhashable again",
        path="agentkit/core/contracts/ports.py",
        before="        return hash((self.run_id, self.version))",
        after="        return hash((self.run_id, self.version, tuple(self.state)))",
        tests=PORTS_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="SearchHit folds query-dependent score into identity, so one document lands in two buckets",
        path="agentkit/core/contracts/ports.py",
        before="        return hash((self.url, self.title))",
        after="        return hash((self.url, self.title, self.score))",
        tests=PORTS_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="FetchResponse hashes the body, making hash O(page size) on every set insert",
        path="agentkit/core/contracts/ports.py",
        before="        return hash((self.url, self.status, self.content_type, self.fetched_at))",
        after="        return hash((self.url, self.status, self.content_type, self.fetched_at, self.body))",
        tests=PORTS_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ToolSchema hashes the JSON Schema body, so every non-empty schema is unhashable",
        path="agentkit/core/contracts/types.py",
        before="        return hash((self.name, self.description))",
        after="        return hash((self.name, self.description, str(self.parameters)))",
        tests=KERNELTYPE_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ChatRequest hashes the whole transcript, making the hash O(turns) in an agent loop",
        path="agentkit/core/contracts/types.py",
        before="        return hash((self.model, self.temperature, self.max_tokens, len(self.messages)))",
        after="        return hash((self.model, self.temperature, self.max_tokens, tuple(self.messages)))",
        tests=KERNELTYPE_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ToolRequest hashes its decoded-JSON arguments, so a nested payload is unhashable",
        path="agentkit/core/contracts/types.py",
        before="        return hash((self.name, self.side_effecting, self.url_arg))",
        after="        return hash((self.name, self.side_effecting, self.url_arg, str(self.arguments)))",
        tests=KERNELTYPE_TESTS,
    ),
    Mutant(
        tag="regress",
        why="Prompt stops being hashable, breaking every dict/set/lru_cache holding one",
        path="agentkit/agents/prompts/prompt.py",
        before="        return hash((self.id, self.version, self.template, self.inputs))",
        after="        return hash(self.bound)",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="regress",
        why="Prompt.bound stops being frozen, so a caller edits a bound prompt after the fact",
        path="agentkit/agents/prompts/prompt.py",
        before="        frozen = deep_freeze(dict(self.bound))",
        after="        frozen = dict(self.bound)",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="regress",
        why="the retry backoff becomes dead code again, turning retries into a hot loop",
        path="agentkit/core/execution/resilience.py",
        before="            await asleep(backoff_delay(attempt, rng=rng))",
        after="            pass",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="regress",
        why="release_probe stops restarting the cooldown, so the breaker never brakes",
        path="agentkit/core/execution/resilience.py",
        before="            self.state = \"open\"\n            self._opened_at = self.clock()",
        after="            self.state = \"open\"",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="regress",
        why="a refused call loses the failure that opened the breaker",
        path="agentkit/core/execution/resilience.py",
        before="                last if last is not None else breaker.last_error",
        after="                last",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="deferred",
        why="a state transition stops taking the lock, so two threads can both be the probe",
        path="agentkit/core/execution/resilience.py",
        before="        with self._lock:\n            if exc is not None:",
        after="        if True:\n            if exc is not None:",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="deferred",
        why="the lock leaks into equality, so two identical breakers compare unequal",
        path="agentkit/core/execution/resilience.py",
        before="default_factory=threading.Lock, repr=False, compare=False",
        after="default_factory=threading.Lock, repr=False, compare=True",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="breaker",
        why="a permanent probe failure wedges the breaker in half_open forever",
        path="agentkit/core/execution/resilience.py",
        before="    def release_probe(self) -> None:",
        after="    def release_probe(self) -> None:\n        return",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="breaker",
        why="a lost probe is never presumed lost, so a silent caller wedges the gate",
        path="agentkit/core/execution/resilience.py",
        before="            if now - self._probe_started_at >= self.cooldown:",
        after="            if False:",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="breaker",
        why="the neutral release grants health credit, treating a 401 as a recovery",
        path="agentkit/core/execution/resilience.py",
        before=(
            '                return  # nothing in flight'
            ' — CLOSED / OPEN are unaffected\n            self.state = "open"'
        ),
        after=(
            '                return  # nothing in flight'
            ' — CLOSED / OPEN are unaffected\n            self.state = "open"'
            "\n            self._fails = 0"
        ),
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="breaker",
        why="backoff overflows on a large attempt number instead of saturating",
        path="agentkit/core/execution/resilience.py",
        before="min(attempt - 1, 1023)",
        after="attempt - 1",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="prompt",
        why="declared inputs are ignored, so placeholders ship to the model verbatim",
        path="agentkit/agents/prompts/prompt.py",
        before='            text = text.replace("{" + name + "}", str(resolved[name]))',
        after="            pass",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="prompt",
        why="a missing input renders a half-filled prompt instead of refusing",
        path="agentkit/agents/prompts/prompt.py",
        before="        if missing or unexpected:",
        after="        if False:",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="prompt",
        why="str.format is used, so a JSON Schema in a template raises or is eaten",
        path="agentkit/agents/prompts/prompt.py",
        before="        text = self.template\n        for name in self.inputs:",
        after=(
            "        return self.template.format(**values).strip()\n"
            "        text = self.template\n"
            "        for name in self.inputs:"
        ),
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="prompt",
        why="values passed to an input-less prompt are silently ignored",
        path="agentkit/agents/prompts/prompt.py",
        before='            self._reject_undeclared(values, verb="passed to render()")',
        after="            pass",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="a control_response is folded into the turn, putting protocol JSON in the answer",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='                    if payload.get("type") == "control_response":',
        after="                    if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="an interrupted turn reports the CLI's ambiguous error subtype instead",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='                state.stop_reason = "interrupted"',
        after="                pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="Ctrl-C during a CLI run is swallowed into a tidy successful-looking "
        "terminal event instead of propagating — which also hides a test double's "
        "ScriptExhausted, the one exception that must never be caught",
        path="agentkit/models/providers/cli/_common.py",
        before="    if fatal_exc is not None and not isinstance(fatal_exc, Exception):",
        after="    if fatal_exc is not None and isinstance(fatal_exc, Exception):",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="the SESSION half of the re-raise is a separate line in a separate method; "
        "deleting it leaves Ctrl-C during a session turn swallowed, and exactly one "
        "test in the suite notices",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="""            yield StreamEvent("final", usage=state.usage, result=result)
            if should_reraise_cancel:
                raise asyncio.CancelledError()
            _reraise_if_not_an_exception(fatal_exc)""",
        after="""            yield StreamEvent("final", usage=state.usage, result=result)
            if should_reraise_cancel:
                raise asyncio.CancelledError()""",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="json.loads on bytes sniffs a BOM, so a non-UTF-8 diagnostic line raises "
        "UnicodeDecodeError rather than JSONDecodeError; narrowing the catch lets one "
        "bad byte abort the run and bill the completed work at $0.00",
        path="agentkit/models/providers/cli/_common.py",
        before="    except ValueError:  # JSONDecodeError *and* UnicodeDecodeError",
        after="    except json.JSONDecodeError:",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="a replayed process that reports its exit code before wait() makes every "
        "persistent session look dead from turn two",
        path="agentkit/core/testing/fakes/_cli_replay.py",
        before="        return self._invocation.returncode\n\n    async def wait(self) -> int:",
        after="        return self._run.returncode\n\n    async def wait(self) -> int:",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="dropping the newline-less fragment at EOF silently loses a recording's "
        "final result payload — cost, usage and session id vanish while the run still "
        "reports partial=False",
        path="agentkit/core/testing/fakes/_cli_replay.py",
        before="            yield blob[start:]\n            return",
        after="            return",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="sorting the queued stderr chunks as whole tuples lets a tie on the stdout "
        "position fall through to comparing BYTES, so a multi-write traceback is "
        "reassembled in alphabetical order",
        path="agentkit/core/testing/fakes/_cli_replay.py",
        before="sorted(run.interleaved_stderr, key=lambda pair: pair[0])",
        after="sorted(run.interleaved_stderr)",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="a double whose terminate() does not stop the child still reports "
        "stop_reason='cancelled', because cancellation outranks the exit code — so a "
        "leaked subprocess is invisible from the result alone",
        path="agentkit/core/testing/fakes/_cli_replay.py",
        before="        self._invocation.terminated = True",
        after="        self._invocation.terminated = False",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="interrupting an idle session sends a request nobody will ever answer",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if proc is None or proc.returncode is not None or not self._turn_active:",
        after="        if proc is None or proc.returncode is not None:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="a turn that ends first strands the interrupt waiter forever",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='                self._fail_pending("the turn ended before the CLI answered")',
        after="                pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="cancel_queued is sent to a CLI that does not support it, and silently ignored",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='            if not self.supports("interrupt_cancel_queued_v1"):',
        after="            if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="cancel_queued uses the plain interrupt subtype, so queued work still runs",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='            subtype = "interrupt_cancel_queued"',
        after='            subtype = "interrupt"',
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="capabilities land only at the terminal event, too late to feature-detect against",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="                        if delta.init:\n                            self._absorb_init(delta.init)",
        after="                        pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="an inline interrupt hangs the application instead of degrading",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="                payload = await asyncio.wait_for(future, timeout=ack_timeout_s)",
        after="                payload = await future",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="the interrupt flag is set only after the ack, which the turn may end before",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        self._interrupted = True\n        try:",
        after="        try:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a turn reads to EOF like a one-shot drive, so the session hangs forever",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='                    if payload.get("type") == "result":\n                        break',
        after='                    if payload.get("type") == "__never":\n                        break',
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="the prompt is passed as an argv argument too, running a turn nobody asked for",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='        if stream_input:',
        after="        if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="turns are not serialised, so two callers interleave one conversation",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        async with self._lock:",
        after="        if True:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a turn is written into a process that has already exited",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="                if proc.returncode is not None:",
        after="                if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a closed session accepts turns, silently starting a fresh conversation",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="                if self._closed or proc is None:",
        after="                if proc is None:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a CLI that closed mid-turn is not noticed, so the turn reports success",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='                    raise _SessionClosed("the CLI closed its output mid-turn; the session is over")',
        after='                    pass',
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a dead session reads as a spawn failure rather than an ended conversation",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="            if isinstance(fatal_exc, _SessionClosed):",
        after="            if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="per-turn structured output is accepted, silently returning prose",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="                    if schema_requested:",
        after="                    if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a cancelled turn leaves the process alive with half an answer in its context",
        path="agentkit/models/providers/cli/claude/driver.py",
        # The guard this used to anchor on became a multi-line condition that
        # also covers a timeout, so the kill ITSELF is the anchor now: it is
        # the one line whose removal leaves the process running.
        before="                    await _terminate(proc, self._cog.terminate_grace_s)",
        after="                    pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="closing the session kills the CLI instead of ending the conversation",
        path="agentkit/models/providers/cli/claude/driver.py",
        before=(
            "                if proc.stdin is not None and not proc.stdin.is_closing():\n"
            "                    proc.stdin.close()"
        ),
        after="                pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="the completed block is re-emitted, so a UI renders every sentence twice",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="                        None if partial else StreamEvent(\"message_delta\", text=text),",
        after="                        StreamEvent(\"message_delta\", text=text),",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="token deltas also accumulate, doubling AgentResult.output",
        path="agentkit/models/providers/cli/claude/driver.py",
        before=(
            "                yield StreamEvent(\"message_delta\", text=chunk), _EventDelta()\n"
            "        elif dtype == \"thinking_delta\":"
        ),
        after=(
            "                yield StreamEvent(\"message_delta\", text=chunk), _EventDelta(text=chunk)\n"
            "        elif dtype == \"thinking_delta\":"
        ),
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="the partial-messages flag is never passed, so no tokens ever arrive",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='        if self.partial_messages:\n            argv += ["--include-partial-messages"]\n',
        after="",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="a skipped MCP server is discarded with the rest of the init payload",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if state.init:\n            # Startup facts an operator needs",
        after="        if False:\n            # Startup facts an operator needs",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="an empty error list is manufactured, so presence stops being the signal",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='                        if k in payload\n                    },',
        after='                    },',
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="provider retries are invisible, leaving a 40s pause unexplained",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='        if subtype == "api_retry":',
        after="        if False:",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="strict_mcp_config alone silently leaves the session with no MCP servers",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if self.strict_mcp_config and not self.mcp_config:",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="a stable prompt prefix is silently a no-op under a replaced system prompt",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='        if self.stable_prompt_prefix and self.system_prompt_mode == "replace":',
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="a typo'd add_dir reaches the CLI and dies three seconds in",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        missing_dirs = [str(d) for d in self.add_dirs if not Path(d).is_dir()]",
        after="        missing_dirs = []",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="a fallback CHAIN is passed as a Python tuple repr instead of a comma list",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='else ",".join(self.fallback_model)',
        after='else str(self.fallback_model)',
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="bare mode without a credential fails with a message pointing at the wrong fix",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if self.bare:\n            self._warn_if_bare_mode_has_no_credential(env)",
        after="        if False:\n            self._warn_if_bare_mode_has_no_credential(env)",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="the bare-mode warning fires even when settings may carry an apiKeyHelper",
        path="agentkit/models/providers/cli/claude/driver.py",
        before=(
            "        if self.settings is not None:\n"
            "            return\n"
            "        if any(env.get(name) for name in _BARE_CREDENTIAL_ENV):"
        ),
        after="        if any(env.get(name) for name in _BARE_CREDENTIAL_ENV):",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="CLI spend is invisible to every meter again ($50 run, $0.00 ledger)",
        path="agentkit/models/providers/cli/claude/driver.py",
        before=(
            "        charge_error = await _charge_meters("
            "ctx, state.usage, enabled=self.meter_spend)"
        ),
        after="        charge_error = None",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the run's headroom is never handed to the CLI, so nothing stops it mid-flight",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if max_budget_usd is not None:\n            argv += [\"--max-budget-usd\", max_budget_usd]\n",
        after="",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the ORIGINAL ceiling is sent instead of the remaining headroom",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='        return f"{headroom:f}"',
        after='        return f"{budget.ceiling():f}"',
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="an exhausted budget still burns a subprocess spawn to learn it is exhausted",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if headroom <= 0:",
        after="        if False:",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the pre-flight refusal is categorised as a spawn failure, so it reads unresumable",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='            elif type(fatal_exc).__name__ == "MeterExceeded":',
        after="            elif False:",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the per-actor envelope is skipped, the bug ActorBudget already had once",
        # Moved: the two byte-identical `_charge_meters` copies were merged
        # into one shared helper, so this invariant (and the two below) now
        # live in `_cli_common.py` and are exercised through BOTH cognitions.
        path="agentkit/models/providers/cli/_common.py",
        before="    actor = getattr(ctx, \"actor_budget\", None)\n    if actor is not None:",
        after="    actor = None\n    if actor is not None:",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="a ceiling crossed by this very run raises, losing a result already paid for",
        path="agentkit/models/providers/cli/_common.py",
        before=(
            "        except Exception as exc:  # noqa: BLE001 — see docstring\n"
            "            note = f\"{type(exc).__name__}: {exc}\""
        ),
        after="        except Exception:\n            raise",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="meter_spend=False still charges the shared envelope",
        path="agentkit/models/providers/cli/_common.py",
        # `self.meter_spend` arrives as the keyword-only `enabled` now. The
        # `call = ...` line stays in the anchor: without it this is a prefix of
        # `_budget_cap`'s identical guard and would match twice.
        before=(
            "    if not enabled or ctx is None:\n"
            "        return None\n"
            "    call = _CliCall(ctx=ctx)"
        ),
        after="    if ctx is None:\n        return None\n    call = _CliCall(ctx=ctx)",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="agent.output is ignored again, so the CLI never validates its answer",
        path="agentkit/models/providers/cli/claude/driver.py",
        before=(
            "        adapter = getattr(agent, \"_output_adapter\", None)\n"
            "        if adapter is None:\n"
            "            return None"
        ),
        after=(
            "        return None\n"
            "        adapter = getattr(agent, \"_output_adapter\", None)\n"
            "        if adapter is None:"
        ),
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="the validated dict is handed back raw instead of the declared type",
        path="agentkit/models/providers/cli/_common.py",
        before="        return adapter.validate(value), None",
        after="        return value, None",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="a success with no structured_output reads as a clean run",
        path="agentkit/models/providers/cli/claude/driver.py",
        before=(
            "            else:\n"
            "                final_partial = True\n"
            "                if final_stop_reason in (None, \"success\"):"
        ),
        after=(
            "            elif False:\n"
            "                final_partial = True\n"
            "                if final_stop_reason in (None, \"success\"):"
        ),
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="a value the Python type rejects is reported as a successful parse",
        path="agentkit/models/providers/cli/claude/driver.py",
        # `_coerce_structured` returns a `(parsed, failure)` pair now instead
        # of setting a `coercion_error` local; neutralising the check on the
        # failure half is the same break.
        before="                if structured_failure is not None:",
        after="                if False:",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="exhausted structured-output retries are categorised as a generic termination",
        path="agentkit/models/providers/cli/_common.py",
        before="    if reason in invalid_output_reasons:\n        return \"invalid_output\"\n",
        after="",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="the field-level coercion diagnostics are dropped, leaving only '1 error(s)'",
        path="agentkit/models/providers/cli/_common.py",
        # The joined sentence became typed `violations` on
        # `StructuredOutputFailure`; starving the list they are parsed from
        # leaves the same bare "N error(s)" summary in `detail`.
        before='        errors = [str(e) for e in (getattr(exc, "errors", None) or ())]',
        after="        errors = []",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="agent.prompt REPLACES the CLI's system prompt, stripping its tool guidance",
        path="agentkit/models/providers/cli/claude/driver.py",
        # The mode is decided once now and consumed by both the inline and the
        # `--system-prompt-file` branch, so forcing the decision covers the
        # pair — the old anchor could only reach the inline one.
        before='            replace = self.system_prompt_mode == "replace"',
        after="            replace = True",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="a resume id is passed as --session-id, silently starting a fresh session",
        path="agentkit/models/providers/cli/claude/driver.py",
        before='            argv += ["--resume", self.resume_session_id]',
        after='            argv += ["--session-id", self.resume_session_id]',
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="a non-UUID session_id burns a subprocess spawn to learn it is invalid",
        path="agentkit/models/providers/cli/claude/driver.py",
        before=(
            "        if self.session_id is not None:\n"
            "            try:\n"
            "                uuid.UUID(str(self.session_id))"
        ),
        after="        if False:\n            try:\n                uuid.UUID(str(self.session_id))",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="a restricted tool set is silently not passed, leaving every tool available",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if self.tools is not None:",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="tools=() emits a bare --tools flag the CLI rejects",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if self.tools == ():",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliflags",
        why="fork_session without a resume reaches the CLI and errors there",
        path="agentkit/models/providers/cli/claude/driver.py",
        before="        if self.fork_session and not (self.continue_session or self.resume_session_id):",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="a backslash path reaches the backend (traversal on a Windows-hosted one)",
        path="agentkit/tools/filesystem/files.py",
        before='        if "\\\\" in raw:',
        after="        if False:",
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="a NUL byte reaches the backend (C-level path truncation)",
        path="agentkit/tools/filesystem/files.py",
        before='        if "\\x00" in raw:',
        after="        if False:",
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="a sibling directory sharing the root's prefix passes confinement",
        path="agentkit/tools/filesystem/files.py",
        before='        if norm != self.root and not norm.startswith(self.root + "/"):',
        after="        if norm != self.root and not norm.startswith(self.root):",
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="rename confines only its source, becoming the escape hatch",
        path="agentkit/tools/filesystem/files.py",
        before='        return await self._fs.rename(path, self._confine(args.get("new_path")))',
        after='        return await self._fs.rename(path, args.get("new_path"))',
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="filetool",
        why="delete wipes the whole root by prefix in one call",
        path="agentkit/tools/filesystem/files.py",
        before="            if path == self.root:",
        after="            if False:",
        tests=FILETOOL_TESTS,
    ),
    Mutant(
        tag="stopreason",
        why="a CLI subprocess that exited non-zero reads as a deliberate stop",
        path="agentkit/models/providers/cli/_common.py",
        before='    if reason in failure_reasons or (reason is not None and reason.startswith("cli_exit_")):',
        after="    if reason in failure_reasons:",
        tests=STOP_REASON_TESTS,
    ),
    Mutant(
        tag="concurrency",
        why="cooperative cancellation is isolated into a Failure slot again",
        path="agentkit/core/execution/concurrency.py",
        before="            except Cancelled:\n",
        after="            except _NeverRaised:\n",
        tests=CONCURRENCY_TESTS,
    ),
    Mutant(
        tag="budget",
        why="no loop consults the actor envelope, so exhausting it stops nothing",
        path="agentkit/models/providers/cli/_common.py",
        before="    actor = getattr(ctx, \"actor_budget\", None)",
        after="    actor = None",
        tests=("tests/source/unit/agents/test_actor_budget.py",),
    ),
    Mutant(
        tag="hitl",
        why="an expired elicitation is treated as an approval",
        path="agentkit/integrations/mcp/approvals.py",
        before='        return self.kind in ("approve", "modify")',
        after='        return self.kind in ("approve", "modify", "expired")',
        tests=HITL_TESTS,
    ),
    Mutant(
        tag="registry",
        why="an unknown model reports capabilities as present instead of UNKNOWN",
        path="agentkit/models/catalog/registry.py",
        before="        return entry.capabilities if entry is not None else ModelCapabilities()",
        after="        return entry.capabilities if entry is not None else ModelCapabilities(\n"
        "            tools=_Y, structured_output=_Y, native_json_schema=_Y, streaming=_Y, vision=_Y\n"
        "        )",
        tests=REGISTRY_TESTS,
    ),
    Mutant(
        tag="registry",
        why="a missing credential silently returns the fake instead of raising",
        path="agentkit/models/catalog/registry.py",
        before="        if fallback is None:\n            raise hard",
        after="        if False:\n            raise hard",
        tests=REGISTRY_TESTS,
    ),
    Mutant(
        tag="registry",
        why="a declared NO capability stops refusing at bind time",
        path="agentkit/models/catalog/registry.py",
        before="        missing = [n for n in (*requires, *derived) if caps.get(n) is Capability.NO]",
        after="        missing = []",
        tests=REGISTRY_TESTS,
    ),
    Mutant(
        tag="registry",
        why="the downgrade warning fires every time instead of once",
        path="agentkit/models/catalog/registry.py",
        before="        if key in self._warned:\n            return",
        after="        if False:\n            return",
        tests=REGISTRY_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="abstract collections fall back to 'string', telling the model to send a JSON string",
        path="agentkit/tools/declarations/schema.py",
        before="    if origin in _ARRAY_ANNOTATIONS:\n        element = _element_annotation(ann)",
        after="    if origin in (list, tuple, set, frozenset):\n        element = _element_annotation(ann)",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="a TypedDict parameter is advertised as a string again, so the body indexes a str",
        path="agentkit/tools/declarations/schema.py",
        before="    if not is_typeddict(ann):\n        return None",
        after="    if True:\n        return None",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="a TypedDict's total=False keys are advertised as required, inverting the contract",
        path="agentkit/tools/declarations/schema.py",
        before='            "required": [key for key in hints if key in required],',
        after='            "required": list(hints),',
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="money",
        why="max_concurrency=0 constructs again, and the first fan-out waits on it forever",
        path="agentkit/tools/execution/policy.py",
        before="        if self.max_concurrency < 1:",
        after="        if False:",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="containers stop describing items, so list[Unit] is an untyped array again",
        path="agentkit/tools/declarations/schema.py",
        before="        return {\"type\": \"array\", \"items\": _json_type(element)}",
        after="        return {\"type\": \"array\"}",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="a heterogeneous tuple claims one element type, advertising items that are a lie",
        path="agentkit/tools/declarations/schema.py",
        before="        if len(args) == 2 and args[1] is Ellipsis:\n            return None if args[0] is Any else args[0]\n        return None",
        after="        return None if args[0] is Any else args[0]",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="elements stop being coerced, so the body gets strings where items promised members",
        path="agentkit/tools/declarations/schema.py",
        before="    if origin in _ARRAY_ANNOTATIONS:\n        return _sequence_coercer(ann, origin)",
        after="    if origin in _ARRAY_ANNOTATIONS:\n        return None",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="set/tuple annotations receive a list again, so the body gets a container it did not ask for",
        path="agentkit/tools/declarations/schema.py",
        before="_REBUILT_CONTAINERS: dict[Any, Any] = {set: set, frozenset: frozenset, tuple: tuple}",
        after="_REBUILT_CONTAINERS: dict[Any, Any] = {}",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="Optional collapses to bare X again, so a required X|None cannot be sent as null",
        path="agentkit/tools/declarations/schema.py",
        before="        nullable = len(real) != len(args)",
        after="        nullable = False",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="*args is silently dropped again, leaving a parameter that can never be filled",
        path="agentkit/tools/declarations/schema.py",
        before="        if p.kind is p.VAR_POSITIONAL:",
        after="        if False:",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="codex",
        why="the cached prefix is not subtracted, so every cached token is billed at "
        "full input rate — and a Codex session is mostly cache, so the estimate and "
        "every meter are wrong by roughly the whole prompt on every turn",
        path="agentkit/models/providers/cli/codex/driver.py",
        before="    return max(0, total_input - cached), cached",
        after="    return total_input, cached",
        tests=CODEX_EVENT_TESTS,
    ),
    Mutant(
        tag="codex",
        why="a completed item re-emits text its updates already streamed, so a UI with "
        "a cursor in it shows every sentence twice while the final answer stays right",
        path="agentkit/models/providers/cli/codex/driver.py",
        before="        already = self.emitted.get(key, 0)",
        after="        already = 0",
        tests=CODEX_EVENT_TESTS,
    ),
    Mutant(
        tag="codex",
        why="an item's text is folded on every phase, so an updated-then-completed "
        "message lands in the answer twice",
        path="agentkit/models/providers/cli/codex/driver.py",
        before=(
            '        return [(ev, _EventDelta(text=text, message_id=item_id or None, '
            'message_done=phase == "completed"))]'
        ),
        after='        return [(ev, _EventDelta(text=text, message_id=item_id or None, message_done=True))]',
        tests=CODEX_EVENT_TESTS,
    ),
    Mutant(
        tag="codex",
        why="the legacy vocabulary stops being parsed, so an older `codex` produces a "
        "run that exits 0, emits no events and returns an empty answer with no stop "
        "reason — silent, plausible and wrong, because forward-compat is 'don't crash'",
        path="agentkit/models/providers/cli/codex/driver.py",
        before='    if "msg" in payload and isinstance(payload.get("msg"), dict):',
        after="    if False:",
        tests=CODEX_EVENT_TESTS,
    ),
    Mutant(
        tag="codex",
        why="a paired tool item emits its call event twice, so an observer counting "
        "tool calls doubles every shell command",
        path="agentkit/models/providers/cli/codex/driver.py",
        before='    if log.first_call(f"call:{item_id}"):',
        after="    if True:",
        tests=CODEX_TESTS,
    ),
    Mutant(
        tag="codex",
        why="an unset sandbox reports no capabilities, so RunPolicy's lethal-trifecta "
        "gate goes blind on the commonest wiring there is — a cognition nobody "
        "configured, which the CLI still runs read-only",
        path="agentkit/models/providers/cli/codex/driver.py",
        before="        return self.sandbox or _DEFAULT_SANDBOX",
        after='        return self.sandbox or "danger-full-access"',
        tests=CODEX_CAPS_TESTS,
    ),
    Mutant(
        tag="codex",
        why="web search stops contributing untrusted_content, so a read-only sandbox "
        "with --search — the most innocuous-looking configuration Codex has — is no "
        "longer refused as the lethal trifecta",
        path="agentkit/models/providers/cli/codex/driver.py",
        before='            tags.update(("untrusted_content", "egress"))',
        after='            tags.update(("egress",))',
        tests=CODEX_CAPS_TESTS,
    ),
    Mutant(
        tag="codex",
        why="the subcommand is emitted before the options instead of after them, which "
        "is the one argv layout `codex exec` cannot parse for both its global and its "
        "parent-only flags",
        path="agentkit/models/providers/cli/codex/driver.py",
        # `--search` is the concrete global flag this is about: it is rejected
        # by `codex exec` and accepted by `codex`, so hoisting the subcommand
        # ahead of it is the exact argv layout the source comment warns about.
        before=(
            "        argv: list[str] = [self.codex_bin]\n"
            "        if self.web_search:\n"
            '            argv += ["--search"]\n'
            '        argv += ["exec"]'
        ),
        after=(
            '        argv: list[str] = [self.codex_bin, "exec"]\n'
            "        if self.web_search:\n"
            '            argv += ["--search"]'
        ),
        tests=CODEX_FLAG_TESTS,
    ),
    Mutant(
        tag="codex",
        why="a config override value is interpolated rather than serialised, so a "
        "string arrives unquoted (a TOML parse error), a bool arrives quoted (the "
        "string 'True'), and a Windows path's backslashes end the value early",
        path="agentkit/models/providers/cli/codex/driver.py",
        before="    if isinstance(value, str):\n        return json.dumps(value)",
        after="    if isinstance(value, str):\n        return value",
        tests=CODEX_FLAG_TESTS,
    ),
    Mutant(
        tag="codex",
        why="a session forgets the thread it started, so every turn opens a fresh "
        "conversation — the model answers each question competently with no memory of "
        "the last, which reads as a bad model rather than a bug",
        path="agentkit/models/providers/cli/codex/driver.py",
        before='            resume = ("resume", self.session_id) if self.session_id else ()',
        after="            resume = ()",
        tests=CODEX_SESSION_TESTS,
    ),
    Mutant(
        tag="codex",
        why="a session cannot be an agent's cognition again: Agent._span_attrs reads "
        "cognition.name on every run, so agent.run raises AttributeError before "
        "reaching the CLI — the bug this attribute was added to fix",
        path="agentkit/models/providers/cli/codex/driver.py",
        before='        return f"{self._cog.name}_session"',
        after='        raise AttributeError("name")',
        tests=CODEX_SESSION_TESTS,
    ),
    Mutant(
        tag="codex",
        why="a final message that is not the JSON --output-schema asked for is reported "
        "as a clean success with parsed=None, so a caller who declared output=Invoice "
        "reads the silence as if the type had simply not been wired",
        path="agentkit/models/providers/cli/codex/driver.py",
        before="            if decode_error is not None:\n                final_partial = True",
        after="            if decode_error is not None:\n                final_partial = False",
        tests=CODEX_SCHEMA_TESTS,
    ),
    Mutant(
        tag="codex",
        why="the schema's scratch directory outlives the run, so every typed run leaks "
        "a file naming the caller's data model",
        path="agentkit/models/providers/cli/codex/driver.py",
        before="            if scratch is not None:\n                shutil.rmtree(scratch, ignore_errors=True)",
        after="            if scratch is None:\n                shutil.rmtree(scratch, ignore_errors=True)",
        tests=CODEX_SCHEMA_TESTS,
    ),
    Mutant(
        tag="codex",
        why="the CLI's spend never reaches the framework's meters, which is the silent "
        "$0.00 ledger the Claude cognition shipped once — the middleware that would "
        "have charged it never runs, because the Invoker is bypassed",
        path="agentkit/models/providers/cli/codex/driver.py",
        before=(
            "        charge_error = await _charge_meters("
            "ctx, usage, enabled=self.meter_spend)"
        ),
        after="        charge_error = None",
        tests=CODEX_BUDGET_TESTS,
    ),
    Mutant(
        tag="codex",
        why="an exhausted budget spawns anyway, and the resumable budget_exhausted stop "
        "reason becomes a completed run that overspent",
        path="agentkit/models/providers/cli/codex/driver.py",
        # Dedented: the pre-flight moved into the `_prepare` closure so a
        # throw here still reaches the terminal event.
        before="            self._refuse_if_budget_exhausted(ctx)",
        after="            pass",
        tests=CODEX_BUDGET_TESTS,
    ),
    Mutant(
        tag="codex",
        why="a computed cost is presented as if the CLI reported one, so a caller bills "
        "against a price-table estimate instead of a provider billing receipt",
        path="agentkit/models/providers/cli/codex/driver.py",
        before='            state.cost_source = CostSource.HOST_PRICING if self.pricing is not None else CostSource.PRICE_TABLE',
        after='            state.cost_source = CostSource.REPORTED',
        tests=CODEX_BUDGET_TESTS,
    ),
    Mutant(
        tag="fakecodex",
        why="the codex double replays past the end of its recording, so a session that "
        "spawns twice for one turn — a retry nobody asked for — is answered with a "
        "stable, plausible, wrong result instead of raising",
        path="agentkit/core/testing/fakes/codex_cli.py",
        before="            if not self._repeat_last or not self._runs:",
        after="            if self._runs:",
        tests=FAKE_CODEX_TESTS,
    ),
    Mutant(
        tag="fakecodex",
        why="codex_turn emits its terminal payload first, so every test asserting on a "
        "stream asserts against an order the binary never produces",
        path="agentkit/core/testing/fakes/codex_cli.py",
        before="        out.append(completed)",
        after="        out.insert(0, completed)",
        tests=FAKE_CODEX_TESTS,
    ),
    Mutant(
        tag="calls",
        why="durable checkpoints drop first-class message context metadata",
        path="agentkit/core/serialization/messages.py",
        before='        "metadata": thaw(message.metadata),',
        after='        "metadata": {},',
        tests=("tests/source/conformance/test_persistence.py",),
    ),
)


# ── runner ───────────────────────────────────────────────────────────────────


# A mutant can make the suite HANG rather than fail — removing a timeout is a
# perfectly good mutation, and the tests that exercise it then wait forever.
# A hang is a finding, not a reason to block the run.
_PER_MUTANT_TIMEOUT_S = 120


# pytest signals "I could not even collect that" with a NON-ZERO exit: 4 for a
# usage error (a path that does not exist) and 5 for nothing collected. Reading
# any non-zero exit as "the suite caught the mutant" is how an entry whose test
# file was renamed away scores a silent kill — not one assertion ran, and the
# invariant is completely unguarded while the gate reports green. The
# reorganisation renamed 34 test files out from under this catalogue and the
# whole job still passed, which is exactly the failure this distinguishes.
_PYTEST_USAGE_ERROR = 4
_PYTEST_NOTHING_COLLECTED = 5

UNRUNNABLE = "unrunnable"


def _run_tests(tests: tuple[str, ...]) -> bool | None | str:
    """``False`` = tests failed (mutant killed), ``True`` = tests passed
    (survived), ``None`` = the run timed out, ``UNRUNNABLE`` = pytest could not
    run the named tests at all, so the result proves nothing either way."""
    if [t for t in tests if not (ROOT / t.split("::")[0]).exists()]:
        return UNRUNNABLE
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
    if proc.returncode in (_PYTEST_USAGE_ERROR, _PYTEST_NOTHING_COLLECTED):
        return UNRUNNABLE
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
    # Source anchors are checked for the parked set too. They are the thing
    # that keeps UNENFORCED restorable: an entry whose code has since been
    # rewritten again is no longer a backlog item, it is noise, and this is
    # where that shows up instead of at the bottom of a file nobody reads.
    for m in (*MUTANTS, *UNENFORCED):
        target = ROOT / m.path
        if not target.exists():
            problems.append(f"{m.path}: file does not exist (renamed or removed?)")
            continue
        count = target.read_text().count(m.before)
        if count != 1:
            problems.append(f"{m.path}: anchor for {m.why!r} appears {count}x, expected 1")
        # A live source anchor with a dead test path is the worse of the two
        # failures: the mutant applies cleanly, pytest exits non-zero because
        # the file is missing, and the run scores it killed. Checking it here
        # is what makes the staleness visible in `--verify` rather than at the
        # bottom of a green log.
        if m in UNENFORCED:
            # Parked precisely BECAUSE its test file is gone; requiring one
            # here would make the gate permanently red over work that is
            # tracked rather than pending.
            continue
        for t in m.tests:
            if not (ROOT / t.split("::")[0]).exists():
                problems.append(f"{m.path}: test path does not exist: {t}")
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
        "--unenforced",
        action="store_true",
        help="list the invariants whose tests the reorganisation deleted, and exit",
    )
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

    if args.unenforced:
        print(
            f"{len(UNENFORCED)} catalogued invariants have no test enforcing them.\n"
            "Their source anchors are live; each becomes a working entry again as\n"
            "soon as a test covers it (move it into MUTANTS and point `tests` at\n"
            "the file).\n"
        )
        by_file: dict[str, list[Mutant]] = {}
        for mut in UNENFORCED:
            by_file.setdefault(mut.path, []).append(mut)
        for path in sorted(by_file):
            print(f"  {path}  ({len(by_file[path])})")
            for mut in by_file[path]:
                print(f"      [{mut.tag}] {mut.why.splitlines()[0][:110]}")
        return 0

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
                # Drop the compiled bytecode for the file we just rewrote.
                #
                # CPython invalidates a ``.pyc`` on (source mtime in WHOLE
                # SECONDS, source size) — so two consecutive mutants that
                # change a file by the same number of bytes within the same
                # second are indistinguishable to it, and the second one runs
                # against the FIRST one's bytecode. Not hypothetical: mutants
                # 63/64/66 all rename a ``def`` in ``providers/base.py`` and
                # all grow it by exactly 7 bytes, so ``_sse_decode`` was
                # reported SURVIVED in a full run and killed when run alone.
                #
                # It lies in both directions — a mutant can also look KILLED
                # because a neighbour's stale bytecode broke the tests — and
                # it is timing-dependent, so it comes and goes between runs.
                shutil.rmtree(target.parent / "__pycache__", ignore_errors=True)
                survived = _run_tests(m.tests)
            finally:
                target.write_text(original)
            mark = (
                "UNRUNNABLE"
                if survived == UNRUNNABLE
                else {True: "SURVIVED", False: "killed  ", None: "TIMEOUT "}[survived]
            )
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
