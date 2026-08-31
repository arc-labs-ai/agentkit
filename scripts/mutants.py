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
    # The backup FILENAME is a LOSSY encoding of the path: `_backup_path` maps
    # "/" to "__", so `agentkit/middlewares/__init__.py` is stored as
    # `agentkit__middlewares____init__.py`, and decoding "__" back to "/" gives
    # `agentkit/middlewares//init/.py` — a directory that does not exist. Every
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
AUDIT2_TESTS = ("tests/middlewares/test_audit_records_failures.py",)
STORELOCK_TESTS = ("tests/adapters/test_stores.py",)
EST_TESTS = (
    "tests/context/test_context_tokens.py",
    "tests/capabilities/test_compaction_strategies.py",
)
OBS_TESTS = ("tests/adapters/test_observer_adapters.py",)
STORE2_TESTS = ("tests/adapters/test_stores.py",)
VEC_TESTS = ("tests/adapters/test_vector_adapters.py",)
PROV_TESTS = ("tests/adapters/test_provider_error_classification.py",)
SSE_TESTS = ("tests/adapters/test_provider_sse_and_tool_calls.py",)
MAP_TESTS = ("tests/adapters/test_callable_llm_mapping.py",)
TOK_TESTS = (
    "tests/context/test_context_tokens.py",
    "tests/capabilities/test_compaction_strategies.py",
)
MCPHTTP_TESTS = ("tests/integrations/mcp/test_http_transport.py",)
SCHEMA2_TESTS = (
    "tests/capabilities/test_output_schema_dataclass.py",
    "tests/capabilities/test_output_schema_attrs.py",
    "tests/capabilities/test_output_schema_pydantic.py",
)
MW_MEMO_TESTS = ("tests/middlewares/test_memoize_tool_identity.py",)
MW_METER_TESTS = ("tests/middlewares/test_meter_stream_and_cache.py",)
MW_SEM_TESTS = ("tests/middlewares/test_semantic_memoize_tool_turns.py",)
MW_TRACE_TESTS = ("tests/middlewares/test_tracing_records_failures.py",)
MW_COMPACT_TESTS = ("tests/middlewares/test_compaction_broken_tracer.py",)
COMPACTION_TESTS = ("tests/capabilities/test_compaction_strategies.py",)
TOOLCOERCE_TESTS = ("tests/tools/test_tool_argument_coercion.py",)
MW_AUDIT_TESTS = ("tests/middlewares/test_audit_records_failures.py",)
STREAMS_TESTS = ("tests/kernel/test_streams.py",)
BREAKER_TESTS = ("tests/kernel/test_kernel.py",)
CONC_TESTS = ("tests/kernel/test_concurrency.py",)
WF_TESTS = ("tests/agents/test_workflow.py",)
CEILING_TESTS = ("tests/agents/test_ceiling_propagation.py",)
SURFACE_TESTS = ("tests/meta/test_public_surface.py",)
PROMPT_TESTS = ("tests/test_prompt_render.py",)
AGENT_PROMPT_TESTS = ("tests/agents/test_agent_prompt_binding.py",)
APPROVAL_TESTS = ("tests/integrations/mcp/test_approvals.py",)
CLI_SESSION_TESTS = ("tests/agents/cognition/test_claude_cli_session.py",)
CLI_STREAM_TESTS = ("tests/agents/cognition/test_claude_cli_streaming.py",)
CLI_BUDGET_TESTS = ("tests/agents/cognition/test_claude_cli_budget.py",)
CLI_SCHEMA_TESTS = ("tests/agents/cognition/test_claude_cli_structured.py",)
CLI_FLAG_TESTS = ("tests/agents/cognition/test_claude_cli_flags.py",)
FAKE_CLI_TESTS = ("tests/testing/test_fake_claude_cli.py",)
MEM_DEDUPE_TESTS = (
    "tests/memory/test_composite_memory_dedupe.py",
    "tests/memory/test_memory_item.py",
)
MEM_VECTOR_TESTS = ("tests/memory/test_vector_memory.py",)
WORKFLOW_MAP_TESTS = ("tests/agents/test_workflow_map.py",)
STORE_PRIM_TESTS = (
    "tests/adapters/test_store_primitives.py",
    "tests/adapters/test_stores.py",
    "tests/meta/test_protocol_conformance.py",
)
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

MEMOIZE_TESTS = (
    "tests/middlewares/test_memoize_side_effects.py",
    "tests/middlewares/test_memoize_chat_key_tool_fields.py",
)
SIGNAL_TESTS = ("tests/agents/test_signal_immutability.py",)
BUS_TESTS = ("tests/runtime/test_event_bus_close_race.py",)
PARSED_TESTS = (
    "tests/kernel/test_result_to_stream_parity.py",
    "tests/middlewares/test_memoize_preserves_parsed.py",
)
MAPPING_TESTS = ("tests/adapters/test_callable_llm_mapping.py",)
PORTS_TESTS = ("tests/kernel/test_value_type_hashability.py",)
FROZEN_TESTS = ("tests/kernel/test_frozen_payloads.py",)
FROZENVAL_TESTS = ("tests/kernel/test_frozen_value_payloads.py",)
FROZENREQ_TESTS = ("tests/kernel/test_frozen_request_payloads.py",)
KERNELTYPE_TESTS = ("tests/kernel/test_kernel.py",)
RESULT_TESTS = ("tests/agents/test_agents.py",)
MEMITEM_TESTS = ("tests/memory/test_memory_item.py",)
SKILL_TESTS = ("tests/skills/test_skill.py",)
CTXDIFF_TESTS = ("tests/context/test_context_working.py",)
VALUETYPE_TESTS = (
    "tests/kernel/test_kernel.py",
    "tests/context/test_context_working.py",
    "tests/meta/test_value_type_invariants.py",
)
REPLAY_TESTS = ("tests/adapters/test_replay_file.py",)

MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        tag="approvals",
        why="auto_allow_when can BROADEN instead of only narrowing, becoming a second way in",
        path="agentkit/integrations/mcp/approvals.py",
        before=(
            "        if tool_name in self.auto_allow and self._arguments_are_auto_allowed(\n"
            "            tool_name, arguments\n"
            "        ):"
        ),
        after=(
            "        if tool_name in self.auto_allow or self._arguments_are_auto_allowed(\n"
            "            tool_name, arguments\n"
            "        ):"
        ),
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="frozen",
        why="deep_freeze goes shallow, so cp.state['a']['b'] = 1 rewrites a durable record again",
        path="agentkit/kernel/_frozen.py",
        before="    if isinstance(value, dict):\n        return FrozenDict({k: deep_freeze(v) for k, v in value.items()})",
        after="    if isinstance(value, dict):\n        return FrozenDict(value)",
        tests=FROZEN_TESTS,
    ),
    Mutant(
        tag="frozen",
        why="FrozenDict loses __reduce__, so deepcopy and pickle break at the checkpoint seam",
        path="agentkit/kernel/_frozen.py",
        before="        return (FrozenDict, (dict(self),))",
        after="        return (dict, (dict(self),))",
        tests=FROZEN_TESTS,
    ),
    Mutant(
        tag="frozen",
        why="deep_freeze stops copying, so the caller keeps a live handle on what they handed over",
        path="agentkit/kernel/_frozen.py",
        before="    if isinstance(value, list):\n        return FrozenList([deep_freeze(v) for v in value])",
        after="    if isinstance(value, list):\n        return FrozenList(value)",
        tests=FROZEN_TESTS,
    ),
    Mutant(
        tag="frozen",
        why="Checkpoint.state stops being frozen, so a persisted record is rewritable in memory",
        path="agentkit/kernel/ports.py",
        before='        object.__setattr__(self, "state", deep_freeze(self.state))',
        after="        pass",
        tests=FROZENVAL_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="AgentResult hashes free-form evals, so the framework's most-returned type breaks again",
        path="agentkit/agents/result.py",
        before="        return hash((self.output, self.usage, self.partial, self.prompt_version, self.stop_reason))",
        after="        return hash((self.output, self.usage, self.partial, str(self.evals)))",
        tests=RESULT_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="WorkflowResult keys on insertion ORDER, so two equal results hash differently",
        path="agentkit/agents/result.py",
        before="        return hash((frozenset(self.outputs), self.usage, self.steps, self.stop_reason))",
        after="        return hash((tuple(self.outputs), self.usage, self.steps, self.stop_reason))",
        tests=RESULT_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="MemoryItem folds backend metadata into identity, so one passage lands in two buckets",
        path="agentkit/memory/base.py",
        before="        return hash((self.content, self.source, self.score))",
        after="        return hash((self.content, self.source, self.score, str(self.metadata)))",
        tests=MEMITEM_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="Skill hashes its cognition, which is a mutable dataclass with __hash__ = None",
        path="agentkit/skills/skill.py",
        before="        return hash((self.name, self.description, self.prompt, self.model))",
        after="        return hash((self.name, self.description, self.prompt, self.model, self.cognition))",
        tests=SKILL_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ContextDiff keys on insertion ORDER, breaking the hash invariant for equal diffs",
        path="agentkit/context/context.py",
        before="                frozenset(self.scratchpad_changes),",
        after="                tuple(self.scratchpad_changes),",
        tests=CTXDIFF_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="Checkpoint hashes its mutable state, so a durable record is unhashable again",
        path="agentkit/kernel/ports.py",
        before="        return hash((self.run_id, self.version))",
        after="        return hash((self.run_id, self.version, tuple(self.state)))",
        tests=PORTS_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="SearchHit folds query-dependent score into identity, so one document lands in two buckets",
        path="agentkit/kernel/ports.py",
        before="        return hash((self.url, self.title))",
        after="        return hash((self.url, self.title, self.score))",
        tests=PORTS_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="FetchResponse hashes the body, making hash O(page size) on every set insert",
        path="agentkit/kernel/ports.py",
        before="        return hash((self.url, self.status, self.content_type, self.fetched_at))",
        after="        return hash((self.url, self.status, self.content_type, self.fetched_at, self.body))",
        tests=PORTS_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="Observation hashes its payload, so it is unhashable the moment it carries a dict",
        path="agentkit/kernel/observation.py",
        before="        return hash((self.run_id, self.agent, self.kind))",
        after="        return hash((self.run_id, self.agent, self.kind, self.payload))",
        tests=PORTS_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ToolSchema hashes the JSON Schema body, so every non-empty schema is unhashable",
        path="agentkit/kernel/types.py",
        before="        return hash((self.name, self.description))",
        after="        return hash((self.name, self.description, str(self.parameters)))",
        tests=KERNELTYPE_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ChatRequest hashes the whole transcript, making the hash O(turns) in an agent loop",
        path="agentkit/kernel/types.py",
        before="        return hash((self.model, self.temperature, self.max_tokens, len(self.messages)))",
        after="        return hash((self.model, self.temperature, self.max_tokens, tuple(self.messages)))",
        tests=KERNELTYPE_TESTS,
    ),
    Mutant(
        tag="hashable",
        why="ToolRequest hashes its decoded-JSON arguments, so a nested payload is unhashable",
        path="agentkit/kernel/types.py",
        before="        return hash((self.name, self.side_effecting, self.url_arg))",
        after="        return hash((self.name, self.side_effecting, self.url_arg, str(self.arguments)))",
        tests=KERNELTYPE_TESTS,
    ),
    Mutant(
        tag="valuetype",
        why="ToolCall is unhashable again, so union merge dies on every tool-using agent",
        path="agentkit/kernel/types.py",
        before="        return hash((self.id, self.name))",
        after="        return hash(self.arguments)",
        tests=VALUETYPE_TESTS,
    ),
    Mutant(
        tag="valuetype",
        why="ToolCall.arguments stops being frozen, so a tool call's payload is editable again",
        path="agentkit/kernel/types.py",
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
        path="agentkit/context/context.py",
        before="        return hash((self.prefix, self.messages, tuple(k for k, _ in self.scratchpad)))",
        after="        return hash((self.prefix, self.messages, self.scratchpad))",
        tests=VALUETYPE_TESTS,
    ),
    Mutant(
        tag="parsed",
        why="a replayed/cached chat result loses its typed object again",
        path="agentkit/kernel/middleware.py",
        before='            parsed=getattr(result, "parsed", None),',
        after="",
        tests=PARSED_TESTS,
    ),
    Mutant(
        tag="parsed",
        why="CallableLLM.stream drops parsed, so chat() and stream() disagree on the same call",
        path="agentkit/adapters/llm/_mapping.py",
        before="        parsed=res.parsed,\n",
        after="",
        tests=MAPPING_TESTS,
    ),
    Mutant(
        tag="bus",
        why="`closed` goes back to being dead state, so a mid-attach subscriber hangs forever",
        path="agentkit/runtime/event_bus.py",
        before="            attached = not channel.closed",
        after="            attached = True",
        tests=BUS_TESTS,
    ),
    Mutant(
        tag="bus",
        why="a late subscriber manufactures a live channel for a closed id and waits on a sentinel nobody sends",
        path="agentkit/runtime/event_bus.py",
        before="            if not reopen and stream_id in self._closed_ids:",
        after="            if False:",
        tests=BUS_TESTS,
    ),
    Mutant(
        tag="bus",
        why="publish stops reopening, so a legitimately reused stream id is poisoned forever",
        path="agentkit/runtime/event_bus.py",
        before="            self._forget_closed(stream_id)",
        after="            pass",
        tests=BUS_TESTS,
    ),
    Mutant(
        tag="bus",
        why="the closed-id set stops mirroring the deque's eviction, so it grows without bound",
        path="agentkit/runtime/event_bus.py",
        before="            self._closed_ids.discard(self._closed_order[0])",
        after="            pass",
        tests=BUS_TESTS,
    ),
    Mutant(
        tag="bus",
        why="re-closing one id in a loop evicts every other tombstone behind it",
        path="agentkit/runtime/event_bus.py",
        before="        if self._closed_capacity == 0 or stream_id in self._closed_ids:",
        after="        if self._closed_capacity == 0:",
        tests=BUS_TESTS,
    ),
    Mutant(
        tag="signals",
        why="a signal payload stops being copied, so the sender keeps editing what it already emitted",
        path="agentkit/agents/control/signals.py",
        before="        return deep_freeze(dict(value))",
        after="        return value",
        tests=SIGNAL_TESTS,
    ),
    Mutant(
        tag="signals",
        why="a bare str payload silently explodes into one option per character",
        path="agentkit/agents/control/signals.py",
        before="    if isinstance(value, (str, bytes)):",
        after="    if False:",
        tests=SIGNAL_TESTS,
    ),
    Mutant(
        tag="signals",
        why="an opaque payload re-enters the hash, so a signal is unhashable whenever it carries one",
        path="agentkit/agents/control/signals.py",
        before="    new_state: StateT | None = field(default=None, hash=False)",
        after="    new_state: StateT | None = field(default=None)",
        tests=SIGNAL_TESTS,
    ),
    Mutant(
        tag="signals",
        why="a signal mapping payload stops being frozen, so an audited used_cost is rewritable",
        path="agentkit/agents/control/signals.py",
        before="        return deep_freeze(dict(value))",
        after="        return dict(value)",
        tests=SIGNAL_TESTS,
    ),
    Mutant(
        tag="replay",
        why="a wrong-shape JSON document escapes get() as a TypeError instead of reading as a miss",
        path="agentkit/adapters/replay/file.py",
        before="            self._failed_reads += 1\n            self._warn_once(\n                \"decode-failed\",",
        after="            raise",
        tests=REPLAY_TESTS,
    ),
    Mutant(
        tag="replay",
        why="a failed rename leaves one orphaned .tmp per call in the operator's data directory",
        path="agentkit/adapters/replay/file.py",
        before="            tmp_path.unlink(missing_ok=True)\n            raise",
        after="            raise",
        tests=REPLAY_TESTS,
    ),
    Mutant(
        tag="regress",
        why="an unbound prompt stops being refused at construction and dies mid-run instead",
        path="agentkit/agents/agent.py",
        before="        self.check_prompt()",
        after="        pass",
        tests=AGENT_PROMPT_TESTS,
    ),
    Mutant(
        tag="regress",
        why="check_prompt reports every declared input instead of only the unbound ones",
        path="agentkit/agents/agent.py",
        before="        missing = sorted(set(p.inputs) - set(p.bound))",
        after="        missing = sorted(p.inputs)",
        tests=AGENT_PROMPT_TESTS,
    ),
    Mutant(
        tag="regress",
        why="memoize() caches a SIDE-EFFECTING tool again — the second send_email is faked",
        path="agentkit/middlewares/memoize.py",
        before="        return allow_side_effects or not _side_effecting(call)",
        after="        return True",
        tests=MEMOIZE_TESTS,
    ),
    Mutant(
        tag="regress",
        why="_side_effecting reads only the REQUEST flag, missing every positionally-built ToolRequest",
        path="agentkit/middlewares/memoize.py",
        before='    return bool(getattr(getattr(r, "tool", None), "side_effecting", False))',
        after="    return False",
        tests=MEMOIZE_TESTS,
    ),
    Mutant(
        tag="regress",
        why="idempotent() loses its opt-in, so an at-least-once retry re-fires the mutation",
        path="agentkit/middlewares/memoize.py",
        before="    return memoize(key=key, store=store, when=_side_effecting, allow_side_effects=True)",
        after="    return memoize(key=key, store=store, when=_side_effecting)",
        tests=MEMOIZE_TESTS,
    ),
    Mutant(
        tag="regress",
        why="the chat key drops tool_calls, so two ReAct branches are served each other's answer",
        path="agentkit/middlewares/memoize.py",
        before="def _message_identity(m: Any) -> dict[str, Any]:",
        after='def _message_identity(m: Any) -> dict[str, Any]:\n    return {"role": m.role, "content": m.content}',
        tests=MEMOIZE_TESTS,
    ),
    Mutant(
        tag="regress",
        why="Prompt stops being hashable, breaking every dict/set/lru_cache holding one",
        path="agentkit/prompts/prompt.py",
        before="        return hash((self.id, self.version, self.template, self.inputs))",
        after="        return hash(self.bound)",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="regress",
        why="Prompt.bound stops being frozen, so a caller edits a bound prompt after the fact",
        path="agentkit/prompts/prompt.py",
        before="        frozen = deep_freeze(dict(self.bound))",
        after="        frozen = dict(self.bound)",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="signals",
        why="a signal SEQUENCE payload stops freezing its elements, so nested dicts stay editable",
        path="agentkit/agents/control/signals.py",
        before="    return tuple(deep_freeze(v) for v in value)",
        after="    return tuple(value)",
        tests=SIGNAL_TESTS,
    ),
    Mutant(
        tag="regress",
        why="the retry backoff becomes dead code again, turning retries into a hot loop",
        path="agentkit/kernel/resilience.py",
        before="            await asleep(backoff_delay(attempt, rng=rng))",
        after="            pass",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="regress",
        why="release_probe stops restarting the cooldown, so the breaker never brakes",
        path="agentkit/kernel/resilience.py",
        before="            self.state = \"open\"\n            self._opened_at = self.clock()",
        after="            self.state = \"open\"",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="regress",
        why="a refused call loses the failure that opened the breaker",
        path="agentkit/kernel/resilience.py",
        before="                last if last is not None else breaker.last_error",
        after="                last",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="schema",
        why="tuple containers fall through unremapped, so nested models keep the wrong alias",
        path="agentkit/capabilities/output_schema/pydantic_adapter.py",
        before="        dumped, (list, tuple)",
        after="        dumped, (list,)",
        tests=SCHEMA2_TESTS,
    ),
    Mutant(
        tag="schema",
        why="the walk stops recursing, fixing only the top level of a nested model",
        path="agentkit/capabilities/output_schema/pydantic_adapter.py",
        before="            rename.get(k, k): _to_validation_keys(children.get(k), v) for k, v in dumped.items()",
        after="            rename.get(k, k): v for k, v in dumped.items()",
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
    # ── the deferred design items ───────────────────────────────────────────
    Mutant(
        tag="deferred",
        why="a caller's summariser raising takes down the run it was observing",
        path="agentkit/adapters/observer/cadence.py",
        before="            except Exception as exc:  # noqa: BLE001 — see above",
        after="            except _NeverRaisedByAnything as exc:",
        tests=OBS_TESTS,
    ),
    Mutant(
        tag="deferred",
        why="the containment widens to BaseException, so a summariser can ignore cancellation",
        path="agentkit/adapters/observer/cadence.py",
        before="            except Exception as exc:  # noqa: BLE001 — see above",
        after="            except BaseException as exc:",
        tests=OBS_TESTS,
    ),
    Mutant(
        tag="deferred",
        why="dropped batches are not counted, so a partial rollup looks complete",
        path="agentkit/adapters/observer/cadence.py",
        before="                self._dropped += len(buf)",
        after="                pass",
        tests=OBS_TESTS,
    ),
    Mutant(
        tag="deferred",
        why="a state transition stops taking the lock, so two threads can both be the probe",
        path="agentkit/kernel/resilience.py",
        before="        with self._lock:\n            if exc is not None:",
        after="        if True:\n            if exc is not None:",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="deferred",
        why="the lock leaks into equality, so two identical breakers compare unequal",
        path="agentkit/kernel/resilience.py",
        before="default_factory=threading.Lock, repr=False, compare=False",
        after="default_factory=threading.Lock, repr=False, compare=True",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="deferred",
        why="a started wave is abandoned mid-flight, dropping siblings",
        path="agentkit/agents/workflow.py",
        before="                if steps >= self.max_steps:",
        after="                if steps >= self.max_steps or len(ready) > 1:",
        tests=WF_TESTS,
    ),
    # ── the leftovers ───────────────────────────────────────────────────────
    Mutant(
        tag="leftover",
        why="the lock table stops reclaiming, leaking one lock per key ever touched",
        path="agentkit/adapters/store/_keylock.py",
        before="        if entry.users == 0 and table.get(key) is entry:",
        after="        if False:",
        tests=STORELOCK_TESTS,
    ),
    # NOTE: "the refcount is zeroed instead of decremented" has no mutant. It
    # is EQUIVALENT, and working out why is worth recording. The decrement runs
    # in the ``finally``, so while callers are in flight nothing has reached it
    # and the count is right either way; on the way out the table ends empty
    # under both spellings. The one scenario that could differ — a caller
    # arriving after a premature delete and building a second lock — cannot
    # happen, because ``get_or_set`` stores the value INSIDE the lock, so that
    # caller hits the cached fast path and never asks for a lock at all.
    #
    # The reference count is still the right design (it is what makes the
    # release rule correct rather than accidentally correct, and it is asserted
    # directly by ``test_the_lock_entry_is_shared_while_callers_are_in_flight``),
    # but forcing a kill here would mean inventing a scenario the code cannot
    # reach.
    Mutant(
        tag="leftover",
        why="an audit record covering three charges reads exactly like one covering one",
        path="agentkit/middlewares/resilience.py",
        before='            call.meta["attempts"] = attempt',
        after="            pass",
        tests=AUDIT2_TESTS,
    ),
    Mutant(
        tag="leftover",
        why="the request-builder pre-check drifts from the compactor again",
        path="agentkit/capabilities/request_builder/base.py",
        before="    return estimate_message_tokens(messages)",
        after='    return sum(len(m.content or "") for m in messages) // 4',
        tests=EST_TESTS,
    ),
    # ── observers / stores / vectors ────────────────────────────────────────
    Mutant(
        tag="obs",
        why="the queue bound counts lifetime emissions again, dropping 8 of 12 with an idle queue",
        path="agentkit/adapters/observer/sinks.py",
        before="            if item.kind not in CRITICAL_KINDS:\n                self._noncritical -= 1",
        after="            if False:\n                self._noncritical -= 1",
        tests=OBS_TESTS,
    ),
    Mutant(
        tag="obs",
        why="the rollup buffer is read after the await again, raising IndexError into emit()",
        path="agentkit/adapters/observer/cadence.py",
        before="        buf, self._buf = self._buf, []",
        after="        buf = self._buf",
        tests=OBS_TESTS,
    ),
    Mutant(
        tag="obs",
        why="close() stops forwarding, so a wrapped rollup loses its trailing summary",
        path="agentkit/adapters/observer/hooks.py",
        before="    async def close(self) -> None:",
        after="    async def _disabled_close(self) -> None:",
        tests=OBS_TESTS,
    ),
    Mutant(
        tag="store",
        why="a cached None reads as a miss, so a null-returning producer re-runs every call",
        path="agentkit/adapters/store/redis.py",
        # Target ``_lookup`` itself, not either ``is not _MISS`` guard: there
        # are two of those (a fast path outside the lock and the decision
        # inside it), so breaking one alone is EQUIVALENT — the other still
        # catches it and only an extra lock acquisition changes. ``_lookup`` is
        # where the miss/stored-null distinction is actually recovered.
        before="        return _MISS if raw is None else json.loads(raw)",
        after="        return _MISS if raw is None else (json.loads(raw) or _MISS)",
        tests=STORE2_TESTS,
    ),
    Mutant(
        tag="vec",
        why="a repeated id inside one upsert is appended twice, breaking upsert-by-id",
        path="agentkit/adapters/vector/in_memory.py",
        before="        bucket.extend((ch, _vec(ch.text)) for ch in incoming.values())",
        after="        bucket.extend((ch, _vec(ch.text)) for ch in chunks)",
        tests=VEC_TESTS,
    ),
    # ── providers ───────────────────────────────────────────────────────────
    Mutant(
        tag="prov",
        why="the permanent-429 message carries a transient marker again, so classify says retry",
        path="agentkit/adapters/llm/providers/base.py",
        before="def _permanent_message(",
        after="def _unused_permanent_message(",
        tests=PROV_TESTS,
    ),
    Mutant(
        tag="prov",
        why="the error body is truncated BEFORE parsing, losing error.type on a long body",
        path="agentkit/adapters/llm/providers/base.py",
        before="def _error_type(",
        after="def _unused_error_type(",
        tests=PROV_TESTS,
    ),
    Mutant(
        tag="prov",
        why="a dict-shaped provider response coerces to an empty result again",
        path="agentkit/adapters/llm/_mapping.py",
        before="def _first_int(",
        after="def _unused_first_int(",
        tests=MAP_TESTS,
    ),
    Mutant(
        tag="prov",
        why="multi-line SSE data frames are dropped whole and silently",
        path="agentkit/adapters/llm/providers/base.py",
        before="def _sse_decode(",
        after="def _unused_sse_decode(",
        tests=SSE_TESTS,
    ),
    # ── token accounting ────────────────────────────────────────────────────
    Mutant(
        tag="tokens",
        why="tool calls stop counting, so compaction never fires on an agentic transcript",
        path="agentkit/context/tokens.py",
        before="def estimate_message_tokens(",
        after="def _unused_estimate_message_tokens(",
        tests=TOK_TESTS,
    ),
    Mutant(
        tag="schema",
        why="an init=False dataclass field crashes with a raw TypeError again",
        path="agentkit/capabilities/output_schema/dataclass_adapter.py",
        before="def _coerce_dataclass(",
        after="def _unused_coerce_dataclass(",
        tests=SCHEMA2_TESTS,
    ),
    Mutant(
        tag="schema",
        why="serialize emits the SERIALIZATION alias, which parse cannot read",
        path="agentkit/capabilities/output_schema/pydantic_adapter.py",
        before="        return cast(dict[str, Any], _to_validation_keys(value, dumped))",
        after="        return dumped",
        tests=SCHEMA2_TESTS,
    ),
    # ── middlewares: bill it, key it, and report the failure ────────────────
    Mutant(
        tag="mw",
        why="tool calls share one memo key again, so every tool returns the first one's result",
        path="agentkit/middlewares/memoize.py",
        before="    if _is_tool_call(call):",
        after="    if False:",
        tests=MW_MEMO_TESTS,
    ),
    Mutant(
        tag="mw",
        why="an abandoned stream escapes the meter, so real tokens are billed unseen",
        path="agentkit/middlewares/meter.py",
        before="        except BaseException:",
        after="        except _NeverRaisedByAnything:",
        tests=MW_METER_TESTS,
    ),
    Mutant(
        tag="mw",
        why="a cache hit is charged again, tripping a ceiling on money never spent",
        path="agentkit/middlewares/meter.py",
        before='        if ctx.call.meta.get("cache_hit"):',
        after="        if False:",
        tests=MW_METER_TESTS,
    ),
    Mutant(
        tag="mw",
        why="a tool-call turn poisons the semantic cache, so the loop returns an empty answer",
        path="agentkit/middlewares/memoize.py",
        before='        if content and not getattr(result, "tool_calls", None):',
        after="        if content is not None:",
        tests=MW_SEM_TESTS,
    ),
    Mutant(
        tag="mw",
        why="a failed call is exported as a successful span, hiding provider errors",
        path="agentkit/middlewares/tracing.py",
        before="            stack.__exit__(type(exc), exc, exc.__traceback__)",
        after="            stack.close()",
        tests=MW_TRACE_TESTS,
    ),
    Mutant(
        tag="mw",
        why="a misbehaving tracer kills the run from the compaction seam",
        path="agentkit/middlewares/compaction.py",
        before="with _safe_trace_span(",
        after="with _unsafe_trace_span(",
        tests=MW_COMPACT_TESTS,
    ),
    Mutant(
        tag="mw",
        why="a failed side effect leaves no audit record at all",
        path="agentkit/middlewares/egress_audit.py",
        before="    async def on_error(",
        after="    async def _disabled_on_error(",
        tests=MW_AUDIT_TESTS,
    ),
    # ── a failure must never look like a clean finish ───────────────────────
    Mutant(
        tag="merge",
        why="a raising source posts the DONE frame again, truncating the fan-in in silence",
        path="agentkit/kernel/streams.py",
        before="            elif error is not None:",
        after="            elif False:",
        tests=STREAMS_TESTS,
    ),
    Mutant(
        tag="merge",
        why="the error is dropped instead of transported, so gather() discards it",
        path="agentkit/kernel/streams.py",
        before="            error = exc",
        after="            error = None",
        tests=STREAMS_TESTS,
    ),
    Mutant(
        tag="breaker",
        why="a permanent probe failure wedges the breaker in half_open forever",
        path="agentkit/kernel/resilience.py",
        before="    def release_probe(self) -> None:",
        after="    def release_probe(self) -> None:\n        return",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="breaker",
        why="a lost probe is never presumed lost, so a silent caller wedges the gate",
        path="agentkit/kernel/resilience.py",
        before="            if now - self._probe_started_at >= self.cooldown:",
        after="            if False:",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="breaker",
        why="the neutral release grants health credit, treating a 401 as a recovery",
        path="agentkit/kernel/resilience.py",
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
        path="agentkit/kernel/resilience.py",
        before="min(attempt - 1, 1023)",
        after="attempt - 1",
        tests=BREAKER_TESTS,
    ),
    Mutant(
        tag="conc",
        why="an abort leaves its siblings running detached, still spending",
        path="agentkit/kernel/concurrency.py",
        before=(
            "        for t in tasks:\n            t.cancel()\n"
            "        if tasks:\n"
            "            _, pending = await asyncio.wait(tasks, timeout=SIBLING_CLEANUP_GRACE_S)"
        ),
        after="        pass",
        tests=CONC_TESTS,
    ),
    # ── a declared ceiling must actually hold ───────────────────────────────
    Mutant(
        tag="workflow",
        why="max_steps is checked after the wave again, so a self-route loops forever",
        path="agentkit/agents/workflow.py",
        before="                if steps >= self.max_steps:",
        after="                if steps >= self.max_steps and pending and steps:",
        tests=WF_TESTS,
    ),
    Mutant(
        tag="workflow",
        why="routes read `done` again, so a second route from a self-routing node KeyErrors",
        path="agentkit/agents/workflow.py",
        before="                        if when(wave_outputs[node.name]):",
        after="                        if when(done[node.name]):",
        tests=WF_TESTS,
    ),
    Mutant(
        tag="workflow",
        why="a tool node drops the tool's url_arg, so egress() never sees a URL to check",
        path="agentkit/agents/workflow.py",
        before='        url_arg = url_arg if url_arg is not None else getattr(tool, "url_arg", None)',
        after="        url_arg = url_arg",
        tests=WF_TESTS,
    ),
    Mutant(
        tag="workflow",
        why="a node can downgrade a side-effecting tool, so a mutation skips the approval gate",
        path="agentkit/agents/workflow.py",
        before='        side_effecting = side_effecting or bool(getattr(tool, "side_effecting", False))',
        after=(
            "        side_effecting = side_effecting if side_effecting else "
            'bool(getattr(tool, "side_effecting", False)) and side_effecting is not False'
        ),
        tests=WF_TESTS,
    ),
    Mutant(
        tag="ceiling",
        why="a budget ceiling below an as_tool boundary is reflected to the model again",
        path="agentkit/agents/cognition/react.py",
        before="        except _ceiling_errors():",
        after="        except ():",
        tests=CEILING_TESTS,
    ),
    Mutant(
        tag="ceiling",
        why="the ceiling reaches the caller wrapped in an ExceptionGroup nobody catches",
        path="agentkit/agents/cognition/react.py",
        before="                    breached = _unwrap_ceiling(group)",
        after="                    breached = None",
        tests=CEILING_TESTS,
    ),
    Mutant(
        tag="ceiling",
        why="a genuine multi-failure fan-out is unwrapped, hiding every failure but one",
        path="agentkit/agents/_agent_helpers.py",
        before="        if isinstance(exc, _ceiling_errors()):\n            return exc",
        after="        return exc",
        tests=CEILING_TESTS,
    ),
    # ── a documented entry point must actually work ─────────────────────────
    Mutant(
        tag="surface",
        why="a submodule shadows an exported callable again (the bug, reintroduced)",
        path="agentkit/middlewares/__init__.py",
        # Mutating the RATCHET to disable its own assertion would prove nothing
        # — a test cannot vouch for itself. This re-creates the defect in the
        # SOURCE instead: bind an exported name to a module, exactly as the
        # ``security.py`` submodule import did, and check the ratchet fails.
        before="from agentkit.middlewares.tracing import tracing",
        after=(
            "from agentkit.middlewares import egress_audit as security\n"
            "from agentkit.middlewares.tracing import tracing"
        ),
        tests=SURFACE_TESTS,
    ),
    Mutant(
        tag="prompt",
        why="declared inputs are ignored, so placeholders ship to the model verbatim",
        path="agentkit/prompts/prompt.py",
        before='            text = text.replace("{" + name + "}", str(resolved[name]))',
        after="            pass",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="prompt",
        why="a missing input renders a half-filled prompt instead of refusing",
        path="agentkit/prompts/prompt.py",
        before="        if missing or unexpected:",
        after="        if False:",
        tests=PROMPT_TESTS,
    ),
    Mutant(
        tag="prompt",
        why="str.format is used, so a JSON Schema in a template raises or is eaten",
        path="agentkit/prompts/prompt.py",
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
        path="agentkit/prompts/prompt.py",
        before='            self._reject_undeclared(values, verb="passed to render()")',
        after="            pass",
        tests=PROMPT_TESTS,
    ),
    # ── interrupt: stop the turn, keep the conversation ─────────────────────
    Mutant(
        tag="cliinterrupt",
        why="a control_response is folded into the turn, putting protocol JSON in the answer",
        path="agentkit/agents/cognition/claude_cli.py",
        before='                    if payload.get("type") == "control_response":',
        after="                    if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="an interrupted turn reports the CLI's ambiguous error subtype instead",
        path="agentkit/agents/cognition/claude_cli.py",
        before='                state.stop_reason = "interrupted"',
        after="                pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="an interrupt is categorised as a failure rather than a deliberate stop",
        path="agentkit/agents/result.py",
        before='    "interrupted": "terminated",',
        after='    "interrupted": "failed",',
        tests=("tests/agents/test_stop_reason_taxonomy.py",),
    ),
    # ── the offline CLI double, and the two bugs it surfaced ────────────────
    Mutant(
        tag="fakecli",
        why="Ctrl-C during a CLI run is swallowed into a tidy successful-looking "
        "terminal event instead of propagating — which also hides a test double's "
        "ScriptExhausted, the one exception that must never be caught",
        path="agentkit/agents/cognition/claude_cli.py",
        before="    if fatal_exc is not None and not isinstance(fatal_exc, Exception):",
        after="    if fatal_exc is not None and isinstance(fatal_exc, Exception):",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="the SESSION half of the re-raise is a separate line in a separate method; "
        "deleting it leaves Ctrl-C during a session turn swallowed, and exactly one "
        "test in the suite notices",
        path="agentkit/agents/cognition/claude_cli.py",
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
        path="agentkit/agents/cognition/claude_cli.py",
        before="    except ValueError:  # JSONDecodeError *and* UnicodeDecodeError",
        after="    except json.JSONDecodeError:",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="a replayed process that reports its exit code before wait() makes every "
        "persistent session look dead from turn two",
        path="agentkit/testing/fakes/claude_cli.py",
        before="        return self._invocation.returncode\n\n    async def wait(self) -> int:",
        after="        return self._run.returncode\n\n    async def wait(self) -> int:",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="dropping the newline-less fragment at EOF silently loses a recording's "
        "final result payload — cost, usage and session id vanish while the run still "
        "reports partial=False",
        path="agentkit/testing/fakes/claude_cli.py",
        before="            yield blob[start:]\n            return",
        after="            return",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="sorting the queued stderr chunks as whole tuples lets a tie on the stdout "
        "position fall through to comparing BYTES, so a multi-write traceback is "
        "reassembled in alphabetical order",
        path="agentkit/testing/fakes/claude_cli.py",
        before="sorted(run.interleaved_stderr, key=lambda pair: pair[0])",
        after="sorted(run.interleaved_stderr)",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="fakecli",
        why="a double whose terminate() does not stop the child still reports "
        "stop_reason='cancelled', because cancellation outranks the exit code — so a "
        "leaked subprocess is invisible from the result alone",
        path="agentkit/testing/fakes/claude_cli.py",
        before="        self._invocation.terminated = True",
        after="        self._invocation.terminated = False",
        tests=FAKE_CLI_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="interrupting an idle session sends a request nobody will ever answer",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if proc is None or proc.returncode is not None or not self._turn_active:",
        after="        if proc is None or proc.returncode is not None:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="a turn that ends first strands the interrupt waiter forever",
        path="agentkit/agents/cognition/claude_cli.py",
        before='                self._fail_pending("the turn ended before the CLI answered")',
        after="                pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="cancel_queued is sent to a CLI that does not support it, and silently ignored",
        path="agentkit/agents/cognition/claude_cli.py",
        before='            if not self.supports("interrupt_cancel_queued_v1"):',
        after="            if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="cancel_queued uses the plain interrupt subtype, so queued work still runs",
        path="agentkit/agents/cognition/claude_cli.py",
        before='            subtype = "interrupt_cancel_queued"',
        after='            subtype = "interrupt"',
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="capabilities land only at the terminal event, too late to feature-detect against",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                        if delta.init:\n                            self._absorb_init(delta.init)",
        after="                        pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="an inline interrupt hangs the application instead of degrading",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                payload = await asyncio.wait_for(future, timeout=ack_timeout_s)",
        after="                payload = await future",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="cliinterrupt",
        why="the interrupt flag is set only after the ack, which the turn may end before",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        self._interrupted = True\n        try:",
        after="        try:",
        tests=CLI_SESSION_TESTS,
    ),
    # ── the CLI's permission prompts reach a person ─────────────────────────
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
        before='            return Decision(kind="expired", note=f"no answer within {self.timeout_s}s")',
        after='            return Decision(kind="approve", note="timed out")',
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="the timeout is not enforced, so a turn can park forever",
        path="agentkit/integrations/mcp/approvals.py",
        before="        if self.timeout_s is None:\n            return await self.asker.ask(request)",
        after="        if True:\n            return await self.asker.ask(request)",
        tests=APPROVAL_TESTS,
    ),
    Mutant(
        tag="approvals",
        why="a modify decision is treated as a plain approval, discarding the reviewer's edits",
        path="agentkit/integrations/mcp/approvals.py",
        before='        if decision.kind == "modify" and isinstance(decision.value, dict):',
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
        before=(
            "        if tool_name in self.auto_allow and self._arguments_are_auto_allowed(\n"
            "            tool_name, arguments\n"
            "        ):"
        ),
        after="        if False:",
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
    # ── one process, many turns ─────────────────────────────────────────────
    Mutant(
        tag="clisession",
        why="a turn reads to EOF like a one-shot drive, so the session hangs forever",
        path="agentkit/agents/cognition/claude_cli.py",
        before='                    if payload.get("type") == "result":\n                        break',
        after='                    if payload.get("type") == "__never":\n                        break',
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="the prompt is passed as an argv argument too, running a turn nobody asked for",
        path="agentkit/agents/cognition/claude_cli.py",
        before='        if stream_input:',
        after="        if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="turns are not serialised, so two callers interleave one conversation",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        async with self._lock:",
        after="        if True:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a turn is written into a process that has already exited",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                if proc.returncode is not None:",
        after="                if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a closed session accepts turns, silently starting a fresh conversation",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                if self._closed or proc is None:",
        after="                if proc is None:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a CLI that closed mid-turn is not noticed, so the turn reports success",
        path="agentkit/agents/cognition/claude_cli.py",
        before='                    raise _SessionClosed(\n                        "the CLI closed its output mid-turn; the session is over"\n                    )',
        after="                    pass",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a dead session reads as a spawn failure rather than an ended conversation",
        path="agentkit/agents/cognition/claude_cli.py",
        before="            if isinstance(fatal_exc, _SessionClosed):",
        after="            if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="per-turn structured output is accepted, silently returning prose",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                    if schema_requested:",
        after="                    if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="a cancelled turn leaves the process alive with half an answer in its context",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                if cancelled and proc is not None and proc.returncode is None:",
        after="                if False:",
        tests=CLI_SESSION_TESTS,
    ),
    Mutant(
        tag="clisession",
        why="closing the session kills the CLI instead of ending the conversation",
        path="agentkit/agents/cognition/claude_cli.py",
        before=(
            "                if proc.stdin is not None and not proc.stdin.is_closing():\n"
            "                    proc.stdin.close()"
        ),
        after="                pass",
        tests=CLI_SESSION_TESTS,
    ),
    # ── token streaming, without showing every sentence twice ───────────────
    Mutant(
        tag="clistream",
        why="the completed block is re-emitted, so a UI renders every sentence twice",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                        None if partial else StreamEvent(\"message_delta\", text=text),",
        after="                        StreamEvent(\"message_delta\", text=text),",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="token deltas also accumulate, doubling AgentResult.output",
        path="agentkit/agents/cognition/claude_cli.py",
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
    # NOTE: "non-text deltas are ignored" has no mutant. Each branch reads its
    # OWN field (``delta["text"]`` / ``delta["thinking"]``), so a signature or
    # tool-argument delta yields an empty string no matter which branch it
    # reaches — every single-token mutation of that dispatch is equivalent. The
    # behaviour is enforced by structure, and pinned by
    # ``test_non_text_deltas_are_ignored``; forcing a kill here would mean
    # adding a decision point that exists only to be mutated.
    Mutant(
        tag="clistream",
        why="the partial-messages flag is never passed, so no tokens ever arrive",
        path="agentkit/agents/cognition/claude_cli.py",
        before='        if self.partial_messages:\n            argv += ["--include-partial-messages"]\n',
        after="",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="a skipped MCP server is discarded with the rest of the init payload",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if state.init:\n            # Startup facts an operator needs",
        after="        if False:\n            # Startup facts an operator needs",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="an empty error list is manufactured, so presence stops being the signal",
        path="agentkit/agents/cognition/claude_cli.py",
        before="                    if k in payload\n                },",
        after="                },",
        tests=CLI_STREAM_TESTS,
    ),
    Mutant(
        tag="clistream",
        why="provider retries are invisible, leaving a 40s pause unexplained",
        path="agentkit/agents/cognition/claude_cli.py",
        before='        if subtype == "api_retry":',
        after="        if False:",
        tests=CLI_STREAM_TESTS,
    ),
    # ── the flags a service ships with ──────────────────────────────────────
    Mutant(
        tag="cliops",
        why="strict_mcp_config alone silently leaves the session with no MCP servers",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if self.strict_mcp_config and not self.mcp_config:",
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="a stable prompt prefix is silently a no-op under a replaced system prompt",
        path="agentkit/agents/cognition/claude_cli.py",
        before='        if self.stable_prompt_prefix and self.system_prompt_mode == "replace":',
        after="        if False:",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="a typo'd add_dir reaches the CLI and dies three seconds in",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        missing_dirs = [str(d) for d in self.add_dirs if not Path(d).is_dir()]",
        after="        missing_dirs = []",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="a fallback CHAIN is passed as a Python tuple repr instead of a comma list",
        path="agentkit/agents/cognition/claude_cli.py",
        before='                else ",".join(self.fallback_model)',
        after="                else str(self.fallback_model)",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="bare mode without a credential fails with a message pointing at the wrong fix",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if self.bare:\n            self._warn_if_bare_mode_has_no_credential(env)",
        after="        if False:\n            self._warn_if_bare_mode_has_no_credential(env)",
        tests=CLI_FLAG_TESTS,
    ),
    Mutant(
        tag="cliops",
        why="the bare-mode warning fires even when settings may carry an apiKeyHelper",
        path="agentkit/agents/cognition/claude_cli.py",
        before=(
            "        if self.settings is not None:\n"
            "            return\n"
            "        if any(env.get(name) for name in _BARE_CREDENTIAL_ENV):"
        ),
        after="        if any(env.get(name) for name in _BARE_CREDENTIAL_ENV):",
        tests=CLI_FLAG_TESTS,
    ),
    # ── CLI spend is on the books, and the ceiling is on the CLI ────────────
    Mutant(
        tag="clibudget",
        why="CLI spend is invisible to every meter again ($50 run, $0.00 ledger)",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        charge_error = await self._charge_meters(ctx, state.usage)",
        after="        charge_error = None",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the run's headroom is never handed to the CLI, so nothing stops it mid-flight",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if max_budget_usd is not None:\n            argv += [\"--max-budget-usd\", max_budget_usd]\n",
        after="",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the ORIGINAL ceiling is sent instead of the remaining headroom",
        path="agentkit/agents/cognition/claude_cli.py",
        before='        return f"{headroom:f}"',
        after='        return f"{budget.ceiling():f}"',
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="an exhausted budget still burns a subprocess spawn to learn it is exhausted",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        if headroom <= 0:",
        after="        if False:",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the pre-flight refusal is categorised as a spawn failure, so it reads unresumable",
        path="agentkit/agents/cognition/claude_cli.py",
        before='            elif type(fatal_exc).__name__ == "MeterExceeded":',
        after="            elif False:",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="the per-actor envelope is skipped, the bug ActorBudget already had once",
        path="agentkit/agents/cognition/claude_cli.py",
        before="        actor = getattr(ctx, \"actor_budget\", None)\n        if actor is not None:",
        after="        actor = None\n        if actor is not None:",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="a ceiling crossed by this very run raises, losing a result already paid for",
        path="agentkit/agents/cognition/claude_cli.py",
        before=(
            "            except Exception as exc:  # noqa: BLE001 — see docstring\n"
            "                note = f\"{type(exc).__name__}: {exc}\""
        ),
        after="            except Exception:\n                raise",
        tests=CLI_BUDGET_TESTS,
    ),
    Mutant(
        tag="clibudget",
        why="meter_spend=False still charges the shared envelope",
        path="agentkit/agents/cognition/claude_cli.py",
        before=(
            "        if not self.meter_spend or ctx is None:\n"
            "            return None\n"
            "        call = _CliCall(ctx=ctx)"
        ),
        after="        if ctx is None:\n            return None\n        call = _CliCall(ctx=ctx)",
        tests=CLI_BUDGET_TESTS,
    ),
    # ── output= types a CLI-delegated run too ───────────────────────────────
    Mutant(
        tag="clischema",
        why="agent.output is ignored again, so the CLI never validates its answer",
        path="agentkit/agents/cognition/claude_cli.py",
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
        path="agentkit/agents/cognition/claude_cli.py",
        before="        return adapter.validate(value), None",
        after="        return value, None",
        tests=CLI_SCHEMA_TESTS,
    ),
    Mutant(
        tag="clischema",
        why="a success with no structured_output reads as a clean run",
        path="agentkit/agents/cognition/claude_cli.py",
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
        before=(
            "    kinds = {_JSON_PRIMITIVES.get(type(v)) for v in vals}\n"
            "    if len(kinds) == 1 and (t := kinds.pop()) is not None:\n"
            "        frag[\"type\"] = t\n"
            "    return frag"
        ),
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
        before=(
            "        with contextlib.suppress(asyncio.QueueEmpty):\n"
            "            self.outbox.get_nowait()  # evict the oldest"
        ),
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
        before=(
            "        termination: TerminationCondition = _run_local_termination(\n"
            "            cognition, MaxTurns(self.max_turns)\n"
            "        )"
        ),
        after=(
            "        termination: TerminationCondition = getattr(cognition, \"termination\", None) or MaxTurns(self.max_turns)"
        ),
        tests=TERM_TESTS,
    ),
    Mutant(
        tag="termination",
        why="cloning the external switch, so set() cannot reach a running loop",
        path="agentkit/agents/control/termination.py",
        before=(
            "    def __deepcopy__(self, memo: dict[int, Any]) -> ExternalTermination:\n"
            "        memo[id(self)] = self\n"
            "        return self\n"
            ""
        ),
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
        before=(
            "        steps, dropped = _validate_plan(steps, children, best_effort=self.best_effort)\n"
            "\n"
            "        total = len(steps)"
        ),
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
        before=(
            "        steps, dropped = _validate_plan(steps, children, best_effort=self.best_effort)\n"
            "        errors.extend(dropped)"
        ),
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
        before=(
            "        if parent_actor_budget is not None:\n"
            "            for idx, slice_ in enumerate(reserved_slices):"
        ),
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
        path="agentkit/middlewares/egress_audit.py",
        before="        if guardrail is None:",
        after="        if False:",
        tests=SECURITY_TESTS,
    ),
    Mutant(
        tag="resilience",
        why="a late success closes an OPEN breaker, bypassing the cooldown",
        path="agentkit/kernel/resilience.py",
        before='            if self.state == "open":\n                return',
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
    Mutant(
        tag="compaction",
        why="the sliding window stops guarding its walk-back, so keep_recent=0 raises IndexError",
        path="agentkit/capabilities/compaction/impls.py",
        # Anchored on the comment above it: all THREE compactors in this file run
        # the identical walk-back, and the other two already guarded it.
        before=(
            "        # ``ImportanceFilteringCompactor`` guards its identical loop the same way.\n"
            '        while keep and start > head and messages[start].role == "tool":'
        ),
        after=(
            "        # ``ImportanceFilteringCompactor`` guards its identical loop the same way.\n"
            '        while start > head and messages[start].role == "tool":'
        ),
        tests=COMPACTION_TESTS,
    ),
    Mutant(
        tag="compaction",
        why="keep_recent stops being clamped, so a negative count reaches past the end of the list",
        path="agentkit/capabilities/compaction/impls.py",
        before="        keep = max(0, self.keep_recent)",
        after="        keep = self.keep_recent",
        tests=COMPACTION_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="abstract collections fall back to 'string', telling the model to send a JSON string",
        path="agentkit/tools/schema.py",
        before="    if origin in _ARRAY_ANNOTATIONS:\n        element = _element_annotation(ann)",
        after="    if origin in (list, tuple, set, frozenset):\n        element = _element_annotation(ann)",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="a TypedDict parameter is advertised as a string again, so the body indexes a str",
        path="agentkit/tools/schema.py",
        before="    if not is_typeddict(ann):\n        return None",
        after="    if True:\n        return None",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="a TypedDict's total=False keys are advertised as required, inverting the contract",
        path="agentkit/tools/schema.py",
        before='            "required": [key for key in hints if key in required],',
        after='            "required": list(hints),',
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="money",
        why="an unrepresentable ceiling escapes as decimal.InvalidOperation, past MoneyPrecisionError",
        path="agentkit/runtime/meter.py",
        before=(
            "        raise MoneyPrecisionError(\n"
            '            f"monetary amount {value!r} is too large to represent at "\n'
            '            f"{MONEY_SCALE} decimal places"\n'
            "        ) from exc"
        ),
        after="        raise",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="money",
        why="max_concurrency=0 constructs again, and the first fan-out waits on it forever",
        path="agentkit/runtime/meter.py",
        before="        if self.max_concurrency < 1:",
        after="        if False:",
        tests=MONEY_TESTS,
    ),
    Mutant(
        tag="workflow",
        why="a typo'd after= stops being caught, so dependent nodes silently never run",
        path="agentkit/agents/workflow.py",
        before="        self._validate_dependencies()",
        after="        pass",
        tests=WF_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="containers stop describing items, so list[Unit] is an untyped array again",
        path="agentkit/tools/schema.py",
        before="        return {\"type\": \"array\", \"items\": _json_type(element)}",
        after="        return {\"type\": \"array\"}",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="a heterogeneous tuple claims one element type, advertising items that are a lie",
        path="agentkit/tools/schema.py",
        before="        if len(args) == 2 and args[1] is Ellipsis:\n            return None if args[0] is Any else args[0]\n        return None",
        after="        return None if args[0] is Any else args[0]",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="elements stop being coerced, so the body gets strings where items promised members",
        path="agentkit/tools/schema.py",
        before="    if origin in _ARRAY_ANNOTATIONS:\n        return _sequence_coercer(ann, origin)",
        after="    if origin in _ARRAY_ANNOTATIONS:\n        return None",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="set/tuple annotations receive a list again, so the body gets a container it did not ask for",
        path="agentkit/tools/schema.py",
        before="_REBUILT_CONTAINERS: dict[Any, Any] = {set: set, frozenset: frozenset, tuple: tuple}",
        after="_REBUILT_CONTAINERS: dict[Any, Any] = {}",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="Optional collapses to bare X again, so a required X|None cannot be sent as null",
        path="agentkit/tools/schema.py",
        before="        nullable = len(real) != len(args)",
        after="        nullable = False",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="toolschema",
        why="*args is silently dropped again, leaving a parameter that can never be filled",
        path="agentkit/tools/schema.py",
        before="        if p.kind is p.VAR_POSITIONAL:",
        after="        if False:",
        tests=TOOLCOERCE_TESTS,
    ),
    Mutant(
        tag="workflow",
        why="dependency validation only checks the first node, so a later typo still slips through",
        path="agentkit/agents/workflow.py",
        before="        for name in self._order:",
        after="        for name in self._order[:1]:",
        tests=WF_TESTS,
    ),
    Mutant(
        tag="memdedupe",
        why=(
            'Restores `item.id is not None`, so an id of "" is admitted as a real shared identity again and every '
            'id-less-but-not-None record in the pool collapses into one group. Killed by '
            'test_an_empty_string_id_is_treated_as_no_id_at_all.'
        ),
        path="agentkit/memory/composite.py",
        before="        if mode == \"id\" and item.id:",
        after="        if mode == \"id\" and item.id is not None:",
        tests=MEM_DEDUPE_TESTS,
    ),
    Mutant(
        tag="memdedupe",
        why=(
            'Removes the blank-content guard so `""`, `"   "` and `"\\n\\t"` share '
            'one digest again and chain unrelated '
            'records together across BOTH dedupe modes. Killed by '
            'test_blank_content_is_not_evidence_that_two_records_are_the_same_fact.'
        ),
        path="agentkit/memory/composite.py",
        before="    stripped = content.strip()\n    if not stripped:\n        return None\n    return hashlib.sha256(stripped.encode(\"utf-8\")).hexdigest()",
        after="    stripped = content.strip()\n    return hashlib.sha256(stripped.encode(\"utf-8\")).hexdigest()",
        tests=MEM_DEDUPE_TESTS,
    ),
    Mutant(
        tag="memdedupe",
        why=(
            "Rebuilds the stamp from each member's own `source`/count of 1 instead of absorbing a prior stamp, "
            're-breaking nested composites (dedupe_sources drops the inner sources, dedupe_count under-reports). '
            'Killed by 2 tests: the nested-composite and stamp-absorption tests.'
        ),
        path="agentkit/memory/composite.py",
        before="            agreed.extend(_prior_sources(m))\n            collapsed += _prior_count(m)",
        after="            agreed.append(m.source)\n            collapsed += 1",
        tests=MEM_DEDUPE_TESTS,
    ),
    Mutant(
        tag="memdedupe",
        why=(
            'Drops the type guard on an absorbed count, so a backend that round-tripped a string `dedupe_count` '
            'through storage propagates a str into an int accumulation and raises inside the query path. Killed by '
            'test_a_corrupt_stamp_from_a_backend_does_not_crash_the_merge.'
        ),
        path="agentkit/memory/composite.py",
        before="    if isinstance(prior, int) and not isinstance(prior, bool) and prior >= 1:\n        return prior",
        after="    if prior is not None:\n        return prior  # type: ignore[no-any-return]",
        tests=MEM_DEDUPE_TESTS,
    ),
    Mutant(
        tag="memdedupe",
        why=(
            "Trusts a foreign source's id as a key into this store's keyspace again — the destructive regression: a "
            'broadcast write of a journal item with row key "3" upserts over vector chunk "3". Killed by '
            'test_a_foreign_sources_id_never_addresses_this_stores_keyspace.'
        ),
        path="agentkit/memory/vector.py",
        before="            own_id = item.id if item.source == self.name else None",
        after="            own_id = item.id",
        tests=MEM_VECTOR_TESTS,
    ),
    Mutant(
        tag="memdedupe",
        why=(
            'Stops stripping the transient dedupe stamp before upsert, so per-query fan-out bookkeeping is persisted '
            'as durable record metadata and (given the absorb fix) inflates its own count on every round trip. Killed '
            'by test_the_dedupe_stamp_is_not_persisted_as_record_metadata.'
        ),
        path="agentkit/memory/vector.py",
        before="            for transient in (DEDUPE_SOURCES_KEY, DEDUPE_COUNT_KEY):\n                metadata.pop(transient, None)",
        after="",
        tests=MEM_VECTOR_TESTS,
    ),
    Mutant(
        tag="workflowmap",
        why=(
            'bounded_by stops being an EXTRA bound and becomes a REPLACEMENT for the tree semaphore, so a '
            "`bounded_by=100` map blows the run's `max_concurrency` entirely. SURVIVED all 29 original tests (the "
            'shipped bounded_by test uses width 2 under the default max_concurrency=8, where both behave '
            'identically). Now killed by test_map_bounded_by_does_not_escape_the_level_semaphore.'
        ),
        path="agentkit/agents/workflow.py",
        before="                if width_sem is not None:\n                    async with level_sem:\n                        return await _work(index, item, slot)\n                return await _work(index, item, slot)",
        after="                return await _work(index, item, slot)",
        tests=WORKFLOW_MAP_TESTS,
    ),
    Mutant(
        tag="workflowmap",
        why=(
            'the identity is truncated WITHOUT the disambiguating digest, so two items sharing a 120-char prefix '
            'collapse onto one identity and a swapped re-expansion resumes silently against mis-threaded element '
            'slots instead of raising MapExpansionChanged. SURVIVED all 29 original tests despite the source comment '
            'claiming this exact risk. Now killed by test_map_long_identities_are_capped_but_stay_distinguishable.'
        ),
        path="agentkit/agents/workflow.py",
        before="    return f\"{s[:_IDENTITY_MAX]}\u2026#{hashlib.sha256(s.encode('utf-8', 'replace')).hexdigest()[:12]}\"",
        after="    return s[:_IDENTITY_MAX]",
        tests=WORKFLOW_MAP_TESTS,
    ),
    Mutant(
        tag="workflowmap",
        why=(
            "a failing checkpointer's exception replaces the element failure the caller needs to see, so "
            "`RuntimeError('c is flaky')` is reported to the operator as `OSError('disk full')`. SURVIVED all 29 "
            'original tests — no test had a store that fails. Now killed by '
            'test_map_checkpointer_failure_does_not_replace_the_element_failure.'
        ),
        path="agentkit/agents/workflow.py",
        before="            with contextlib.suppress(Exception):",
        after="            if True:",
        tests=WORKFLOW_MAP_TESTS,
    ),
    Mutant(
        tag="workflowmap",
        why=(
            "the author's `prompt=` is dropped and every runnable element gets the default task instead — a run that "
            'used a prompt nobody wrote and reported success. SURVIVED all 29 original tests: `prompt` appeared zero '
            'times in the test file. Now killed by test_map_prompt_builds_the_task_for_a_runnable_element.'
        ),
        path="agentkit/agents/workflow.py",
        before="                    p = (\n                        prompt(item, goal)\n                        if prompt is not None\n                        else _default_prompt({\"item\": item}, goal)\n                    )",
        after="                    p = _default_prompt({\"item\": item}, goal)",
        tests=WORKFLOW_MAP_TESTS,
    ),
    Mutant(
        tag="workflowmap",
        why=(
            'the identity cap is removed, so a 500-element map over fat payloads writes every payload in full into '
            'the expansion record and therefore into EVERY checkpoint the run takes. SURVIVED all 29 original tests. '
            'Now killed by test_map_long_identities_are_capped_but_stay_distinguishable.'
        ),
        path="agentkit/agents/workflow.py",
        before="_IDENTITY_MAX = 120",
        after="_IDENTITY_MAX = 10**9",
        tests=WORKFLOW_MAP_TESTS,
    ),
    Mutant(
        tag="workflowmap",
        why=(
            'a `prompt=` the framework cannot deliver is silently dropped instead of refused, so a map whose `each` '
            'returns plain data runs with the prompt never invoked and completes green. This is the mutant that '
            'reproduces the ORIGINAL shipped behaviour, which I changed. Killed by '
            'test_map_prompt_with_a_non_runnable_element_is_refused_not_dropped.'
        ),
        path="agentkit/agents/workflow.py",
        before="                if prompt is not None and not callable(runner):",
        after="                if False:",
        tests=WORKFLOW_MAP_TESTS,
    ),
    Mutant(
        tag="storeprim",
        why="`by` stops being validated, so the four backends diverge: memory/file return a float from a method annotated -> int and poison the key, while Redis reports 1 for increment(k, 1.5)",
        path="agentkit/adapters/store/memory.py",
        before="        check_by(key, by)",
        after="",
        tests=STORE_PRIM_TESTS,
    ),
    Mutant(
        tag="storeprim",
        why="FileStore's increment goes back to a plain setdefault lock table, leaking one asyncio.Lock per key forever \u2014 the exact regression `_keylock` exists to prevent",
        path="agentkit/adapters/store/file.py",
        before="        check_by(key, by)\n        self._warn_ttl_ignored(ttl)\n        async with key_lock(self._locks, key):",
        after="        check_by(key, by)\n        self._warn_ttl_ignored(ttl)\n        lock = self._locks.setdefault(key, asyncio.Lock())\n        async with lock:",
        tests=STORE_PRIM_TESTS,
    ),
    Mutant(
        tag="storeprim",
        why="the Postgres error path re-acquires a SECOND pooled connection while holding the first, so increment deadlocks outright on a max_size=1 pool \u2014 invisible until the pool is bounded",
        path="agentkit/adapters/store/postgres.py",
        before="        if total is None:\n            # The guard excluded the row, so something non-integer is there.\n            # Re-read it to name the type in the error \u2014 but OUTSIDE the\n            # ``acquire`` above, because ``self.get`` takes a connection of its\n            # own. Asking the pool for a second connection while still holding\n            # the first deadlocks outright on ``max_size=1`` (measured: the\n            # call hangs forever), and on any pool size it doubles the\n            # connections a concurrent burst of bad increments needs, so N\n            # workers each holding one and waiting for another is a deadlock at\n            # every size. The error path must not be the expensive one.\n            raise not_a_counter(key, await self.get(key))",
        after="            if total is None:\n                # The guard excluded the row, so something non-integer is there.\n                # Re-read it to name the type in the error \u2014 but OUTSIDE the\n                # ``acquire`` above, because ``self.get`` takes a connection of its\n                # own. Asking the pool for a second connection while still holding\n                # the first deadlocks outright on ``max_size=1`` (measured: the\n                # call hangs forever), and on any pool size it doubles the\n                # connections a concurrent burst of bad increments needs, so N\n                # workers each holding one and waiting for another is a deadlock at\n                # every size. The error path must not be the expensive one.\n                raise not_a_counter(key, await self.get(key))",
        tests=STORE_PRIM_TESTS,
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
