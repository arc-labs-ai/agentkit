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
WORKFLOW_TESTS = ("tests/agents/test_workflow.py",)
REGISTRY_TESTS = (
    "tests/adapters/test_model_registry.py",
    "tests/agents/test_agent_capability_binding.py",
)

MUTANTS: tuple[Mutant, ...] = (
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
        before='                        ctx, run_id, context, usage, i + 1, repaired, status="suspended"',
        after="                        ctx, run_id, context, usage, i + 1, repaired",
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
