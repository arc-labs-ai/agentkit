"""Workflow — explicit control: a developer-authored typed graph.

The deterministic counterpart to the emergent team loop. You author the path as a graph of named
**nodes** with data-dependency **edges** (`after`) and optional **conditional routes** (branch / bounded
loop-back). The engine topologically schedules: each wave runs every *ready* node (all deps satisfied)
concurrently under the tree semaphore, threads each node's typed output to its dependents, then evaluates
routes. It reuses the whole spine — `gather_bounded`/`ctx.semaphore()` for concurrency, `ctx.check_cancelled()`
for abort, `ctx.emit()` for observation, the run `Budget` for cost — and the same `Suspended`/`ctx.store`
suspend/resume the loop uses for **human-gate** nodes. Bounded by `max_steps` so a cycle can never
run forever.

Node kinds: `agent` · `coordinator` · `fn` (pure) · `tool` · `human_gate` · `subworkflow` · `map`.
`run()` executes to completion or a suspend; `resume(run_id, decisions, ctx)` continues from a human
gate.

**`map` is the one node whose shape is not a fact about the source.** Every other builder authors a
node, so the graph is fully known before the run; `map` authors ONE node that expands into N element
runs at execution time, where N comes from data a previous node just produced. That is what a
plan-then-execute application needs and what neither an all-authored `Workflow` nor a `PlanPolicy`
(which cannot express the rest of the structure) could give it. See :meth:`Workflow.map`.

**Durable-resume contract.** A human-gate suspend checkpoints `{goal, done, steps}` to `ctx.store`, where
`done` maps each completed node to its output. For resume to survive a *real* (serializing) store — not
just `InMemoryStore`, which keeps live objects — those outputs must be serializable by that store. The
built-in node kinds satisfy this (`agent`/`coordinator` → str, `subworkflow` → the child's `outputs`
dict); a custom `fn`/`tool` node whose output crosses a gate must likewise return a serializable value.
A `map` node additionally records its EXPANSION in `done` — one entry per finished element
under `"<node>[<i>]"` plus an identity list under `"<node>#expansion"` — so a resume can tell which
elements finished and re-run only the rest. Those keys are part of the checkpoint and of
`WorkflowResult.outputs`; see :meth:`Workflow.map` for why recording the expansion, not only its
results, is what makes resume across a dynamically sized node correct.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from agentkit.agents.result import Suspended, WorkflowResult
from agentkit.capabilities.checkpointer import resolve_checkpointer
from agentkit.kernel.concurrency import gather_best_effort, gather_bounded
from agentkit.kernel.errors import AgentkitError, CheckpointerError, Failure
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolRequest, Usage

OnExisting = Literal["fail", "resume", "start_fresh"]


#: How a ``map`` node's element results and expansion record are keyed in ``done``.
#: Positional (``"impl[0]"``) rather than content-addressed, because a name a human
#: can read is what makes a stuck checkpoint diagnosable — and because the identity
#: list below is what actually defends the positions.
_EXPANSION_SUFFIX = "#expansion"
_ELEMENT_KEY = re.compile(r"^(?P<owner>.+)\[(?P<index>\d+)\]$")

#: Identity strings longer than this are truncated and disambiguated with a digest.
#: The record goes into every checkpoint the run writes, so a 500-element map over
#: fat payloads would otherwise double the snapshot; the digest keeps the guard's
#: discriminating power at full strength while the readable prefix keeps it
#: debuggable. Truncating WITHOUT the digest would have silently weakened the guard.
_IDENTITY_MAX = 120


def _reserved_owner(name: str) -> str | None:
    """Which map node, if any, owns ``name`` in the ``done`` namespace.

    ``"impl[3]"`` and ``"impl#expansion"`` both answer ``"impl"``. Used by
    ``_add`` to refuse a graph where an authored node and a map's element
    records would fight over the same ``done`` key — a collision that would
    otherwise surface as a node output silently replaced by an element
    result (or vice versa), which no reader would ever attribute correctly.
    """
    if name.endswith(_EXPANSION_SUFFIX):
        owner = name[: -len(_EXPANSION_SUFFIX)]
        return owner or None
    m = _ELEMENT_KEY.match(name)
    return m.group("owner") if m is not None else None


def _clear_map_records(done: dict[str, Any], name: str) -> None:
    """Drop every ``done`` entry a map node owns — its element results AND its
    expansion record. Used by the loop-back path; see the call site for what
    survives when this is skipped."""
    for k in [k for k in done if _reserved_owner(k) == name]:
        done.pop(k, None)


def _default_identity(item: Any) -> str:
    """The default per-element identity for a ``map`` expansion record.

    ``str(item)`` — the readable choice — with a length cap and a digest tail so a
    long payload cannot bloat every checkpoint the run writes. See ``key=`` on
    :meth:`Workflow.map` for when ``str`` is the wrong answer: a plain object
    inherits ``object.__repr__``, which embeds a MEMORY ADDRESS, so a re-derived
    expansion would look changed even though the items are the same. That is a
    loud, actionable failure rather than a silent one, but ``key=`` is how you
    avoid it.
    """
    s = str(item)
    if len(s) <= _IDENTITY_MAX:
        return s
    return f"{s[:_IDENTITY_MAX]}…#{hashlib.sha256(s.encode('utf-8', 'replace')).hexdigest()[:12]}"


class MapExpansionChanged(AgentkitError):
    """A ``map`` node re-expanded to something different from what its checkpoint recorded.

    Raised on RESUME, and deliberately loud. Element results are keyed by POSITION
    (``"impl[0]"``), so reusing them against a collection whose contents or order
    moved threads one element's output into another element's slot — a wrong answer
    that completes successfully and looks right. Refusing is the only honest option:
    the framework cannot know whether the drift was a reordered ``set``, a re-queried
    database, or a genuine change of plan.

    Fixes, in order of preference: make ``over=`` deterministic given the same
    inputs (sort it); pass ``key=`` so identity comes from a stable field rather
    than ``str(item)``; or start the run fresh.
    """


@dataclass
class _Node:
    name: str
    kind: str
    after: tuple[str, ...]
    run: Callable[
        [dict[str, Any], str, Any], Awaitable[tuple[Any, Usage]]
    ]  # (inputs, goal, ctx) -> (output, usage)
    gate: bool = False
    # A ``map`` node cannot be driven through ``run`` above: expansion needs the
    # live ``done`` map (to skip elements a previous attempt finished) and the
    # current step count (to checkpoint partial progress), neither of which is on
    # that signature. Rather than widen every node kind's signature for one of
    # them, a map carries a second entry point and ``_execute`` prefers it.
    expand: (
        Callable[
            [dict[str, Any], str, Any, dict[str, Any], int], Awaitable[tuple[Any, Usage]]
        ]
        | None
    ) = None


def _warn_unpersisted_gate(gate: str, run_id: str) -> None:
    """Announce a human-gate suspend that cannot be resumed.

    ``stacklevel=4`` aims the warning at the caller's ``run()`` rather than at
    the depths of ``_execute``.
    """
    import warnings

    warnings.warn(
        f"workflow gate {gate!r} suspended run {run_id!r} but no durable seam is wired "
        "on the RunContext, so NOTHING WAS PERSISTED and Workflow.resume() will raise "
        f"\"no suspended workflow {run_id!r} to resume\". Wire either "
        "Services(checkpointer=Checkpointer(port=...)) or Services(store=...) to make "
        "this suspend resumable.",
        UserWarning,
        stacklevel=4,
    )


def _as_tuple(after: Any) -> tuple[str, ...]:
    if not after:
        return ()
    return (after,) if isinstance(after, str) else tuple(after)


def _arity(f: Callable[..., Any]) -> int:
    try:
        return len(inspect.signature(f).parameters)
    except (ValueError, TypeError):
        return 1


def _default_prompt(inputs: dict[str, Any], goal: str) -> str:
    if not inputs:
        return goal
    body = "\n".join(f"[{k}] {v}" for k, v in inputs.items())
    return f"{goal}\n\n{body}"


class Workflow:
    """A typed node/edge graph with a deterministic, concurrent, bounded engine."""

    def __init__(self, name: str = "workflow", *, max_steps: int = 100) -> None:
        """``max_steps`` is a never-hang backstop, checked BETWEEN waves.

        A wave runs whole once it starts, so the count can overshoot by up to
        (widest wave − 1): three independent root nodes with ``max_steps=2``
        run all three and report ``steps=3``. That surprises people, so it is
        worth stating rather than discovering.

        It is deliberate and it is BOUNDED. Stopping mid-wave would drop
        siblings that are already running — a partial, confusing state — and
        the overshoot cannot run away, because the graph's width is static and
        the check happens before every wave. The guarantee is "a cycle cannot
        spin forever", not "the node count never exceeds N"; if you need the
        latter, keep your waves narrow.
        """
        self.name = name
        self.max_steps = max_steps
        self._nodes: dict[str, _Node] = {}
        self._order: list[str] = []
        self._routes: dict[str, list[tuple[Callable[[Any], bool], str]]] = {}

    # ---- builders --------------------------------------------------------------------------------

    def _add(self, node: _Node) -> str:
        if node.name in self._nodes:
            raise ValueError(f"duplicate node {node.name!r}")
        self._check_map_namespace(node)
        self._nodes[node.name] = node
        self._order.append(node.name)
        return node.name

    def _check_map_namespace(self, node: _Node) -> None:
        """Refuse a graph where an authored node and a map's ``done`` records collide.

        A map writes ``"impl[0]"`` and ``"impl#expansion"`` into the same ``done`` map
        that holds node outputs, so an authored node called ``impl[0]`` and a map called
        ``impl`` are two writers of one key. Whoever runs second wins, and the loser's
        value is read by its dependents as if it were its own — a wrong answer that
        raises nothing. Both declaration orders are checked, because ``after=`` may
        forward-reference and there is no "declare maps first" rule to lean on.
        """
        if node.expand is not None:
            clashes = [n for n in self._order if _reserved_owner(n) == node.name]
            if clashes:
                raise ValueError(
                    f"map node {node.name!r} records its expansion in ``done`` under "
                    f"{sorted(clashes)!r}, and those name existing nodes. Rename the map "
                    "or the node — a shared key means one silently overwrites the other."
                )
        # NOT an ``else``. A map can also be the VICTIM: ``map("a")`` followed by
        # ``map("a[0]")`` puts the second map's own output on the first map's element
        # key, and checking only one direction let that pair through.
        owner = _reserved_owner(node.name)
        if owner is not None and owner in self._nodes and self._nodes[owner].expand is not None:
            raise ValueError(
                f"node {node.name!r} collides with the ``done`` namespace of map node "
                f"{owner!r} (its elements are recorded as {owner!r}[i] and "
                f"{owner + _EXPANSION_SUFFIX!r}). Rename one of them."
            )

    def agent(
        self,
        name: str,
        agent: Any,
        *,
        after: Any = (),
        prompt: Callable[[dict[str, Any], str], str] | None = None,
    ) -> str:
        deps = _as_tuple(after)

        async def run(inputs: dict[str, Any], goal: str, ctx: Ctx) -> tuple[Any, Usage]:
            p = prompt(inputs, goal) if prompt else _default_prompt(inputs, goal)
            res = await agent.run(p, ctx.child())
            return res.output, res.usage

        return self._add(_Node(name, "agent", deps, run))

    def fn(self, name: str, f: Callable[..., Any], *, after: Any = ()) -> str:
        """A pure function node. `f` is sync or async, called `f(inputs)` or `f(inputs, goal)` by arity."""
        deps = _as_tuple(after)

        async def run(inputs: dict[str, Any], goal: str, ctx: Ctx) -> tuple[Any, Usage]:
            out = f(inputs, goal) if _arity(f) >= 2 else f(inputs)
            if inspect.isawaitable(out):
                out = await out
            return out, Usage()

        return self._add(_Node(name, "fn", deps, run))

    def tool(
        self,
        name: str,
        tool: Any,
        *,
        after: Any = (),
        args: Callable[..., dict[str, Any]] | None = None,
        side_effecting: bool = False,
        url_arg: str | None = None,
    ) -> str:
        deps = _as_tuple(after)
        # Fall back to the TOOL's own declaration, exactly as the ReAct cognition
        # does (``_tool_request``). These kwargs used to be passed straight through,
        # so a tool built with ``FunctionTool(..., side_effecting=True, url_arg="url")``
        # reached the invoker as ``side_effecting=False, url_arg=None`` unless the
        # graph author happened to restate both. Measured: an egress()-guarded
        # workflow fetched https://evil.com/x with an allowlist of example.com
        # (a real SSRF-guard bypass — Egress only checks when ``url_arg`` is set),
        # and idempotent() executed the same charge twice (its ``when`` predicate
        # reads ``request.side_effecting``). An explicit kwarg still wins, so a node
        # can mark a tool side-effecting/URL-bearing that did not declare itself so.
        #
        # Both are ESCALATE-ONLY, and deliberately: a graph author cannot pass
        # ``side_effecting=False`` to downgrade a tool that declares itself
        # side-effecting, nor ``url_arg=None`` to suppress an egress check the
        # tool asked for. These are safety flags — the tool's author knows what
        # it does, and a node should not be able to quietly opt out of a guard
        # on its behalf. Pinned by
        # ``test_workflow_tool_node_cannot_downgrade_a_side_effecting_tool``.
        side_effecting = side_effecting or bool(getattr(tool, "side_effecting", False))
        url_arg = url_arg if url_arg is not None else getattr(tool, "url_arg", None)

        async def run(inputs: dict[str, Any], goal: str, ctx: Ctx) -> tuple[Any, Usage]:
            a = {} if args is None else (args(inputs, goal) if _arity(args) >= 2 else args(inputs))
            req = ToolRequest(
                name=getattr(tool, "name", name),
                arguments=a,
                tool=tool,
                side_effecting=side_effecting,
                url_arg=url_arg,
            )
            return await ctx.invoker.invoke_tool(req, ctx), Usage()

        return self._add(_Node(name, "tool", deps, run))

    def coordinator(
        self,
        name: str,
        coordinator: Any,
        *,
        after: Any = (),
        prompt: Callable[[dict[str, Any], str], str] | None = None,
    ) -> str:
        """A coordinator `Agent` (or any runnable with `async run(task, ctx)`) as a graph node
        — emergent inside explicit. Output is the coordinator's last transcript message; usage
        is merged.
        """
        deps = _as_tuple(after)

        async def run(inputs: dict[str, Any], goal: str, ctx: Ctx) -> tuple[Any, Usage]:
            p = prompt(inputs, goal) if prompt else _default_prompt(inputs, goal)
            res = await coordinator.run(p, ctx.child())
            # AgentResult — output is the last assistant reply by construction
            last = getattr(res, "output", "") or ""
            return last, res.usage

        return self._add(_Node(name, "coordinator", deps, run))

    def human_gate(self, name: str, *, after: Any = ()) -> str:
        """A node that suspends for a human decision; its output is the decision passed to `resume`."""

        async def run(
            inputs: dict[str, Any], goal: str, ctx: Ctx
        ) -> tuple[Any, Usage]:  # never called directly — the engine injects the decision
            return None, Usage()

        return self._add(_Node(name, "human_gate", _as_tuple(after), run, gate=True))

    def subworkflow(self, name: str, child: Workflow, *, after: Any = ()) -> str:
        deps = _as_tuple(after)

        async def run(inputs: dict[str, Any], goal: str, ctx: Ctx) -> tuple[Any, Usage]:
            res = await child.run(_default_prompt(inputs, goal), ctx.child())
            # Output the child's `outputs` dict (serializable + indexable downstream), NOT the
            # WorkflowResult object — so a checkpoint survives a real (serializing) store, not just
            # InMemoryStore. See the durable-resume contract in the module docstring.
            return res.outputs, res.usage

        return self._add(_Node(name, "subworkflow", deps, run))

    def map(
        self,
        name: str,
        *,
        over: Callable[..., Any],
        each: Callable[..., Any],
        after: Any = (),
        bounded_by: int | None = None,
        best_effort: bool = False,
        prompt: Callable[[Any, str], str] | None = None,
        key: Callable[[Any], str] | None = None,
    ) -> str:
        """ONE node that expands into N element runs, with N decided at RUNTIME.

        ``over(inputs[, goal])`` returns the collection (any iterable — it is
        materialised once, so a generator is safe); ``each(item[, index])`` returns
        the worker for one element. The node's output is the list of element outputs
        in expansion order, so a dependent reads it exactly like any other node's::

            wf.map("implement", over=lambda d: d["plan"].requirements,
                   each=lambda item: agent_for(item), after="plan", bounded_by=4)
            wf.human_gate("review", after="implement")

        **Why this belongs in the framework rather than in an application's ``fn`` node.**
        An application can already fan out inside a plain ``fn``. What it cannot do is
        make that fan-out RESUMABLE, because the engine's durable record is the ``done``
        map and a hand-rolled fan-out is opaque to it: a run that dies after eight of ten
        elements resumes by re-running all ten. So a map records its EXPANSION, not only
        its result:

        * ``done["impl[i]"]`` — one entry per FINISHED element, written as it finishes,
          so a later attempt skips it. A failed element (``best_effort``) is deliberately
          NOT recorded, because a resume must retry it.
        * ``done["impl#expansion"]`` — the ordered identity list, written BEFORE any
          element runs. This is the half that is easy to leave out and impossible to add
          later: without it, resume has element results keyed by position and no way to
          know the positions still mean the same things. With it, a drifted expansion
          raises :class:`MapExpansionChanged` instead of threading element 2's output
          into element 0's slot.

        Both keys are visible in ``WorkflowResult.outputs``; that is the point — "which
        elements finished" is the question you have at 3am.

        **Determinism is the caller's half of the contract.** ``over`` must return the
        same collection in the same order given the same inputs. It usually does for
        free (its inputs come from ``done``, which the checkpoint restored), but a ``set``
        comprehension or a re-queried table will not. Identity defaults to ``str(item)``;
        pass ``key=`` when your items are objects whose ``repr`` embeds an address, or
        when ``str`` is large.

        **Concurrency reuses the spine, one level down.** Elements run under
        ``gather_bounded`` against ``ctx.child().semaphore()`` — the pool at depth+1, NOT
        the wave's own pool. That is not a stylistic choice: the wave already holds a
        permit at its depth for this very node, so drawing elements from the same pool
        deadlocks at ``max_concurrency=1`` and at any cap once the ancestors' outstanding
        permits reach it. See ``Budget.semaphore`` for the full argument. ``bounded_by``
        adds a node-local width on TOP of that, acquired first so a narrow map cannot sit
        on level permits it is not using.

        ``ctx.check_cancelled()`` runs at the top of every element, so an abort stops the
        expansion at the next element boundary rather than after all N. A map counts as
        ONE step against ``max_steps`` — the graph grew wider, not longer.

        ``each`` may return: a runnable (anything with ``run(task, ctx)`` — an ``Agent``,
        a coordinator, a ``Workflow``); an awaitable (``each=lambda i: work(i)`` over an
        async ``work``); a callable (invoked ``worker(item[, goal])``); or, if none of
        those, the element's RESULT itself (``each=lambda i: i.upper()``). The one shape
        to avoid is returning a callable you meant as data — it will be called.

        ``prompt(item, goal)`` builds the task string for the RUNNABLE shape only (the
        default is ``_default_prompt({"item": item}, goal)``); the other three shapes are
        handed the item itself and have nowhere to put a prompt. Passing ``prompt=`` with
        a non-runnable element raises rather than dropping it — a prompt the author wrote
        and the framework ignored is a wrong run that reports success.
        """
        if bounded_by is not None and bounded_by < 1:
            # Refuse at CONSTRUCTION, where it is free to fix — the same rule the
            # meters apply to a ceiling. A width of zero is a fan-out that reserves
            # nothing and can never run: the "looks like it ran and did nothing"
            # failure ``_at_least_one`` exists to prevent in ``kernel.concurrency``.
            raise ValueError(
                f"map node {name!r}: bounded_by must be >= 1, got {bounded_by}. "
                "Omit it for 'as wide as the tree semaphore allows'."
            )
        deps = _as_tuple(after)
        identity = key if key is not None else _default_identity
        exp_key = f"{name}{_EXPANSION_SUFFIX}"

        async def _record_partial(ctx: Any, goal: str, done: dict[str, Any], steps: int) -> None:
            """Persist how far the expansion got, then let the failure through.

            Without this a map is resumable in principle and never in practice: the
            wave's ``gather_bounded`` cancels the siblings and the exception unwinds
            out of ``_execute``, taking the live ``done`` — and every finished element
            in it — with it. The human-gate suspend is the only other writer, and a map
            that failed has not reached one.

            ``RUNNING``, not ``SUSPENDED``: nothing is waiting on a human. It is also
            not terminal, so ``Checkpointer.resume``'s default filter still hands it
            back. Exceptions from the checkpointer itself are swallowed on purpose —
            record-keeping must never replace the failure the caller needs to see.
            """
            cp = resolve_checkpointer(ctx)
            if cp is None:
                return
            with contextlib.suppress(Exception):
                await cp.snapshot(
                    ctx.correlation_id,
                    {"goal": goal, "done": done, "steps": steps},
                    status=CheckpointStatus.RUNNING,
                    ctx=ctx,
                )

        async def expand(
            inputs: dict[str, Any], goal: str, ctx: Any, done: dict[str, Any], steps: int
        ) -> tuple[Any, Usage]:
            # Materialise ONCE. ``over`` may hand back a generator, and the expansion is
            # read at least twice (identities, then elements) — a second pass over a
            # consumed iterator yields nothing, which is a map that silently expands to
            # width zero and a downstream node that silently sees an empty list.
            items = tuple(over(inputs, goal) if _arity(over) >= 2 else over(inputs))
            identities = [identity(it) for it in items]
            recorded = done.get(exp_key)
            if recorded is not None and list(recorded) != identities:
                raise MapExpansionChanged(
                    f"map node {name!r} expanded to {identities!r}, but its checkpoint "
                    f"recorded {list(recorded)!r}. Element results are keyed by position, "
                    "so reusing them here would thread one element's output into "
                    "another's slot. Make ``over`` deterministic, pass ``key=``, or start "
                    "the run fresh."
                )
            # Written BEFORE any element runs, so a failure two elements in still leaves
            # a checkpoint that knows what the expansion WAS.
            done[exp_key] = identities
            # The one fact about this run that is NOT in the authored graph. A reader
            # tailing observations sees every other node's shape in the source; the
            # width of a map is only knowable here, and "did it expand to 3 or to
            # 300?" is the first question asked of a run that cost more than expected.
            await ctx.emit(
                "progress",
                f"{name} expanded to {len(items)}",
                agent=name,
                payload={"node": name, "elements": len(items)},
            )

            # depth+1, for both the permit pool and the element contexts — see the
            # concurrency paragraph in the docstring.
            child = ctx.child()
            level_sem = child.semaphore()
            width_sem = asyncio.Semaphore(bounded_by) if bounded_by is not None else None

            async def _work(index: int, item: Any, slot: str) -> tuple[Any, Usage]:
                worker = each(item, index) if _arity(each) >= 2 else each(item)
                runner = getattr(worker, "run", None)
                usage = Usage()
                if prompt is not None and not callable(runner):
                    # ``prompt`` only has a receiver when the element is a RUNNABLE —
                    # the other three shapes are handed the item, not a task string.
                    # Dropping it silently is the worst option available: the author
                    # wrote a prompt, the framework ran something else, and nothing
                    # anywhere says so. ``each`` resolves per element, so this is the
                    # first moment the mismatch is knowable; saying it loudly here beats
                    # a run that quietly ignored half its configuration.
                    raise ValueError(
                        f"map node {name!r}: prompt= was given, but element {index} "
                        f"resolved to {type(worker).__name__}, which takes no prompt — "
                        "only a runnable (anything with ``run(task, ctx)``) is handed "
                        "one. Drop prompt=, or return a runnable from each=."
                    )
                if callable(runner):
                    p = (
                        prompt(item, goal)
                        if prompt is not None
                        else _default_prompt({"item": item}, goal)
                    )
                    res = await runner(p, child.child())
                    out = getattr(res, "output", None)
                    if out is None and hasattr(res, "outputs"):
                        # A child ``Workflow``: hand on its ``outputs`` dict for the same
                        # reason ``subworkflow`` does — a ``WorkflowResult`` object does
                        # not survive a serializing store, and this value goes into
                        # ``done`` and therefore into every later checkpoint.
                        out = res.outputs
                    usage = getattr(res, "usage", None) or Usage()
                elif inspect.isawaitable(worker):
                    out = await worker
                elif callable(worker):
                    out = worker(item, goal) if _arity(worker) >= 2 else worker(item)
                    if inspect.isawaitable(out):
                        out = await out
                else:
                    out = worker
                # Commit PER ELEMENT, not once per wave. This is the line a resume
                # depends on: when element 3 raises, elements 0-2 are already in ``done``
                # and ``_record_partial`` has something worth persisting.
                done[slot] = out
                return out, usage

            async def _element(index: int, item: Any) -> tuple[Any, Usage]:
                slot = f"{name}[{index}]"
                if slot in done:  # a previous attempt finished this one
                    return done[slot], Usage()
                ctx.check_cancelled()
                if width_sem is not None:
                    async with level_sem:
                        return await _work(index, item, slot)
                return await _work(index, item, slot)

            coros: list[Awaitable[tuple[Any, Usage]]] = [
                _element(i, it) for i, it in enumerate(items)
            ]
            # ``bounded_by`` is the OUTER bound and the level pool the inner one, so a
            # narrow map waits on its own permit before taking a level permit it would
            # only sit on. The reverse nesting is deadlock-free too, but it lets a
            # ``bounded_by=1`` map hold every level permit while running one element,
            # starving a sibling map in the same wave.
            outer = width_sem if width_sem is not None else level_sem
            try:
                raw: list[Any] = (
                    list(await gather_best_effort(coros, sem=outer))
                    if best_effort
                    else list(await gather_bounded(coros, sem=outer))
                )
            except BaseException:
                await _record_partial(ctx, goal, done, steps)
                raise

            outputs: list[Any] = []
            total = Usage()
            for r in raw:
                if isinstance(r, Failure):  # best_effort — the slot IS the failure
                    outputs.append(r)
                    continue
                out, u = r
                outputs.append(out)
                total = total + u
            return outputs, total

        async def run(inputs: dict[str, Any], goal: str, ctx: Ctx) -> tuple[Any, Usage]:
            # Unreachable: ``_execute`` prefers ``expand`` for any node that has one.
            # Present because ``_Node.run`` is not optional, and a named error beats a
            # ``None`` call if a future refactor ever routes a map through here.
            raise RuntimeError(f"map node {name!r} must be executed through ``expand``")

        return self._add(_Node(name, "map", deps, run, expand=expand))

    def route(self, from_: str, *, when: Callable[[Any], bool], to: str) -> Workflow:
        """Conditional edge: after `from_` runs, if `when(output)` is true, (re)activate `to`. A route to
        an ancestor is a **bounded loop-back** (guarded by `max_steps`)."""
        if from_ not in self._nodes or to not in self._nodes:
            raise KeyError("route endpoints must be existing nodes")
        self._routes.setdefault(from_, []).append((when, to))
        return self

    # ---- engine ----------------------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Every name in every node's ``after`` must be a node in this graph.

        ``route()`` has always refused an endpoint that names nothing. ``after``
        — the other way to name a node, and the one written far more often — did
        not, so a single typo produced a graph whose dependent nodes could never
        become ready: their dependency was never going to enter ``done``. The
        run completed with a partial ``outputs`` map and ``stop_reason
        "deadlock"``, which named the symptom but not the cause, and only for a
        caller who thought to look past ``outputs``.

        Checked HERE rather than in ``_add`` because ``after`` may legitimately
        forward-reference a node built later — declaration order is not
        execution order, and graphs written that way work. ``_execute`` is the
        first moment the graph is complete, and it is still before any node has
        run, so the failure is total rather than partial.

        Raises ``KeyError`` to match what ``route()`` already raises for the
        same class of mistake."""
        for name in self._order:
            unknown = [dep for dep in self._nodes[name].after if dep not in self._nodes]
            if unknown:
                known = ", ".join(sorted(self._nodes)) or "<none>"
                raise KeyError(
                    f"node {name!r} depends on unknown node(s) {unknown!r}; "
                    f"this workflow defines: {known}"
                )

    def _forward_closure(self, start: str) -> set[str]:
        """`start` plus every node that (transitively) depends on it — what a loop-back must re-run."""
        rev: dict[str, list[str]] = {}
        for n in self._nodes.values():
            for d in n.after:
                rev.setdefault(d, []).append(n.name)
        seen: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(rev.get(cur, []))
        return seen

    async def run(
        self,
        goal: str,
        ctx: Ctx,
        *,
        decisions: dict[str, Any] | None = None,
        on_existing: OnExisting = "start_fresh",
    ) -> WorkflowResult:
        """Execute the workflow to completion or a human-gate suspend.

        ``on_existing`` controls what happens when a checkpoint already
        exists under ``ctx.correlation_id`` on ``ctx.checkpointer``:

        * ``"start_fresh"`` (default, preserved historical behaviour) —
          ignore any existing checkpoint and run from step 1. Callers
          who want silent overwrite (the old default) keep the default.
        * ``"resume"`` — consult ``ctx.checkpointer.resume(run_id)``
          (which itself filters terminal ``DONE``/``FAILED`` snapshots
          by default). If a resumable checkpoint exists, replay from
          it; otherwise start fresh. The checkpoint's ``state`` is
          expected to carry ``{"goal", "done", "steps"}`` (the shape
          the Workflow itself writes at a human-gate suspend), so a
          checkpoint written by a previous ``run`` / ``resume`` picks
          up cleanly.
        * ``"fail"`` — consult the checkpointer for ANY snapshot
          (terminal or not) and raise ``CheckpointerError`` if one
          exists. This is the idempotency-guard mode: use when the
          caller must not silently re-run a job that already has
          persisted state.

        When ``ctx.checkpointer`` is ``None``, the non-``start_fresh``
        modes degrade cleanly: ``"fail"`` cannot detect a prior run so
        it proceeds; ``"resume"`` has nowhere to resume from and
        starts fresh.
        """
        if on_existing != "start_fresh":
            cpt = getattr(ctx, "checkpointer", None)
            if cpt is not None:
                run_id = ctx.correlation_id
                if on_existing == "fail":
                    # ``include_terminal=True`` so a completed / failed
                    # snapshot still trips the guard — the intent is
                    # "this run_id has ANY persisted state".
                    existing = await cpt.resume(run_id, include_terminal=True)
                    if existing is not None:
                        raise CheckpointerError(f"run {run_id!r} already exists")
                elif on_existing == "resume":
                    # Default terminal-filter applies — a DONE/FAILED
                    # checkpoint is NOT resumable and we fall through
                    # to the fresh path.
                    cp = await cpt.resume(run_id)
                    if cp is not None:
                        state = cp.state
                        return await self._execute(
                            state.get("goal", goal),
                            ctx,
                            # ``dict(...)`` around the deepcopy, not just the
                            # deepcopy: the restored map comes off a Checkpoint
                            # whose payload is now deep-frozen, and
                            # ``FrozenDict.__deepcopy__`` faithfully returns
                            # another FrozenDict — so the "mutable working copy"
                            # this line exists to produce was itself frozen and
                            # the ``done[node.name] = out`` commit raised.
                            # Unwrapping the TOP level restores the working
                            # copy. Node OUTPUTS stay frozen, which is right:
                            # they came from a durable record, and
                            # ``WorkflowResult`` re-freezes them on the way out.
                            done=dict(copy.deepcopy(state.get("done", {}))),
                            decisions=dict(decisions or {}),
                            steps=state.get("steps", 0),
                        )
        return await self._execute(goal, ctx, done={}, decisions=dict(decisions or {}), steps=0)

    async def resume(self, run_id: str, decisions: dict[str, Any], ctx: Ctx) -> WorkflowResult:
        """Continue a gate-suspended run. Reads through the same seam
        ``_execute`` wrote to — see ``resolve_checkpointer``."""
        cp = resolve_checkpointer(ctx)
        state = None
        if cp is not None:
            checkpoint = await cp.resume(run_id)
            state = checkpoint.state if checkpoint is not None else None
        if not state:
            raise ValueError(f"no suspended workflow {run_id!r} to resume")
        saved = state
        # Deep-copy the persisted ``done`` map. ``InMemoryStore`` returns
        # the same object reference it was handed at ``set`` time, so a
        # shallow ``dict(saved["done"])`` would leave inner values
        # (lists, dicts) aliased between the live workflow state (which
        # is exposed to callers as ``WorkflowResult.outputs``) and the
        # persisted checkpoint. A caller mutating a node output
        # post-resume would then corrupt the record. Deep-copy at the
        # store boundary keeps the persisted view immutable from the
        # caller's perspective.
        result = await self._execute(
            saved["goal"],
            ctx,
            # Same reason as the ``on_existing="resume"`` path above: unwrap
            # the top level so the wave loop can commit into it.
            done=dict(copy.deepcopy(saved["done"])),
            decisions=dict(decisions),
            steps=saved["steps"],
        )
        if result.stop_reason != "suspended" and cp is not None:
            # Terminal → reclaim the checkpoint so a naive "resume if anything
            # exists" wiring cannot replay a finished run. There is ONE seam to
            # clear: ``resolve_checkpointer``. The gate-suspend path in
            # ``_execute`` writes through it and nowhere else, so whatever this
            # resume read is the only record of the run.
            #
            # The record the gate wrote is marked SUSPENDED, and nothing
            # downgrades it on the way out — so leaving it behind does not just
            # waste a slot, it leaves a FINISHED run advertising itself as
            # resumable. Measured by disabling this branch on a
            # prep -> gate -> act -> ship graph: after the run completed, a
            # second ``resume(run_id, {"approve": "yes"})`` returned
            # ``complete`` again with ``act`` and ``ship`` each executed twice
            # (2 instead of 1), and ``run(..., decisions=..., on_existing="resume")``
            # re-executed both the same way. Deleting is what makes the second
            # call raise "no suspended workflow ... to resume" instead.
            await cp.delete(run_id)
        return result

    async def _execute(
        self,
        goal: str,
        ctx: Ctx,
        *,
        done: dict[str, Any],
        decisions: dict[str, Any],
        steps: int,
    ) -> WorkflowResult:
        self._validate_dependencies()
        usage = Usage()
        pending = {name for name in self._order if name not in done}

        with ctx.trace.span(
            "invoke_workflow",
            "client",
            **{
                "gen_ai.workflow.name": self.name,
                "gen_ai.workflow.autonomy": getattr(ctx, "autonomy", None) or "auto",
                "agentkit.workflow.nodes_count": len(self._nodes),
                "agentkit.workflow.id": id(self),
            },
        ):
            await ctx.emit(
                "run_start", f"workflow {self.name}", payload={"nodes": len(self._nodes)}
            )
            while pending:
                ctx.check_cancelled()
                ready = [
                    self._nodes[n]
                    for n in self._order
                    if n in pending and all(d in done for d in self._nodes[n].after)
                ]
                if not ready:
                    await ctx.emit(
                        "error", "workflow deadlocked", payload={"pending": sorted(pending)}
                    )
                    return WorkflowResult(done, usage, steps, "deadlock")

                # Cycle/size backstop — checked BEFORE the wave, once per wave.
                # It used to be checked after the wave, guarded by ``and pending``,
                # which let two families of run escape the bound:
                #   * a self-route (``route("a", …, to="a")``) leaves ``pending``
                #     EMPTY at the post-wave check — the route that re-arms ``a``
                #     is evaluated a few lines later — so the guard never fired.
                #     Measured: max_steps=20, node ran 39,782 times in 3s and never
                #     terminated. A route to a genuine ancestor (``b → a``) happened
                #     to be bounded only because its wave left a sibling pending.
                #   * ``max_steps=0`` ran one node (measured steps=1) because no
                #     check preceded the first wave.
                # ``pending`` is non-empty by the ``while`` condition, so the guard
                # needs no extra qualifier here: reaching the top of the loop with
                # the budget spent and work left IS the max_steps stop.
                if steps >= self.max_steps:
                    await ctx.emit(
                        "error", "workflow hit max_steps", payload={"max": self.max_steps}
                    )
                    return WorkflowResult(done, usage, steps, "max_steps")

                gate = next((n for n in ready if n.gate and n.name not in decisions), None)
                if gate is not None:  # suspend for a human decision
                    run_id = ctx.correlation_id
                    cp = resolve_checkpointer(ctx)
                    if cp is None:
                        # Returning a ``Suspended`` we could not persist is a
                        # silent, well-formed failure: ``run()`` reports a
                        # resumable state, and the truth only emerges later —
                        # usually in a different process — as
                        # "no suspended workflow <id> to resume", with nothing
                        # pointing back at the missing seam.
                        _warn_unpersisted_gate(gate.name, run_id)
                    else:
                        # ``Checkpointer.snapshot`` deep-copies ``state`` at the
                        # seam, so the live ``done`` map (handed to the caller as
                        # ``WorkflowResult.outputs``) cannot alias the persisted
                        # record. Marked SUSPENDED so auto-resume can tell
                        # "waiting on a human" from "engine in motion" — the
                        # status this path could not express while it wrote
                        # through a bare KV.
                        await cp.snapshot(
                            run_id,
                            {"goal": goal, "done": done, "steps": steps},
                            status=CheckpointStatus.SUSPENDED,
                            ctx=ctx,
                        )
                    await ctx.emit(
                        "interrupt", f"awaiting decision: {gate.name}", payload={"gate": gate.name}
                    )
                    susp = Suspended(
                        run_id=run_id, pending=(gate.name,), reason="awaiting_decision"
                    )
                    return WorkflowResult(done, usage, steps, "suspended", suspended=susp)

                # ``at_step`` is a DEFAULT ARGUMENT, not a closure read, and that is
                # load-bearing: a map checkpoints its partial progress against this
                # number, while the commit loop below reassigns ``steps`` as the wave
                # lands. A late-bound read would stamp the resume point with a count
                # that includes siblings the failing wave never committed, so a resume
                # would start from a step number no snapshot corresponds to. The
                # default binds the PRE-wave value, once, when the wave starts.
                async def _one(node: _Node, at_step: int = steps) -> tuple[Any, Usage]:
                    if node.gate:
                        return decisions[node.name], Usage()
                    inputs = {d: done[d] for d in node.after}
                    if node.expand is not None:
                        # A ``map``: it needs the live ``done`` (to skip elements a
                        # previous attempt already finished) and a step count to
                        # checkpoint partial progress against.
                        return await node.expand(inputs, goal, ctx, done, at_step)
                    return await node.run(inputs, goal, ctx)

                outs = await gather_bounded([_one(n) for n in ready], sem=ctx.semaphore())
                # ``strict=True`` ensures the wave length matches the
                # result length. A silent truncation from
                # ``gather_bounded`` (custom semaphore returning fewer
                # items than ``ready``) would drop a sibling's output
                # with no error, leaving ``done`` / ``pending``
                # inconsistent. Also validate each result is a 2-tuple —
                # a node ``fn`` that forgets to return ``Usage`` would
                # otherwise raise a confusing tuple-unpacking ValueError
                # deep inside the loop; the structured message here
                # names the offending node.
                if len(outs) != len(ready):
                    raise RuntimeError(
                        f"Workflow wave returned {len(outs)} results for "
                        f"{len(ready)} ready nodes (names: "
                        f"{[n.name for n in ready]}). gather_bounded "
                        "must preserve order + arity."
                    )
                for node, raw in zip(ready, outs, strict=True):
                    if not (isinstance(raw, tuple) and len(raw) == 2):
                        raise RuntimeError(
                            f"Workflow node {node.name!r} returned "
                            f"{type(raw).__name__} (expected (output, Usage) "
                            "tuple). Every ``fn`` must return both."
                        )
                    out, u = raw
                    done[node.name] = out  # commit the WHOLE wave first —
                    usage = usage + u  # never drop a sibling whose work
                    pending.discard(node.name)  # (and budget) was already spent
                    steps += 1
                    await ctx.emit(
                        "summary", f"{node.name} done", agent=node.name, payload={"step": steps}
                    )
                # Snapshot each node's output BEFORE evaluating routes: a route whose
                # forward-closure contains its own source (a self-route, or any loop-back
                # onto an ancestor of the source) pops that entry from ``done``, so a
                # SECOND route from the same node then read a deleted key. Measured:
                # a node with ``route(a→a)`` plus ``route(a→other)`` raised
                # ``KeyError: 'a'`` mid-wave. Routing decides on the outputs the wave
                # produced, not on what an earlier route has since cleared.
                wave_outputs = {n.name: done[n.name] for n in ready}
                for node in ready:  # conditional routes, after the wave
                    for when, to in self._routes.get(node.name, []):
                        if when(wave_outputs[node.name]):
                            for name in self._forward_closure(
                                to
                            ):  # loop-back: clear + re-run downstream
                                done.pop(name, None)
                                pending.add(name)
                                if self._nodes[
                                    name
                                ].gate:  # a gate in a loop must RE-prompt the human,
                                    decisions.pop(
                                        name, None
                                    )  # not silently reuse the stale decision
                                if self._nodes[name].expand is not None:
                                    # A map's element results and expansion record live
                                    # in ``done`` under their OWN keys, so popping the
                                    # node name alone leaves them behind. The re-run then
                                    # re-expands against a stale record: a loop whose
                                    # collection shrinks raises ``MapExpansionChanged``,
                                    # and one whose collection is unchanged reuses every
                                    # element result — a "loop" that recomputes nothing.
                                    _clear_map_records(done, name)

            await ctx.emit("result", "workflow complete", payload={"steps": steps})
            return WorkflowResult(done, usage, steps, "complete")


__all__ = ["MapExpansionChanged", "Workflow", "WorkflowResult"]
