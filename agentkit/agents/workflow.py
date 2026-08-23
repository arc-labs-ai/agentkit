"""Workflow — explicit control: a developer-authored typed graph.

The deterministic counterpart to the emergent team loop. You author the path as a graph of named
**nodes** with data-dependency **edges** (`after`) and optional **conditional routes** (branch / bounded
loop-back). The engine topologically schedules: each wave runs every *ready* node (all deps satisfied)
concurrently under the tree semaphore, threads each node's typed output to its dependents, then evaluates
routes. It reuses the whole spine — `gather_bounded`/`ctx.semaphore()` for concurrency, `ctx.check_cancelled()`
for abort, `ctx.emit()` for observation, the run `Budget` for cost — and the same `Suspended`/`ctx.store`
suspend/resume the loop uses for **human-gate** nodes. Bounded by `max_steps` so a cycle can never
run forever.

Node kinds: `agent` · `coordinator` · `fn` (pure) · `tool` · `human_gate` · `subworkflow`. `run()`
executes to completion or a suspend; `resume(run_id, decisions, ctx)` continues from a human gate.

**Durable-resume contract.** A human-gate suspend checkpoints `{goal, done, steps}` to `ctx.store`, where
`done` maps each completed node to its output. For resume to survive a *real* (serializing) store — not
just `InMemoryStore`, which keeps live objects — those outputs must be serializable by that store. The
built-in node kinds satisfy this (`agent`/`coordinator` → str, `subworkflow` → the child's `outputs`
dict); a custom `fn`/`tool` node whose output crosses a gate must likewise return a serializable value.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from agentkit.agents.result import Suspended, WorkflowResult
from agentkit.capabilities.checkpointer import resolve_checkpointer
from agentkit.kernel.concurrency import gather_bounded
from agentkit.kernel.errors import CheckpointerError
from agentkit.kernel.ports import CheckpointStatus
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolRequest, Usage

OnExisting = Literal["fail", "resume", "start_fresh"]


@dataclass
class _Node:
    name: str
    kind: str
    after: tuple[str, ...]
    run: Callable[
        [dict[str, Any], str, Any], Awaitable[tuple[Any, Usage]]
    ]  # (inputs, goal, ctx) -> (output, usage)
    gate: bool = False


def _ckpt_key(run_id: str) -> str:
    return f"workflow:{run_id}"


async def _read_legacy_checkpoint(ctx: Ctx, run_id: str) -> dict[str, Any] | None:
    """Read a suspend written by a version that persisted straight to ``ctx.store``.

    Workflow used to write ``{"goal", "done", "steps"}`` at ``workflow:<run_id>``
    on the raw store, while the checkpointer bridge writes at
    ``checkpoint:<run_id>``. Switching seams without this would orphan every
    IN-FLIGHT human gate across an upgrade — and a suspended run is precisely
    the state that cannot be reconstructed, because it is waiting on a person.
    Read-only and best-effort; safe to delete once no old suspends can remain.
    """
    store = getattr(ctx, "store", None)
    if store is None:
        return None
    with contextlib.suppress(Exception):
        legacy = await store.get(_ckpt_key(run_id))
        if legacy:
            return dict(legacy)
    return None


async def _delete_legacy_checkpoint(ctx: Ctx, run_id: str) -> None:
    """Reclaim a legacy-format checkpoint on terminal completion."""
    store = getattr(ctx, "store", None)
    if store is None:
        return
    with contextlib.suppress(Exception):
        await store.delete(_ckpt_key(run_id))


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
        self._nodes[node.name] = node
        self._order.append(node.name)
        return node.name

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

    def route(self, from_: str, *, when: Callable[[Any], bool], to: str) -> Workflow:
        """Conditional edge: after `from_` runs, if `when(output)` is true, (re)activate `to`. A route to
        an ancestor is a **bounded loop-back** (guarded by `max_steps`)."""
        if from_ not in self._nodes or to not in self._nodes:
            raise KeyError("route endpoints must be existing nodes")
        self._routes.setdefault(from_, []).append((when, to))
        return self

    # ---- engine ----------------------------------------------------------------------------------

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
        if state is None:
            state = await _read_legacy_checkpoint(ctx, run_id)
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
        if result.stop_reason != "suspended":
            # Terminal → reclaim the checkpoint so a naive
            # "resume if anything exists" wiring cannot replay a finished run.
            # Both seams are cleared: the run may have been suspended by an
            # older version that wrote the legacy KV key.
            if cp is not None:
                await cp.delete(run_id)
            await _delete_legacy_checkpoint(ctx, run_id)
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

                async def _one(node: _Node) -> tuple[Any, Usage]:
                    if node.gate:
                        return decisions[node.name], Usage()
                    return await node.run({d: done[d] for d in node.after}, goal, ctx)

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

            await ctx.emit("result", "workflow complete", payload={"steps": steps})
            return WorkflowResult(done, usage, steps, "complete")


__all__ = ["Workflow", "WorkflowResult"]
