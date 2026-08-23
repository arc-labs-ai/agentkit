"""WorkingContext — an agent's working memory.

Distinct from ``RunContext`` (which is *execution wiring* — identity,
budget meters, services). ``WorkingContext`` is *what the agent
knows right now* — a unified state object combining four axes:

* **Transcript axis** — a cache-stable ``PrefixContext`` (system +
  grounding) plus a mutable tail of ``messages``. The shape every
  LLM call sees.
* **Scratchpad axis** — a free-form dict of cross-step notes. The
  agent's working notepad.
* **Journal axis** — an append-only ``MutationJournal`` of typed
  authored history with a watermark. Used by hierarchical agents
  to stream progress upward without re-shipping mutations the
  parent has already absorbed.

Long-term recall over a vector / KV store is no longer a
``WorkingContext`` slot — it lives on the agent as
``Agent.memory: MemorySource | None``. The cognition queries memory
and grounds the prompt via the ``RequestBuilder`` seam.

The three axes above are orthogonal — agents use what they need. A
single-shot chat agent uses transcript + scratchpad only. A
long-lived sub-agent uses all four.

This module owns three closely-related types — ``WorkingContext``,
``FrozenContext`` (its immutable snapshot), ``ContextDiff`` (its
structural delta), and ``MutationJournal`` (the journal axis) — all
in one place. These classes co-evolve; splitting them across modules
would be bookkeeping rather than separation of concerns.

This module is intentionally narrow: pure data operations only — no
LLM calls, no retrieval, no compaction. Callers compose those around
it explicitly. Side-effects live with their owners (``Compactor``,
``Retriever``, ``Invoker``).

Concurrency model. Private by default (``fork()`` gives an
independent copy, ``merge()`` aggregates a child's knowledge into a
parent). A multi-agent team shares one ``WorkingContext`` as a
blackboard by passing it as ``context=``. The simple mutators
(``append`` / ``extend`` / ``note`` / ``update_scratchpad`` /
``clear_messages``) are sync and GIL-atomic — concurrent coroutines
on a single event loop can't interleave inside any one call. For
MULTI-step atomicity (a sequence of writes with an ``await`` in the
middle, or a read-modify-write across an await), set ``shared=True``
and wrap the sequence in ``async with ctx.lock:``  or pass a closure
to ``await ctx.apply_locked(fn)``. Mutators do NOT auto-lock on
``shared=True``: the lock is exposed as a public primitive and the
boundary is explicit.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar

from agentkit.context.prefix import PrefixContext
from agentkit.context.tokens import ApproxTokenCounter, TokenCounter
from agentkit.kernel.types import Message

if TYPE_CHECKING:
    from agentkit.context.scope import ContextScope

# ── Type variable for the journal axis ──────────────────────────────

JournalEntryT = TypeVar("JournalEntryT")
"""The type of entries stored in the journal axis. Project code
parameterises this with its own mutation/event union (typically a
discriminated union of domain mutation kinds, each stamped with
``agent_id`` for attribution that survives merges)."""


# ── MutationJournal — the journal axis ──────────────────────────────


class MutationJournal(Generic[JournalEntryT]):
    """Append-only journal of typed entries + watermark for streaming.

    Used by hierarchical agents to stream progress upward without
    re-shipping entries the parent has already absorbed. The watermark
    tracks how far the consumer (parent) has acknowledged:

    - ``record(entry)`` / ``apply(entries)`` — append; never rewrites.
    - ``diff()`` — uncommitted suffix (past the watermark).
    - ``mark_committed()`` — advance watermark; called after a parent
      has absorbed the diff.
    - ``view()`` — full read-only view (committed + uncommitted).

    Attribution invariant: callers may stamp ``agent_id`` (or any
    other field) on entries before recording. The journal never
    rewrites entries — author attribution survives every merge into
    a parent's journal.

    No locking — the journal is private to ONE agent. Cross-agent
    merging goes through signal protocols, not the journal directly.
    """

    __slots__ = ("_committed_index", "_entries")

    def __init__(self, initial: Iterable[JournalEntryT] | None = None) -> None:
        """Build a journal, optionally seeded with prior entries.

        Seed entries are treated as already-committed (watermark
        starts past them) — typical use is re-hydrating a journal
        from a persisted snapshot where the entries have already
        been observed by the parent.
        """
        self._entries: list[JournalEntryT] = list(initial) if initial is not None else []
        self._committed_index: int = len(self._entries)

    def record(self, entry: JournalEntryT) -> None:
        """Append one entry. Journal-semantics — never rewritten."""
        self._entries.append(entry)

    def apply(self, entries: Iterable[JournalEntryT]) -> None:
        """Merge a batch of entries into the journal in arrival order.

        Used by the parent's merge loop to absorb a child's diff: each
        incoming entry keeps whatever attribution it already carries
        so the audit trail survives.

        Idempotency: callers MUST not double-apply the same diff. The
        journal trusts the merge loop's ack-style protocol.
        """
        self._entries.extend(entries)

    def view(self) -> tuple[JournalEntryT, ...]:
        """All entries in record order (committed + uncommitted)."""
        return tuple(self._entries)

    def diff(self) -> list[JournalEntryT]:
        """Uncommitted suffix — everything appended past the watermark."""
        return list(self._entries[self._committed_index :])

    def has_uncommitted(self) -> bool:
        """True if there's anything past the watermark."""
        return self._committed_index < len(self._entries)

    def size(self) -> int:
        """Total entries recorded so far."""
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def mark_committed(self) -> None:
        """Advance the watermark past every entry currently in the
        journal. Called after a parent has acknowledged absorbing the
        diff (or after the final-delta has been shipped)."""
        self._committed_index = len(self._entries)

    @property
    def committed_index(self) -> int:
        """Diagnostic — how many entries have been ack'd."""
        return self._committed_index


# ── ContextDiff — structural delta between two snapshots ────────────


@dataclass(frozen=True)
class ContextDiff:
    """A structural diff between two ``WorkingContext`` snapshots.

    Returned by ``WorkingContext.diff(other)``. The shape is
    intentionally minimal — added / removed message tuples and a flat
    scratchpad-changes dict.

    ``messages_added`` / ``messages_removed`` are tuples (frozen) so
    a ``ContextDiff`` is itself hashable / pickleable. Order is
    preserved from the source contexts.

    ``scratchpad_changes`` is a dict mapping each changed key to its
    new value in ``self`` (the right-hand side). A key present in
    ``other`` but absent in ``self`` appears here with value ``None``
    to signal removal; ambiguity with a ``None``-valued live key is
    accepted for a debug helper.

    ``prefix_changed`` is True when the two prefixes compare unequal.
    """

    messages_added: tuple[Message, ...]
    messages_removed: tuple[Message, ...]
    scratchpad_changes: dict[str, Any]
    prefix_changed: bool


# ── FrozenContext — immutable snapshot ──────────────────────────────


@dataclass(frozen=True)
class FrozenContext:
    """Immutable snapshot of a ``WorkingContext`` at a point in time.

    Use when handing a context across an agent boundary without
    sharing mutation: a parent that wants to brief a child but
    doesn't want the child's writes to leak back.

    All fields are immutable: ``PrefixContext`` is frozen,
    ``messages`` is a tuple, ``scratchpad`` is a sorted tuple of
    (key, value) pairs (so equality is deterministic), and
    ``journal_entries`` is a tuple. A snapshot is therefore safe to
    share across agent boundaries without locking and to use as a
    memoization-cache key — see ``__hash__`` for what "hashable" can
    and cannot mean when two of those tuples hold arbitrary
    user objects.

    Round-tripping: ``WorkingContext(prefix=f.prefix,
    messages=list(f.messages), scratchpad=dict(f.scratchpad))`` is
    functionally equivalent to the source (modulo identity).
    """

    prefix: PrefixContext
    messages: tuple[Message, ...]
    scratchpad: tuple[tuple[str, object], ...]
    journal_entries: tuple[Any, ...] = ()

    def __hash__(self) -> int:
        """Hash the transcript axis + the scratchpad KEYS. Never the scratchpad
        VALUES, never the journal entries.

        The docstring above has always promised a memoization-cache key, and
        the type could not deliver it. The tuples are frozen CONTAINERS, but
        two of them hold objects the framework does not own, so the generated
        all-fields hash inherited whatever the caller put in. Measured before
        this fix::

            hash(WorkingContext().note("k", "v").freeze())        1304396713497489601
            hash(WorkingContext().note("plan", {}).freeze())      TypeError: unhashable type: 'dict'

        ``update_scratchpad({"plan": {...}})`` is the DOCUMENTED API, so the
        promise broke on ordinary use, and it broke by value rather than by
        type — the same code path works until a caller stores a dict.

        Values are excluded because there is no honest alternative: a
        scratchpad value is any Python object, and a snapshot that is hashable
        only when the agent happened to store scalars is not a cache key, it is
        a trap. Keys are kept because they are ``str`` by construction and cost
        nothing (O(#keys) over interned strings), and they are the part that
        actually varies between the snapshots a memoizer sees — a context that
        has recorded a ``plan`` note differs from one that has not, even when
        the plan itself is opaque.

        ``journal_entries`` is excluded outright rather than reduced to a key
        like the scratchpad: entries are project-defined mutation types
        parameterising ``JournalEntryT``, very often plain dicts, so there is no
        ``str``-typed part to lift out — only the COUNT would be safe, and a
        count is a poor discriminator. The journal already has its own
        watermark-based identity (``MutationJournal.diff``), which a cache key
        has no business duplicating.

        Excluding all of that is sound rather than a workaround: ``__eq__``
        still compares every field, and the hash invariant only requires EQUAL
        objects to hash equally. Two snapshots differing only in a scratchpad
        VALUE or a journal entry collide into one bucket and ``__eq__``
        separates them there — a correct cache, one extra comparison.

        Cost is O(#messages + #keys) and linear in practice — tuple hashes are
        not cached, though the ``str`` hashes underneath them are. Measured on
        tool-calling transcripts: 5.2 µs at 10 messages, 41 µs at 100, 396 µs
        at 1000. Fine for a cache key computed once per snapshot; if you find
        yourself hashing the same ``FrozenContext`` in a tight loop, hold the
        key rather than recomputing it. That linearity is inherent to a
        STRUCTURAL hash — the alternative is an identity hash, which would
        break the equal-objects-hash-equally invariant this type needs to be a
        cache key at all.

        No ``__reduce__`` here, unlike ``ToolCall`` / ``Prompt`` /
        ``SignalEnvelope``: ``FrozenContext`` holds no ``MappingProxyType`` of
        its own, so once ``ToolCall`` pickles, so does a snapshot containing
        one. A scratchpad holding a genuinely unpicklable value (an open
        socket, a lambda) still fails to pickle — correctly, and that failure
        is the caller's object, not this type's shape. Hashing such a snapshot
        works, which is the point of splitting the two.
        """
        return hash((self.prefix, self.messages, tuple(k for k, _ in self.scratchpad)))


# ── WorkingContext — the unified primary ────────────────────────────


@dataclass
class WorkingContext:
    """An agent's working memory — four orthogonal axes.

    Required:
        ``prefix`` — the cache-stable head (system + grounding +
            optional schema block). Empty ``PrefixContext()`` is the
            right default when none is set yet.

    Axes (each optional, each independent):
        ``messages``    — per-turn LLM transcript tail. Mutating
                          methods touch only this list.
        ``scratchpad``  — cross-step notes. Last-write-wins.
        ``journal``     — append-only authored history with
                          watermark. Used by hierarchical agents.

    Long-term recall is no longer a context slot — see
    ``Agent.memory: MemorySource | None`` (auto-wired into the
    ``RequestBuilder`` grounder).

    Knobs:
        ``token_counter`` — pluggable ``TokenCounter`` for ``tokens()``.
                            Defaults to ``ApproxTokenCounter`` (chars/4).
        ``limit``  — hard token ceiling. ``None`` = no enforcement
                     (caller's policy still owns the actual abort).
        ``shared`` — team-blackboard hint. ``False`` (default) →
                     single-coroutine use; the lock is never touched.
                     ``True`` → multi-coroutine blackboard. The simple
                     sync mutators are still GIL-atomic and don't
                     auto-lock; for MULTI-step atomicity (sequences
                     with awaits between writes, read-modify-write)
                     wrap in ``async with ctx.lock:`` or
                     ``await ctx.apply_locked(fn)``.

    A single-shot chat agent uses transcript + scratchpad only; the
    journal stays empty. A long-lived hierarchical agent uses all
    three axes.
    """

    prefix: PrefixContext = field(default_factory=PrefixContext)
    messages: list[Message] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    journal: MutationJournal[Any] = field(default_factory=MutationJournal)
    token_counter: TokenCounter = field(default_factory=ApproxTokenCounter)
    limit: int | None = None
    shared: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    # ── transcript axis — append + clear ──────────────────────────────

    def append(self, *messages: Message) -> Self:
        """Add messages to the transcript tail."""
        self.messages.extend(messages)
        return self

    def extend(self, messages: Iterable[Message]) -> Self:
        """Add multiple messages to the transcript tail."""
        self.messages.extend(messages)
        return self

    def clear_messages(self) -> Self:
        """Clear the tail. Prefix, scratchpad, journal untouched."""
        self.messages.clear()
        return self

    # ── scratchpad axis ──────────────────────────────────────────────

    def note(self, key: str, value: Any) -> Self:
        """Set a scratchpad value (last-write-wins)."""
        self.scratchpad[key] = value
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Read a scratchpad value."""
        return self.scratchpad.get(key, default)

    def update_scratchpad(self, data: dict[str, Any]) -> Self:
        """Bulk-update the scratchpad."""
        self.scratchpad.update(data)
        return self

    # ── concurrency primitives ──────────────────────────────────────

    @property
    def lock(self) -> asyncio.Lock:
        """The internal asyncio lock — exposed as a public primitive
        when ``shared=True`` so callers can wrap multi-step mutation
        sequences in ``async with ctx.lock:`` to keep cross-coroutine
        atomicity.

        Acquire when EITHER of these holds:
        - A sequence of two or more mutations with an ``await`` between
          them must observe a consistent state to other coroutines.
        - A read-modify-write pattern (read scratchpad value, await
          some computation, write back) must not be lost to a
          concurrent writer.

        Sync single-step mutators (``append`` / ``note`` / etc.) do NOT
        need to be wrapped — they're GIL-atomic. The lock is for
        sequencing across ``await``s. ``shared=True`` does NOT auto-lock
        mutators; this property is the primitive callers use explicitly."""
        return self._lock

    async def apply_locked(self, fn: Callable[[Self], Any]) -> Self:
        """Run a synchronous mutation closure under ``self.lock`` for
        cross-coroutine atomicity.

        Example::

            async def add_pair(ctx):
                ctx.append(user_msg)
                await some_async_validation(ctx)
                ctx.append(assistant_msg)

            await ctx.apply_locked(add_pair)

        Both appends and the validation in between are serialized
        against any other coroutine that also goes through
        ``apply_locked`` (or ``async with ctx.lock``). Callers that
        DON'T need cross-coroutine safety should just call the
        mutators directly — this helper is opt-in.

        Supports async closures too: if ``fn`` returns an awaitable,
        it is awaited inside the lock."""
        async with self._lock:
            result = fn(self)
            if inspect.isawaitable(result):
                await result
        return self

    # ── view + composition ──────────────────────────────────────────

    def fork(self) -> WorkingContext:
        """An independent copy — own tail, deep-copied scratchpad,
        new journal seeded from current entries.

        The prefix is shared by reference (it's frozen — safe to
        share). The token counter / limit / shared flag are
        inherited; the lock is fresh.
        """
        new_journal: MutationJournal[Any] = MutationJournal(list(self.journal.view()))
        return WorkingContext(
            prefix=self.prefix,
            messages=list(self.messages),
            scratchpad=copy.deepcopy(self.scratchpad),
            journal=new_journal,
            token_counter=self.token_counter,
            limit=self.limit,
            shared=self.shared,
        )

    def merge(
        self,
        other: WorkingContext,
        *,
        mode: Literal["concat", "union"] = "concat",
    ) -> Self:
        """Aggregate another context's knowledge into this one.

        ``mode="concat"`` (default) appends the other's messages
        verbatim and bulk-updates the scratchpad (last-write-wins).
        Also applies the other's full journal entries — preserving
        each entry's attribution (the journal never rewrites).

        ``mode="union"`` deduplicates messages by value before
        appending, so a parent merging two siblings doesn't end up
        with the same observation twice. "By value" means ``__eq__``:
        the ``set`` below only uses ``Message.__hash__`` to pick a
        bucket, so two tool-call messages that share a call id but
        differ in arguments collide and both survive, as they must.
        That hash exists only because ``ToolCall`` defines one — this
        line raised ``TypeError: unhashable type: 'dict'`` for every
        message carrying tool calls, i.e. on every real fan-in.
        """
        if mode == "union":
            seen = set(self.messages)
            for m in other.messages:
                if m not in seen:
                    self.messages.append(m)
                    seen.add(m)
        else:
            self.messages.extend(other.messages)
        self.scratchpad.update(other.scratchpad)
        if other.journal.size() > 0:
            self.journal.apply(other.journal.view())
        return self

    def slice(self, scope: ContextScope) -> WorkingContext:
        """Return a NEW WorkingContext containing only messages that
        match ``scope``.

        Prefix, scratchpad, and journal are all inherited unchanged
        — slicing is a view operation on the tail, not a re-shape of
        the other axes. Useful for parent→child briefings.
        """
        compute_indices = getattr(scope, "_surviving_indices", None)
        if compute_indices is not None:
            keep = compute_indices(self.messages)
            kept = [m for i, m in enumerate(self.messages) if i in keep]
        else:
            kept = [m for i, m in enumerate(self.messages) if scope.matches(m, i)]
        return WorkingContext(
            prefix=self.prefix,
            messages=kept,
            scratchpad=dict(self.scratchpad),
            journal=MutationJournal(list(self.journal.view())),
            token_counter=self.token_counter,
            limit=self.limit,
            shared=False,  # a slice is a private view by default
        )

    def diff(self, other: WorkingContext) -> ContextDiff:
        """Replay/debug helper: what changed between two snapshots.

        Computes ``self - other`` semantics: messages in ``self`` but
        not ``other`` are ``messages_added``; the reverse are
        ``messages_removed``. Scratchpad entries follow the same logic
        with ``None`` signalling removal. ``prefix_changed`` is a
        structural equality check.

        Journal is intentionally NOT included in the diff — the
        journal has its own watermark-based diff via
        ``journal.diff()`` which is the right semantic for streaming.
        """
        self_msgs = list(self.messages)
        other_msgs = list(other.messages)

        # List-based difference preserving order + duplicate counts.
        other_remaining = list(other_msgs)
        added: list[Message] = []
        for m in self_msgs:
            try:
                other_remaining.remove(m)
            except ValueError:
                added.append(m)
        self_remaining = list(self_msgs)
        removed: list[Message] = []
        for m in other_msgs:
            try:
                self_remaining.remove(m)
            except ValueError:
                removed.append(m)

        changes: dict[str, Any] = {}
        for k, v in self.scratchpad.items():
            if other.scratchpad.get(k) != v or k not in other.scratchpad:
                changes[k] = v
        for k in other.scratchpad:
            if k not in self.scratchpad:
                changes[k] = None

        return ContextDiff(
            messages_added=tuple(added),
            messages_removed=tuple(removed),
            scratchpad_changes=changes,
            prefix_changed=self.prefix != other.prefix,
        )

    def freeze(self) -> FrozenContext:
        """Immutable snapshot — safe to share across agent boundaries
        without locking. Equality is structural."""
        return FrozenContext(
            prefix=self.prefix,
            messages=tuple(self.messages),
            scratchpad=tuple(sorted(self.scratchpad.items())),
            journal_entries=tuple(self.journal.view()),
        )

    # ── token math ───────────────────────────────────────────────────

    async def tokens(self) -> int:
        """Current token usage estimate via the configured counter.

        Includes prefix + messages. Async because the counter is async
        (it may call a remote counting endpoint or warm a lazy
        encoder); in the common ``ApproxTokenCounter`` path this is a
        single cheap await.
        """
        return await self.token_counter.estimate(self.assembled())

    def size(self) -> int:
        """Number of messages in the tail. Cheap — no token math."""
        return len(self.messages)

    def assembled(self) -> list[Message]:
        """The full message list the provider sees: ``prefix.as_messages()
        + self.messages``.

        Returns a new list each call so a caller mutating it doesn't
        corrupt the live tail.
        """
        return self.prefix.as_messages() + list(self.messages)


__all__ = [
    "ContextDiff",
    "FrozenContext",
    "JournalEntryT",
    "MutationJournal",
    "WorkingContext",
]
