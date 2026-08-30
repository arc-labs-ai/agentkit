"""Meter — ONE protocol for governing a unit of work, with two instances at two scopes.

A meter is one concept — *guard a ceiling, then charge usage* — applied at two scopes: `Budget`
per run and `Quota` per tenant. Both implement the `Meter` protocol and are driven by the single
`meter` middleware, which guards every `ctx.meters` before a call and charges them after. `Budget`
is also the run's depth/concurrency authority (it owns the shared semaphore and `max_depth`).

Money is :class:`~decimal.Decimal` here, not ``float``. Binary floating point cannot represent
``0.01``, so a hundred one-cent charges summed as floats land at ``1.0000000000000007`` and a
metered run cannot be reconciled to the cent. ``Budget`` keeps an exact ``Decimal`` ledger and
exposes ``spent_usd`` as a float MIRROR of it for display and for every existing reader.

``Budget`` also accumulates the whole :class:`~agentkit.kernel.types.Usage`, not just a cost
scalar. ``Usage`` arrives carrying input / output / cache-read / cache-write token counts and
already defines ``__add__``; reducing it to one number forced every application to re-aggregate
what the framework had already seen.

And ``charge()`` returns a :class:`Charge` verdict rather than only ever raising. Raising from
inside ``charge()`` aborts the run mid-call — before the cognition reaches its checkpoint write —
so exhausting a budget destroyed everything spent up to that point. With ``on_exceeded="stop"``
the caller gets a verdict it can act on, and the tool-loop cognition writes a ``suspended``
checkpoint before it stops. See :class:`Budget` for why ``"raise"`` is still the default.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal, Protocol, runtime_checkable

from agentkit.kernel.types import Usage

# Money is tracked to six decimal places — a tenth of a millicent. That is the
# scale ``pricing.cost()`` already rounds to, and it is fine enough that a
# per-call cost never quantizes to zero (the cheapest realistic call, a handful
# of tokens on a $0.08/1M cache-read rate, is ~1e-7 USD... which WOULD round to
# zero, hence "accumulate, then quantize" below rather than quantizing each
# charge).
MONEY_SCALE = 6
_QUANTUM = Decimal(1).scaleb(-MONEY_SCALE)  # Decimal("0.000001")
_CENTS = Decimal("0.01")

# Distinguishes "never normalised" from "normalised from ``None``" (which is a
# legitimate ceiling value meaning unlimited). A plain ``None`` default would
# make the two indistinguishable and skip the first re-derivation.
_UNSET_CEILING: Any = object()


class MeterExceeded(RuntimeError):
    """A ceiling (run cost/calls or tenant RPM/TPM/$) was crossed → the caller degrades gracefully."""


class MoneyPrecisionError(ValueError):
    """A monetary ceiling was given more precision than the ledger keeps.

    Raised at CONSTRUCTION only. Silently rounding a ceiling changes what the
    operator asked for; refusing tells them at wiring time, when it is free to
    fix. Charges are handled the other way round (quantized, not refused) —
    see :meth:`Budget.charge`, where raising mid-run would re-create the exact
    unrecoverable abort this module exists to remove.
    """


def to_money(value: Decimal | float | int | str, *, strict: bool = False) -> Decimal:
    """Coerce a monetary amount to an exact ``Decimal`` at :data:`MONEY_SCALE`.

    Floats go through ``str()`` first: ``Decimal(0.01)`` is
    ``0.01000000000000000020816681711721685...`` (the binary value), whereas
    ``Decimal(str(0.01))`` is exactly ``0.01`` — the number the caller meant.

    ``strict=True`` refuses input carrying more precision than the ledger
    keeps instead of quantizing it.
    """
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise MoneyPrecisionError(f"not a monetary amount: {value!r}") from exc
    if not amount.is_finite():
        raise MoneyPrecisionError(f"monetary amount must be finite, got {value!r}")
    try:
        quantized = amount.quantize(_QUANTUM, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        # ``quantize`` raises when the result would need more digits than the
        # decimal context allows — i.e. the amount is too large to hold at
        # ``MONEY_SCALE``. The guard above only wraps the PARSE, so this escaped
        # as a bare ``decimal.InvalidOperation``: every other malformed ceiling
        # ('abc', nan, inf, excess precision) surfaced as ``MoneyPrecisionError``,
        # and the one that didn't is the one an operator hits by leaving an
        # exponent in a config value.
        raise MoneyPrecisionError(
            f"monetary amount {value!r} is too large to represent at "
            f"{MONEY_SCALE} decimal places"
        ) from exc
    if strict and quantized != amount:
        raise MoneyPrecisionError(
            f"{value!r} carries more than {MONEY_SCALE} decimal places of precision. "
            f"Round it yourself, or accept {quantized} explicitly — the ledger will not "
            f"silently discard the difference for you."
        )
    return quantized


def _fmt(amount: Decimal) -> str:
    """Render money for a human-readable message: no trailing zero padding
    from the fixed scale, and never scientific notation.

    ``Decimal("0.020000")`` prints as ``0.02``; ``Decimal("100.000000")`` as
    ``100``. Plain ``str()`` would show the padding and ``normalize()`` alone
    would turn 100 into ``1E+2``, so both are wrong for an operator-facing
    ceiling message.
    """
    return format(amount.normalize(), "f")


@dataclass(frozen=True)
class Charge:
    """The verdict a meter returns from ``guard``/``charge``.

    A value, not an exception, so the caller decides what happens next. The
    tool-loop cognition uses ``ok=False`` to write a checkpoint and stop
    cleanly; a batch driver might use it to switch to a cheaper model; a
    fire-and-forget caller ignores it and lets ``on_exceeded="raise"`` do the
    old thing.

    ``usage`` is the meter's cumulative token totals, so a caller reading the
    verdict never has to re-aggregate what the framework already summed.
    """

    ok: bool
    reason: str = ""
    spent: Decimal = Decimal(0)  # EXACT cumulative spend. Named ``spent``, not
    # ``spent_usd``, on purpose: ``Budget.spent_usd`` is
    # the lossy float mirror, and one name meaning two
    # types across two classes is how a float creeps
    # back into a ledger. ``spent``/``remaining`` are
    # Decimal everywhere; ``*_usd`` is float everywhere.
    remaining: Decimal | None = None  # headroom; None when no ceiling is configured
    calls: int = 0
    usage: Usage = field(default_factory=Usage)  # the meter's CUMULATIVE token totals,
    # so a caller acting on a verdict never has to go
    # back to the meter for the numbers that justified
    # it. Empty for a meter that tracks no usage.

    def raise_if_exceeded(self) -> Charge:
        """Convert the verdict back into the exception, for a caller who wants
        the old control flow at one specific site. Returns self when ok, so it
        chains: ``(await budget.charge(...)).raise_if_exceeded()``."""
        if not self.ok:
            raise MeterExceeded(self.reason)
        return self


@runtime_checkable
class Meter(Protocol):
    """Guard a ceiling, then charge usage.

    Both methods return a :class:`Charge`. Implementations written against the
    older ``-> None`` signature still satisfy this at runtime and still work
    through the ``meter`` middleware, which treats ``None`` as "no verdict
    offered" — see ``middlewares/meter.py``. New implementations should return
    a verdict so a caller can degrade instead of only ever catching.
    """

    async def guard(self, call: Any) -> Charge | None: ...  # before the work runs
    async def charge(self, call: Any, usage: Usage) -> Charge | None: ...  # after it returns


def _est_tokens(call: Any) -> int:
    """≈4 chars/token estimate from a chat request's messages (0 for tool calls)."""
    msgs = getattr(getattr(call, "request", None), "messages", None) or []
    return sum(len(getattr(m, "content", "") or "") for m in msgs) // 4


@dataclass
class Budget:
    """The per-run meter + the run's depth/concurrency authority (one instance per agent-tree run).

    **Money.** ``max_cost_usd`` accepts a ``Decimal``, ``float``, ``int``, or
    ``str`` and is normalised to an exact ``Decimal``. A ceiling with more than
    :data:`MONEY_SCALE` decimal places raises :class:`MoneyPrecisionError` at
    construction rather than being silently rounded.

    **Tokens.** ``usage`` accumulates the full :class:`Usage` — input, output,
    cache-read and cache-write tokens survive to the end of a multi-agent run,
    because ``Budget`` is shared by reference across ``ctx.child()``.

    **Exhaustion.** ``on_exceeded`` picks the behaviour when a ceiling is
    crossed:

    * ``"raise"`` (default) — ``MeterExceeded`` from inside ``charge``, exactly
      as before.
    * ``"stop"`` — ``charge`` returns ``Charge(ok=False, ...)`` and raises
      nothing. The caller acts on it; the ReAct cognition writes a
      ``suspended`` checkpoint and ends the run with
      ``stop_reason="budget_exhausted"``, so the spend is recoverable.

    ``"raise"`` remains the default deliberately. Flipping it would silently
    change the control flow of every existing wiring — a run that used to
    abort would now continue past its ceiling in any caller that ignores the
    return value, which is a worse failure than the one being fixed. Callers
    opt into recoverability; a future major version can reconsider.

    **Known property, not a bug.** ``_check`` compares ``spent > ceiling``
    AFTER the work has run, so a budget is always overrun by at most one
    call's cost — there is no pre-flight estimate. Setting a ceiling slightly
    below the true limit is the mitigation; ``on_exceeded="stop"`` makes the
    overshoot recoverable rather than fatal.
    """

    max_cost_usd: Decimal | float | int | str | None = None
    max_calls: int | None = None
    max_depth: int = 4
    max_concurrency: int = 8
    spent_usd: float = 0.0  # FLOAT MIRROR of the exact ledger — see below.
    calls: int = 0
    on_exceeded: Literal["raise", "stop"] = "raise"
    usage: Usage = field(default_factory=Usage)  # cumulative token counts
    _spent: Decimal = field(default=Decimal(0), compare=False, repr=False)
    _ceiling: Decimal | None = field(default=None, compare=False, repr=False)
    _ceiling_src: Any = field(default=_UNSET_CEILING, compare=False, repr=False)
    _lock: Any = field(default=None, compare=False, repr=False)
    _sems: dict[int, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        # The non-money ceilings, held to the same standard as the money one.
        # This class already refuses a malformed ``max_cost_usd`` at
        # construction — where it is free to fix — and used to accept anything
        # at all on the other three axes. That asymmetry had a sharp edge:
        # ``max_concurrency=0`` builds an ``asyncio.Semaphore(0)``, so the first
        # fan-out waits forever on a permit that is never issued. No exception,
        # no log line, just a run that never returns. A negative value reached
        # the same constructor and surfaced as a bare ``ValueError`` from inside
        # ``ctx.semaphore()``, pointing at the framework rather than the config.
        #
        # Zero is meaningful on two of these axes and not on the third, so they
        # are checked separately rather than with one shared predicate:
        # ``max_depth=0`` is "run the root, never fan out" and ``max_calls=0``
        # is "make no calls", but there is no run at all with zero permits.
        if self.max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be at least 1, got {self.max_concurrency}. "
                "A value of 0 does not mean 'unbounded' — it is a semaphore with no "
                "permits, and the first fan-out would wait on it forever."
            )
        if self.max_depth < 0:
            raise ValueError(f"max_depth must not be negative, got {self.max_depth}")
        if self.max_calls is not None and self.max_calls < 0:
            raise ValueError(f"max_calls must not be negative, got {self.max_calls}")
        # Normalise the ceiling STRICTLY here. A ceiling is the operator's
        # stated intent, so quietly rounding it is wrong; a charge is a
        # measurement, so quietly quantizing it is right. Different inputs,
        # different rules.
        self._renormalise_ceiling(strict=True)
        # ``spent_usd`` stays an init-accepting FIELD rather than becoming a
        # read-only property, because rebuilding a Budget from a checkpoint —
        # ``Budget(spent_usd=saved.state["spent_usd"])`` — is a DOCUMENTED
        # path (docs/mental-models/03). Seeding the exact ledger from it keeps
        # that wiring working; from here on ``_spent`` is authoritative and
        # ``spent_usd`` is re-derived from it after every charge, so the two
        # can never drift.
        self._spent = to_money(self.spent_usd)
        self._sync_mirror()

    # -- the exact ledger, and its float mirror -------------------------------

    def _sync_mirror(self) -> None:
        """Re-derive the lossy ``spent_usd`` float from the exact ledger.

        Keeping a mirror rather than deprecating the name means ~20 doc
        references, 28 test references, and every application reading
        ``budget.spent_usd`` keep working untouched. The float is for display
        and arithmetic-that-doesn't-matter; :meth:`spent` is for money.
        """
        self.spent_usd = float(self._spent)

    def spent(self) -> Decimal:
        """The exact amount spent. THIS is the number to reconcile against —
        ``spent_usd`` is a float rendering of it and cannot be summed
        losslessly."""
        return self._spent

    def spent_cents(self) -> Decimal:
        """The exact spend rounded to whole cents, for invoicing.

        Quantization happens HERE, at read time, never per-charge: rounding
        each charge to cents would round every sub-cent call to zero and
        undercount the entire run."""
        return self._spent.quantize(_CENTS, rounding=ROUND_HALF_UP)

    def remaining_usd(self) -> float | None:
        """Headroom as a float — the shape the tracing middleware and existing
        callers expect. ``None`` when no ceiling is set."""
        exact = self.remaining()
        return None if exact is None else float(exact)

    def remaining(self) -> Decimal | None:
        """Headroom, exactly. ``None`` when no ceiling is set."""
        ceiling = self.ceiling()
        if ceiling is None:
            return None
        return max(Decimal(0), ceiling - self._spent)

    def _renormalise_ceiling(self, *, strict: bool = False) -> None:
        """Re-derive the exact ceiling from ``max_cost_usd`` and remember the
        source value we derived it from."""
        raw = self.max_cost_usd
        self._ceiling_src = raw
        self._ceiling = None if raw is None else to_money(raw, strict=strict)

    def ceiling(self) -> Decimal | None:
        """The normalised ceiling, exactly.

        Re-derives when ``max_cost_usd`` has been ASSIGNED since the last
        normalisation. ``Budget`` is a mutable dataclass and
        ``budget.max_cost_usd = 10.0`` is the obvious way to raise a ceiling
        and resume an exhausted run — it is what the spend recipe tells an
        operator to do. Caching the normalised value at construction and never
        looking again would make that assignment silently do nothing, which is
        a worse failure than the float ledger this class was rewritten to
        remove.

        The re-derivation is NON-strict, unlike the one in ``__post_init__``:
        this runs on a read path reached from inside ``charge()``, and raising
        ``MoneyPrecisionError`` there would abort a run mid-flight — exactly
        the unrecoverable abort being designed out. Precision is refused at
        construction, where it is free to fix; a post-construction assignment
        is quantized.
        """
        if self.max_cost_usd != self._ceiling_src:
            self._renormalise_ceiling()
        return self._ceiling

    # -- verdicts -------------------------------------------------------------

    def _verdict(self) -> Charge:
        """Evaluate both ceilings against the current books. Pure — no raising,
        no mutation — so ``guard`` and ``charge`` share one definition of
        "exceeded" and cannot drift apart."""
        reason = ""
        ceiling = self.ceiling()  # re-derives if max_cost_usd was reassigned
        if ceiling is not None and self._spent > ceiling:
            reason = f"cost ${_fmt(self._spent)} > ${_fmt(ceiling)}"
        elif self.max_calls is not None and self.calls > self.max_calls:
            reason = f"calls {self.calls} > {self.max_calls}"
        return Charge(
            ok=not reason,
            reason=reason,
            spent=self._spent,
            remaining=self.remaining(),
            calls=self.calls,
            usage=self.usage,
        )

    def exhausted(self) -> bool:
        """Has a ceiling been crossed? A cheap synchronous read for a caller
        that wants to check between units of work — the ReAct loop uses it
        after each chat call to decide whether to checkpoint and stop."""
        return not self._verdict().ok

    def verdict(self) -> Charge:
        """The current verdict without charging anything."""
        return self._verdict()

    def _get_lock(self) -> asyncio.Lock:
        # Lazy so the lock binds to the running loop at first use, never to
        # some other loop that happened to exist at Budget-construction time.
        if self._lock is None:
            self._lock = asyncio.Lock()
        lock: asyncio.Lock = self._lock
        return lock

    async def guard(self, call: Any = None) -> Charge:
        """Check the ceilings before the work runs."""
        async with self._get_lock():
            verdict = self._verdict()
        if not verdict.ok and self.on_exceeded == "raise":
            raise MeterExceeded(verdict.reason)
        return verdict

    async def charge(self, call: Any, usage: Usage) -> Charge:
        """Accrue ``usage`` and return the resulting verdict.

        The books are updated BEFORE the ceiling is evaluated, so an
        over-ceiling call is still recorded — the spend really happened and
        dropping it would make the ledger lie.

        A charge is QUANTIZED to :data:`MONEY_SCALE`, not refused, even though
        an over-precise ceiling is refused at construction. Refusing here would
        mean a custom ``pricing=`` callable returning full float precision
        aborts a run mid-flight — re-creating the unrecoverable abort this
        class was rewritten to remove. The ceiling is intent (refuse); a charge
        is a measurement (record).

        ``Usage`` is frozen, so ``self.usage`` is REPLACED via its ``__add__``,
        never mutated in place.
        """
        async with self._get_lock():
            self.usage = self.usage + usage
            self._spent += to_money(usage.cost_usd)
            self._sync_mirror()
            self.calls += 1
            verdict = self._verdict()
        # Raise OUTSIDE the lock: an exception unwinding through ``async with``
        # releases it correctly either way, but keeping the raise out of the
        # critical section makes it obvious that no waiter can be stranded.
        if not verdict.ok and self.on_exceeded == "raise":
            raise MeterExceeded(verdict.reason)
        return verdict

    def semaphore(self, depth: int = 0) -> asyncio.Semaphore:
        """The concurrency permit pool for one DEPTH of the agent tree.

        One semaphore per depth, not one for the whole tree. A single shared
        semaphore deadlocks, and not hypothetically: a parent's fan-out holds
        its permits for the ENTIRE duration of each child run, so a nested
        fan-out draws from a pool its own ancestors have already drained. With
        ``max_concurrency=2``, an agent dispatching two ``as_tool`` sub-agents
        that each dispatch their own tools hangs forever — the two outer
        permits are held until the inner runs finish, and the inner runs cannot
        start without a permit. Deeper trees hit the same wall at any cap once
        the ancestors' outstanding acquisitions reach it.

        Keying on depth breaks the cycle structurally: an ancestor at depth d
        can only ever hold permits from pool d, and its children draw from
        pool d+1, so no acquisition can ever wait on a permit held by its own
        ancestor. Every nesting boundary in the framework goes through
        ``ctx.child()`` (``as_tool``, ``run_agents``, the coordinator
        policies), so depth genuinely increments at each level.

        The trade is an honest one: the bound is now ``max_concurrency`` PER
        LEVEL rather than across the whole tree, so worst-case in-flight work
        is ``max_concurrency * (max_depth + 1)``. Set ``max_concurrency`` with
        that in mind. A single tree-wide cap cannot be both deadlock-free and
        respected by nested acquisition — a re-entrant permit would let one
        level's fan-out multiply without limit, which is a weaker bound than
        per-level, not a stronger one.

        Lazy per depth, so the semaphore binds to the running loop at first
        use rather than to whatever loop existed at Budget construction.
        """
        sem = self._sems.get(depth)
        if sem is None:
            sem = asyncio.Semaphore(self.max_concurrency)
            self._sems[depth] = sem
        assert isinstance(sem, asyncio.Semaphore)
        return sem


@dataclass
class Quota:
    """The per-tenant meter — rolling RPM/TPM/$ windows keyed by `scope.key()`, independent of Budget.
    The noisy-neighbor guard and chargeback source. In-memory ref; prod swaps a Redis-backed Meter.

    Unlike :class:`Budget`, ``Quota`` enforces on ``guard`` (before the work)
    rather than on ``charge`` (after it), so its ``charge`` has nothing to
    refuse — it returns an ``ok`` verdict for protocol symmetry. ``guard``
    honours the same ``on_exceeded`` switch so a caller can drive both meters
    with one control flow.
    """

    max_rpm: int | None = None
    max_tpm: int | None = None
    max_usd: Decimal | float | int | str | None = None
    window: float = 60.0
    clock: Callable[[], float] = time.monotonic
    on_exceeded: Literal["raise", "stop"] = "raise"
    _ceiling: Decimal | None = field(default=None, compare=False, repr=False)
    # Monotonic-ish marker for the last full eviction sweep. Seeded to
    # ``-inf`` so the first ``guard`` sweeps immediately rather than waiting
    # out a window on a freshly built Quota.
    _last_sweep: float = field(default=float("-inf"), compare=False, repr=False)
    _reqs: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _charges: dict[str, list[tuple[float, int, Decimal]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _lock: Any = field(default=None, compare=False, repr=False)

    _ceiling_src: Any = field(default=_UNSET_CEILING, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._renormalise_ceiling(strict=True)

    def _renormalise_ceiling(self, *, strict: bool = False) -> None:
        self._ceiling_src = self.max_usd
        self._ceiling = None if self.max_usd is None else to_money(self.max_usd, strict=strict)

    def ceiling(self) -> Decimal | None:
        """The normalised per-window ceiling. Re-derives on a post-construction
        assignment to ``max_usd`` — see ``Budget.ceiling`` for the rationale."""
        if self.max_usd != self._ceiling_src:
            self._renormalise_ceiling()
        return self._ceiling

    def _prune(self, key: str, now: float) -> None:
        """Drop out-of-window entries for ONE key, then drop the key itself if
        it has gone empty.

        Deleting the empty key matters because ``_reqs``/``_charges`` are
        ``defaultdict``s keyed by ``Scope.key()``. A service whose scope
        carries a per-user or per-request id — which is the whole point of a
        tenant key — would otherwise accumulate one permanently-retained dict
        entry per distinct scope ever seen, holding an empty list forever.
        """
        cutoff = now - self.window
        reqs = [t for t in self._reqs[key] if t > cutoff]
        charges = [c for c in self._charges[key] if c[0] > cutoff]
        if reqs:
            self._reqs[key] = reqs
        else:
            self._reqs.pop(key, None)
        if charges:
            self._charges[key] = charges
        else:
            self._charges.pop(key, None)

    def _sweep(self, now: float) -> None:
        """Evict every key whose whole window has expired.

        ``_prune`` only ever touches the key being guarded, so a tenant that
        goes quiet is never revisited and its entry leaks. Measured: 5000
        distinct scopes left 5000 retained keys long after every window had
        expired. This sweeps the rest, at most once per ``window`` — O(keys)
        amortised over the window length, which is negligible beside the
        per-call work it sits next to.
        """
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        cutoff = now - self.window
        for key in [k for k, v in self._reqs.items() if not any(t > cutoff for t in v)]:
            self._reqs.pop(key, None)
        for key in [k for k, v in self._charges.items() if not any(c[0] > cutoff for c in v)]:
            self._charges.pop(key, None)

    def _get_lock(self) -> asyncio.Lock:
        # Lazy — see Budget._get_lock for the loop-binding rationale.
        if self._lock is None:
            self._lock = asyncio.Lock()
        lock: asyncio.Lock = self._lock
        return lock

    def spent_in_window(self, key: str) -> Decimal:
        """Exact spend for one tenant inside the current window — the
        chargeback number, summed in ``Decimal`` so it reconciles."""
        return sum((c for _, _, c in self._charges[key]), Decimal(0))

    async def guard(self, call: Any) -> Charge:
        now = self.clock()
        key = call.ctx.scope.key()
        est = _est_tokens(call)
        async with self._get_lock():  # prune+read+count is one atomic critical section (no awaits)
            self._sweep(now)  # evict long-dead tenants; at most once per window
            self._prune(key, now)
            reqs = len(self._reqs[key])
            toks = sum(t for _, t, _ in self._charges[key])
            cost = self.spent_in_window(key)
            reason = ""
            if self.max_rpm is not None and reqs >= self.max_rpm:
                reason = f"{key}: {reqs} req ≥ {self.max_rpm} rpm"
            elif self.max_tpm is not None and toks + est > self.max_tpm:
                reason = f"{key}: {toks}+{est} tok > {self.max_tpm} tpm"
            else:
                ceiling = self.ceiling()
                if ceiling is not None and cost >= ceiling:
                    reason = f"{key}: ${_fmt(cost)} ≥ ${_fmt(ceiling)} per window"
            if not reason:
                self._reqs[key].append(
                    now
                )  # count this attempt toward RPM (a made request, even if it later fails)
        verdict = Charge(ok=not reason, reason=reason, spent=cost)
        if not verdict.ok and self.on_exceeded == "raise":
            raise MeterExceeded(verdict.reason)
        return verdict

    async def charge(self, call: Any, usage: Usage) -> Charge:
        key = call.ctx.scope.key()
        async with self._get_lock():
            self._charges[key].append((self.clock(), usage.total_tokens, to_money(usage.cost_usd)))
            spent = self.spent_in_window(key)
        # ``usage`` is deliberately NOT set: Quota keeps per-window token
        # counts partitioned by tenant, not one cumulative ``Usage``, and
        # returning this call's usage under a field documented as cumulative
        # would be a quieter lie than leaving it empty.
        return Charge(ok=True, spent=spent)


__all__ = [
    "MONEY_SCALE",
    "Budget",
    "Charge",
    "Meter",
    "MeterExceeded",
    "MoneyPrecisionError",
    "Quota",
    "to_money",
]
