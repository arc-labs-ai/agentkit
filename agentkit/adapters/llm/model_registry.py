"""Model registry — resolve a model name to a wired ``LLMPort``, and refuse a
capability mismatch before any spend.

Two problems, one object. They are the same problem wearing different hats:

**The last mile (Brief 3).** ``agentkit.client`` ships ``claude()`` /
``openai()`` / ``deepseek()`` / ``openrouter()``, and every one of them
requires an explicit ``api_key=``. The package read no provider environment
variable anywhere. So every application wrote the same bootstrap — read the
key, pick the provider, handle the optional-extra ``ImportError``, fall back
without failing silently. In one production codebase that was ~710 lines and
the only domain-free code in its engine.

**The silent capability failure (Brief 5).** A model is a string.
``claude(model="claude-sonnet-4-6")`` says nothing about what it can do, so
nothing catches a role bound to a model that lacks a needed capability. That
failure is silent AND well-formed: an agent asked to read images, bound to a
model that cannot see, returns a structurally valid answer citing evidence it
never read.

Both are answered by knowing things about a model name. So this module keeps
ONE table: a :class:`ModelEntry` carries the provider that serves it and the
capabilities it declares, and a :class:`ProviderEntry` carries the environment
variable and the (lazily imported) factory. Resolution and capability checking
read the same rows.

Design commitments, each of which is load-bearing:

* **Nothing is guessed.** An unregistered model reports every capability as
  :attr:`Capability.UNKNOWN`, never ``False`` and never ``True``. A framework
  that guesses ``True`` reintroduces the exact silent failure it exists to
  catch; one that guesses ``False`` refuses every self-hosted model on day
  one. ``UNKNOWN`` is the honest third answer, and what to DO about it is the
  caller's policy (``on_unknown=``).
* **A missing extra degrades loudly.** If ``httpx`` isn't installed we raise
  :class:`MissingProviderExtra` naming the extra. We never silently fall
  through to the fake — a fake serving production traffic is the worst
  outcome in this module's problem space.
* **Fallback is explicit.** ``resolve()`` with no credentials raises unless
  the caller passed ``fallback=``. See :meth:`ModelRegistry.resolve` for why
  this deliberately differs from "warn and downgrade".
* **A credential never appears in a message.** Not in an error, not in a
  warning, not in ``repr``. Pinned by test.
* **Rules over a prefix table.** ``register_rule`` lets an application map
  its own naming convention onto a provider without forking a hardcoded
  prefix list.

This layer sits ABOVE the explicit factories and does not replace them.
``providers.claude(api_key=...)`` keeps working untouched; this is the
from-configuration path for callers who don't want to name a provider in code.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agentkit.kernel.errors import AgentkitError

# ── errors ───────────────────────────────────────────────────────────────────


class RegistryError(AgentkitError):
    """Base for every failure this module raises — one except clause covers
    the whole bootstrap."""


class UnknownModel(RegistryError):
    """No registered provider claims this model name.

    Raised instead of a ``KeyError`` so the message can name what IS
    registered; a bare ``KeyError('gpt-5')`` tells the caller nothing about
    how to fix their config."""


class ProviderNotConfigured(RegistryError):
    """A provider was selected but no credential was found in the environment.

    The message names the environment variables that were CHECKED — never
    their values, and never a partial/masked value either. A masked
    credential is still a credential leak's first half."""


class MissingProviderExtra(RegistryError):
    """The provider's optional dependency isn't installed.

    Deliberately fatal rather than a downgrade: "your transport is missing"
    is a wiring bug the operator must see, and quietly serving a fake instead
    is precisely the silent failure this module exists to prevent."""


class CapabilityMismatch(RegistryError):
    """A model was bound to a role needing a capability it does not have.

    Raised at construction / first bind, never on the call — the entire value
    is catching it before money is spent."""


# ── capabilities ─────────────────────────────────────────────────────────────


class Capability(StrEnum):
    """Tri-state, because two-state is what causes the bug.

    A ``bool`` forces every unrecognised model into either "has it" (silent
    wrong answers) or "doesn't" (refuses everything custom). A ``bool | None``
    is worse than either, because ``if caps.vision:`` silently reads ``None``
    as absent and the type system won't stop you. A three-valued enum makes
    the third case impossible to ignore at the read site.
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


# The declarable capability names, in the order a mismatch report lists them.
# ``context_window`` is deliberately NOT here — it is an int, not a
# tri-state, and is requested via ``min_context_window=`` instead.
CAPABILITY_NAMES: tuple[str, ...] = (
    "tools",
    "structured_output",
    "native_json_schema",
    "streaming",
    "vision",
)


@dataclass(frozen=True)
class ModelCapabilities:
    """What one model can do. Every field defaults to ``UNKNOWN`` so a
    partially-filled entry never over-claims.

    ``vision`` is declared but not yet checkable end-to-end: ``Message.content``
    is a ``str`` and both provider adapters map it to text only, so there is
    no multimodal input path to mismatch against today. It is here so the
    declaration is ready when that path lands, and so an application that
    routes images outside the framework can still consult it.
    """

    tools: Capability = Capability.UNKNOWN
    """Provider-native tool/function calling (what ``ReActCognition`` needs)."""

    structured_output: Capability = Capability.UNKNOWN
    """Can be steered to emit parseable JSON at all — via native mode OR the
    prompt-injection fallback. Practically every instruct model: ``NO`` means
    a completion-only model that cannot follow the schema block."""

    native_json_schema: Capability = Capability.UNKNOWN
    """Accepts a provider-native ``response_format={"type": "json_schema"}``.
    Distinct from ``structured_output``: this is the strict server-side mode,
    not the prompt fallback. Read by ``_infer_response_format``."""

    streaming: Capability = Capability.UNKNOWN
    """Server-sent incremental deltas (as opposed to one-shot completion)."""

    vision: Capability = Capability.UNKNOWN
    """Accepts image input. See the class docstring on why this is
    forward-looking."""

    context_window: int | None = None
    """Total context in tokens, or ``None`` when unknown. Lets compaction size
    itself against a real number instead of a guess."""

    def get(self, name: str) -> Capability:
        """Look one capability up by name. An unrecognised NAME is a
        programming error (a typo in ``requires=``) and raises immediately,
        rather than degrading into ``UNKNOWN`` and silently never
        checking anything."""
        if name not in CAPABILITY_NAMES:
            raise ValueError(
                f"unknown capability {name!r}; declarable capabilities are: "
                f"{', '.join(CAPABILITY_NAMES)}"
            )
        value: Capability = getattr(self, name)
        return value

    @property
    def all_unknown(self) -> bool:
        """True when nothing at all is declared — i.e. this model is not in
        the registry and the object is a placeholder."""
        return all(getattr(self, n) is Capability.UNKNOWN for n in CAPABILITY_NAMES)


# Shorthands for the table below. The table is read far more often than it is
# written, and ``tools=YES`` reads better in a row than ``tools=Capability.YES``.
_Y = Capability.YES
_N = Capability.NO


# ── entries ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelEntry:
    """One model: who serves it, and what it can do."""

    name: str  # canonical family name, lowercase, undated
    provider: str  # a registered ProviderEntry.name
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


@dataclass(frozen=True)
class ProviderEntry:
    """One provider: how to authenticate it and how to build its client.

    ``factory`` is a dotted ``module:attr`` string rather than a callable so
    that registering a provider costs NO import. ``agentkit.adapters.llm.providers``
    raises ``ImportError`` at module load when the ``http`` extra is absent —
    importing it eagerly here would make the whole registry unimportable on a
    zero-dependency install, which is the opposite of what it is for.
    """

    name: str
    env_vars: tuple[str, ...]  # checked in order; first one set wins
    factory: str  # "package.module:callable"
    extra: str = "http"  # the pip extra to name when the import fails
    default_model: str | None = None
    accepts_model_kwarg: bool = True  # factory takes model=; False → api_key only


# ── model-name normalisation ─────────────────────────────────────────────────

# Providers echo a DATED release id on every response — Anthropic's
# ``claude-haiku-4-5-20251001`` (undashed date) and OpenAI's
# ``gpt-4o-mini-2024-07-18`` (dashed). Both shapes, one pattern.
_DATE_SUFFIX = re.compile(r"-\d{4}-?\d{2}-?\d{2}$")


def normalize_model_name(model: str) -> tuple[str, ...]:
    """Candidate lookup keys for a model name, most specific first.

    ``("anthropic/claude-sonnet-4-6-20250101")`` →
    ``("anthropic/claude-sonnet-4-6-20250101", "claude-sonnet-4-6-20250101",
       "anthropic/claude-sonnet-4-6", "claude-sonnet-4-6")``

    Deliberately mirrors the candidate ORDER in
    ``adapters/llm/providers/pricing.py::_lookup`` — exact, then strip an
    OpenRouter-style ``provider/`` prefix, then strip the trailing date, then
    both. It is a separate function rather than a shared import because the
    pricing table is owned elsewhere and stays untouched; the cost of the
    duplication is one drift risk, which
    ``tests/adapters/test_model_registry.py::test_normalisation_agrees_with_pricing_lookup``
    pins by asserting both resolve the same names.

    Returned as a de-duplicated tuple so callers can just iterate.
    """
    name = model.strip().lower()
    bare = name.split("/", 1)[-1]
    candidates = (name, bare, _DATE_SUFFIX.sub("", name), _DATE_SUFFIX.sub("", bare))
    seen: dict[str, None] = {}
    for c in candidates:
        if c:
            seen.setdefault(c, None)
    return tuple(seen)


# ── the registry ─────────────────────────────────────────────────────────────

# What a name→provider rule looks like: given a normalised model name, return
# a provider name or None. Applications register their own so an internal
# naming convention ("acme-llm-v3") routes without forking a prefix table.
Rule = Callable[[str], str | None]


class ModelRegistry:
    """Model/provider knowledge, and the two operations that read it:
    :meth:`resolve` (name → wired ``LLMPort``) and :meth:`check` (name +
    requirements → refuse or pass).

    Instantiable so tests get an isolated table; the process-wide default is
    :func:`registry`.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderEntry] = {}
        self._models: dict[str, ModelEntry] = {}
        self._rules: list[Rule] = []
        # One-shot warning latch. Per-instance, not module-global, so a test
        # can observe the warning on a fresh registry and so "exactly once"
        # means once per configured registry rather than once per process
        # (which would make the second differently-wired app in a process
        # silently miss its own downgrade notice).
        self._warned: set[str] = set()

    # -- registration ---------------------------------------------------------

    def register_provider(self, entry: ProviderEntry) -> ModelRegistry:
        """Add or replace a provider. Returns self so registration chains."""
        self._providers[entry.name] = entry
        return self

    def register_model(self, entry: ModelEntry) -> ModelRegistry:
        """Add or replace a model row.

        The provider is NOT validated here: registration order shouldn't
        matter, and an application may legitimately declare capabilities for a
        model whose provider it registers later (or never, if it only ever
        calls :meth:`check`).
        """
        self._models[entry.name.lower()] = entry
        return self

    def register_rule(self, rule: Rule) -> ModelRegistry:
        """Add a name→provider rule, consulted when no ``ModelEntry`` matches.

        The most-recently-added rule wins: the built-in rules act as the
        fallback, and an application's rule (registered after them) takes
        precedence. A rule returns ``None`` to abstain.

            reg.register_rule(lambda n: "openai" if n.startswith("acme-") else None)
        """
        self._rules.append(rule)
        return self

    # -- lookup ---------------------------------------------------------------

    def entry_for(self, model: str) -> ModelEntry | None:
        """The registered row for a model, trying each normalised candidate."""
        for candidate in normalize_model_name(model):
            hit = self._models.get(candidate)
            if hit is not None:
                return hit
        return None

    def provider_for(self, model: str) -> str | None:
        """Which provider serves this model, or ``None`` if nothing claims it.

        A registered ``ModelEntry`` wins over any rule — an explicit row is
        always more authoritative than a pattern.

        Rules are the OUTER loop, most-recently-registered first: each rule
        sees EVERY normalised candidate before the next rule is consulted.
        Iterating candidates outermost meant the first candidate decided the
        answer, so the coarsest rule that matched the raw name won and the
        later, more specific rules never saw the bare name at all — measured:
        ``provider_for('anthropic/claude-sonnet-9')`` returned ``openrouter``
        because ``by_openrouter_prefix`` matched the slash before
        ``by_family`` was offered ``claude-sonnet-9``, contradicting both
        ``register_rule``'s promise and ``_builtin_rules``' own comment.
        Reversing the rule order is the short-circuiting form of "run every
        rule, last non-``None`` wins", which is what :meth:`register_rule`
        documents.
        """
        entry = self.entry_for(model)
        if entry is not None:
            return entry.provider
        candidates = normalize_model_name(model)
        for rule in reversed(self._rules):
            for candidate in candidates:
                provider = rule(candidate)
                if provider is not None:
                    return provider
        return None

    def provider_entry(self, name: str) -> ProviderEntry | None:
        """The registered row for a provider, or ``None``. Lets a caller read a
        provider's ``default_model`` / ``env_vars`` without reaching into the
        registry's internals."""
        return self._providers.get(name)

    def capabilities(self, model: str) -> ModelCapabilities:
        """What this model can do. An unregistered model yields an
        all-``UNKNOWN`` placeholder — never a guess in either direction."""
        entry = self.entry_for(model)
        return entry.capabilities if entry is not None else ModelCapabilities()

    def describe(self) -> str:
        """A human-readable inventory, used in error messages so a refusal
        tells the operator what they COULD have asked for."""
        models = ", ".join(sorted(self._models)) or "(none)"
        providers = ", ".join(sorted(self._providers)) or "(none)"
        return f"registered providers: {providers}\nregistered models: {models}"

    # -- capability checking --------------------------------------------------

    def check(
        self,
        model: str,
        requires: tuple[str, ...] = (),
        *,
        derived: tuple[str, ...] = (),
        min_context_window: int | None = None,
        on_unknown: str = "warn",
        subject: str = "",
    ) -> None:
        """Refuse a capability mismatch, or return silently.

        ``requires`` names capabilities from :data:`CAPABILITY_NAMES` that the
        caller DECLARED. ``derived`` names ones the framework INFERRED from
        wiring (a tool-holding cognition implies ``tools``). Both raise on a
        declared :attr:`Capability.NO` — that is an unambiguous mismatch and
        no policy makes it acceptable — but only ``requires`` participates in
        the ``UNKNOWN`` policy below.

        That asymmetry is the point. A derived requirement is the framework's
        own inference, so nagging about it for every unregistered model name
        would put a warning on essentially every development wiring and teach
        people to filter the category out — at which point the real
        ``requires=`` warnings go unread too. Derived requirements therefore
        stay silent on ``UNKNOWN`` and speak only when a model has been
        explicitly declared incapable.

        :attr:`Capability.UNKNOWN` on a DECLARED requirement is governed by
        ``on_unknown``:

        * ``"warn"`` (default) — emit a ``UserWarning`` and continue. This is
          the default because refusing on unknown would break every
          self-hosted, fine-tuned, or newly-released model name the moment
          this module shipped, and the briefs require existing wiring to keep
          working.
        * ``"refuse"`` — raise. The right setting for a production service
          that pins its models; makes "we don't know" a deployment-time stop.
        * ``"allow"`` — say nothing. For a caller that has verified out of
          band.

        ``subject`` names the thing being bound (an agent name), so the
        message points at the wiring rather than at the framework.
        """
        if on_unknown not in ("warn", "refuse", "allow"):
            raise ValueError(f"on_unknown must be 'warn' | 'refuse' | 'allow', got {on_unknown!r}")

        caps = self.capabilities(model)
        who = f"{subject!r} " if subject else ""

        # A declared NO is fatal whether the requirement was declared or derived.
        missing = [n for n in (*requires, *derived) if caps.get(n) is Capability.NO]
        if missing:
            raise CapabilityMismatch(
                f"{who}requires {', '.join(sorted(missing))} but model {model!r} "
                f"declares it as unsupported. Bind a model that supports it, or drop "
                f"the requirement.\n{self.describe()}"
            )

        if min_context_window is not None and caps.context_window is not None:
            if caps.context_window < min_context_window:
                raise CapabilityMismatch(
                    f"{who}needs a context window of at least {min_context_window} tokens "
                    f"but model {model!r} declares {caps.context_window}."
                )

        unknown = [n for n in requires if caps.get(n) is Capability.UNKNOWN]
        if min_context_window is not None and caps.context_window is None:
            unknown.append("context_window")
        if not unknown or on_unknown == "allow":
            return

        detail = (
            f"{who}requires {', '.join(sorted(unknown))} but model {model!r} is not in the "
            f"model registry, so its capabilities are UNKNOWN — not assumed present. "
            f"Register it with ModelRegistry.register_model(...) to make this a real check."
        )
        if on_unknown == "refuse":
            raise CapabilityMismatch(detail + f"\n{self.describe()}")
        self._warn_once(f"unknown-caps:{model}:{','.join(sorted(unknown))}", detail)

    # -- resolution -----------------------------------------------------------

    def resolve(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        fallback: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Any:
        """Build an ``LLMPort`` for ``model`` (or an explicit ``provider``),
        reading credentials from the environment.

            llm = resolve("claude-sonnet-4-6")          # reads ANTHROPIC_API_KEY
            llm = resolve(provider="openai")            # reads OPENAI_API_KEY
            llm = resolve("gpt-4o", fallback="fake")    # degrade, loudly, if unkeyed

        ``fallback`` is OPT-IN and defaults to ``None`` (raise). This
        deliberately differs from "quietly return the fake and warn": in a
        server process a ``UserWarning`` goes to a log nobody reads, and the
        result is fabricated completions served as real ones. Making the
        caller type ``fallback="fake"`` keeps the escape hatch one keystroke
        away for a notebook while refusing to make it the accidental default
        for a deployment. When it IS requested, the downgrade warns exactly
        once per registry.

        ``env`` is injectable so tests never mutate ``os.environ``.

        A missing optional extra raises :class:`MissingProviderExtra` and is
        NEVER absorbed into the fallback — a broken install must not
        masquerade as a missing credential.
        """
        env = os.environ if env is None else env

        name = provider or (self.provider_for(model) if model else None)
        if name is None:
            if model is None:
                raise UnknownModel(f"resolve() needs a model or a provider.\n{self.describe()}")
            return self._fall_back(
                fallback,
                reason=f"no registered provider serves model {model!r}",
                hard=UnknownModel(
                    f"no registered provider serves model {model!r}.\n{self.describe()}"
                ),
            )

        entry = self._providers.get(name)
        if entry is None:
            raise UnknownModel(f"no provider named {name!r} is registered.\n{self.describe()}")

        key = api_key or self._credential(entry, env)
        if key is None:
            return self._fall_back(
                fallback,
                reason=(
                    f"provider {entry.name!r} has no credential "
                    f"(checked {', '.join(entry.env_vars)})"
                ),
                hard=ProviderNotConfigured(
                    f"provider {entry.name!r} needs a credential. Set one of: "
                    f"{', '.join(entry.env_vars)}. "
                    f"(Or pass api_key= explicitly, or fallback= to degrade.)"
                ),
            )

        factory = self._load_factory(entry)
        chosen = model or entry.default_model
        kwargs: dict[str, Any] = {"api_key": key}
        if entry.accepts_model_kwarg and chosen:
            kwargs["model"] = chosen
        return factory(**kwargs)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _credential(entry: ProviderEntry, env: Mapping[str, str]) -> str | None:
        """First non-empty value among the provider's env vars.

        Empty-string is treated as ABSENT, not as a credential: a shell that
        exports ``OPENAI_API_KEY=`` is a misconfiguration, and passing ``""``
        to a provider produces a confusing 401 far from the cause.
        """
        for var in entry.env_vars:
            value = env.get(var)
            if value:
                return value
        return None

    def _load_factory(self, entry: ProviderEntry) -> Callable[..., Any]:
        """Import the provider factory, translating a missing extra into a
        message that names the extra. Loud, never silent."""
        import importlib

        module_path, _, attr = entry.factory.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise MissingProviderExtra(
                f"provider {entry.name!r} needs an optional dependency that isn't installed. "
                f"Install it with: pip install 'arc-agentkit[{entry.extra}]'. "
                f"(Underlying import error: {exc})"
            ) from exc
        factory: Callable[..., Any] = getattr(module, attr)
        return factory

    def _fall_back(self, fallback: str | None, *, reason: str, hard: RegistryError) -> Any:
        """Either degrade to the requested fallback (warning once) or raise.

        ``reason`` NEVER contains a credential — it names the environment
        variables consulted, not their contents.
        """
        if fallback is None:
            raise hard
        if fallback != "fake":
            raise RegistryError(
                f"unknown fallback {fallback!r}; the only supported value is 'fake'."
            )
        self._warn_once(
            f"fallback:{reason}",
            f"agentkit is falling back to a FAKE LLM because {reason}. "
            f"Responses will be canned, not real. Set the credential, or pass an explicit "
            f"api_key=, before running anything that matters.",
        )
        # Imported here, never at module scope. ``agentkit.testing`` is the
        # test-double boundary and ``tests/meta/test_public_api_surface.py``
        # pins that it stays off the production import path; a top-level
        # import here would drag a fake into every production wiring that
        # merely touched the registry.
        from agentkit.testing import FakeLLM

        return FakeLLM()

    def _warn_once(self, key: str, message: str) -> None:
        import warnings

        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)


# ── the built-in table ───────────────────────────────────────────────────────
#
# Capabilities are declared PER MODEL, not inferred from the name — that is
# Brief 5's constraint and the reason this is a table rather than a set of
# ``startswith`` checks. The rules further down only route a name to a
# provider when no row matches; they never invent capabilities.
#
# This table, like the price table it sits beside, WILL go stale. It is a
# convenience default. An application pins its own truth by registering rows
# over it — ``register_model`` replaces by name.


def _anthropic(name: str, window: int) -> ModelEntry:
    """Anthropic models: tools yes, streaming yes, vision yes; no native
    ``json_schema`` response_format (the equivalent is tool_use, which the
    provider adapter does not wire yet — see ``_infer_response_format``)."""
    return ModelEntry(
        name=name,
        provider="anthropic",
        capabilities=ModelCapabilities(
            tools=_Y,
            structured_output=_Y,
            native_json_schema=_N,
            streaming=_Y,
            vision=_Y,
            context_window=window,
        ),
    )


def _openai(name: str, window: int, *, vision: Capability = _Y) -> ModelEntry:
    """OpenAI models: the only family with native ``json_schema`` mode wired
    end-to-end in this framework today."""
    return ModelEntry(
        name=name,
        provider="openai",
        capabilities=ModelCapabilities(
            tools=_Y,
            structured_output=_Y,
            native_json_schema=_Y,
            streaming=_Y,
            vision=vision,
            context_window=window,
        ),
    )


def _deepseek(name: str, window: int, *, tools: Capability = _Y) -> ModelEntry:
    """DeepSeek models: text-only, no image input."""
    return ModelEntry(
        name=name,
        provider="deepseek",
        capabilities=ModelCapabilities(
            tools=tools,
            structured_output=_Y,
            native_json_schema=_N,
            streaming=_Y,
            vision=_N,
            context_window=window,
        ),
    )


_BUILTIN_PROVIDERS = (
    ProviderEntry(
        name="anthropic",
        env_vars=("ANTHROPIC_API_KEY",),
        factory="agentkit.adapters.llm.providers:claude",
        default_model="claude-sonnet-4-6",
    ),
    ProviderEntry(
        name="openai",
        env_vars=("OPENAI_API_KEY",),
        factory="agentkit.adapters.llm.providers:openai",
    ),
    ProviderEntry(
        name="deepseek",
        env_vars=("DEEPSEEK_API_KEY",),
        factory="agentkit.adapters.llm.providers:deepseek",
        default_model="deepseek-chat",
    ),
    ProviderEntry(
        name="openrouter",
        env_vars=("OPENROUTER_API_KEY",),
        factory="agentkit.adapters.llm.providers:openrouter",
    ),
)

_BUILTIN_MODELS = (
    _anthropic("claude-sonnet-4-6", 200_000),
    _anthropic("claude-haiku-4-5", 200_000),
    _anthropic("claude-opus-4-1", 200_000),
    _openai("gpt-4o", 128_000),
    _openai("gpt-4o-mini", 128_000),
    _openai("gpt-4.1", 1_047_576),
    _openai("gpt-4.1-mini", 1_047_576),
    _deepseek("deepseek-chat", 64_000),
    # deepseek-reasoner does not accept tools — the canonical example of a
    # capability that is real, checkable today, and silently wrong without a
    # declaration: bind it to a ReActCognition and the tools are simply ignored.
    _deepseek("deepseek-reasoner", 64_000, tools=_N),
)


def _builtin_rules() -> list[Rule]:
    """Name→provider fallbacks for models NOT in the table above.

    These route only; they never claim a capability. A ``gpt-`` model we've
    never heard of resolves to the OpenAI provider (correct) and reports
    every capability as UNKNOWN (also correct).
    """

    def by_openrouter_prefix(name: str) -> str | None:
        # OpenRouter names are "vendor/model". If the vendor half isn't one we
        # serve directly, OpenRouter is the plausible route.
        return "openrouter" if "/" in name else None

    def by_family(name: str) -> str | None:
        if name.startswith(("claude-", "anthropic.")):
            return "anthropic"
        if name.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-")):
            return "openai"
        if name.startswith("deepseek"):
            return "deepseek"
        return None

    # ``by_family`` last so it wins over the coarse slash rule for a name like
    # "anthropic/claude-sonnet-9" — ``provider_for`` consults rules
    # most-recently-registered first and offers each one every normalised
    # candidate, so ``by_family`` gets to see the bare "claude-sonnet-9"
    # before ``by_openrouter_prefix`` is asked about the slash. A vendor we do
    # NOT serve directly ("meta-llama/llama-4") still falls through to
    # OpenRouter, which is exactly what that rule's comment claims.
    return [by_openrouter_prefix, by_family]


def default_registry() -> ModelRegistry:
    """A freshly-populated registry with the built-in table. Each call returns
    an INDEPENDENT instance — tests take one of these instead of mutating the
    process-wide default."""
    reg = ModelRegistry()
    for p in _BUILTIN_PROVIDERS:
        reg.register_provider(p)
    for m in _BUILTIN_MODELS:
        reg.register_model(m)
    for r in _builtin_rules():
        reg.register_rule(r)
    return reg


_DEFAULT: ModelRegistry | None = None


def registry() -> ModelRegistry:
    """The process-wide registry. Built on first use, then shared — so an
    application's ``register_model`` / ``register_rule`` calls at startup are
    visible to every later ``Agent`` construction."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = default_registry()
    return _DEFAULT


# ── module-level conveniences ────────────────────────────────────────────────
#
# The common cases, so an application never has to hold a registry object.


def resolve_llm(
    model: str | None = None,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    fallback: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    """:meth:`ModelRegistry.resolve` against the process-wide registry."""
    return registry().resolve(
        model, provider=provider, api_key=api_key, fallback=fallback, env=env
    )


def model_capabilities(model: str) -> ModelCapabilities:
    """:meth:`ModelRegistry.capabilities` against the process-wide registry."""
    return registry().capabilities(model)


def require_capabilities(
    model: str,
    requires: tuple[str, ...] = (),
    *,
    derived: tuple[str, ...] = (),
    min_context_window: int | None = None,
    on_unknown: str = "warn",
    subject: str = "",
) -> None:
    """:meth:`ModelRegistry.check` against the process-wide registry."""
    registry().check(
        model,
        requires,
        derived=derived,
        min_context_window=min_context_window,
        on_unknown=on_unknown,
        subject=subject,
    )


def register_model(entry: ModelEntry) -> None:
    """Declare (or override) a model row on the process-wide registry."""
    registry().register_model(entry)


def register_provider(entry: ProviderEntry) -> None:
    """Declare (or override) a provider on the process-wide registry."""
    registry().register_provider(entry)


def register_rule(rule: Rule) -> None:
    """Add a name→provider rule to the process-wide registry."""
    registry().register_rule(rule)


__all__ = [
    "CAPABILITY_NAMES",
    "Capability",
    "CapabilityMismatch",
    "MissingProviderExtra",
    "ModelCapabilities",
    "ModelEntry",
    "ModelRegistry",
    "ProviderEntry",
    "ProviderNotConfigured",
    "RegistryError",
    "Rule",
    "UnknownModel",
    "default_registry",
    "model_capabilities",
    "normalize_model_name",
    "register_model",
    "register_provider",
    "register_rule",
    "registry",
    "require_capabilities",
    "resolve_llm",
]
