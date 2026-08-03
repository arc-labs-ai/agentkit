"""Tool-writing conventions baked into `@tool` / `FunctionTool.from_callable` (per Anthropic's
"Writing tools for AI agents"): a poorly-specified tool fails at *registration* time, not runtime.
We assert the four wiring contracts the framework's safety primitives depend on:

  - Description is the spec  → docstring must clear a >=30-char floor or registration raises.
  - Side-effecting is declared → `@tool` rejects an implicit/missing `side_effecting=` keyword.
  - Idempotency is opt-in    → `idempotent=True` round-trips on the dataclass.
  - URL allowlisting hook    → `url_arg` round-trips on the dataclass (unchanged behavior).

All offline & deterministic; no LLM, no real I/O."""

import asyncio

import pytest

from agentkit.tools import FunctionTool, ToolDefinitionError, ToolRegistry, tool

# ---- happy path: a correctly-declared tool registers fine -------------------------------------


def test_well_declared_tool_registers_and_round_trips_fields():
    @tool(side_effecting=False, idempotent=True)
    def search(query: str) -> str:
        """Search the indexed corpus for `query` and return the top hit."""
        return f"hit for {query}"

    assert isinstance(search, FunctionTool)
    assert search.name == "search"
    assert search.description.startswith("Search the indexed corpus")
    assert search.side_effecting is False
    assert search.idempotent is True
    # the description floor passes, so the schema description matches the docstring's first line
    assert (
        search.schema.description == "Search the indexed corpus for `query` and return the top hit."
    )


def test_side_effecting_true_round_trips():
    @tool(side_effecting=True)
    def purge(path: str) -> str:
        """Permanently delete the file at `path` from the durable store."""
        return f"purged {path}"

    assert purge.side_effecting is True
    assert purge.idempotent is False  # default


# ---- description floor: a thin docstring fails at decoration time -----------------------------


def test_short_docstring_raises_tool_definition_error_at_decoration():
    with pytest.raises(ToolDefinitionError) as exc:

        @tool(side_effecting=False)
        def thin(x: str) -> str:
            """too short"""
            return x

    msg = str(exc.value)
    assert "at least 30 characters" in msg and "'thin'" in msg


def test_missing_docstring_raises_tool_definition_error_at_decoration():
    with pytest.raises(ToolDefinitionError):

        @tool(side_effecting=False)
        def no_doc(x: str) -> str:
            return x


def test_from_callable_without_docstring_raises():
    def naked(x: str) -> str:
        return x

    with pytest.raises(ToolDefinitionError) as exc:
        FunctionTool.from_callable(naked)
    assert "'naked'" in str(exc.value)


def test_explicit_description_overrides_missing_docstring():
    # An author can still wire a callable that has no docstring by passing an explicit description
    # — the floor is on the resolved description text, not on the docstring per se.
    def no_doc(x: str) -> str:
        return x

    t = FunctionTool.from_callable(
        no_doc,
        description="Echo `x` back unchanged; trivial pass-through for testing.",
        side_effecting=False,
    )
    assert t.description.startswith("Echo `x` back unchanged")


def test_explicit_description_below_floor_still_raises():
    def no_doc(x: str) -> str:
        return x

    with pytest.raises(ToolDefinitionError):
        FunctionTool.from_callable(no_doc, description="too short", side_effecting=False)


# ---- side_effecting is REQUIRED on @tool ------------------------------------------------------


def test_missing_side_effecting_raises_at_decoration():
    with pytest.raises(ToolDefinitionError) as exc:

        @tool()
        def fetch(url: str) -> str:
            """Fetch the resource at `url` and return its body as text."""
            return f"body of {url}"

    assert "side_effecting" in str(exc.value)


def test_bare_decorator_form_without_side_effecting_raises():
    # `@tool` (no parens, no kwargs) — applied directly to the function — must also fail,
    # because we never silently default `side_effecting`.
    with pytest.raises(ToolDefinitionError):

        @tool
        def fetch(url: str) -> str:
            """Fetch the resource at `url` and return its body as text."""
            return f"body of {url}"


# ---- idempotent + url_arg round-trip on the dataclass -----------------------------------------


def test_idempotent_field_round_trips_via_decorator():
    @tool(side_effecting=False, idempotent=True)
    def lookup(key: str) -> str:
        """Look up the value associated with `key` in the read-only index."""
        return f"value({key})"

    assert lookup.idempotent is True
    assert lookup.side_effecting is False


def test_url_arg_round_trips_unchanged():
    # url_arg is the FetchPort/egress hook; tightening tool conventions must not regress it.
    @tool(side_effecting=False, url_arg="url")
    def fetch(url: str) -> str:
        """Fetch the resource at `url` and return its body; egress-gated path."""
        return f"body of {url}"

    assert fetch.url_arg == "url"


# ---- ToolRegistry.from_tools surfaces the same errors -----------------------------------------


def test_registry_from_tools_surfaces_thin_docstring_error():
    def bad(x: str) -> str:
        """short"""
        return x

    with pytest.raises(ToolDefinitionError):
        ToolRegistry.from_tools([bad])


def test_registry_from_tools_accepts_well_declared_callable():
    def good(query: str) -> str:
        """Search the corpus for `query` and return the top hit (read-only)."""
        return f"hit for {query}"

    reg = ToolRegistry.from_tools([good])
    assert reg.names() == ["good"]
    t = reg.get("good")
    # plain-callable path defaults side_effecting=False (the read-only default); idempotent stays False
    assert t.side_effecting is False
    assert t.idempotent is False


# ---- the decorated tool still executes ---------------------------------------------------------


def test_decorated_tool_still_runs():
    @tool(side_effecting=False)
    def echo(msg: str) -> str:
        """Return `msg` back to the caller verbatim (test stub)."""
        return msg

    out = asyncio.run(echo.run({"msg": "hello"}, ctx=None))
    assert out == "hello"
