"""A tool call is checked against the signature the model was shown.

Two halves of one contract, both previously broken in the direction that hides
the problem:

**The call.** Unknown argument names were dropped silently. A parameter with a
DEFAULT then ran with that default, so a model calling
``notify(message="page the on-call, prod is down")`` against
``def notify(msg: str = "default message")`` got back ``"sent: default
message"`` — a side-effecting tool reporting success for something it was never
asked to do. Nothing downstream could tell. A missing required argument fared
slightly better, surfacing as a raw ``TypeError: search() missing 1 required
positional argument``, which names the Python function rather than the tool and
tells the model nothing about what it may pass.

**The schema.** A structured parameter (Pydantic / dataclass / attrs) was
advertised as ``{"type": "string"}``, so the model sent a string and the
function received a ``str`` where its annotation promised an object — the schema
was instructing the model to call the tool wrongly. An ``Enum`` parameter was
advertised as a bare string with no ``enum`` list, leaving the model free to
invent a member.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass

import pytest

from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import ToolArgumentError, tool
from agentkit.tools.schema import _build_schema, _json_type

CTX = make_test_ctx(llm=FakeLLM("x"))


def _run(t, args):  # noqa: ANN001, ANN202
    return asyncio.run(t.run(args, CTX))


@tool(side_effecting=True)
def notify(msg: str = "default message") -> str:
    """Send a notification to the on-call engineer with the given message."""
    return f"sent: {msg}"


@tool(side_effecting=False)
def search(query: str, limit: int = 5) -> str:
    """Search the corpus for documents matching the given query string."""
    return f"{limit} results for {query}"


# ── 1. the call is validated ────────────────────────────────────────────────


def test_an_unknown_argument_is_refused_not_dropped() -> None:
    """THE regression. Before: ``sent: default message`` — success, for a
    notification nobody asked for."""
    with pytest.raises(ToolArgumentError) as exc:
        _run(notify, {"message": "PAGE THE ONCALL, prod is down"})

    err = exc.value
    assert err.tool_name == "notify"
    assert err.unexpected == ("message",)
    assert err.accepted == ("msg",)
    # The message has to be actionable by the MODEL — it names the tool, the
    # offending key and the accepted set, because the model authored the call
    # and is the only party that can fix it.
    text = str(err)
    assert "notify" in text and "message" in text and "msg" in text


def test_a_missing_required_argument_names_the_tool() -> None:
    """Was a raw ``TypeError`` naming the Python function."""
    with pytest.raises(ToolArgumentError) as exc:
        _run(search, {})
    assert exc.value.missing == ("query",)


def test_a_typo_reports_both_halves() -> None:
    """One typo is simultaneously an unexpected key and a missing one; the model
    can only correct it if it sees both."""
    with pytest.raises(ToolArgumentError) as exc:
        _run(search, {"querry": "x"})
    assert exc.value.unexpected == ("querry",) and exc.value.missing == ("query",)


def test_a_valid_call_is_unaffected() -> None:
    """The positive control, including an omitted OPTIONAL argument — the check
    must not turn defaults into requirements."""
    assert _run(search, {"query": "octopus"}) == "5 results for octopus"
    assert _run(search, {"query": "octopus", "limit": 2}) == "2 results for octopus"


def test_it_is_a_value_error() -> None:
    """``ToolArgumentError`` subclasses ``ValueError`` so the framework's
    existing per-tool isolation keeps catching it."""
    assert issubclass(ToolArgumentError, ValueError)


# ── 2. **kwargs is the opt-out ──────────────────────────────────────────────


def test_a_tool_declaring_kwargs_receives_the_extras() -> None:
    """Declaring ``**kwargs`` is the author saying "I accept keys I did not
    enumerate". Before, such a tool was neither strict nor permissive: the
    extras were dropped and it never saw one."""

    @tool(side_effecting=False)
    def flexible(a: str, **extra: object) -> str:
        """Accept arbitrary extra keyword arguments alongside the declared one."""
        return f"a={a} extra={sorted(extra)}"

    assert _run(flexible, {"a": "1", "b": 2, "c": 3}) == "a=1 extra=['b', 'c']"


def test_kwargs_does_not_excuse_a_missing_required_argument() -> None:
    """Permissive about EXTRA keys is not permissive about absent ones."""

    @tool(side_effecting=False)
    def flexible(a: str, **extra: object) -> str:
        """Accept arbitrary extra keyword arguments alongside the declared one."""
        return "ok"

    with pytest.raises(ToolArgumentError):
        _run(flexible, {"b": 2})


# ── 3. a ctx key from the model is dropped, never injected ──────────────────


def test_a_model_supplied_ctx_key_cannot_reach_the_tool() -> None:
    """``ctx`` is injected by the framework and is not part of the advertised
    schema. A model naming it must neither override the real one nor trip the
    unknown-argument check (it is invisible to the model by construction)."""
    seen = {}

    @tool(side_effecting=False)
    def peek(a: str, ctx=None) -> str:  # noqa: ANN001
        """Record which run context object the framework injected for this call."""
        seen["ctx"] = ctx
        return "ok"

    assert _run(peek, {"a": "1", "ctx": {"evil": True}}) == "ok"
    assert seen["ctx"] is CTX


# ── 4. the advertised schema tells the truth ────────────────────────────────


@dataclass
class Filter:
    field: str
    op: str


def test_a_structured_parameter_advertises_an_object() -> None:
    """Was ``{"type": "string"}`` — the schema instructing the model to send the
    wrong shape."""

    def find(f: Filter, limit: int = 10) -> str:
        """Find records matching the structured filter, up to the given limit."""
        return "ok"

    props = _build_schema(find, "find", "Find records matching a structured filter.").parameters[
        "properties"
    ]
    assert props["f"]["type"] == "object"
    assert set(props["f"]["properties"]) == {"field", "op"}
    assert props["limit"] == {"type": "integer"}  # the primitive path is untouched


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Priority(enum.Enum):
    LOW = 1
    HIGH = 2


def test_an_enum_parameter_advertises_its_members() -> None:
    """Same treatment ``Literal`` already got — otherwise the model is free to
    invent a member and the tool raises on a value the schema implied was
    fine."""
    assert _json_type(Colour) == {"enum": ["red", "blue"], "type": "string"}
    assert _json_type(Priority) == {"enum": [1, 2], "type": "integer"}


def test_a_mixed_enum_declares_values_without_a_type() -> None:
    """Heterogeneous members have no single JSON type; listing the values
    without pinning one is honest, and matches the ``Literal`` branch."""

    class Mixed(enum.Enum):
        A = "a"
        B = 2

    assert _json_type(Mixed) == {"enum": ["a", 2]}


def test_an_unknown_annotation_still_degrades_to_string() -> None:
    """The fallback stays: schema inference is best-effort and must never fail
    tool construction."""

    class Opaque:
        pass

    assert _json_type(Opaque) == {"type": "string"}
