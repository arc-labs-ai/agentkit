"""PydanticAdapter — pin the contract end-to-end:

    1. ``json_schema()`` produces a schema that mirrors the model's field types
       (list[str], int, Optional[float] all show up correctly).
    2. ``parse(valid_json_str)`` returns a typed model instance.
    3. ``parse(invalid_json_str)`` raises :class:`OutputCoercionError` carrying
       the per-field validation errors so the retry middleware can reflect them
       back to the model.
    4. ``serialize(instance)`` round-trips back to the dict shape ``parse`` can
       reconsume.

The whole module is skipped when pydantic is not installed (it's an OPTIONAL
dep)."""

from __future__ import annotations

import json

import pytest

pydantic = pytest.importorskip("pydantic")

from agentkit.capabilities.output_schema import (  # noqa: E402  — must follow importorskip
    OutputCoercionError,
    PydanticAdapter,
)


class Report(pydantic.BaseModel):
    """Mixed model that exercises list, primitive, and Optional[primitive]."""

    title: str
    tags: list[str]
    word_count: int
    confidence: float | None = None


def test_json_schema_reflects_field_types():
    adapter = PydanticAdapter(Report)
    schema = adapter.json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["title"]["type"] == "string"
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    assert props["word_count"]["type"] == "integer"
    # ``confidence`` is Optional[float] → either an anyOf with null or a typed
    # union list, depending on pydantic version. Both are acceptable.
    conf = props["confidence"]
    if "type" in conf:
        types = conf["type"] if isinstance(conf["type"], list) else [conf["type"]]
        assert "null" in types or "number" in types
    else:
        assert any("null" in str(v) or "number" in str(v) for v in conf.values())


def test_parse_returns_typed_instance():
    adapter = PydanticAdapter(Report)
    payload = {"title": "ok", "tags": ["a", "b"], "word_count": 42}
    inst = adapter.parse(json.dumps(payload))
    assert isinstance(inst, Report)
    assert inst.title == "ok"
    assert inst.tags == ["a", "b"]
    assert inst.word_count == 42
    assert inst.confidence is None


def test_parse_accepts_dict_too():
    """Provider structured-output modes return an already-parsed dict; the
    adapter must accept both string and dict inputs."""
    adapter = PydanticAdapter(Report)
    inst = adapter.parse({"title": "x", "tags": [], "word_count": 1})
    assert isinstance(inst, Report)
    assert inst.title == "x"


def test_parse_invalid_raises_with_errors():
    adapter = PydanticAdapter(Report)
    # word_count is an int — passing a string triggers Pydantic validation.
    bad = json.dumps({"title": "ok", "tags": [], "word_count": "not-a-number"})
    with pytest.raises(OutputCoercionError) as ei:
        adapter.parse(bad)
    err = ei.value
    assert err.raw == bad
    assert err.errors, "errors list should not be empty"
    assert any("word_count" in e for e in err.errors)


def test_parse_invalid_json_string_raises():
    adapter = PydanticAdapter(Report)
    with pytest.raises(OutputCoercionError):
        adapter.parse("not json at all")


def test_parse_strips_markdown_json_fences():
    """Claude routinely wraps JSON output in ```json … ``` even when asked
    for raw JSON. The adapter must peel the fence before parsing — otherwise
    every Reader / Synthesizer / Critic call would fail with InvalidJSON and
    burn retries until the model coincidentally stops fencing."""
    adapter = PydanticAdapter(Report)
    fenced = '```json\n{"title": "t", "tags": ["x"], "word_count": 3, "confidence": 0.5}\n```'
    result = adapter.parse(fenced)
    assert result == Report(title="t", tags=["x"], word_count=3, confidence=0.5)


def test_parse_strips_bare_triple_backtick_fences():
    """Some models emit ``` without the ``json`` language tag. Still safe to
    strip — valid JSON cannot start with a backtick."""
    adapter = PydanticAdapter(Report)
    fenced = '```\n{"title": "t", "tags": [], "word_count": 1, "confidence": 0.1}\n```'
    result = adapter.parse(fenced)
    assert result.title == "t"


def test_parse_passes_raw_json_through_unchanged():
    """The fence-strip helper must be a no-op for properly formatted raw
    JSON — no spurious transforms that could corrupt edge-case payloads."""
    adapter = PydanticAdapter(Report)
    raw = '{"title": "t", "tags": [], "word_count": 1, "confidence": 0.1}'
    result = adapter.parse(raw)
    assert result.title == "t"


def test_serialize_round_trips():
    adapter = PydanticAdapter(Report)
    inst = Report(title="t", tags=["x"], word_count=3, confidence=0.5)
    dumped = adapter.serialize(inst)
    assert dumped == {"title": "t", "tags": ["x"], "word_count": 3, "confidence": 0.5}
    # And parse(dumped) reconstructs the same instance.
    rebuilt = adapter.parse(dumped)
    assert rebuilt == inst


def test_name_defaults_to_class_name():
    adapter = PydanticAdapter(Report)
    assert adapter.name == "Report"


def test_name_override():
    adapter = PydanticAdapter(Report, name="my_custom_report")
    assert adapter.name == "my_custom_report"


def test_python_type_is_the_model_class():
    adapter = PydanticAdapter(Report)
    assert adapter.python_type is Report


def test_partial_parse_returns_partial_model():
    """Streaming partial parse: the tolerant JSON path closes the open
    string and ``model_construct`` builds the partial without validating
    missing required fields."""
    adapter = PydanticAdapter(Report)
    partial = adapter.partial_parse('{"title":"par')
    assert partial is not None
    assert isinstance(partial, Report)
    # ``model_dump(exclude_unset=True)`` shows only the fields the partial
    # stream actually populated — missing required ``tags`` / ``word_count``
    # are intentionally absent.
    assert partial.model_dump(exclude_unset=True) == {"title": "par"}


def test_partial_parse_empty_is_none():
    """Zero useful structure → ``None`` (caller pumps another delta)."""
    adapter = PydanticAdapter(Report)
    assert adapter.partial_parse("") is None
    assert adapter.partial_parse("   ") is None


def test_constructor_rejects_non_basemodel():
    with pytest.raises(TypeError):
        PydanticAdapter(dict)  # type: ignore[arg-type]
