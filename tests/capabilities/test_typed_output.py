"""Agent typed/validated output — dependency-free `parse` + reflect-error-to-model repair (offline)."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentkit.agents import Agent
from agentkit.kernel.types import Usage
from agentkit.testing import FakeLLM, Turn, make_test_ctx


def _run(agent, ctx):
    return asyncio.run(agent.run("task", ctx))


def test_returns_raw_when_no_parser():
    r = _run(Agent(name="a", model="m"), make_test_ctx(llm=FakeLLM('{"x": 1}')))
    assert r.output == '{"x": 1}' and r.parsed is None and not r.partial


def test_parse_returns_typed_object():
    r = _run(Agent(name="a", model="m", parse=json.loads), make_test_ctx(llm=FakeLLM('{"x": 1}')))
    assert r.parsed == {"x": 1} and not r.partial


def test_reflect_and_retry_recovers():
    # first reply is invalid JSON → error reflected back → second reply parses.
    llm = FakeLLM.script([Turn(content="not json at all"), Turn(content='{"x": 2}')])
    r = _run(Agent(name="a", model="m", parse=json.loads, max_repairs=1), make_test_ctx(llm=llm))
    assert r.parsed == {"x": 2} and not r.partial
    assert llm.calls == 2  # one repair round


def test_repair_exhausted_returns_partial():
    llm = FakeLLM("never valid json")  # every attempt returns the same un-parseable text
    r = _run(Agent(name="a", model="m", parse=json.loads, max_repairs=1), make_test_ctx(llm=llm))
    assert r.partial and r.parsed is None
    assert r.evals["stop_reason"] == "invalid_output" and "error" in r.evals
    assert llm.calls == 2  # initial attempt + 1 repair, then give up


def test_usage_accumulates_across_repairs():
    llm = FakeLLM.script(
        [
            Turn(content="bad", usage=Usage(1, 1, 0.001)),
            Turn(content="{}", usage=Usage(1, 1, 0.001)),
        ]
    )
    r = _run(Agent(name="a", model="m", parse=json.loads), make_test_ctx(llm=llm))
    assert r.parsed == {} and r.usage.cost_usd == pytest.approx(0.002)


def test_custom_validator_can_enforce_a_schema():
    # `parse` is arbitrary: here it asserts shape, not just JSON-wellformedness.
    def parse_label(text: str) -> str:
        obj = json.loads(text)
        if obj.get("label") not in {"faithful", "unfaithful"}:
            raise ValueError("label not in vocab")
        return obj["label"]

    ok = _run(
        Agent(name="a", model="m", parse=parse_label),
        make_test_ctx(llm=FakeLLM('{"label": "faithful"}')),
    )
    assert ok.parsed == "faithful"
    bad = _run(
        Agent(name="a", model="m", parse=parse_label, max_repairs=0),
        make_test_ctx(llm=FakeLLM('{"label": "maybe"}')),
    )
    assert bad.partial and bad.parsed is None
