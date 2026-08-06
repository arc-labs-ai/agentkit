"""Integration-test shared fixtures + skip markers.

Real-provider tests are gated on API-key env vars so CI (which has no keys)
skips them cleanly. Mocked tests run everywhere.

Cost discipline: every real call uses ``max_tokens=200`` (or less) and
uses the cheapest current model families — ``claude-haiku-4-5`` and
``gpt-4o-mini`` — never Opus / GPT-5. Total spend across the suite is
budgeted under ~$3.
"""

from __future__ import annotations

import os

import pytest

# The gates. A real-call test decorates itself with @requires_anthropic /
# @requires_openai; without keys the test skips cleanly.
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real-provider integration test",
)

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-provider integration test",
)


# Small hard caps — every real-call test uses these constants so the whole
# suite is under one cost knob.
HAIKU_MODEL = "claude-haiku-4-5"
OPENAI_MINI_MODEL = "gpt-4o-mini"  # gpt-5-mini not on billing during this run
MAX_TOKENS = 200


@pytest.fixture
def anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("no ANTHROPIC_API_KEY")
    return key


@pytest.fixture
def openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("no OPENAI_API_KEY")
    return key
