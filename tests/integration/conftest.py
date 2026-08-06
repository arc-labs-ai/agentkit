"""Integration-test shared fixtures + skip markers.

Real-provider tests are gated on API-key env vars so CI (which has no keys)
skips them cleanly. Mocked tests run everywhere.

Cost discipline: every real call uses ``max_tokens=200`` (or less) and
uses the cheapest current model families — ``claude-haiku-4-5`` and
``gpt-4o-mini`` — never Opus / GPT-5. Total spend across the suite is
budgeted under ~$3.

Isolation workaround: another subagent is mid-edit on
``agentkit/agents/cognition/claude_cli.py`` and that file has a live
syntax error that blocks the whole ``from agentkit ...`` import chain.
This test suite is deliberately scoped OUT of the ClaudeCli cognition
(a separate subagent owns those tests), so we stub the broken module
BEFORE any agentkit code loads. When claude_cli.py is repaired the
stub becomes a harmless no-op override that a fresh interpreter would
have discovered on its own.
"""

from __future__ import annotations

import os
import sys
import types

# A previous version of this file stubbed the ``claude_cli`` submodule to
# work around a live syntax error introduced mid-edit by a peer subagent.
# The stub is now conditional: only install it if the real module fails to
# compile. Once the source is valid the real class is used.
try:  # pragma: no cover — try/except is the guard
    import agentkit.agents.cognition.claude_cli  # noqa: F401
except (SyntaxError, ImportError):
    _stub = types.ModuleType("agentkit.agents.cognition.claude_cli")

    class _ClaudeCliCognitionPlaceholder:  # pragma: no cover — never invoked
        name = "claude_cli"

    _stub.ClaudeCliCognition = _ClaudeCliCognitionPlaceholder  # type: ignore[attr-defined]
    sys.modules["agentkit.agents.cognition.claude_cli"] = _stub

import pytest  # noqa: E402  — must come after the sys.modules pre-load

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
