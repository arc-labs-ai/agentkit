"""Test-tree pytest config.

Adds the agentkit project root to `sys.path` so the `examples/` top-level (which is NOT a
distributed package — it's a runnable-demo dir intentionally kept out of the wheel) can be
imported as `examples.multi_agent` from `tests/test_examples.py`. Without this, pytest invoked
from any working directory other than `/agentkit/` itself can't resolve the `examples` import
and `tests/test_examples.py` errors out at collection. Keeping examples out of the wheel and
on the test sys.path captures the intent: examples ship with the source tree, never with the
installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENTKIT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENTKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTKIT_ROOT))

# The tests directory itself, so shared test-only helpers import by bare name
# (`from _assertions import assert_money`). Needed because the suite runs under
# `--import-mode=importlib`, which deliberately does NOT put a test module's
# own directory on `sys.path` — that is the behaviour which lets two
# same-named test modules coexist in different packages, and it is worth
# keeping. Helper modules are underscore-prefixed so pytest never collects
# them as tests.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


# ── hypothesis profiles ──────────────────────────────────────────────────────
# Registered here rather than in a test module because ``--hypothesis-profile``
# is resolved at CONFIG time, before any test file is imported.
#
# Only ``deep`` is registered. An earlier version also overrode the DEFAULT
# profile to 250 examples, which silently made every other property test in
# the repo search 2.5x harder and added ~12s to the full run — a cost that
# unrelated files paid for a convenience in one of them. The ordinary run
# therefore uses Hypothesis's own default; ``deep`` is for a pre-release sweep:
#
#     pytest tests/agents/cognition/test_cli_parser_properties.py \
#         --hypothesis-profile=deep
from hypothesis import HealthCheck
from hypothesis import settings as _hypothesis_settings

_hypothesis_settings.register_profile(
    "deep",
    max_examples=4000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
