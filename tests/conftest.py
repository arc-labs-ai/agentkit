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
