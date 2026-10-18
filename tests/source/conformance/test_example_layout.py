"""Each runnable example owns a directory and a module entry point."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = ("call_runtime", "single_agent", "agent_runner_tools", "react_tools_and_events")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", EXAMPLES)
def test_examples_have_isolated_runnable_entrypoints(name):
    assert (ROOT / "examples" / name / "__main__.py").is_file()
    result = subprocess.run(
        [sys.executable, "-m", f"tests.examples.{name}"], cwd=ROOT.parent, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
