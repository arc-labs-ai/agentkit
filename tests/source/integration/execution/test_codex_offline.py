"""Real Codex sandbox inventory without provider dispatch or credentials.

Read-only write denial is demonstrated separately from native-tool denial and
credential secrecy. These tests do not authorize a contained execution profile.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys

import pytest

from agentkit.models.providers.cli._common import _terminate


@pytest.mark.asyncio
@pytest.mark.skipif(
    shutil.which("codex") is None or os.getenv("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="real Codex CLI unavailable or explicitly disabled",
)
async def test_real_codex_read_only_denies_write_but_can_read_an_outside_fake_credential(tmp_path):
    home, work, other = (tmp_path / name for name in ("home", "work", "other-job"))
    for path in (home, work, other):
        path.mkdir()
    fake_credential = other / "auth.json"
    fake_credential.write_text("planted-not-real")
    target = work / "prohibited-write"
    script = """
import json,sys
from pathlib import Path
read = Path(sys.argv[1]).read_text() == 'planted-not-real'
try:
    Path(sys.argv[2]).write_text('violation')
except PermissionError:
    denied = True
else:
    denied = False
print(json.dumps({'outside_read': read, 'write_denied': denied}))
"""
    process = await asyncio.create_subprocess_exec(
        shutil.which("codex"),
        "sandbox",
        "-c",
        'sandbox_mode="read-only"',
        "--",
        sys.executable,
        "-c",
        script,
        str(fake_credential),
        str(target),
        cwd=work,
        env={"HOME": str(home), "CODEX_HOME": str(home), "PATH": os.defpath},
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 10)
    finally:
        await _terminate(process, 0.1)
    assert process.returncode == 0, "The real sandbox must run the planted probe"
    assert json.loads(output) == {"outside_read": True, "write_denied": True}
    assert not target.exists()
