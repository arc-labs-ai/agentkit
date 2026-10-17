"""Workspace permission and actual OS process-boundary checks.

Every canary is synthetic and created under pytest's temporary directory.
These tests never load root evaluation credentials or contact model services.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from tests.evals.sandbox.workspace import SandboxUnavailable, WorkspaceSandbox


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source.py").write_text("VALUE = 1\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_visible.py").write_text("def test_ok():\n    assert True\n")
    sandbox = WorkspaceSandbox(root, test_timeout_s=10)
    sandbox.allow_write_paths(["source.py"])
    sandbox.allow_test_paths(["tests/test_visible.py"])
    return sandbox


@pytest.fixture
def os_workspace(workspace):
    if not WorkspaceSandbox.process_available():
        pytest.skip("actual process isolation requires macOS sandbox-exec or Linux bubblewrap")
    return workspace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["../outside.txt", "/etc/passwd", "tests/../../outside.txt", ".env.synthetic", "nested/.env.synthetic"]
)
async def test_workspace_read_and_write_refuse_escape_or_secret_paths(workspace, path):
    with pytest.raises(PermissionError):
        await workspace.read_file(path)
    with pytest.raises(PermissionError):
        await workspace.write_file(path, "forbidden")
    assert workspace.audit == []


@pytest.mark.asyncio
async def test_write_grant_is_exact_and_test_files_remain_immutable(workspace):
    assert await workspace.write_file("source.py", "VALUE = 2\n") == "Updated source.py"
    assert await workspace.read_file("source.py") == "VALUE = 2\n"
    with pytest.raises(PermissionError):
        await workspace.write_file("tests/test_visible.py", "def test_ok():\n    pass\n")
    with pytest.raises(PermissionError):
        await workspace.write_file("new_source.py", "VALUE = 3\n")
    assert (workspace.root / "tests" / "test_visible.py").read_text().endswith("assert True\n")
    assert workspace.audit[0] == {"operation": "write", "path": "source.py"}


@pytest.mark.asyncio
@pytest.mark.parametrize("directory_link", [False, True])
async def test_workspace_tools_refuse_file_and_directory_symlinks(workspace, tmp_path, directory_link):
    external = tmp_path / "outside"
    external.mkdir()
    canary = external / "synthetic-canary.txt"
    canary.write_text("SYNTHETIC_PRIVATE_CANARY")
    if directory_link:
        (workspace.root / "linked").symlink_to(external, target_is_directory=True)
        path = "linked/synthetic-canary.txt"
    else:
        (workspace.root / "linked.txt").symlink_to(canary)
        path = "linked.txt"
    with pytest.raises(PermissionError):
        await workspace.read_file(path)
    with pytest.raises(PermissionError):
        workspace.allow_write_paths([path])
    assert all("linked" not in item for item in await workspace.list_files())


@pytest.mark.asyncio
async def test_model_tools_hide_private_grading_and_process_scratch(workspace):
    for directory in (".grading", ".process", "__pycache__"):
        (workspace.root / directory).mkdir()
        (workspace.root / directory / "private.txt").write_text("synthetic private data")
    assert await workspace.list_files() == ["source.py", "tests/test_visible.py"]
    with pytest.raises(PermissionError, match="grader"):
        await workspace.read_file(".grading/private.txt")


@pytest.mark.asyncio
async def test_file_read_and_write_bounds_apply_before_unbounded_content_is_returned(workspace):
    workspace.MAX_FILE_BYTES = 16
    (workspace.root / "large.txt").write_text("x" * 17)
    with pytest.raises(ValueError, match="bounded read"):
        await workspace.read_file("large.txt")
    with pytest.raises(ValueError, match="size bound"):
        await workspace.write_file("source.py", "x" * 17)
    assert (workspace.root / "source.py").read_text() == "VALUE = 1\n"


@pytest.mark.asyncio
async def test_unsupported_platform_explicitly_refuses_process_execution(workspace, monkeypatch):
    monkeypatch.setattr(WorkspaceSandbox, "process_available", staticmethod(lambda: False))
    with pytest.raises(SandboxUnavailable):
        await workspace.run_tests(["tests/test_visible.py"])
    assert workspace.test_runs == []


@pytest.mark.asyncio
async def test_test_path_allowlist_is_checked_before_process_creation(workspace, monkeypatch):
    monkeypatch.setattr(WorkspaceSandbox, "process_available", staticmethod(lambda: True))

    async def unexpected_process(*args, **kwargs):
        pytest.fail("unapproved test path created a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_process)
    with pytest.raises(PermissionError, match="approved"):
        await workspace.run_tests(["tests/not_approved.py"])
    with pytest.raises(PermissionError):
        await workspace.run_tests(["../not_approved.py"])
    with pytest.raises(ValueError):
        await workspace.run_tests([])
    with pytest.raises(ValueError):
        await workspace.run_tests(["tests/test_visible.py"] * 33)


async def _heartbeat_stopped(path: Path, *, settle_s: float = 0.75) -> bool:
    """Has whatever was writing ``path`` stopped?

    Liveness is observed through a file the sandboxed process writes, never
    through its pid. ``--unshare-all`` includes ``--unshare-pid``, so a pid
    captured INSIDE the sandbox is namespace-local — typically a single digit —
    and ``os.kill`` on that number from the host addresses a completely
    different, usually root-owned, process. That reads as
    ``PermissionError: [Errno 1] Operation not permitted`` rather than as the
    wrong question it is. macOS has no pid namespace, which is why pid-based
    probes passed there and only ever failed on Linux.

    The scratch directory is a writable bind mount shared with the host, so the
    heartbeat crosses the boundary that pids cannot.
    """
    before = await asyncio.to_thread(path.read_text)
    await asyncio.sleep(settle_s)
    after = await asyncio.to_thread(path.read_text)
    return after == before


def install_test(sandbox: WorkspaceSandbox, source: str) -> None:
    (sandbox.root / "tests" / "test_visible.py").write_text(source)


@pytest.mark.asyncio
async def test_actual_sandbox_runs_approved_tests_and_records_reviewable_result(os_workspace):
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    # The output is the message, because the sandbox's failures are reported BY
    # the sandbox: a bare `assert 1 == 0` here says nothing, while the output
    # carries the actual `bwrap: execvp ...` or inner pytest traceback.
    assert result["return_code"] == 0 and "1 passed" in result["output"], result["output"]
    assert result["paths"] == ["tests/test_visible.py"]
    assert result["duration_s"] >= 0 and not result["truncated"]
    assert os_workspace.test_runs == [result]


@pytest.mark.asyncio
async def test_actual_test_process_receives_no_parent_credentials_or_python_injection(os_workspace, monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AGENTKIT_PRIVATE_CANARY",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(name, "SYNTHETIC_PARENT_ONLY_CANARY")
    install_test(
        os_workspace,
        """
import os

def test_clean_environment():
    forbidden = (
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN',
        'CLAUDE_CODE_OAUTH_TOKEN', 'AGENTKIT_PRIVATE_CANARY', 'PYTHONPATH',
    )
    assert not any(name in os.environ for name in forbidden)
    assert os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
    assert os.environ['HOME'].endswith('/.process')
""",
    )
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    assert result["return_code"] == 0, result["output"]
    assert "SYNTHETIC_PARENT_ONLY_CANARY" not in result["output"]


@pytest.mark.asyncio
async def test_actual_process_cannot_read_or_write_a_synthetic_file_outside_workspace(os_workspace, tmp_path):
    canary = tmp_path / "synthetic-private-canary.txt"
    canary.write_text("SYNTHETIC_OUTSIDE_CANARY")
    target = json.dumps(str(canary))
    install_test(
        os_workspace,
        f"""
from pathlib import Path
import pytest

def test_outside_file_denied():
    target = Path({target})
    with pytest.raises(OSError):
        target.read_text()
        target.write_text('attempted overwrite')
""",
    )
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    assert result["return_code"] == 0, result["output"]
    assert canary.read_text() == "SYNTHETIC_OUTSIDE_CANARY"
    assert "SYNTHETIC_OUTSIDE_CANARY" not in result["output"]


@pytest.mark.asyncio
async def test_actual_process_network_denied_even_for_a_listening_loopback_service(os_workspace):
    accepted = []

    async def connection(reader, writer):
        accepted.append(True)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    install_test(
        os_workspace,
        f"""
import socket
import pytest

def test_network_denied():
    with pytest.raises(OSError):
        socket.create_connection(('127.0.0.1', {port}), timeout=1)
""",
    )
    try:
        result = await os_workspace.run_tests(["tests/test_visible.py"])
    finally:
        server.close()
        await server.wait_closed()
    assert result["return_code"] == 0, result["output"]
    assert accepted == []


@pytest.mark.asyncio
async def test_actual_process_output_is_bounded_while_pipe_is_fully_drained(os_workspace):
    os_workspace.MAX_OUTPUT_BYTES = 4096
    install_test(
        os_workspace,
        """
def test_large_failed_output():
    print('OUTPUT-CANARY-' + 'x' * 300000)
    assert False, 'expected fixture failure'
""",
    )
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    assert result["return_code"] == 1 and result["truncated"]
    assert len(result["output"].encode()) <= os_workspace.MAX_OUTPUT_BYTES
    assert result["duration_s"] < os_workspace.test_timeout_s


@pytest.mark.asyncio
async def test_actual_process_timeout_kills_descendant_process_group(os_workspace):
    os_workspace.test_timeout_s = 1.5
    beat = os_workspace.root / ".process" / "descendant.beat"
    child = (
        "import time\n"
        "from pathlib import Path\n"
        f"p = Path({str(beat)!r})\n"
        "while True:\n"
        "    p.write_text(str(time.monotonic()))\n"
        "    time.sleep(0.05)\n"
    )
    install_test(
        os_workspace,
        f"""
import subprocess
import sys
import time

def test_hanging_descendant():
    subprocess.Popen([sys.executable, '-I', '-c', {child!r}])
    time.sleep(60)
""",
    )
    with pytest.raises(TimeoutError):
        await os_workspace.run_tests(["tests/test_visible.py"])
    assert beat.exists(), "timeout occurred before the descendant probe could start"
    assert await _heartbeat_stopped(beat), "sandbox timeout left a descendant running"
    assert os_workspace.test_runs == []


@pytest.mark.asyncio
async def test_actual_process_cannot_modify_immutable_tests_outside_write_grant(os_workspace):
    install_test(
        os_workspace,
        """
from pathlib import Path
import pytest

def test_immutable_file_denied():
    with pytest.raises(OSError):
        Path('tests/test_visible.py').write_text('tampered')
""",
    )
    original = (os_workspace.root / "tests" / "test_visible.py").read_text()
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    assert result["return_code"] == 0, result["output"]
    assert (os_workspace.root / "tests" / "test_visible.py").read_text() == original


@pytest.mark.asyncio
async def test_actual_process_cannot_read_outside_canary_through_workspace_symlink(os_workspace, tmp_path):
    canary = tmp_path / "synthetic-symlink-canary.txt"
    canary.write_text("SYNTHETIC_SYMLINK_CANARY")
    (os_workspace.root / "linked-canary.txt").symlink_to(canary)
    install_test(
        os_workspace,
        """
from pathlib import Path
import pytest

def test_symlink_target_denied():
    with pytest.raises(OSError):
        Path('linked-canary.txt').read_text()
""",
    )
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    assert result["return_code"] == 0, result["output"]
    assert "SYNTHETIC_SYMLINK_CANARY" not in result["output"]


@pytest.mark.asyncio
async def test_python_test_process_cannot_write_even_the_parent_tool_approved_source(os_workspace):
    install_test(
        os_workspace,
        """
from pathlib import Path
import pytest

def test_parent_only_source_writes():
    with pytest.raises(OSError):
        Path('source.py').write_text('VALUE = 99')
""",
    )
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    assert result["return_code"] == 0, result["output"]
    assert (os_workspace.root / "source.py").read_text() == "VALUE = 1\n"
    assert await os_workspace.write_file("source.py", "VALUE = 2\n") == "Updated source.py"


@pytest.mark.asyncio
async def test_cancelling_actual_test_process_kills_it_before_its_long_deadline(os_workspace):
    os_workspace.test_timeout_s = 30
    beat = os_workspace.root / ".process" / "running-test.beat"
    install_test(
        os_workspace,
        f"""
import time
from pathlib import Path

def test_wait_until_cancelled():
    p = Path({str(beat)!r})
    while True:
        p.write_text(str(time.monotonic()))
        time.sleep(0.05)
""",
    )
    task = asyncio.create_task(os_workspace.run_tests(["tests/test_visible.py"]))
    for _ in range(100):
        if beat.exists():
            break
        await asyncio.sleep(0.02)
    try:
        assert beat.exists(), "isolated test process did not reach the cancellation probe"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await _heartbeat_stopped(beat), "cancellation left the test process running"
        assert os_workspace.test_runs == []
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_bare_interpreter_uses_explicit_parent_pytest_packages_without_pythonpath(os_workspace, monkeypatch):
    """Reproduce the uv-overlay failure with a real dependency-free interpreter."""
    import subprocess

    bare = str(await asyncio.to_thread(Path(sys.executable).resolve))
    clean = {"PATH": os.defpath, "HOME": str(os_workspace.root / ".process")}
    discovery = await asyncio.to_thread(
        subprocess.run,
        [bare, "-I", "-c", "import importlib.util; print(importlib.util.find_spec('pytest') is None)"],
        env=clean,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    if discovery.stdout.strip() != "True":
        pytest.skip("base interpreter already supplies pytest; overlay dependency gap cannot be reproduced")
    monkeypatch.setattr(sys, "executable", bare)
    monkeypatch.setenv("PYTHONPATH", "SYNTHETIC_UNAUTHORIZED_IMPORT_PATH")
    result = await os_workspace.run_tests(["tests/test_visible.py"])
    assert result["return_code"] == 0, result["output"]
    assert "1 passed" in result["output"]


def test_runtime_package_access_does_not_authorize_project_root_or_parent_pythonpath(workspace, tmp_path, monkeypatch):
    unauthorized = tmp_path / "not-a-package-root"
    unauthorized.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(unauthorized))
    profile = workspace._profile()
    for entry in workspace._runtime_sites:
        assert entry.is_dir()
        assert f"(subpath {json.dumps(str(entry))})" in profile
    assert f"(subpath {json.dumps(str(unauthorized))})" not in profile
    assert f"(subpath {json.dumps(str(workspace.root.parent))})" not in profile
