"""Real local processes, no provider network or credentials.

These test lifecycle and demonstrate subprocess limitations. They deliberately
make no claim that either provider's native tools are contained.
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from agentkit.models.providers.cli import CliInvocation, SubprocessExecutionBackend
from agentkit.models.providers.cli._common import CliTimeouts, _CliLaunch, _CliRunOutcome, _run_cli_process


@pytest.mark.asyncio
@pytest.mark.parametrize("script", ["print('{}')", "print('{}'); raise RuntimeError('planted crash')"])
async def test_backend_cleanup_is_called_after_success_and_crash(script):
    backend = SubprocessExecutionBackend()
    closed = []

    async def spawn(*argv, **kwargs):
        process = await backend.start(CliInvocation(tuple(argv), kwargs))
        original = process.aclose

        async def close():
            closed.append(True)
            await original()

        process.aclose = close
        return process

    async def handle(payload):
        if False:
            yield payload

    outcome = _CliRunOutcome()
    async for _ in _run_cli_process(
        prepare=lambda: _CliLaunch([sys.executable, "-c", script], env={}),
        handle=handle,
        spawn=spawn,
        semaphore=asyncio.BoundedSemaphore(1),
        timeouts=CliTimeouts(total=5),
        terminate_grace_s=0.1,
        ctx=None,
        outcome=outcome,
    ):
        pass
    assert closed == [True]
    assert outcome.return_code == (1 if "RuntimeError" in script else 0)


@pytest.mark.asyncio
async def test_subprocess_has_no_filesystem_or_credential_boundary(tmp_path):
    """An outside fake credential is readable: never claim this backend isolates."""
    outside = tmp_path / "other-job"
    outside.mkdir()
    credential = outside / "auth.json"
    credential.write_text("planted-not-a-credential")
    work = tmp_path / "job"
    work.mkdir()
    script = "from pathlib import Path; import sys; p=Path(sys.argv[1]); print(p.read_text()); p.write_text('modified')"
    process = await SubprocessExecutionBackend().start(
        CliInvocation(
            (sys.executable, "-c", script, str(credential)),
            {"env": {}, "cwd": str(work), "start_new_session": True, "stdout": asyncio.subprocess.PIPE},
        )
    )
    try:
        assert await process.stdout.read() == b"planted-not-a-credential\n"
        assert await process.wait() == 0
        assert credential.read_text() == "modified"
    finally:
        await process.aclose()


@pytest.mark.asyncio
async def test_cancellation_uses_backend_tree_termination_and_cleanup(tmp_path):
    backend = SubprocessExecutionBackend()
    called = []
    ready = asyncio.Event()
    cancelled = asyncio.Event()
    child_pid = None
    # Child reports liveness by holding a TCP port, avoiding zombie/PID races.
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()
    child = (
        "import socket,time; s=socket.socket(); s.bind(('127.0.0.1'," + str(port) + ")); "
        "s.listen(); print('ready',flush=True); time.sleep(300)"
    )
    script = (
        "import subprocess,sys,json,time; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdout=subprocess.PIPE); "
        "p.stdout.readline(); print(json.dumps({'pid':p.pid}),flush=True); time.sleep(300)"
    )

    async def spawn(*argv, **kwargs):
        process = await backend.start(CliInvocation(tuple(argv), kwargs))
        original_terminate, original_close = process.terminate_tree, process.aclose

        async def terminate(grace):
            called.append("terminate")
            await original_terminate(grace)

        async def close():
            called.append("close")
            await original_close()

        process.terminate_tree, process.aclose = terminate, close
        return process

    async def handle(payload):
        nonlocal child_pid
        child_pid = payload["pid"]
        ready.set()
        if False:
            yield payload

    def check():
        if cancelled.is_set():
            raise RuntimeError("cancelled")

    async def run():
        outcome = _CliRunOutcome()
        async for _ in _run_cli_process(
            prepare=lambda: _CliLaunch([sys.executable, "-c", script, child], env={}),
            handle=handle,
            spawn=spawn,
            semaphore=asyncio.BoundedSemaphore(1),
            timeouts=CliTimeouts(total=5),
            terminate_grace_s=0.1,
            ctx=SimpleNamespace(check_cancelled=check),
            outcome=outcome,
        ):
            pass
        return outcome

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(ready.wait(), 5)
        cancelled.set()
        outcome = await asyncio.wait_for(task, 5)
        assert outcome.cancelled
        assert "terminate" in called and "close" in called
        # The planted descendant released its listening socket.
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", port)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,name", [("claude_code", ".credentials.json"), ("codex", "auth.json")])
async def test_real_process_rotation_returns_fake_credentials_once_and_removes_file(mode, name, tmp_path):
    from agentkit.models.providers.cli.isolation import IsolatedHome

    changed = []

    async def persist(files):
        changed.append(dict(files))

    lifecycle = await IsolatedHome.credential_lifecycle(tmp_path, mode, {name: b"fake-old"}, persist)
    async with lifecycle:
        process = await SubprocessExecutionBackend().start(
            CliInvocation(
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'fake-new')",
                    str(tmp_path / name),
                ),
                {"env": {}, "start_new_session": True},
            )
        )
        try:
            assert await process.wait() == 0
        finally:
            await process.aclose()
    assert changed == [{name: b"fake-new"}]
    assert not (tmp_path / name).exists()


@pytest.mark.asyncio
async def test_subprocess_does_not_claim_or_enforce_private_network_denial():
    reached = asyncio.Event()

    async def accept(reader, writer):
        reached.set()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    backend = SubprocessExecutionBackend()
    process = await backend.start(
        CliInvocation(
            (
                sys.executable,
                "-c",
                "import socket,sys; socket.create_connection(('127.0.0.1',int(sys.argv[1]))).close()",
                str(port),
            ),
            {"env": {}, "start_new_session": True},
        )
    )
    try:
        assert await process.wait() == 0
        await asyncio.wait_for(reached.wait(), 1)
        assert not (await backend.inspect()).guarantees
    finally:
        await process.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_backend_cleanup_connection_loss_and_stall_are_bounded(monkeypatch):
    from agentkit.models.providers.cli import _common

    monkeypatch.setattr(_common, "_STDERR_DRAIN_S", 0.02)
    backend = SubprocessExecutionBackend()
    entered = asyncio.Event()

    async def spawn(*argv, **kwargs):
        process = await backend.start(CliInvocation(tuple(argv), kwargs))

        async def close():
            entered.set()
            await asyncio.Future()  # Planted supervisor that never answers cleanup.

        process.aclose = close
        return process

    async def handle(payload):
        if False:
            yield payload

    async def run():
        outcome = _CliRunOutcome()
        async for _ in _run_cli_process(
            prepare=lambda: _CliLaunch([sys.executable, "-c", "print('{}')"], env={}),
            handle=handle,
            spawn=spawn,
            semaphore=asyncio.BoundedSemaphore(1),
            timeouts=CliTimeouts(total=1),
            terminate_grace_s=0.1,
            ctx=None,
            outcome=outcome,
        ):
            pass
        return outcome

    outcome = await asyncio.wait_for(run(), 1)
    assert entered.is_set() and outcome.cleanup_failed
