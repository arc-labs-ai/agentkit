"""Agentkit-owned Claude PreToolUse bridge; transport failures deny execution.

The standalone client uses only stdlib so isolated Python can run the hook.
Callback authority remains in the host process, never in a CLI settings file.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import shlex
import socket
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentkit.models.contracts.types import PreToolUse

_MAX_BYTES = 1_048_576
_CALLBACK_TIMEOUT_S = 5.0
_CLIENT_TIMEOUT_S = 8.0
_CLIENT_SCRIPT = str(Path(__file__).resolve())


def _decision(allowed: bool, reason: str = "") -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if allowed else "deny",
            "permissionDecisionReason": reason,
        }
    }


class NativeToolFence:
    """Per-call native policy endpoint, private configuration and hook command."""

    def __init__(self, callback: PreToolUse, emit: Any, *, hosted_tools: tuple[str, ...] = ()):
        self.callback, self.emit = callback, emit
        self.hosted_tools = frozenset(hosted_tools)
        self._token = secrets.token_urlsafe(32)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._server: asyncio.Server | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self.settings: dict[str, Any] = {}

    async def __aenter__(self) -> NativeToolFence:
        if not hasattr(socket, "AF_UNIX"):
            raise NotImplementedError("native tool fences require Unix-domain sockets")
        self._directory = tempfile.TemporaryDirectory(prefix="agentkit-native-", dir="/tmp")
        root = Path(self._directory.name)
        config = root / "client.json"
        endpoint = str(root / "policy.sock")
        try:
            self._server = await asyncio.start_unix_server(self._handle, endpoint, limit=_MAX_BYTES + 1)
            with os.fdopen(os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as target:
                json.dump({"socket": endpoint, "token": self._token}, target)
            client = shlex.join([sys.executable, "-I", _CLIENT_SCRIPT, "--client-config", str(config)])
            fallback = shlex.quote(json.dumps(_decision(False, "native tool policy unavailable")))
            # Claude treats many nonzero hook exits as nonblocking errors. A
            # missing interpreter/script therefore also emits an explicit deny.
            command = client + " || { printf '%s\\n' " + fallback + "; exit 0; }"
            self.settings = {
                "hooks": {
                    "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": command, "timeout": 10}]}]
                }
            }
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    def merged_settings(self, hooks: Any = None) -> dict[str, Any]:
        merged = dict(hooks or {})
        previous = merged.get("PreToolUse", ())
        if not isinstance(previous, (list, tuple)):
            raise ValueError("PreToolUse hooks must be a sequence")
        merged["PreToolUse"] = [*previous, *self.settings["hooks"]["PreToolUse"]]
        return {"hooks": merged}

    async def __aexit__(self, *exc: Any) -> None:
        """Tear the fence down, and finish doing so even if cancelled.

        ``client.json`` holds the shared secret the hook authenticates with, so
        a teardown that stops halfway leaves a live credential on disk next to a
        socket path that names it. Both awaits below used to sit in front of the
        directory removal, so cancelling a teardown with a decision in flight
        left the directory, the token file and the socket behind — measured, one
        leaked directory per cancelled attempt, cleaned up only whenever the
        ``TemporaryDirectory`` finalizer happened to run.

        The shape is the one ``integrations/mcp/serve.py`` already uses for the
        same hazard: the work runs as a task that cancellation cannot abandon,
        the caller's ``CancelledError`` is re-raised after it finishes, and the
        removal itself is in a ``finally`` so it happens even when the shutdown
        raises.
        """
        cleanup = asyncio.ensure_future(self._teardown())
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                interrupted = True
        cleanup.result()
        if interrupted:
            raise asyncio.CancelledError

    async def _teardown(self) -> None:
        """Stop serving, let any decision in flight finish, then remove the secret.

        The ordering is deliberate and is NOT a missed cancellation
        opportunity. Cancelling an in-flight handler would leave the hook with
        no answer to print, and Claude reads a decision-less hook as a
        nonblocking error — which means the tool RUNS. So an in-flight decision
        is allowed to complete, and it always completes: the callback deadline
        is shorter than the client's, which is shorter than the CLI's, and every
        unusable answer becomes a deny. Waiting out that deadline is the price
        of failing closed, and it is the right price.
        """
        try:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                self._server = None
            pending = tuple(self._tasks)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        from agentkit.core.serialization._immutable import deep_freeze
        from agentkit.models.contracts.constants import EventKind
        from agentkit.models.contracts.types import Allow, Deny
        from agentkit.models.providers._events import _emit

        current = asyncio.current_task()
        assert current is not None
        self._tasks.add(current)
        answer = _decision(False, "native tool policy unavailable")
        try:
            async with asyncio.timeout(_CLIENT_TIMEOUT_S):
                raw = await reader.readline()
                if len(raw) > _MAX_BYTES:
                    raise ValueError("hook request too large")
                request = json.loads(raw)
                if (
                    not isinstance(request, dict)
                    or not isinstance(request.get("token"), str)
                    or not hmac.compare_digest(request["token"], self._token)
                ):
                    raise ValueError("unauthorized hook request")
                payload = request.get("payload")
                if (
                    not isinstance(payload, dict)
                    or not isinstance(payload.get("tool_name"), str)
                    or not payload["tool_name"]
                    or not isinstance(payload.get("tool_input"), dict)
                ):
                    raise ValueError("invalid hook payload")
                name, arguments = payload["tool_name"], payload["tool_input"]
                if name in self.hosted_tools:
                    # These exact tools are backed by agentkit's own ToolSet
                    # and keep its grant/approval policy at the MCP server.
                    answer = _decision(True)
                else:
                    try:
                        async with asyncio.timeout(_CALLBACK_TIMEOUT_S):
                            result = await self.callback(name, deep_freeze(arguments))
                        if not isinstance(result, (Allow, Deny)):
                            raise TypeError("native callback must return Allow or Deny")
                        refused = isinstance(result, Deny)
                        reason = result.reason if isinstance(result, Deny) else ""
                    except Exception:
                        refused, reason = True, "native tool policy callback failed"
                    answer = _decision(not refused, reason)
                    await _emit(
                        self.emit,
                        EventKind.TOOL_CALL,
                        name=name,
                        tool_call_id=payload.get("tool_use_id"),
                        refused=refused,
                        native=True,
                        reason=reason,
                    )
            writer.write(json.dumps(answer).encode() + b"\n")
            await writer.drain()
        except Exception:
            writer.write(json.dumps(_decision(False, "native tool policy unavailable")).encode() + b"\n")
            try:
                await writer.drain()
            except (OSError, ConnectionError):
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass
            finally:
                self._tasks.discard(current)


def _client(config_path: str) -> dict[str, Any]:
    try:
        raw = sys.stdin.buffer.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError("hook payload too large")
        payload = json.loads(raw)
        config = json.loads(Path(config_path).read_text())
        request = json.dumps({"token": config["token"], "payload": payload}).encode() + b"\n"
        if len(request) > _MAX_BYTES:
            raise ValueError("hook request too large")
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(_CLIENT_TIMEOUT_S)
            connection.connect(config["socket"])
            connection.sendall(request)
            with connection.makefile("rb") as stream:
                response = json.loads(stream.readline(_MAX_BYTES + 1))
        if not isinstance(response, dict):
            raise ValueError("invalid hook response")
        specific = response.get("hookSpecificOutput", {})
        if specific.get("hookEventName") != "PreToolUse" or specific.get("permissionDecision") not in ("allow", "deny"):
            raise ValueError("invalid hook decision")
        return response
    except Exception:
        return _decision(False, "native tool policy unavailable")


if __name__ == "__main__":
    response = (
        _client(sys.argv[2])
        if len(sys.argv) == 3 and sys.argv[1] == "--client-config"
        else _decision(False, "native tool policy unavailable")
    )
    print(json.dumps(response))
