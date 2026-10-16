"""Opt-in live checks for Claude Code and Codex CLI login files.

Run with ``python -m tests.evals.cli_auth``. The selected auth file is copied
into agentkit's isolated CLI home; it is never copied to a report. Forced
refresh rewrites only the in-memory copy and requires an explicit opt-in because
OAuth refresh-token rotation can invalidate the source session.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.constants import FailureClass, UnpricedPolicy
from agentkit.models.contracts.types import Binding, CallLimits, Credential, CredentialRef
from agentkit.models.providers.cli.calls import CliAdapter
from tests.evals.engine.config import DEFAULT_MODELS
from tests.evals.engine.environment import runtime_versions
from tests.evals.engine.reporting import source_digest, write_report

_AUTH_FILES = {
    Mode.CLAUDE_CODE: ".credentials.json",
    Mode.CODEX: "auth.json",
}
_DEFAULT_LOGIN_MODELS = {
    Mode.CLAUDE_CODE: DEFAULT_MODELS[Mode.CLAUDE_CODE],
    # Codex CLI's ChatGPT login catalog is distinct from OpenAI API model
    # availability; use a current CLI account model for this auth-only probe.
    Mode.CODEX: "gpt-6.1-sol",
}


def _resolve_auth_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=True)


def _decode_jwt_payload(token: str) -> tuple[dict[str, Any], str] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    return (payload, parts[2]) if isinstance(payload, dict) else None


def _set_jwt_expired(token: str) -> str | None:
    parts = token.split(".")
    decoded = _decode_jwt_payload(token)
    if len(parts) != 3 or decoded is None:
        return None
    payload, signature = decoded
    payload["exp"] = 0
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    return f"{parts[0]}.{encoded}.{signature}"


def force_expired_access_token(mode: Mode, document: Any) -> Any:
    """Expire supported access-token metadata in a disposable JSON copy."""
    changed = False

    def visit(value: Any, key: str = "") -> Any:
        nonlocal changed
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for child_key, child in value.items():
                folded = child_key.casefold().replace("_", "")
                if mode == Mode.CLAUDE_CODE and folded in {"expiresat", "expiry", "expires"}:
                    out[child_key] = 0
                    changed = True
                elif mode == Mode.CODEX and child_key == "access_token" and isinstance(child, str):
                    expired = _set_jwt_expired(child)
                    if expired is not None:
                        out[child_key] = expired
                        changed = True
                    else:
                        out[child_key] = child
                else:
                    out[child_key] = visit(child, child_key)
            return out
        if isinstance(value, list):
            return [visit(child, key) for child in value]
        return value

    result = visit(document)
    if not changed:
        raise ValueError("auth file has no supported access-token expiry field to force")
    return result


def access_token_expiry(mode: Mode, document: Any) -> float | None:
    """Read only expiry metadata; never return or log token content."""
    expiries: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                folded = key.casefold().replace("_", "")
                if mode == Mode.CLAUDE_CODE and folded in {"expiresat", "expiry", "expires"}:
                    if isinstance(child, (int, float)) and not isinstance(child, bool):
                        expiry = float(child)
                        expiries.append(expiry / 1000 if expiry > 10_000_000_000 else expiry)
                elif mode == Mode.CODEX and key == "access_token" and isinstance(child, str):
                    claims = _decode_jwt_payload(child)
                    if claims and isinstance(claims[0].get("exp"), (int, float)):
                        expiries.append(float(claims[0]["exp"]))
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return min(expiries) if expiries else None


def _default_auth_path(mode: Mode) -> Path:
    if mode == Mode.CLAUDE_CODE:
        return Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / _AUTH_FILES[mode]
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / _AUTH_FILES[mode]


async def run_login_check(
    mode: Mode,
    auth_file: Path | None,
    *,
    keychain_probe: bool = False,
    force_refresh: bool = False,
    allow_refresh_rotation: bool = False,
    report_dir: Path,
) -> tuple[Path, bool]:
    if mode not in _AUTH_FILES:
        raise ValueError("CLI login checks support Claude Code and Codex only")
    if keychain_probe and (mode != Mode.CLAUDE_CODE or auth_file is not None):
        raise ValueError("the keychain probe is Claude-only and does not accept --auth-file")
    if not keychain_probe and auth_file is None:
        raise ValueError("--auth-file is required unless --keychain-probe is selected")
    if force_refresh and not allow_refresh_rotation:
        raise ValueError("--force-refresh requires --allow-refresh-token-rotation")

    filename = _AUTH_FILES[mode]
    if keychain_probe:
        auth_bytes = b"{}"
        auth_source = "macos_keychain_probe"
        force_expired = False
    else:
        assert auth_file is not None
        auth_path = await asyncio.to_thread(_resolve_auth_path, auth_file)
        if not await asyncio.to_thread(auth_path.is_file):
            raise ValueError("auth file must be a regular file")
        default_auth_path = await asyncio.to_thread(lambda: _default_auth_path(mode).expanduser().resolve())
        if force_refresh and auth_path == default_auth_path:
            raise ValueError("forced refresh cannot use the default login file; supply a disposable login copy")
        auth_bytes = await asyncio.to_thread(auth_path.read_bytes)
        auth_source = "isolated_login_file"
        force_expired = force_refresh

    try:
        document = json.loads(auth_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("auth file must contain valid JSON") from exc
    if force_expired:
        document = force_expired_access_token(mode, document)
        auth_bytes = json.dumps(document, separators=(",", ":")).encode()
    else:
        expires = access_token_expiry(mode, document)
        if not keychain_probe and (expires is None or expires - time.time() < 3600):
            raise ValueError("auth file has unknown or near-term expiry; use a disposable login for forced refresh")

    changed_files: set[str] = set()

    async def on_changed(files: Any) -> None:
        changed_files.update(name for name, value in files.items() if isinstance(value, bytes))

    credential = CredentialRef(lambda: Credential(home={filename: auth_bytes}, on_changed=on_changed))
    model = os.environ.get(f"AGENTKIT_EVAL_MODEL_{mode.upper()}", _DEFAULT_LOGIN_MODELS[mode])
    events: list[Any] = []
    async with CallRuntime(
        adapters={mode: CliAdapter()},
        pricing=PriceTable.default(),
        unpriced_policy=UnpricedPolicy.PROCEED_UNPRICED,
    ) as runtime:
        answer = await runtime.ask(
            Binding(mode, model, credential),
            "Reply with exactly AUTH_OK.",
            limits=CallLimits(max_turns=1, max_output_tokens=512, wall_clock_s=90),
            events=events.append,
            call_id=f"cli-login/{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}",
        )
    passed = (
        answer.outcome == "answered"
        and answer.text.strip() == "AUTH_OK"
        and (not force_expired or filename in changed_files)
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = report_dir / run_id / "summary.json"
    failure_class = answer.failure.classification if answer.failure else None
    report = {
        "run_id": run_id,
        "source_digest": source_digest(),
        "runtime_versions": await runtime_versions((mode,)),
        "mode": mode,
        "model": model,
        "auth_source": auth_source,
        "forced_expiry": force_expired,
        "changed_files": sorted(changed_files),
        "outcome": answer.outcome,
        "failure_class": failure_class if failure_class != FailureClass.UNKNOWN else "unknown",
        "attempts": answer.attempts,
        "usage_complete": answer.usage_complete,
        "cost_usd": str(answer.cost_usd) if answer.cost_usd is not None else None,
        "cost_source": answer.cost_source,
        "passed": passed,
    }
    write_report(target, report, ())
    return target, passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated real CLI login and refresh evaluation")
    parser.add_argument("--mode", required=True, type=Mode, choices=(Mode.CLAUDE_CODE, Mode.CODEX))
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--keychain-probe", action="store_true", help="probe macOS Claude Keychain-backed login")
    parser.add_argument("--force-refresh", action="store_true", help="expire only the isolated access-token copy")
    parser.add_argument(
        "--allow-refresh-token-rotation",
        action="store_true",
        help="confirm the supplied login is disposable; forced refresh may rotate its refresh token",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("tests/evals/reports/cli-auth"))
    args = parser.parse_args()
    path, passed = asyncio.run(
        run_login_check(
            args.mode,
            args.auth_file,
            keychain_probe=args.keychain_probe,
            force_refresh=args.force_refresh,
            allow_refresh_rotation=args.allow_refresh_token_rotation,
            report_dir=args.report_dir,
        )
    )
    print(f"Report: {path}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
