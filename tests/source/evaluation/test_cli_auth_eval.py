from __future__ import annotations

import base64
import json
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentkit.core.contracts.types import Mode
from tests.evals import cli_auth
from tests.evals.cli_auth import access_token_expiry, force_expired_access_token


def _jwt(claims: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.signature"


def test_codex_refresh_probe_expires_only_the_access_token_in_its_copy() -> None:
    token = _jwt({"exp": 1_900_000_000, "sub": "not-recorded"})
    document = {"tokens": {"access_token": token, "refresh_token": "untouched"}}
    expired = force_expired_access_token(Mode.CODEX, document)
    assert access_token_expiry(Mode.CODEX, document) == 1_900_000_000
    assert access_token_expiry(Mode.CODEX, expired) == 0
    assert expired["tokens"]["refresh_token"] == "untouched"
    assert document["tokens"]["access_token"] == token


def test_claude_refresh_probe_expires_supported_metadata_in_its_copy() -> None:
    document = {"claudeAiOauth": {"expiresAt": 1_900_000_000_000, "refreshToken": "untouched"}}
    expired = force_expired_access_token(Mode.CLAUDE_CODE, document)
    assert access_token_expiry(Mode.CLAUDE_CODE, document) == 1_900_000_000
    assert access_token_expiry(Mode.CLAUDE_CODE, expired) == 0
    assert expired["claudeAiOauth"]["refreshToken"] == "untouched"
    assert document["claudeAiOauth"]["expiresAt"] == 1_900_000_000_000


def test_forced_refresh_refuses_unknown_token_expiry_shapes() -> None:
    with pytest.raises(ValueError, match="expiry field"):
        force_expired_access_token(Mode.CLAUDE_CODE, {"credential": "opaque"})


@pytest.mark.asyncio
async def test_codex_login_check_accepts_custom_auth_file_when_default_home_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "clean-home"))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("AGENTKIT_SKIP_REAL_CLI", "1")
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({"tokens": {"access_token": _jwt({"exp": int(time.time()) + 86_400}), "refresh_token": "opaque"}})
    )

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeRuntime:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def ask(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                outcome="answered",
                text="AUTH_OK",
                failure=None,
                attempts=1,
                usage_complete=True,
                cost_usd=Decimal("0"),
                cost_source="reported",
            )

    monkeypatch.setattr(cli_auth, "CallRuntime", FakeRuntime)
    report, passed = await cli_auth.run_login_check(
        Mode.CODEX,
        auth_file,
        report_dir=tmp_path / "reports",
    )

    assert passed
    assert report.is_file()
