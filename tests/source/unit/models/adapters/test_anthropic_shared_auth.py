"""Legacy and unified transports share authentication without orchestration coupling."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from agentkit.agents.prompts.composition import inline_prompt
from agentkit.core.contracts.types import Message, Mode
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef, ModelSettings
from agentkit.models.providers.auth import ANTHROPIC_API_VERSION, AnthropicCredential, AuthKind, anthropic_credential
from agentkit.models.providers.frontiers import AnthropicLLM
from agentkit.models.providers.frontiers.calls import ApiAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credentials,expected",
    [
        ({"ANTHROPIC_API_KEY": "sk-ant-api-fixture"}, {"x-api-key": "sk-ant-api-fixture"}),
        ({"ANTHROPIC_API_KEY": "sk-ant-oat-fixture"}, {"authorization": "Bearer sk-ant-oat-fixture"}),
        (
            {"ANTHROPIC_API_KEY": "ignored", "ANTHROPIC_AUTH_TOKEN": "opaque-bearer"},
            {"authorization": "Bearer opaque-bearer"},
        ),
    ],
)
async def test_chat_stream_and_unified_api_use_same_credential_policy(credentials, expected):
    observed = []

    def handler(request):
        observed.append(dict(request.headers))
        if json.loads(request.content).get("stream"):
            frames = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "answer"}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"},
            ]
            return httpx.Response(200, text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames))
        return httpx.Response(
            200,
            json={
                "model": "test",
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        legacy = AnthropicLLM(
            api_key=credentials.get("ANTHROPIC_API_KEY", ""),
            auth_token=credentials.get("ANTHROPIC_AUTH_TOKEN"),
            pricing=lambda _model, _usage: 0.0,  # Authentication fixtures have no billing.
            client=client,
        )
        assert (await legacy.chat(messages=[Message("user", "task")], model="test")).content == "answer"
        assert (
            "".join(
                delta.text
                for delta in [delta async for delta in legacy.stream(messages=[Message("user", "task")], model="test")]
            )
            == "answer"
        )
        adapter = ApiAdapter(client=client)
        result = await adapter.turn(
            Binding(Mode.ANTHROPIC_API, "test", CredentialRef(lambda: credentials)),
            inline_prompt("task"),
            settings=ModelSettings(),
            limits=CallLimits(max_output_tokens=64),
            schema=None,
            tools=None,
            session=None,
            cancel=None,
            emit=lambda _: None,
        )
        assert result.messages[-1] == "answer" and not result.failure
    assert len(observed) == 3
    for headers in observed:
        assert {key: headers[key] for key in ("authorization", "x-api-key") if key in headers} == expected
        assert headers["anthropic-version"] == ANTHROPIC_API_VERSION


def test_legacy_empty_credentials_and_explicit_custom_headers_remain_compatible():
    provider = AnthropicLLM(api_key="", version="2099-01-01", extra_headers={"x-fixture": "custom"})
    assert provider._headers()["x-api-key"] == ""
    assert provider._headers()["anthropic-version"] == "2099-01-01"
    assert provider._headers()["x-fixture"] == "custom"


def test_existing_import_paths_are_identity_preserving_and_picklable():
    from agentkit.models.contracts.constants import AuthKind as CompatibilityKind
    from agentkit.models.providers.auth import AnthropicCredential as CompatibilityCredential

    assert CompatibilityCredential is AnthropicCredential and CompatibilityKind is AuthKind
    credential = anthropic_credential({"ANTHROPIC_AUTH_TOKEN": "synthetic-secret"})
    assert credential == pickle.loads(pickle.dumps(credential))
    assert "synthetic-secret" not in repr(credential)


def test_shared_auth_import_requires_no_optional_dependencies_or_call_runtime():
    root = Path(__file__).resolve().parents[5]
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "from agentkit.models.providers.auth import AnthropicCredential,AuthKind;"
        "assert AnthropicCredential('fixture').kind is AuthKind.API_KEY;"
        "assert 'agentkit.calls' not in sys.modules;"
        "assert 'httpx' not in sys.modules;assert 'anthropic' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(root)], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_provider_import_does_not_load_call_orchestration():
    root = Path(__file__).resolve().parents[5]
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "from agentkit.models.providers.frontiers import AnthropicLLM;"
        "assert 'agentkit.calls' not in sys.modules;assert 'anthropic' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
