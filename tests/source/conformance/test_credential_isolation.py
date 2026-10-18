"""Reachability facts behind ``docs/design/credential-isolation.md``.

``test_isolation.py`` already covers that a credential is installed at 0600,
only under a mode allowlist, and always removed. None of that answers the
question containment is blocked on: once installed, WHO can read it. These
tests encode the reachability answers, which are the load-bearing ones, and
they are the cheap offline half of the evidence in that document.

Every test here is deliberately synchronous. ``ASYNC230``/``ASYNC240`` forbid
``open()`` and ``pathlib``/``os.path`` inside an async body, and all of these
assertions are filesystem assertions, so the two that need a coroutine drive it
with ``asyncio.run`` instead of being written async.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agentkit.models.providers.cli._common import _build_child_env
from agentkit.models.providers.cli.isolation import IsolatedHome

REAL_SECRET = b'{"claudeAiOauth":{"accessToken":"synthetic-probe-value-never-a-real-token"}}'


def _contains(root: Path, needle: bytes) -> list[str]:
    """Every file under ``root`` whose bytes contain ``needle``."""
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            if needle in path.read_bytes():
                hits.append(str(path.relative_to(root)))
    return hits


def test_an_installed_credential_is_readable_by_any_child_process_of_the_job(tmp_path):
    """0600 is not a boundary against the CLI's own native tools.

    This is the whole reason credential isolation is a hard problem. The CLI
    must read this file, its ``Read``/``Bash`` tools run as the same user in
    the same process tree, and so they inherit exactly the access the reader
    has. A child process standing in for a native tool proves it: the mode
    keeps OTHER users out and does nothing about the job itself. Any claim that
    installing at 0600 contains the credential has to fail this test.
    """
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    IsolatedHome.install_credentials(home, "claude_code", {".credentials.json": REAL_SECRET})
    target = home / ".credentials.json"
    assert target.stat().st_mode & 0o777 == 0o600

    # The stand-in for a native tool: an ordinary child process, same uid, no
    # special privilege, reading the path by name.
    child = subprocess.run(
        [sys.executable, "-I", "-c", "import sys;sys.stdout.buffer.write(open(sys.argv[1],'rb').read())", str(target)],
        capture_output=True,
        check=False,
    )
    assert child.returncode == 0
    assert child.stdout == REAL_SECRET, "a same-user child reads the installed credential verbatim"


def test_the_isolated_env_policy_passes_the_host_home_through_unchanged():
    """``env_policy="isolated"`` does not hide the host's own credentials.

    Worth pinning because the policy's NAME invites the opposite reading. The
    host ``HOME`` is on the passthrough list, so under this policy alone a job
    still sees ``~/.claude.json``, ``~/.codex/auth.json`` and (on macOS) the
    login keychain, which lives under ``$HOME/Library/Keychains``. What
    actually redirects the home is the explicit ``HOME`` override the CLI
    adapter layers on top. If that override is ever dropped, this test keeps
    saying where the protection really came from.
    """
    env, removed = _build_child_env(policy="isolated", credential_vars=("ANTHROPIC_API_KEY",))
    assert "HOME" in env and env["HOME"] == os.environ["HOME"]
    assert removed == ()

    # And the override is what makes the home private, not the policy.
    overridden, _ = _build_child_env(
        policy="isolated",
        credential_vars=("ANTHROPIC_API_KEY",),
        overrides={"HOME": "/nonexistent/owned-home"},
    )
    assert overridden["HOME"] == "/nonexistent/owned-home"


def test_substituting_the_credential_at_install_keeps_the_real_one_out_of_the_job(tmp_path):
    """``install_credentials`` is the single chokepoint a broker would use.

    Mechanism (c) in the design document depends on this property: exactly one
    function writes credential material into a job home, so replacing what it
    writes with a short-lived job-scoped token means the genuine credential
    never enters the home at all. The assertion is the one that matters for
    that claim — the real secret's bytes appear NOWHERE under the job root.
    """
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    broker_token = b'{"claudeAiOauth":{"accessToken":"job-scoped-broker-token"}}'
    IsolatedHome.install_credentials(home, "claude_code", {".credentials.json": broker_token})

    assert _contains(home, b"job-scoped-broker-token") == [".credentials.json"]
    assert _contains(home, b"synthetic-probe-value-never-a-real-token") == []


def test_dropping_the_credential_after_startup_leaves_no_credential_bytes_behind(tmp_path):
    """Mount-then-drop is only sound if the drop is actually complete.

    Both CLIs were observed reading their credential once at startup and
    serving later turns from memory, which makes dropping it mid-session
    viable. That is only worth anything if nothing else under the home still
    holds a copy, so this pins teardown emptying the home of those bytes
    rather than merely unlinking one known name. Driven with ``asyncio.run``
    because the lifecycle is a coroutine while every assertion is a path one.
    """
    home = tmp_path / "home"
    home.mkdir(mode=0o700)

    async def drive() -> None:
        lifecycle = await IsolatedHome.credential_lifecycle(home, "codex", {"auth.json": REAL_SECRET})
        async with lifecycle:
            pass

    assert _contains(home, REAL_SECRET) == []
    asyncio.run(drive())
    assert _contains(home, REAL_SECRET) == [], "teardown must leave no credential bytes under the job home"
    assert not (home / "auth.json").exists()


def test_a_minted_codex_credential_needs_a_three_segment_jwt_id_token(tmp_path):
    """Codex fails OPEN on a malformed ``id_token``, so a minter must not.

    Measured against codex 0.162.1/0.163.0: with ``tokens.id_token`` set to a
    non-JWT string the CLI issued its request with **no ``Authorization``
    header at all** rather than refusing, while ``tokens.access_token`` itself
    may be any opaque value. A broker that minted a credential with a sloppy
    ``id_token`` would therefore produce jobs that silently authenticate as
    nobody, which surfaces as a confusing vendor error rather than as a
    credential problem.

    This encodes the shape requirement so whoever writes the real minter has a
    red test to write it against, and asserts the negative case too so the
    requirement cannot be satisfied vacuously.
    """

    def mint(id_token: str) -> bytes:
        return json.dumps(
            {
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": id_token,
                    "access_token": "job-scoped-broker-token",
                    "refresh_token": "job-scoped-broker-refresh",
                    "account_id": "00000000-0000-4000-8000-000000000000",
                },
            }
        ).encode()

    def acceptable(blob: bytes) -> bool:
        """The shape Codex was observed to require before it sends a Bearer."""
        tokens = json.loads(blob)["tokens"]
        segments = tokens["id_token"].split(".")
        if len(segments) != 3 or not all(segments):
            return False
        padded = segments[1] + "=" * (-len(segments[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        auth = claims.get("https://api.openai.com/auth") or {}
        return bool(auth.get("chatgpt_account_id")) and bool(tokens["access_token"])

    good = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
        + "."
        + base64.urlsafe_b64encode(
            json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct", "chatgpt_plan_type": "pro"}}).encode()
        )
        .decode()
        .rstrip("=")
        + ".sig"
    )
    assert acceptable(mint(good))
    assert not acceptable(mint("not-a-jwt-at-all"))
    assert not acceptable(mint("only.two"))

    # And the shape agentkit would actually install round-trips through the
    # single chokepoint without the real credential coming near it.
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    IsolatedHome.install_credentials(home, "codex", {"auth.json": mint(good)})
    assert acceptable((home / "auth.json").read_bytes())
    assert _contains(home, b"synthetic-probe-value-never-a-real-token") == []


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="sandbox-exec read-deny is a macOS-specific enforcement primitive",
)
def test_sandbox_exec_denies_one_credential_path_to_a_child_and_its_grandchildren(tmp_path):
    """An enforcement point outside the CLI process does exist on macOS.

    Mechanism (a) needs a way to make one path unreachable to the job without
    rewriting the job, and needs that denial to survive the shell the native
    ``Bash`` tool would spawn. Both hold here. What this test deliberately does
    NOT show is the CLI authenticating under such a profile: the CLI reads the
    same path its tools would, so a static deny on the credential blocks the
    reader too. That gap is why the design document records (a) as infeasible
    as a static policy and viable only as a post-startup drop.
    """
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    IsolatedHome.install_credentials(home, "codex", {"auth.json": REAL_SECRET})
    # The sandbox profile matches on the resolved path; on macOS ``tmp_path``
    # lives under a /var -> /private/var symlink, so a literal of the
    # unresolved path would never match and the test would pass vacuously.
    target = Path(os.path.realpath(home / "auth.json"))
    profile = tmp_path / "deny.sb"
    profile.write_text(f'(version 1)\n(allow default)\n(deny file-read* (literal "{target}"))\n')

    unsandboxed = subprocess.run(["/bin/cat", str(target)], capture_output=True, check=False)
    assert unsandboxed.returncode == 0 and unsandboxed.stdout == REAL_SECRET

    direct = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/cat", str(target)],
        capture_output=True,
        check=False,
    )
    assert direct.returncode != 0 and REAL_SECRET not in direct.stdout

    # A native shell tool is a grandchild, so the denial has to be inherited.
    grandchild = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/sh", "-c", f"cat {target}"],
        capture_output=True,
        check=False,
    )
    assert grandchild.returncode != 0 and REAL_SECRET not in grandchild.stdout

    # The sandbox is a read-deny on one path, not a broken process.
    working = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/echo", "ok"],
        capture_output=True,
        check=False,
    )
    assert working.returncode == 0 and working.stdout.strip() == b"ok"
