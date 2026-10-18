"""Does the minter produce a credential the CLI accepts, and refuse one it mishandles?

The design question this closes is in ``docs/design/credential-isolation.md``:
a credential installed into a job home is readable by the job's own native
tools, because the reader and the tools are the same user in the same process
tree. Permissions cannot change that. The mechanism that survives it is to
install a token that is worthless outside the job, and this is the piece that
formats such a token into each CLI's native login shape.

Two kinds of test here, and the second kind is the reason the module exists:

- the mint is ACCEPTED where it has to be — it installs through the real
  chokepoint, under the real allowlist, and the real secret appears nowhere;
- the mint is REFUSED wherever a wrong one would fail SILENTLY. Codex fails
  OPEN on a malformed ``id_token`` (measured: no ``Authorization`` header at
  all rather than an error), and a caller handing the real login through would
  install exactly what minting exists to keep out while looking like it worked.
  Both of those are green-looking wrong answers, so each has a test that the
  refusal is what happens.

Synchronous throughout: ``ASYNC230``/``ASYNC240`` forbid ``open()`` and
``pathlib`` inside an async body, and these are filesystem assertions.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.models.providers.cli.isolation import IsolatedHome
from agentkit.models.providers.cli.minting import (
    CODEX_AUTH_CLAIM,
    LOGIN_FILENAMES,
    MAX_SCOPE_SECONDS,
    JobScope,
    MintingError,
    mint_credential,
    mint_home,
    verify_minted_home,
)

REAL_SECRET = b'{"claudeAiOauth":{"accessToken":"synthetic-probe-value-never-a-real-token"}}'
ACCOUNT = "00000000-0000-4000-8000-000000000000"


def _scope(**overrides: object) -> JobScope:
    fields: dict[str, object] = {
        "job_id": "job-7",
        "model": "claude-opus-5",
        "expires_at": time.time() + 600,
        "account_id": ACCOUNT,
    }
    fields.update(overrides)
    return JobScope(**fields)  # type: ignore[arg-type]


def _home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    return home


def _claims(id_token: str) -> dict:
    segment = id_token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))


@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
def test_a_mint_installs_through_the_real_chokepoint_for_both_modes(tmp_path, mode) -> None:
    """The mint has to satisfy ``install_credentials``' own allowlist.

    This is the load-bearing cross-check of the whole step: the minter is only
    useful if its output is exactly what the one function that writes
    credential material into a job home will accept. A mint with the wrong
    filename would raise "credential file is outside the mode allowlist" at
    dispatch time, in production, and only for the mode nobody minted for
    locally.
    """
    home = _home(tmp_path)
    files = mint_home(mode, token="job-scoped-opaque-token", scope=_scope())
    installed = IsolatedHome.install_credentials(home, mode.value, files)

    assert set(installed) == {LOGIN_FILENAMES[mode]}
    target = home / LOGIN_FILENAMES[mode]
    assert target.stat().st_mode & 0o777 == 0o600
    assert b"job-scoped-opaque-token" in target.read_bytes()


@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
def test_the_minted_filename_matches_the_install_allowlist_exactly(mode) -> None:
    """A mode added to one table and not the other is a dispatch-time failure.

    ``LOGIN_FILENAMES`` and the allowlist inside ``install_credentials`` are two
    statements of the same fact, kept in two places because one belongs to the
    minter and one to the installer. Asserted by CONSTRUCTION rather than by
    reading the literal: an empty mint for a mode is accepted, and a mint whose
    name is outside the allowlist raises.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        home = Path(raw) / "home"
        home.mkdir(mode=0o700)
        IsolatedHome.install_credentials(home, mode.value, {LOGIN_FILENAMES[mode]: b"{}"})

    with tempfile.TemporaryDirectory() as raw:
        other = Path(raw) / "home"
        other.mkdir(mode=0o700)
        with pytest.raises(ValueError, match="outside the mode allowlist"):
            IsolatedHome.install_credentials(other, mode.value, {"not-the-login-file.json": b"{}"})


def test_a_mint_never_carries_the_real_credential_into_the_job(tmp_path) -> None:
    """The property the whole mechanism rests on, asserted on the bytes on disk.

    Not "the minter was called correctly" — the real secret's bytes appear
    NOWHERE under the job home. That is the only form of this assertion that
    stays true if the blob shape changes.
    """
    home = _home(tmp_path)
    files = mint_home(Mode.CLAUDE_CODE, token="job-scoped-opaque-token", scope=_scope())
    IsolatedHome.install_credentials(home, "claude_code", files)

    present = [p for p in home.rglob("*") if p.is_file() and REAL_SECRET in p.read_bytes()]
    assert present == []
    assert b"synthetic-probe-value-never-a-real-token" not in (home / ".credentials.json").read_bytes()


def test_handing_the_real_login_to_the_minter_is_refused_not_formatted() -> None:
    """The mistake that would look like success.

    A caller who wires the operator's saved session straight into ``token``
    installs exactly what minting exists to keep out, and every test of the
    happy path still passes: a credential is installed, at 0600, under the
    allowlist, and the run works — because it is the real credential. So the
    minter refuses anything carrying the markers of a real login rather than
    formatting it.
    """
    for leaked in (
        REAL_SECRET.decode(),
        '{"claudeAiOauth":{"accessToken":"x"}}',
        "sk-ant-api03-not-a-job-token",
        "sk-proj-not-a-job-token",
    ):
        with pytest.raises(MintingError, match="real saved-session credential"):
            mint_home(Mode.CLAUDE_CODE, token=leaked, scope=_scope())


def test_a_codex_mint_carries_the_three_segment_jwt_codex_requires() -> None:
    """Codex fails OPEN on a malformed ``id_token``, so the mint must not be sloppy.

    Measured against codex 0.162.1/0.163.0: a non-JWT ``tokens.id_token`` makes
    the CLI issue its request with no ``Authorization`` header at all rather
    than refusing, while ``access_token`` may be any opaque string. A job that
    authenticates as nobody surfaces as a confusing vendor error, not as a
    credential problem, which is why the shape is asserted here rather than
    left to the broker to discover.
    """
    files = mint_home(Mode.CODEX, token="job-scoped-opaque-token", scope=_scope())
    blob = json.loads(files["auth.json"])
    id_token = blob["tokens"]["id_token"]

    segments = id_token.split(".")
    assert len(segments) == 3 and all(segments), "codex treats a short token as malformed and sends nothing"
    assert _claims(id_token)[CODEX_AUTH_CLAIM]["chatgpt_account_id"] == ACCOUNT
    assert blob["tokens"]["access_token"] == "job-scoped-opaque-token"
    assert blob["OPENAI_API_KEY"] is None, "a minted session must not look like an API-key login"


@pytest.mark.parametrize(
    "id_token",
    ["not-a-jwt-at-all", "only.two", "has..empty", ""],
    ids=["no-dots", "two-segments", "empty-middle", "empty"],
)
def test_verify_refuses_every_id_token_shape_codex_fails_open_on(id_token) -> None:
    """The refusal is checked against hand-built blobs, not only the minter's own.

    ``mint_home`` verifies its own output, which protects against a change to
    the shape it writes. It does not protect a host that mints elsewhere, so
    the check is exposed and tested independently — and tested on the negative
    cases, or it could pass vacuously.
    """
    blob = {
        "OPENAI_API_KEY": None,
        "tokens": {"id_token": id_token, "access_token": "job-scoped-opaque-token", "account_id": ACCOUNT},
    }
    with pytest.raises(MintingError, match="three-segment JWT"):
        verify_minted_home(Mode.CODEX, {"auth.json": json.dumps(blob).encode()})


@pytest.mark.parametrize(
    "claims",
    [
        {"job_id": "job-7"},
        {CODEX_AUTH_CLAIM: {}},
        {CODEX_AUTH_CLAIM: {"chatgpt_account_id": ""}},
        {CODEX_AUTH_CLAIM: {"chatgpt_account_id": None}},
        {CODEX_AUTH_CLAIM: "not-an-object"},
    ],
    ids=["claim-absent", "claim-empty", "account-blank", "account-null", "claim-not-an-object"],
)
def test_verify_refuses_a_well_formed_jwt_that_names_no_account(claims) -> None:
    """Three segments is necessary and not sufficient.

    Codex reads the account out of the ``https://api.openai.com/auth`` claim. A
    structurally valid JWT without a usable one passes a shape check and still
    leaves the CLI with nobody to authenticate as.

    Parametrized over PRESENT-but-useless as well as absent, because the two
    take different branches: an absent claim fails the ``isinstance`` test, and
    a claim carrying a blank or null account only fails the emptiness test. A
    single absent case would leave the emptiness half of the check unenforced
    while looking like it covered the requirement.
    """
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    blob = {
        "tokens": {"id_token": f"{header}.{body}.unsigned", "access_token": "t", "account_id": ACCOUNT},
    }
    with pytest.raises(MintingError, match="chatgpt_account_id"):
        verify_minted_home(Mode.CODEX, {"auth.json": json.dumps(blob).encode()})


def test_codex_without_an_account_in_scope_is_refused_rather_than_minted_blank() -> None:
    """A ``None`` account would mint a JWT that passes a shape check and names nobody."""
    with pytest.raises(MintingError, match="account id"):
        mint_home(Mode.CODEX, token="job-scoped-opaque-token", scope=_scope(account_id=None))


@pytest.mark.parametrize(
    ("offset", "message"),
    [(-1, "already past"), (MAX_SCOPE_SECONDS + 60, "ceiling")],
    ids=["expired", "beyond-ceiling"],
)
def test_a_deadline_outside_the_admitted_window_is_refused(offset, message) -> None:
    """The deadline is part of the mechanism, not metadata.

    A minted token's worth is bounded by scope and lifetime; an unbounded
    deadline gives a native tool that reads it the same window the operator's
    own login has, which is the situation minting exists to end. So an expired
    deadline and one past the ceiling are both refused at mint time rather than
    left for the broker to notice.

    The offset is resolved against the clock HERE rather than in the parameter
    list. Captured at collection time it drifts by however long the rest of the
    suite takes to reach this test: a deadline set 60s past the ceiling stops
    being past the ceiling after 60s of other tests, so the beyond-ceiling case
    passed alone and failed in a full run. A relative offset cannot rot that
    way.
    """
    with pytest.raises(MintingError, match=message):
        mint_home(Mode.CLAUDE_CODE, token="job-scoped-opaque-token", scope=_scope(expires_at=time.time() + offset))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"job_id": "  "}, "job id"), ({"model": ""}, "model"), ({"expires_at": "soon"}, "epoch deadline")],
    ids=["blank-job", "blank-model", "non-numeric-deadline"],
)
def test_a_scope_missing_any_of_its_three_bounds_is_refused(overrides, message) -> None:
    """All three bounds are required because the broker needs all three.

    "Is this request inside what the job was admitted to do" cannot be answered
    without the job, the model and the deadline, and a scope that omits one
    produces a token whose only bound is the operator's account.
    """
    with pytest.raises(MintingError, match=message):
        _scope(**overrides)


def test_the_minted_token_carries_its_scope_back_to_the_broker() -> None:
    """The broker reads the scope out of the credential rather than being told it.

    Minting and enforcing are different processes, so the scope has to travel
    with the token. Claude carries the deadline in its native ``expiresAt``
    (milliseconds, which is the unit that file uses); Codex carries the whole
    scope in the id_token claims.
    """
    scope = _scope()
    claude = json.loads(mint_home(Mode.CLAUDE_CODE, token="t-1", scope=scope)[".credentials.json"])
    assert claude["claudeAiOauth"]["expiresAt"] == int(scope.expires_at * 1000)

    codex = json.loads(mint_home(Mode.CODEX, token="t-1", scope=scope)["auth.json"])
    claims = _claims(codex["tokens"]["id_token"])
    assert claims["job_id"] == "job-7"
    assert claims["model"] == "claude-opus-5"
    assert claims["exp"] == int(scope.expires_at)


def test_a_minted_credential_attaches_no_refresh_callback() -> None:
    """The job provably cannot refresh, so there is no rotated credential to persist.

    ``credential_lifecycle`` calls ``on_changed`` when the installed file's
    bytes differ at teardown. For a minted token that would hand the host a
    job-scoped artifact to write back over the operator's real login. Refresh
    belongs to the broker, which is step 3 of the design's sequencing, so the
    minted credential deliberately carries no callback.
    """
    credential = mint_credential(Mode.CLAUDE_CODE, token="job-scoped-opaque-token", scope=_scope())
    assert credential.on_changed is None
    assert credential.env == {}
    assert set(credential.home) == {".credentials.json"}


@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.ANTHROPIC_API])
def test_minting_is_refused_for_a_mode_that_reads_no_login_file(mode) -> None:
    """The API modes authenticate from the environment, so a home mint is meaningless.

    Refused rather than returning an empty mapping: an empty mint would install
    nothing and read as "credentials handled", which is the kind of silent
    success this module is built to avoid.
    """
    with pytest.raises(MintingError, match="does not read a login file"):
        mint_home(mode, token="job-scoped-opaque-token", scope=_scope())


def test_verify_refuses_a_mint_with_more_files_than_the_one_login(tmp_path) -> None:
    """Extra files in a mint are a smuggling channel, not a convenience."""
    files = {".credentials.json": b'{"claudeAiOauth":{"accessToken":"t"}}', "settings.json": b"{}"}
    with pytest.raises(MintingError, match="exactly .credentials.json"):
        verify_minted_home(Mode.CLAUDE_CODE, files)


@pytest.mark.parametrize(
    ("token", "message"),
    [
        ("", "non-empty token"),
        ("   ", "non-empty token"),
        ("job-token\nsecond-line", "single line"),
        ("job-token\rsecond-line", "single line"),
    ],
    ids=["empty", "blank", "newline", "carriage-return"],
)
def test_a_token_that_is_not_one_opaque_line_is_refused(token, message) -> None:
    """A multi-line token is a file-shape hazard, not a cosmetic complaint.

    The token is interpolated into a JSON login file the CLI parses; an
    embedded newline is the shape a smuggled second field takes. Refused at the
    mint because that is the last point where the value is still attributable
    to the caller that supplied it.
    """
    with pytest.raises(MintingError, match=message):
        mint_home(Mode.CLAUDE_CODE, token=token, scope=_scope())


def test_a_scope_that_is_not_a_job_scope_is_refused() -> None:
    """A mapping that merely looks like a scope carries no validation.

    ``JobScope.__post_init__`` is where the three bounds are enforced, so
    accepting a duck-typed stand-in would let an unbounded scope through the
    one check that exists for it.
    """
    with pytest.raises(MintingError, match="requires a JobScope"):
        mint_home(Mode.CLAUDE_CODE, token="job-scoped-opaque-token", scope={"job_id": "job-7"})  # type: ignore[arg-type]


@pytest.mark.parametrize("account_id", ["", "   "], ids=["empty", "blank"])
def test_a_blank_account_id_in_scope_is_refused_rather_than_carried(account_id) -> None:
    """``None`` means "not applicable"; blank means a bug upstream.

    The two are different and only one is legitimate, so a blank string is
    refused instead of being treated as absent — otherwise it would reach the
    Codex id_token as a present-but-useless claim.
    """
    with pytest.raises(MintingError, match="non-empty string"):
        _scope(account_id=account_id)


@pytest.mark.parametrize(
    ("mode", "files", "message"),
    [
        (Mode.CLAUDE_CODE, {".credentials.json": b"not json at all"}, "not JSON"),
        (Mode.CLAUDE_CODE, {".credentials.json": b'["a", "list"]'}, "not an object"),
        (Mode.CLAUDE_CODE, {".credentials.json": b"{}"}, "no access token"),
        (Mode.CLAUDE_CODE, {".credentials.json": b'{"claudeAiOauth": {}}'}, "no access token"),
        (Mode.CLAUDE_CODE, {".credentials.json": b'{"claudeAiOauth": "nope"}'}, "no access token"),
        (Mode.CODEX, {"auth.json": b"{}"}, "no tokens object"),
        (Mode.CODEX, {"auth.json": b'{"tokens": "nope"}'}, "no tokens object"),
        (Mode.CODEX, {"auth.json": b'{"tokens": {"id_token": "a.b.c"}}'}, "no access token"),
        (Mode.CODEX, {"auth.json": b'{"tokens": {"id_token": 7, "access_token": "t"}}'}, "must be a string"),
    ],
    ids=[
        "claude-not-json", "claude-not-object", "claude-no-oauth", "claude-empty-oauth",
        "claude-oauth-not-object", "codex-no-tokens", "codex-tokens-not-object",
        "codex-no-access-token", "codex-id-token-not-a-string",
    ],
)
def test_verify_refuses_every_malformed_login_shape(mode, files, message) -> None:
    """The fail-closed branches, exercised rather than assumed.

    ``verify_minted_home`` exists so a mint that would leave the CLI
    unauthenticated raises instead of installing. Every branch of it is a
    refusal, so an unexercised branch is an unproven refusal — and this module
    is worth nothing except for its refusals.
    """
    with pytest.raises(MintingError, match=message):
        verify_minted_home(mode, files)


def test_verify_refuses_a_login_file_for_a_mode_that_reads_none() -> None:
    """The same refusal as ``mint_home``, reached through the checking entry point."""
    with pytest.raises(MintingError, match="does not read a login file"):
        verify_minted_home(Mode.OPENAI_API, {".credentials.json": b"{}"})


def test_verify_refuses_an_id_token_whose_payload_is_not_an_object() -> None:
    """A JWT body that decodes to a scalar has no claims to read.

    Three non-empty segments is a shape test, and base64 of ``"hello"`` passes
    it. The claim lookup would then be attempted on a string, so the payload
    type is checked before it is read.
    """
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    body = base64.urlsafe_b64encode(b'"just-a-string"').decode().rstrip("=")
    blob = {"tokens": {"id_token": f"{header}.{body}.unsigned", "access_token": "t"}}
    with pytest.raises(MintingError, match="payload is not an object"):
        verify_minted_home(Mode.CODEX, {"auth.json": json.dumps(blob).encode()})
