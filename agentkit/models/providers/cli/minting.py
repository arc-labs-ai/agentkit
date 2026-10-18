"""Job-scoped credential minting: the step that keeps the real login out of the job.

WHY THIS MODULE EXISTS
----------------------
``docs/design/credential-isolation.md`` settles the question containment was
blocked on. A credential installed into a job home at 0600 is NOT contained:
the CLI must read it, and the CLI's own ``Read``/``Bash`` tools run as the same
user in the same process tree, so they inherit exactly the access the reader
has. ``tests/source/conformance/test_credential_isolation.py`` pins that with a
child process reading the installed file verbatim. Permissions cannot close
that channel, because the reader and the tools are the same principal.

The only mechanism that survives it is to make what the job holds worthless
outside the job. That is what this module builds: given an opaque token the
HOST minted for one job, it produces the native login-file shape each CLI
accepts, so ``IsolatedHome.install_credentials`` — the single chokepoint that
writes credential material into a job home — installs a job-scoped token and
the operator's real saved session never enters the home at all.

WHAT THIS DOES AND DOES NOT ESTABLISH
-------------------------------------
It does NOT contain anything, and nothing here should be read as a containment
claim:

- A native tool in the job can still read the minted token. That is accepted by
  design; the point is what the token is worth. Forwarded to the vendor
  unchanged it returns ``401`` at zero cost, so a tool that reads it gains
  exactly the authority the BROKER grants it.
- The blast radius therefore moves from "the operator's whole provider account"
  to "whatever this job was admitted to do", and the second is only as narrow
  as the scope below. That is why ``JobScope`` is required rather than
  optional: scoping per job, per model and per deadline is part of the
  mechanism, not a later hardening step.
- The job still reaches the broker, so a tool can spend the job's own quota
  within its admitted scope. That residual is bounded, not closed.

This module does not mint the token's VALUE and holds no secret. It formats a
token the caller already has, and refuses shapes the CLIs mishandle. Issuing,
scoping and revoking the token is the broker's job (step 2 of the design's
sequencing); this is step 1, which needs no CLI change and no new mechanism.

WHY IT FAILS CLOSED ON CODEX
----------------------------
Measured against codex 0.162.1/0.163.0: with ``tokens.id_token`` set to a
non-JWT string the CLI issues its request with **no ``Authorization`` header at
all** rather than refusing, while ``tokens.access_token`` itself may be any
opaque value. A minter that got the ``id_token`` shape wrong would therefore
produce jobs that silently authenticate as nobody, surfacing as a confusing
vendor error rather than as a credential problem. So every mint is verified
against that observed requirement before it is returned, and a mint that would
fail open raises instead.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.types import Credential

#: The claim namespace Codex reads the account out of, and the one the broker
#: reads the scope back out of. Measured, not documented by the vendor.
CODEX_AUTH_CLAIM: Final = "https://api.openai.com/auth"

#: A mint is refused beyond this horizon. A deadline is part of what makes the
#: token worth little outside the job, so an unbounded one defeats the purpose;
#: this is deliberately shorter than any plausible saved-session lifetime.
MAX_SCOPE_SECONDS: Final = 12 * 60 * 60

#: The login filename each mode's binary reads, matching the allowlist in
#: ``IsolatedHome.install_credentials``. Kept here so a new mode cannot be
#: minted for without also being admitted there.
LOGIN_FILENAMES: Final[Mapping[Mode, str]] = {
    Mode.CLAUDE_CODE: ".credentials.json",
    Mode.CODEX: "auth.json",
}

#: Substrings that mean the caller handed us a real saved-session blob instead
#: of a minted token. Passing the real credential through would install exactly
#: what this module exists to keep out, and it would look like it worked, so it
#: is refused rather than formatted.
_REAL_CREDENTIAL_MARKERS: Final = (
    "claudeAiOauth", "refresh_token", "refreshToken", "OPENAI_API_KEY", "sk-ant-", "sk-proj-",
)


class MintingError(ValueError):
    """A mint was refused because it would be unsafe or silently ineffective."""


@dataclass(frozen=True)
class JobScope:
    """What a minted token is admitted to do, and until when.

    Every field is required. The broker needs all three to answer "is this
    request inside what the job was admitted to do", and a mint that omits any
    of them produces a token whose only bound is the operator's own account —
    which is the situation this module exists to end.
    """

    job_id: str
    model: str
    expires_at: float
    account_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id.strip():
            raise MintingError("a minted credential requires a job id")
        if not isinstance(self.model, str) or not self.model.strip():
            raise MintingError("a minted credential requires the model it is admitted for")
        if not isinstance(self.expires_at, (int, float)) or isinstance(self.expires_at, bool):
            raise MintingError("a minted credential requires a numeric epoch deadline")
        if self.account_id is not None and (not isinstance(self.account_id, str) or not self.account_id.strip()):
            raise MintingError("account id must be a non-empty string when given")

    def remaining_seconds(self, *, now: float | None = None) -> float:
        return self.expires_at - (time.time() if now is None else now)

    def claims(self) -> dict[str, object]:
        """The scope as the broker reads it back out of a minted token."""
        return {"job_id": self.job_id, "model": self.model, "exp": int(self.expires_at)}


def _check_token(token: str) -> None:
    if not isinstance(token, str) or not token.strip():
        raise MintingError("a minted credential requires a non-empty token")
    if any(marker in token for marker in _REAL_CREDENTIAL_MARKERS) or token.lstrip().startswith("{"):
        # The caller passed the real login through. Formatting it would install
        # the operator's credential into the job and look like success.
        raise MintingError(
            "token looks like a real saved-session credential rather than a job-scoped mint; "
            "mint a job token in the parent and pass that, never the login itself"
        )
    if "\n" in token or "\r" in token:
        raise MintingError("a minted token must be a single line")


def _check_scope(scope: JobScope, *, now: float | None = None) -> None:
    if not isinstance(scope, JobScope):
        raise MintingError("a minted credential requires a JobScope")
    remaining = scope.remaining_seconds(now=now)
    if remaining <= 0:
        raise MintingError("minted credential deadline is already past")
    if remaining > MAX_SCOPE_SECONDS:
        raise MintingError(f"minted credential deadline exceeds the {MAX_SCOPE_SECONDS}s ceiling")


def _b64url(payload: Mapping[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")


def _decode_segment(segment: str) -> dict[str, object]:
    padded = segment + "=" * (-len(segment) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded))
    if not isinstance(decoded, dict):
        raise MintingError("minted id_token payload is not an object")
    return decoded


def _codex_id_token(scope: JobScope) -> str:
    """A JWT of the shape Codex was OBSERVED to require before it sends a Bearer.

    Unsigned (``alg: none``) on purpose: nothing downstream verifies it, the
    broker authenticates the bearer token instead, and minting a signature here
    would imply a trust relationship that does not exist. The claims carry the
    scope so the broker can read back what the job was admitted to do.
    """
    header = _b64url({"alg": "none", "typ": "JWT"})
    body = _b64url(
        {
            CODEX_AUTH_CLAIM: {
                "chatgpt_account_id": scope.account_id,
                "chatgpt_plan_type": "pro",
            },
            **scope.claims(),
        }
    )
    # A third segment must be present and non-empty: Codex splits on "." and
    # treats a two-segment token as malformed, which is the fail-open case.
    return f"{header}.{body}.unsigned"


def mint_home(mode: Mode, *, token: str, scope: JobScope, now: float | None = None) -> dict[str, bytes]:
    """The native login files for ``mode``, carrying ``token`` and nothing else.

    The return value is what ``IsolatedHome.install_credentials`` accepts, so a
    host wires minting by substituting this for the real login at that one
    chokepoint. Raises :class:`MintingError` rather than returning a blob the
    CLI would mishandle.
    """
    mode = Mode(mode)
    if mode not in LOGIN_FILENAMES:
        raise MintingError(f"{mode.value} does not read a login file from the job home")
    _check_token(token)
    _check_scope(scope, now=now)

    blob: dict[str, object]
    if mode == Mode.CLAUDE_CODE:
        # Claude accepts an opaque access token; the deadline is milliseconds
        # here because that is the unit the native file uses.
        blob = {
            "claudeAiOauth": {
                "accessToken": token,
                "expiresAt": int(scope.expires_at * 1000),
                "scopes": ["user:inference"],
                "subscriptionType": "max",
            }
        }
    else:
        if scope.account_id is None:
            raise MintingError("codex requires an account id in the job scope to mint a usable id_token")
        blob = {
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": _codex_id_token(scope),
                "access_token": token,
                "account_id": scope.account_id,
            },
            "last_refresh": None,
        }

    files = {LOGIN_FILENAMES[mode]: json.dumps(blob).encode()}
    # Verify our OWN output before handing it back, so a change to the shape
    # above cannot ship a mint that fails open.
    verify_minted_home(mode, files)
    return files


def mint_credential(mode: Mode, *, token: str, scope: JobScope, now: float | None = None) -> Credential:
    """``mint_home`` wrapped as the :class:`Credential` a ``CredentialRef`` returns.

    No ``on_changed`` callback is attached: the job cannot refresh a minted
    token — by design and by observation — so there is no rotated credential to
    persist. Refresh belongs to the broker, which owns the real credential.
    """
    return Credential(home=mint_home(mode, token=token, scope=scope, now=now))


def verify_minted_home(mode: Mode, files: Mapping[str, bytes]) -> None:
    """Refuse a login blob the CLI would accept while sending no credential.

    Separate from ``mint_home`` so a host that mints elsewhere can run the same
    check, and so the check is testable against hand-built blobs.
    """
    mode = Mode(mode)
    expected = LOGIN_FILENAMES.get(mode)
    if expected is None:
        raise MintingError(f"{mode.value} does not read a login file from the job home")
    if set(files) != {expected}:
        raise MintingError(f"{mode.value} mint must contain exactly {expected}")
    try:
        blob = json.loads(files[expected])
    except (ValueError, TypeError) as exc:
        raise MintingError("minted login file is not JSON") from exc
    if not isinstance(blob, dict):
        raise MintingError("minted login file is not an object")

    if mode == Mode.CLAUDE_CODE:
        oauth = blob.get("claudeAiOauth")
        if not isinstance(oauth, dict) or not oauth.get("accessToken"):
            raise MintingError("minted claude login carries no access token")
        return

    tokens = blob.get("tokens")
    if not isinstance(tokens, dict):
        raise MintingError("minted codex login has no tokens object")
    if not tokens.get("access_token"):
        raise MintingError("minted codex login carries no access token")
    id_token = tokens.get("id_token")
    if not isinstance(id_token, str):
        raise MintingError("minted codex id_token must be a string")
    segments = id_token.split(".")
    if len(segments) != 3 or not all(segments):
        # The fail-open case: codex sends NO Authorization header for this.
        raise MintingError(
            "minted codex id_token is not a three-segment JWT; codex fails OPEN on this "
            "and would authenticate as nobody rather than refusing"
        )
    auth = _decode_segment(segments[1]).get(CODEX_AUTH_CLAIM)
    if not isinstance(auth, dict) or not auth.get("chatgpt_account_id"):
        raise MintingError(f"minted codex id_token carries no {CODEX_AUTH_CLAIM}.chatgpt_account_id claim")
