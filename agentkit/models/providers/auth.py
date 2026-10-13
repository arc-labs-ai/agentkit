"""Explicit API-key versus bearer authentication, independent of transports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class AuthKind(StrEnum):
    API_KEY = "api_key"
    BEARER = "bearer"


ANTHROPIC_API_VERSION = "2023-06-01"


@dataclass(frozen=True)
class AnthropicCredential:
    secret: str = field(repr=False)
    kind: AuthKind = AuthKind.API_KEY

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", AuthKind(self.kind))
        if not self.secret:
            raise ValueError("credential secret must be nonempty")

    def headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {self.secret}"} if self.kind == AuthKind.BEARER else {"x-api-key": self.secret}
        )

    def cli_env(self) -> dict[str, str]:
        return (
            {"ANTHROPIC_AUTH_TOKEN": self.secret}
            if self.kind == AuthKind.BEARER
            else {"ANTHROPIC_API_KEY": self.secret}
        )


def anthropic_credential(values: Mapping[str, str]) -> AnthropicCredential | None:
    """Prefer explicit bearer credentials; recognize OAuth tokens in legacy key slots.

    Resolution is repeated per call so a host credential resolver can refresh
    short-lived OAuth tokens. No tokens are stored in sessions or reports.
    """
    token = values.get("ANTHROPIC_AUTH_TOKEN")
    if token:
        return AnthropicCredential(token, AuthKind.BEARER)
    key = values.get("ANTHROPIC_API_KEY")
    if key:
        return AnthropicCredential(key, AuthKind.BEARER if key.startswith("sk-ant-oat") else AuthKind.API_KEY)
    return None


__all__ = ["AuthKind", "ANTHROPIC_API_VERSION", "AnthropicCredential", "anthropic_credential"]
