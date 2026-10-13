"""Published provider endpoint defaults; every client also accepts a custom URL."""

from enum import StrEnum


class ApiEndpoint(StrEnum):
    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com"
    DEEPSEEK = "https://api.deepseek.com/v1"
    OPENROUTER = "https://openrouter.ai/api/v1"


class ApiPath(StrEnum):
    RESPONSES = "/responses"
    ANTHROPIC_MESSAGES = "/v1/messages"
    CHAT_COMPLETIONS = "/chat/completions"


__all__ = ["ApiEndpoint", "ApiPath"]
