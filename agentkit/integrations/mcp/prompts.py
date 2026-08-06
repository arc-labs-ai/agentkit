"""Adapt an MCP server's prompts into agentkit ``Prompt`` objects.

MCP prompts come in two flavors:

- **Static** — no arguments, or every argument is optional. These
  materialize to a fully-rendered ``Prompt`` here (we call
  ``get_prompt(name)`` at adaptation time and store the resulting text
  as the template).
- **Argumented** — one or more REQUIRED arguments. These cannot be
  reduced to a static template without knowing the args, so v1 drops
  them and logs a warning. Callers who need argumented prompts should
  reach for ``MCPClient.get_prompt(name, args)`` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentkit.prompts.prompt import Prompt

if TYPE_CHECKING:
    from mcp import types as mcp_types

    from agentkit.integrations.mcp.client import MCPClient


_LOGGER = logging.getLogger(__name__)


def _requires_args(mcp_prompt: mcp_types.Prompt) -> bool:
    """A prompt needs args if any declared argument is ``required=True``."""
    for arg in mcp_prompt.arguments or []:
        if arg.required:
            return True
    return False


async def mcp_prompts(client: MCPClient) -> dict[str, Prompt]:
    """Return the MCP server's prompts as a ``{name: Prompt}`` dict.

    Static prompts are rendered eagerly by calling ``get_prompt(name)``;
    the rendered text becomes ``Prompt.template``. Argumented prompts
    (any ``required=True`` argument) are dropped with a warning — v1
    can't materialize a template we don't have the args for.
    """
    prompts_out: dict[str, Prompt] = {}
    for mcp_prompt in await client.list_prompts():
        if _requires_args(mcp_prompt):
            _LOGGER.warning(
                "mcp_prompts: dropping %r — MCP prompt requires arguments, "
                "cannot render statically. Use MCPClient.get_prompt(name, args) directly.",
                mcp_prompt.name,
            )
            continue
        try:
            rendered = await client.get_prompt(mcp_prompt.name)
        except Exception:  # noqa: BLE001 — a broken server prompt must not sink the batch
            _LOGGER.warning(
                "mcp_prompts: failed to render %r — skipping.",
                mcp_prompt.name,
                exc_info=True,
            )
            continue
        input_names = tuple(arg.name for arg in mcp_prompt.arguments or [])
        prompts_out[mcp_prompt.name] = Prompt(
            id=mcp_prompt.name,
            version="mcp:1",
            template=rendered,
            inputs=input_names,
        )
    return prompts_out


__all__ = ["mcp_prompts"]
