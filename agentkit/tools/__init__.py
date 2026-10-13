"""Tool Protocol stack — peer of ``agentkit.state.memory`` and ``agentkit.agents.skills``.

The ``Tool`` Protocol is the Command-pattern surface every action an agent can request must
satisfy. ``FunctionTool`` is the canonical impl wrapping a Python callable, with the ``@tool``
decorator as the ergonomic front door. ``ToolRegistry`` is the Composite holding many tools
behind a single lookup-by-name surface.
"""

from agentkit.tools.adapters.agent import as_tool, render_result
from agentkit.tools.declarations.base import Tool
from agentkit.tools.declarations.errors import ToolArgumentError, ToolDefinitionError, ToolShapeError
from agentkit.tools.declarations.function import FunctionTool, tool
from agentkit.tools.filesystem import FileTool, InMemoryFiles, StoreFiles
from agentkit.tools.registry.registry import ToolRegistry

__all__ = [
    "FileTool",
    "FunctionTool",
    "InMemoryFiles",
    "StoreFiles",
    "Tool",
    "ToolArgumentError",
    "ToolDefinitionError",
    "ToolRegistry",
    "ToolShapeError",
    "as_tool",
    "render_result",
    "tool",
]
