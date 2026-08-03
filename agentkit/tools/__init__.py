"""Tool Protocol stack — peer of ``agentkit.memory`` and ``agentkit.skills``.

The ``Tool`` Protocol is the Command-pattern surface every action an agent can request must
satisfy. ``FunctionTool`` is the canonical impl wrapping a Python callable, with the ``@tool``
decorator as the ergonomic front door. ``ToolRegistry`` is the Composite holding many tools
behind a single lookup-by-name surface.
"""

from agentkit.tools.base import Tool
from agentkit.tools.errors import ToolDefinitionError, ToolShapeError
from agentkit.tools.file_tool import FileTool, InMemoryFiles
from agentkit.tools.from_agent import as_tool, render_result
from agentkit.tools.function import FunctionTool, tool
from agentkit.tools.registry import ToolRegistry

__all__ = [
    "FileTool",
    "FunctionTool",
    "InMemoryFiles",
    "Tool",
    "ToolDefinitionError",
    "ToolRegistry",
    "ToolShapeError",
    "as_tool",
    "render_result",
    "tool",
]
