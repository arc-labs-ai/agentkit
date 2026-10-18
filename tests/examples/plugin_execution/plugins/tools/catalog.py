"""Declarations are composed separately from their host implementations."""

from pydantic import BaseModel, ConfigDict

from agentkit.tools import FunctionTool, ToolRegistry


class RecordAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record_id: str
    status: str


def tool_catalog(handler) -> ToolRegistry:
    return ToolRegistry.from_tools(
        [
            FunctionTool.from_callable(handler, name="lookup_record", side_effecting=False, output_schema=RecordAnswer),
        ]
    )
