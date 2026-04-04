from __future__ import annotations

from schemas import ToolResult
from tools.tools import BaseTool, build_tool_output
from utils.timezone import isoformat


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = (
        "Return the current local date and time for the running environment. "
        "Use this tool when the answer depends on the current time."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, object]) -> ToolResult:
        current_time = isoformat()
        return ToolResult(
            output=build_tool_output(
                success=True,
                data={
                    "current_time": current_time,
                },
            ),
            success=True,
        )
