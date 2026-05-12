from __future__ import annotations

from typing import Any

from schemas import ToolResult
from tools.tool_base import BaseTool, build_tool_output
from utils.time.time import isoformat


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "Return the current local date and time. Use when the answer depends on the current time."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
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
