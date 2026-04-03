from __future__ import annotations

import os
from pathlib import Path

from schemas import ToolResult, build_error
from tools.tools import BaseTool, build_tool_output


class FileTool(BaseTool):
    name = "file"
    description = (
        "Read, write, or append a local file. "
        "Use this tool when you need to inspect file contents or modify a file in the workspace. "
        "The write and append actions change filesystem state."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The file operation to perform.",
                "enum": ["read", "write", "append"],
            },
            "path": {
                "type": "string",
                "description": "The local filesystem path to the target file.",
            },
            "content": {
                "type": "string",
                "description": "The text content to write or append. Required when action is write or append.",
            },
        },
        "required": ["action", "path"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, object]) -> ToolResult:
        action = str(arguments.get("action", "")).strip().lower()
        path_value = str(arguments.get("path", "")).strip()
        content = str(arguments.get("content", ""))

        if action not in {"read", "write", "append"}:
            error = build_error("TOOL_ARGUMENT_ERROR", "File tool action must be read, write, or append.")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )
        if not path_value:
            error = build_error("TOOL_ARGUMENT_ERROR", "File tool requires a non-empty path.")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        target_path = self._resolve_target_path(path_value)
        try:
            if action == "read":
                content = target_path.read_text(encoding="utf-8")
                return ToolResult(
                    output=build_tool_output(
                        success=True,
                        data={
                            "action": action,
                            "path": str(target_path),
                            "content": content,
                        },
                    ),
                    success=True,
                )

            target_path.parent.mkdir(parents=True, exist_ok=True)
            if action == "write":
                target_path.write_text(content, encoding="utf-8")
                return ToolResult(
                    output=build_tool_output(
                        success=True,
                        data={
                            "action": action,
                            "path": str(target_path),
                            "bytes_written": len(content.encode("utf-8")),
                        },
                    ),
                    success=True,
                )

            with target_path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(content)
            return ToolResult(
                output=build_tool_output(
                    success=True,
                    data={
                        "action": action,
                        "path": str(target_path),
                        "bytes_written": len(content.encode("utf-8")),
                    },
                ),
                success=True,
            )
        except Exception as exc:
            error = build_error("FILE_TOOL_ERROR", f"File tool failed: {exc}")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

    @staticmethod
    def _resolve_target_path(path_value: str) -> Path:
        target_path = Path(path_value).expanduser()
        if target_path.is_absolute():
            return target_path
        task_runtime_dir = os.environ.get("NANOAGENT_TASK_RUNTIME_DIR")
        if task_runtime_dir:
            return Path(task_runtime_dir).expanduser() / target_path
        return target_path
