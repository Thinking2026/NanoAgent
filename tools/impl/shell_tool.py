from __future__ import annotations

import subprocess
from pathlib import Path

from schemas import ToolResult, build_error
from tools.tools import BaseTool, build_tool_output


class ShellTool(BaseTool):
    name = "shell"
    description = (
        "Run a shell command in the local environment and return the command output. "
        "Use this tool when information or file changes require command-line execution. "
        "The command may read files, modify files, or affect local system state."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "The maximum number of seconds to wait before stopping the command.",
                "default": 15,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, object]) -> ToolResult:
        command = str(arguments.get("command", "")).strip()
        if not command:
            error = build_error("TOOL_ARGUMENT_ERROR", "Shell tool requires a non-empty command.")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        timeout = int(arguments.get("timeout", 15))
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            error = build_error("SHELL_TIMEOUT", f"Shell command timed out after {timeout} seconds.")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )
        except Exception as exc:
            error = build_error("SHELL_EXECUTION_ERROR", f"Shell command failed to start: {exc}")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        output = completed.stdout.strip()
        error_output = completed.stderr.strip()
        if completed.returncode != 0:
            message = error_output or output or f"Command exited with code {completed.returncode}"
            error = build_error("SHELL_COMMAND_FAILED", message)
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        return ToolResult(
            output=build_tool_output(
                success=True,
                data={
                    "command": command,
                    "stdout": output,
                    "stderr": error_output,
                    "exit_code": completed.returncode,
                },
            ),
            success=True,
        )
