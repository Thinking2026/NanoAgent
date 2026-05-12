from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from schemas import (
    SHELL_COMMAND_FAILED,
    SHELL_EXECUTION_ERROR,
    SHELL_TIMEOUT,
    TOOL_ARGUMENT_ERROR,
    ToolResult,
    build_pipeline_error,
)
from tools.tool_base import BaseTool, build_tool_output
from utils.env_util.runtime_env import get_project_root, get_task_runtime_dir


class ShellTool(BaseTool):
    name = "shell"
    description = (
        "Run a shell command and return its output. "
        "Executes in the task workspace directory via the system shell (pipes, redirects, env vars supported). "
        "Non-zero exit code is treated as failure; stderr is used as the error message when available. "
        "Avoid long-running or interactive commands; use timeout to cap execution time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute in the task workspace directory.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds before killing the command. Defaults to 15.",
                "default": 15,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        command = str(arguments.get("command", "")).strip()
        if not command:
            error = build_pipeline_error(TOOL_ARGUMENT_ERROR, "Shell tool requires a non-empty command.")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        timeout = int(arguments.get("timeout", 15))
        working_directory = self._working_directory()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_directory,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            error = build_pipeline_error(SHELL_TIMEOUT, f"Shell command timed out after {timeout} seconds.")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )
        except Exception as exc:
            error = build_pipeline_error(SHELL_EXECUTION_ERROR, f"Shell command failed to start: {exc}")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        output = completed.stdout.strip()
        error_output = completed.stderr.strip()
        if completed.returncode != 0:
            message = error_output or output or f"Command exited with code {completed.returncode}"
            error = build_pipeline_error(SHELL_COMMAND_FAILED, message)
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
                    "cwd": str(working_directory),
                    "stdout": output,
                    "stderr": error_output,
                    "exit_code": completed.returncode,
                },
            ),
            success=True,
        )

    @staticmethod
    def _working_directory() -> Path:
        try:
            path = get_task_runtime_dir()
            path.mkdir(parents=True, exist_ok=True)
            return path
        except RuntimeError:
            return get_project_root()
