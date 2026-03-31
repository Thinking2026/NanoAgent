from __future__ import annotations

import importlib
import inspect
import pkgutil
import time
from abc import ABC, abstractmethod
from typing import Any

from schemas import AgentError, ToolCall, ToolResult, build_error
from tools.tools import BaseTool


class ToolRegistry:
    def __init__(
        self,
        tools: list[BaseTool] | None = None,
        timeout_retry_max_attempts: int = 4,
        timeout_retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0),
    ) -> None:
        self._tools = {tool.name: tool for tool in (tools or [])}
        self._timeout_retry_max_attempts = timeout_retry_max_attempts
        self._timeout_retry_delays = self._normalize_retry_delays(
            timeout_retry_delays,
            timeout_retry_max_attempts,
        )
        self._router = ToolChainRouter(
            self._tools.values(),
            timeout_retry_max_attempts=self._timeout_retry_max_attempts,
            timeout_retry_delays=self._timeout_retry_delays,
        )

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        self._router = ToolChainRouter(
            self._tools.values(),
            timeout_retry_max_attempts=self._timeout_retry_max_attempts,
            timeout_retry_delays=self._timeout_retry_delays,
        )

    def auto_register(
        self,
        module_names: list[str] | None = None,
        package_name: str | None = None,
    ) -> None:
        for tool in discover_tools(module_names=module_names, package_name=package_name):
            self.register(tool)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        llm_raw_tool_call_id: str | None = None,
    ) -> ToolResult:
        return self._router.route(
            ToolCall(
                name=name,
                arguments=arguments,
                llm_raw_tool_call_id=llm_raw_tool_call_id,
            )
        )

    @staticmethod
    def _normalize_retry_delays(
        retry_delays: tuple[float, ...],
        max_attempts: int,
    ) -> tuple[float, ...]:
        target_length = max(0, max_attempts - 1)
        if target_length == 0:
            return ()
        delays = [delay for delay in retry_delays if delay > 0]
        if not delays:
            delays = [1.0]
        while len(delays) < target_length:
            delays.append(delays[-1] * 2)
        return tuple(delays[:target_length])


class BaseToolHandler(ABC):
    def __init__(self) -> None:
        self._next_handler: BaseToolHandler | None = None

    def set_next(self, handler: BaseToolHandler) -> BaseToolHandler:
        self._next_handler = handler
        return handler

    def handle(self, tool_call: ToolCall) -> ToolResult:
        if self.can_handle(tool_call):
            return self.process(tool_call)
        if self._next_handler is not None:
            return self._next_handler.handle(tool_call)
        return ToolResult(
            output="",
            llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
            success=False,
            error=build_error("TOOL_NOT_FOUND", f"Unknown tool: {tool_call.name}"),
        )

    @abstractmethod
    def can_handle(self, tool_call: ToolCall) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process(self, tool_call: ToolCall) -> ToolResult:
        raise NotImplementedError


class ToolHandlerNode(BaseToolHandler):
    def __init__(
        self,
        tool: BaseTool,
        timeout_retry_max_attempts: int,
        timeout_retry_delays: tuple[float, ...],
    ) -> None:
        super().__init__()
        self._tool = tool
        self._timeout_retry_max_attempts = timeout_retry_max_attempts
        self._timeout_retry_delays = timeout_retry_delays

    def can_handle(self, tool_call: ToolCall) -> bool:
        return self._tool.can_handle(tool_call.name)

    def process(self, tool_call: ToolCall) -> ToolResult:
        total_attempts = self._timeout_retry_max_attempts
        for attempt_idx in range(total_attempts):
            try:
                result = self._tool.run(tool_call.arguments)
                result.llm_raw_tool_call_id = tool_call.llm_raw_tool_call_id
                return result
            except TimeoutError as exc:
                if attempt_idx < total_attempts - 1:
                    time.sleep(self._timeout_retry_delays[attempt_idx])
                    continue
                return ToolResult(
                    output="",
                    llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
                    success=False,
                    error=build_error(
                        "TOOL_TIMEOUT",
                        (
                            f"Tool `{tool_call.name}` timed out after {total_attempts} attempts: {exc}"
                        ),
                    ),
                )
            except AgentError as exc:
                if "TIMEOUT" in exc.code and attempt_idx < total_attempts - 1:
                    time.sleep(self._timeout_retry_delays[attempt_idx])
                    continue
                return ToolResult(
                    output="",
                    llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
                    success=False,
                    error=exc,
                )
            except Exception as exc:
                return ToolResult(
                    output="",
                    llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
                    success=False,
                    error=build_error(
                        "TOOL_EXECUTION_ERROR",
                        f"Tool `{tool_call.name}` failed unexpectedly: {exc}",
                    ),
                )
        return ToolResult(
            output="",
            llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
            success=False,
            error=build_error("TOOL_TIMEOUT", f"Tool `{tool_call.name}` timed out."),
        )


class FallbackToolHandler(BaseToolHandler):
    def can_handle(self, tool_call: ToolCall) -> bool:
        return True

    def process(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            output="",
            llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
            success=False,
            error=build_error("TOOL_NOT_FOUND", f"Unknown tool: {tool_call.name}"),
        )


class ToolChainRouter:
    def __init__(
        self,
        tools: list[BaseTool] | Any,
        timeout_retry_max_attempts: int,
        timeout_retry_delays: tuple[float, ...],
    ) -> None:
        self._root_handler = self._build_chain(
            list(tools),
            timeout_retry_max_attempts=timeout_retry_max_attempts,
            timeout_retry_delays=timeout_retry_delays,
        )

    def route(self, tool_call: ToolCall) -> ToolResult:
        return self._root_handler.handle(tool_call)

    @staticmethod
    def _build_chain(
        tools: list[BaseTool],
        timeout_retry_max_attempts: int,
        timeout_retry_delays: tuple[float, ...],
    ) -> BaseToolHandler:
        fallback = FallbackToolHandler()
        if not tools:
            return fallback

        handlers = [
            ToolHandlerNode(
                tool,
                timeout_retry_max_attempts=timeout_retry_max_attempts,
                timeout_retry_delays=timeout_retry_delays,
            )
            for tool in tools
        ]
        current = handlers[0]
        root = current
        for handler in handlers[1:]:
            current = current.set_next(handler)
        current.set_next(fallback)
        return root


def discover_tools(
    module_names: list[str] | None = None,
    package_name: str | None = None,
) -> list[BaseTool]:
    discovered_modules = []

    for module_name in module_names or []:
        module = _safe_import(module_name)
        if module is not None:
            discovered_modules.append(module)

    if package_name:
        discovered_modules.extend(_discover_package_modules(package_name))

    tools: dict[str, BaseTool] = {}
    for module in discovered_modules:
        if module is None:
            continue
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if not issubclass(candidate, BaseTool) or candidate is BaseTool:
                continue
            if inspect.isabstract(candidate):
                continue
            try:
                tool = candidate()
            except TypeError:
                continue
            tools[tool.name] = tool
    return list(tools.values())


def create_default_tool_registry(
    module_names: list[str] | None = None,
    package_name: str | None = "tools.impl",
    timeout_retry_max_attempts: int = 4,
    timeout_retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> ToolRegistry:
    registry = ToolRegistry(
        timeout_retry_max_attempts=timeout_retry_max_attempts,
        timeout_retry_delays=timeout_retry_delays,
    )
    registry.auto_register(module_names=module_names, package_name=package_name)
    return registry


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _discover_package_modules(package_name: str) -> list[Any]:
    package = _safe_import(package_name)
    if package is None:
        return []

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return [package]

    modules = [package]
    for module_info in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        module = _safe_import(module_info.name)
        if module is not None:
            modules.append(module)
    return modules
