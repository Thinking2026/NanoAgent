from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from schemas import ToolCall, ToolResult


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def can_handle(self, tool_name: str) -> bool:
        return self.name == tool_name

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the provided text back to the agent."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to echo back."}
        },
        "required": ["text"],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        text = str(arguments.get("text", ""))
        return ToolResult(call_id="", output=text, success=True)


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "Return the current local time for the running environment."
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        current_time = datetime.now().isoformat(timespec="seconds")
        return ToolResult(call_id="", output=current_time, success=True)


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools = {tool.name: tool for tool in (tools or [])}
        self._router = ToolChainRouter(self._tools.values())

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        self._router = ToolChainRouter(self._tools.values())

    def auto_register(
        self,
        module_names: list[str] | None = None,
        package_name: str | None = None,
    ) -> None:
        for tool in discover_tools(module_names=module_names, package_name=package_name):
            self.register(tool)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any], call_id: str) -> ToolResult:
        return self._router.route(ToolCall(name=name, arguments=arguments, call_id=call_id))


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
            call_id=tool_call.call_id,
            output="",
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

    @abstractmethod
    def can_handle(self, tool_call: ToolCall) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process(self, tool_call: ToolCall) -> ToolResult:
        raise NotImplementedError


class ToolHandlerNode(BaseToolHandler):
    def __init__(self, tool: BaseTool) -> None:
        super().__init__()
        self._tool = tool

    def can_handle(self, tool_call: ToolCall) -> bool:
        return self._tool.can_handle(tool_call.name)

    def process(self, tool_call: ToolCall) -> ToolResult:
        result = self._tool.run(tool_call.arguments)
        result.call_id = tool_call.call_id
        return result


class FallbackToolHandler(BaseToolHandler):
    def can_handle(self, tool_call: ToolCall) -> bool:
        return True

    def process(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=tool_call.call_id,
            output="",
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )


class ToolChainRouter:
    def __init__(self, tools: list[BaseTool] | Any) -> None:
        self._root_handler = self._build_chain(list(tools))

    def route(self, tool_call: ToolCall) -> ToolResult:
        return self._root_handler.handle(tool_call)

    @staticmethod
    def _build_chain(tools: list[BaseTool]) -> BaseToolHandler:
        fallback = FallbackToolHandler()
        if not tools:
            return fallback

        handlers = [ToolHandlerNode(tool) for tool in tools]
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
    discovered_modules = [_safe_import(__name__)]

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
    package_name: str | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
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
