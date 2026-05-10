from __future__ import annotations

import importlib
import inspect
import pkgutil
import time
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Optional

from schemas import (
    TOOL_EXECUTION_ERROR,
    TOOL_NOT_FOUND,
    TOOL_TIMEOUT,
    PipelineError,
    ToolCall,
    ToolResult,
    build_pipeline_error,
)
from infra.observability.tracing import Span, Tracer
from tools.tool_base import BaseTool
from utils.log.log import Logger, zap


class ToolRegistry:
    def __init__(
        self,
        tools: list[BaseTool] | None = None,
        timeout_retry_max_attempts: int = 3,
        timeout_retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0),
        tracer: Tracer | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._tools = {tool.name: tool for tool in (tools or [])}
        self._timeout_retry_max_attempts = timeout_retry_max_attempts
        self._timeout_retry_delays = self._normalize_retry_delays(
            timeout_retry_delays,
            timeout_retry_max_attempts,
        )
        self._tracer = tracer
        self._logger = logger or Logger.get_instance()
        self._router = ToolChainRouter(self._tools.values())

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        self._router = ToolChainRouter(self._tools.values())

    def auto_register(
        self,
        module_names: list[str] | None = None,
        package_name: str | None = None,
    ) -> None:
        tools = discover_tools(module_names=module_names, package_name=package_name)
        for tool in tools:
            self.register(tool)
        self._logger.info(
            "Tools auto-registered",
            zap.any("count", len(tools)),
            zap.any("names", [t.name for t in tools]),
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def get_tool_schemas_for(self, names: list[str]) -> list[dict[str, Any]]:
        name_set = set(names)
        return [tool.schema() for name, tool in self._tools.items() if name in name_set]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def validate_arguments(self, tool_call: ToolCall) -> list[str]:
        """Return a list of missing required argument names, or [] if valid."""
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return []
        required: list[str] = tool.parameters.get("required", [])
        return [key for key in required if key not in tool_call.arguments]

    def reset_all(self) -> None:
        for tool in self._tools.values():
            tool.reset()

    def execute(
        self,
        tool_call: ToolCall,
        arguments: dict[str, Any] | None = None,
        llm_raw_tool_call_id: str | None = None,
    ) -> ToolResult:
        if not isinstance(tool_call, ToolCall):
            tool_call = ToolCall(
                name=str(tool_call),
                arguments=arguments or {},
                llm_raw_tool_call_id=llm_raw_tool_call_id,
            )
        self._logger.info(
            "Tool execution start",
            zap.any("tool_name", tool_call.name),
            zap.any("argument_keys", list(tool_call.arguments.keys())),
        )
        with self._start_span(
            f"tool.execute.{tool_call.name}",
            attributes={
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "llm_raw_tool_call_id": tool_call.llm_raw_tool_call_id,
            },
        ) as span:
            result = self._execute_with_retry(tool_call)
            span.add_attributes(
                {
                    "success": result.success,
                    "error_code": None if result.error is None else result.error.code,
                    "error_message": None if result.error is None else result.error.message,
                }
            )
            if not result.success and result.error is not None and self._logger is not None:
                self._logger.error(
                    "Tool call failed",
                    zap.any("tool_name", tool_call.name),
                    zap.any("error_code", result.error.code),
                    zap.any("error_message", result.error.message),
                    zap.any("arguments", tool_call.arguments),
                )
            else:
                self._logger.info(
                    "Tool execution succeeded",
                    zap.any("tool_name", tool_call.name),
                    zap.any("output_length", len(result.output or "")),
                )
            return result

    def _execute_with_retry(self, tool_call: ToolCall) -> ToolResult:
        total_attempts = self._timeout_retry_max_attempts
        result: ToolResult | None = None
        for attempt_idx in range(total_attempts):
            self._logger.info(
                "Tool execution attempt",
                zap.any("tool_name", tool_call.name),
                zap.any("attempt", attempt_idx + 1),
                zap.any("max_attempts", total_attempts),
            )
            result = self._router.route(tool_call)
            if result.success:
                return result
            if result.error is not None and "TIMEOUT" in result.error.code:
                if attempt_idx < total_attempts - 1:
                    self._logger.warning(
                        "Tool timed out, retrying",
                        zap.any("tool_name", tool_call.name),
                        zap.any("attempt", attempt_idx + 1),
                        zap.any("delay_seconds", self._timeout_retry_delays[attempt_idx]),
                    )
                    time.sleep(self._timeout_retry_delays[attempt_idx])
                    continue
            else:
                # Non-timeout failure: no point retrying
                break

        return result  # type: ignore[return-value]

    def _start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        if self._tracer is None:
            return Span(None)
        return self._tracer.start_span(name=name, type="tool", attributes=attributes)

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
            error=build_pipeline_error(TOOL_NOT_FOUND, f"Unknown tool: {tool_call.name}"),
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
        try:
            result = self._tool.run(tool_call.arguments)
            result.llm_raw_tool_call_id = tool_call.llm_raw_tool_call_id
            return result
        except TimeoutError as exc:
            return ToolResult(
                output="",
                llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
                success=False,
                error=build_pipeline_error(TOOL_TIMEOUT, f"Tool `{tool_call.name}` timed out: {exc}"),
            )
        except PipelineError as exc:
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
                error=build_pipeline_error(
                    TOOL_EXECUTION_ERROR,
                    f"Tool `{tool_call.name}` failed unexpectedly: {exc}",
                ),
            )


class FallbackToolHandler(BaseToolHandler):
    def can_handle(self, tool_call: ToolCall) -> bool:
        return True

    def process(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            output="",
            llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
            success=False,
            error=build_pipeline_error(TOOL_NOT_FOUND, f"Unknown tool: {tool_call.name}"),
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
    tracer: Tracer | None = None,
    logger: Logger | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(
        timeout_retry_max_attempts=timeout_retry_max_attempts,
        timeout_retry_delays=timeout_retry_delays,
        tracer=tracer,
        logger=logger,
    )
    registry.auto_register(module_names=module_names, package_name=package_name)
    return registry


def _safe_import(module_name: str) -> Optional[ModuleType]:
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
