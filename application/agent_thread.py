from __future__ import annotations

import threading
from typing import Callable

from agent import Agent, ReActAgent
from config import JsonConfig
from context.shared_context import SharedContext
from llm import (
    BaseLLMClient,
    ClaudeLLMClient,
    DeepSeekLLMClient,
    FallbackLLMClient,
    LLMProviderRegistry,
    OpenAILLMClient,
    QwenLLMClient,
)
from llm.message_formatter import MessageFormatter
from queue.message_queue import AgentToUserQueue, UserToAgentQueue
from rag.rag_service import RAGService
from rag.storage import ChromaDBStorage, FileStorage, SQLiteStorage, StorageRegistry
from schemas import AgentError, ChatMessage, SessionStatus, build_error
from tools import ToolRegistry, create_default_tool_registry
from utils.log import Logger, zap
from utils.thread_event import ThreadEvent


class AgentThread(threading.Thread):
    def __init__(
        self,
        user_to_agent_queue: UserToAgentQueue,
        agent_to_user_queue: AgentToUserQueue,
        shared_context: SharedContext,
        config: JsonConfig,
        stop_event: ThreadEvent,
        stop_callback: Callable[[str | None], None],
        logger: Logger,
    ) -> None:
        super().__init__(name="AgentThread", daemon=False)
        self._user_to_agent_queue = user_to_agent_queue
        self._agent_to_user_queue = agent_to_user_queue
        self._shared_context = shared_context
        self._config = config
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._logger = logger
        self._storage_registry: StorageRegistry | None = None
        self._storage = None
        self._rag_service: RAGService | None = None
        self._message_formatter: MessageFormatter | None = None
        self._tool_registry: ToolRegistry | None = None
        self._llm_client: BaseLLMClient | None = None
        self._agent: Agent | None = None
        self._base_system_prompt = self._shared_context.get_system_prompt()
        self._max_tool_iterations = int(self._config.get("agent.max_tool_iterations", 3)
        )
        self._max_react_attempt_iterations = int(
            self._config.get("agent.max_react_attempt_iterations", 20)
        )
        try:
            self._storage_registry = self._build_storage_registry()
            self._storage = self._build_storage()
            self._rag_service = self._build_rag_service()
            self._message_formatter = self._build_message_formatter()
            self._tool_registry = self._build_tool_registry()
            self._llm_client = self._build_llm_client()
            self._agent = self._build_agent()
        except Exception:
            self.release_resources()
            self.stop()
            raise

    def stop(self) -> None:
        self._stop_callback(self.name)

    def reset(self) -> None:
        if self._agent is not None:
            self._agent.reset()
        self._shared_context.archive_current_task()
        self._shared_context.set_session_status(SessionStatus.NEW_TASK)
        self._restore_base_system_prompt()

    def release_resources(self) -> None:
        self.reset()
        if self._agent is not None:
            self._agent.release_resources()
        if self._storage_registry is not None:
            self._storage_registry.close_all()
        self._agent = None
        self._llm_client = None
        self._tool_registry = None
        self._message_formatter = None
        self._rag_service = None
        self._storage = None
        self._storage_registry = None

    def _build_storage_registry(self) -> StorageRegistry:
        #TODO 暂时还用不上这里，先把桩代码写了，后续完善
        file_storage = FileStorage(
            self._config.get("storage.file.path", "runtime/nanoagent_soul.json")
        )
        sqlite_path = self._config.get("storage.sqlite.path", "runtime/nanoagent_local_storage.db")
        sqlite_storage = SQLiteStorage(sqlite_path)
        sqlite_storage.seed(file_storage.get_documents())
        storages = [file_storage, sqlite_storage]

        chromadb_path = self._config.get("storage.chromadb.persist_directory")
        if chromadb_path:
            chromadb_storage = ChromaDBStorage(
                persist_directory=chromadb_path,
                collection_name=self._config.get("storage.chromadb.collection_name", "nanoagent_collection"),
            )
            if not chromadb_storage.get_documents():
                chromadb_storage.upsert_documents(file_storage.get_documents())
            storages.append(chromadb_storage)

        return StorageRegistry(storages)

    def _build_storage(self):
        backend_name = self._config.get("storage.backend", "file")
        return self._storage_registry.get(backend_name)

    def _build_rag_service(self) -> RAGService:
        return RAGService(self._storage)

    @staticmethod
    def _build_message_formatter() -> MessageFormatter:
        return MessageFormatter()

    def _build_tool_registry(self) -> ToolRegistry:
        package_name = self._config.get("tools.package")
        module_names = self._config.get("tools.modules", [])
        if not isinstance(module_names, list):
            module_names = []
        return create_default_tool_registry(
            module_names=module_names,
            package_name=package_name,
        )

    def _build_llm_client(self) -> BaseLLMClient:
        registry = LLMProviderRegistry()
        default_provider_name = self._config.get("llm.provider", "openai")
        configured_priority = self._config.get("llm.priority_chain")
        provider_priority: list[str]
        if isinstance(configured_priority, list) and configured_priority:
            provider_priority = [str(item) for item in configured_priority if str(item).strip()]
        else:
            provider_priority = [default_provider_name]

        provider_settings = self._config.get("llm.provider_settings", {})
        if not isinstance(provider_settings, dict):
            provider_settings = {}

        for provider_name in provider_priority:
            overrides = provider_settings.get(provider_name, {})
            if not isinstance(overrides, dict):
                overrides = {}
            provider = self._create_llm_provider(provider_name, overrides)
            registry.register(provider)

        return FallbackLLMClient(
            registry=registry,
            provider_priority=provider_priority,
            retry_delays=(1.0, 2.0, 4.0),
        )

    def _create_llm_provider(
        self,
        provider_name: str,
        overrides: dict,
    ) -> BaseLLMClient:
        global_api_key = self._config.get("llm.api_key")
        global_timeout = float(self._config.get("llm.timeout", 60.0))

        api_key = overrides.get("api_key", global_api_key)
        timeout = float(overrides.get("timeout", global_timeout))

        if provider_name == "openai":
            return OpenAILLMClient.from_settings(
                api_key=api_key,
                model=overrides.get("model", self._config.get("llm.model", "gpt-4.1-mini")),
                base_url=overrides.get("base_url", self._config.get("llm.base_url", "https://api.openai.com/v1")),
                timeout=timeout,
            )
        if provider_name == "qwen":
            return QwenLLMClient.from_settings(
                api_key=api_key,
                model=overrides.get("model", "qwen-plus"),
                base_url=overrides.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                timeout=timeout,
            )
        if provider_name == "deepseek":
            return DeepSeekLLMClient.from_settings(
                api_key=api_key,
                model=overrides.get("model", "deepseek-chat"),
                base_url=overrides.get("base_url", "https://api.deepseek.com/v1"),
                timeout=timeout,
            )
        if provider_name == "claude":
            return ClaudeLLMClient.from_settings(
                api_key=api_key,
                model=overrides.get("model", "claude-3-5-sonnet-latest"),
                base_url=overrides.get("base_url", "https://api.anthropic.com"),
                timeout=timeout,
                max_tokens=int(overrides.get("max_tokens", self._config.get("llm.max_tokens", 1024))),
                anthropic_version=overrides.get(
                    "anthropic_version",
                    self._config.get("llm.anthropic_version", "2023-06-01"),
                ),
            )
        raise build_error("LLM_PROVIDER_NOT_FOUND", f"Unsupported LLM provider: {provider_name}")

    def _build_agent(self) -> Agent:
        return ReActAgent(
            shared_context=self._shared_context,
            message_formatter=self._message_formatter,
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
            rag_service=self._rag_service,
            max_tool_iterations=self._max_tool_iterations,
        )

    def run(self) -> None:
        try:
            while not self._stop_event.is_set() and not self._is_any_queue_closed():
                session_status = self._shared_context.get_session_status()

                if (
                    session_status == SessionStatus.IN_PROGRESS
                    and self._agent is not None
                    and self._agent.get_react_attempt_iterations() > self._max_react_attempt_iterations
                ):
                    self._agent_to_user_queue.send_agent_message(
                        ChatMessage(
                            role="assistant",
                            content="Sorry, this question is too hard, i can not solve",
                        )
                    )
                    self.reset()
                    continue

                incoming_message = self._wait_for_user_message(session_status)
                if incoming_message is None and session_status == SessionStatus.NEW_TASK:
                    continue  # No new user message, loop back and check stop condition or wait again   

                try:
                    execution_result = self._agent.run(session_status, incoming_message)
                    for message in execution_result.user_messages:
                        self._agent_to_user_queue.send_agent_message(message)
                    if execution_result.should_reset:
                        self.reset()
                except Exception as exc:
                    normalized_error = self._normalize_error(exc)
                    self._logger.error(
                        "Agent thread execution failed",
                        zap.any("error", normalized_error),
                    )
                    break
        except Exception as exc:
            self._logger.error("Agent thread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()

    def _wait_for_user_message(
        self,
        session_status: SessionStatus,
    ) -> ChatMessage | None:
        while not self._stop_event.is_set() and not self._user_to_agent_queue.is_closed():
            user_message = self._user_to_agent_queue.get_user_message(timeout=2)
            if user_message is not None:
                return user_message
            if session_status == SessionStatus.IN_PROGRESS:
                return None
        return None

    def _is_any_queue_closed(self) -> bool:
        return self._user_to_agent_queue.is_closed() or self._agent_to_user_queue.is_closed()

    def _restore_base_system_prompt(self) -> None:
        with self._shared_context._lock:
            self._shared_context._system_prompt = self._base_system_prompt

    @staticmethod
    def _normalize_error(exc: Exception) -> AgentError:
        if isinstance(exc, AgentError):
            return exc
        return build_error("UNEXPECTED_ERROR", f"Agent encountered an unexpected error: {exc}")
