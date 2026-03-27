from __future__ import annotations

import threading

from agent import Agent, ReActAgent
from config import JsonConfig
from context.shared_context import SharedContext
from llm import (
    BaseLLMClient,
    ClaudeLLMClient,
    DeepSeekLLMClient,
    DynamicLLMClient,
    LLMProviderRegistry,
    OpenAILLMClient,
    QwenLLMClient,
)
from llm.message_formatter import MessageFormatter
from queue.message_queue import MessageQueue
from rag.rag_service import RAGService
from rag.storage import ChromaDBStorage, FileStorage, SQLiteStorage, StorageRegistry
from schemas import AgentError, ChatMessage, SessionStatus, SystemMessage, build_error
from tools import ToolRegistry, create_default_tool_registry
from utils.log import Logger, zap


class AgentThread(threading.Thread):
    def __init__(
        self,
        message_queue: MessageQueue,
        shared_context: SharedContext,
        config: JsonConfig,
        stop_event: threading.Event,
        logger: Logger,
    ) -> None:
        super().__init__(name="AgentThread", daemon=False)
        self._message_queue = message_queue
        self._shared_context = shared_context
        self._config = config
        self._stop_event = stop_event
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
        self._stop_event.set()

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
        provider_name = self._config.get("llm.provider", "openai")
        if provider_name == "openai":
            provider = OpenAILLMClient.from_settings(
                api_key=self._config.get("llm.api_key"),
                model=self._config.get("llm.model", "gpt-4.1-mini"),
                base_url=self._config.get("llm.base_url", "https://api.openai.com/v1"),
                timeout=float(self._config.get("llm.timeout", 60.0)),
            )
            registry.register(provider)
        elif provider_name == "qwen":
            provider = QwenLLMClient.from_settings(
                api_key=self._config.get("llm.api_key"),
                model=self._config.get("llm.model", "qwen-plus"),
                base_url=self._config.get("llm.base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                timeout=float(self._config.get("llm.timeout", 60.0)),
            )
            registry.register(provider)
        elif provider_name == "deepseek":
            provider = DeepSeekLLMClient.from_settings(
                api_key=self._config.get("llm.api_key"),
                model=self._config.get("llm.model", "deepseek-chat"),
                base_url=self._config.get("llm.base_url", "https://api.deepseek.com/v1"),
                timeout=float(self._config.get("llm.timeout", 60.0)),
            )
            registry.register(provider)
        elif provider_name == "claude":
            provider = ClaudeLLMClient.from_settings(
                api_key=self._config.get("llm.api_key"),
                model=self._config.get("llm.model", "claude-3-5-sonnet-latest"),
                base_url=self._config.get("llm.base_url", "https://api.anthropic.com"),
                timeout=float(self._config.get("llm.timeout", 60.0)),
                max_tokens=int(self._config.get("llm.max_tokens", 1024)),
                anthropic_version=self._config.get("llm.anthropic_version", "2023-06-01"),
            )
            registry.register(provider)
        else:
            raise build_error("LLM_PROVIDER_NOT_FOUND", f"Unsupported LLM provider: {provider_name}")
        return DynamicLLMClient(registry, default_provider=provider_name)

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
            while not self._stop_event.is_set() and not self._message_queue.is_closed():
                session_status = self._shared_context.get_session_status()

                if (
                    session_status == SessionStatus.IN_PROGRESS
                    and self._agent is not None
                    and self._agent.get_react_attempt_iterations() > self._max_react_attempt_iterations
                ):
                    self._message_queue.send_agent_message(
                        ChatMessage(
                            role="assistant",
                            content="Sorry, this question is too hard, i can not solve",
                        )
                    )
                    self.reset()
                    continue

                incoming_message = self._wait_for_user_message(session_status)
                if isinstance(incoming_message, SystemMessage):
                    if self._handle_system_message(incoming_message):
                        break
                    continue

                try:
                    execution_result = self._agent.run(session_status, incoming_message)
                    for message in execution_result.user_messages:
                        self._message_queue.send_agent_message(message)
                    if execution_result.should_reset:
                        self.reset()
                except Exception as exc:
                    normalized_error = self._normalize_error(exc)
                    self._logger.error(
                        "Agent thread execution failed",
                        zap.any("error", normalized_error),
                    )
                    self.stop()
                    break
        except Exception as exc:
            self._logger.error("Agent thread crashed", zap.any("error", exc))
            self.stop()
        finally:
            self.release_resources()

    def _wait_for_user_message(
        self,
        session_status: SessionStatus,
    ) -> ChatMessage | SystemMessage | None:
        timeout = None if session_status == SessionStatus.NEW_TASK else 5.0
        while not self._stop_event.is_set() and not self._message_queue.is_closed():
            user_message = self._message_queue.get_user_message(timeout=timeout)
            if user_message is not None:
                return user_message
            if session_status == SessionStatus.IN_PROGRESS:
                return None
        return SystemMessage(command="shutdown", content="queue closed")

    def _handle_system_message(self, message: SystemMessage) -> bool:
        if message.command in {"quit", "shutdown"}:
            self.reset()
            return True
        return False

    def _restore_base_system_prompt(self) -> None:
        with self._shared_context._lock:
            self._shared_context._system_prompt = self._base_system_prompt

    @staticmethod
    def _normalize_error(exc: Exception) -> AgentError:
        if isinstance(exc, AgentError):
            return exc
        return build_error("UNEXPECTED_ERROR", f"Agent encountered an unexpected error: {exc}")
