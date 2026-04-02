from __future__ import annotations

import threading
from typing import Callable

from agent import Agent, ReActAgent
from agent.impl import ReActAgentContext
from config import ConfigValueReader, JsonConfig
from context.agent_context import AgentContext
from context.formatter import MessageFormatter
from context.session import Session
from llm import (
    BaseLLMClient,
    ClaudeLLMClient,
    DeepSeekLLMClient,
    FallbackLLMClient,
    LLMProviderRegistry,
    OpenAILLMClient,
    QwenLLMClient,
)
from queue.message_queue import AgentToUserQueue, UserToAgentQueue
from rag.rag_service import RAGService
from rag.storage import ChromaDBStorage, FileStorage, SQLiteStorage, StorageRegistry
from schemas import AgentError, ChatMessage, SessionStatus, build_error
from tracing import SpanHandle, Tracer
from tools import ToolRegistry, create_default_tool_registry
from utils.log import Logger, zap
from utils.thread_event import ThreadEvent


class AgentThread(threading.Thread):
    def __init__(
        self,
        user_to_agent_queue: UserToAgentQueue,
        agent_to_user_queue: AgentToUserQueue,
        config: JsonConfig,
        stop_event: ThreadEvent,
        stop_callback: Callable[[str | None], None],
        logger: Logger,
    ) -> None:
        super().__init__(name="AgentThread", daemon=False)
        self._user_to_agent_queue = user_to_agent_queue
        self._agent_to_user_queue = agent_to_user_queue
        self._agent_context: AgentContext = self._build_agent_context()
        self._session = Session()
        self._config = config
        self._config_value_reader = ConfigValueReader(config)
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._logger = logger
        self._user_message_wait_timeout_seconds = self._config_value_reader.positive_float(
            "agent.latency.agent_user_message_wait_timeout_seconds",
            2.0,
        )
        self._storage_registry: StorageRegistry | None = None
        self._storage = None
        self._rag_service: RAGService | None = None
        self._message_formatter: MessageFormatter | None = None
        self._tool_registry: ToolRegistry | None = None
        self._llm_client: BaseLLMClient | None = None
        self._agent: Agent | None = None
        self._tracer: Tracer | None = None
        self._session_span: SpanHandle | None = None
        self._base_system_prompt = self._agent_context.get_system_prompt()
        self._load_agent_config()
        self._load_llm_config()
        self._load_tool_config()
        self._load_tracing_config()
        try:
            self._tracer = self._build_tracer()
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

    @staticmethod
    def _build_agent_context() -> AgentContext:
        return ReActAgentContext()

    def stop(self) -> None:
        self._stop_callback(self.name)

    def reset(self) -> None:
        self._finish_session_trace()
        if self._agent is not None:
            self._agent.reset()
        self._restore_base_system_prompt()

    def release_resources(self) -> None:
        self.reset()
        if self._agent is not None:
            self._agent.release_resources()
        self._agent_context.release()
        if self._storage_registry is not None:
            self._storage_registry.close_all()
        self._agent = None
        self._llm_client = None
        self._tool_registry = None
        self._message_formatter = None
        self._rag_service = None
        self._storage = None
        self._storage_registry = None
        self._session_span = None

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
        return RAGService(self._storage, tracer=self._tracer)

    def _load_tracing_config(self) -> None:
        self._tracing_enabled = bool(self._config.get("tracing.enabled", True))
        self._tracing_output_path = self._config.get("tracing.output_path", "runtime/traces.jsonl")
        self._tracing_capture_payloads = bool(
            self._config.get("tracing.capture_payloads", False)
        )
        self._tracing_max_content_length = self._config_value_reader.positive_int(
            "tracing.max_content_length",
            default=1000,
        )

    def _build_tracer(self) -> Tracer:
        return Tracer(
            enabled=self._tracing_enabled,
            output_path=self._tracing_output_path,
            capture_payloads=self._tracing_capture_payloads,
            max_content_length=self._tracing_max_content_length,
        )

    def _load_agent_config(self) -> None:
        self._max_tool_iterations = int(self._config.get("agent.max_tool_iterations", 3))
        self._max_react_attempt_iterations = int(
            self._config.get("agent.max_react_attempt_iterations", 20)
        )

    def _load_llm_config(self) -> None:
        self._llm_retry_max_attempts = int(self._config.get("llm.retry.max_attempts", 4))
        self._llm_retry_delays = self._config_value_reader.retry_delays(
            "llm.retry.backoff_seconds",
        )
        self._llm_context_trimming_enabled = bool(
            self._config.get("llm.context_trimming.enabled", True)
        )
        self._llm_context_max_messages = self._config_value_reader.positive_int(
            "llm.context_trimming.max_messages",
            default=40,
        )

    def _load_tool_config(self) -> None:
        self._tool_retry_max_attempts = int(self._config.get("tools.retry.max_attempts", 4))
        self._tool_retry_delays = self._config_value_reader.retry_delays(
            "tools.retry.backoff_seconds",
        )

    def _build_message_formatter(self) -> MessageFormatter:
        if not self._llm_context_trimming_enabled:
            return MessageFormatter(max_messages=None)
        return MessageFormatter(max_messages=self._llm_context_max_messages)

    def _build_tool_registry(self) -> ToolRegistry:
        package_name = self._config.get("tools.package")
        module_names = self._config.get("tools.modules", [])
        if not isinstance(module_names, list):
            module_names = []
        return create_default_tool_registry(
            module_names=module_names,
            package_name=package_name,
            timeout_retry_max_attempts=self._tool_retry_max_attempts,
            timeout_retry_delays=self._tool_retry_delays,
            tracer=self._tracer,
        )

    def _build_llm_client(self) -> BaseLLMClient:
        registry = LLMProviderRegistry()
        default_provider_name = self._config.get("llm.provider", "openai")
        enable_provider_fallback = bool(
            self._config.get("llm.enable_provider_fallback", False)
        )
        configured_priority = self._config.get("llm.priority_chain")
        provider_priority: list[str]
        if enable_provider_fallback and isinstance(configured_priority, list) and configured_priority:
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
            provider.set_tracer(self._tracer)
            registry.register(provider)

        return FallbackLLMClient(
            registry=registry,
            provider_priority=provider_priority,
            max_attempts=self._llm_retry_max_attempts,
            retry_delays=self._llm_retry_delays,
            enable_provider_fallback=enable_provider_fallback,
        )

    def _create_llm_provider(
        self,
        provider_name: str,
        overrides: dict,
    ) -> BaseLLMClient:
        global_timeout = float(self._config.get("llm.timeout", 60.0))

        api_key = None
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
            agent_context=self._agent_context,
            session=self._session,
            message_formatter=self._message_formatter,
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
            rag_service=self._rag_service,
            max_tool_iterations=self._max_tool_iterations,
        )

    def run(self) -> None:
        try:
            while self._is_running():
                session_status = self._session.get_status()

                if (
                    session_status == SessionStatus.IN_PROGRESS
                    and self._agent is not None
                    and self._agent.get_react_attempt_iterations() > self._max_react_attempt_iterations
                ):
                    self._agent_to_user_queue.send_agent_message(
                        ChatMessage(
                            role="assistant",
                            content="Sorry, this question is too hard, i can not solve",
                            metadata={"session_status": SessionStatus.NEW_TASK},
                        )
                    )
                    self.reset()
                    continue

                incoming_message = self._wait_for_user_message(session_status)
                if incoming_message is None and session_status == SessionStatus.NEW_TASK: #排除法，此时说明收到的停止信号了
                    continue  # No new user message, loop back and check stop condition or wait again   
                if session_status == SessionStatus.NEW_TASK and incoming_message is not None:
                    self._start_session_trace(incoming_message)
                    self._record_user_input_trace(incoming_message, input_type="question")
                    self._agent.begin_session()
                    self._agent_to_user_queue.send_agent_message(
                        ChatMessage(
                            role="assistant",
                            content="",
                            metadata={
                                "control": True,
                                "session_status": SessionStatus.IN_PROGRESS,
                            },
                        )
                    )
                    session_status = self._session.get_status()
                elif incoming_message is not None:
                    self._record_user_input_trace(incoming_message, input_type="hint")

                try:
                    execution_result = self._agent.run(session_status, incoming_message)
                    for message in execution_result.user_messages:
                        self._agent_to_user_queue.send_agent_message(message)
                    if execution_result.error is not None:
                        self._logger.error(
                            "Agent execution returned an internal error",
                            zap.any("trace_id", None if self._tracer is None else self._tracer.current_trace_id()),
                            zap.any("span_id", None if self._tracer is None else self._tracer.current_span_id()),
                            zap.any("error", execution_result.error),
                        )
                    if execution_result.should_reset:
                        if not execution_result.user_messages:
                            self._agent_to_user_queue.send_agent_message(
                                ChatMessage(
                                    role="assistant",
                                    content="",
                                    metadata={
                                        "control": True,
                                        "session_status": SessionStatus.NEW_TASK,
                                    },
                                )
                            )
                        self._finish_session_trace(error=execution_result.error)
                        self.reset()
                except Exception as exc:
                    normalized_error = self._normalize_error(exc)
                    self._finish_session_trace(error=normalized_error)
                    self._logger.error(
                        "Agent thread execution failed",
                        zap.any("trace_id", None if self._tracer is None else self._tracer.current_trace_id()),
                        zap.any("span_id", None if self._tracer is None else self._tracer.current_span_id()),
                        zap.any("error", normalized_error),
                    )
                    break
        except Exception as exc:
            self._finish_session_trace(error=exc)
            self._logger.error("Agent thread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()

    def _start_session_trace(self, user_message: ChatMessage) -> None:
        if self._tracer is None or self._session_span is not None:
            return
        self._session_span = self._tracer.start_trace(
            "session",
            attributes={
                "thread": self.name,
                "session_status": self._session.get_status(),
                "user_message": user_message.content,
            },
        )

    def _finish_session_trace(self, error: Exception | AgentError | None = None) -> None:
        if self._session_span is None:
            return
        status = "error" if error is not None else "ok"
        self._session_span.finish(status=status, error=error)
        self._session_span = None

    def _record_user_input_trace(
        self,
        user_message: ChatMessage,
        input_type: str,
    ) -> None:
        if self._tracer is None:
            return
        with self._tracer.start_span(
            name="user.input",
            kind="input",
            attributes={
                "input_type": input_type,
                "role": user_message.role,
                "content": user_message.content,
            },
        ):
            return None

    def _wait_for_user_message(
        self,
        session_status: SessionStatus,
    ) -> ChatMessage | None: #两种情况下会返回None：1. 收到停止信号 2. 任务进行中但没有收到用户消息（此时agent可以继续执行之前的任务）
        while self._is_running():
            user_message = self._user_to_agent_queue.get_user_message(
                timeout=self._user_message_wait_timeout_seconds
            )
            if user_message is not None:
                return user_message
            if session_status == SessionStatus.IN_PROGRESS:
                return None
        return None

    def _is_running(self) -> bool:
        return not self._stop_event.is_set() and not self._is_any_queue_closed()

    def _is_any_queue_closed(self) -> bool:
        return self._user_to_agent_queue.is_closed() or self._agent_to_user_queue.is_closed()

    def _restore_base_system_prompt(self) -> None:
        with self._agent_context._lock:
            self._agent_context._system_prompt = self._base_system_prompt

    @staticmethod
    def _normalize_error(exc: Exception) -> AgentError:
        if isinstance(exc, AgentError):
            return exc
        return build_error("UNEXPECTED_ERROR", f"Agent encountered an unexpected error: {exc}")
