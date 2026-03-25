from __future__ import annotations

import threading

from config import JsonConfig
from context.shared_context import SharedContext
from llm.llm_api import (
    BaseLLMClient,
    DynamicLLMClient,
    LLMProviderRegistry,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)
from llm.message_formatter import MessageFormatter
from queue.message_queue import MessageQueue
from rag.rag_service import RAGService
from rag.storage import InMemoryStorage, SQLiteStorage, StorageRegistry
from schemas import AgentEvent, ChatMessage
from tools import ToolRegistry, create_default_tool_registry


class AgentThread(threading.Thread):
    def __init__(
        self,
        message_queue: MessageQueue,
        shared_context: SharedContext,
        config: JsonConfig,
        max_tool_iterations: int | None = None,
    ) -> None:
        super().__init__(name="AgentThread", daemon=True)
        self._message_queue = message_queue
        self._shared_context = shared_context
        self._config = config
        self._storage_registry = self._build_storage_registry()
        self._storage = self._build_storage()
        self._rag_service = self._build_rag_service()
        self._message_formatter = self._build_message_formatter()
        self._tool_registry = self._build_tool_registry()
        self._llm_client = self._build_llm_client()
        self._max_tool_iterations = max_tool_iterations or int(
            self._config.get("agent.max_tool_iterations", 3)
        )
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _build_storage_registry(self) -> StorageRegistry:
        memory_storage = InMemoryStorage()
        sqlite_path = self._config.get("storage.sqlite.path", "runtime/agent_storage.db")
        sqlite_storage = SQLiteStorage(sqlite_path)
        sqlite_storage.seed(memory_storage.get_documents())
        return StorageRegistry([memory_storage, sqlite_storage])

    def _build_storage(self):
        backend_name = self._config.get("storage.backend", "memory")
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
        registry = LLMProviderRegistry([MockLLMClient()])
        provider_name = self._config.get("llm.provider", "mock")
        if provider_name == "openai_compatible":
            openai_client = OpenAICompatibleLLMClient.from_settings(
                api_key=self._config.get("llm.api_key"),
                model=self._config.get("llm.model", "gpt-4.1-mini"),
                base_url=self._config.get("llm.base_url", "https://api.openai.com/v1"),
                timeout=float(self._config.get("llm.timeout", 60.0)),
            )
            registry.register(openai_client)
        return DynamicLLMClient(registry, default_provider=provider_name)

    def run(self) -> None:
        while not self._stop_event.is_set() and not self._message_queue.is_closed():
            user_message = self._message_queue.get_user_message(timeout=0.2)
            if user_message is None:
                continue

            if user_message.metadata.get("control") == "shutdown":
                self._message_queue.close()
                self.stop()
                break

            try:
                self._handle_user_message(user_message)
            except Exception as exc:
                error_message = ChatMessage(
                    role="assistant",
                    content=f"Agent 处理消息时发生异常：{exc}",
                )
                self._message_queue.send_agent_message(error_message)

    def _handle_user_message(self, user_message: ChatMessage) -> None:
        self._shared_context.append_message(user_message)
        self._shared_context.append_event(
            AgentEvent(event_type="user_message_received", payload={"content": user_message.content})
        )

        working_messages = self._shared_context.get_conversation()
        rag_context = self._rag_service.retrieve(user_message.content)

        for iteration in range(self._max_tool_iterations + 1):
            request = self._message_formatter.build_request(
                system_prompt=self._shared_context.system_prompt,
                conversation=working_messages,
                tools=self._tool_registry.get_tool_schemas(),
                context=rag_context,
            )
            response = self._message_formatter.parse_response(self._llm_client.generate(request))
            assistant_message = response.assistant_message
            self._shared_context.append_message(assistant_message)

            if not response.tool_calls:
                self._message_queue.send_agent_message(assistant_message)
                self._shared_context.append_event(
                    AgentEvent(event_type="assistant_answered", payload={"content": assistant_message.content})
                )
                return

            if iteration >= self._max_tool_iterations:
                fallback = ChatMessage(
                    role="assistant",
                    content="工具调用次数超过上限，本轮先停止，避免进入死循环。",
                )
                self._shared_context.append_message(fallback)
                self._message_queue.send_agent_message(fallback)
                return

            for tool_call in response.tool_calls:
                result = self._tool_registry.execute(
                    tool_call.name,
                    tool_call.arguments,
                    tool_call.call_id,
                )
                self._shared_context.append_event(
                    AgentEvent(
                        event_type="tool_executed",
                        payload={
                            "tool_name": tool_call.name,
                            "success": result.success,
                            "error": result.error,
                        },
                    )
                )
                observation_text = result.output if result.success else f"Tool error: {result.error}"
                observation = self._message_formatter.format_tool_observation(
                    tool_name=tool_call.name,
                    output=observation_text,
                    call_id=tool_call.call_id,
                )
                working_messages = working_messages + [assistant_message, observation]
                self._shared_context.append_message(observation)
