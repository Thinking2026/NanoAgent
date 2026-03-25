from __future__ import annotations

import os

from agent_thread import AgentThread
from llm_api import (
    DynamicLLMClient,
    LLMProviderRegistry,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)
from message_formatter import MessageFormatter
from message_queue import MessageQueue
from rag_service import RAGService
from shared_context import SharedContext
from storage import InMemoryStorage, SQLiteStorage, StorageRegistry
from tools import create_default_tool_registry
from user_thread import UserThread


def build_system_prompt() -> str: #TODO
    return (
        "You are a prototype AI agent. "
        "Use external context when it helps, call tools when necessary, "
        "and keep answers clear and concise."
    )


def main() -> None:
    message_queue = MessageQueue()
    shared_context = SharedContext(system_prompt=build_system_prompt())
    memory_storage = InMemoryStorage()
    sqlite_storage = SQLiteStorage("runtime/agent_storage.db")
    sqlite_storage.seed(memory_storage.get_documents())
    storage_registry = StorageRegistry([memory_storage, sqlite_storage])
    storage = storage_registry.get("memory")
    rag_service = RAGService(storage)
    message_formatter = MessageFormatter()
    tool_registry = create_default_tool_registry()
    llm_registry = LLMProviderRegistry([MockLLMClient()])
    default_provider = "mock"
    if os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"):
        llm_registry.register(OpenAICompatibleLLMClient.from_env())
        default_provider = os.getenv("LLM_PROVIDER", "openai_compatible")
    llm_client = DynamicLLMClient(llm_registry, default_provider=default_provider)

    agent_thread = AgentThread(
        message_queue=message_queue,
        shared_context=shared_context,
        rag_service=rag_service,
        llm_client=llm_client,
        message_formatter=message_formatter,
        tool_registry=tool_registry,
    )
    user_thread = UserThread(
        message_queue=message_queue,
        message_formatter=message_formatter,
    )

    agent_thread.start()
    user_thread.start()

    try:
        user_thread.join()
    except KeyboardInterrupt:
        message_queue.close()
        user_thread.stop()
    finally:
        agent_thread.stop()
        agent_thread.join(timeout=1)


if __name__ == "__main__":
    main()
