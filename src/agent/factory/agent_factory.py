from __future__ import annotations

import os
from typing import TYPE_CHECKING

from config import ConfigReader
from infra.db.bootstrap_documents import load_seed_documents
from infra.db.impl.chromadb_storage import ChromaDBStorage
from infra.db.impl.mysql_storage import MySQLStorage
from infra.db.impl.sqlite_storage import SQLiteStorage
from infra.db.registry import StorageRegistry
from llm.llm_gateway import LLMGateway
from schemas.errors import STORAGE_CONFIG_ERROR, build_pipeline_error
from schemas.event_bus import EventBus
from schemas.ids import TaskId
from schemas.task import LLMProviderCapabilities, Plan, Task
from tools import (
    create_default_tool_registry,
    SQLQueryTool,
    build_sql_query_tool_name,
    build_sql_query_tool_description,
    SQLSchemaTool,
    build_sql_schema_tool_name,
    build_sql_schema_tool_description,
    VectorSearchTool,
    build_vector_search_tool_name,
    build_vector_search_tool_description,
    VectorSchemaTool,
    build_vector_schema_tool_name,
    build_vector_schema_tool_description,
)
from tools.tool_registry import ToolRegistry
from utils.log.log import Logger

from agent.application.pipeline import Pipeline
from agent.models.analysis.analyzer import Analyzer
from agent.models.context.context_manager import ContextManager
from agent.models.evaluate.quality_evaluator import QualityEvaluator
from agent.models.executor.stage_executor import StageExecutor
from agent.models.knowledge.knowledge_loader import KnowledgeLoader
from agent.models.knowledge.knowledge_manager import KnowledgeManager
from agent.models.model_routing.provider_router import ModelSelector
from agent.models.personality.user_preference import PersonalityManager
from agent.models.plan.planner import Planner
from agent.models.reasoning.impl.react.react_strategy import ReActStrategy
from agent.models.reasoning.reasoning_manager import ReasoningManager
from agent.application.driver import PipelineDriver

from infra.observability.tracing import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer

if TYPE_CHECKING:
    from agent.application.pipeline_thread import PipelineThread

class AgentFactory:
    """Builds a fully-wired Pipeline from AgentConfig.

    Single assembly point: domain objects do not know about config format.
    """

    def __init__(self, config: ConfigReader) -> None:
        self._config = config
        self._logger = Logger.get_instance()

    @classmethod
    def from_config(cls, config: ConfigReader) -> AgentFactory:
        return cls(config)

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    def build_renderer(self) -> PromptRenderer:
        if not hasattr(self, "_renderer"):
            self._renderer = Jinja2PromptRenderer()
        return self._renderer

    def build_storage_registry(self) -> StorageRegistry:
        seed_documents = load_seed_documents(
            self._config.get("storage.file.path", "tests/runtime/nanoagent_soul.json")
        )
        storages = [SQLiteStorage(self._build_sqlite_databases())]

        chromadb_path = self._config.get("storage.chromadb.persist_directory")
        if chromadb_path:
            collections = self._build_chromadb_collections()
            chromadb = ChromaDBStorage(
                persist_directory=chromadb_path,
                collections=collections,
            )
            bootstrap = self._config.get("storage.chromadb.bootstrap_collection")
            if isinstance(bootstrap, str) and bootstrap.strip():
                if not chromadb.get_documents(bootstrap):
                    chromadb.upsert_documents(bootstrap, seed_documents)
            storages.append(chromadb)

        mysql_host = str(self._config.get("storage.mysql.host", "")).strip()
        if mysql_host:
            storages.append(MySQLStorage(
                host=mysql_host,
                port=int(self._config.get("storage.mysql.port", 3306)),
                user=os.getenv("MYSQL_USER", ""),
                password=os.getenv("MYSQL_PASSWORD", ""),
                allowed_databases=self._build_mysql_databases(),
                charset=str(self._config.get("storage.mysql.charset", "utf8mb4")),
            ))

        return StorageRegistry(storages)

    def build_tool_registry(self, tracer: Tracer):
        storage_registry = self.build_storage_registry()
        package_name = self._config.get("tools.package", "tools.impl")
        if not isinstance(package_name, str) or not package_name.strip():
            package_name = "tools.impl"
        module_names = self._config.get("tools.modules", [])
        if not isinstance(module_names, list):
            module_names = []
        registry = create_default_tool_registry(
            module_names=module_names,
            package_name=package_name,
            timeout_retry_max_attempts=int(self._config.get("tools.retry.max_attempts", 4)),
            timeout_retry_delays=self._config.retry_delays("tools.retry.backoff_seconds"),
            tracer=tracer,
            logger=self._logger,
        )
        if storage_registry:
            self._register_storage_tools(registry, storage_registry)
        return registry

    # ------------------------------------------------------------------
    # LLM gateway
    # ------------------------------------------------------------------

    def build_llm_gateway(self, tracer: Tracer) -> LLMGateway:
        return LLMGateway(
            config=self._config,
            tracer=tracer,
            logger=self._logger,
        )

    # ------------------------------------------------------------------
    # Domain objects
    # ------------------------------------------------------------------
    def build_model_selector(self, tracer: Tracer) -> ModelSelector:
        priority_chain = self._config.get("llm.priority_chain", ["deepseek"])
        if not isinstance(priority_chain, list) or not priority_chain:
            priority_chain = ["deepseek"]

        capabilities: list[LLMProviderCapabilities] = []
        for name in priority_chain:
            cap_cfg = self._config.get(f"llm.provider_settings.{name}.capabilities", {})
            if not isinstance(cap_cfg, dict):
                cap_cfg = {}
            capabilities.append(LLMProviderCapabilities(
                name=name,
                cognitive_complexity=list(cap_cfg.get("cognitive_complexity", ["simple", "medium", "complex"])),
                best_scenarios=list(cap_cfg.get("best_scenarios", [])),
                top_strengths=list(cap_cfg.get("top_strengths", [])),
                cost_tier=str(cap_cfg.get("cost_tier", "medium")),
                latency_tier=str(cap_cfg.get("latency_tier", "medium")),
                context_size=int(cap_cfg.get("context_size",
                    self._config.get(f"llm.provider_settings.{name}.context_window", 32000))),
            ))

        return ModelSelector(
            config=self._config,
            logger=self._logger,
            tracer=tracer,
            provider_capabilities=capabilities,
            enable_fallback=bool(self._config.get("llm.enable_provider_fallback", True)),
        )

    def build_context_manager(self, tracer: Tracer, llm_gateway: LLMGateway, tool_registry: ToolRegistry) -> ContextManager:
        return ContextManager(config=self._config, logger=self._logger, tracer=tracer, llm_gateway=llm_gateway, tool_registry=tool_registry)

    def build_knowledge_loader(self, tracer: Tracer)-> KnowledgeLoader:
        return KnowledgeLoader(config=self._config, logger=self._logger, tracer=tracer, renderer=self.build_renderer())

    def build_personality_manager(self, tracer: Tracer) -> PersonalityManager:
        return PersonalityManager(config=self._config, logger=self._logger, tracer=tracer, renderer=self.build_renderer())

    def build_analyzer(self, tracer: Tracer, event_bus: EventBus) -> Analyzer:
        return Analyzer(config=self._config, logger=self._logger, tracer=tracer, event_bus=event_bus, renderer=self.build_renderer())

    def build_reasoning_manager(self, tracer: Tracer, llm_gateway: LLMGateway) -> ReasoningManager:
        strategy = ReActStrategy()
        return ReasoningManager(config=self._config, logger=self._logger, tracer=tracer, llm_gateway=llm_gateway, strategy=strategy)

    def build_stage_executor(
        self,
        tracer: Tracer,
        reasoning_manager: ReasoningManager,
        context_manager: ContextManager,
        quality_evaluator: QualityEvaluator,
        knowledge_loader: KnowledgeLoader,
        planner: Planner,
        llm_gateway: LLMGateway,
        event_bus: EventBus,
        model_selector: ModelSelector,
        tool_registry: ToolRegistry,
    ) -> StageExecutor:
        return StageExecutor(
            config=self._config,
            logger=self._logger,
            tracer=tracer,
            reasoning_manager=reasoning_manager,
            context_manager=context_manager,
            quality_evaluator=quality_evaluator,
            knowledge_loader=knowledge_loader,
            planner=planner,
            llm_gateway=llm_gateway,
            event_bus=event_bus,
            model_selector=model_selector,
            renderer=self.build_renderer(),
            tool_registry=tool_registry,
        )

    def build_planner(self, tracer: Tracer, event_bus: EventBus, evaluator: QualityEvaluator) -> Planner:
        return Planner(config=self._config, logger=self._logger, tracer=tracer, event_bus=event_bus, evaluator=evaluator, renderer=self.build_renderer())

    def build_quality_evaluator(self, tracer: Tracer) -> QualityEvaluator:
        return QualityEvaluator(config=self._config, logger=self._logger, tracer=tracer, renderer=self.build_renderer())

    def build_knowledge_manager(self, tracer: Tracer)-> KnowledgeManager:
        return KnowledgeManager(config=self._config, logger=self._logger, tracer=tracer, renderer=self.build_renderer())

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def build_tracer(self) -> Tracer:
        tracing_enabled = bool(self._config.get("tracing.enabled", True))
        tracing_output_path = self._config.get("tracing.output_path", "var/tracing/traces.jsonl")
        payload_redaction_enabled = bool(
            self._config.get(
                "tracing.payload_redaction_enabled",
                not bool(self._config.get("tracing.capture_payloads", False)),
            )
        )
        max_content_length = self._config.positive_int("tracing.max_content_length", default=1000)
        return Tracer(
            enabled=tracing_enabled,
            output_path=tracing_output_path,
            payload_redaction_enabled=payload_redaction_enabled,
            max_content_length=max_content_length,
        )

    def build_pipeline_driver(self, thread: PipelineThread, event_bus: EventBus) -> PipelineDriver:
        return PipelineDriver(
            event_bus=event_bus,
            thread=thread,
        )

    def build_pipeline(self, event_bus: EventBus) -> Pipeline:
        return Pipeline(config=self._config, agent_factory=self, logger=self._logger, event_bus=event_bus)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _primary_provider_name(self) -> str:
        chain = self._config.get("llm.priority_chain", ["deepseek"])
        if isinstance(chain, list) and chain:
            return str(chain[0])
        return "deepseek"

    def _build_sqlite_databases(self) -> dict[str, str]:
        sqlite_config = self._config.get("storage.sqlite", {})
        if not isinstance(sqlite_config, dict):
            sqlite_config = {}
        configured = sqlite_config.get("allowed_databases") or sqlite_config.get("databases")
        databases: dict[str, str] = {}
        if isinstance(configured, dict):
            for name, path in configured.items():
                if str(name).strip() and str(path).strip():
                    databases[str(name).strip()] = str(path).strip()
        fallback_path = str(sqlite_config.get("path", "")).strip()
        if fallback_path:
            databases.setdefault(self._derive_sqlite_alias(fallback_path), fallback_path)
        if not databases:
            databases["local_storage"] = "var/storage/nanoagent_local_storage.db"
        return databases

    def _build_mysql_databases(self) -> list[str]:
        mysql_config = self._config.get("storage.mysql", {})
        if not isinstance(mysql_config, dict):
            mysql_config = {}
        configured = mysql_config.get("allowed_databases")
        if isinstance(configured, list):
            dbs = [str(d).strip() for d in configured if str(d).strip()]
            if dbs:
                return dbs
        fallback = str(mysql_config.get("database", "")).strip()
        if fallback:
            return [fallback]
        raise build_pipeline_error(
            STORAGE_CONFIG_ERROR,
            "MySQL storage requires `storage.mysql.allowed_databases` or `storage.mysql.database`.",
        )

    def _build_chromadb_collections(self) -> list[str]:
        chromadb_config = self._config.get("storage.chromadb", {})
        if not isinstance(chromadb_config, dict):
            chromadb_config = {}
        configured = chromadb_config.get("allowed_collections") or chromadb_config.get("collections")
        if isinstance(configured, list):
            cols = [str(c).strip() for c in configured if str(c).strip()]
            if cols:
                return cols
        fallback = str(chromadb_config.get("collection_name", "")).strip()
        if fallback:
            return [fallback]
        return ["agent_documents"]

    @staticmethod
    def _derive_sqlite_alias(path_value: str) -> str:
        path = str(path_value).strip()
        if path.endswith(".db"):
            path = path[:-3]
        return path.rsplit("/", 1)[-1] or "sqlite"

    def _register_storage_tools(self, tool_registry, storage_registry: StorageRegistry) -> None:
        for backend_name in storage_registry.list_backends():
            storage = storage_registry.get(backend_name)
            if backend_name in {"sqlite", "mysql"}:
                resources = ", ".join(storage.list_resources()) or "<none>"
                tool_registry.register(SQLSchemaTool(
                    name=build_sql_schema_tool_name(backend_name),
                    description=build_sql_schema_tool_description(backend_name, resources),
                    storage=storage,
                    backend_name=backend_name,
                ))
                tool_registry.register(SQLQueryTool(
                    name=build_sql_query_tool_name(backend_name),
                    description=build_sql_query_tool_description(backend_name, resources),
                    storage=storage,
                    backend_name=backend_name,
                ))
            elif backend_name == "chromadb":
                resources = ", ".join(storage.list_resources()) or "<none>"
                tool_registry.register(VectorSchemaTool(
                    name=build_vector_schema_tool_name(backend_name),
                    description=build_vector_schema_tool_description(backend_name, resources),
                    storage=storage,
                    backend_name=backend_name,
                ))
                tool_registry.register(VectorSearchTool(
                    name=build_vector_search_tool_name(backend_name),
                    description=build_vector_search_tool_description(backend_name, resources),
                    storage=storage,
                    backend_name=backend_name,
                ))
