from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent.events.events import UserClarificationRequested
from infra.observability.tracing.tracer import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.errors import JSON_LOAD_ERROR, TASK_ANALYSIS_LOW_CONFIDENCE, build_json_error, build_pipeline_error
from schemas.ids import TaskId, UserId
from schemas.task import (
    COMPLEXITY_MAP,
    RelatedKnowledgeEntry,
    RelatedUserPreferenceEntry,
    Task,
    TaskAnalysis,
    TaskComplexity,
    TaskConstraint,
    TaskEntity,
    TaskStatus,
    ToolMatch,
    RiskItem,
)
from schemas.types import LLMMessage, UnifiedLLMRequest
from utils.log.log import Logger, zap
from utils.time.time import now

if TYPE_CHECKING:
    from agent.application.driver import PipelineDriver
    from agent.models.knowledge.knowledge_loader import KnowledgeLoader
    from agent.models.personality.user_preference import PersonalityManager
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway
    from schemas.event_bus import EventBus
    from tools.tool_registry import ToolRegistry

class Analyzer:
    """Extracts task features via LLM and enriches the Task with knowledge and preferences."""

    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer, event_bus: EventBus, renderer: PromptRenderer | None = None):
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()
        self._driver: PipelineDriver | None = None

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver

    def analyze(
        self,
        task_id: TaskId,
        user_id: UserId,
        task_description: str,
        llm_gateway: LLMGateway,
        knowledge_loader: KnowledgeLoader,
        personality_manager: PersonalityManager,
        tool_registry: ToolRegistry,
    ) -> Task:
        tool_schemas = tool_registry.get_tool_schemas()
        self._logger.info(
            "Task analysis started",
            zap.any("task_id", task_id),
            zap.any("user_id", user_id),
            zap.any("task_length", len(task_description)),
            zap.any("tool_schema_count", len(tool_schemas)),
        )

        with self._tracer.start_span(
            "analyzer.load_user_preference",
            "analysis",
            {"task_id": task_id},
        ):
            user_preference_context = self._load_user_preference(personality_manager)

        with self._tracer.start_span(
            "analyzer.extract_analysis",
            "analysis",
            {
                "task_id": task_id,
                "tool_schema_count": len(tool_schemas),
                "user_preference_context_length": len(user_preference_context),
            },
        ) as span:
            analysis = self._extract_analysis(
                task_description, tool_schemas,
                user_preference_context,
                llm_gateway,
            )
            span.add_attributes(
                {
                    "task_type": analysis.task_type,
                    "confidence": analysis.confidence,
                    "estimated_steps": analysis.estimated_steps,
                    "tool_matches": len(analysis.tool_matches),
                    "entities": len(analysis.entities),
                }
            )

        if (0.6 > analysis.confidence) and (0 < len(analysis.implicit_needs)) and (self._driver is not None): #TODO 0.6放到配置config.json中，新的顶级分节项analyzer
            self._logger.info(
                "Task analysis requires clarification",
                zap.any("task_id", task_id),
                zap.any("confidence", analysis.confidence),
                zap.any("question_count", len(analysis.implicit_needs)),
            )
            with self._tracer.start_span(
                "analyzer.clarification",
                "analysis",
                {
                    "task_id": task_id,
                    "confidence": analysis.confidence,
                    "question_count": len(analysis.implicit_needs),
                },
            ) as span:
                analysis = self._run_clarification(
                    task_id, analysis, task_description,
                    tool_schemas, user_preference_context,
                    llm_gateway,
                )
                span.add_attributes({"confidence_after": analysis.confidence})
        else:
            raise build_pipeline_error(
                TASK_ANALYSIS_LOW_CONFIDENCE,
                f"Task analysis confidence too low ({analysis.confidence:.2f}) after clarification",
            )

        partial_task = self._build_task(task_id, user_id, task_description, analysis, [], [])

        with self._tracer.start_span("analyzer.query_user_preferences", "analysis", {"task_id": task_id}) as span:
            raw_preferences = personality_manager.query_related_user_preference(partial_task, llm_gateway) or []
            span.add_attributes({"raw_preference_count": len(raw_preferences)})
        with self._tracer.start_span("analyzer.query_knowledge", "analysis", {"task_id": task_id}) as span:
            raw_knowledge = knowledge_loader.query_related_knowledge(partial_task, llm_gateway) or []
            span.add_attributes({"raw_knowledge_count": len(raw_knowledge)})

        min_pref_conf: float = self._config.get("analyzer.min_confidence.user_preference", 0.6) if self._config else 0.6
        min_know_conf: float = self._config.get("analyzer.min_confidence.knowledge_entry", 0.6) if self._config else 0.6

        related_preferences = [
            RelatedUserPreferenceEntry(entry=e, confidence=self._score_preference_entry(e, analysis))
            for e in raw_preferences
            if self._score_preference_entry(e, analysis) >= min_pref_conf
        ]
        related_knowledge = [
            RelatedKnowledgeEntry(entry=e, confidence=self._score_knowledge_entry(e, analysis))
            for e in raw_knowledge
            if self._score_knowledge_entry(e, analysis) >= min_know_conf
        ]

        task = self._build_task(task_id, user_id, task_description, analysis, related_preferences, related_knowledge)

        self._logger.info(
            "Task analysis complete",
            zap.any("task_id", task.id),
            zap.any("task_type", task.task_type),
            zap.any("confidence", task.confidence),
            zap.any("complexity_level", task.complexity.level),
            zap.any("tool_matches", len(task.tool_matches)),
            zap.any("entities", len(task.entities)),
            zap.any("constraints", len(task.action_constraints)),
            zap.any("preference_count", len(related_preferences)),
            zap.any("knowledge_count", len(related_knowledge)),
        )
        return task

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_analysis(
        self,
        task_description: str,
        tool_schemas: list[dict],
        preference_context: str,
        llm_gateway: LLMGateway,
        clarification_context: str = "",
    ) -> TaskAnalysis:
        tool_schemas_simplified = [
            {"name": s["function"]["name"],
             "description": s["function"].get("description", ""),
             "parameters": s["function"].get("parameters", {})}
            for s in tool_schemas
        ]
        prompt = self._renderer.render("analyzer/user_prompt.j2", {
            "task_description": task_description,
            "tool_schemas": tool_schemas_simplified,
            "clarification_context": clarification_context,
            "preference_context": preference_context,
        })
        system_prompt = self._renderer.render("analyzer/system.j2", {})

        provider = self._config.get("llm.analyzer_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "Calling LLM for task analysis",
            zap.any("provider", provider),
            zap.any("prompt_length", len(prompt)),
            zap.any("tool_schema_count", len(tool_schemas)),
            zap.any("has_clarification_context", bool(clarification_context)),
        )
        response = llm_gateway.generate(
            UnifiedLLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.0,
            ),
            provider,
        )
        content = response.assistant_message.content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
            content = "\n".join(inner)
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self._logger.error(
                "Failed to parse task analysis response",
                zap.any("error", exc),
                zap.any("content_length", len(content)),
            )
            raise build_json_error(code=JSON_LOAD_ERROR, message=f"Failed to parse LLM analysis response: {exc}")
        parsed = self._parse_analysis(raw)
        self._logger.info(
            "Task analysis response parsed",
            zap.any("task_type", parsed.task_type),
            zap.any("confidence", parsed.confidence),
            zap.any("estimated_steps", parsed.estimated_steps),
            zap.any("tool_matches", len(parsed.tool_matches)),
        )
        return parsed

    def _parse_analysis(self, raw: dict) -> TaskAnalysis:
        entities = [
            TaskEntity(
                type=e.get("type", ""),
                value=e.get("value", ""),
                raw=e.get("raw", ""),
                normalized=bool(e.get("normalized", False)),
            )
            for e in raw.get("entities", [])
            if isinstance(e, dict)
        ]
        raw_constraints = raw.get("action_constraints", raw.get("constraints", []))
        constraints = [
            TaskConstraint(
                description=c.get("description", ""),
                strict=bool(c.get("strict", False)),
                source=c.get("source", "implicit"),
            )
            for c in raw_constraints
            if isinstance(c, dict)
        ]
        tool_matches = [
            ToolMatch(
                tool_name=m.get("tool_name", ""),
                match_score=float(m.get("match_score", 0.0)),
                reasoning=m.get("reasoning", ""),
            )
            for m in raw.get("tool_matches", [])
            if isinstance(m, dict)
        ]
        risks = [
            RiskItem(
                category=r.get("category", ""),
                description=r.get("description", ""),
                severity=r.get("severity", "low"),
            )
            for r in raw.get("risks", [])
            if isinstance(r, dict)
        ]
        return TaskAnalysis(
            task_type=str(raw.get("task_type", "")),
            task_goal=str(raw.get("task_goal", "")),
            intent=str(raw.get("intent", "")),
            entities=entities,
            action_constraints=constraints,
            tool_matches=tool_matches,
            complexity_level=int(raw.get("complexity_level", 2)),
            estimated_steps=int(raw.get("estimated_steps", 1)),
            output_constraints=str(raw.get("output_constraints", "")),
            notes=str(raw.get("notes", "")),
            implicit_needs=list(raw.get("implicit_needs", [])),
            risks=risks,
            confidence=float(raw.get("confidence", 1.0)),
        )

    def _run_clarification(
        self,
        task_id: TaskId,
        analysis: TaskAnalysis,
        task_description: str,
        tool_schemas: list[dict],
        preference_context: str,
        llm_gateway: LLMGateway,
    ) -> TaskAnalysis:
        combined_question = "\n".join(
            f"{i}. {q}" for i, q in enumerate(analysis.implicit_needs, start=1)#TODO analysis.implicit_needs没检查就找用户 
        )
        self._logger.info(
            "Publishing analysis clarification request",
            zap.any("task_id", task_id),
            zap.any("question", combined_question),
        )
        self._event_bus.publish(UserClarificationRequested(
            task_id=task_id,
            order="1",
            question=combined_question,
            content=combined_question,
        ))
        cmd = self._driver.loop_user_messages(timeout=300.0)
        clarification = cmd.content if cmd is not None else ""
        self._logger.info(
            "Analysis clarification received",
            zap.any("task_id", task_id),
            zap.any("has_clarification", bool(clarification)),
            zap.any("clarification_length", len(clarification)),
        )

        clarification_context = (
            f"\nClarification questions asked:\n{combined_question}"
            f"\nUser's answer: {clarification}"
        )
        analysis = self._extract_analysis(
            task_description, tool_schemas,
            preference_context,
            llm_gateway, clarification_context,
        )

        if analysis.confidence < 0.6:
            self._logger.error(
                "Task analysis confidence remains too low after clarification",
                zap.any("task_id", task_id),
                zap.any("confidence", analysis.confidence),
            )
            raise build_pipeline_error(
                TASK_ANALYSIS_LOW_CONFIDENCE,
                f"Task analysis confidence too low ({analysis.confidence:.2f}) after clarification",
            )

        return analysis

    def _build_task(
        self,
        task_id: TaskId,
        user_id: UserId,
        task_description: str,
        analysis: TaskAnalysis,
        related_preferences: list[RelatedUserPreferenceEntry],
        related_knowledge: list[RelatedKnowledgeEntry],
    ) -> Task:
        min_tool_conf: float = self._config.get("analyzer.min_confidence.tool_match", 0.5) if self._config else 0.5
        tool_matches = [m for m in analysis.tool_matches if m.match_score >= min_tool_conf]
        required_tools = [m.tool_name for m in tool_matches]
        complexity = COMPLEXITY_MAP.get(analysis.complexity_level, TaskComplexity(level=analysis.complexity_level))
        return Task(
            id=task_id,
            user_id=user_id,
            description=task_description,
            created_at=now(),
            status=TaskStatus.CREATED,
            task_type=analysis.task_type,
            task_goal=analysis.task_goal,
            intent=analysis.intent,
            complexity=complexity,
            required_tools=required_tools,
            tool_matches=tool_matches,
            output_constraints=analysis.output_constraints,
            notes=analysis.notes,
            entities=analysis.entities,
            action_constraints=analysis.action_constraints,
            risks=analysis.risks,
            confidence=analysis.confidence,
            estimated_steps=analysis.estimated_steps,
            related_user_preference_entries=related_preferences,
            related_knowledge_entries=related_knowledge,
        )

    def _load_user_preference(self, personality_manager: PersonalityManager) -> str:
        try:
            entries = personality_manager.load_all_preferences()
            if not entries:
                return ""
            lines = [f"- {e.content}" for e in entries[:5]]
            return "\n".join(lines)
        except Exception:
            return ""

    def _score_preference_entry(self, entry, analysis: TaskAnalysis) -> float:
        task_tokens = set(analysis.intent.lower().split())
        task_tokens.update(analysis.task_type.lower().split())
        task_tokens.update(analysis.task_goal.lower().split())
        task_tokens.update(analysis.notes.lower().split())
        task_tokens.update(e.value.lower() for e in analysis.entities)
        entry_tokens = set(kw.lower() for kw in entry.keywords)
        entry_tokens.update(entry.content.lower().split()[:20])
        overlap = len(task_tokens & entry_tokens)
        if overlap == 0:
            return 0.5
        return min(0.5 + overlap * 0.1, 1.0)

    def _score_knowledge_entry(self, entry, analysis: TaskAnalysis) -> float:
        task_tokens = set(analysis.intent.lower().split())
        task_tokens.update(analysis.task_type.lower().split())
        task_tokens.update(analysis.task_goal.lower().split())
        task_tokens.update(analysis.notes.lower().split())
        task_tokens.update(e.value.lower() for e in analysis.entities)
        entry_tokens = set(tag.lower() for tag in entry.tags)
        entry_tokens.update(entry.title.lower().split())
        overlap = len(task_tokens & entry_tokens)
        if overlap == 0:
            return 0.5
        return min(0.5 + overlap * 0.1, 1.0)

