from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.events.events import UserClarificationRequested
from infra.observability.tracing.tracer import Tracer
from schemas.errors import JSON_LOAD_ERROR, TASK_ANALYSIS_LOW_CONFIDENCE, build_json_error, build_pipeline_error
from schemas.ids import TaskId, UserId
from schemas.task import (
    RelatedKnowledgeEntry,
    RelatedUserPreferenceEntry,
    ReasoningType,
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

_ANALYZE_SYSTEM_PROMPT = """\
You are an expert task analysis engine. Given a user task description, available tools, \
user preferences, and domain knowledge, produce a structured JSON analysis.

## Output Schema
Return a single JSON object with exactly these keys:

{
  "task_type": string,
  "task_goal": string,
  "intent": string,
  "entities": [{"type": string, "value": string, "raw": string, "normalized": boolean}],
  "action_constraints": [{"description": string, "strict": boolean, "source": string}],
  "tool_matches": [{"tool_name": string, "match_score": float, "required_params": [string], "reasoning": string}],
  "complexity_level": integer,
  "estimated_steps": integer,
  "reasoning_depth": string,
  "output_constraints": string,
  "notes": string,
  "implicit_needs": [string],
  "risks": [{"category": string, "description": string, "severity": string}],
  "confidence": float
}

## Field Definitions

task_type: short category label, such as: "data_analysis" | "code_generation" | "search" | \
"qa" | "file_operation" | "calculation" | "copywriting" | "technical writing"

task_goal: the top-level user goal if clearly inferable (e.g. "understand stock performance"), \
empty string "" if the goal cannot be confidently inferred

intent: one sentence describing the clarified intent behind the request (the TRUE goal, \
not just the surface action)

entities: all entities that affect tool calls — extract every stock code, date, filename, \
URL, number, person name, location, or search query term
  - type: "stock_code" | "date" | "filename" | "url" | "number" | "person" | "location" | "query_term"
  - value: normalized value
  - raw: original text as user wrote it
  - normalized: true if value differs from raw

action_constraints: explicit constraints (user stated) and implicit constraints (reasonably inferred)
  - strict: true = hard constraint that MUST be satisfied; false = soft preference
  - source: "explicit" (user stated it) | "implicit" (reasonably inferred)
  Examples of implicit constraints: querying stock prices implies need for real-time data; \
  generating a report implies need for structured formatting

tool_matches: only include tools with match_score >= 0.5
  - match_score: 0.9-1.0 = tool fully covers the need; 0.7-0.89 = covers core need but needs \
    combination or param conversion; 0.5-0.69 = partial/auxiliary; < 0.5 = exclude
  - required_params: ONLY the parameter names this specific task will use (not all tool params)
  - reasoning: one sentence explaining why this tool is needed

complexity_level: 1-4 (see "Complexity Mapping")
estimated_steps: number of execution steps in the plan (not analysis steps)
reasoning_depth: "single-step reasoning" | "multi-step reasoning"
output_constraints: format/length/language constraints on the output, "" if none
notes: any other relevant observations, "" if none

implicit_needs: list of clarification questions when confidence < 0.6, else []
  These are questions the agent needs answered to proceed confidently.

risks: risk items to flag
  - category: "data_staleness" | "cost_overflow" | "ambiguity" | "missing_tool" | "scope_creep"
  - severity: "low" | "medium" | "high"

confidence: 0.0-1.0 (see scoring criteria below)

## Entity Normalization Rules
- Dates: convert relative ("yesterday", "last week") to ISO 8601 (YYYY-MM-DD) using today as anchor. \
  For vague relative times ("recently", "lately"), use a structured description like "relative:recent" \
  as the value and set normalized=false.
- Stock codes: uppercase, remove exchange prefix if ambiguous (e.g. "apple stock" → "AAPL")
- Numbers: keep as string, strip currency symbols into a separate entity if present
- URLs: normalize scheme to lowercase

## Tool Matching Instructions
Read each tool's description and parameter schema carefully before scoring.
Base the score on how well the tool covers the task's core need — not just the tool name.
List in required_params ONLY the parameter names this specific task will use.

## Complexity Mapping
L1 (level=1): single-step, template-based, low hallucination risk — greetings, formatting, tagging, simple extraction
L2 (level=2): single-step reasoning, common sense, short context — customer service, email drafting, basic translation
L3 (level=3): multi-step reasoning, code, analysis — code review, data analysis, report generation
L4 (level=4): deep reasoning, creativity, long chain-of-thought — architecture design, math proofs, strategy planning

## Confidence Scoring Criteria
Start at 1.0 and subtract:
  -0.2 if task_type is ambiguous (multiple plausible interpretations)
  -0.15 if key entities are missing or unclear (e.g. "analyze the data" with no file specified)
  -0.1 if required tools are uncertain
  -0.1 if output format is unspecified but matters for this task
  -0.05 per additional ambiguity factor

## Confidence Thresholds
  confidence < 0.6: set task_type = "clarification_needed", list questions in implicit_needs
  confidence < 0.4: set task_type = "rejection_required"

Respond with only valid JSON. No markdown fences."""


class Analyzer:
    """Extracts task features via LLM and enriches the Task with knowledge and preferences."""

    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer, event_bus: EventBus):
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._driver: PipelineDriver | None = None

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver

    def analyze(
        self,
        user_id: UserId,
        task_description: str,
        llm_gateway: LLMGateway,
        knowledge_loader: KnowledgeLoader,
        personality_manager: PersonalityManager,
        tool_registry: ToolRegistry,
    ) -> Task:
        task_id = TaskId(str(uuid4()))
        tool_schemas = tool_registry.get_tool_schemas()

        preference_context = self._build_preference_context(personality_manager)
        knowledge_context = self._build_knowledge_context(knowledge_loader)

        analysis = self._extract_analysis(
            task_description, tool_schemas,
            preference_context, knowledge_context,
            llm_gateway,
        )

        if analysis.confidence < 0.6 and self._driver is not None:
            analysis = self._run_clarification(
                task_id, analysis, task_description,
                tool_schemas, preference_context, knowledge_context,
                llm_gateway,
            )

        partial_task = self._build_task(task_id, user_id, task_description, analysis, [], [])

        raw_preferences = personality_manager.query_related_user_preference(partial_task, llm_gateway) or []
        raw_knowledge = knowledge_loader.query_related_knowledge(partial_task, llm_gateway) or []

        related_preferences = [
            RelatedUserPreferenceEntry(entry=e, confidence=self._score_preference_entry(e, analysis))
            for e in raw_preferences
        ]
        related_knowledge = [
            RelatedKnowledgeEntry(entry=e, confidence=self._score_knowledge_entry(e, analysis))
            for e in raw_knowledge
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
        knowledge_context: str,
        llm_gateway: LLMGateway,
        clarification_context: str = "",
    ) -> TaskAnalysis:
        tools_block = json.dumps(
            [{"name": s["function"]["name"],
              "description": s["function"].get("description", ""),
              "parameters": s["function"].get("parameters", {})}
             for s in tool_schemas],
            ensure_ascii=False,
            indent=2,
        )
        parts = [f"Task description:\n{task_description}"]
        if clarification_context:
            parts.append(clarification_context)
        parts.append(f"\nAvailable tools (name, description, parameters):\n{tools_block}")
        if preference_context:
            parts.append(f"\nUser preferences (for context):\n{preference_context}")
        if knowledge_context:
            parts.append(f"\nDomain knowledge (for context):\n{knowledge_context}")
        prompt = "\n".join(parts)

        provider = self._config.get("llm.analyzer_provider", ["deepseek"])[0] if self._config else "deepseek"
        response = llm_gateway.generate(
            UnifiedLLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                system_prompt=_ANALYZE_SYSTEM_PROMPT,
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
            raise build_json_error(code=JSON_LOAD_ERROR, message=f"Failed to parse LLM analysis response: {exc}")
        return self._parse_analysis(raw)

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
        constraints = [
            TaskConstraint(
                description=c.get("description", ""),
                strict=bool(c.get("strict", False)),
                source=c.get("source", "implicit"),
            )
            for c in raw.get("constraints", [])
            if isinstance(c, dict)
        ]
        tool_matches = [
            ToolMatch(
                tool_name=m.get("tool_name", ""),
                match_score=float(m.get("match_score", 0.0)),
                required_params=list(m.get("required_params", [])),
                reasoning=m.get("reasoning", ""),
            )
            for m in raw.get("tool_matches", [])
            if isinstance(m, dict) and float(m.get("match_score", 0.0)) >= 0.5
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
            reasoning_depth=str(raw.get("reasoning_depth", "single-step reasoning")),
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
        knowledge_context: str,
        llm_gateway: LLMGateway,
    ) -> TaskAnalysis:
        combined_question = "\n".join(
            f"{i}. {q}" for i, q in enumerate(analysis.implicit_needs, start=1)
        )
        self._event_bus.publish(UserClarificationRequested(
            task_id=task_id,
            order="1",
            question=combined_question,
            content=combined_question,
        ))
        cmd = self._driver.loop_user_messages(timeout=300.0)
        clarification = cmd.content if cmd is not None else ""

        clarification_context = (
            f"\nClarification questions asked:\n{combined_question}"
            f"\nUser's answer: {clarification}"
        )
        analysis = self._extract_analysis(
            task_description, tool_schemas,
            preference_context, knowledge_context,
            llm_gateway, clarification_context,
        )

        if analysis.confidence < 0.6:
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
        required_tools = [m.tool_name for m in analysis.tool_matches]
        return Task(
            id=task_id,
            user_id=user_id,
            description=task_description,
            created_at=now(),
            status=TaskStatus.CREATED,
            task_type=analysis.task_type,
            task_goal=analysis.task_goal,
            intent=analysis.intent,
            complexity=TaskComplexity(
                level=analysis.complexity_level,
            ),
            required_tools=required_tools,
            tool_matches=analysis.tool_matches,
            reasoning_depth=_parse_reasoning_depth(analysis.reasoning_depth),
            output_constraints=analysis.output_constraints,
            notes=analysis.notes,
            entities=analysis.entities,
            action_constraints=analysis.constraints,
            risks=analysis.risks,
            confidence=analysis.confidence,
            estimated_steps=analysis.estimated_steps,
            related_user_preference_entries=related_preferences,
            related_knowledge_entries=related_knowledge,
        )

    def _build_preference_context(self, personality_manager: PersonalityManager) -> str:
        try:
            entries = personality_manager.load_all_preferences()
            if not entries:
                return ""
            lines = [f"- {e.content}" for e in entries[:5]]
            return "\n".join(lines)
        except Exception:
            return ""

    def _build_knowledge_context(self, knowledge_loader: KnowledgeLoader) -> str:
        try:
            entries = knowledge_loader.load_all_entries()
            if not entries:
                return ""
            lines = [f"- [{e.title}] {e.content}" for e in entries[:5]]
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


def _parse_reasoning_depth(value: str) -> ReasoningType:
    if value == ReasoningType.MULTI_STEP.value:
        return ReasoningType.MULTI_STEP
    return ReasoningType.SINGLE_STEP
