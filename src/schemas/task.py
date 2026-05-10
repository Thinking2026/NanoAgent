from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.time.time import now as _time_now

from schemas.ids import (
    PlanId,
    PlanStepId,
    TaskId,
    UserId,
)
from schemas.types import LLMMessage, LLMResponse, ToolCall 

class StageStatus(str, Enum):
    RUNNING      = "RUNNING"
    COMPLETED    = "COMPLETED"
    PAUSED       = "PAUSED"
    SUCCESS      = "SUCCESS"
    INTERRUPTED  = "INTERRUPTED"
    FAILED       = "FAILED"

class PlanUpdateTrigger(str, Enum):
    STAGE_EVAL_FAILED    = "STAGE_EVAL_FAILED"
    USER_GUIDANCE        = "USER_GUIDANCE"
    QUALITY_CHECK_FAILED = "QUALITY_CHECK_FAILED"

class EvaluationTarget(str, Enum):
    TASK_RESULT  = "TASK_RESULT"
    STAGE_RESULT = "STAGE_RESULT"
    PLAN         = "PLAN"

@dataclass(frozen=True)
class EvaluationReport:
    target_type: EvaluationTarget   # "task" | "stage" | "plan"
    target_id: str
    passed: bool
    feedback: str
    evaluated_at: datetime
    need_user_clarification: bool = field(default=False)
    clarification_question: str = field(default="")

@dataclass(frozen=True)
class LLMProviderCapabilities:
    name: str
    cognitive_complexity: list[str] #认知复杂度
    best_scenarios: list[str]
    top_strengths: list[str]
    cost_tier: str
    latency_tier: str
    context_size: int

@dataclass(frozen=True)
class ModelRoutingDecision:
    primary: str
    fallbacks: list[str] = field(default_factory=list)

@dataclass(slots=True)
class UserPreferenceEntry:
    user_id:    str
    keywords:   list[str]
    content:    str
    created_at: datetime = field(default_factory=_time_now)

@dataclass(slots=True)
class KnowledgeEntry:
    entry_id:   str
    title:      str
    tags:       list[str]
    content:    str
    created_at: datetime = field(default_factory=_time_now)

@dataclass(frozen=True)
class RelatedUserPreferenceEntry:
    entry: UserPreferenceEntry
    confidence: float  # 0-1

@dataclass(frozen=True)
class RelatedKnowledgeEntry:
    entry: KnowledgeEntry
    confidence: float  # 0-1

class NextDecisionType(str, Enum):
    TOOL_CALL            = "TOOL_CALL"
    FINAL_ANSWER         = "FINAL_ANSWER"
    CONTINUE             = "CONTINUE"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"
    PAUSED               = "PAUSED"

@dataclass(frozen=True)
class NextDecision:
    decision_type: NextDecisionType
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: LLMMessage | None = None
    raw_response: LLMResponse | None = None
    message: str = ""
    answer: str = ""

@dataclass(frozen=True)
class PlanStep:
    id: PlanStepId
    goal: str
    description: str
    order: int
    key_results: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class Plan:
    id: PlanId
    task_id: TaskId
    step_list: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=_time_now)

    @property
    def step_count(self) -> int:
        return len(self.step_list)

class PlanChangeReason(str, Enum):
    PLAN_EVALUATE_FAILED     = "PLAN_EVALUATE_FAILED"
    TASK_RESULT_EVALUATED    = "TASK_RESULT_EVALUATED"
    STAGE_RESULT_EVALUATED   = "STAGE_RESULT_EVALUATED"

@dataclass(frozen=True)
class PlanVersion:
    plan: Plan
    version: int
    change_reason: PlanChangeReason
    changed_at: datetime = field(default_factory=_time_now)


@dataclass(frozen=True)
class TaskEntity:
    type: str       # "stock_code" | "date" | "filename" | "url" | "number" | "person" | "location" | "query_term"
    value: str      # normalized value (ISO date, uppercase ticker, etc.)
    raw: str        # original text as user wrote it
    normalized: bool = False  # True if value differs from raw

@dataclass(frozen=True)
class TaskConstraint:
    description: str
    strict: bool = False    # True = hard constraint, False = soft preference
    source: str = "implicit"  # "explicit" | "implicit"

@dataclass(frozen=True)
class ToolMatch:
    tool_name: str
    match_score: float                      # 0.5–1.0 (< 0.5 excluded)
    required_params: list[str] = field(default_factory=list)  # only params needed for this task
    reasoning: str = ""                     # one sentence why this tool is needed

@dataclass(frozen=True)
class RiskItem:
    category: str   # "data_staleness" | "cost_overflow" | "ambiguity" | "missing_tool" | "scope_creep"
    description: str
    severity: str = "low"  # "low" | "medium" | "high"

@dataclass(frozen=True)
class TaskAnalysis:
    task_type: str
    task_goal: str
    intent: str
    entities: list[TaskEntity]
    action_constraints: list[TaskConstraint]
    tool_matches: list[ToolMatch]
    complexity_level: int
    estimated_steps: int
    reasoning_depth: str
    output_constraints: str
    notes: str
    implicit_needs: list[str]  # clarification questions when confidence < 0.6
    risks: list[RiskItem]
    confidence: float

@dataclass(frozen=True)
class TaskComplexity:
    level: int
    features: list[str] = field(default_factory=list) #这个难度的任务有什么特征
    use_cases: list[str] = field(default_factory=list) #这个难度一般是什么任务

L1 = TaskComplexity(
    level=1,
    features=["单步", "模板化", "低幻觉要求"],
    use_cases=["寒暄", "格式化", "标签分类", "简单提取"],
)

L2 = TaskComplexity(
    level=2,
    features=["单步推理", "常识", "短上下文"],
    use_cases=["客服问答", "邮件起草", "基础翻译"],
)

L3 = TaskComplexity(
    level=3,
    features=["多步推理", "代码", "分析"],
    use_cases=["代码审查", "数据分析", "报告生成"],
)

L4 = TaskComplexity(
    level=4,
    features=["深度推理", "创意", "长链思维"],
    use_cases=["架构设计", "数学证明", "策略规划"],
)

class ReasoningType(str, Enum):
    SINGLE_STEP          = "single-step reasoning"
    MULTI_STEP           = "multi-step reasoning"

class TaskStatus(str, Enum):
    CREATED       = "CREATED"
    RUNNING       = "RUNNING"
    PAUSED        = "PAUSED"
    CANCELLED     = "CANCELLED"
    SUCCESS       = "SUCCESS"
    FAILED        = "FAILED"

@dataclass(frozen=True)
class Task:
    id: TaskId
    user_id: UserId
    description: str
    created_at: datetime
    status: TaskStatus = TaskStatus.CREATED
    task_type: str = ""
    intent: str = ""
    complexity: TaskComplexity = field(default_factory=lambda: TaskComplexity(level=2))
    required_tools: list[str] = field(default_factory=list)
    reasoning_depth: ReasoningType = ReasoningType.SINGLE_STEP
    output_constraints: str = ""
    notes: str = ""
    related_user_preference_entries: list[RelatedUserPreferenceEntry] = field(default_factory=list)
    related_knowledge_entries: list[RelatedKnowledgeEntry] = field(default_factory=list)
    task_goal: str = ""
    entities: list[TaskEntity] = field(default_factory=list)
    action_constraints: list[TaskConstraint] = field(default_factory=list)
    tool_matches: list[ToolMatch] = field(default_factory=list)
    risks: list[RiskItem] = field(default_factory=list)
    confidence: float = 1.0
    estimated_steps: int = 0

@dataclass(frozen=True)
class TaskResult:
    task_id: TaskId
    succeeded: bool
    result: str
    error_reason: str
    delivered_at: datetime

__all__ = [
    "StageStatus",
    "CheckpointEntry",
    "EvaluationRecord",
    "KnowledgeEntryStatus",
    "KnowledgeExtracted",
    "KnowledgeIndexed",
    "ProviderCapabilities",
    "ModelRoutingDecision",
    "RelatedPreferenceEntry",
    "RelatedKnowledgeEntry",
    "PlanStep",
    "Plan",
    "TaskFeature",
    "PlanUpdateTrigger",
    "NextDecisionType",
    "NextDecision",
    "Task",
    "TaskResult",
    "PlanVersion",
]
