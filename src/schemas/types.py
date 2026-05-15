from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from schemas.errors import PipelineError
from schemas.ids import CheckpointId, PlanStepId, StageId, TaskId, UserId
from utils.time.time import now as _time_now

#本文件引入的类型只能依赖内置类型或者文件中已经引入的类型，不能依赖其他文件中定义的类型，否则会导致循环依赖问题

LLMRole = Literal["user", "assistant", "tool"]
ALL_ROLES = ("system", "user", "assistant", "tool")
DEFAULT_CLARIFICATION = "Try to solve the problem on your own based on your understanding"

@dataclass(slots=True)
class ContextWindow:
    """Snapshot of the conversation context assembled by ContextManager."""
    messages: list[LLMMessage]
    tool_schemas: list[dict[str, Any]] | None = None

@dataclass(slots=True)
class UnifiedLLMRequest:
    messages: list[LLMMessage]
    system_prompt: str | None = None
    tool_schemas: list[dict[str, Any]] | None = None
    max_tokens: int = 393216
    temperature: float = 0.0
    model_override: str | None = None
    json_mode: bool = False
    json_required_keys: list[str] | None = None

@dataclass(slots=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass(slots=True)
class LLMResponse:
    assistant_message: LLMMessage
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: LLMUsage | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class LLMMessage:
    """留作未来扩展，从context_message应该经过比如渲染引擎转换变成LLMMessage，然后转成UnifiedLLMRequest发送给LLM"""
    role: LLMRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    llm_raw_tool_call_id: str | None = None

@dataclass(slots=True)
class ToolResult:
    output: str
    llm_raw_tool_call_id: str | None = None
    success: bool = True
    error: PipelineError | None = None


@dataclass(slots=True)
class SQLQueryRequest:
    statement: str
    database: str | None = None
    params: list[Any] | tuple[Any, ...] | dict[str, Any] | None = None
    max_rows: int = 100


@dataclass(slots=True)
class VectorSearchRequest:
    query: str
    collection: str | None = None
    top_k: int = 3
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KeyValueGetRequest:
    key: str


@dataclass(slots=True)
class KeyValueSetRequest:
    key: str
    value: Any
    ttl_seconds: int | None = None


@dataclass(slots=True)
class ClientMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentExecutionResult:
    user_messages: list[ClientMessage] = field(default_factory=list)
    error: PipelineError | None = None
    task_completed: bool = False

@dataclass(slots=True)
class RoleBudget:
    role: str
    ratio: float
    token_budget: int

@dataclass(slots=True)
class BudgetResult:
    strategy: str
    total_budget: int
    reserve_ratio: float
    reserved_tokens: int        # for LLM response + future summary calls
    available_tokens: int       # total_budget - reserved_tokens
    role_budgets: dict[str, RoleBudget] = field(default_factory=dict)

class UserMsgType(str, Enum):
    """Canonical types for messages arriving from the user side."""
    NEW_TASK           = "NEW_TASK"             # User submit a new task
    CANCEL             = "CANCEL"               # User cancel the running task
    GUIDANCE           = "GUIDANCE"             # User subimit gudiance during task solving
    CLARIFICATION      = "CLARIFICATION"        # User submit clarification
    RESUME             = "RESUME"               # User request resume task solving
    CHECKPOINT_RUN     = "CHECKPOINT_RUN"       # User request restore from checkpoint
    AGENT_PROGESS      = "AGENT_PROGESS_NOTIFY" # Agent notify client the progress of task solving


@dataclass(frozen=True)
class UserMessage:
    """Normalised command produced by PipelineDriver from a raw UIMessage."""
    msg_type: UserMsgType
    task_id: TaskId | None = None
    user_id: UserId | None = None
    content: str = None
    metadata: dict[str, Any] = field(default_factory=dict)

class UserCommandType(str, Enum):
    NEW_TASK        = "NEW_TASK"         # Submit a new task
    CANCEL          = "CANCEL"           # Cancel the running task
    GUIDANCE        = "GUIDANCE"         # Mid-task steering / correction
    CLARIFICATION   = "CLARIFICATION"    # Reply to a clarification request
    RESUME          = "RESUME"           # Resume after a B-class pause
    CHECKPOINT_RUN  = "CHECKPOINT_RUN"   # Resume from latest checkpoint


class StageStatus(str, Enum):
    RUNNING      = "RUNNING"
    COMPLETED    = "COMPLETED"
    PAUSED       = "PAUSED"
    SUCCESS      = "SUCCESS"
    INTERRUPTED  = "INTERRUPTED"
    FAILED       = "FAILED"


@dataclass(frozen=True)
class StepInput:
    source: str
    value: str
    step_ref: int | None = None
    constraint_note: str = ""


@dataclass(frozen=True)
class StepDependency:
    step_order: int
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Stage:
    id: StageId
    task_id: TaskId
    plan_step_id: PlanStepId
    order: int
    goal: str
    description: str
    key_results: list[str] = field(default_factory=list)
    inputs: list[StepInput] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    action_constraints: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    dependencies: list[StepDependency] = field(default_factory=list)
    execution_notes: str = ""
    output_constraints: str = ""
    status: StageStatus = StageStatus.RUNNING
    result: str = ""
    iteration_count: int = 0
    started_at: datetime = field(default_factory=_time_now)
    completed_at: datetime | None = None

    def increment_iteration(self) -> None:
        self.iteration_count += 1

    def complete(self, result: str) -> None:
        self.status = StageStatus.COMPLETED
        self.result = result
        self.completed_at = _time_now()

    def fail(self, reason: str = "") -> None:
        self.status = StageStatus.FAILED
        self.result = reason
        self.completed_at = _time_now()

    def pause(self, reason: str = "") -> None:
        self.status = StageStatus.PAUSED
        self.result = reason

    def get_step_order(self) -> int:
        return self.order

    def get_stage_index(self) -> int:
        return self.order - 1
