from __future__ import annotations

"""Shared task-context seeding.

Both Pipeline (initial run / task-level recovery) and StageExecutor (REPLAN_ALL)
need to (a) filter tool schemas by the combined analyzer+planner score and
(b) re-inject the rewritten task message plus the make_plan tool-call pair that
starts every reasoning trace. Those two operations lived in three near-identical
copies before; this is the single implementation.
"""

from typing import TYPE_CHECKING
from uuid import uuid4

from config import ConfigReader
from infra.rendering_engine import PromptRenderer
from schemas.task import Plan, Task
from tools.tool_registry import ToolRegistry
from utils.log.log import Logger

from agent.models.context.context_manager import ToolResultMetadata, ToolUseMetadata

if TYPE_CHECKING:
    from agent.models.context.context_manager import ContextManager

_PLAN_TOOL_NAME = "make_plan"
_PLAN_ANNOUNCEMENT = "I have analyzed the task. I will now create an execution plan."


class TaskContextSeeder:
    """Filters tool schemas and seeds the reasoning trace starting point."""

    def __init__(
        self,
        config: ConfigReader,
        logger: Logger,
        renderer: PromptRenderer,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
    ) -> None:
        self._config = config
        self._logger = logger
        self._renderer = renderer
        self._context_manager = context_manager
        self._tool_registry = tool_registry

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    def apply_tool_filter(self, task: Task, *, reason: str = "") -> list[str]:
        """Keep tools whose analyzer or planner score clears the threshold.

        Returns the kept tool names. Schemas are left untouched when nothing
        clears the threshold, so the agent never ends up with zero tools.
        """
        threshold = float(self._config.get("planner.tool_score_filter_threshold", 0.65))
        score_map = {m.tool_name: m for m in task.tool_matches}
        kept = [
            name for name, m in score_map.items()
            if max(m.match_score, m.planner_score) >= threshold
        ]
        if not kept:
            return []

        self._context_manager.set_tool_schemas(self._tool_registry.get_tool_schemas_for(kept))
        self._logger.info("Tool schemas filtered for task",
            task_id=task.id, threshold=threshold, reason=reason,
            total_tools=len(score_map), filtered_count=len(kept), kept_tools=kept)
        return kept

    # ------------------------------------------------------------------
    # Reasoning trace seeding
    # ------------------------------------------------------------------

    def seed(self, task: Task, plan: Plan, task_description: str) -> None:
        """Inject the rewritten task message and the plan as a tool-call pair.

        Called once for a fresh task and again after any context reset, so the
        reasoning trace always starts from the same shape.
        """
        self.add_rewritten_task_message(task, task_description)
        self.add_plan_messages(plan, task_description)

    def add_rewritten_task_message(self, task: Task, task_description: str) -> None:
        rewritten = self._renderer.render("pipeline/rewritten_task_message.j2", {
            "task_description": task_description,
            "task": task,
        }).rstrip()
        self._context_manager.add_message("user", rewritten)

    def add_plan_messages(self, plan: Plan, task_description: str) -> None:
        """Wrap the plan in a synthetic make_plan tool call so it reads as a step."""
        tool_call_id = str(uuid4())
        self._context_manager.add_message(
            "assistant",
            _PLAN_ANNOUNCEMENT,
            tool_use=ToolUseMetadata(
                tool_call_id=tool_call_id,
                tool_name=_PLAN_TOOL_NAME,
                tool_arguments={"task_description": task_description},
                extra_calls=(),
            ),
        )
        self._context_manager.add_message(
            "tool",
            self.render_plan_content(plan),
            tool_result=ToolResultMetadata(
                tool_call_id=tool_call_id,
                tool_name=_PLAN_TOOL_NAME,
                success=True,
            ),
        )

    def render_plan_content(self, plan: Plan) -> str:
        return self._renderer.render("pipeline/plan_content.j2", {"plan": plan}).rstrip()
