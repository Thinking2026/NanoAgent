from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from uuid import uuid4

from utils.time.time import now as _time_now

from agent.application.driver import PipelineDriver
from agent.events.events import *
from agent.models.context.context_manager import ToolResultMetadata, ToolUseMetadata
from config.config import ConfigReader
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.event_bus import EventBus
from schemas.ids import TaskId, UserId
from schemas.task import Plan, Task, TaskRecoveryAction, TaskResult
from utils.log.log import Logger

if TYPE_CHECKING:
    from agent.factory.agent_factory import AgentFactory
    from infra.observability.tracing import Span

class Pipeline:
    """Application-layer orchestrator for the full task lifecycle.

    Implements the three-level loop from TD.md:
      Task Level  → plan → execute stages → quality check
      Stage Level → handled by StageExecutor.execute()
      Reasoning   → handled by StageExecutor._execute_stage()

    User signals (cancel / guidance / clarification / resume) are delivered via
    the public control methods, which are safe to call from any thread.
    """

    def __init__(
        self,
        config: ConfigReader,
        logger: Logger,
        agent_factory: AgentFactory,
        event_bus: EventBus,
        renderer: PromptRenderer | None = None,
    ) -> None:
        self._config = config
        self._agent_factory = agent_factory
        self._logger = logger
        self._event_bus = event_bus
        self._tracer = self._agent_factory.build_tracer()
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()

        self._analyzer = self._agent_factory.build_analyzer(self._tracer, self._event_bus)
        self._quality_evaluator = self._agent_factory.build_quality_evaluator(self._tracer, self._event_bus)
        self._knowledge_manager = self._agent_factory.build_knowledge_manager(self._tracer, self._event_bus)
        self._knowledge_loader = self._agent_factory.build_knowledge_loader(self._tracer, self._event_bus)
        self._model_selector = self._agent_factory.build_model_selector(self._tracer, self._event_bus)
        self._personality_manager = self._agent_factory.build_personality_manager(self._tracer, self._event_bus)

        self._planner = self._agent_factory.build_planner(
            tracer=self._tracer, 
            event_bus=self._event_bus,
            evaluator=self._quality_evaluator)

        self._llm_gateway = self._agent_factory.build_llm_gateway(self._tracer, self._event_bus)
        self._reasoning_manager = self._agent_factory.build_reasoning_manager(self._tracer, self._llm_gateway)

        self._tool_registry = self._agent_factory.build_tool_registry(self._tracer)
        self._context_manager = self._agent_factory.build_context_manager(tracer=self._tracer, llm_gateway=self._llm_gateway, tool_registry=self._tool_registry)

        self._stage_executor = self._agent_factory.build_stage_executor(
                tracer=self._tracer,
                reasoning_manager=self._reasoning_manager,
                context_manager=self._context_manager,
                quality_evaluator=self._quality_evaluator,
                knowledge_loader=self._knowledge_loader,
                planner=self._planner,
                llm_gateway=self._llm_gateway,
                event_bus=self._event_bus,
                model_selector=self._model_selector,
                tool_registry=self._tool_registry,
            )

        self._max_task_retries = int(self._config.get("agent.max_quality_retries", 2))
        self._knowledge_snippet_chars = int(self._config.get("pipeline.knowledge_snippet_chars", 3000))
        self._preference_snippet_chars = int(self._config.get("pipeline.preference_snippet_chars", 2000))

        self._task: Task | None = None
        self._session_span: Span | None = None

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver
        self._analyzer.set_driver(driver)
        self._stage_executor.set_driver(driver)
        self._planner.set_driver(driver)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, user_id: UserId, task_description: str) -> TaskResult:
        self._context_manager.reset()
        self._task_description_cache = task_description
        self._start_session_trace(task_description)
        task_id = TaskId(str(uuid4()))
        self._logger.info("Pipeline run started", user_id=user_id, task=task_description)

        # ── 1.1 分析Task特征 ──────────────────────────────────────────
        self._event_bus.publish(
            TaskAnalysisStarted.with_meta(task_id=task_id, task_description=task_description)
        )
        try:
            with self._tracer.start_span("pipeline.analyze_task", "pipeline", user_id=user_id, task_id=task_id):
                task = self._analyzer.analyze(
                    task_id=task_id,
                    user_id=user_id,
                    task_description=task_description,
                    llm_gateway=self._llm_gateway,
                    knowledge_loader=self._knowledge_loader,
                    personality_manager=self._personality_manager,
                    tool_registry=self._tool_registry,
                )
        except Exception as exc:
            self._logger.error("Pipeline task analysis failed", error=exc)
            self._finish_session_trace(error=str(exc))
            raise
        self._task = task

        rewritten = self._build_rewritten_task_message(task_description, task)
        self._context_manager.add_message("user", rewritten)

        # 1.1.4 发布"分析报告已出"事件
        self._event_bus.publish(
            TaskAnalysisSucceed.with_meta(task_id=task.id, task_type=task.task_type, task_goal=task.task_goal)
        )
        # ── 1.2 根据Task特征匹配处理模型 ──────────────────────────────
        try:
            with self._tracer.start_span("pipeline.route_model", "pipeline", task_id=task.id):
                self._model_selector.initialize_routing(task=self._task)
        except Exception as exc:
            self._logger.error("Pipeline model routing failed", task_id=task.id, error=exc)
            self._finish_session_trace(error=str(exc))
            raise
        self._logger.info("Model routing complete",
            task_id=task.id,
            current_provider=self._model_selector.get_current_provider())

        # ── 1.3 制定并评审执行计划（含重试循环）──────────────────────
        self._event_bus.publish(
            PlanGenerateStarted.with_meta(task_id=task.id)
        )
        try:
            with self._tracer.start_span("pipeline.make_plan", "pipeline", task_id=task.id):
                plan, task = self._planner.make_plan(task, self._llm_gateway)
        except Exception as exc:
            self._logger.error("Pipeline plan generation failed", task_id=task.id, error=exc)
            self._finish_session_trace(error=str(exc))
            raise
        
        # ── 1.3.1 Filter tools by combined analyzer + planner score ──────
        threshold: float = float(self._config.get("planner.tool_score_filter_threshold", 0.65))
        score_map = {m.tool_name: m for m in task.tool_matches}
        filtered_tool_names: list[str] = [
            name for name, m in score_map.items()
            if max(m.match_score, m.planner_score) >= threshold
        ]
        if filtered_tool_names:
            filtered_schemas = self._tool_registry.get_tool_schemas_for(filtered_tool_names)
            self._context_manager.set_tool_schemas(filtered_schemas)
            self._logger.info("the tool need to use for this task are decided",
                task_id=task.id, threshold=threshold,
                total_tools=len(score_map), filtered_count=len(filtered_tool_names),
                kept_tools=filtered_tool_names)

        # 1.3.2.1.1 发布"执行计划已确定"事件
        _plan_goals = " → ".join(s.goal[:35] for s in plan.step_list[:4])
        _plan_suffix = "..." if len(plan.step_list) > 4 else ""
        self._event_bus.publish(
            PlanGenerateSucceed.with_meta(task_id=task.id,
                plan=f"[{len(plan.step_list)} steps] {_plan_goals}{_plan_suffix}",
                steps=len(plan.step_list))
        )

        # 使用工具调用消息来包装plan
        plan_tool_call_id = str(uuid4())
        self._context_manager.add_message(
            "assistant",
            "I have analyzed the task. I will now create an execution plan.",
            tool_use=ToolUseMetadata(
                tool_call_id=plan_tool_call_id,
                tool_name="make_plan",
                tool_arguments={"task_description": task_description},
                extra_calls=(),
            ),
        )
        self._context_manager.add_message(
            "tool",
            self._build_plan_content(plan),
            tool_result=ToolResultMetadata(
                tool_call_id=plan_tool_call_id,
                tool_name="make_plan",
                success=True,
            ),
        )

        # ── 1.4 发布"Task已开始执行"事件 ─────────────────────────────
        self._context_manager.set_task(task)
        self._context_manager.set_plan(plan)
        self._stage_executor.set_task_description(task_description)
        self._stage_executor.set_task_output_constraints(task.output_constraints or "")
        self._stage_executor.set_task_goal(task.task_goal or "")
        self._stage_executor.set_task_intent(task.intent or "")
        self._event_bus.publish(
            TaskExecutionStarted.with_meta(task_id=task.id, progress=f"Starting {len(plan.step_list)}-step plan")
        )

        # ── 1.5 按照计划执行 ──────────────────────────────────────────
        current_task_retries = 0
        while True:
            try:
                self._logger.info("Stage execution loop started", task_id=task.id, plan_id=plan.id, current_retry_time=current_task_retries, step_count=len(plan.step_list))
                with self._tracer.start_span("pipeline.execute_plan", "pipeline", task_id=task.id, plan_id=plan.id, current_retry_time=current_task_retries, step_count=len(plan.step_list)):
                    raw_result = self._stage_executor.execute(plan=plan)
            except Exception as exc:
                self._logger.error("Task execute failed in pipeline",
                    task_id=task.id, plan_id=plan.id, error=exc)
                self._finish_session_trace(error=str(exc))
                raise

            # 1.5.2 执行失败
            if raw_result is None:
                event = TaskExecutionFailed.with_meta(task_id=task.id)
                self._event_bus.publish(event)
                result = self._failed_result(task.id, "Stage execution failed")
                self._logger.error("Task execute failed in pipeline, got None result", task_id=task.id)
                self._finish_session_trace(error=result.error_reason or None)
                return result

            # 1.5.1 执行成功 → 评审任务结果
            try:
                with self._tracer.start_span("pipeline.evaluate_task_result", "pipeline",
                    task_id=task.id, result_length=len(raw_result)):
                    review = self._quality_evaluator.evaluate_task_result(
                        task=task, result=raw_result, llmgateway=self._llm_gateway
                    )
            except Exception as exc:
                self._logger.error("Task result evaluation failed", task_id=task.id, error=exc)
                self._finish_session_trace(error=str(exc))
                raise

            if review.passed:
                # 1.5.1.1.1 异步提取任务经验和知识
                #self._extract_knowledge_async(task, raw_result)
                # 1.5.1.1.2 从用户建议里总结用户偏好并落地
                #self._extract_preferences_async(task_description)
                # 1.5.1.1.3 发布"Task执行结果信息"事件
                self._event_bus.publish(
                    TaskExecutionSucceed.with_meta(task_id=task.id, result=raw_result)
                )
                result = TaskResult(
                    task_id=task.id,
                    succeeded=True,
                    result=raw_result,
                    error_reason="",
                    delivered_at=_time_now(),
                )
                self._logger.info("Task result evaluate succeeded",
                    task_id=task.id, curernt_task_retries=current_task_retries,
                    result_length=len(raw_result))
                self._finish_session_trace()
                return result

            # 1.5.1.2 评审不通过 → 根据 LLM 建议的恢复策略处理
            current_task_retries += 1
            if current_task_retries > self._max_task_retries:
                event = TaskExecutionFailed.with_meta(
                    task_id=task.id, result="Exceed maximum retries but still failed"
                )
                self._event_bus.publish(event)
                result = self._failed_result(task.id, "Task Result quality check failed after max retries")
                self._logger.error("Task result evaluate failed after max retries",
                    task_id=task.id, max_retries=self._max_task_retries,
                    feedback=review.feedback)
                self._finish_session_trace(error=result.error_reason or None)
                return result

            action = review.recovery_action or TaskRecoveryAction.REPLAN_ALL
            self._logger.info("Start to retry or replan after evaluate rejected",
                task_id=task.id,
                action=action.value if hasattr(action, "value") else str(action),
                retry=current_task_retries, feedback=review.feedback)
            try:
                with self._tracer.start_span("pipeline.task_recovery", "pipeline",
                    task_id=task.id,
                    action=action.value if hasattr(action, "value") else str(action),
                    retry=current_task_retries):
                    plan, task = self._apply_task_recovery(action, task, plan, review.feedback)
            except Exception as exc:
                self._logger.error("Pipeline task recovery failed",
                    task_id=task.id,
                    action=action.value if hasattr(action, "value") else str(action),
                    error=exc)
                self._finish_session_trace(error=str(exc))
                raise

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------

    def _apply_task_recovery(
        self,
        action: TaskRecoveryAction,
        task: Task,
        plan: Plan,
        feedback: str,
    ) -> tuple[Plan, Task]:
        """根据 LLM 建议的恢复模式重置上下文并（可选地）更新计划。"""
        self._stage_executor.reset()  # 两种模式都需要清空 ctx_window
        self._stage_executor.set_task_recovery_feedback(feedback)

        if action == TaskRecoveryAction.RETRY_SAME_PLAN:
            self._logger.info("Retrying same plan", task_id=task.id, plan_id=plan.id)
            self._reinject_task_context(task, plan)
            return plan, task

        # REPLAN_ALL：重新生成整个计划
        self._logger.info("Renewing full plan", task_id=task.id, plan_id=plan.id)
        plan, task = self._planner.renew_plan(task=task, feedback=feedback, llm_api=self._llm_gateway)
        self._context_manager.set_task(task)
        self._context_manager.set_plan(plan)
        threshold: float = float(self._config.get("planner.tool_score_filter_threshold", 0.65))
        score_map = {m.tool_name: m for m in task.tool_matches}
        filtered_tool_names: list[str] = [
            name for name, m in score_map.items()
            if max(m.match_score, m.planner_score) >= threshold
        ]
        if filtered_tool_names:
            filtered_schemas = self._tool_registry.get_tool_schemas_for(filtered_tool_names)
            self._context_manager.set_tool_schemas(filtered_schemas)
            self._logger.info("Tool schemas updated after task replan",
                task_id=task.id, threshold=threshold,
                filtered_count=len(filtered_tool_names), kept_tools=filtered_tool_names)
        self._reinject_task_context(task, plan)
        return plan, task

    def _reinject_task_context(self, task: Task, plan: Plan) -> None:
        """reset 后重新注入 rewritten_task_message 和 plan tool call，恢复推理轨迹起点。"""
        rewritten = self._build_rewritten_task_message(self._task_description_cache, task)
        self._context_manager.add_message("user", rewritten)
        plan_tool_call_id = str(uuid4())
        self._context_manager.add_message(
            "assistant",
            "I have analyzed the task. I will now create an execution plan.",
            tool_use=ToolUseMetadata(
                tool_call_id=plan_tool_call_id,
                tool_name="make_plan",
                tool_arguments={"task_description": self._task_description_cache},
                extra_calls=(),
            ),
        )
        self._context_manager.add_message(
            "tool",
            self._build_plan_content(plan),
            tool_result=ToolResultMetadata(
                tool_call_id=plan_tool_call_id,
                tool_name="make_plan",
                success=True,
            ),
        )

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------

    def _start_session_trace(self, task_description: str) -> None:
        if self._tracer is None or self._session_span is not None:
            return
        self._session_span = self._tracer.start_trace(
            "session",
            attributes={"task": task_description},
        )
        self._logger.info("Session trace started", trace_id=self._session_span.trace_id)

    def _finish_session_trace(self, error: str | None = None) -> None:
        if self._session_span is None:
            return
        status = "error" if error else "ok"
        self._session_span.finish(status=status, error=error)
        self._logger.info("Session trace finished", status=status, error=error)
        self._session_span = None

    # ------------------------------------------------------------------
    # Async side-effects
    # ------------------------------------------------------------------

    def _build_conversation_snippet(self, max_chars: int) -> str | None:
        history = self._stage_executor.get_conversation_history()
        if not history:
            return None
        filtered = [m for m in history if m.role in ("user", "assistant")]
        if not filtered:
            return None
        joined = "\n".join(f"{m.role}: {m.content}" for m in filtered)
        if len(joined) <= max_chars:
            return joined
        truncated = joined[-max_chars:]
        first_newline = truncated.find("\n")
        if first_newline > 0:
            truncated = truncated[first_newline + 1:]
        return truncated

    def _extract_knowledge_async(self, task: Task, result: str) -> None:
        snippet = self._build_conversation_snippet(self._knowledge_snippet_chars)

        def _run() -> None:
            try:
                self._logger.info("Async knowledge extraction started",
                    task_id=task.id, result_length=len(result), has_snippet=snippet is not None)
                self._knowledge_manager.extract_and_save(task, result, self._llm_gateway, snippet)
                self._logger.info("Async knowledge extraction finished", task_id=task.id)
            except Exception as exc:
                self._logger.error("Async knowledge extraction failed", error=exc)

        threading.Thread(target=_run, daemon=True).start()

    def _extract_preferences_async(self, task_description: str) -> None:
        snippet = self._build_conversation_snippet(self._preference_snippet_chars)

        def _run() -> None:
            try:
                self._logger.info("Async preference extraction started",
                    task_length=len(task_description), has_snippet=snippet is not None)
                self._personality_manager.extract_and_save_user_preference(
                    task_description, self._llm_gateway, snippet
                )
                self._logger.info("Async preference extraction finished")
            except Exception as exc:
                self._logger.error("Async preference extraction failed", error=exc)

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_analysis_report(self, task: Task) -> str:
        lines = [
            f"Task analysis complete:",
            f"  type: {task.task_type}",
            f"  intent: {task.intent}",
            f"  complexity: {task.complexity.level}/5",
        ]
        if task.required_tools:
            lines.append(f"  tools: {', '.join(task.required_tools)}")
        if task.related_knowledge:
            lines.append(f"  knowledge: {len(task.related_knowledge)} chars")
        if task.related_user_preference:
            lines.append(f"  preference: {len(task.related_user_preference)} chars")
        return "\n".join(lines)

    @staticmethod
    def _build_rewritten_task_message(task_description: str, task: Task) -> str:
        from infra.rendering_engine import Jinja2PromptRenderer
        renderer = Jinja2PromptRenderer()
        return renderer.render("pipeline/rewritten_task_message.j2", {
            "task_description": task_description,
            "task": task,
        }).rstrip()

    @staticmethod
    def _build_plan_content(plan: Plan) -> str:
        from infra.rendering_engine import Jinja2PromptRenderer
        renderer = Jinja2PromptRenderer()
        return renderer.render("pipeline/plan_content.j2", {"plan": plan}).rstrip()

    def _update_reasoning_gateway(self, provider_name: str) -> None:
        self._llm_gateway.switch_provider(provider_name)

    def _cancelled_result(self, task_id: TaskId) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            succeeded=False,
            result="",
            error_reason="Task cancelled by user",
            delivered_at=_time_now(),
        )

    def _failed_result(self, task_id: TaskId, reason: str) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            succeeded=False,
            result="",
            error_reason=reason,
            delivered_at=_time_now(),
        )
