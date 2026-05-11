from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from uuid import uuid4

from utils.time.time import now as _time_now

from agent.application.driver import PipelineDriver
from agent.events.events import (
    ExecutionPlanFinalized,
    TaskAnalysisCompleted,
    TaskExecutionFailed,
    TaskExecutionStarted,
    TaskResultProduced,
)
from agent.factory.agent_factory import AgentFactory
from agent.models.context.context_manager import ToolResultMetadata, ToolUseMetadata
from config.config import ConfigReader
from schemas.event_bus import EventBus
from schemas.ids import TaskId, UserId
from schemas.task import Plan, Task, TaskRecoveryAction, TaskResult
from utils.log.log import Logger, zap

if TYPE_CHECKING:
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
    ) -> None:
        self._config = config
        self._agent_factory = agent_factory
        self._logger = logger
        self._tracer = self._agent_factory.build_tracer()
        self._event_bus = event_bus

        self._analyzer = self._agent_factory.build_analyzer(self._tracer, self._event_bus)
        self._quality_evaluator = self._agent_factory.build_quality_evaluator(self._tracer)
        self._knowledge_manager = self._agent_factory.build_knowledge_manager(self._tracer)
        self._knowledge_loader = self._agent_factory.build_knowledge_loader(self._tracer)
        self._model_selector = self._agent_factory.build_model_selector(self._tracer)
        self._personality_manager = self._agent_factory.build_personality_manager(self._tracer)
        self._planner = self._agent_factory.build_planner(
            tracer=self._tracer, 
            event_bus=self._event_bus,
            evaluator=self._quality_evaluator)

        self._llm_gateway = self._agent_factory.build_llm_gateway(self._tracer)
        self._reasoning_manager = AgentFactory.build_reasoning_manager(self._llm_gateway)

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
            )

        self._max_task_retries = int(self._config.get("agent.max_quality_retries", 2))

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
        self._start_session_trace(task_description)
        self._logger.info(
            "Pipeline run started",
            zap.any("user_id", user_id),
            zap.any("task", task_description),
            zap.any("trace_id", self._tracer.current_trace_id() if self._tracer else None),
        )

        # ── 1.1 分析Task特征 ──────────────────────────────────────────
        try:
            with self._tracer.start_span("pipeline.analyze_task", "pipeline", {"user_id": user_id}):
                task = self._analyzer.analyze(
                    user_id=user_id,
                    task_description=task_description,
                    llm_gateway=self._llm_gateway,
                    knowledge_loader=self._knowledge_loader,
                    personality_manager=self._personality_manager,
                    tool_registry=self._tool_registry,
                )
        except Exception as exc:
            self._logger.error("Pipeline task analysis failed", zap.any("error", exc))
            self._finish_session_trace(error=str(exc))
            raise
        self._task = task

        rewritten = self._build_rewritten_task_message(task_description, task)
        self._context_manager.add_message("user", rewritten)

        # 1.1.4 发布"分析报告已出"事件
        self._event_bus.publish(
            TaskAnalysisCompleted(task_id=task.id, content=task.intent) #TODO 优化UI信息
        )
        # ── 1.2 根据Task特征匹配处理模型 ──────────────────────────────
        try:
            with self._tracer.start_span("pipeline.route_model", "pipeline", {"task_id": task.id}):
                routing = self._model_selector.route(task=self._task, enable_fallback=True)
        except Exception as exc:
            self._logger.error("Pipeline model routing failed", zap.any("task_id", task.id), zap.any("error", exc))
            self._finish_session_trace(error=str(exc))
            raise
        provider_chain = [routing.primary] + routing.fallbacks
        self._logger.info(
            "Model routing complete",
            zap.any("task_id", task.id),
            zap.any("provider_chain", provider_chain),
        )

        # ── 1.3 制定并评审执行计划（含重试循环）──────────────────────
        try:
            with self._tracer.start_span("pipeline.make_plan", "pipeline", {"task_id": task.id}):
                plan = self._planner.make_plan(task, self._llm_gateway)
        except Exception as exc:
            self._logger.error("Pipeline plan generation failed", zap.any("task_id", task.id), zap.any("error", exc))
            self._finish_session_trace(error=str(exc))
            raise
        if plan is None:
            event = TaskExecutionFailed(task_id=task.id, content="Failed to produce a valid plan")
            self._event_bus.publish(event)
            result = self._failed_result(task.id, "Failed to produce a valid plan after retries")
            self._logger.error(
                "Pipeline failed to produce plan",
                zap.any("task_id", task.id),
            )
            self._finish_session_trace(error=result.error_reason or None)
            return result

        # 1.3.2.1.1 发布"执行计划已确定"事件
        self._event_bus.publish(
            ExecutionPlanFinalized(task_id=task.id, plan_id=plan.id, content="")
        )

        # 注入计划为模拟工具调用对
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
        self._event_bus.publish(
            TaskExecutionStarted(task_id=task.id, content="")
        )

        # ── 1.5 按照计划执行 ──────────────────────────────────────────
        current_task_retries = 0
        while True:
            try:
                self._logger.info(
                    "Stage execution loop started",
                    zap.any("task_id", task.id),
                    zap.any("plan_id", plan.id),
                    zap.any("task_retry", current_task_retries),
                    zap.any("step_count", len(plan.step_list)),
                )
                with self._tracer.start_span(
                    "pipeline.execute_plan",
                    "pipeline",
                    {
                        "task_id": task.id,
                        "plan_id": plan.id,
                        "task_retry": current_task_retries,
                        "step_count": len(plan.step_list),
                        "provider_chain": provider_chain,
                    },
                ):
                    raw_result = self._stage_executor.execute(plan=plan, provider_chain=provider_chain)
            except Exception as exc:
                self._logger.error(
                    "Pipeline plan execution raised",
                    zap.any("task_id", task.id),
                    zap.any("plan_id", plan.id),
                    zap.any("error", exc),
                )
                self._finish_session_trace(error=str(exc))
                raise

            # 1.5.2 执行失败
            if raw_result is None:
                event = TaskExecutionFailed(task_id=task.id, content="Stage execution failed")
                self._event_bus.publish(event)
                result = self._failed_result(task.id, "Stage execution failed")
                self._logger.error("Pipeline stage execution failed", zap.any("task_id", task.id))
                self._finish_session_trace(error=result.error_reason or None)
                return result

            # 1.5.1 执行成功 → 评审任务结果
            try:
                with self._tracer.start_span(
                    "pipeline.evaluate_task_result",
                    "pipeline",
                    {"task_id": task.id, "result_length": len(raw_result)},
                ):
                    review = self._quality_evaluator.evaluate_task_result(
                        task=task, result=raw_result, llmgateway=self._llm_gateway
                    )
            except Exception as exc:
                self._logger.error(
                    "Pipeline task result evaluation failed",
                    zap.any("task_id", task.id),
                    zap.any("error", exc),
                )
                self._finish_session_trace(error=str(exc))
                raise

            if review.passed:
                # 1.5.1.1.1 异步提取任务经验和知识
                self._extract_knowledge_async(task_description, raw_result)
                # 1.5.1.1.2 从用户建议里总结用户偏好并落地
                self._extract_preferences_async(task_description)
                # 1.5.1.1.3 发布"Task执行结果信息"事件
                self._event_bus.publish(
                    TaskResultProduced(task_id=task.id, content=raw_result)
                )
                result = TaskResult(
                    task_id=task.id,
                    succeeded=True,
                    result=raw_result,
                    error_reason="",
                    delivered_at=_time_now(),
                )
                self._logger.info(
                    "Pipeline run succeeded",
                    zap.any("task_id", task.id),
                    zap.any("task_retries", current_task_retries),
                    zap.any("result_length", len(raw_result)),
                )
                self._finish_session_trace()
                return result

            # 1.5.1.2 评审不通过 → 根据 LLM 建议的恢复策略处理
            current_task_retries += 1
            if current_task_retries > self._max_task_retries:
                event = TaskExecutionFailed(
                    task_id=task.id, content="Quality check failed after retries"
                )
                self._event_bus.publish(event)
                result = self._failed_result(task.id, "Quality check failed after retries")
                self._logger.error(
                    "Pipeline quality check failed after retries",
                    zap.any("task_id", task.id),
                    zap.any("max_retries", self._max_task_retries),
                    zap.any("feedback", review.feedback),
                )
                self._finish_session_trace(error=result.error_reason or None)
                return result

            action = review.recovery_action or TaskRecoveryAction.REPLAN_ALL
            self._logger.info(
                "Applying task recovery",
                zap.any("task_id", task.id),
                zap.any("action", action.value if hasattr(action, "value") else str(action)),
                zap.any("retry", current_task_retries),
                zap.any("feedback", review.feedback),
            )
            try:
                with self._tracer.start_span(
                    "pipeline.task_recovery",
                    "pipeline",
                    {
                        "task_id": task.id,
                        "action": action.value if hasattr(action, "value") else str(action),
                        "retry": current_task_retries,
                    },
                ):
                    plan = self._apply_task_recovery(action, task, plan, review.feedback)
            except Exception as exc:
                self._logger.error(
                    "Pipeline task recovery failed",
                    zap.any("task_id", task.id),
                    zap.any("action", action.value if hasattr(action, "value") else str(action)),
                    zap.any("error", exc),
                )
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
    ) -> Plan:
        """根据 LLM 建议的恢复模式重置上下文并（可选地）更新计划。"""
        self._stage_executor.reset()  # 两种模式都需要清空 ctx_window

        if action == TaskRecoveryAction.RETRY_SAME_PLAN:
            self._logger.info("Retrying same plan", zap.any("task_id", task.id), zap.any("plan_id", plan.id))
            return plan  # 计划不变，直接重试

        # REPLAN_ALL：重新生成整个计划
        self._logger.info("Renewing full plan", zap.any("task_id", task.id), zap.any("plan_id", plan.id))
        plan = self._planner.renew_plan(task=task, feedback=feedback, llm_api=self._llm_gateway)
        self._context_manager.set_plan(plan)
        return plan

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
        self._logger.info(
            "Session trace started",
            zap.any("trace_id", self._session_span.trace_id),
        )

    def _finish_session_trace(self, error: str | None = None) -> None:
        if self._session_span is None:
            return
        status = "error" if error else "ok"
        self._session_span.finish(status=status, error=error)
        self._logger.info(
            "Session trace finished",
            zap.any("status", status),
            zap.any("error", error),
        )
        self._session_span = None

    # ------------------------------------------------------------------
    # Async side-effects
    # ------------------------------------------------------------------

    def _extract_knowledge_async(self, task_description: str, result: str) -> None:
        summary = f"Task: {task_description}\nResult: {result}"

        def _run() -> None:
            try:
                self._logger.info("Async knowledge extraction started", zap.any("summary_length", len(summary)))
                self._knowledge_manager.extract_and_save(summary, self._llm_gateway)
                self._logger.info("Async knowledge extraction finished")
            except Exception as exc:
                self._logger.error("Async knowledge extraction failed", zap.any("error", exc))

        threading.Thread(target=_run, daemon=True).start()

    def _extract_preferences_async(self, task_description: str) -> None:
        def _run() -> None:
            try:
                self._logger.info("Async preference extraction started", zap.any("task_length", len(task_description)))
                self._personality_manager.extract_and_save_user_preference(
                    task_description, self._llm_gateway
                )
                self._logger.info("Async preference extraction finished")
            except Exception as exc:
                self._logger.error("Async preference extraction failed", zap.any("error", exc))

        threading.Thread(target=_run, daemon=True).start()

    def _save_checkpoint_async(self, task_id: TaskId, stage_order: int) -> None:
        conversation = self._stage_executor.get_conversation_history()

        def _save() -> None:
            try:
                self._checkpoint_processor.save(task_id, stage_order, conversation)
            except Exception:
                pass

        threading.Thread(target=_save, daemon=True).start()

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
        if task.related_knowledge_entries:
            lines.append(f"  knowledge entries: {len(task.related_knowledge_entries)}")
        if task.related_user_preference_entries:
            lines.append(f"  preference entries: {len(task.related_user_preference_entries)}")
        return "\n".join(lines)

    @staticmethod
    def _build_rewritten_task_message(task_description: str, task: Task) -> str:#TODO 优化这里
        lines = [
            "## Task",
            "",
            f"**Original request:** {task_description}",
            f"**Clarified intent:** {task.intent}",
            f"**Task type:** {task.task_type}",
        ]
        if task.output_constraints:
            lines.append(f"**Output constraints:** {task.output_constraints}")
        if task.notes:
            lines.append(f"**Notes:** {task.notes}")
        return "\n".join(lines)

    @staticmethod
    def _build_plan_content(plan: Plan) -> str:
        lines = ["## Execution Plan", ""]
        for step in plan.step_list:
            lines.append(f"Step {step.order}: {step.goal}")
            lines.append(f"  Description: {step.description}")
            if step.key_results:
                lines.append("  Key results:")
                for kr in step.key_results:
                    lines.append(f"    - {kr}")
            if step.inputs:
                lines.append(f"  Inputs: {', '.join(step.inputs)}")
            if step.required_tools:
                lines.append(f"  Tools: {', '.join(step.required_tools)}")
            if step.constraints:
                lines.append("  Constraints:")
                for constraint in step.constraints:
                    lines.append(f"    - {constraint}")
            if step.risks:
                lines.append("  Risks/checks:")
                for risk in step.risks:
                    lines.append(f"    - {risk}")
            if step.dependencies:
                lines.append(f"  Depends on steps: {', '.join(str(i) for i in step.dependencies)}")
            if step.execution_notes:
                lines.append(f"  Execution notes: {step.execution_notes}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _update_reasoning_gateway(self, provider_name: str) -> None:
        self._llm_gateway.switch_provider(provider_name)

    @staticmethod
    def _next_provider_index(provider_chain: list[str], current_index: int) -> int | None:
        next_index = current_index + 1
        return next_index if next_index < len(provider_chain) else None

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
