from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from utils.time.time import now as _time_now

from agent.application.driver import PipelineDriver
from agent.events.events import *
from agent.models.checkpoint.checkpoint_manager import CheckpointManager
from agent.models.context.task_context_seeder import TaskContextSeeder
from config.config import ConfigReader
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.event_bus import EventBus
from schemas.ids import TaskId, UserId
from schemas.task import EvaluationReport, Plan, Task, TaskRecoveryAction, TaskResult
from schemas.types import CheckpointData, UserMsgType
from utils.log.log import Logger

if TYPE_CHECKING:
    from agent.factory.agent_factory import AgentFactory
    from infra.observability.tracing import Span


class Pipeline:
    """Application-layer orchestrator for the full task lifecycle.

    Implements the three-level loop from TD.md:
      Task Level  → analyze → plan → execute stages → quality check → recover
      Stage Level → handled by StageExecutor.execute()
      Reasoning   → handled by StageExecutor._execute_stage()

    Every task-level phase runs through `_phase()`, which owns the tracing span
    and the failure path (log → failure event → close session trace → re-raise),
    so the lifecycle methods below read as the sequence of phases only.

    User signals (cancel / guidance / clarification / resume) are delivered via
    the driver, which is safe to call from any thread.
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
        self._agent_reach_tool_registrar = self._agent_factory.build_agent_reach_tool_registrar()
        self._agent_reach_tool_registrar.register_healthy_tools(self._tool_registry)
        self._context_manager = self._agent_factory.build_context_manager(
            tracer=self._tracer, llm_gateway=self._llm_gateway, tool_registry=self._tool_registry)

        self._context_seeder = TaskContextSeeder(
            config=self._config,
            logger=self._logger,
            renderer=self._renderer,
            context_manager=self._context_manager,
            tool_registry=self._tool_registry,
        )

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
        self._task_description_cache: str = ""
        self._session_span: Span | None = None

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver
        self._analyzer.set_driver(driver)
        self._stage_executor.set_driver(driver)
        self._planner.set_driver(driver)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, user_id: UserId, task_description: str,
            msg_type: UserMsgType = UserMsgType.NEW_TASK) -> TaskResult:
        if msg_type == UserMsgType.LOAD_CHECKPOINT:
            return self._run_from_checkpoint(task_description)
        return self._run_new_task(user_id, task_description)

    def _run_new_task(self, user_id: UserId, task_description: str) -> TaskResult:
        """Task Level: analyze → route model → plan → execute stages."""
        task_id = self._prepare_new_run(task_description)
        task = self._analyze_task(task_id, user_id, task_description)
        self._route_model(task)
        plan, task = self._make_plan(task)
        self._seed_execution_context(task, plan, task_description)
        return self._execute_stages(task, plan, task_description)

    def _run_from_checkpoint(self, checkpoint_path: str) -> TaskResult:
        """Restore state from a checkpoint file and resume from the next step."""
        data = CheckpointManager.load(checkpoint_path)
        self._logger.info("Resuming from checkpoint",
            checkpoint_path=checkpoint_path,
            task_id=data.task_id,
            next_step=data.completed_step_index + 1)

        self._context_manager.reset()
        self._context_manager.replace_conversation_history(data.conversation_history)
        self._context_manager.set_tool_schemas(data.tool_schemas)
        self._context_manager.set_task(data.task)
        self._context_manager.set_plan(data.plan)
        self._model_selector.restore_state(data.model_selector_state)
        self._stage_executor.set_task_description(data.task_description)
        self._stage_executor.set_task_output_constraints(data.task_output_constraints)
        self._stage_executor.set_task_goal(data.task_goal)
        self._stage_executor.set_task_intent(data.task_intent)
        self._stage_executor.set_task_recovery_feedback(data.task_recovery_feedback)
        self._task = data.task
        self._task_description_cache = data.task_description
        self._start_session_trace(data.task_description)

        self._event_bus.publish(
            TaskResumed.with_meta(
                task_id=data.task_id,
                progress=f"从 checkpoint 恢复，将从第 {data.completed_step_index + 2} 步继续。",
                total_steps=len(data.plan.step_list),
            )
        )
        return self._execute_stages(data.task, data.plan, data.task_description,
                                    start_step_index=data.completed_step_index + 1)

    # ------------------------------------------------------------------
    # Task Level loop
    # ------------------------------------------------------------------

    def _execute_stages(
        self,
        task: Task,
        plan: Plan,
        task_description: str,
        start_step_index: int = 0,
    ) -> TaskResult:
        """Run the plan, review the deliverable, recover on rejection.

        Shared by new tasks and checkpoint resumes. Each iteration either
        delivers a result, fails terminally, or produces a plan to retry with.
        """
        self._arm_checkpointing(task, plan, task_description)

        retries = 0
        while True:
            raw_result = self._run_plan(task, plan, start_step_index, retries)
            if raw_result is None:
                return self._on_plan_execution_failed(task)

            review = self._review_task_result(task, raw_result)
            if review.passed:
                return self._deliver_success(task, raw_result, review, task_description, retries)

            retries += 1
            if retries > self._max_task_retries:
                return self._on_task_retries_exhausted(task, review, retries)

            plan, task = self._recover_task(task, plan, review, retries)
            # After recovery, always restart from step 0
            start_step_index = 0

    # ── Task Level phases ─────────────────────────────────────────────

    def _prepare_new_run(self, task_description: str) -> TaskId:
        self._context_manager.reset()
        self._task_description_cache = task_description
        self._start_session_trace(task_description)
        return TaskId(str(uuid4()))

    def _analyze_task(self, task_id: TaskId, user_id: UserId, task_description: str) -> Task:
        """1.1 分析 Task 特征，并把重写后的任务描述注入上下文。"""
        self._logger.info("Pipeline run started", user_id=user_id, task=task_description)
        self._event_bus.publish(
            TaskAnalysisStarted.with_meta(task_id=task_id, task_description=task_description)
        )

        task = self._phase(
            "analyze_task",
            lambda: self._analyzer.analyze(
                task_id=task_id,
                user_id=user_id,
                task_description=task_description,
                llm_gateway=self._llm_gateway,
                knowledge_loader=self._knowledge_loader,
                personality_manager=self._personality_manager,
                tool_registry=self._tool_registry,
            ),
            failure_event=lambda exc: TaskAnalysisFailed.with_meta(task_id=task_id, error=exc),
            user_id=user_id, task_id=task_id,
        )
        self._task = task

        self._context_seeder.add_rewritten_task_message(task, task_description)
        self._event_bus.publish(
            TaskAnalysisSucceed.with_meta(
                task_id=task.id,
                task_type=task.task_type,
                task_goal=task.task_goal,
                intent=task.intent,
                complexity=f"{task.complexity.level}/5",
                estimated_steps=task.estimated_steps,
                required_tools=task.required_tools,
                risks=[risk.description for risk in task.risks[:3]],
            )
        )
        return task

    def _route_model(self, task: Task) -> None:
        """1.2 根据 Task 特征匹配处理模型。"""
        self._phase(
            "route_model",
            lambda: self._model_selector.initialize_routing(task=task),
            task_id=task.id,
        )
        self._logger.info("Model routing complete",
            task_id=task.id,
            current_provider=self._model_selector.get_current_provider())

    def _make_plan(self, task: Task) -> tuple[Plan, Task]:
        """1.3 制定并评审执行计划（重试循环在 Planner 内部）。"""
        self._event_bus.publish(
            PlanGenerateStarted.with_meta(
                task_id=task.id,
                task_goal=task.task_goal,
                estimated_steps=task.estimated_steps,
            )
        )
        return self._phase(
            "make_plan",
            lambda: self._planner.make_plan(task, self._llm_gateway),
            failure_event=lambda exc: PlanGenerateFailed.with_meta(task_id=task.id, error=exc),
            task_id=task.id,
        )

    def _seed_execution_context(self, task: Task, plan: Plan, task_description: str) -> None:
        """1.4 过滤工具、注入计划、把任务级约束交给 StageExecutor，然后宣告开始执行。"""
        kept_tools = self._context_seeder.apply_tool_filter(task, reason="plan_ready")
        self._publish_plan_ready(task, plan, kept_tools)
        self._context_seeder.add_plan_messages(plan, task_description)

        self._context_manager.set_task(task)
        self._context_manager.set_plan(plan)
        self._stage_executor.set_task_description(task_description)
        self._stage_executor.set_task_output_constraints(task.output_constraints or "")
        self._stage_executor.set_task_goal(task.task_goal or "")
        self._stage_executor.set_task_intent(task.intent or "")

        self._event_bus.publish(
            TaskExecutionStarted.with_meta(
                task_id=task.id,
                progress=f"Starting {len(plan.step_list)}-step plan",
                total_steps=len(plan.step_list),
                task_goal=task.task_goal,
            )
        )

    def _run_plan(self, task: Task, plan: Plan, start_step_index: int, retries: int) -> str | None:
        """1.5 调用 Stage 级循环。返回 None 表示阶段执行未能产出结果。"""
        self._logger.info("Stage execution loop started", task_id=task.id, plan_id=plan.id,
            current_retry_time=retries, step_count=len(plan.step_list))
        return self._phase(
            "execute_plan",
            lambda: self._stage_executor.execute(plan=plan, start_step_index=start_step_index),
            log_message="Task execute failed in pipeline",
            task_id=task.id, plan_id=plan.id,
            current_retry_time=retries, step_count=len(plan.step_list),
        )

    def _review_task_result(self, task: Task, raw_result: str) -> EvaluationReport:
        """1.5.1 执行成功 → 评审任务结果。"""
        return self._phase(
            "evaluate_task_result",
            lambda: self._quality_evaluator.evaluate_task_result(
                task=task, result=raw_result, llmgateway=self._llm_gateway),
            log_message="Task result evaluation failed",
            task_id=task.id, result_length=len(raw_result),
        )

    def _recover_task(
        self, task: Task, plan: Plan, review: EvaluationReport, retries: int
    ) -> tuple[Plan, Task]:
        """1.5.1.2 评审不通过 → 按 LLM 建议的恢复策略重试或重新规划。"""
        action = review.recovery_action or TaskRecoveryAction.REPLAN_ALL
        label = _action_label(action)
        self._logger.info("Start to retry or replan after evaluate rejected",
            task_id=task.id, action=label, retry=retries, feedback=review.feedback)
        self._event_bus.publish(
            RePlanStarted.with_meta(
                task_id=task.id,
                reason="最终结果复核未通过，需要恢复执行。",
                feedback=review.feedback,
                recovery_action=label,
                retry=retries,
                max_retries=self._max_task_retries,
            )
        )

        plan, task = self._phase(
            "task_recovery",
            lambda: self._apply_task_recovery(action, task, plan, review.feedback),
            failure_event=lambda exc: RePlanFailed.with_meta(
                task_id=task.id, recovery_action=label, error=exc),
            log_message="Pipeline task recovery failed",
            task_id=task.id, action=label, retry=retries,
        )

        self._event_bus.publish(
            RePlanSucceed.with_meta(
                task_id=task.id,
                recovery_action=label,
                steps=len(plan.step_list),
                plan=" → ".join(s.goal[:35] for s in plan.step_list[:4]),
            )
        )
        return plan, task

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
            self._context_seeder.seed(task, plan, self._task_description_cache)
            return plan, task

        # REPLAN_ALL：重新生成整个计划
        self._logger.info("Renewing full plan", task_id=task.id, plan_id=plan.id)
        plan, task = self._planner.renew_plan(
            task=task, feedback=feedback, llm_api=self._llm_gateway)
        self._context_manager.set_task(task)
        self._context_manager.set_plan(plan)
        self._context_seeder.apply_tool_filter(task, reason="task_replan_all")
        self._context_seeder.seed(task, plan, self._task_description_cache)
        return plan, task

    # ── Terminal outcomes ─────────────────────────────────────────────

    def _deliver_success(
        self,
        task: Task,
        raw_result: str,
        review: EvaluationReport,
        task_description: str,
        retries: int,
    ) -> TaskResult:
        self._event_bus.publish(
            TaskExecutionSucceed.with_meta(
                task_id=task.id,
                result=raw_result,
                feedback=review.feedback,
            )
        )
        self._logger.info("Task result evaluate succeeded",
            task_id=task.id, curernt_task_retries=retries, result_length=len(raw_result))

        # Post-delivery learning: both run on daemon threads, never block delivery.
        self._extract_knowledge_async(task, raw_result)
        self._extract_preferences_async(task_description)

        self._finish_session_trace()
        return TaskResult(
            task_id=task.id,
            succeeded=True,
            result=raw_result,
            error_reason="",
            delivered_at=_time_now(),
        )

    def _on_plan_execution_failed(self, task: Task) -> TaskResult:
        """1.5.2 阶段执行未产出结果。"""
        self._event_bus.publish(
            TaskExecutionFailed.with_meta(
                task_id=task.id,
                reason="阶段执行未能产出结果，任务已停止。",
            )
        )
        self._logger.error("Task execute failed in pipeline, got None result", task_id=task.id)
        return self._fail(task.id, "Stage execution failed")

    def _on_task_retries_exhausted(
        self, task: Task, review: EvaluationReport, retries: int
    ) -> TaskResult:
        self._event_bus.publish(
            TaskExecutionFailed.with_meta(
                task_id=task.id,
                reason="最终结果多次复核仍未通过。",
                result="Exceed maximum retries but still failed",
                feedback=review.feedback,
                retry=retries,
                max_retries=self._max_task_retries,
            )
        )
        self._logger.error("Task result evaluate failed after max retries",
            task_id=task.id, max_retries=self._max_task_retries, feedback=review.feedback)
        return self._fail(task.id, "Task Result quality check failed after max retries")

    # ------------------------------------------------------------------
    # Phase wrapper
    # ------------------------------------------------------------------

    def _phase(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        failure_event: Callable[[Exception], DomainEvent] | None = None,
        log_message: str | None = None,
        **span_attrs: Any,
    ) -> Any:
        """Run one task-level phase inside a span, with a uniform failure path.

        On failure: log → publish *failure_event* (when given) → close the
        session trace → re-raise. Every task-level phase goes through here so
        the lifecycle methods stay free of try/except/span boilerplate.
        """
        try:
            with self._tracer.start_span(f"pipeline.{name}", "pipeline", **span_attrs):
                return fn()
        except Exception as exc:
            self._logger.error(log_message or f"Pipeline {name} failed", error=exc, **span_attrs)
            if failure_event is not None:
                self._event_bus.publish(failure_event(exc))
            self._finish_session_trace(error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _arm_checkpointing(self, task: Task, plan: Plan, task_description: str) -> None:
        """Save a checkpoint each time a non-final stage passes its evaluation."""
        def _on_stage_success(step_index: int) -> None:
            data = self._build_checkpoint_data(task, plan, task_description, step_index)
            CheckpointManager.save_async(data, str(task.id))

        self._stage_executor.set_stage_success_callback(_on_stage_success)

    def _build_checkpoint_data(
        self,
        task: Task,
        plan: Plan,
        task_description: str,
        completed_step_index: int,
    ) -> CheckpointData:
        return CheckpointData(
            task_id=task.id,
            user_id=task.user_id,
            task_description=task_description,
            task=task,
            plan=plan,
            completed_step_index=completed_step_index,
            conversation_history=self._context_manager.get_conversation_history(),
            tool_schemas=self._context_manager.get_tool_schemas(),
            model_selector_state=self._model_selector.get_state(),
            task_output_constraints=self._stage_executor.get_task_output_constraints(),
            task_goal=self._stage_executor.get_task_goal(),
            task_intent=self._stage_executor.get_task_intent(),
            task_recovery_feedback=self._stage_executor.get_task_recovery_feedback(),
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_plan_ready(self, task: Task, plan: Plan, kept_tools: list[str]) -> None:
        goals = " → ".join(s.goal[:35] for s in plan.step_list[:4])
        suffix = "..." if len(plan.step_list) > 4 else ""
        self._event_bus.publish(
            PlanGenerateSucceed.with_meta(
                task_id=task.id,
                plan=f"[{len(plan.step_list)} steps] {goals}{suffix}",
                steps=len(plan.step_list),
                required_tools=kept_tools or task.required_tools,
            )
        )

    def _fail(self, task_id: TaskId, reason: str) -> TaskResult:
        self._finish_session_trace(error=reason)
        return TaskResult(
            task_id=task_id,
            succeeded=False,
            result="",
            error_reason=reason,
            delivered_at=_time_now(),
        )


def _action_label(action) -> str:
    """Recovery action → its string value, tolerating a plain string."""
    return action.value if hasattr(action, "value") else str(action)
