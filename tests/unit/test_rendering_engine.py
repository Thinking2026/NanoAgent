from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infra.rendering_engine import Jinja2PromptRenderer, TemplateNotFoundError, TemplateRenderError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def renderer():
    return Jinja2PromptRenderer()


def _make_step_input(source="entity", value="AAPL", step_ref=None, constraint_note=""):
    from schemas.task import StepInput
    return StepInput(source=source, value=value, step_ref=step_ref, constraint_note=constraint_note)


def _make_step_dependency(step_order=1, depends_on=None):
    from schemas.task import StepDependency
    return StepDependency(step_order=step_order, depends_on=depends_on or ["output_constraints"])


def _make_constraint(description="Use real-time data", strict=True, source="explicit"):
    from schemas.task import TaskConstraint
    return TaskConstraint(description=description, strict=strict, source=source)


def _make_entity(type_="stock_code", value="AAPL", raw="apple stock", normalized=True):
    from schemas.task import TaskEntity
    return TaskEntity(type=type_, value=value, raw=raw, normalized=normalized)


def _make_risk(category="data_staleness", description="Data may be stale", severity="high"):
    from schemas.task import RiskItem
    return RiskItem(category=category, description=description, severity=severity)


def _make_task(
    description="Test task",
    task_type="general",
    task_goal="complete the task",
    intent="complete the task",
    entities=None,
    action_constraints=None,
    risks=None,
    estimated_steps=1,
):
    from schemas.ids import TaskId, UserId
    from schemas.task import Task
    from utils.time.time import now
    from uuid import uuid4
    return Task(
        id=TaskId(str(uuid4())),
        user_id=UserId("user-1"),
        description=description,
        created_at=now(),
        task_type=task_type,
        task_goal=task_goal,
        intent=intent,
        entities=entities or [],
        action_constraints=action_constraints or [],
        risks=risks or [],
        estimated_steps=estimated_steps,
    )


# ---------------------------------------------------------------------------
# Filter tests via render_string
# ---------------------------------------------------------------------------

class TestFilterTojson:
    def test_basic_dict(self, renderer):
        result = renderer.render_string('{{ data | tojson }}', {"data": {"key": "value"}})
        assert json.loads(result) == {"key": "value"}

    def test_list(self, renderer):
        result = renderer.render_string('{{ data | tojson }}', {"data": [1, 2, 3]})
        assert json.loads(result) == [1, 2, 3]

    def test_ensure_ascii_false(self, renderer):
        result = renderer.render_string('{{ data | tojson(ensure_ascii=False) }}', {"data": {"name": "中文"}})
        assert "中文" in result

    def test_indent(self, renderer):
        result = renderer.render_string('{{ data | tojson(indent=2) }}', {"data": {"a": 1}})
        assert "\n" in result


class TestFilterFormatStepInputs:
    def test_empty_list(self, renderer):
        result = renderer.render_string('{{ items | format_step_inputs }}', {"items": []})
        assert "(none)" in result

    def test_entity_input(self, renderer):
        inp = _make_step_input(source="entity", value="AAPL", constraint_note="use real-time")
        result = renderer.render_string('{{ items | format_step_inputs }}', {"items": [inp]})
        assert "[entity]" in result
        assert "AAPL" in result
        assert "use real-time" in result

    def test_prior_step_input(self, renderer):
        inp = _make_step_input(source="prior_step", value="closing prices", step_ref=1)
        result = renderer.render_string('{{ items | format_step_inputs }}', {"items": [inp]})
        assert "prior_step" in result
        assert "from step 1" in result

    def test_no_constraint_note(self, renderer):
        inp = _make_step_input(source="file", value="data.csv", constraint_note="")
        result = renderer.render_string('{{ items | format_step_inputs }}', {"items": [inp]})
        assert "data.csv" in result
        assert "constraint:" not in result


class TestFilterFormatDependencies:
    def test_empty_list(self, renderer):
        result = renderer.render_string('{{ deps | format_dependencies }}', {"deps": []})
        assert "(none)" in result

    def test_single_dependency(self, renderer):
        dep = _make_step_dependency(step_order=2, depends_on=["key_results"])
        result = renderer.render_string('{{ deps | format_dependencies }}', {"deps": [dep]})
        assert "step 2" in result
        assert "key_results" in result


class TestFilterFormatConstraints:
    def test_empty_list(self, renderer):
        result = renderer.render_string('{{ cs | format_constraints }}', {"cs": []})
        assert "(none)" in result

    def test_strict_constraint(self, renderer):
        c = _make_constraint(description="Must use HTTPS", strict=True, source="explicit")
        result = renderer.render_string('{{ cs | format_constraints }}', {"cs": [c]})
        assert "STRICT" in result
        assert "Must use HTTPS" in result

    def test_soft_constraint(self, renderer):
        c = _make_constraint(description="Prefer JSON output", strict=False, source="implicit")
        result = renderer.render_string('{{ cs | format_constraints }}', {"cs": [c]})
        assert "soft" in result
        assert "Prefer JSON output" in result


class TestFilterFormatEntities:
    def test_empty_list(self, renderer):
        result = renderer.render_string('{{ es | format_entities }}', {"es": []})
        assert "(none)" in result

    def test_normalized_entity(self, renderer):
        e = _make_entity(type_="stock_code", value="AAPL", raw="apple stock", normalized=True)
        result = renderer.render_string('{{ es | format_entities }}', {"es": [e]})
        assert "[stock_code]" in result
        assert "AAPL" in result
        assert "apple stock" in result

    def test_non_normalized_entity(self, renderer):
        e = _make_entity(type_="date", value="2024-01-01", raw="2024-01-01", normalized=False)
        result = renderer.render_string('{{ es | format_entities }}', {"es": [e]})
        assert "2024-01-01" in result
        assert "raw:" not in result


class TestFilterFormatRisks:
    def test_empty_list(self, renderer):
        result = renderer.render_string('{{ rs | format_risks }}', {"rs": []})
        assert "(none)" in result

    def test_risk_item(self, renderer):
        r = _make_risk(category="data_staleness", description="Data may be stale", severity="high")
        result = renderer.render_string('{{ rs | format_risks }}', {"rs": [r]})
        assert "HIGH" in result
        assert "data_staleness" in result
        assert "Data may be stale" in result

    def test_string_risk(self, renderer):
        result = renderer.render_string('{{ rs | format_risks }}', {"rs": ["ambiguity"]})
        assert "ambiguity" in result


class TestFilterOrNone:
    def test_non_empty_string(self, renderer):
        result = renderer.render_string('{{ val | or_none }}', {"val": "hello"})
        assert result == "hello"

    def test_empty_string(self, renderer):
        result = renderer.render_string('{{ val | or_none }}', {"val": ""})
        assert result == "(none)"

    def test_none_value(self, renderer):
        result = renderer.render_string('{{ val | or_none }}', {"val": None})
        assert result == "(none)"


# ---------------------------------------------------------------------------
# Renderer core tests
# ---------------------------------------------------------------------------

class TestJinja2RendererCore:
    def test_render_string_basic(self, renderer):
        result = renderer.render_string("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_render_string_conditional(self, renderer):
        result = renderer.render_string(
            "{% if show %}visible{% endif %}",
            {"show": True},
        )
        assert result == "visible"

    def test_render_string_loop(self, renderer):
        result = renderer.render_string(
            "{% for item in items %}{{ item }},{% endfor %}",
            {"items": ["a", "b", "c"]},
        )
        assert result == "a,b,c,"

    def test_render_string_missing_variable_raises(self, renderer):
        with pytest.raises(TemplateRenderError):
            renderer.render_string("{{ missing_var }}", {})

    def test_render_template_not_found_raises(self, renderer):
        with pytest.raises(TemplateNotFoundError):
            renderer.render("nonexistent/template.j2", {})

    def test_trim_blocks_no_extra_blank_lines(self, renderer):
        result = renderer.render_string(
            "before\n{% if True %}\nafter\n{% endif %}",
            {},
        )
        assert "\n\n" not in result


# ---------------------------------------------------------------------------
# Template smoke tests — verify each template renders without error
# ---------------------------------------------------------------------------

class TestTemplateSmoke:
    """Render each template with minimal valid context and assert no exception."""

    def test_analyzer_system(self, renderer):
        result = renderer.render("analyzer/system.j2", {})
        assert "task analysis" in result.lower()

    def test_analyzer_user_prompt_minimal(self, renderer):
        result = renderer.render("analyzer/user_prompt.j2", {
            "task_description": "Analyze AAPL stock",
            "tool_schemas": [{"name": "get_price", "description": "Gets price", "parameters": {}}],
            "clarification_context": "",
            "preference_context": "",
        })
        assert "AAPL" in result
        assert "get_price" in result

    def test_analyzer_user_prompt_with_clarification(self, renderer):
        result = renderer.render("analyzer/user_prompt.j2", {
            "task_description": "Analyze AAPL",
            "tool_schemas": [],
            "clarification_context": "User said: use 2024 data",
            "preference_context": "Prefer concise output",
        })
        assert "User said: use 2024 data" in result
        assert "Prefer concise output" in result

    def test_quality_evaluator_system(self, renderer):
        result = renderer.render("quality_evaluator/system.j2", {})
        assert len(result) > 10

    def test_quality_evaluator_evaluate_plan(self, renderer):
        from schemas.task import Plan
        from schemas.ids import PlanId
        from utils.time.time import now
        from uuid import uuid4
        task = _make_task(description="Test task")
        plan = Plan(
            id=PlanId(str(uuid4())),
            task_id=task.id,
            step_list=[],
            created_at=now(),
        )
        result = renderer.render("quality_evaluator/evaluate_plan.j2", {
            "task": task,
            "plan": plan,
        })
        assert "Test task" in result

    def test_quality_evaluator_evaluate_task_result(self, renderer):
        task = _make_task(description="Test task")
        result = renderer.render("quality_evaluator/evaluate_task_result.j2", {
            "task": task,
            "result": "The answer is 42.",
        })
        assert "42" in result

    def test_quality_evaluator_evaluate_stage_result(self, renderer):
        from schemas.task import PlanStep
        from schemas.ids import PlanStepId
        from uuid import uuid4
        step = PlanStep(
            id=PlanStepId(str(uuid4())),
            goal="Fetch data",
            description="Call API",
            order=1,
            key_results=["Data fetched"],
            inputs=[],
            required_tools=[],
            action_constraints=[],
            risks=[],
            dependencies=[],
            execution_notes="",
            output_constraints="DataFrame",
        )
        result = renderer.render("quality_evaluator/evaluate_stage_result.j2", {
            "step": step,
            "result": "Got 252 rows",
        })
        assert "Fetch data" in result

    def test_knowledge_loader_system(self, renderer):
        result = renderer.render("knowledge_loader/system.j2", {})
        assert len(result) > 10

    def test_knowledge_loader_query_prompt(self, renderer):
        from types import SimpleNamespace
        task = _make_task(description="Find relevant knowledge")
        result = renderer.render("knowledge_loader/query_prompt.j2", {
            "task": task,
            "entries": [
                SimpleNamespace(
                    doc_title="SOP-001",
                    doc_type="runbook",
                    file_name="sop.md",
                    chunk_index=0,
                    content="Do X",
                )
            ],
        })
        assert "Find relevant knowledge" in result
        assert "SOP-001" in result

    def test_knowledge_manager_system(self, renderer):
        result = renderer.render("knowledge_manager/system.j2", {})
        assert len(result) > 10

    def test_personality_manager_system_extract(self, renderer):
        result = renderer.render("personality_manager/system_extract.j2", {})
        assert len(result) > 10

    def test_personality_manager_system_query(self, renderer):
        result = renderer.render("personality_manager/system_query.j2", {})
        assert len(result) > 10

    def test_personality_manager_query_prompt(self, renderer):
        task = _make_task(description="Write a report", task_type="copywriting")
        result = renderer.render("personality_manager/query_prompt.j2", {
            "task": task,
            "entries": [{"user_id": "u1", "keywords": ["concise"], "content": "Prefer bullet points"}],
        })
        assert "Write a report" in result
        assert "Prefer bullet points" in result

    def test_pipeline_plan_content(self, renderer):
        from schemas.task import Plan, PlanStep
        from schemas.ids import PlanId, PlanStepId, TaskId
        from utils.time.time import now
        from uuid import uuid4
        step = PlanStep(
            id=PlanStepId(str(uuid4())),
            goal="Fetch data",
            description="Call the API",
            order=1,
            key_results=["Data fetched"],
            inputs=[],
            required_tools=["api_tool"],
            action_constraints=[],
            risks=[],
            dependencies=[],
            execution_notes="",
            output_constraints="DataFrame",
        )
        plan = Plan(
            id=PlanId(str(uuid4())),
            task_id=TaskId(str(uuid4())),
            step_list=[step],
            created_at=now(),
        )
        result = renderer.render("pipeline/plan_content.j2", {"plan": plan})
        assert "Fetch data" in result

    def test_planner_system_make_plan(self, renderer):
        result = renderer.render("planner/system_make_plan.j2", {})
        assert "plan" in result.lower()

    def test_planner_system_renew_plan(self, renderer):
        result = renderer.render("planner/system_renew_plan.j2", {})
        assert len(result) > 10

    def test_planner_system_renew_step(self, renderer):
        result = renderer.render("planner/system_renew_step.j2", {})
        assert len(result) > 10

    def test_planner_system_renew_from_step(self, renderer):
        result = renderer.render("planner/system_renew_from_step.j2", {})
        assert len(result) > 10

    def test_planner_make_plan_user(self, renderer):
        task = _make_task(
            description="Analyze AAPL stock performance in 2024",
            task_type="data_analysis",
            task_goal="understand stock performance",
            intent="analyze AAPL closing prices",
            entities=[_make_entity()],
            action_constraints=[_make_constraint()],
            risks=[_make_risk()],
            estimated_steps=3,
        )
        result = renderer.render("planner/make_plan_user.j2", {
            "task": task,
            "extra_context": "",
        })
        assert "AAPL" in result or "analyze" in result.lower()

    def test_planner_make_plan_user_with_extra_context(self, renderer):
        task = _make_task(description="Analyze data")
        result = renderer.render("planner/make_plan_user.j2", {
            "task": task,
            "extra_context": "User clarification: use 2024 data only",
        })
        assert "User clarification" in result

    def test_planner_renew_plan_user(self, renderer):
        task = _make_task(description="Analyze data")
        result = renderer.render("planner/renew_plan_user.j2", {
            "task": task,
            "feedback": "Step 2 was missing a validation check",
        })
        assert "Step 2 was missing" in result

    def test_planner_renew_step_user(self, renderer):
        from schemas.task import PlanStep
        from schemas.ids import PlanStepId
        from uuid import uuid4
        step = PlanStep(
            id=PlanStepId(str(uuid4())),
            goal="Fetch data",
            description="Call API",
            order=2,
            key_results=["Data fetched"],
            inputs=[],
            required_tools=[],
            action_constraints=[],
            risks=[],
            dependencies=[],
            execution_notes="",
            output_constraints="DataFrame",
        )
        task = _make_task(task_goal="analyze data", intent="get stock prices")
        result = renderer.render("planner/renew_step_user.j2", {
            "task": task,
            "step": step,
            "step_dict": {"goal": "Fetch data", "description": "Call API"},
            "feedback": "Add error handling",
        })
        assert "Add error handling" in result

    def test_planner_renew_from_step_user(self, renderer):
        from schemas.task import PlanStep
        from schemas.ids import PlanStepId
        from uuid import uuid4
        step = PlanStep(
            id=PlanStepId(str(uuid4())),
            goal="Analyze data",
            description="Run analysis",
            order=2,
            key_results=[],
            inputs=[],
            required_tools=[],
            action_constraints=[],
            risks=[],
            dependencies=[],
            execution_notes="",
            output_constraints="",
        )
        task = _make_task(task_goal="analyze data", intent="get insights")
        result = renderer.render("planner/renew_from_step_user.j2", {
            "task": task,
            "preserved_steps": [],
            "steps_to_revise": [step],
            "steps_to_revise_dicts": [{"goal": "Analyze data"}],
            "feedback": "Include risk mitigation",
        })
        assert "Include risk mitigation" in result

    def test_stage_executor_stage_prompt(self, renderer):
        from schemas.task import PlanStep
        from schemas.ids import PlanStepId
        from uuid import uuid4
        step = PlanStep(
            id=PlanStepId(str(uuid4())),
            goal="Fetch AAPL prices",
            description="Call get_stock_price with ticker=AAPL",
            order=1,
            key_results=["252 rows returned"],
            inputs=[_make_step_input()],
            required_tools=["get_stock_price"],
            action_constraints=["Use real-time data"],
            risks=["data_staleness"],
            dependencies=[],
            execution_notes="ticker=AAPL, start=2024-01-01",
            output_constraints="DataFrame with date and close columns",
        )
        result = renderer.render("stage_executor/stage_prompt.j2", {
            "stage": step,
            "total_steps": 3,
        })
        assert "Fetch AAPL prices" in result
        assert "252 rows" in result
