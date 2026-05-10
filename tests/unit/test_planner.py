from __future__ import annotations

import json

from agent.models.plan.planner import _build_plan, _parse_steps
from schemas.ids import TaskId


def test_parse_steps_accepts_plan_object():
    content = json.dumps(
        {
            "steps": [
                {
                    "goal": "Gather inputs",
                    "description": "Collect the entities and constraints needed for execution.",
                    "key_results": ["Inputs are enumerated"],
                    "inputs": ["task.entities"],
                    "required_tools": ["search"],
                    "constraints": ["Respect explicit constraints"],
                    "risks": ["Ambiguous entity"],
                    "dependencies": [],
                    "execution_notes": "Verify missing data before continuing.",
                }
            ]
        }
    )

    raw_steps = _parse_steps(content)

    assert raw_steps[0]["goal"] == "Gather inputs"


def test_build_plan_maps_enriched_step_fields():
    raw_steps = [
        {
            "goal": "Verify inputs",
            "description": "Check the task entities and constraints.",
            "key_results": ["Entities confirmed"],
            "inputs": ["task.description", "task.entities"],
            "required_tools": ["search"],
            "constraints": ["Use only matched tools"],
            "risks": ["data_staleness"],
            "dependencies": ["1"],
            "execution_notes": "Normalize tool parameters before calling tools.",
        }
    ]

    plan = _build_plan(TaskId("task-1"), raw_steps)
    step = plan.step_list[0]

    assert step.goal == "Verify inputs"
    assert step.key_results == ["Entities confirmed"]
    assert step.inputs == ["task.description", "task.entities"]
    assert step.required_tools == ["search"]
    assert step.constraints == ["Use only matched tools"]
    assert step.risks == ["data_staleness"]
    assert step.dependencies == [1]
    assert step.execution_notes == "Normalize tool parameters before calling tools."


def test_build_plan_defaults_new_fields_for_legacy_steps():
    plan = _build_plan(
        TaskId("task-1"),
        [{"goal": "Do work", "description": "Run the legacy step.", "key_results": ["Done"]}],
    )
    step = plan.step_list[0]

    assert step.inputs == []
    assert step.required_tools == []
    assert step.constraints == []
    assert step.risks == []
    assert step.dependencies == []
    assert step.execution_notes == ""
