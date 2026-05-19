from __future__ import annotations

import pickle
import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.models.checkpoint.checkpoint_manager import CheckpointManager
from schemas.types import (
    CheckpointData,
    LLMMessage,
    ModelSelectorState,
    UserCommandType,
    UserMsgType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint_data(tmp_path: Path, step: int = 2) -> CheckpointData:
    return CheckpointData(
        task_id="task-abc",
        user_id="user-1",
        task_description="do something",
        task=None,
        plan=None,
        completed_step_index=step,
        conversation_history=[LLMMessage(role="user", content="hello")],
        tool_schemas=[{"name": "shell"}],
        model_selector_state=ModelSelectorState(
            current_provider="openai",
            priority_list=["openai", "anthropic"],
            breakers={},
        ),
        task_output_constraints="",
        task_goal="goal",
        task_intent="intent",
        task_recovery_feedback="",
    )


# ---------------------------------------------------------------------------
# CheckpointManager tests
# ---------------------------------------------------------------------------

class TestCheckpointManagerSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        data = _make_checkpoint_data(tmp_path, step=3)
        with patch.object(CheckpointManager, "BASE_DIR", tmp_path):
            CheckpointManager._save(data, "my_task")
            saved = list((tmp_path / "my_task").glob("*.cpt"))
            assert len(saved) == 1
            loaded = CheckpointManager.load(saved[0])
        assert loaded.task_id == data.task_id
        assert loaded.completed_step_index == data.completed_step_index
        assert loaded.task_description == data.task_description
        assert loaded.model_selector_state.current_provider == "openai"
        assert len(loaded.conversation_history) == 1

    def test_filename_format(self, tmp_path: Path) -> None:
        data = _make_checkpoint_data(tmp_path, step=1)
        with patch.object(CheckpointManager, "BASE_DIR", tmp_path):
            CheckpointManager._save(data, "task_name")
            files = list((tmp_path / "task_name").glob("*.cpt"))
        assert len(files) == 1
        name = files[0].name
        # {task_id}_stage_{step+1}_{YYYYMMDD}_{HHMMSS}.cpt
        pattern = r"^task-abc_stage_2_\d{8}_\d{6}\.cpt$"
        assert re.match(pattern, name), f"Unexpected filename: {name}"

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        data = _make_checkpoint_data(tmp_path)
        task_dir = tmp_path / "brand_new_task"
        assert not task_dir.exists()
        with patch.object(CheckpointManager, "BASE_DIR", tmp_path):
            CheckpointManager._save(data, "brand_new_task")
        assert task_dir.exists()

    def test_save_async_completes(self, tmp_path: Path) -> None:
        data = _make_checkpoint_data(tmp_path)
        with patch.object(CheckpointManager, "BASE_DIR", tmp_path):
            CheckpointManager.save_async(data, "async_task")
            deadline = time.monotonic() + 5.0
            files: list[Path] = []
            while time.monotonic() < deadline:
                files = list((tmp_path / "async_task").glob("*.cpt"))
                if files:
                    break
                time.sleep(0.05)
        assert files, "Checkpoint file was not created by save_async"

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load("/nonexistent/path/file.cpt")

    def test_conversation_history_preserved(self, tmp_path: Path) -> None:
        msgs = [
            LLMMessage(role="user", content="task"),
            LLMMessage(role="assistant", content="ok"),
        ]
        data = CheckpointData(
            task_id="task-abc",
            user_id="user-1",
            task_description="do something",
            task=None,
            plan=None,
            completed_step_index=2,
            conversation_history=msgs,
            tool_schemas=[],
            model_selector_state=ModelSelectorState(
                current_provider="openai",
                priority_list=["openai"],
                breakers={},
            ),
            task_output_constraints="",
            task_goal="goal",
            task_intent="intent",
            task_recovery_feedback="",
        )
        with patch.object(CheckpointManager, "BASE_DIR", tmp_path):
            CheckpointManager._save(data, "hist_task")
            files = list((tmp_path / "hist_task").glob("*.cpt"))
            loaded = CheckpointManager.load(files[0])
        assert len(loaded.conversation_history) == 2
        assert loaded.conversation_history[1].content == "ok"


# ---------------------------------------------------------------------------
# New enum values
# ---------------------------------------------------------------------------

class TestNewEnumValues:
    def test_user_msg_type_load_checkpoint_exists(self) -> None:
        assert UserMsgType.LOAD_CHECKPOINT == "LOAD_CHECKPOINT"

    def test_user_command_type_load_checkpoint_exists(self) -> None:
        assert UserCommandType.LOAD_CHECKPOINT == "LOAD_CHECKPOINT"

    def test_enums_are_serializable(self) -> None:
        data = pickle.dumps(UserMsgType.LOAD_CHECKPOINT)
        assert pickle.loads(data) == UserMsgType.LOAD_CHECKPOINT


# ---------------------------------------------------------------------------
# StageExecutor callback
# ---------------------------------------------------------------------------

class TestStageExecutorCallback:
    def test_callback_is_none_by_default(self) -> None:
        from agent.models.executor.stage_executor import StageExecutor
        executor = MagicMock(spec=StageExecutor)
        executor._on_stage_success = None
        assert executor._on_stage_success is None

    def test_set_stage_success_callback(self) -> None:
        from agent.models.executor.stage_executor import StageExecutor
        # Build a minimal executor via MagicMock to test the setter contract
        called_with: list[int] = []

        def cb(idx: int) -> None:
            called_with.append(idx)

        # Use a real partial instance via __new__ to avoid full DI
        executor = object.__new__(StageExecutor)
        executor._on_stage_success = None
        StageExecutor.set_stage_success_callback(executor, cb)
        executor._on_stage_success(5)
        assert called_with == [5]


# ---------------------------------------------------------------------------
# ModelSelectorState
# ---------------------------------------------------------------------------

class TestModelSelectorState:
    def test_pickle_roundtrip(self) -> None:
        from agent.models.model_routing.circuit_breaker import ProviderCircuitBreaker
        breaker = ProviderCircuitBreaker(provider_name="anthropic")
        state = ModelSelectorState(
            current_provider="anthropic",
            priority_list=["anthropic", "openai"],
            breakers={"anthropic": breaker},
        )
        raw = pickle.dumps(state)
        loaded = pickle.loads(raw)
        assert loaded.current_provider == "anthropic"
        assert loaded.priority_list == ["anthropic", "openai"]
        assert "anthropic" in loaded.breakers


# ---------------------------------------------------------------------------
# Pipeline routing
# ---------------------------------------------------------------------------

class TestPipelineRouting:
    def test_run_dispatches_to_run_new_task(self) -> None:
        from agent.application.pipeline import Pipeline
        pipeline = object.__new__(Pipeline)
        pipeline._run_new_task = MagicMock(return_value="new_result")
        pipeline._run_from_checkpoint = MagicMock(return_value="ckpt_result")

        result = Pipeline.run(pipeline, user_id="u1", task_description="do it")
        pipeline._run_new_task.assert_called_once_with("u1", "do it")
        pipeline._run_from_checkpoint.assert_not_called()

    def test_run_dispatches_to_run_from_checkpoint(self) -> None:
        from agent.application.pipeline import Pipeline
        from schemas.types import UserMsgType
        pipeline = object.__new__(Pipeline)
        pipeline._run_new_task = MagicMock(return_value="new_result")
        pipeline._run_from_checkpoint = MagicMock(return_value="ckpt_result")

        result = Pipeline.run(
            pipeline,
            user_id="u1",
            task_description="/path/to/file.cpt",
            msg_type=UserMsgType.LOAD_CHECKPOINT,
        )
        pipeline._run_from_checkpoint.assert_called_once_with("/path/to/file.cpt")
        pipeline._run_new_task.assert_not_called()


# ---------------------------------------------------------------------------
# PipelineThread routing
# ---------------------------------------------------------------------------

class TestPipelineThreadRouting:
    def test_load_checkpoint_routes_to_submit_checkpoint(self) -> None:
        from schemas.types import UserMessage, UserMsgType

        msg = UserMessage(
            msg_type=UserMsgType.LOAD_CHECKPOINT,
            user_id="u1",
            content="/tmp/task-abc_stage_2_20260101_120000.cpt",
        )
        driver = MagicMock()
        driver.submit_checkpoint.return_value = MagicMock(succeeded=True, task_id="t1")
        driver.submit_task.return_value = MagicMock(succeeded=True, task_id="t1")

        if msg.msg_type == UserMsgType.LOAD_CHECKPOINT:
            result = driver.submit_checkpoint(
                user_id=msg.user_id,
                checkpoint_path=msg.content.strip(),
            )
        else:
            result = driver.submit_task(
                user_id=msg.user_id,
                task_description=msg.content.strip(),
            )

        driver.submit_checkpoint.assert_called_once_with(
            user_id="u1",
            checkpoint_path="/tmp/task-abc_stage_2_20260101_120000.cpt",
        )
        driver.submit_task.assert_not_called()
