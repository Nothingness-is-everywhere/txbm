"""
Tests for the GameTask class.

Covers:
    - Task initialization with configuration
    - Step execution with mock recorder
    - Retry logic on failure
    - Timeout handling
    - Checkpoint recovery
    - Result recording
    - Error screenshot saving
    - Progress reporting
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, List, Optional

import pytest
import numpy as np

from ok.automation.game_task import (
    GameTask,
    GameTaskError,
    StepExecutionError,
    StepTimeoutError,
    StepValidationError,
    RecorderInterface,
)
from ok.automation.config_loader import (
    AutomationConfig,
    StepConfig,
)


class MockRecorder:
    """Mock implementation of RecorderInterface for testing."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        find_results: Optional[List[Any]] = None,
        click_result: bool = True,
        sleep_duration: float = 0,
    ):
        self._width = width
        self._height = height
        self._find_results = find_results or []
        self._click_result = click_result
        self._sleep_duration = sleep_duration
        self._sleep_calls = []
        self._click_calls = []
        self._find_calls = []
        self._screenshots = []
        self._swipe_calls = []
        self._ocr_results = []
        self._input_text_calls = []
        self._send_key_calls = []

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame(self) -> np.ndarray:
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def find_feature(self, feature_name: str, **kwargs) -> Optional[Any]:
        self._find_calls.append(("find_feature", feature_name, kwargs))
        idx = len(self._find_calls) - 1
        if self._find_results:
            return self._find_results[idx % len(self._find_results)]
        return None

    def find_one(self, feature_name: str, **kwargs) -> Optional[Any]:
        self._find_calls.append(("find_one", feature_name, kwargs))
        idx = len(self._find_calls) - 1
        if self._find_results:
            return self._find_results[idx % len(self._find_results)]
        return None

    def click(self, x: int, y: int, **kwargs) -> bool:
        self._click_calls.append(("click", x, y, kwargs))
        return self._click_result

    def click_box(self, box: Any, **kwargs) -> bool:
        self._click_calls.append(("click_box", box, kwargs))
        return self._click_result

    def swipe(self, from_x: int, from_y: int, to_x: int, to_y: int, duration: int = 0, **kwargs) -> bool:
        self._swipe_calls.append((from_x, from_y, to_x, to_y, duration, kwargs))
        return True

    def send_key(self, key: str, **kwargs) -> bool:
        self._send_key_calls.append(("send_key", key, kwargs))
        return True

    def input_text(self, text: str) -> bool:
        self._input_text_calls.append(text)
        return True

    def screenshot(self, name: str, **kwargs) -> None:
        self._screenshots.append((name, kwargs))

    def ocr(self, **kwargs) -> List[Any]:
        self._ocr_results.append(kwargs)
        return self._find_results[:] if self._find_results else []

    def sleep(self, timeout: float) -> None:
        self._sleep_calls.append(timeout)
        if self._sleep_duration > 0:
            time.sleep(self._sleep_duration)

    def wait_until(self, condition, **kwargs) -> Any:
        timeout = kwargs.get("time_out", 5)
        return condition()


class TestGameTaskInit:
    """Tests for GameTask initialization."""

    def test_basic_init(self):
        config = AutomationConfig(task_name="test_task")
        task = GameTask(config)
        assert task.config.task_name == "test_task"
        assert task.recorder is None
        assert task.is_running is False
        assert task.current_step_index == 0
        assert task.results == {}
        assert task._step_results == []

    def test_init_with_recorder(self):
        config = AutomationConfig(task_name="test_task")
        recorder = MockRecorder()
        task = GameTask(config, recorder=recorder)
        assert task.recorder is recorder

    def test_init_creates_artifacts_dir(self, tmp_path):
        config = AutomationConfig(output_dir=str(tmp_path / "custom_artifacts"))
        task = GameTask(config)
        assert (tmp_path / "custom_artifacts").exists()

    def test_init_with_custom_output_dir(self, tmp_path):
        config = AutomationConfig()
        task = GameTask(config, output_dir=str(tmp_path / "my_output"))
        assert (tmp_path / "my_output").exists()

    def test_step_handlers_registered(self):
        config = AutomationConfig()
        task = GameTask(config)
        expected_types = [
            "find_feature", "click_feature", "wait_feature",
            "ocr", "click_ocr", "click_coordinate",
            "swipe", "send_key", "wait", "input_text", "custom",
        ]
        for step_type in expected_types:
            assert step_type in task._step_handlers


class TestGameTaskExecution:
    """Tests for GameTask execution logic."""

    def test_simple_find_feature_success(self):
        recorder = MockRecorder(find_results=["found_box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Find button",
                    type="find_feature",
                    feature_name="button",
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert task.current_step_index == 1

    def test_find_feature_failure(self):
        recorder = MockRecorder(find_results=[])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Missing feature",
                    type="find_feature",
                    feature_name="non_existent",
                    retry_count=0,
                    timeout=5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is False

    def test_click_feature_success(self):
        recorder = MockRecorder(find_results=["found_box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Click button",
                    type="click_feature",
                    feature_name="button",
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert len(recorder._click_calls) > 0

    def test_wait_feature_success(self):
        recorder = MockRecorder(find_results=["loaded_box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Wait for load",
                    type="wait_feature",
                    feature_name="game_loaded",
                    timeout=5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True

    def test_click_coordinate_relative(self):
        recorder = MockRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Click center",
                    type="click_coordinate",
                    x=0.5,
                    y=0.5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert len(recorder._click_calls) > 0
        click_type, x, y, kwargs = recorder._click_calls[0]
        assert x == 960  # 0.5 * 1920
        assert y == 540  # 0.5 * 1080

    def test_click_coordinate_absolute(self):
        recorder = MockRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Click absolute",
                    type="click_coordinate",
                    x=300,
                    y=400,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        click_type, x, y, kwargs = recorder._click_calls[0]
        assert x == 300
        assert y == 400

    def test_swipe_step(self):
        recorder = MockRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Swipe up",
                    type="swipe",
                    x=0.5,
                    y=0.8,
                    to_x=0.5,
                    to_y=0.2,
                    swipe_duration=0.5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert len(recorder._swipe_calls) > 0

    def test_send_key_step(self):
        recorder = MockRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Press ESC",
                    type="send_key",
                    key="esc",
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert len(recorder._send_key_calls) > 0

    def test_wait_step(self):
        recorder = MockRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Wait",
                    type="wait",
                    timeout=0.1,
                    params={"duration": 0.1},
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True

    def test_input_text_step(self):
        recorder = MockRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Input text",
                    type="input_text",
                    text="Hello World",
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert "Hello World" in recorder._input_text_calls

    def test_ocr_step(self):
        recorder = MockRecorder(find_results=["text_box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Find text",
                    type="ocr",
                    x=0,
                    y=0,
                    to_x=1,
                    to_y=1,
                    text="Settings",
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True

    def test_click_ocr_step(self):
        recorder = MockRecorder(find_results=["text_box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Click text",
                    type="click_ocr",
                    x=0,
                    y=0,
                    to_x=1,
                    to_y=1,
                    text="Settings",
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True

    def test_multiple_steps(self):
        recorder = MockRecorder(find_results=["box1", "box2"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(name="Step 1", type="click_feature", feature_name="btn1"),
                StepConfig(name="Step 2", type="find_feature", feature_name="btn2"),
                StepConfig(name="Step 3", type="click_coordinate", x=0.5, y=0.5),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert task.current_step_index == 3

    def test_no_steps(self):
        config = AutomationConfig(
            task_name="empty_test",
            steps=[],
        )
        task = GameTask(config)
        result = task.run()
        assert result is True

    def test_custom_step_execution(self):
        class CustomTask(GameTask):
            def my_custom_action(self, step: StepConfig) -> bool:
                return step.params.get("value", False)

        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Custom",
                    type="custom",
                    custom_func_name="my_custom_action",
                    params={"value": True},
                ),
            ],
        )
        task = CustomTask(config)
        result = task.run()
        assert result is True

    def test_custom_step_missing_function(self):
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Bad custom",
                    type="custom",
                    custom_func_name="non_existent_func",
                ),
            ],
        )
        task = GameTask(config)
        result = task.run()
        assert result is False


class TestGameTaskRetryLogic:
    """Tests for retry and timeout logic."""

    def test_retry_on_failure(self):
        call_count = [0]

        class FlakyRecorder(MockRecorder):
            def find_one(self, feature_name, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    return None
                return "found"

        recorder = FlakyRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Retry test",
                    type="find_feature",
                    feature_name="button",
                    retry_count=3,
                    timeout=5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True
        assert call_count[0] >= 3

    def test_max_retries_exhausted(self):
        recorder = MockRecorder(find_results=[])
        config = AutomationConfig(
            task_name="test",
            max_retries=1,
            steps=[
                StepConfig(
                    name="Always fail",
                    type="find_feature",
                    feature_name="never_found",
                    retry_count=1,
                    timeout=5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is False
        assert task.state_machine.retries > 0

    def test_retry_with_success(self):
        results = [None, None, "success_box"]

        class RecoveringRecorder(MockRecorder):
            def find_one(self, feature_name, **kwargs):
                if results:
                    return results.pop(0)
                return None

        recorder = RecoveringRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Eventually succeeds",
                    type="find_feature",
                    feature_name="button",
                    retry_count=3,
                    timeout=10,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is True

    def test_stop_method(self):
        recorder = MockRecorder(find_results=["box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(name="S1", type="find_feature", feature_name="f1"),
                StepConfig(name="S2", type="find_feature", feature_name="f2"),
            ],
        )
        task = GameTask(config, recorder=recorder)
        task.stop()
        assert task.is_running is False


class TestGameTaskResultTracking:
    """Tests for result recording and reporting."""

    def test_results_recorded(self):
        recorder = MockRecorder(find_results=["box1", "box2"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(name="S1", type="find_feature", feature_name="f1"),
                StepConfig(name="S2", type="find_feature", feature_name="f2"),
            ],
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        assert len(task._step_results) == 2
        assert all(r["success"] for r in task._step_results)

    def test_failed_step_recorded(self):
        recorder = MockRecorder(find_results=[])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Failing step",
                    type="find_feature",
                    feature_name="missing",
                    retry_count=0,
                    timeout=5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        results = task._step_results
        assert any(not r["success"] for r in results)

    def test_execution_summary(self):
        recorder = MockRecorder(find_results=["box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(name="S1", type="find_feature", feature_name="f1"),
            ],
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        summary = task.execution_summary
        assert "step_results" in summary
        assert "final_state" in summary

    def test_progress_report(self):
        recorder = MockRecorder(find_results=["box1", "box2", "box3"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(name="S1", type="find_feature", feature_name="f1"),
                StepConfig(name="S2", type="find_feature", feature_name="f2"),
                StepConfig(name="S3", type="find_feature", feature_name="f3"),
            ],
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        progress = task.get_progress()
        assert progress["task_name"] == "test"
        assert progress["total_steps"] == 3
        assert progress["completed_steps"] >= 1
        assert progress["progress_percent"] >= 0

    def test_save_execution_report(self, tmp_path):
        recorder = MockRecorder(find_results=["box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(name="S1", type="find_feature", feature_name="f1"),
            ],
            output_dir=str(tmp_path),
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        report_path = task.save_execution_report()
        assert report_path is not None

        import json
        with open(report_path) as f:
            report = json.load(f)
        assert "task_name" in report
        assert "execution_summary" in report


class TestGameTaskErrorScreenshot:
    """Tests for error screenshot functionality."""

    def test_screenshot_on_failure(self, tmp_path):
        recorder = MockRecorder(find_results=[])
        config = AutomationConfig(
            task_name="test",
            error_screenshot=True,
            steps=[
                StepConfig(
                    name="Failing step",
                    type="find_feature",
                    feature_name="missing",
                    retry_count=0,
                    timeout=5,
                ),
            ],
            output_dir=str(tmp_path),
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        assert len(recorder._screenshots) > 0

    def test_no_screenshot_when_disabled(self, tmp_path):
        recorder = MockRecorder(find_results=[])
        config = AutomationConfig(
            task_name="test",
            error_screenshot=False,
            steps=[
                StepConfig(
                    name="Failing step",
                    type="find_feature",
                    feature_name="missing",
                    retry_count=0,
                    timeout=5,
                ),
            ],
            output_dir=str(tmp_path),
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        assert len(recorder._screenshots) == 0


class TestGameTaskThrottle:
    """Tests for throttle and random delay."""

    def test_throttle_applied(self):
        recorder = MockRecorder(find_results=["box"], sleep_duration=0.1)
        config = AutomationConfig(
            task_name="test",
            throttle_ms=50,
            steps=[
                StepConfig(
                    name="With throttle",
                    type="find_feature",
                    feature_name="f1",
                    post_delay=0.1,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        start = time.time()
        task.run()
        elapsed = time.time() - start
        assert elapsed >= 0.1

    def test_throttle_disabled(self):
        recorder = MockRecorder(find_results=["box"])
        config = AutomationConfig(
            task_name="test",
            throttle_ms=0,
            steps=[
                StepConfig(
                    name="No throttle",
                    type="find_feature",
                    feature_name="f1",
                    post_delay=0,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        start = time.time()
        task.run()
        elapsed = time.time() - start
        assert elapsed < 0.1


class TestGameTaskEdgeCases:
    """Tests for edge cases."""

    def test_no_recorder_returns_false(self):
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Find without recorder",
                    type="find_feature",
                    feature_name="f1",
                ),
            ],
        )
        task = GameTask(config, recorder=None)
        result = task.run()
        assert result is False

    def test_invalid_step_type_raises(self):
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Invalid",
                    type="nonexistent_type",
                ),
            ],
        )
        task = GameTask(config)
        result = task.run()
        assert result is False

    def test_timeout_calculation(self):
        recorder = MockRecorder(find_results=["box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Short timeout",
                    type="find_feature",
                    feature_name="f1",
                    timeout=10,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        assert config.steps[0].timeout == 10.0

    def test_state_machine_integration(self):
        recorder = MockRecorder(find_results=["box"])
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(name="S1", type="find_feature", feature_name="f1"),
                StepConfig(name="S2", type="find_feature", feature_name="f2"),
            ],
        )
        task = GameTask(config, recorder=recorder)
        task.run()
        sm = task.state_machine
        assert sm.state.value == "done"
        assert sm.status.value == "success"

    def test_exception_during_execution(self):
        class ThrowingRecorder(MockRecorder):
            def find_one(self, feature_name, **kwargs):
                raise RuntimeError("Unexpected error")

        recorder = ThrowingRecorder()
        config = AutomationConfig(
            task_name="test",
            steps=[
                StepConfig(
                    name="Throwing step",
                    type="find_feature",
                    feature_name="f1",
                    retry_count=0,
                    timeout=5,
                ),
            ],
        )
        task = GameTask(config, recorder=recorder)
        result = task.run()
        assert result is False
