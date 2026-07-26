"""
Game task base class with state machine, retry logic, and checkpoint recovery.

Provides a reusable framework for game automation tasks with:
    - State machine-driven step execution
    - Configurable retry and timeout handling
    - Checkpoint-based interruption recovery
    - Structured execution logging
    - Integration with ok-script's BaseTask
"""

import time
import logging
import traceback
import random
from typing import Callable, Optional, List, Dict, Any, Protocol
from pathlib import Path
from datetime import datetime

from ok.automation.state_machine import GameStateMachine, TaskState, TaskStatus
from ok.automation.config_loader import AutomationConfig, StepConfig


class GameTaskError(Exception):
    """Base exception for game task errors."""

    pass


class StepExecutionError(GameTaskError):
    """Error during step execution."""

    def __init__(self, step_name: str, original_error: Exception):
        self.step_name = step_name
        self.original_error = original_error
        super().__init__(f"Step '{step_name}' failed: {original_error}")


class StepTimeoutError(GameTaskError):
    """Timeout during step execution."""

    def __init__(self, step_name: str, timeout: float):
        self.step_name = step_name
        self.timeout = timeout
        super().__init__(f"Step '{step_name}' timed out after {timeout}s")


class StepValidationError(GameTaskError):
    """Error in step configuration validation."""

    pass


class RecorderInterface(Protocol):
    """Protocol for objects that can perform game automation actions."""

    def find_feature(self, feature_name: str, **kwargs) -> Optional[Any]: ...

    def find_one(self, feature_name: str, **kwargs) -> Optional[Any]: ...

    def click(self, x: int, y: int, **kwargs) -> bool: ...

    def click_box(self, box: Any, **kwargs) -> bool: ...

    def swipe(self, from_x: int, from_y: int, to_x: int, to_y: int, **kwargs) -> bool: ...

    def send_key(self, key: str, **kwargs) -> bool: ...

    def input_text(self, text: str) -> bool: ...

    def screenshot(self, name: str, **kwargs) -> None: ...

    def ocr(self, **kwargs) -> List[Any]: ...

    def sleep(self, timeout: float) -> None: ...

    def wait_until(self, condition: Callable, **kwargs) -> Any: ...

    @property
    def frame(self) -> Any: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


class GameTask:
    """
    Base class for game automation tasks.

    Provides state machine-driven execution with retry, timeout, and checkpoint
    recovery capabilities. Integrates with ok-script's BaseTask via the
    RecorderInterface protocol.

    Usage:
        class MyGameTask(GameTask):
            def _execute_find_feature(self, step):
                box = self.recorder.find_one(step.feature_name)
                return box is not None

        config = AutomationConfig(
            task_name="my_task",
            steps=[StepConfig(name="Find button", type="find_feature", feature_name="btn")]
        )
        task = MyGameTask(config, recorder=my_recorder)
        task.run()

    Attributes:
        config: The automation configuration.
        state_machine: The underlying state machine.
        recorder: Interface for performing game automation actions.
        current_step_index: Index of the currently executing step.
        results: Dictionary storing step execution results.
        artifacts_dir: Directory for saving screenshots and logs.
    """

    def __init__(
        self,
        config: AutomationConfig,
        recorder: Optional[RecorderInterface] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the game task.

        Args:
            config: Automation configuration with steps and parameters.
            recorder: Interface for game automation actions (ok-script BaseTask).
            output_dir: Directory for artifacts (screenshots, logs).
        """
        self.config = config
        self.recorder = recorder
        self.state_machine = GameStateMachine(
            max_retries=config.max_retries,
            step_timeout=30.0,
            checkpoint_interval=1,
        )
        self.current_step_index: int = 0
        self.results: Dict[str, Any] = {}
        self._step_results: List[Dict[str, Any]] = []
        self._start_time: float = 0
        self._artifacts_dir: Path = Path(
            output_dir or config.output_dir or "artifacts"
        )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"GameTask.{config.task_name}")
        self._is_running: bool = False
        self._step_handlers: Dict[str, Callable] = {
            "find_feature": self._execute_find_feature,
            "click_feature": self._execute_click_feature,
            "wait_feature": self._execute_wait_feature,
            "ocr": self._execute_ocr,
            "click_ocr": self._execute_click_ocr,
            "click_coordinate": self._execute_click_coordinate,
            "swipe": self._execute_swipe,
            "send_key": self._execute_send_key,
            "wait": self._execute_wait,
            "input_text": self._execute_input_text,
            "custom": self._execute_custom,
        }

    @property
    def is_running(self) -> bool:
        """Check if the task is currently running."""
        return self._is_running

    @property
    def execution_summary(self) -> Dict[str, Any]:
        """Get execution summary from the state machine."""
        summary = self.state_machine.get_execution_summary()
        summary["step_results"] = self._step_results
        return summary

    def run(self) -> bool:
        """
        Execute the full task flow through the state machine.

        Returns:
            True if the task completed successfully.

        Raises:
            GameTaskError: On unrecoverable errors.
        """
        self._is_running = True
        self._start_time = time.time()
        self.current_step_index = 0

        self._logger.info(f"Starting task: {self.config.task_name}")
        self.state_machine.reset()
        self.state_machine.navigate("Task started")

        try:
            if not self.config.steps:
                self._logger.info("No steps to execute, completing task")
                self.state_machine.execute("Empty task - no steps")
                self.state_machine.verify("Empty task verified")
                self.state_machine.complete("Empty task completed")
                return True

            for i, step in enumerate(self.config.steps):
                if self.state_machine.is_terminal:
                    break

                self.current_step_index = i
                step_result = self._execute_step_with_retry(step, i)

                if not step_result:
                    self._logger.error(
                        f"Step '{step.name}' failed after all retries"
                    )
                    self.state_machine.fail(
                        f"Step '{step.name}' failed"
                    )
                    self._save_error_screenshot(step)
                    break

                self._record_step_result(step, step_result, success=True)

            if self.state_machine.state == TaskState.FAIL:
                return False

            self.state_machine.complete("All steps executed successfully")
            self.current_step_index = len(self.config.steps)
            self._logger.info(
                f"Task '{self.config.task_name}' completed successfully "
                f"in {self.state_machine.elapsed_time:.2f}s"
            )
            return True

        except GameTaskError:
            self._logger.error("GameTaskError occurred")
            self.state_machine.fail("GameTaskError")
            return False

        except Exception as e:
            self._logger.error(f"Task execution error: {e}")
            self._logger.error(traceback.format_exc())
            self.state_machine.fail(f"Unexpected error: {e}")
            self._save_error_screenshot()
            self._record_step_result(
                self.config.steps[self.current_step_index]
                if self.current_step_index < len(self.config.steps)
                else None,
                None,
                success=False,
                error=str(e),
            )
            return False

        finally:
            self._is_running = False

    def stop(self) -> None:
        """Stop the task execution gracefully."""
        self._is_running = False
        self._logger.info("Task stopped by user")

    def _execute_step_with_retry(
        self, step: StepConfig, index: int
    ) -> bool:
        """
        Execute a step with retry logic and timeout.

        Args:
            step: Step configuration to execute.
            index: Step index in the flow.

        Returns:
            True if the step executed successfully.
        """
        max_retries = min(step.retry_count, self.config.max_retries)

        for attempt in range(max_retries + 1):
            if self.state_machine.is_terminal:
                return False

            if attempt > 0:
                self._logger.info(
                    f"Retrying step '{step.name}' (attempt {attempt + 1}/{max_retries + 1})"
                )
                self.state_machine.retry(
                    f"Retry {attempt + 1}/{max_retries + 1}"
                )
                self._apply_throttle(step.interval)

            try:
                self.state_machine.execute(f"Executing step: {step.name}")
                result = self._execute_step(step)

                if result:
                    self._logger.info(f"Step '{step.name}' completed successfully")
                    self.state_machine.verify(f"Step verified: {step.name}")
                    self._apply_throttle(step.post_delay)
                    return True
                else:
                    self._logger.warning(
                        f"Step '{step.name}' returned False (attempt {attempt + 1})"
                    )
                    self._record_step_result(
                        step, None, success=False, error="Step returned False"
                    )
                    if attempt >= max_retries:
                        return False

            except StepTimeoutError as e:
                self._logger.warning(
                    f"Step '{step.name}' timed out: {e}"
                )
                self._record_step_result(
                    step, None, success=False, error=str(e)
                )
                if attempt >= max_retries:
                    return False

            except Exception as e:
                self._logger.warning(
                    f"Step '{step.name}' failed (attempt {attempt + 1}): {e}"
                )
                self._record_step_result(
                    step, None, success=False, error=str(e)
                )
                if attempt >= max_retries:
                    return False

        return False

    def _execute_step(self, step: StepConfig) -> bool:
        """
        Execute a single step by dispatching to the appropriate handler.

        Args:
            step: Step configuration.

        Returns:
            True if the step was successful.

        Raises:
            StepTimeoutError: If the step exceeds its timeout.
            GameTaskError: If the step handler raises an exception.
        """
        handler = self._step_handlers.get(step.type)
        if not handler:
            raise GameTaskError(
                f"Unknown step type: {step.type}"
            )

        timeout = step.timeout
        start_time = time.time()

        try:
            result = handler(step)

            if time.time() - start_time > timeout:
                raise StepTimeoutError(step.name, timeout)

            return result

        except StepTimeoutError:
            raise
        except GameTaskError:
            raise
        except Exception as e:
            raise StepExecutionError(step.name, e)

    def _execute_find_feature(self, step: StepConfig) -> bool:
        """Execute a find_feature step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for find_feature")
            return False

        box = self.recorder.find_one(
            step.feature_name,
            threshold=step.threshold,
        )
        found = box is not None
        self._logger.debug(
            f"find_feature '{step.feature_name}': found={found}, box={box}"
        )
        return found

    def _execute_click_feature(self, step: StepConfig) -> bool:
        """Execute a click_feature step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for click_feature")
            return False

        box = self.recorder.find_one(
            step.feature_name,
            threshold=step.threshold,
        )
        if box is None:
            self._logger.warning(f"Feature '{step.feature_name}' not found")
            return False

        result = self.recorder.click_box(box)
        self._logger.debug(
            f"click_feature '{step.feature_name}': clicked={result}"
        )
        self._apply_throttle(step.post_delay)
        return result

    def _execute_wait_feature(self, step: StepConfig) -> bool:
        """Execute a wait_feature step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for wait_feature")
            return False

        def condition():
            return self.recorder.find_one(
                step.feature_name,
                threshold=step.threshold,
            )

        try:
            result = self.recorder.wait_until(
                condition,
                time_out=int(step.timeout),
            )
            found = result is not None
            self._logger.debug(
                f"wait_feature '{step.feature_name}': found={found}"
            )
            return found
        except Exception as e:
            self._logger.warning(f"wait_feature failed: {e}")
            return False

    def _execute_ocr(self, step: StepConfig) -> bool:
        """Execute an OCR step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for ocr")
            return False

        boxes = self.recorder.ocr(
            x=step.x or 0,
            y=step.y or 0,
            to_x=step.to_x or 1,
            to_y=step.to_y or 1,
            match=step.text,
        )
        found = len(boxes) > 0
        self._logger.debug(f"ocr '{step.text}': found={found}")
        return found

    def _execute_click_ocr(self, step: StepConfig) -> bool:
        """Execute a click_ocr step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for click_ocr")
            return False

        boxes = self.recorder.ocr(
            x=step.x or 0,
            y=step.y or 0,
            to_x=step.to_x or 1,
            to_y=step.to_y or 1,
            match=step.text,
        )
        if not boxes:
            self._logger.warning(f"OCR text '{step.text}' not found")
            return False

        result = self.recorder.click_box(boxes[0])
        self._logger.debug(
            f"click_ocr '{step.text}': clicked={result}"
        )
        self._apply_throttle(step.post_delay)
        return result

    def _execute_click_coordinate(self, step: StepConfig) -> bool:
        """Execute a click_coordinate step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for click_coordinate")
            return False

        x = step.x or 0.5
        y = step.y or 0.5

        if 0 <= x <= 1 and 0 <= y <= 1:
            pixel_x = int(self.recorder.width * x)
            pixel_y = int(self.recorder.height * y)
        else:
            pixel_x = int(x)
            pixel_y = int(y)

        result = self.recorder.click(pixel_x, pixel_y)
        self._logger.debug(
            f"click_coordinate ({pixel_x}, {pixel_y}): result={result}"
        )
        self._apply_throttle(step.post_delay)
        return result

    def _execute_swipe(self, step: StepConfig) -> bool:
        """Execute a swipe step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for swipe")
            return False

        from_x = int(self.recorder.width * (step.x or 0.5))
        from_y = int(self.recorder.height * (step.y or 0.5))
        to_x = int(self.recorder.width * (step.to_x or 0.5))
        to_y = int(self.recorder.height * (step.to_y or 0.5))

        duration_ms = int(step.swipe_duration * 1000)
        self.recorder.swipe(from_x, from_y, to_x, to_y, duration_ms)
        self._logger.debug(
            f"swipe ({from_x},{from_y}) -> ({to_x},{to_y}): duration={duration_ms}ms"
        )
        self._apply_throttle(step.post_delay)
        return True

    def _execute_send_key(self, step: StepConfig) -> bool:
        """Execute a send_key step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for send_key")
            return False

        result = self.recorder.send_key(step.key)
        self._logger.debug(f"send_key '{step.key}': result={result}")
        self._apply_throttle(step.post_delay)
        return result

    def _execute_wait(self, step: StepConfig) -> bool:
        """Execute a wait step."""
        wait_time = step.params.get("duration", step.timeout)
        if self.recorder:
            self.recorder.sleep(wait_time)
        else:
            time.sleep(wait_time)
        self._logger.debug(f"wait: {wait_time}s")
        return True

    def _execute_input_text(self, step: StepConfig) -> bool:
        """Execute an input_text step."""
        if self.recorder is None:
            self._logger.warning("No recorder available for input_text")
            return False

        result = self.recorder.input_text(step.text or "")
        self._logger.debug(f"input_text '{step.text}': result={result}")
        self._apply_throttle(step.post_delay)
        return result

    def _execute_custom(self, step: StepConfig) -> bool:
        """Execute a custom step by calling a registered function."""
        func_name = step.custom_func_name
        if not func_name:
            raise GameTaskError("Custom step has no function name")

        custom_func = getattr(self, func_name, None)
        if custom_func is None:
            raise GameTaskError(
                f"Custom function '{func_name}' not found on task"
            )

        result = custom_func(step)
        self._logger.debug(f"custom '{func_name}': result={result}")
        return bool(result)

    def _apply_throttle(self, delay: float) -> None:
        """Apply throttle delay with optional random variation."""
        if delay <= 0:
            return

        throttle_ms = self.config.throttle_ms / 1000.0
        actual_delay = max(delay, throttle_ms)

        if self.config.random_delay > 0:
            variation = random.uniform(
                -self.config.random_delay * actual_delay,
                self.config.random_delay * actual_delay,
            )
            actual_delay = max(0, actual_delay + variation)

        if self.recorder:
            self.recorder.sleep(actual_delay)
        else:
            time.sleep(actual_delay)

    def _record_step_result(
        self,
        step: Optional[StepConfig],
        result: Any,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        Record the result of a step execution.

        Args:
            step: The step that was executed.
            result: The result returned by the step.
            success: Whether the step was successful.
            error: Error message if the step failed.
        """
        step_name = step.name if step else "unknown"
        record = {
            "step_name": step_name,
            "step_type": step.type if step else "unknown",
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "duration": self.state_machine.elapsed_time,
            "error": error,
        }

        self._step_results.append(record)

        key = f"step_{step_name}"
        self.results[key] = {
            "success": success,
            "result": str(result) if result else None,
            "error": error,
        }

    def _save_error_screenshot(
        self, step: Optional[StepConfig] = None
    ) -> Optional[str]:
        """
        Save a screenshot when an error occurs.

        Args:
            step: The step that failed.

        Returns:
            Path to the saved screenshot, or None.
        """
        if not self.config.error_screenshot or self.recorder is None:
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_name = step.name if step else "unknown"
            filename = f"error_{timestamp}_{step_name}.png"
            filepath = self._artifacts_dir / filename

            self.recorder.screenshot(str(filepath))
            self._logger.info(f"Error screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            self._logger.warning(f"Failed to save error screenshot: {e}")
            return None

    def save_execution_report(self) -> str:
        """
        Save the execution report to the artifacts directory.

        Returns:
            Path to the saved report file.
        """
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self._artifacts_dir / f"report_{timestamp}.json"

        report = {
            "task_name": self.config.task_name,
            "description": self.config.description,
            "execution_summary": self.execution_summary,
            "state_machine_summary": self.state_machine.get_execution_summary(),
            "step_results": self._step_results,
            "start_time": datetime.fromtimestamp(
                self._start_time
            ).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": round(self.state_machine.elapsed_time, 2),
        }

        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._logger.info(f"Execution report saved: {report_file}")
        return str(report_file)

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current task progress.

        Returns:
            Progress information dictionary.
        """
        total_steps = len(self.config.steps)
        completed_steps = len(
            [r for r in self._step_results if r.get("success")]
        )

        return {
            "task_name": self.config.task_name,
            "current_state": self.state_machine.state.value,
            "current_status": self.state_machine.status.value,
            "current_step_index": self.current_step_index,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "progress_percent": (
                round(completed_steps / total_steps * 100, 1)
                if total_steps > 0
                else 0
            ),
            "elapsed_time": round(self.state_machine.elapsed_time, 2),
            "retries": self.state_machine.retries,
            "is_running": self._is_running,
        }
