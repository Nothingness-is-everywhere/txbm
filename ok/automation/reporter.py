"""
Structured logging and execution report generation for automation tasks.

Provides:
    - Structured log entries with timestamps, task names, steps, actions, and results
    - Automatic screenshot saving on failure
    - JSON report generation with success/failure statistics and timing data
    - Integration with ok-script's existing logging system
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    """
    A structured log entry for task execution.

    Attributes:
        timestamp: When the entry was created.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        task_name: Name of the task.
        step_name: Name of the step (if applicable).
        action: Action performed (e.g., 'find_feature', 'click').
        result: Result of the action ('success', 'failed', 'timeout').
        duration_ms: Duration of the action in milliseconds.
        message: Human-readable message.
        details: Additional context details.
    """

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    level: str = "INFO"
    task_name: str = ""
    step_name: str = ""
    action: str = ""
    result: str = ""
    duration_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepReport:
    """
    Report for a single step execution.

    Attributes:
        step_name: Name of the step.
        step_type: Type of the step.
        success: Whether the step succeeded.
        duration_ms: Execution duration in milliseconds.
        retries: Number of retries performed.
        error: Error message if the step failed.
        timestamp: When the step was executed.
    """

    step_name: str
    step_type: str
    success: bool
    duration_ms: float = 0.0
    retries: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TaskReport:
    """
    Complete execution report for a task.

    Attributes:
        task_name: Name of the task.
        description: Task description.
        start_time: Task start timestamp.
        end_time: Task end timestamp.
        total_duration_ms: Total execution duration.
        success: Whether the task completed successfully.
        total_steps: Total number of steps.
        successful_steps: Number of successful steps.
        failed_steps: Number of failed steps.
        total_retries: Total number of retries across all steps.
        average_step_duration_ms: Average duration per step.
        steps: List of individual step reports.
        screenshots: List of saved screenshot paths.
        logs: All structured log entries.
        state_machine_summary: Summary from the state machine.
    """

    task_name: str
    description: str = ""
    start_time: str = ""
    end_time: str = ""
    total_duration_ms: float = 0.0
    success: bool = False
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    total_retries: int = 0
    average_step_duration_ms: float = 0.0
    steps: List[StepReport] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    logs: List[LogEntry] = field(default_factory=list)
    state_machine_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            "task_name": self.task_name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "total_retries": self.total_retries,
            "average_step_duration_ms": round(
                self.average_step_duration_ms, 2
            ),
            "steps": [asdict(step) for step in self.steps],
            "screenshots": self.screenshots,
            "log_count": len(self.logs),
            "state_machine_summary": self.state_machine_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert the report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class TaskReporter:
    """
    Collects structured logs and generates execution reports for automation tasks.

    Usage:
        reporter = TaskReporter("my_task", output_dir="artifacts")
        reporter.log_step_start("Step 1", "find_feature")
        reporter.log_step_end("Step 1", True, duration_ms=150.5)
        report = reporter.generate_report(success=True)
        reporter.save_report(report)
    """

    def __init__(
        self,
        task_name: str,
        description: str = "",
        output_dir: str = "artifacts",
    ):
        """
        Initialize the task reporter.

        Args:
            task_name: Name of the task being reported.
            description: Description of the task.
            output_dir: Directory for saving reports and screenshots.
        """
        self.task_name = task_name
        self.description = description
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.start_timestamp = datetime.now().isoformat()

        self._logs: List[LogEntry] = []
        self._step_reports: List[StepReport] = []
        self._screenshots: List[str] = []
        self._step_start_times: Dict[str, float] = {}
        self._step_retry_counts: Dict[str, int] = {}
        self._state_machine_summary: Dict[str, Any] = {}

        self._logger = logging.getLogger(f"TaskReporter.{task_name}")

    def log(
        self,
        step_name: str = "",
        action: str = "",
        result: str = "",
        duration_ms: float = 0.0,
        level: str = "INFO",
        message: str = "",
        **details: Any,
    ) -> LogEntry:
        """
        Add a structured log entry.

        Args:
            step_name: Name of the step.
            action: Action performed.
            result: Result of the action.
            duration_ms: Duration in milliseconds.
            level: Log level.
            message: Message text.
            **details: Additional key-value pairs.

        Returns:
            The created log entry.
        """
        entry = LogEntry(
            task_name=self.task_name,
            step_name=step_name,
            action=action,
            result=result,
            duration_ms=duration_ms,
            level=level,
            message=message,
            details=details,
        )
        self._logs.append(entry)

        log_func = getattr(self._logger, level.lower(), self._logger.info)
        log_func(
            f"[{self.task_name}] {step_name} {action} {result} "
            f"({duration_ms:.1f}ms) - {message}"
        )

        return entry

    def log_step_start(
        self, step_name: str, step_type: str = ""
    ) -> None:
        """
        Log the start of a step execution.

        Args:
            step_name: Name of the step.
            step_type: Type of the step.
        """
        self._step_start_times[step_name] = time.time()
        self.log(
            step_name=step_name,
            action=step_type or "start",
            result="started",
            message=f"Step '{step_name}' started",
            level="DEBUG",
        )

    def log_step_end(
        self,
        step_name: str,
        success: bool,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        step_type: str = "",
    ) -> StepReport:
        """
        Log the completion of a step execution.

        Args:
            step_name: Name of the step.
            success: Whether the step succeeded.
            duration_ms: Duration in milliseconds.
            error: Error message if failed.
            step_type: Type of the step.

        Returns:
            The created step report.
        """
        if duration_ms <= 0 and step_name in self._step_start_times:
            duration_ms = (
                time.time() - self._step_start_times.pop(step_name)
            ) * 1000

        retries = self._step_retry_counts.get(step_name, 0)
        if step_name in self._step_retry_counts:
            self._step_retry_counts[step_name] = 0

        report = StepReport(
            step_name=step_name,
            step_type=step_type,
            success=success,
            duration_ms=duration_ms,
            retries=retries,
            error=error,
        )
        self._step_reports.append(report)

        self.log(
            step_name=step_name,
            action=step_type or "end",
            result="success" if success else "failed",
            duration_ms=duration_ms,
            error=error or "",
            message=(
                f"Step '{step_name}' {'succeeded' if success else 'failed'}"
                f" (retries={retries})"
            ),
            level="INFO" if success else "ERROR",
        )

        return report

    def log_step_retry(self, step_name: str) -> None:
        """
        Log a step retry.

        Args:
            step_name: Name of the step being retried.
        """
        self._step_retry_counts[step_name] = (
            self._step_retry_counts.get(step_name, 0) + 1
        )
        self.log(
            step_name=step_name,
            action="retry",
            result="retrying",
            retry_count=self._step_retry_counts[step_name],
            message=(
                f"Step '{step_name}' retry "
                f"#{self._step_retry_counts[step_name]}"
            ),
            level="WARNING",
        )

    def log_screenshot(self, filepath: str) -> None:
        """
        Record a saved screenshot.

        Args:
            filepath: Path to the screenshot file.
        """
        self._screenshots.append(filepath)
        self.log(
            action="screenshot",
            result="saved",
            filepath=filepath,
            message=f"Screenshot saved: {filepath}",
            level="WARNING",
        )

    def set_state_machine_summary(
        self, summary: Dict[str, Any]
    ) -> None:
        """
        Set the state machine execution summary.

        Args:
            summary: State machine summary dictionary.
        """
        self._state_machine_summary = summary

    def generate_report(self, success: bool) -> TaskReport:
        """
        Generate a complete execution report.

        Args:
            success: Whether the task completed successfully.

        Returns:
            Complete task report.
        """
        end_time = time.time()
        total_duration_ms = (end_time - self.start_time) * 1000

        successful_steps = sum(
            1 for s in self._step_reports if s.success
        )
        failed_steps = sum(
            1 for s in self._step_reports if not s.success
        )
        total_retries = sum(s.retries for s in self._step_reports)

        avg_duration = (
            total_duration_ms / len(self._step_reports)
            if self._step_reports
            else 0
        )

        return TaskReport(
            task_name=self.task_name,
            description=self.description,
            start_time=self.start_timestamp,
            end_time=datetime.now().isoformat(),
            total_duration_ms=total_duration_ms,
            success=success,
            total_steps=len(self._step_reports),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            total_retries=total_retries,
            average_step_duration_ms=avg_duration,
            steps=self._step_reports,
            screenshots=self._screenshots,
            logs=self._logs,
            state_machine_summary=self._state_machine_summary,
        )

    def save_report(
        self, report: TaskReport, filename: Optional[str] = None
    ) -> str:
        """
        Save the execution report to a JSON file.

        Args:
            report: The task report to save.
            filename: Optional filename (auto-generated if not provided).

        Returns:
            Path to the saved report file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{self.task_name}_{timestamp}.json"

        filepath = self.output_dir / filename
        filepath.write_text(
            report.to_json(),
            encoding="utf-8",
        )

        self._logger.info(f"Report saved: {filepath}")
        return str(filepath)

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get all collected logs as a list of dictionaries.

        Returns:
            List of log entries as dictionaries.
        """
        return [asdict(entry) for entry in self._logs]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Statistics dictionary.
        """
        total = len(self._step_reports)
        successes = sum(
            1 for s in self._step_reports if s.success
        )
        failures = total - successes

        return {
            "total_steps": total,
            "success_rate": (
                round(successes / total * 100, 1) if total > 0 else 0
            ),
            "failed_steps": failures,
            "total_retries": sum(
                s.retries for s in self._step_reports
            ),
            "average_step_duration_ms": (
                round(
                    sum(s.duration_ms for s in self._step_reports) / total,
                    2,
                )
                if total > 0
                else 0
            ),
            "total_duration_ms": round(
                time.time() - self.start_time, 2
            )
            * 1000,
            "log_count": len(self._logs),
        }
