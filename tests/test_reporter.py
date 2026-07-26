"""
Tests for the TaskReporter class.

Covers:
    - Log entry creation
    - Step start/end logging
    - Retry logging
    - Screenshot logging
    - Report generation
    - Report saving
    - Statistics computation
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ok.automation.reporter import (
    LogEntry,
    StepReport,
    TaskReport,
    TaskReporter,
)


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_default_values(self):
        entry = LogEntry()
        assert entry.level == "INFO"
        assert entry.task_name == ""
        assert entry.step_name == ""
        assert entry.action == ""
        assert entry.result == ""
        assert entry.duration_ms == 0.0

    def test_custom_values(self):
        entry = LogEntry(
            task_name="test",
            step_name="step1",
            action="click",
            result="success",
            duration_ms=150.5,
            level="DEBUG",
            message="Click completed",
            details={"key": "value"},
        )
        assert entry.task_name == "test"
        assert entry.step_name == "step1"
        assert entry.action == "click"
        assert entry.result == "success"
        assert entry.duration_ms == 150.5
        assert entry.level == "DEBUG"
        assert entry.details == {"key": "value"}


class TestStepReport:
    """Tests for StepReport dataclass."""

    def test_default_values(self):
        report = StepReport(
            step_name="test",
            step_type="click",
            success=True,
        )
        assert report.duration_ms == 0.0
        assert report.retries == 0
        assert report.error is None
        assert report.timestamp != ""

    def test_failed_report(self):
        report = StepReport(
            step_name="failed_step",
            step_type="ocr",
            success=False,
            duration_ms=2000.0,
            retries=3,
            error="Timeout after 2s",
        )
        assert report.success is False
        assert report.retries == 3
        assert report.error == "Timeout after 2s"


class TestTaskReport:
    """Tests for TaskReport dataclass."""

    def test_default_values(self):
        report = TaskReport(task_name="test")
        assert report.total_steps == 0
        assert report.successful_steps == 0
        assert report.failed_steps == 0
        assert report.total_retries == 0
        assert report.steps == []
        assert report.logs == []

    def test_to_dict(self):
        steps = [
            StepReport(step_name="s1", step_type="click", success=True),
            StepReport(step_name="s2", step_type="wait", success=False),
        ]
        logs = [
            LogEntry(task_name="test", message="log1"),
            LogEntry(task_name="test", message="log2"),
        ]
        report = TaskReport(
            task_name="test",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:05:00",
            total_duration_ms=300000.0,
            success=True,
            total_steps=2,
            successful_steps=1,
            failed_steps=1,
            steps=steps,
            logs=logs,
        )
        d = report.to_dict()
        assert d["task_name"] == "test"
        assert d["success"] is True
        assert d["total_steps"] == 2
        assert d["log_count"] == 2
        assert len(d["steps"]) == 2

    def test_to_json(self):
        report = TaskReport(task_name="json_test")
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["task_name"] == "json_test"


class TestTaskReporter:
    """Tests for TaskReporter class."""

    def test_initialization(self, tmp_path):
        reporter = TaskReporter("test_task", output_dir=str(tmp_path))
        assert reporter.task_name == "test_task"
        assert reporter.description == ""
        assert (tmp_path).exists()

    def test_log_entry(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        entry = reporter.log(
            step_name="step1",
            action="click",
            result="success",
            duration_ms=100.5,
            message="Clicked successfully",
        )
        assert isinstance(entry, LogEntry)
        assert entry.step_name == "step1"
        assert entry.action == "click"
        assert entry.result == "success"

        logs = reporter.get_logs()
        assert len(logs) == 1

    def test_log_step_start(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("my_step", "find_feature")
        logs = reporter.get_logs()
        assert len(logs) == 1
        assert logs[0]["action"] == "find_feature"
        assert logs[0]["result"] == "started"

    def test_log_step_end_success(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("my_step", "click")
        report = reporter.log_step_end("my_step", True, duration_ms=50.0)
        assert report.success is True
        assert report.duration_ms == 50.0
        assert len(reporter._step_reports) == 1

    def test_log_step_end_failure(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("bad_step", "ocr")
        report = reporter.log_step_end(
            "bad_step", False, duration_ms=1000.0, error="OCR failed"
        )
        assert report.success is False
        assert report.error == "OCR failed"

    def test_log_step_retry(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_retry("my_step")
        reporter.log_step_retry("my_step")
        assert reporter._step_retry_counts["my_step"] == 2

        logs = reporter.get_logs()
        retry_logs = [l for l in logs if l["action"] == "retry"]
        assert len(retry_logs) == 2

    def test_retry_count_in_step_end(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("step1", "click")
        reporter.log_step_retry("step1")
        reporter.log_step_retry("step1")
        report = reporter.log_step_end("step1", True)
        assert report.retries == 2

    def test_log_screenshot(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_screenshot("/path/to/screenshot.png")
        assert len(reporter._screenshots) == 1
        assert "/path/to/screenshot.png" in reporter._screenshots

    def test_generate_report_success(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("s1", "click")
        reporter.log_step_end("s1", True, duration_ms=100.0)
        reporter.log_step_start("s2", "wait")
        reporter.log_step_end("s2", True, duration_ms=200.0)

        report = reporter.generate_report(success=True)
        assert isinstance(report, TaskReport)
        assert report.task_name == "test"
        assert report.success is True
        assert report.total_steps == 2
        assert report.successful_steps == 2
        assert report.failed_steps == 0
        assert len(report.steps) == 2
        assert report.total_duration_ms > 0

    def test_generate_report_failure(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("s1", "click")
        reporter.log_step_end("s1", True, duration_ms=100.0)
        reporter.log_step_start("s2", "ocr")
        reporter.log_step_end("s2", False, duration_ms=5000.0, error="Timeout")

        report = reporter.generate_report(success=False)
        assert report.success is False
        assert report.successful_steps == 1
        assert report.failed_steps == 1
        assert report.total_retries == 0

    def test_set_state_machine_summary(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        summary = {"state": "done", "retries": 0}
        reporter.set_state_machine_summary(summary)
        report = reporter.generate_report(success=True)
        assert report.state_machine_summary == summary

    def test_save_report(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("s1", "click")
        reporter.log_step_end("s1", True)
        report = reporter.generate_report(success=True)
        saved_path = reporter.save_report(report)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".json")

        with open(saved_path) as f:
            saved_data = json.load(f)
        assert saved_data["task_name"] == "test"
        assert saved_data["success"] is True

    def test_save_report_custom_filename(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        report = reporter.generate_report(success=True)
        saved_path = reporter.save_report(report, filename="custom_report.json")
        assert os.path.basename(saved_path) == "custom_report.json"

    def test_get_statistics(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log_step_start("s1", "click")
        reporter.log_step_end("s1", True, duration_ms=100.0)
        reporter.log_step_start("s2", "click")
        reporter.log_step_retry("s2")
        reporter.log_step_end("s2", True, duration_ms=150.0)
        reporter.log_step_start("s3", "ocr")
        reporter.log_step_end("s3", False, duration_ms=3000.0)

        stats = reporter.get_statistics()
        assert stats["total_steps"] == 3
        assert stats["success_rate"] == pytest.approx(66.7, rel=0.01)
        assert stats["failed_steps"] == 1
        assert stats["total_retries"] == 1
        assert stats["log_count"] > 0

    def test_empty_statistics(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        stats = reporter.get_statistics()
        assert stats["total_steps"] == 0
        assert stats["success_rate"] == 0

    def test_log_levels(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log(message="debug msg", level="DEBUG")
        reporter.log(message="info msg", level="INFO")
        reporter.log(message="warning msg", level="WARNING")
        reporter.log(message="error msg", level="ERROR")

        logs = reporter.get_logs()
        levels = {l["level"] for l in logs}
        assert "DEBUG" in levels
        assert "INFO" in levels
        assert "WARNING" in levels
        assert "ERROR" in levels

    def test_get_logs_returns_dicts(self, tmp_path):
        reporter = TaskReporter("test", output_dir=str(tmp_path))
        reporter.log(message="test")
        logs = reporter.get_logs()
        assert isinstance(logs, list)
        assert isinstance(logs[0], dict)
