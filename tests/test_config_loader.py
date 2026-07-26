"""
Tests for the ConfigLoader class.

Covers:
    - Loading configuration from dictionaries
    - Loading configuration from JSON files
    - Loading configuration from YAML files
    - Validation of configuration rules
    - Default configuration generation
    - Error handling for invalid configurations
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ok.automation.config_loader import (
    AutomationConfig,
    StepConfig,
    ConfigLoader,
    StepType,
    RecognitionMode,
)


class TestStepConfig:
    """Tests for StepConfig dataclass."""

    def test_default_values(self):
        step = StepConfig()
        assert step.name == ""
        assert step.type == "find_feature"
        assert step.feature_name is None
        assert step.threshold == 0.8
        assert step.retry_count == 3
        assert step.timeout == 30.0
        assert step.params == {}

    def test_custom_values(self):
        step = StepConfig(
            name="Test Step",
            type="click_feature",
            feature_name="test_button",
            threshold=0.9,
            retry_count=5,
            timeout=60.0,
            x=0.5,
            y=0.3,
        )
        assert step.name == "Test Step"
        assert step.type == "click_feature"
        assert step.feature_name == "test_button"
        assert step.threshold == 0.9
        assert step.retry_count == 5
        assert step.timeout == 60.0
        assert step.x == 0.5
        assert step.y == 0.3

    def test_to_dict(self):
        step = StepConfig(
            name="Export Step",
            type="ocr",
            text="Hello",
            x=0.1,
            y=0.2,
        )
        d = step.to_dict()
        assert d["name"] == "Export Step"
        assert d["type"] == "ocr"
        assert d["text"] == "Hello"
        assert d["x"] == 0.1
        assert d["y"] == 0.2
        assert "params" in d


class TestAutomationConfig:
    """Tests for AutomationConfig dataclass."""

    def test_default_values(self):
        config = AutomationConfig()
        assert config.task_name == "unnamed_task"
        assert config.description == ""
        assert config.global_timeout == 300.0
        assert config.max_retries == 3
        assert config.steps == []
        assert config.error_screenshot is True
        assert config.resume_enabled is True

    def test_custom_values(self):
        steps = [
            StepConfig(name="Step 1", type="find_feature"),
            StepConfig(name="Step 2", type="click_feature"),
        ]
        config = AutomationConfig(
            task_name="My Task",
            description="Test task",
            global_timeout=600.0,
            max_retries=5,
            steps=steps,
        )
        assert config.task_name == "My Task"
        assert config.description == "Test task"
        assert config.global_timeout == 600.0
        assert config.max_retries == 5
        assert len(config.steps) == 2

    def test_to_dict(self):
        config = AutomationConfig(
            task_name="Dict Test",
            steps=[StepConfig(name="S1", type="wait")],
        )
        d = config.to_dict()
        assert d["task_name"] == "Dict Test"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["name"] == "S1"


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_from_dict_minimal(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "minimal",
            "steps": [{"name": "step1", "type": "find_feature"}],
        })
        assert config.task_name == "minimal"
        assert len(config.steps) == 1
        assert config.steps[0].name == "step1"
        assert config.steps[0].type == "find_feature"

    def test_from_dict_full(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "full",
            "description": "Full config test",
            "global_timeout": 600,
            "max_retries": 5,
            "throttle_ms": 200,
            "random_delay": 0.2,
            "steps": [
                {
                    "name": "Step 1",
                    "type": "click_feature",
                    "feature_name": "btn",
                    "threshold": 0.9,
                    "timeout": 15,
                    "retry_count": 4,
                    "post_delay": 1.0,
                },
                {
                    "name": "Step 2",
                    "type": "swipe",
                    "x": 0.5,
                    "y": 0.8,
                    "to_x": 0.5,
                    "to_y": 0.2,
                    "swipe_duration": 1.0,
                },
            ],
        })
        assert config.task_name == "full"
        assert config.global_timeout == 600.0
        assert config.max_retries == 5
        assert config.throttle_ms == 200
        assert len(config.steps) == 2

        step1 = config.steps[0]
        assert step1.name == "Step 1"
        assert step1.type == "click_feature"
        assert step1.feature_name == "btn"
        assert step1.threshold == 0.9
        assert step1.retry_count == 4

    def test_load_json_file(self):
        loader = ConfigLoader()
        config_data = {
            "task_name": "json_test",
            "steps": [
                {"name": "find", "type": "find_feature", "feature_name": "test"}
            ],
        }
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            temp_path = f.name

        try:
            config = loader.load(temp_path)
            assert config.task_name == "json_test"
            assert len(config.steps) == 1
        finally:
            os.unlink(temp_path)

    def test_load_yaml_file(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        loader = ConfigLoader()
        yaml_content = """
task_name: yaml_test
steps:
  - name: find_button
    type: click_feature
    feature_name: button
    threshold: 0.85
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = loader.load(temp_path)
            assert config.task_name == "yaml_test"
            assert len(config.steps) == 1
            assert config.steps[0].feature_name == "button"
            assert config.steps[0].threshold == 0.85
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.yaml")

    def test_load_unsupported_format(self):
        loader = ConfigLoader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("dummy content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                loader.load(temp_path)
            assert "Unsupported file format" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_validate_valid_config(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "valid",
            "steps": [
                {"name": "find", "type": "find_feature", "feature_name": "test"},
            ],
        })
        errors = loader.validate(config)
        assert len(errors) == 0

    def test_validate_missing_task_name(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "steps": [{"name": "s1", "type": "find_feature"}],
        })
        config.task_name = ""
        errors = loader.validate(config)
        assert any("task_name" in e for e in errors)

    def test_validate_invalid_timeout(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "global_timeout": -1,
            "steps": [{"name": "s1", "type": "find_feature"}],
        })
        errors = loader.validate(config)
        assert any("global_timeout" in e for e in errors)

    def test_validate_no_steps(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [],
        })
        errors = loader.validate(config)
        assert any("step" in e.lower() for e in errors)

    def test_validate_invalid_step_type(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "invalid_type"},
            ],
        })
        errors = loader.validate(config)
        assert any("step type" in e.lower() for e in errors)

    def test_validate_missing_feature_name(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "find_feature"},
            ],
        })
        errors = loader.validate(config)
        assert any("feature_name" in e for e in errors)

    def test_validate_invalid_threshold(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "find_feature", "feature_name": "test", "threshold": 1.5},
            ],
        })
        errors = loader.validate(config)
        assert any("threshold" in e for e in errors)

    def test_validate_missing_click_coordinate(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "click_coordinate"},
            ],
        })
        errors = loader.validate(config)
        assert any("x and y" in e.lower() or "coordinate" in e.lower() for e in errors)

    def test_validate_missing_swipe_coordinates(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "swipe", "x": 0.5, "y": 0.5},
            ],
        })
        errors = loader.validate(config)
        assert any("to_x" in e or "to_y" in e or "End coordinates" in e for e in errors)

    def test_validate_missing_key(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "send_key"},
            ],
        })
        errors = loader.validate(config)
        assert any("key" in e for e in errors)

    def test_validate_missing_text(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "input_text"},
            ],
        })
        errors = loader.validate(config)
        assert any("text" in e.lower() for e in errors)

    def test_validate_missing_custom_func(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "s1", "type": "custom"},
            ],
        })
        errors = loader.validate(config)
        assert any("custom_func_name" in e for e in errors)

    def test_get_default_config(self):
        loader = ConfigLoader()
        config = loader.get_default_config()
        assert isinstance(config, AutomationConfig)
        assert config.task_name == "example_task"
        assert len(config.steps) > 0

    def test_default_config_validates(self):
        loader = ConfigLoader()
        config = loader.get_default_config()
        errors = loader.validate(config)
        assert len(errors) == 0

    def test_multiple_validation_errors(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "",
            "global_timeout": -1,
            "throttle_ms": -1,
            "max_retries": -1,
            "steps": [
                {"name": "", "type": "invalid", "threshold": 2.0},
            ],
        })
        errors = loader.validate(config)
        assert len(errors) >= 4

    def test_errors_property(self):
        loader = ConfigLoader()
        assert loader.errors == []

        config = loader.from_dict({"task_name": ""})
        loader.validate(config)
        assert len(loader.errors) > 0


class TestStepTypes:
    """Tests for step type validation."""

    def test_all_valid_types_listed(self):
        valid_types = [
            "find_feature",
            "click_feature",
            "wait_feature",
            "ocr",
            "click_ocr",
            "click_coordinate",
            "swipe",
            "send_key",
            "wait",
            "input_text",
            "custom",
        ]
        loader = ConfigLoader()
        for t in valid_types:
            assert t in loader.VALID_STEP_TYPES

    def test_invalid_type_rejected(self):
        loader = ConfigLoader()
        config = loader.from_dict({
            "task_name": "test",
            "steps": [
                {"name": "bad", "type": "non_existent_type"},
            ],
        })
        errors = loader.validate(config)
        assert len(errors) > 0

    def test_each_type_accepted(self):
        loader = ConfigLoader()
        for step_type in loader.VALID_STEP_TYPES:
            step_data = {"name": f"test_{step_type}", "type": step_type}
            
            if step_type in ("find_feature", "click_feature", "wait_feature"):
                step_data["feature_name"] = "test_feature"
            elif step_type == "click_coordinate":
                step_data["x"] = 0.5
                step_data["y"] = 0.5
            elif step_type == "ocr":
                step_data["x"] = 0.0
                step_data["y"] = 0.0
            elif step_type == "click_ocr":
                step_data["x"] = 0.0
                step_data["y"] = 0.0
                step_data["text"] = "test"
            elif step_type == "swipe":
                step_data["x"] = 0.0
                step_data["y"] = 0.0
                step_data["to_x"] = 1.0
                step_data["to_y"] = 1.0
            elif step_type == "send_key":
                step_data["key"] = "enter"
            elif step_type == "input_text":
                step_data["text"] = "hello"
            elif step_type == "custom":
                step_data["custom_func_name"] = "test_func"

            config = loader.from_dict({
                "task_name": f"test_{step_type}",
                "steps": [step_data],
            })
            errors = loader.validate(config)
            assert len(errors) == 0, f"Type {step_type} failed: {errors}"
