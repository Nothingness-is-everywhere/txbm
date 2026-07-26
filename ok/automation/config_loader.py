"""
Configuration loader and validator for automation task flows.

Supports YAML and JSON configuration files with comprehensive validation
for task steps, coordinates, thresholds, retry parameters, and timeouts.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


StepType = Literal[
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

RecognitionMode = Literal["template", "ocr", "color", "any"]


@dataclass
class StepConfig:
    """
    Configuration for a single automation step.

    Attributes:
        name: Human-readable step name for logging.
        type: The type of action to perform.
        feature_name: Name of the feature/template to find (for find_feature/click_feature).
        x: Relative X coordinate (0.0-1.0) or pixel X.
        y: Relative Y coordinate (0.0-1.0) or pixel Y.
        to_x: End X coordinate for swipe or region end.
        to_y: End Y coordinate for swipe or region end.
        width: Width for region-based operations.
        height: Height for region-based operations.
        threshold: Confidence threshold for recognition (0.0-1.0).
        retry_count: Number of retries for this step.
        timeout: Timeout in seconds for this step.
        interval: Interval between retries in seconds.
        post_delay: Delay after step execution in seconds.
        swipe_duration: Duration of swipe gesture in seconds.
        text: Text to input (for input_text step).
        key: Key to send (for send_key step).
        recognition_mode: Primary recognition method to use.
        verify_feature: Feature to verify after execution.
        custom_func_name: Function name for custom steps.
        params: Additional parameters for the step.
    """

    name: str = ""
    type: StepType = "find_feature"
    feature_name: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    to_x: Optional[float] = None
    to_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    threshold: float = 0.8
    retry_count: int = 3
    timeout: float = 30.0
    interval: float = 0.5
    post_delay: float = 0.5
    swipe_duration: float = 0.5
    text: Optional[str] = None
    key: Optional[str] = None
    recognition_mode: RecognitionMode = "template"
    verify_feature: Optional[str] = None
    custom_func_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert step config to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "feature_name": self.feature_name,
            "x": self.x,
            "y": self.y,
            "to_x": self.to_x,
            "to_y": self.to_y,
            "width": self.width,
            "height": self.height,
            "threshold": self.threshold,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "interval": self.interval,
            "post_delay": self.post_delay,
            "swipe_duration": self.swipe_duration,
            "text": self.text,
            "key": self.key,
            "recognition_mode": self.recognition_mode,
            "verify_feature": self.verify_feature,
            "custom_func_name": self.custom_func_name,
            "params": self.params,
        }


@dataclass
class AutomationConfig:
    """
    Complete configuration for an automation task flow.

    Attributes:
        task_name: Unique identifier for the task.
        description: Human-readable description.
        device_config: Device connection configuration.
        capture_config: Screenshot method configuration.
        interaction_config: Input method configuration.
        template_config: Template matching configuration.
        global_timeout: Maximum total execution time in seconds.
        max_retries: Maximum retries per step.
        steps: Ordered list of step configurations.
        error_screenshot: Whether to save screenshot on failure.
        output_dir: Directory for logs and artifacts.
        resume_enabled: Whether checkpoint recovery is enabled.
        throttle_ms: Minimum delay between actions in milliseconds.
        random_delay: Add random delay variation (0.0-1.0 range multiplier).
    """

    task_name: str = "unnamed_task"
    description: str = ""
    device_config: Dict[str, Any] = field(default_factory=dict)
    capture_config: Dict[str, Any] = field(default_factory=dict)
    interaction_config: Dict[str, Any] = field(default_factory=dict)
    template_config: Dict[str, Any] = field(default_factory=dict)
    global_timeout: float = 300.0
    max_retries: int = 3
    steps: List[StepConfig] = field(default_factory=list)
    error_screenshot: bool = True
    output_dir: str = "artifacts"
    resume_enabled: bool = True
    throttle_ms: int = 100
    random_delay: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "task_name": self.task_name,
            "description": self.description,
            "device_config": self.device_config,
            "capture_config": self.capture_config,
            "interaction_config": self.interaction_config,
            "template_config": self.template_config,
            "global_timeout": self.global_timeout,
            "max_retries": self.max_retries,
            "steps": [s.to_dict() for s in self.steps],
            "error_screenshot": self.error_screenshot,
            "output_dir": self.output_dir,
            "resume_enabled": self.resume_enabled,
            "throttle_ms": self.throttle_ms,
            "random_delay": self.random_delay,
        }


class ConfigLoader:
    """
    Loads and validates automation task configurations from YAML or JSON files.

    Usage:
        loader = ConfigLoader()
        config = loader.load("task_config.yaml")
        errors = loader.validate(config)
        if not errors:
            print("Configuration is valid")
    """

    REQUIRED_STEP_FIELDS = {"name", "type"}
    VALID_STEP_TYPES: List[str] = [
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

    def __init__(self) -> None:
        """Initialize the config loader."""
        self._errors: List[str] = []

    @property
    def errors(self) -> List[str]:
        """Get validation errors from last validation run."""
        return list(self._errors)

    def load(self, file_path: str) -> AutomationConfig:
        """
        Load configuration from a file (YAML or JSON).

        Args:
            file_path: Path to the configuration file.

        Returns:
            Parsed AutomationConfig object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
            json.JSONDecodeError: If JSON parsing fails.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        suffix = path.suffix.lower()
        content = path.read_text(encoding="utf-8")

        if suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required to load YAML files. "
                    "Install it with: pip install pyyaml"
                )
            data = yaml.safe_load(content)
        elif suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Use .yaml, .yml, or .json"
            )

        return self._parse_dict(data)

    def from_dict(self, data: Dict[str, Any]) -> AutomationConfig:
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            Parsed AutomationConfig object.
        """
        return self._parse_dict(data)

    def _parse_dict(self, data: Dict[str, Any]) -> AutomationConfig:
        """
        Parse a dictionary into an AutomationConfig.

        Args:
            data: Raw configuration dictionary.

        Returns:
            Parsed configuration.
        """
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a dictionary")

        steps_raw = data.get("steps", [])
        if not isinstance(steps_raw, list):
            raise ValueError("'steps' must be a list")

        steps = [self._parse_step(step) for step in steps_raw]

        return AutomationConfig(
            task_name=data.get("task_name", "unnamed_task"),
            description=data.get("description", ""),
            device_config=data.get("device_config", {}),
            capture_config=data.get("capture_config", {}),
            interaction_config=data.get("interaction_config", {}),
            template_config=data.get("template_config", {}),
            global_timeout=float(data.get("global_timeout", 300.0)),
            max_retries=int(data.get("max_retries", 3)),
            steps=steps,
            error_screenshot=data.get("error_screenshot", True),
            output_dir=data.get("output_dir", "artifacts"),
            resume_enabled=data.get("resume_enabled", True),
            throttle_ms=int(data.get("throttle_ms", 100)),
            random_delay=float(data.get("random_delay", 0.1)),
        )

    def _parse_step(self, step_data: Dict[str, Any]) -> StepConfig:
        """
        Parse a single step configuration.

        Args:
            step_data: Raw step dictionary.

        Returns:
            Parsed StepConfig.
        """
        return StepConfig(
            name=step_data.get("name", ""),
            type=step_data.get("type", "find_feature"),
            feature_name=step_data.get("feature_name"),
            x=step_data.get("x"),
            y=step_data.get("y"),
            to_x=step_data.get("to_x"),
            to_y=step_data.get("to_y"),
            width=step_data.get("width"),
            height=step_data.get("height"),
            threshold=float(step_data.get("threshold", 0.8)),
            retry_count=int(step_data.get("retry_count", 3)),
            timeout=float(step_data.get("timeout", 30.0)),
            interval=float(step_data.get("interval", 0.5)),
            post_delay=float(step_data.get("post_delay", 0.5)),
            swipe_duration=float(step_data.get("swipe_duration", 0.5)),
            text=step_data.get("text"),
            key=step_data.get("key"),
            recognition_mode=step_data.get("recognition_mode", "template"),
            verify_feature=step_data.get("verify_feature"),
            custom_func_name=step_data.get("custom_func_name"),
            params=step_data.get("params", {}),
        )

    def validate(self, config: AutomationConfig) -> List[str]:
        """
        Validate the configuration and return a list of errors.

        Args:
            config: Configuration to validate.

        Returns:
            List of error messages. Empty list if valid.
        """
        self._errors = []

        if not config.task_name:
            self._errors.append("task_name is required")

        if config.global_timeout <= 0:
            self._errors.append("global_timeout must be positive")

        if config.max_retries < 0:
            self._errors.append("max_retries cannot be negative")

        if config.throttle_ms < 0:
            self._errors.append("throttle_ms cannot be negative")

        if not config.steps:
            self._errors.append("At least one step is required")
        else:
            for i, step in enumerate(config.steps):
                self._validate_step(step, i)

        return self._errors

    def _validate_step(self, step: StepConfig, index: int) -> None:
        """
        Validate a single step configuration.

        Args:
            step: Step configuration to validate.
            index: Step index for error messages.
        """
        prefix = f"Step {index} ({step.name or 'unnamed'}): "

        if step.type not in self.VALID_STEP_TYPES:
            self._errors.append(
                f"{prefix}Invalid step type '{step.type}'. "
                f"Valid types: {self.VALID_STEP_TYPES}"
            )

        if step.threshold < 0.0 or step.threshold > 1.0:
            self._errors.append(
                f"{prefix}threshold must be between 0.0 and 1.0"
            )

        if step.timeout <= 0:
            self._errors.append(f"{prefix}timeout must be positive")

        if step.retry_count < 0:
            self._errors.append(f"{prefix}retry_count cannot be negative")

        if step.type in ("find_feature", "click_feature", "wait_feature"):
            if not step.feature_name:
                self._errors.append(
                    f"{prefix}feature_name is required for {step.type}"
                )

        if step.type == "click_coordinate":
            if step.x is None or step.y is None:
                self._errors.append(
                    f"{prefix}x and y coordinates are required for click_coordinate"
                )

        if step.type == "ocr":
            if step.x is None or step.y is None:
                self._errors.append(
                    f"{prefix}x and y coordinates are required for ocr"
                )

        if step.type == "swipe":
            if step.x is None or step.y is None:
                self._errors.append(
                    f"{prefix}Start coordinates (x, y) are required for swipe"
                )
            if step.to_x is None or step.to_y is None:
                self._errors.append(
                    f"{prefix}End coordinates (to_x, to_y) are required for swipe"
                )

        if step.type == "send_key":
            if not step.key:
                self._errors.append(
                    f"{prefix}key is required for send_key"
                )

        if step.type == "input_text":
            if not step.text:
                self._errors.append(
                    f"{prefix}text is required for input_text"
                )

        if step.type == "custom":
            if not step.custom_func_name:
                self._errors.append(
                    f"{prefix}custom_func_name is required for custom steps"
                )

        for coord_attr in ["x", "y", "to_x", "to_y"]:
            val = getattr(step, coord_attr)
            if val is not None:
                if not (-1 <= val <= 1 or val >= 0):
                    self._errors.append(
                        f"{prefix}{coord_attr} should be relative (0.0-1.0) or absolute pixel value"
                    )

    def get_default_config(self) -> AutomationConfig:
        """
        Get a default configuration template.

        Returns:
            Default AutomationConfig with example values.
        """
        return AutomationConfig(
            task_name="example_task",
            description="Example automation task configuration",
            global_timeout=300.0,
            max_retries=3,
            steps=[
                StepConfig(
                    name="Find start button",
                    type="click_feature",
                    feature_name="start_button",
                    threshold=0.85,
                    timeout=10.0,
                    retry_count=3,
                ),
                StepConfig(
                    name="Wait for game to load",
                    type="wait_feature",
                    feature_name="game_loaded",
                    timeout=30.0,
                    recognition_mode="template",
                ),
                StepConfig(
                    name="Click attack button",
                    type="click_feature",
                    feature_name="attack_button",
                    threshold=0.80,
                ),
                StepConfig(
                    name="Verify result",
                    type="find_feature",
                    feature_name="victory_screen",
                    threshold=0.90,
                    timeout=15.0,
                ),
            ],
        )
