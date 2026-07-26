"""
ok-script automation module.

Provides game task automation with state machine, configuration-driven workflows,
retry logic, checkpoint recovery, and structured reporting.
"""

from ok.automation.state_machine import (
    TaskState,
    TaskStatus,
    GameStateMachine,
    StateTransition,
)
from ok.automation.config_loader import (
    AutomationConfig,
    StepConfig,
    ConfigLoader,
)
from ok.automation.game_task import GameTask
from ok.automation.reporter import TaskReporter
from ok.automation.stamina_reader import (
    StaminaReader,
    StaminaState,
    StaminaValue,
    get_stamina,
)

__all__ = [
    "TaskState",
    "TaskStatus",
    "GameStateMachine",
    "StateTransition",
    "AutomationConfig",
    "StepConfig",
    "ConfigLoader",
    "GameTask",
    "TaskReporter",
    "StaminaReader",
    "StaminaState",
    "StaminaValue",
    "get_stamina",
]
