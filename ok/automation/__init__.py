"""
ok-script automation module.

Provides game task automation with state machine, configuration-driven workflows,
retry logic, checkpoint recovery, and structured reporting.

Stamina public variables (updated by update_global_stamina()):
    EXPEDITION_STAMINA       - 当前远征体力值 (int)
    TRAINING_STAMINA         - 当前训练体力值 (int)
    EXPEDITION_STAMINA_MAX   - 远征体力上限 (int, 默认150)
    TRAINING_STAMINA_MAX     - 训练体力上限 (int, 默认50)
    EXPEDITION_STAMINA_VALUE - 远征体力完整值对象 (StaminaValue)
    TRAINING_STAMINA_VALUE   - 训练体力完整值对象 (StaminaValue)
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
    DigitClassifier,
    get_stamina,
    init_stamina,
    update_global_stamina,
    EXPEDITION_STAMINA,
    TRAINING_STAMINA,
    EXPEDITION_STAMINA_MAX,
    TRAINING_STAMINA_MAX,
    EXPEDITION_STAMINA_VALUE,
    TRAINING_STAMINA_VALUE,
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
    "DigitClassifier",
    "get_stamina",
    "init_stamina",
    "update_global_stamina",
    "EXPEDITION_STAMINA",
    "TRAINING_STAMINA",
    "EXPEDITION_STAMINA_MAX",
    "TRAINING_STAMINA_MAX",
    "EXPEDITION_STAMINA_VALUE",
    "TRAINING_STAMINA_VALUE",
]
