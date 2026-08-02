"""
Trigger category classification.

Every trigger belongs to exactly one category. Categories drive per-category
cooldown and let the scheduler avoid running several expensive triggers of the
same kind back-to-back.

触发器分类：用于分类级冷却与并发控制。
"""

from enum import Enum


class TriggerCategory(str, Enum):
    """All trigger categories used by the unified scheduler.

    使用 ``str`` 作为基类便于 JSON 序列化与配置按名称引用。
    """

    # 网络类：断线、超时、重连提示
    NETWORK = "network"

    # 战斗/场景类：战斗结算、场景识别
    BATTLE_SCENE = "battle_scene"

    # UI 弹窗类：公告、签到、奖励、广告等通用弹窗
    UI_POPUP = "ui_popup"

    # 任务流程类：日常/周常任务进度推进
    TASK_FLOW = "task_flow"

    # 异常恢复类：卡死、无响应、回到主界面等恢复动作
    RECOVERY = "recovery"

    @classmethod
    def from_value(cls, value) -> "TriggerCategory":
        """Parse a category from a string/enum, raising ``ValueError`` if unknown."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for member in cls:
                if member.value == value or member.name == value:
                    return member
        raise ValueError(f"Unknown trigger category: {value!r}")
