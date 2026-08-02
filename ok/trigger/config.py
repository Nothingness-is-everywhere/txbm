"""
Centralized trigger framework configuration.

Loads ``configs/trigger_config.json`` and merges:

  1. Built-in defaults (``DEFAULTS``).
  2. Category-level defaults (``category_defaults[category]``).
  3. Per-trigger overrides (``trigger_overrides[trigger_name]``).

The rollback feature flag ``enable_managed_trigger`` can be flipped at runtime
via the environment variable ``OK_TRIGGER_MANAGED`` (``0``/``false`` disables
the central scheduler and restores the legacy round-robin selection).

配置中心化：分类级默认 + 触发器级覆盖 + 环境变量回滚开关。
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ok.trigger.categories import TriggerCategory

logger = logging.getLogger("TriggerConfig")

# Repo root: ok/trigger/config.py -> repo root is two parents up .../ok/trigger
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Canonical default shipped in the package (tracked in git).
_PACKAGE_DEFAULT_PATH = Path(__file__).resolve().parent / "default_trigger_config.json"
# Optional per-deployment override (gitignored, machine-local).
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "trigger_config.json"


# Built-in framework defaults. Values are intentionally conservative so that
# the system never busy-loops even with an empty config file.
DEFAULTS: Dict[str, Any] = {
    # Master switch for the central scheduler. When False, TaskExecutor falls
    # back to the legacy round-robin trigger selection.
    "enable_managed_trigger": True,
    # Global minimum spacing (seconds) between any two trigger runs. Prevents
    # back-to-back expensive detections from different triggers.
    "global_min_interval": 1.0,
    # Rolling budget window (seconds) used for rate limiting.
    "budget_window_seconds": 60.0,
    # Maximum trigger runs within the budget window (0 = unlimited).
    "budget_runs_per_window": 60,
    # Maximum OCR calls within the budget window (0 = unlimited).
    "budget_ocr_per_window": 30,
    # Maximum template-match calls within the budget window (0 = unlimited).
    "budget_match_per_window": 120,
    # Hard timeout (seconds) for a single handle() call. 0 = no timeout guard.
    "handle_timeout_seconds": 8.0,
    # Hard timeout (seconds) for a single check() call. 0 = no timeout guard.
    "check_timeout_seconds": 3.0,
    # Interval (seconds) between metrics summary log lines. 0 = disabled.
    "metrics_log_interval_seconds": 60.0,
    # Default per-trigger parameters applied when a category/trigger does not
    # override them.
    "trigger": {
        "min_interval": 3.0,
        "cooldown_seconds": 5.0,
        "dedup_window_seconds": 8.0,
        "max_retry": 3,
        "timeout_seconds": 8.0,
        "priority": 50,
        "trigger_mode": "polling",
    },
}

# Category-level defaults. Network/recovery triggers are higher priority and
# shorter interval; UI popups and task-flow triggers are lower frequency.
CATEGORY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    TriggerCategory.NETWORK.value: {
        "min_interval": 3.0,
        "cooldown_seconds": 5.0,
        "dedup_window_seconds": 8.0,
        "priority": 80,
        "trigger_mode": "polling",
    },
    TriggerCategory.RECOVERY.value: {
        "min_interval": 5.0,
        "cooldown_seconds": 10.0,
        "dedup_window_seconds": 10.0,
        "priority": 70,
        "trigger_mode": "polling",
    },
    TriggerCategory.UI_POPUP.value: {
        "min_interval": 8.0,
        "cooldown_seconds": 15.0,
        "dedup_window_seconds": 15.0,
        "priority": 50,
        "trigger_mode": "polling",
    },
    TriggerCategory.BATTLE_SCENE.value: {
        "min_interval": 5.0,
        "cooldown_seconds": 8.0,
        "dedup_window_seconds": 8.0,
        "priority": 60,
        "trigger_mode": "hybrid",
    },
    TriggerCategory.TASK_FLOW.value: {
        "min_interval": 10.0,
        "cooldown_seconds": 20.0,
        "dedup_window_seconds": 20.0,
        "priority": 40,
        "trigger_mode": "polling",
    },
}


@dataclass
class TriggerParams:
    """Effective parameters for a single trigger after config merge."""

    min_interval: float = 3.0
    cooldown_seconds: float = 5.0
    dedup_window_seconds: float = 8.0
    max_retry: int = 3
    timeout_seconds: float = 8.0
    priority: int = 50
    trigger_mode: str = "polling"
    extra: Dict[str, Any] = field(default_factory=dict)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class TriggerFrameworkConfig:
    """Loads and merges trigger framework configuration.

    配置加载与合并：DEFAULTS < category_defaults < trigger_overrides。
    """

    def __init__(self, path: Optional[str] = None, raw: Optional[Dict[str, Any]] = None):
        self._path = Path(path) if path else _DEFAULT_CONFIG_PATH
        if raw is not None:
            self._raw: Dict[str, Any] = dict(raw)
        else:
            # Base = package default (tracked); overlay = deployment override.
            base = self._load_json(_PACKAGE_DEFAULT_PATH)
            overlay = self._load_json(self._path)
            self._raw = self._merge(base, overlay)
        self._category_overrides: Dict[str, Dict[str, Any]] = (
            self._raw.get("category_defaults", {}) or {}
        )
        self._trigger_overrides: Dict[str, Dict[str, Any]] = (
            self._raw.get("trigger_overrides", {}) or {}
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load a JSON object from ``path``, returning {} on any failure."""
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning(f"trigger config root is not an object, ignoring: {path}")
                return {}
            logger.info(f"loaded trigger config: {path}")
            return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"failed to load trigger config ({path}): {e}")
            return {}

    @staticmethod
    def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Shallow merge with deep-merge for nested dict values.

        overlay 的同名键覆盖 base；category_defaults / trigger_overrides 按
        子键合并，避免部署覆盖意外清空整个分类默认表。
        """
        merged = dict(base)
        for key, val in overlay.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **val}
            else:
                merged[key] = val
        return merged

    # ------------------------------------------------------------------
    # Global settings
    # ------------------------------------------------------------------
    @property
    def enable_managed_trigger(self) -> bool:
        """Master feature flag. Env var ``OK_TRIGGER_MANAGED`` takes priority."""
        env = os.environ.get("OK_TRIGGER_MANAGED")
        if env is not None and env.strip() != "":
            return env.strip().lower() not in ("0", "false", "no", "off")
        return bool(self._raw.get("enable_managed_trigger", DEFAULTS["enable_managed_trigger"]))

    def get(self, key: str, default: Any = None) -> Any:
        """Read a top-level global setting with built-in default fallback."""
        if key in self._raw:
            return self._raw[key]
        return DEFAULTS.get(key, default)

    # ------------------------------------------------------------------
    # Per-trigger effective params
    # ------------------------------------------------------------------
    def effective_params(
        self,
        name: str,
        category: TriggerCategory,
        declared: Optional[Dict[str, Any]] = None,
    ) -> TriggerParams:
        """Compute the effective params for a trigger.

        Merge order (later wins):
          framework default -> category default -> category override (config)
          -> trigger override (config) -> declared (class attributes)
        """
        merged: Dict[str, Any] = dict(DEFAULTS["trigger"])
        cat_key = category.value if isinstance(category, TriggerCategory) else str(category)
        if cat_key in CATEGORY_DEFAULTS:
            merged.update(CATEGORY_DEFAULTS[cat_key])
        if cat_key in self._category_overrides:
            merged.update(self._category_overrides[cat_key])
        if name in self._trigger_overrides:
            merged.update(self._trigger_overrides[name])
        if declared:
            for k, v in declared.items():
                if v is not None:
                    merged[k] = v

        extra = {k: v for k, v in merged.items() if k not in {
            "min_interval", "cooldown_seconds", "dedup_window_seconds",
            "max_retry", "timeout_seconds", "priority", "trigger_mode",
        }}
        return TriggerParams(
            min_interval=_coerce_float(merged.get("min_interval"), DEFAULTS["trigger"]["min_interval"]),
            cooldown_seconds=_coerce_float(merged.get("cooldown_seconds"), DEFAULTS["trigger"]["cooldown_seconds"]),
            dedup_window_seconds=_coerce_float(merged.get("dedup_window_seconds"), DEFAULTS["trigger"]["dedup_window_seconds"]),
            max_retry=_coerce_int(merged.get("max_retry"), DEFAULTS["trigger"]["max_retry"]),
            timeout_seconds=_coerce_float(merged.get("timeout_seconds"), DEFAULTS["trigger"]["timeout_seconds"]),
            priority=_coerce_int(merged.get("priority"), DEFAULTS["trigger"]["priority"]),
            trigger_mode=str(merged.get("trigger_mode", "polling")),
            extra=extra,
        )


# Module-level shared instance (lazy). The executor constructs its own copy;
# this is a convenience for tests / headless usage.
_shared: Optional[TriggerFrameworkConfig] = None


def get_shared_config() -> TriggerFrameworkConfig:
    global _shared
    if _shared is None:
        _shared = TriggerFrameworkConfig()
    return _shared
