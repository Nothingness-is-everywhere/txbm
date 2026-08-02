"""
TriggerScheduler: single point of trigger selection.

Replaces the per-trigger busy-loop with one scheduler that, each cycle, picks
the highest-priority eligible managed trigger. Eligibility is decided by the
shared ``TriggerThrottle.cool_to_run`` (per-trigger min_interval + cooldown,
per-category cooldown, global min-interval). Dedup and budget are applied
inside ``ManagedTriggerTask.run()`` after ``check()``.

The scheduler is consulted by ``TaskExecutor.next_task`` only when the managed
feature flag is on; otherwise the legacy round-robin path is used (rollback).

统一调度器：按优先级 + 节流选取下一个待运行的受管触发器。
"""

import logging
import time
from typing import List, Optional

from ok.trigger.base import ManagedTriggerTask
from ok.trigger.config import TriggerFrameworkConfig
from ok.trigger.metrics import TriggerMetrics
from ok.trigger.throttle import TriggerThrottle

logger = logging.getLogger("TriggerScheduler")


class TriggerScheduler:
    """Owns the shared throttle / metrics / config and selects triggers."""

    def __init__(self, config: Optional[TriggerFrameworkConfig] = None):
        self.config = config or TriggerFrameworkConfig()
        self.throttle = TriggerThrottle(
            global_min_interval=float(self.config.get("global_min_interval", 1.0)),
            budget_window_seconds=float(self.config.get("budget_window_seconds", 60.0)),
            budget_runs_per_window=int(self.config.get("budget_runs_per_window", 60)),
            budget_ocr_per_window=int(self.config.get("budget_ocr_per_window", 30)),
            budget_match_per_window=int(self.config.get("budget_match_per_window", 120)),
        )
        self.metrics = TriggerMetrics(
            log_interval_seconds=float(self.config.get("metrics_log_interval_seconds", 60.0))
        )

    # ------------------------------------------------------------------
    # Registration / refresh
    # ------------------------------------------------------------------
    def refresh(self, tasks: List) -> None:
        """Recompute effective params for all managed triggers.

        在配置变更或任务初始化后调用，确保每个受管触发器读取最新参数。
        """
        for task in tasks:
            if isinstance(task, ManagedTriggerTask):
                task._refresh_params()

    def has_managed(self, tasks: List) -> bool:
        return any(isinstance(t, ManagedTriggerTask) for t in tasks)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(self, tasks: List, now: Optional[float] = None) -> Optional[ManagedTriggerTask]:
        """Return the highest-priority eligible managed trigger, or None.

        选取规则：
          1. 仅受管触发器（ManagedTriggerTask）且 enabled。
          2. 通过 throttle.cool_to_run（min_interval / cooldown / 分类 / 全局）。
          3. 按 priority 降序；同优先级按列表顺序保持稳定。
        """
        if now is None:
            now = time.time()
        candidates: List[ManagedTriggerTask] = []
        for task in tasks:
            if not isinstance(task, ManagedTriggerTask):
                continue
            if not getattr(task, "_enabled", False):
                continue
            if task._params_dirty:
                task._refresh_params()
            allowed, reason = self.throttle.cool_to_run(
                task.name, task.category, task._params, now
            )
            if allowed:
                candidates.append(task)
            else:
                # Record the cooldown block for observability.
                self.metrics.record_throttle(task.name, task.category, "cooldown")
                logger.debug(f"{task.name}: not eligible ({reason})")

        if not candidates:
            return None
        # Stable sort by priority descending (Python sort is stable).
        candidates.sort(key=lambda t: t._params.priority, reverse=True)
        chosen = candidates[0]
        self.metrics.record_selected(chosen.name, chosen.category)
        return chosen

    # ------------------------------------------------------------------
    # Wait budget (replaces legacy next_trigger_delay for managed triggers)
    # ------------------------------------------------------------------
    def next_trigger_delay(self, tasks: List, now: Optional[float] = None,
                           default: float = 1.0) -> float:
        """Seconds until the next managed trigger becomes eligible.

        受管模式下替代 legacy next_trigger_delay，给出精确的等待时长。
        """
        if now is None:
            now = time.time()
        delays: List[float] = []
        for task in tasks:
            if not isinstance(task, ManagedTriggerTask):
                continue
            if not getattr(task, "_enabled", False):
                continue
            if task._params_dirty:
                task._refresh_params()
            last = self.throttle._last_run.get(task.name, 0.0)  # noqa: SLF001
            remaining = task._params.min_interval - (now - last)
            delays.append(max(0.0, remaining))
        if not delays:
            return default
        return max(0.0, min(delays))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def log_summary(self) -> None:
        """Force-emit a metrics summary (e.g. on shutdown)."""
        self.metrics.maybe_log_summary(force=True)
