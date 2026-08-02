"""
Trigger observability metrics.

Counts per-trigger / per-category / global events so the scheduler's behavior
can be inspected and before/after performance compared. A periodic summary is
logged to the standard logger.

可观测性：每触发器/分类/全局计数 + 周期性摘要日志。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ok.trigger.categories import TriggerCategory

logger = logging.getLogger("TriggerMetrics")


@dataclass
class TriggerCounters:
    """Per-trigger counters."""

    selected: int = 0          # chosen by scheduler.select
    checked: int = 0           # check() ran
    handled: int = 0           # handle() ran
    success: int = 0           # handle() succeeded
    failed: int = 0            # handle() ran but failed
    timeout: int = 0           # check/handle exceeded timeout
    cooldown_hits: int = 0     # blocked by cool_to_run
    dedup_hits: int = 0        # blocked by dedup
    budget_hits: int = 0       # blocked by budget
    retries: int = 0           # internal retries consumed
    ocr_calls: int = 0
    match_calls: int = 0
    screenshot_calls: int = 0
    total_handle_seconds: float = 0.0


class TriggerMetrics:
    """Thread-safe metrics store shared by all managed triggers.

    线程安全的指标存储，由调度器与 ManagedTriggerTask 共同写入。
    """

    def __init__(self, log_interval_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._counters: Dict[str, TriggerCounters] = {}
        self._category_counters: Dict[str, TriggerCounters] = {}
        self._last_summary_time: float = time.time()
        self._log_interval = log_interval_seconds

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _get(self, name: str) -> TriggerCounters:
        if name not in self._counters:
            self._counters[name] = TriggerCounters()
        return self._counters[name]

    def _get_cat(self, category: TriggerCategory) -> TriggerCounters:
        key = category.value if isinstance(category, TriggerCategory) else str(category)
        if key not in self._category_counters:
            self._category_counters[key] = TriggerCounters()
        return self._category_counters[key]

    def record_selected(self, name: str, category: TriggerCategory) -> None:
        with self._lock:
            self._get(name).selected += 1
            self._get_cat(category).selected += 1

    def record_checked(self, name: str, category: TriggerCategory) -> None:
        with self._lock:
            self._get(name).checked += 1
            self._get_cat(category).checked += 1

    def record_throttle(
        self, name: str, category: TriggerCategory, kind: str
    ) -> None:
        """Record a throttle block. ``kind`` in {cooldown, dedup, budget}."""
        with self._lock:
            c = self._get(name)
            cc = self._get_cat(category)
            if kind == "dedup":
                c.dedup_hits += 1
                cc.dedup_hits += 1
            elif kind.startswith("budget"):
                c.budget_hits += 1
                cc.budget_hits += 1
            else:
                c.cooldown_hits += 1
                cc.cooldown_hits += 1

    def record_run(
        self,
        name: str,
        category: TriggerCategory,
        handled: bool,
        success: bool,
        timeout: bool = False,
        elapsed: float = 0.0,
        ocr_calls: int = 0,
        match_calls: int = 0,
        screenshot_calls: int = 0,
        retries: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Record the outcome of a full check+handle cycle."""
        with self._lock:
            c = self._get(name)
            cc = self._get_cat(category)
            if handled:
                c.handled += 1
                cc.handled += 1
                if success:
                    c.success += 1
                    cc.success += 1
                else:
                    c.failed += 1
                    cc.failed += 1
            if timeout:
                c.timeout += 1
                cc.timeout += 1
            c.retries += retries
            cc.retries += retries
            c.ocr_calls += ocr_calls
            cc.ocr_calls += ocr_calls
            c.match_calls += match_calls
            cc.match_calls += match_calls
            c.screenshot_calls += screenshot_calls
            cc.screenshot_calls += screenshot_calls
            c.total_handle_seconds += max(0.0, elapsed)
            cc.total_handle_seconds += max(0.0, elapsed)
        self.maybe_log_summary()

    # ------------------------------------------------------------------
    # Periodic summary
    # ------------------------------------------------------------------
    def maybe_log_summary(self, force: bool = False) -> None:
        """Emit a structured summary if the log interval has elapsed."""
        if self._log_interval <= 0 and not force:
            return
        now = time.time()
        if not force and now - self._last_summary_time < self._log_interval:
            return
        self._last_summary_time = now
        with self._lock:
            lines: List[str] = ["trigger metrics summary:"]
            for name, c in self._counters.items():
                lines.append(
                    f"  [{name}] sel={c.selected} chk={c.checked} hnd={c.handled} "
                    f"ok={c.success} fail={c.failed} tmo={c.timeout} "
                    f"cd={c.cooldown_hits} dup={c.dedup_hits} bud={c.budget_hits} "
                    f"ocr={c.ocr_calls} match={c.match_calls} shot={c.screenshot_calls} "
                    f"retries={c.retries} sec={c.total_handle_seconds:.1f}"
                )
            logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Snapshot (for tests / external reporting)
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return {
                name: {
                    "selected": c.selected,
                    "checked": c.checked,
                    "handled": c.handled,
                    "success": c.success,
                    "failed": c.failed,
                    "timeout": c.timeout,
                    "cooldown_hits": c.cooldown_hits,
                    "dedup_hits": c.dedup_hits,
                    "budget_hits": c.budget_hits,
                    "ocr_calls": c.ocr_calls,
                    "match_calls": c.match_calls,
                    "screenshot_calls": c.screenshot_calls,
                    "retries": c.retries,
                    "total_handle_seconds": c.total_handle_seconds,
                }
                for name, c in self._counters.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._category_counters.clear()
            self._last_summary_time = time.time()
