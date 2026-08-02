"""
Throttle / cooldown / budget / dedup primitives.

The throttle enforces four layers of rate limiting:

  1. Per-trigger cooldown: a trigger may not run again within
     ``params.cooldown_seconds`` of its last *handled* run.
  2. Per-category cooldown: triggers of the same category share a cooldown to
     avoid running several similar expensive checks back-to-back.
  3. Global min-interval + rolling-window budget: limits total runs / OCR /
     match calls across all triggers within ``budget_window_seconds``.
  4. Dedup: two consecutive decisions with the same fingerprint inside
     ``params.dedup_window_seconds`` are skipped.

All checks are O(1) amortized and use monotonic time semantics (callers pass
``now``). OCR/match budget events are stored as ``(timestamp, count)`` tuples so
the rolling window can be purged correctly.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from ok.trigger.categories import TriggerCategory
from ok.trigger.config import TriggerParams

logger = logging.getLogger("TriggerThrottle")


@dataclass
class AdmitDecision:
    """Result of a full admit check (post-``check()``)."""

    allowed: bool
    reason: str = ""
    # One of: ok, dedup, budget_runs, budget_ocr, budget_match
    kind: str = "ok"


class TriggerThrottle:
    """Stateful throttle shared by all managed triggers via the scheduler.

    共享节流器：单触发器冷却 / 分类冷却 / 全局预算 / 去重。
    """

    def __init__(
        self,
        global_min_interval: float = 1.0,
        budget_window_seconds: float = 60.0,
        budget_runs_per_window: int = 60,
        budget_ocr_per_window: int = 30,
        budget_match_per_window: int = 120,
    ):
        self.global_min_interval = global_min_interval
        self.budget_window_seconds = budget_window_seconds
        self.budget_runs_per_window = budget_runs_per_window
        self.budget_ocr_per_window = budget_ocr_per_window
        self.budget_match_per_window = budget_match_per_window

        # Per-trigger last handled time (for cooldown + min_interval).
        self._last_run: Dict[str, float] = {}
        # Per-trigger last fingerprint + time (for dedup).
        self._last_fp: Dict[str, Tuple[str, float]] = {}
        # Per-category last handled time.
        self._last_category_run: Dict[str, float] = {}
        # Global last run timestamp (for global_min_interval).
        self._last_global_run: float = 0.0

        # Rolling-window budget counters.
        self._run_events: Deque[float] = deque()                  # timestamps
        self._ocr_events: Deque[Tuple[float, int]] = deque()      # (ts, count)
        self._match_events: Deque[Tuple[float, int]] = deque()    # (ts, count)

    # ------------------------------------------------------------------
    # Cheap pre-check (used by scheduler.select before check())
    # ------------------------------------------------------------------
    def cool_to_run(
        self,
        name: str,
        category: TriggerCategory,
        params: TriggerParams,
        now: float,
    ) -> Tuple[bool, str]:
        """Return (allowed, reason) for the cheap selection-time filter.

        考量：单触发器 min_interval / 单触发器 cooldown / 分类 cooldown / 全局最小间隔。
        不含 dedup 与预算（需要 TriggerDecision / 资源计数）。
        """
        last = self._last_run.get(name, 0.0)
        elapsed = now - last
        if elapsed < params.min_interval:
            return False, f"min_interval({params.min_interval:.1f}s): {elapsed:.1f}s"
        if elapsed < params.cooldown_seconds:
            return False, f"cooldown({params.cooldown_seconds:.1f}s): {elapsed:.1f}s"
        cat_key = category.value if isinstance(category, TriggerCategory) else str(category)
        cat_last = self._last_category_run.get(cat_key, 0.0)
        # Category cooldown reuses the trigger's cooldown_seconds as shared
        # spacing within the category.
        if now - cat_last < params.cooldown_seconds:
            return False, f"category({cat_key}) cooldown: {now - cat_last:.1f}s"
        if now - self._last_global_run < self.global_min_interval:
            return False, f"global_min_interval: {now - self._last_global_run:.1f}s"
        return True, "ok"

    # ------------------------------------------------------------------
    # Full admit (used after check(), needs decision fingerprint + cost)
    # ------------------------------------------------------------------
    def admit(
        self,
        name: str,
        fingerprint: str,
        now: float,
        dedup_window_seconds: float,
    ) -> AdmitDecision:
        """Full admit check after ``check()`` produced a fingerprint.

        考量：dedup + 全局运行次数预算。cooldown 已在 cool_to_run 中校验。
        """
        # Dedup: same fingerprint within the dedup window -> skip.
        if fingerprint:
            last_fp, last_t = self._last_fp.get(name, ("", 0.0))
            if fingerprint == last_fp and now - last_t < dedup_window_seconds:
                return AdmitDecision(False, f"dedup({fingerprint}): {now - last_t:.1f}s", "dedup")

        # Budget: rolling window for run count.
        self._purge(now)
        if self.budget_runs_per_window and len(self._run_events) >= self.budget_runs_per_window:
            return AdmitDecision(False, "budget_runs exceeded", "budget_runs")
        return AdmitDecision(True, "ok", "ok")

    def budget_available(self, ocr_calls: int, match_calls: int, now: float) -> AdmitDecision:
        """Check whether a predicted cost fits the OCR/match budget.

        使用即将消耗的 OCR/匹配次数预测预算是否充足。
        """
        self._purge(now)
        if (self.budget_ocr_per_window
                and self._sum(self._ocr_events) + ocr_calls > self.budget_ocr_per_window):
            return AdmitDecision(False, "budget_ocr exceeded", "budget_ocr")
        if (self.budget_match_per_window
                and self._sum(self._match_events) + match_calls > self.budget_match_per_window):
            return AdmitDecision(False, "budget_match exceeded", "budget_match")
        return AdmitDecision(True, "ok", "ok")

    # ------------------------------------------------------------------
    # Record outcomes
    # ------------------------------------------------------------------
    def record_run(
        self,
        name: str,
        category: TriggerCategory,
        fingerprint: str,
        now: float,
    ) -> None:
        """Record that a trigger ran (handled). Updates cooldowns + dedup + budget."""
        self._last_run[name] = now
        cat_key = category.value if isinstance(category, TriggerCategory) else str(category)
        self._last_category_run[cat_key] = now
        self._last_global_run = now
        if fingerprint:
            self._last_fp[name] = (fingerprint, now)
        self._run_events.append(now)

    def record_cost(self, ocr_calls: int, match_calls: int, now: float) -> None:
        """Record resource consumption from a ``TriggerResult``."""
        if ocr_calls:
            self._ocr_events.append((now, ocr_calls))
        if match_calls:
            self._match_events.append((now, match_calls))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, int]:
        """Return current rolling-window counters (for metrics)."""
        now_like = self._last_global_run
        self._purge(now_like)
        return {
            "runs_in_window": len(self._run_events),
            "ocr_in_window": self._sum(self._ocr_events),
            "match_in_window": self._sum(self._match_events),
        }

    def reset(self) -> None:
        """Clear all throttle state (used by tests / rollback)."""
        self._last_run.clear()
        self._last_fp.clear()
        self._last_category_run.clear()
        self._last_global_run = 0.0
        self._run_events.clear()
        self._ocr_events.clear()
        self._match_events.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _purge(self, now: float) -> None:
        """Drop budget events older than the rolling window."""
        cutoff = now - self.budget_window_seconds
        while self._run_events and self._run_events[0] < cutoff:
            self._run_events.popleft()
        while self._ocr_events and self._ocr_events[0][0] < cutoff:
            self._ocr_events.popleft()
        while self._match_events and self._match_events[0][0] < cutoff:
            self._match_events.popleft()

    @staticmethod
    def _sum(events: Deque[Tuple[float, int]]) -> int:
        return sum(count for _, count in events)
