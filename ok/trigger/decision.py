"""
Trigger decision / result data structures.

The ``check()`` phase inspects the current frame/context and returns a
``TriggerDecision`` describing whether the trigger should act, plus a
fingerprint (for dedup) and a cost estimate (for budgeting). The ``handle()``
phase performs the action and returns a ``TriggerResult`` with outcome and
resource counters.

check/handle 两阶段契约：check 产出决策，handle 执行动作并回报资源消耗。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TriggerDecision:
    """Outcome of the (cheap) detection phase.

    Attributes:
        should_handle: Whether ``handle()`` should run.
        reason: Human-readable reason for logging/metrics.
        fingerprint: Short signature of the detected event. Two consecutive
            decisions with the same fingerprint inside ``dedup_window_seconds``
            are considered duplicates and skipped.
        cost_estimate: Relative cost hint (0.0 ~ 1.0+) used by the budget
            controller to weigh expensive detections. Leave 0.0 if unknown.
    """

    should_handle: bool = False
    reason: str = ""
    fingerprint: str = ""
    cost_estimate: float = 0.0


@dataclass
class TriggerResult:
    """Outcome of the (expensive) handling phase.

    Attributes:
        handled: True if the trigger performed an action this run.
        success: True if the action succeeded.
        error: Optional error message when the run failed softly.
        ocr_calls: Number of OCR invocations consumed (for budget/metrics).
        match_calls: Number of template-match invocations consumed.
        screenshot_calls: Number of screenshot/capture invocations consumed.
        retries: Number of internal retries performed.
        elapsed: Wall-clock seconds spent in ``handle()``.
    """

    handled: bool = False
    success: bool = False
    error: Optional[str] = None
    ocr_calls: int = 0
    match_calls: int = 0
    screenshot_calls: int = 0
    retries: int = 0
    elapsed: float = 0.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def skip(cls, reason: str = "") -> "TriggerResult":
        """Build a result indicating nothing was done (e.g. no event detected)."""
        return cls(handled=False, success=False, error=reason or None)

    @classmethod
    def ok(cls, **counters) -> "TriggerResult":
        """Build a successful result with optional resource counters."""
        return cls(handled=True, success=True, **counters)

    @classmethod
    def fail(cls, error: str, **counters) -> "TriggerResult":
        """Build a failed result with optional resource counters."""
        return cls(handled=True, success=False, error=error, **counters)
