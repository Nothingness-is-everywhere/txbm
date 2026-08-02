"""
ManagedTriggerTask base class.

Replaces ad-hoc busy-loop triggers with a two-phase contract:

  - ``check(context) -> TriggerDecision``  (cheap detection)
  - ``handle(context) -> TriggerResult``   (expensive action)

The base ``run()`` wires the two phases through the shared
``TriggerThrottle`` / ``TriggerMetrics`` / ``TriggerFrameworkConfig`` owned by
the executor's ``TriggerScheduler``. It guarantees:

  - ``run()`` never raises (a soft failure is recorded and ``False`` returned),
    so a transient detection error can no longer permanently disable the
    trigger (the legacy executor disables any task whose ``run()`` raises).
  - per-trigger / category / global cooldown, dedup, and budget are enforced.
  - check/handle are bounded by a watchdog timeout.
  - structured metrics are recorded for every cycle.

When the managed-scheduler feature flag is off (rollback), ``run()`` still
applies the per-trigger guardrails (cooldown/dedup/timeout/metrics); only the
central priority/budget/category selection reverts to round-robin.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ok.task.task import TriggerTask
from ok.trigger.categories import TriggerCategory
from ok.trigger.config import TriggerFrameworkConfig, TriggerParams, get_shared_config
from ok.trigger.decision import TriggerDecision, TriggerResult
from ok.trigger.metrics import TriggerMetrics
from ok.trigger.throttle import TriggerThrottle

logger = logging.getLogger("ManagedTrigger")


@dataclass
class TriggerContext:
    """Per-run context handed to ``check()`` / ``handle()``.

    ``handle()`` mutates the cost accumulators so the base class can report
    them in metrics / budget without inspecting the trigger's internals.
    """

    frame: Optional[np.ndarray] = None
    executor: Any = None
    now: float = field(default_factory=time.time)
    deadline: float = 0.0  # absolute time after which handle should bail
    # Mutable cost accumulators updated by handle().
    ocr_calls: int = 0
    match_calls: int = 0
    screenshot_calls: int = 0
    retries: int = 0

    def timed_out(self) -> bool:
        """Cooperative timeout flag handle() loops may check."""
        return self.deadline > 0 and time.time() > self.deadline


class ManagedTriggerTask(TriggerTask):
    """Base class for all triggers governed by the unified scheduler.

    子类需实现 ``check`` 与 ``handle``；其余节流/超时/指标由基类统一处理。
    """

    # --- Declared defaults (overridable by config) ---
    category: TriggerCategory = TriggerCategory.UI_POPUP
    priority: int = 50
    trigger_mode: str = "polling"  # event | polling | hybrid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TriggerTask sets trigger_interval=0 (== always fire). Override with a
        # safe default so the legacy round-robin path never busy-loops even
        # before the scheduler computes effective params.
        self.trigger_interval = 3
        # Filled by _refresh_params() from centralized config.
        self._params: TriggerParams = TriggerParams()
        self._params_dirty = True

    # ------------------------------------------------------------------
    # Framework wiring
    # ------------------------------------------------------------------
    def _framework(self):
        """Return (config, throttle, metrics) from the executor scheduler, with
        local fallbacks for headless / unit-test usage."""
        scheduler = getattr(getattr(self, "_executor", None), "trigger_scheduler", None)
        if scheduler is not None:
            return scheduler.config, scheduler.throttle, scheduler.metrics
        # Local fallback (no executor): use shared config + private instances.
        cfg = get_shared_config()
        return cfg, _LocalThrottleHolder.throttle, _LocalThrottleHolder.metrics

    def _refresh_params(self) -> None:
        """(Re)compute effective params from centralized config and sync
        ``trigger_interval`` so the legacy round-robin path stays consistent."""
        cfg, _, _ = self._framework()
        declared = {
            "priority": self.priority,
            "trigger_mode": self.trigger_mode,
        }
        self._params = cfg.effective_params(self.name, self.category, declared)
        # Keep legacy should_trigger() cadence in sync with min_interval.
        self.trigger_interval = self._params.min_interval
        self._params_dirty = False

    def on_create(self):
        # TriggerTask.on_create reads _enabled from config; keep that behavior.
        super().on_create()
        self._refresh_params()

    def post_init(self):
        super().post_init()
        self._refresh_params()

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    def check(self, context: TriggerContext) -> TriggerDecision:
        """Cheap detection. Subclasses must override.

        默认实现：不处理。子类必须重写并返回 TriggerDecision。
        """
        return TriggerDecision(should_handle=False, reason="not_implemented")

    def handle(self, context: TriggerContext) -> TriggerResult:
        """Expensive action. Subclasses must override.

        默认实现：返回未处理。子类必须重写并返回 TriggerResult。
        """
        return TriggerResult.skip("not_implemented")

    # ------------------------------------------------------------------
    # should_trigger (legacy round-robin compatibility)
    # ------------------------------------------------------------------
    def should_trigger(self) -> bool:
        """Legacy hook used by the round-robin path / rollback mode.

        Honors ``min_interval`` and per-trigger cooldown via the shared throttle
        when available, else falls back to ``trigger_interval`` semantics.
        """
        if self._params_dirty:
            self._refresh_params()
        _, throttle, _ = self._framework()
        now = time.time()
        if isinstance(throttle, TriggerThrottle):
            allowed, _ = throttle.cool_to_run(self.name, self.category, self._params, now)
            # cool_to_run is stricter than the old trigger_interval check; allow
            # the legacy path to still tick so the executor keeps cycling, but
            # only actually fire when allowed. We return allowed so the executor
            # either runs us or waits via next_trigger_delay().
            return allowed
        # Fallback: old behavior based on trigger_interval.
        if self.trigger_interval == 0:
            return True
        now_t = time.time()
        if now_t - self.last_trigger_time > self.trigger_interval:
            self.last_trigger_time = now_t
            return True
        return False

    # ------------------------------------------------------------------
    # Main entry point (called by the executor)
    # ------------------------------------------------------------------
    def run(self) -> bool:
        """Safe, instrumented entry point. Never raises.

        返回 True 表示本次处理了事件（或确定无需处理），False 表示检测到但未处理。
        """
        if self._params_dirty:
            self._refresh_params()
        cfg, throttle, metrics = self._framework()
        now = time.time()
        context = TriggerContext(
            frame=self._current_frame(),
            executor=getattr(self, "_executor", None),
            now=now,
        )

        # --- 1. check() with watchdog timeout ---
        decision, check_timed_out = self._run_with_timeout(
            self.check, cfg.get("check_timeout_seconds", 0.0), context
        )
        if check_timed_out or decision is None:
            logger.warning(f"{self.name}: check() timed out")
            metrics.record_run(self.name, self.category, handled=False, success=False,
                               timeout=True, elapsed=0.0)
            return False
        metrics.record_checked(self.name, self.category)

        if not decision.should_handle:
            # No event detected: counts as a clean no-op cycle.
            metrics.record_run(self.name, self.category, handled=False, success=True,
                               elapsed=0.0)
            return False

        # --- 2. admit via dedup + budget ---
        admit = throttle.admit(
            self.name, decision.fingerprint, now, self._params.dedup_window_seconds
        )
        if not admit.allowed:
            kind = "dedup" if admit.kind == "dedup" else "budget"
            metrics.record_throttle(self.name, self.category, kind)
            logger.info(f"{self.name}: skipped ({admit.reason})")
            return False

        # --- 3. handle() with watchdog timeout ---
        context.deadline = (
            now + self._params.timeout_seconds if self._params.timeout_seconds > 0 else 0.0
        )
        start = time.time()
        result, handle_timed_out = self._run_with_timeout(
            self.handle, self._params.timeout_seconds, context
        )
        elapsed = time.time() - start
        if handle_timed_out or result is None:
            logger.warning(f"{self.name}: handle() timed out after {elapsed:.1f}s")
            result = TriggerResult.fail("handle_timeout", elapsed=elapsed)

        # Merge cooperative cost counters from the context.
        result.ocr_calls = result.ocr_calls or context.ocr_calls
        result.match_calls = result.match_calls or context.match_calls
        result.screenshot_calls = result.screenshot_calls or context.screenshot_calls
        result.retries = result.retries or context.retries
        result.elapsed = result.elapsed or elapsed

        # --- 4. record cooldown / dedup / budget / metrics ---
        throttle.record_run(self.name, self.category, decision.fingerprint, time.time())
        throttle.record_cost(result.ocr_calls, result.match_calls, time.time())
        metrics.record_run(
            self.name, self.category,
            handled=result.handled,
            success=result.success,
            timeout=handle_timed_out,
            elapsed=result.elapsed,
            ocr_calls=result.ocr_calls,
            match_calls=result.match_calls,
            screenshot_calls=result.screenshot_calls,
            retries=result.retries,
            error=result.error,
        )
        if result.handled:
            logger.info(
                f"{self.name}: handled success={result.success} "
                f"ocr={result.ocr_calls} match={result.match_calls} "
                f"elapsed={result.elapsed:.2f}s"
                + (f" error={result.error}" if result.error else "")
            )
        return result.handled and result.success

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_frame(self) -> Optional[np.ndarray]:
        """Best-effort current frame without forcing a capture (avoids extra cost)."""
        executor = getattr(self, "_executor", None)
        if executor is None:
            return None
        frame = getattr(executor, "_frame", None)
        if frame is not None:
            return frame
        # Fall back to the frame property (forces a capture) only if needed.
        try:
            return executor.nullable_frame()
        except Exception:
            return None

    def _run_with_timeout(self, fn, timeout: float, context: TriggerContext):
        """Run ``fn(context)`` in a daemon thread, bounded by ``timeout``.

        Returns (result, timed_out). On timeout the result is None and the
        orphan daemon thread continues to completion (hard cancellation of
        cv2/OCR is unsafe). Callers must keep timeouts generous.
        """
        if not timeout or timeout <= 0:
            try:
                return fn(context), False
            except Exception as e:
                logger.warning(f"{self.name}: {fn.__name__} raised {e}")
                return None, False

        box: dict = {"result": None, "error": None}
        done = threading.Event()

        def _target():
            try:
                box["result"] = fn(context)
            except Exception as e:  # never propagate to the executor
                box["error"] = e
            finally:
                done.set()

        t = threading.Thread(target=_target, name=f"{self.name}-{fn.__name__}", daemon=True)
        t.start()
        finished = done.wait(timeout=timeout)
        if not finished:
            return None, True
        if box["error"] is not None:
            logger.warning(f"{self.name}: {fn.__name__} raised {box['error']}")
            return None, False
        return box["result"], False


class _LocalThrottleHolder:
    """Process-local fallback throttle/metrics for headless/test usage."""

    throttle: TriggerThrottle = TriggerThrottle()
    metrics: TriggerMetrics = TriggerMetrics()
