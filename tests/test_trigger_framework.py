"""
Unit tests for the unified trigger framework (ok.trigger).

Run with:
    python -m pytest tests/test_trigger_framework.py -v

Covers:
  - TriggerFrameworkConfig: category/override merge + env rollback flag
  - TriggerThrottle: cooldown / dedup / budget
  - TriggerScheduler: priority selection, cooldown skip, next_trigger_delay
  - ManagedTriggerTask.run(): check->handle flow, no-raise safety, timeout
  - Rollback switch wired into TaskExecutor._managed_trigger_enabled
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ok.trigger.categories import TriggerCategory
from ok.trigger.config import TriggerFrameworkConfig, TriggerParams
from ok.trigger.decision import TriggerDecision, TriggerResult
from ok.trigger.metrics import TriggerMetrics
from ok.trigger.throttle import TriggerThrottle
from ok.trigger.scheduler import TriggerScheduler
from ok.trigger.base import ManagedTriggerTask, TriggerContext


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
class TestTriggerConfig(unittest.TestCase):
    def test_defaults_when_no_file(self):
        cfg = TriggerFrameworkConfig(path=str(_PROJECT_ROOT / "configs" / "__nonexistent__.json"))
        self.assertTrue(cfg.enable_managed_trigger)
        p = cfg.effective_params("X", TriggerCategory.NETWORK)
        self.assertEqual(p.priority, 80)  # category default
        self.assertGreater(p.min_interval, 0)

    def test_category_default_applied(self):
        cfg = TriggerFrameworkConfig(raw={})
        p = cfg.effective_params("any", TriggerCategory.UI_POPUP)
        self.assertEqual(p.priority, 50)
        self.assertEqual(p.min_interval, 8.0)

    def test_trigger_override_wins(self):
        cfg = TriggerFrameworkConfig(raw={
            "trigger_overrides": {"MyTrigger": {"priority": 99, "min_interval": 1.0}}
        })
        p = cfg.effective_params("MyTrigger", TriggerCategory.NETWORK)
        self.assertEqual(p.priority, 99)      # override
        self.assertEqual(p.min_interval, 1.0) # override
        self.assertEqual(p.cooldown_seconds, 5.0)  # category default retained

    def test_declared_attrs_merged(self):
        cfg = TriggerFrameworkConfig(raw={})
        p = cfg.effective_params("X", TriggerCategory.NETWORK,
                                 declared={"priority": 77})
        self.assertEqual(p.priority, 77)

    def test_env_override_disables_managed(self):
        cfg = TriggerFrameworkConfig(raw={"enable_managed_trigger": True})
        self.assertTrue(cfg.enable_managed_trigger)
        old = os.environ.get("OK_TRIGGER_MANAGED")
        try:
            os.environ["OK_TRIGGER_MANAGED"] = "0"
            self.assertFalse(cfg.enable_managed_trigger)
            os.environ["OK_TRIGGER_MANAGED"] = "false"
            self.assertFalse(cfg.enable_managed_trigger)
            os.environ["OK_TRIGGER_MANAGED"] = "1"
            self.assertTrue(cfg.enable_managed_trigger)
        finally:
            if old is None:
                os.environ.pop("OK_TRIGGER_MANAGED", None)
            else:
                os.environ["OK_TRIGGER_MANAGED"] = old


# --------------------------------------------------------------------------
# Throttle
# --------------------------------------------------------------------------
class TestTriggerThrottle(unittest.TestCase):
    def setUp(self):
        self.throttle = TriggerThrottle(
            global_min_interval=0.0,  # disable global spacing for unit tests
            budget_window_seconds=60.0,
            budget_runs_per_window=3,
            budget_ocr_per_window=5,
            budget_match_per_window=10,
        )
        self.params = TriggerParams(min_interval=2.0, cooldown_seconds=2.0,
                                    dedup_window_seconds=5.0)
        self.cat = TriggerCategory.NETWORK

    def test_min_interval_blocks(self):
        now = 100.0
        self.throttle.record_run("t", self.cat, "fp1", now)
        allowed, _ = self.throttle.cool_to_run("t", self.cat, self.params, now + 1.0)
        self.assertFalse(allowed)
        allowed, _ = self.throttle.cool_to_run("t", self.cat, self.params, now + 3.0)
        self.assertTrue(allowed)

    def test_category_cooldown_blocks_other_trigger(self):
        now = 100.0
        self.throttle.record_run("t1", self.cat, "fp1", now)
        # Different trigger, same category, within category cooldown.
        allowed, _ = self.throttle.cool_to_run("t2", self.cat, self.params, now + 1.0)
        self.assertFalse(allowed)

    def test_dedup_blocks_same_fingerprint(self):
        now = 100.0
        admit = self.throttle.admit("t", "fp1", now, dedup_window_seconds=5.0)
        self.assertTrue(admit.allowed)
        self.throttle.record_run("t", self.cat, "fp1", now)
        admit2 = self.throttle.admit("t", "fp1", now + 1.0, dedup_window_seconds=5.0)
        self.assertFalse(admit2.allowed)
        self.assertEqual(admit2.kind, "dedup")

    def test_dedup_allows_different_fingerprint(self):
        now = 100.0
        self.throttle.record_run("t", self.cat, "fp1", now)
        admit = self.throttle.admit("t", "fp2", now + 1.0, dedup_window_seconds=5.0)
        self.assertTrue(admit.allowed)

    def test_budget_runs_limit(self):
        now = 100.0
        for i in range(3):
            self.throttle.record_run("t", self.cat, f"fp{i}", now + i)
        admit = self.throttle.admit("t", "fp_new", now + 10, dedup_window_seconds=5.0)
        self.assertFalse(admit.allowed)
        self.assertEqual(admit.kind, "budget_runs")

    def test_budget_ocr_match(self):
        now = 100.0
        # Fill OCR budget.
        a = self.throttle.budget_available(ocr_calls=5, match_calls=0, now=now)
        self.assertTrue(a.allowed)
        self.throttle.record_cost(5, 0, now)
        a2 = self.throttle.budget_available(ocr_calls=1, match_calls=0, now=now + 1)
        self.assertFalse(a2.allowed)
        self.assertEqual(a2.kind, "budget_ocr")


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------
class _FakeManagedTask(ManagedTriggerTask):
    """Minimal ManagedTriggerTask stand-in built via __new__ (avoids heavy init)."""
    def __init__(self, name, category, priority, min_interval, enabled=True):
        self.name = name
        self.category = category
        self._enabled = enabled
        self._params = TriggerParams(priority=priority, min_interval=min_interval,
                                     cooldown_seconds=min_interval)
        self._params_dirty = False


class TestTriggerScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = TriggerFrameworkConfig(raw={
            "global_min_interval": 0.0,
            "budget_runs_per_window": 100,
        })
        self.sched = TriggerScheduler(self.cfg)

    def test_select_highest_priority(self):
        t_low = _FakeManagedTask("low", TriggerCategory.UI_POPUP, priority=40, min_interval=0.0)
        t_high = _FakeManagedTask("high", TriggerCategory.NETWORK, priority=80, min_interval=0.0)
        chosen = self.sched.select([t_low, t_high])
        self.assertIs(chosen, t_high)

    def test_select_skips_cooldown(self):
        t = _FakeManagedTask("net", TriggerCategory.NETWORK, priority=80, min_interval=10.0)
        # Record a run so it enters cooldown.
        self.sched.throttle.record_run("net", TriggerCategory.NETWORK, "fp", time.time())
        chosen = self.sched.select([t])
        self.assertIsNone(chosen)

    def test_select_skips_disabled(self):
        t = _FakeManagedTask("net", TriggerCategory.NETWORK, priority=80, min_interval=0.0,
                             enabled=False)
        self.assertIsNone(self.sched.select([t]))

    def test_next_trigger_delay(self):
        t = _FakeManagedTask("net", TriggerCategory.NETWORK, priority=80, min_interval=5.0)
        now = time.time()
        self.sched.throttle.record_run("net", TriggerCategory.NETWORK, "fp", now)
        delay = self.sched.next_trigger_delay([t], now=now, default=1.0)
        self.assertGreater(delay, 4.0)
        self.assertLessEqual(delay, 5.0)

    def test_no_managed_returns_none(self):
        # Plain object (not ManagedTriggerTask) is ignored.
        plain = MagicMock()
        self.assertIsNone(self.sched.select([plain]))


# --------------------------------------------------------------------------
# ManagedTriggerTask.run()
# --------------------------------------------------------------------------
class _DummyTrigger(ManagedTriggerTask):
    category = TriggerCategory.UI_POPUP

    def __init__(self):  # bypass heavy TriggerTask.__init__
        pass

    def check(self, context):
        return self._check(context)

    def handle(self, context):
        return self._handle(context)


def _make_dummy(executor):
    t = _DummyTrigger.__new__(_DummyTrigger)
    t.name = "Dummy"
    t.category = TriggerCategory.UI_POPUP
    t._executor = executor
    t._params = TriggerParams(min_interval=0.0, cooldown_seconds=0.0,
                              dedup_window_seconds=0.0, timeout_seconds=2.0,
                              priority=50)
    t._params_dirty = False
    t._check = MagicMock(return_value=TriggerDecision(should_handle=False, reason="none"))
    t._handle = MagicMock(return_value=TriggerResult.skip())
    return t


def _make_executor_with_scheduler():
    cfg = TriggerFrameworkConfig(raw={
        "global_min_interval": 0.0,
        "budget_runs_per_window": 1000,
        "budget_ocr_per_window": 1000,
        "budget_match_per_window": 1000,
        "check_timeout_seconds": 2.0,
        "metrics_log_interval_seconds": 0,
    })
    sched = TriggerScheduler(cfg)
    executor = MagicMock()
    executor.trigger_scheduler = sched
    executor._frame = np.zeros((10, 10, 3), dtype=np.uint8)
    executor.nullable_frame.return_value = executor._frame
    return executor, sched


class TestManagedTriggerRun(unittest.TestCase):
    def test_no_event_returns_false(self):
        executor, sched = _make_executor_with_scheduler()
        t = _make_dummy(executor)
        t._check.return_value = TriggerDecision(should_handle=False)
        self.assertFalse(t.run())
        self.assertEqual(sched.metrics.snapshot()["Dummy"]["checked"], 1)
        self.assertEqual(sched.metrics.snapshot()["Dummy"]["handled"], 0)

    def test_handle_success(self):
        executor, sched = _make_executor_with_scheduler()
        t = _make_dummy(executor)
        t._check.return_value = TriggerDecision(should_handle=True, fingerprint="fp1")
        t._handle.return_value = TriggerResult.ok(match_calls=3)
        self.assertTrue(t.run())
        snap = sched.metrics.snapshot()["Dummy"]
        self.assertEqual(snap["handled"], 1)
        self.assertEqual(snap["success"], 1)
        self.assertEqual(snap["match_calls"], 3)

    def test_run_never_raises_on_check_error(self):
        executor, sched = _make_executor_with_scheduler()
        t = _make_dummy(executor)
        t._check.side_effect = RuntimeError("boom")
        # Must not propagate; returns False.
        self.assertFalse(t.run())

    def test_run_never_raises_on_handle_error(self):
        executor, sched = _make_executor_with_scheduler()
        t = _make_dummy(executor)
        t._check.return_value = TriggerDecision(should_handle=True, fingerprint="fp1")
        t._handle.side_effect = RuntimeError("boom")
        self.assertFalse(t.run())

    def test_dedup_skips_handle(self):
        executor, sched = _make_executor_with_scheduler()
        t = _make_dummy(executor)
        t._params.dedup_window_seconds = 10.0
        t._check.return_value = TriggerDecision(should_handle=True, fingerprint="fp1")
        t._handle.return_value = TriggerResult.ok()
        self.assertTrue(t.run())           # first run handled
        t._handle.return_value = TriggerResult.ok()
        self.assertFalse(t.run())          # second run deduped -> not handled
        self.assertEqual(sched.metrics.snapshot()["Dummy"]["dedup_hits"], 1)
        self.assertEqual(sched.metrics.snapshot()["Dummy"]["handled"], 1)

    def test_handle_timeout(self):
        executor, sched = _make_executor_with_scheduler()
        t = _make_dummy(executor)
        t._params.timeout_seconds = 0.2
        t._check.return_value = TriggerDecision(should_handle=True, fingerprint="fp1")

        def slow_handle(ctx):
            time.sleep(0.6)
            return TriggerResult.ok()

        t._handle = slow_handle
        self.assertFalse(t.run())
        self.assertEqual(sched.metrics.snapshot()["Dummy"]["timeout"], 1)


# --------------------------------------------------------------------------
# Rollback switch in TaskExecutor
# --------------------------------------------------------------------------
class TestRollbackSwitch(unittest.TestCase):
    def test_managed_flag_reads_env(self):
        # Import lazily so PySide6 import cost is paid only when this test runs.
        from ok.task.TaskExecutor import TaskExecutor
        cfg = TriggerFrameworkConfig(raw={"enable_managed_trigger": True})
        executor = TaskExecutor.__new__(TaskExecutor)
        executor.trigger_scheduler = TriggerScheduler(cfg)
        self.assertTrue(executor._managed_trigger_enabled())
        old = os.environ.get("OK_TRIGGER_MANAGED")
        try:
            os.environ["OK_TRIGGER_MANAGED"] = "0"
            self.assertFalse(executor._managed_trigger_enabled())
        finally:
            if old is None:
                os.environ.pop("OK_TRIGGER_MANAGED", None)
            else:
                os.environ["OK_TRIGGER_MANAGED"] = old


if __name__ == "__main__":
    unittest.main()
