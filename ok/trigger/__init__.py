"""
Unified trigger framework package.

Provides a low-overhead, extensible, observable architecture for periodic
trigger tasks:

  - ``ManagedTriggerTask``: base class replacing ad-hoc busy-loop triggers with
    a ``check() -> TriggerDecision`` / ``handle() -> TriggerResult`` contract.
  - ``TriggerScheduler``: single point that selects which trigger runs next,
    applying priority, ``min_interval``, per-trigger / per-category cooldown,
    global budget, and dedup.
  - ``TriggerThrottle``: cooldown / dedup / budget primitives.
  - ``TriggerMetrics``: per-trigger and global counters for observability.
  - ``TriggerFrameworkConfig``: centralized config (category defaults +
    per-trigger overrides + rollback feature flag).

The scheduler integrates into ``TaskExecutor.next_task`` behind a feature flag
(``enable_managed_trigger`` / ``OK_TRIGGER_MANAGED`` env) so the old round-robin
path can be restored without code changes.

触发器统一调度框架包：低开销、可扩展、可观测。
"""

from ok.trigger.categories import TriggerCategory
from ok.trigger.decision import TriggerDecision, TriggerResult
from ok.trigger.config import TriggerFrameworkConfig
from ok.trigger.throttle import TriggerThrottle, AdmitDecision
from ok.trigger.metrics import TriggerMetrics
from ok.trigger.base import ManagedTriggerTask
from ok.trigger.scheduler import TriggerScheduler

__all__ = [
    "TriggerCategory",
    "TriggerDecision",
    "TriggerResult",
    "TriggerFrameworkConfig",
    "TriggerThrottle",
    "AdmitDecision",
    "TriggerMetrics",
    "ManagedTriggerTask",
    "TriggerScheduler",
]
