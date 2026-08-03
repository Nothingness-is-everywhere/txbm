import threading
import time
from unittest.mock import MagicMock

import pytest

from ok.task.TaskExecutor import TaskExecutor
from ok.task.exceptions import FinishedException


def make_executor(min_interval, last_frame_time):
    """Build a TaskExecutor shell with only the attrs next_frame touches."""
    ex = TaskExecutor.__new__(TaskExecutor)
    ex._min_frame_interval = min_interval
    ex._last_frame_time = last_frame_time
    ex.exit_event = threading.Event()
    ex.reset_scene = MagicMock()
    ex._wait_for_activity = MagicMock(return_value=False)
    return ex


def test_next_frame_throttle_waits_when_frame_too_recent():
    ex = make_executor(min_interval=0.05, last_frame_time=time.time())
    ex.exit_event.set()  # skip the capture loop; we only care about the throttle wait

    with pytest.raises(FinishedException):
        ex.next_frame(time_out=6)

    ex._wait_for_activity.assert_called_once()
    waited = ex._wait_for_activity.call_args[0][0]
    assert 0 < waited <= 0.05


def test_next_frame_no_throttle_when_disabled():
    ex = make_executor(min_interval=0.0, last_frame_time=time.time())
    ex.exit_event.set()

    with pytest.raises(FinishedException):
        ex.next_frame(time_out=6)

    ex._wait_for_activity.assert_not_called()


def test_next_frame_no_throttle_on_first_frame():
    ex = make_executor(min_interval=0.05, last_frame_time=0)
    ex.exit_event.set()

    with pytest.raises(FinishedException):
        ex.next_frame(time_out=6)

    ex._wait_for_activity.assert_not_called()


def test_next_frame_throttle_capped_by_timeout():
    # timeout smaller than the min interval -> wait capped to the remaining timeout
    ex = make_executor(min_interval=5.0, last_frame_time=time.time())
    ex.exit_event.set()

    with pytest.raises(FinishedException):
        ex.next_frame(time_out=0.1)

    ex._wait_for_activity.assert_called_once()
    waited = ex._wait_for_activity.call_args[0][0]
    assert 0 < waited <= 0.1
