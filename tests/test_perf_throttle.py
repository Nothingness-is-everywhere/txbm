import threading
import time
from unittest.mock import MagicMock, patch

from ok.device.DeviceManager import DeviceManager


def make_manager():
    manager = DeviceManager.__new__(DeviceManager)
    manager.exit_event = threading.Event()
    manager._adb_lock = threading.Lock()
    manager._adb_connect_state = {}
    manager._last_refresh_time = 0.0
    manager._last_kill_adb_time = 0.0
    manager._refresh_cooldown = 2.0
    manager._adb_backoff_initial = 1.0
    manager._adb_backoff_max = 8.0
    manager._adb_kill_cooldown = 10.0
    return manager


# ---------------------------------------------------------------------------
# adb_connect exponential backoff
# ---------------------------------------------------------------------------
def test_backoff_sequence_is_1_2_4_8_capped():
    manager = make_manager()
    manager._record_adb_connect_failure('a')
    assert manager._adb_connect_state['a']['backoff'] == 1.0
    manager._record_adb_connect_failure('a')
    assert manager._adb_connect_state['a']['backoff'] == 2.0
    manager._record_adb_connect_failure('a')
    assert manager._adb_connect_state['a']['backoff'] == 4.0
    manager._record_adb_connect_failure('a')
    assert manager._adb_connect_state['a']['backoff'] == 8.0
    manager._record_adb_connect_failure('a')
    assert manager._adb_connect_state['a']['backoff'] == 8.0  # capped


def test_backoff_remaining_is_zero_when_no_state():
    manager = make_manager()
    assert manager._adb_connect_backoff_remaining('a') == 0.0


def test_backoff_remaining_positive_within_window():
    manager = make_manager()
    manager._record_adb_connect_failure('a')  # 1s backoff
    remaining = manager._adb_connect_backoff_remaining('a')
    assert 0 < remaining <= 1.0


def test_clear_backoff_resets_state():
    manager = make_manager()
    manager._record_adb_connect_failure('a')
    assert 'a' in manager._adb_connect_state
    manager._clear_adb_connect_backoff('a')
    assert 'a' not in manager._adb_connect_state
    assert manager._adb_connect_backoff_remaining('a') == 0.0


def test_adb_connect_skips_during_backoff_window():
    manager = make_manager()
    mock_adb = MagicMock()
    manager._adb = mock_adb  # bypass the lazy `adb` property
    manager._record_adb_connect_failure('127.0.0.1:7555')  # 1s backoff

    result = manager.adb_connect('127.0.0.1:7555')

    assert result is None
    # Backoff returns before touching adb.list(), avoiding thrash.
    mock_adb.list.assert_not_called()


def test_adb_connect_attempts_after_backoff_expires():
    manager = make_manager()
    mock_adb = MagicMock()
    mock_adb.list.return_value = []  # no devices connected
    manager._adb = mock_adb
    # Simulate a backoff whose window already expired.
    manager._adb_connect_state = {'127.0.0.1:7555': {'next_allowed': time.time() - 1, 'backoff': 1.0}}

    manager.adb_connect('127.0.0.1:7555')

    # Backoff expired -> adb.list() was called (may be called multiple times
    # due to recursive confirm + debug f-string, but at least once).
    mock_adb.list.assert_called()
    assert mock_adb.list.call_count >= 1


# ---------------------------------------------------------------------------
# do_refresh cooldown
# ---------------------------------------------------------------------------
def _stub_refresh_internals(manager):
    manager.refresh_emulators = MagicMock()
    manager.refresh_phones = MagicMock()
    manager.update_pc_device = MagicMock()
    manager.update_browser_device = MagicMock()
    manager.do_start = MagicMock()
    manager.device_dict = {}  # do_refresh logs self.device_dict at the end


def test_do_refresh_skipped_within_cooldown():
    manager = make_manager()
    _stub_refresh_internals(manager)
    manager._last_refresh_time = time.time()  # just refreshed

    manager.do_refresh()

    manager.refresh_emulators.assert_not_called()
    manager.do_start.assert_not_called()


def test_do_refresh_runs_after_cooldown():
    manager = make_manager()
    _stub_refresh_internals(manager)
    manager._last_refresh_time = time.time() - 5  # past the 2s cooldown

    manager.do_refresh()

    manager.refresh_emulators.assert_called_once()
    manager.do_start.assert_called_once()


def test_do_refresh_cooldown_disabled_runs_immediately():
    manager = make_manager()
    manager._refresh_cooldown = 0.0
    _stub_refresh_internals(manager)

    manager.do_refresh()
    manager.do_refresh()  # second call should also run since cooldown disabled

    assert manager.refresh_emulators.call_count == 2
    assert manager.do_start.call_count == 2


# ---------------------------------------------------------------------------
# try_kill_adb rate limiting
# ---------------------------------------------------------------------------
def test_try_kill_adb_skipped_within_cooldown():
    manager = make_manager()
    manager._last_kill_adb_time = time.time()  # just killed

    with patch('psutil.process_iter') as pi:
        manager.try_kill_adb(Exception('x'))
        pi.assert_not_called()  # rate-limited before touching psutil


def test_try_kill_adb_runs_after_cooldown():
    manager = make_manager()
    manager._last_kill_adb_time = time.time() - 20  # past the 10s cooldown

    with patch('psutil.process_iter', return_value=[]):
        manager.try_kill_adb(Exception('x'))
        # process_iter was exercised; no real processes killed (return_value=[])
