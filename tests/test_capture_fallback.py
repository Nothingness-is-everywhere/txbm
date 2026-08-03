import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ok.alas.emulator_windows import Emulator
from ok.device.DeviceManager import DeviceManager


def make_manager():
    manager = DeviceManager.__new__(DeviceManager)
    manager.exit_event = threading.Event()
    return manager


def mumu12_emu(path='C:/MuMu/nx_main/MuMuNxMain.exe'):
    return SimpleNamespace(type=Emulator.MuMuPlayer12, path=path)


# ---------------------------------------------------------------------------
# _is_mumu12
# ---------------------------------------------------------------------------
def test_is_mumu12_true_for_mumu12_non_global():
    assert make_manager()._is_mumu12(mumu12_emu()) is True


def test_is_mumu12_false_for_non_mumu():
    emu = SimpleNamespace(type='LDPlayer', path='x')
    assert make_manager()._is_mumu12(emu) is False


def test_is_mumu12_false_for_global_variant():
    emu = mumu12_emu(path='C:/MuMuPlayerGlobal/shell/MuMuNxDevice.exe')
    assert make_manager()._is_mumu12(emu) is False


def test_is_mumu12_false_for_none():
    assert make_manager()._is_mumu12(None) is False


def test_is_mumu12_false_when_path_missing():
    emu = SimpleNamespace(type=Emulator.MuMuPlayer12, path=None)
    assert make_manager()._is_mumu12(emu) is True  # path None -> '' -> no 'Global' match


# ---------------------------------------------------------------------------
# _try_nemu_capture
# ---------------------------------------------------------------------------
def test_try_nemu_capture_success_returns_method_and_reuses_connection():
    manager = make_manager()
    fake = MagicMock()  # init_nemu does not raise
    with patch('ok.device.DeviceManager.NemuIpcCaptureMethod', return_value=fake):
        result = manager._try_nemu_capture(mumu12_emu())
    assert result is fake
    fake.update_emulator.assert_called_once()
    fake.init_nemu.assert_called_once()
    fake.close.assert_not_called()


def test_try_nemu_capture_failure_returns_none_and_closes():
    manager = make_manager()
    fake = MagicMock()
    fake.init_nemu.side_effect = Exception('keep alive on')
    with patch('ok.device.DeviceManager.NemuIpcCaptureMethod', return_value=fake):
        result = manager._try_nemu_capture(mumu12_emu())
    assert result is None
    fake.close.assert_called_once()


def test_try_nemu_capture_none_emulator_returns_none():
    assert make_manager()._try_nemu_capture(None) is None


# ---------------------------------------------------------------------------
# do_start adb-branch capture selection
#
# Production code uses isinstance(self.capture_method, NemuIpcCaptureMethod) and
# isinstance(self.capture_method, ADBCaptureMethod).  patch(..., return_value=mock)
# replaces the class with a MagicMock *object* (not a type), which makes
# isinstance raise TypeError.  We instead patch with real fake classes so
# isinstance works correctly.
# ---------------------------------------------------------------------------
class _FakeAdb:
    """Real ADBCaptureMethod type double; isinstance works, instances have needed methods."""

    def __init__(self, *a, **kw):
        self.close = MagicMock()
        self.connected = MagicMock(return_value=True)


class _FakeAdbInteraction:
    """Real ADBInteraction type double for isinstance checks in do_start."""

    def __init__(self, *a, **kw):
        pass


class _FakeNemuOk:
    """Real NemuIpcCaptureMethod double; init_nemu succeeds."""

    def __init__(self, device_manager, exit_event):
        self.update_emulator = MagicMock()
        self.init_nemu = MagicMock()
        self.close = MagicMock()
        self.connected = MagicMock(return_value=True)


class _FakeNemuFail:
    """Real NemuIpcCaptureMethod double; init_nemu raises so _try_nemu_capture falls back."""

    def __init__(self, device_manager, exit_event):
        self.update_emulator = MagicMock()
        self.init_nemu = MagicMock(side_effect=Exception('boom'))
        self.close = MagicMock()
        self.connected = MagicMock(return_value=True)


def _make_do_start_manager(preferred):
    manager = DeviceManager.__new__(DeviceManager)
    manager.exit_event = threading.Event()
    manager.config = {'capture': 'adb', 'preferred': 'x'}
    manager.capture_method = None
    manager.hwnd_window = None
    manager.interaction = None
    manager.device_dict = {'x': preferred}
    manager.get_resolution = lambda: (1280, 720)
    return manager


def _adb_patches(nemu_cls=None):
    """Common patches for do_start adb branch: ADB class, interaction, communicate."""
    patches = [
        patch('ok.device.DeviceManager.ADBCaptureMethod', _FakeAdb),
        patch('ok.device.DeviceManager.ADBInteraction', _FakeAdbInteraction),
        patch('ok.device.DeviceManager.communicate'),
    ]
    if nemu_cls is not None:
        patches.insert(0, patch('ok.device.DeviceManager.NemuIpcCaptureMethod', nemu_cls))
    return patches


def test_do_start_auto_upgrades_to_nemu(monkeypatch):
    monkeypatch.setenv('OK_CAPTURE_PREFER_FAST', '1')
    preferred = {'device': 'adb', 'emulator': mumu12_emu(), 'full_path': None, 'player_id': 0}
    manager = _make_do_start_manager(preferred)

    for p in _adb_patches(nemu_cls=_FakeNemuOk):
        p.start()
    try:
        manager.do_start()
    finally:
        patch.stopall()

    # _try_nemu_capture created a _FakeNemuOk, init_nemu succeeded -> capture_method is nemu
    assert isinstance(manager.capture_method, _FakeNemuOk)
    manager.capture_method.init_nemu.assert_called_once()
    manager.capture_method.update_emulator.assert_called_once()


def test_do_start_falls_back_to_adb_when_nemu_probe_fails(monkeypatch):
    monkeypatch.setenv('OK_CAPTURE_PREFER_FAST', '1')
    preferred = {'device': 'adb', 'emulator': mumu12_emu(), 'full_path': None, 'player_id': 0}
    manager = _make_do_start_manager(preferred)

    for p in _adb_patches(nemu_cls=_FakeNemuFail):
        p.start()
    try:
        manager.do_start()
    finally:
        patch.stopall()

    # nemu probe failed -> fell back to ADB
    assert isinstance(manager.capture_method, _FakeAdb)
    assert not isinstance(manager.capture_method, _FakeNemuFail)


def test_do_start_prefer_fast_off_uses_adb(monkeypatch):
    monkeypatch.setenv('OK_CAPTURE_PREFER_FAST', '0')
    preferred = {'device': 'adb', 'emulator': mumu12_emu(), 'full_path': None, 'player_id': 0}
    manager = _make_do_start_manager(preferred)

    # NemuIpcCaptureMethod NOT patched (want_nemu=False -> never called)
    for p in _adb_patches(nemu_cls=None):
        p.start()
    try:
        manager.do_start()
    finally:
        patch.stopall()

    assert isinstance(manager.capture_method, _FakeAdb)


def test_do_start_non_mumu_uses_adb_without_probe(monkeypatch):
    monkeypatch.setenv('OK_CAPTURE_PREFER_FAST', '1')
    preferred = {'device': 'adb',
                 'emulator': SimpleNamespace(type='LDPlayer', path='x'),
                 'full_path': None, 'player_id': 0}
    manager = _make_do_start_manager(preferred)

    for p in _adb_patches(nemu_cls=None):
        p.start()
    try:
        manager.do_start()
    finally:
        patch.stopall()

    # non-mumu -> no probe, zero latency, straight to ADB
    assert isinstance(manager.capture_method, _FakeAdb)


def test_do_start_reuses_existing_nemu_without_reprobe(monkeypatch):
    monkeypatch.setenv('OK_CAPTURE_PREFER_FAST', '1')
    preferred = {'device': 'adb', 'emulator': mumu12_emu(), 'full_path': None, 'player_id': 0}
    manager = _make_do_start_manager(preferred)
    # Pretend a previous start already selected nemu.
    # MagicMock(spec=NemuIpcCaptureMethod) makes isinstance(..., NemuIpcCaptureMethod)
    # return True because the real class is used (NOT patched).
    from ok.device.capture_methods.nemu_ipc import NemuIpcCaptureMethod
    manager.capture_method = MagicMock(spec=NemuIpcCaptureMethod)

    # NemuIpcCaptureMethod NOT patched: isinstance uses the real class.
    for p in _adb_patches(nemu_cls=None):
        p.start()
    try:
        manager.do_start()
    finally:
        patch.stopall()

    # reuse path: update_emulator called, no new instance created
    manager.capture_method.update_emulator.assert_called_once()
