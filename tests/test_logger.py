import logging
import time

from ok.util.logger import (
    BatchedCommunicateHandler,
    CommunicateHandler,
    _ok_logger,
    config_logger,
)


def test_config_logger_does_not_create_file_log_during_pytest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config_logger({"debug": False})

    assert not (tmp_path / "logs").exists()


# ---------------------------------------------------------------------------
# BatchedCommunicateHandler
# ---------------------------------------------------------------------------
class _FakeLog:
    def __init__(self):
        self.emitted = []

    def emit(self, levelno, msg):
        self.emitted.append((levelno, msg))


class _FakeCommunicate:
    def __init__(self):
        self.log = _FakeLog()


def _record(message):
    return logging.LogRecord("ok", logging.INFO, __file__, 1, message, None, None)


def test_batched_handler_flushes_in_batches():
    handler = BatchedCommunicateHandler(flush_interval=0.05, max_batch=200)
    handler.setFormatter(logging.Formatter("%(message)s"))
    fake = _FakeCommunicate()
    handler._communicate = fake

    for i in range(5):
        handler.emit(_record(f"msg{i}"))

    time.sleep(0.15)  # allow the daemon to flush
    handler.stop()

    assert [msg for _, msg in fake.log.emitted] == [f"msg{i}" for i in range(5)]


def test_batched_handler_truncates_to_max_batch():
    # Long flush interval so the daemon does not interfere; we flush manually.
    handler = BatchedCommunicateHandler(flush_interval=10.0, max_batch=3)
    handler.setFormatter(logging.Formatter("%(message)s"))
    fake = _FakeCommunicate()
    handler._communicate = fake

    for i in range(10):
        handler.emit(_record(f"m{i}"))

    handler._flush()
    handler.stop()

    # Keeps only the latest max_batch records to protect the GUI event queue.
    assert len(fake.log.emitted) == 3
    assert fake.log.emitted[-1][1] == "m9"


def test_config_logger_uses_batched_handler_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OK_LOG_BATCH_DISABLE", raising=False)

    config_logger({"debug": False})

    assert any(isinstance(h, BatchedCommunicateHandler) for h in _ok_logger.handlers)


def test_config_logger_falls_back_to_direct_handler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OK_LOG_BATCH_DISABLE", "1")

    config_logger({"debug": False})

    assert any(
        isinstance(h, CommunicateHandler) and not isinstance(h, BatchedCommunicateHandler)
        for h in _ok_logger.handlers
    )

