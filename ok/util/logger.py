import argparse
import inspect
import logging
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections import deque
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
from typing import Optional

from ok.util.file import ensure_dir_for_file, get_relative_path

_ok_log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(threadName)s %(message)s')
_ok_logger = logging.getLogger("ok")
_OK_STDOUT_HANDLER = "_ok_stdout_handler"
_file_listener = None
_file_handler = None
_batched_handler = None


def _ensure_default_console_logger():
    if _ok_logger.handlers:
        return
    stdout_handler = logging.StreamHandler(sys.stdout)
    setattr(stdout_handler, _OK_STDOUT_HANDLER, True)
    stdout_handler.setFormatter(_ok_log_formatter)
    stdout_handler.setLevel(logging.DEBUG)
    _ok_logger.addHandler(stdout_handler)
    _ok_logger.setLevel(logging.DEBUG)


class Logger:
    def __init__(self, name: str):
        _ensure_default_console_logger()
        self.logger = _ok_logger
        self.name = name.split('.')[-1]

    def debug(self, message):
        self.logger.debug(f"{self.name}:{message}")

    def info(self, message):
        self.logger.info(f"{self.name}:{message}")

    def warning(self, message):
        self.logger.warning(f"{self.name}:{message}")

    def error(self, message, exception: Optional[Exception] = None):
        stack_trace_str = self.exception_to_str(exception)
        self.logger.error(f"{self.name}:{message} {stack_trace_str}")

    def critical(self, message):
        self.logger.critical(f"{self.name}:{message}")

    @staticmethod
    def call_stack() -> str:
        stack = ""
        for frame_info in inspect.stack():
            frame = frame_info.frame
            stack += (f"  File: {frame.f_code.co_filename}, "
                      f"Line: {frame.f_lineno}, "
                      f"Function: {frame.f_code.co_name}\n")
        return stack

    @staticmethod
    def get_logger(name: str):
        return Logger(name)

    @staticmethod
    def exception_to_str(exception: Exception) -> str:
        stack_trace_str = ""
        if exception is not None:
            try:
                stack_trace_str = traceback.format_exc()
            except Exception as e:
                stack_trace_str = f"Error formatting exception: {str(e)}"
        return stack_trace_str


def _log_exception_handler(exc_type, exc_value, exc_traceback):
    if _ok_logger is not None:
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        _ok_logger.error(f"Uncaught exception: {tb_text}")


sys.excepthook = _log_exception_handler


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.ERROR


class SafeFileHandler(TimedRotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.namer = self._rotation_filename

    def _rotation_filename(self, default_name):
        dir_name, file_name = os.path.split(default_name)
        base_name = os.path.basename(self.baseFilename)
        prefix = base_name + '.'
        if base_name.endswith('.log') and file_name.startswith(prefix):
            suffix = file_name[len(prefix):]
            stem = base_name[:-4]
            return os.path.join(dir_name, f'{stem}.{suffix}.log')
        return default_name

    def emit(self, record):
        try:
            if self.stream and not self.stream.closed:
                super().emit(record)
            else:
                raise ValueError("I/O operation on closed file.")
        except Exception:
            self.handleError(record)

    def getFilesToDelete(self):
        if self.backupCount <= 0:
            return []

        dir_name, base_name = os.path.split(self.baseFilename)
        if not base_name.endswith('.log'):
            return super().getFilesToDelete()

        stem = base_name[:-4]
        date_pattern = self.extMatch.pattern
        new_pattern = re.compile(rf'^{re.escape(stem)}\.({date_pattern})\.log$')
        legacy_pattern = re.compile(rf'^{re.escape(base_name)}\.({date_pattern})$')
        result = []

        for file_name in os.listdir(dir_name):
            new_match = new_pattern.fullmatch(file_name)
            legacy_match = legacy_pattern.fullmatch(file_name)
            if new_match or legacy_match:
                file_path = os.path.join(dir_name, file_name)
                result.append((os.path.getmtime(file_path), os.path.getctime(file_path), file_path))

        if len(result) <= self.backupCount:
            return []

        result.sort()
        return [path for _, _, path in result[:len(result) - self.backupCount]]


class CommunicateHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        from ok.gui.Communicate import communicate
        self.communicate = communicate

    def emit(self, record):
        log_message = self.format(record)
        self.communicate.log.emit(record.levelno, log_message)


class BatchedCommunicateHandler(logging.Handler):
    """Buffers log records and flushes them to the GUI thread at a bounded rate.

    Worker 线程只往线程安全 deque 追加（不触 UI）；守护线程按固定间隔批量 flush，
    限制 communicate.log 信号注入 GUI 事件队列的速率，防 debug 日志风暴拖慢主线程。
    回退：OK_LOG_BATCH_DISABLE=1 时 config_logger 改用直发 CommunicateHandler。
    """

    def __init__(self, flush_interval=0.15, max_batch=200):
        super().__init__()
        self._buf = deque()
        self._lock = threading.Lock()
        self._flush_interval = max(0.01, float(flush_interval))
        self._max_batch = max(1, int(max_batch))
        self._stop = False
        self._communicate = None
        self._thread = threading.Thread(target=self._loop, name="LogBatcher", daemon=True)
        self._thread.start()

    def emit(self, record):
        try:
            msg = self.format(record)
            with self._lock:
                self._buf.append((record.levelno, msg))
                # 防突发日志无界增长：超过 4 倍上限时丢弃最旧
                if len(self._buf) > self._max_batch * 4:
                    self._buf.popleft()
        except Exception:
            self.handleError(record)

    def _loop(self):
        while not self._stop:
            time.sleep(self._flush_interval)
            self._flush()
        self._flush()

    def _flush(self):
        with self._lock:
            if not self._buf:
                return
            batch = list(self._buf)
            self._buf.clear()
        if self._communicate is None:
            try:
                from ok.gui.Communicate import communicate
                self._communicate = communicate
            except Exception:
                return
        # 单次 flush 最多 max_batch 条，超量保留最新（保护 GUI 事件队列）
        if len(batch) > self._max_batch:
            batch = batch[-self._max_batch:]
        log_signal = self._communicate.log
        for levelno, msg in batch:
            log_signal.emit(levelno, msg)

    def stop(self):
        self._stop = True


def config_logger(config=None, name='ok-script'):
    global _file_listener, _file_handler, _batched_handler

    if _file_listener is not None:
        _file_listener.stop()
        _file_listener = None
    if _file_handler is not None:
        _file_handler.close()
        _file_handler = None
    if _batched_handler is not None:
        _batched_handler.stop()
        _batched_handler = None

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--parent_pid', type=int, help='Parent process ID', default=0)
    args, _ = parser.parse_known_args()

    if not config:
        config = {'debug': True}
    if config.get('debug'):
        _ok_logger.setLevel(logging.DEBUG)
    else:
        _ok_logger.setLevel(logging.INFO)

    if os.environ.get('OK_LOG_BATCH_DISABLE', '0') in ('1', 'true', 'yes', 'on'):
        # 回退：直发 handler，每条日志立即 emit（旧行为）
        communicate_handler = CommunicateHandler()
    else:
        # 默认：批处理 handler，worker 线程入队、守护线程按间隔批量 flush
        try:
            flush_interval = float(os.environ.get('OK_LOG_FLUSH_MS', '150')) / 1000.0
        except (TypeError, ValueError):
            flush_interval = 0.15
        try:
            max_batch = int(os.environ.get('OK_LOG_BATCH_MAX', '200'))
        except (TypeError, ValueError):
            max_batch = 200
        communicate_handler = BatchedCommunicateHandler(
            flush_interval=flush_interval, max_batch=max_batch)
        _batched_handler = communicate_handler
    communicate_handler.setFormatter(_ok_log_formatter)
    existing_stdout_handler = _get_stdout_handler()
    _ok_logger.handlers = [existing_stdout_handler] if existing_stdout_handler and args.parent_pid == 0 else []

    if args.parent_pid == 0:
        stdout_handler = existing_stdout_handler or logging.StreamHandler(sys.stdout)
        setattr(stdout_handler, _OK_STDOUT_HANDLER, True)
        stdout_handler.setFormatter(_ok_log_formatter)
        stdout_handler.filters = []
        stdout_handler.addFilter(InfoFilter())
        if config.get('debug'):
            stdout_handler.setLevel(logging.DEBUG)
        else:
            stdout_handler.setLevel(logging.INFO)
        if stdout_handler not in _ok_logger.handlers:
            _ok_logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_ok_log_formatter)
    stderr_handler.setLevel(logging.ERROR)
    _ok_logger.addHandler(stderr_handler)

    _ok_logger.addHandler(communicate_handler)
    logging.getLogger().handlers = []

    if _should_skip_file_logging(config):
        return

    logger_file = get_relative_path(os.path.join('logs', name + '.log'))
    ensure_dir_for_file(logger_file)

    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)
    _ok_logger.addHandler(queue_handler)

    _file_handler = SafeFileHandler(logger_file, when="midnight", interval=1,
                                    backupCount=7, encoding='utf-8')
    _file_handler.setFormatter(_ok_log_formatter)
    _file_handler.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)

    _file_listener = QueueListener(log_queue, _file_handler)
    _file_listener.start()


def _should_skip_file_logging(config):
    if config and config.get('disable_file_log'):
        return True
    return os.environ.get("OK_DISABLE_FILE_LOG") == "1" or "pytest" in sys.modules


def _get_stdout_handler():
    for handler in _ok_logger.handlers:
        if getattr(handler, _OK_STDOUT_HANDLER, False):
            return handler
    return None
