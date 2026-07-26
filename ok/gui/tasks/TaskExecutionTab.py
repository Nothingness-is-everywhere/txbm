import time
from collections import deque
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QVBoxLayout, QWidget
)
from qfluentwidgets import (
    BodyLabel, FluentIcon, ProgressRing, StrongBodyLabel,
    ToolButton
)

from ok import Logger, og
from ok.gui.Communicate import communicate
from ok.gui.widget.Tab import Tab
from ok.gui.widget.Card import Card

logger = Logger.get_logger(__name__)

STATUS_MAP = {
    "completed": "已完成",
    "failed": "失败",
    "running": "运行中",
}

TYPE_MAP = {
    "trigger": "实时触发",
    "onetime": "周常日常",
}

CATEGORY_TITLE_MAP = {
    "trigger": "实时触发 - 执行状态",
    "onetime": "周常日常 - 执行状态",
    "all": "执行状态",
}

CATEGORY_CURRENT_TITLE_MAP = {
    "trigger": "当前运行的触发器任务",
    "onetime": "当前运行的日常任务",
    "all": "当前运行任务",
}

CATEGORY_HISTORY_TITLE_MAP = {
    "trigger": "触发器执行历史",
    "onetime": "日常任务执行历史",
    "all": "执行历史",
}


class TaskExecutionRecord:
    """单个任务执行记录"""

    def __init__(self, task_name: str, task_type: str):
        self.task_name = task_name
        self.task_type = task_type
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.status = "running"
        self.error_message: Optional[str] = None
        self.info: Dict = {}

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def complete(self, status: str = "completed", error_message: str = None):
        self.end_time = time.time()
        self.status = status
        self.error_message = error_message


class TaskExecutionTab(Tab):
    """任务执行情况标签页"""

    MAX_HISTORY = 50

    def __init__(self, task_category: str = "all"):
        super().__init__()
        self.task_category = task_category
        self.setObjectName(f"TaskExecutionTab_{task_category}")

        self.execution_history: deque = deque(maxlen=self.MAX_HISTORY)
        self.current_records: Dict[str, TaskExecutionRecord] = {}

        self._init_ui()
        self._connect_signals()
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._refresh_display)
        self._update_timer.start(500)

        self._refresh_display()

    def _init_ui(self):
        category_title = CATEGORY_TITLE_MAP.get(self.task_category, "执行状态")
        current_title = CATEGORY_CURRENT_TITLE_MAP.get(self.task_category, "当前运行任务")
        history_title = CATEGORY_HISTORY_TITLE_MAP.get(self.task_category, "执行历史")

        self.status_card = Card(self.tr(category_title), QWidget())
        status_content = self.status_card.widget
        status_layout = QVBoxLayout(status_content)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        self.status_row = QHBoxLayout()
        self.status_row.setSpacing(20)

        self.overall_status_label = BodyLabel(self.tr("空闲"))
        self.overall_status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.status_row.addWidget(self.overall_status_label)

        self.status_row.addStretch()

        self.total_label = BodyLabel("0")
        self.total_label.setStyleSheet("font-size: 14px;")
        self.success_label = BodyLabel("0")
        self.success_label.setStyleSheet("font-size: 14px; color: #2ecc71;")
        self.running_label = BodyLabel("0")
        self.running_label.setStyleSheet("font-size: 14px; color: #3498db;")

        self.status_row.addWidget(BodyLabel(self.tr("总数:")))
        self.status_row.addWidget(self.total_label)
        self.status_row.addSpacing(15)
        self.status_row.addWidget(BodyLabel(self.tr("成功:")))
        self.status_row.addWidget(self.success_label)
        self.status_row.addSpacing(15)
        self.status_row.addWidget(BodyLabel(self.tr("运行中:")))
        self.status_row.addWidget(self.running_label)

        status_layout.addLayout(self.status_row)

        self.progress_row = QHBoxLayout()
        self.progress_row.setSpacing(20)

        self.progress_ring = ProgressRing()
        self.progress_ring.setRange(0, 100)
        self.progress_ring.setValue(0)
        self.progress_ring.setStrokeWidth(8)
        self.progress_ring.setFixedSize(120, 120)

        self.progress_info_layout = QVBoxLayout()
        self.progress_info_layout.setSpacing(4)

        self.current_task_label = StrongBodyLabel(self.tr("当前无运行任务"))
        self.current_task_label.setWordWrap(True)
        self.elapsed_label = BodyLabel("")
        self.elapsed_label.setStyleSheet("color: gray;")

        self.progress_info_layout.addWidget(self.current_task_label)
        self.progress_info_layout.addWidget(self.elapsed_label)
        self.progress_info_layout.addStretch()

        self.progress_row.addWidget(self.progress_ring)
        self.progress_row.addLayout(self.progress_info_layout, 1)

        status_layout.addLayout(self.progress_row)

        self.add_widget(self.status_card)

        self.current_card = Card(self.tr(current_title), QWidget())
        self.current_content = self.current_card.widget
        self.current_layout = QVBoxLayout(self.current_content)
        self.current_layout.setContentsMargins(0, 0, 0, 0)
        self.current_layout.setSpacing(8)
        self.add_widget(self.current_card)

        self.history_card = Card(self.tr(history_title), QWidget())
        self.history_content = self.history_card.widget
        self.history_layout = QVBoxLayout(self.history_content)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(6)
        self.add_widget(self.history_card, 1)

        self.empty_history_label = BodyLabel(self.tr("暂无执行记录"))
        self.empty_history_label.setAlignment(Qt.AlignCenter)
        self.empty_history_label.setStyleSheet("color: gray; padding: 20px;")
        self.history_layout.addWidget(self.empty_history_label)

    def _connect_signals(self):
        communicate.task.connect(self._on_task_changed)
        communicate.task_done.connect(self._on_task_done)
        communicate.executor_paused.connect(self._on_executor_paused)

    def _get_task_type(self, task) -> str:
        if hasattr(task, 'trigger_interval') and task.trigger_interval > 0:
            return "trigger"
        return "onetime"

    def _is_task_in_category(self, task_type: str) -> bool:
        if self.task_category == "all":
            return True
        return task_type == self.task_category

    def _on_task_changed(self, task):
        if task is None:
            return

        task_type = self._get_task_type(task)
        if not self._is_task_in_category(task_type):
            return

        task_name = task.name if hasattr(task, 'name') else str(task)

        if task_name not in self.current_records or self.current_records[task_name].status != "running":
            record = TaskExecutionRecord(task_name, task_type)
            record.info = task.info if hasattr(task, 'info') and task.info else {}
            self.current_records[task_name] = record

    def _on_task_done(self, task):
        if task is None:
            return

        task_type = self._get_task_type(task)
        if not self._is_task_in_category(task_type):
            return

        task_name = task.name if hasattr(task, 'name') else str(task)

        if task_name in self.current_records:
            record = self.current_records.pop(task_name)
            if hasattr(task, 'info') and 'Error' in str(task.info):
                record.complete("failed", str(task.info))
            else:
                record.complete("completed")
            record.info = task.info if hasattr(task, 'info') else {}
            self.execution_history.append(record)

    def _on_executor_paused(self, paused: bool):
        if paused:
            self.overall_status_label.setText(self.tr("已暂停"))
            self.overall_status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #f39c12;")

    def _refresh_display(self):
        try:
            self._update_status_display()
            self._update_current_tasks_display()
            self._update_history_display()
        except Exception as e:
            logger.debug(f"TaskExecutionTab refresh error: {e}")

    def _update_status_display(self):
        executor = og.executor

        if executor is None:
            return

        if executor.paused:
            status_text = self.tr("已暂停")
            color = "#f39c12"
        elif executor.current_task is not None and executor.current_task.running:
            task_type = self._get_task_type(executor.current_task)
            if self._is_task_in_category(task_type):
                status_text = self.tr("运行中")
                color = "#2ecc71"
            else:
                status_text = self.tr("空闲")
                color = "#95a5a6"
        else:
            status_text = self.tr("空闲")
            color = "#95a5a6"

        self.overall_status_label.setText(status_text)
        self.overall_status_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")

        filtered_history = [r for r in self.execution_history if self._is_task_in_category(r.task_type)]
        filtered_current = {k: v for k, v in self.current_records.items() if self._is_task_in_category(v.task_type)}

        total = len(filtered_history) + len(filtered_current)
        success = sum(1 for r in filtered_history if r.status == "completed")
        running = len(filtered_current)

        self.total_label.setText(str(total))
        self.success_label.setText(str(success))
        self.running_label.setText(str(running))

        if executor.current_task is not None and executor.current_task.running:
            task = executor.current_task
            task_type = self._get_task_type(task)
            if self._is_task_in_category(task_type):
                self.current_task_label.setText(og.app.tr(task.name))
                elapsed = time.time() - task.start_time if hasattr(task, 'start_time') and task.start_time > 0 else 0
                self.elapsed_label.setText(self._format_duration(elapsed))
                self.progress_ring.setValue(75)
                self.progress_ring.setRange(0, 100)
            else:
                self.current_task_label.setText(self.tr("当前无运行任务"))
                self.elapsed_label.setText("")
                self.progress_ring.setValue(0)
        else:
            self.current_task_label.setText(self.tr("当前无运行任务"))
            self.elapsed_label.setText("")
            self.progress_ring.setValue(0)

    def _update_current_tasks_display(self):
        while self.current_layout.count():
            item = self.current_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        executor = og.executor
        if executor is None:
            return

        active_tasks = []
        if executor.current_task is not None:
            task_type = self._get_task_type(executor.current_task)
            if self._is_task_in_category(task_type):
                active_tasks.append(executor.current_task)

        visible_current = [r for r in self.current_records.values() if r.status == "running"
                          and self._is_task_in_category(r.task_type)]

        if not active_tasks and not visible_current:
            no_task_label = BodyLabel(self.tr("当前无运行任务"))
            no_task_label.setAlignment(Qt.AlignCenter)
            no_task_label.setStyleSheet("color: gray; padding: 15px;")
            self.current_layout.addWidget(no_task_label)
            return

        for task in active_tasks:
            task_widget = self._create_task_widget(task)
            self.current_layout.addWidget(task_widget)

    def _create_task_widget(self, task) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        indicator = QLabel()
        indicator.setFixedSize(10, 10)
        indicator.setStyleSheet("""
            QLabel {
                background-color: #2ecc71;
                border-radius: 5px;
            }
        """)
        layout.addWidget(indicator)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_label = StrongBodyLabel(og.app.tr(task.name))
        info_layout.addWidget(name_label)

        elapsed = time.time() - task.start_time if hasattr(task, 'start_time') and task.start_time > 0 else 0
        time_label = BodyLabel(self._format_duration(elapsed))
        time_label.setStyleSheet("color: gray; font-size: 12px;")
        info_layout.addWidget(time_label)

        if hasattr(task, 'info') and task.info:
            for key, value in list(task.info.items())[:3]:
                detail_label = BodyLabel(f"{og.app.tr(key)}: {og.app.tr(str(value))}")
                detail_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
                info_layout.addWidget(detail_label)

        layout.addLayout(info_layout, 1)

        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)

        if task.enabled and task.running:
            pause_btn = ToolButton(FluentIcon.PAUSE)
            pause_btn.setToolTip(self.tr("暂停"))
            pause_btn.clicked.connect(lambda: task.pause())
            button_layout.addWidget(pause_btn)

            stop_btn = ToolButton(FluentIcon.CANCEL)
            stop_btn.setToolTip(self.tr("停止"))
            stop_btn.clicked.connect(lambda: task.disable())
            button_layout.addWidget(stop_btn)

        layout.addLayout(button_layout)

        widget.setStyleSheet("""
            QWidget {
                background-color: rgba(46, 204, 113, 20);
                border-radius: 8px;
            }
        """)

        return widget

    def _update_history_display(self):
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        filtered_history = [r for r in self.execution_history if self._is_task_in_category(r.task_type)]

        if not filtered_history:
            self.empty_history_label = BodyLabel(self.tr("暂无执行记录"))
            self.empty_history_label.setAlignment(Qt.AlignCenter)
            self.empty_history_label.setStyleSheet("color: gray; padding: 20px;")
            self.history_layout.addWidget(self.empty_history_label)
            return

        for record in reversed(filtered_history):
            record_widget = self._create_history_record_widget(record)
            self.history_layout.addWidget(record_widget)

        self.history_layout.addStretch()

    def _create_history_record_widget(self, record: TaskExecutionRecord) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        icon_label = QLabel()
        icon_label.setFixedSize(24, 24)

        if record.status == "completed":
            icon_label.setStyleSheet("color: #2ecc71;")
            icon_label.setText("✓")
        elif record.status == "failed":
            icon_label.setStyleSheet("color: #e74c3c;")
            icon_label.setText("✗")
        else:
            icon_label.setStyleSheet("color: #3498db;")
            icon_label.setText("●")

        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(icon_label.styleSheet() + "font-size: 16px; font-weight: bold;")
        layout.addWidget(icon_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_row = QHBoxLayout()
        name_label = StrongBodyLabel(og.app.tr(record.task_name))
        name_row.addWidget(name_label)
        name_row.addStretch()

        type_label = BodyLabel(f"[{TYPE_MAP.get(record.task_type, record.task_type)}]")
        type_label.setStyleSheet("color: gray; font-size: 11px;")
        name_row.addWidget(type_label)

        info_layout.addLayout(name_row)

        time_row = QHBoxLayout()
        duration_label = BodyLabel(self._format_duration(record.duration))
        duration_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        time_row.addWidget(duration_label)
        time_row.addStretch()

        if record.end_time:
            from datetime import datetime
            end_time_str = datetime.fromtimestamp(record.end_time).strftime("%H:%M:%S")
            time_label = BodyLabel(end_time_str)
            time_label.setStyleSheet("color: #95a5a6; font-size: 11px;")
            time_row.addWidget(time_label)

        info_layout.addLayout(time_row)

        if record.status == "failed" and record.error_message:
            error_label = BodyLabel(og.app.tr(str(record.error_message))[:100])
            error_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
            error_label.setWordWrap(True)
            info_layout.addWidget(error_label)

        layout.addLayout(info_layout, 1)

        status_label = BodyLabel(STATUS_MAP.get(record.status, record.status))
        if record.status == "completed":
            status_label.setStyleSheet("color: #2ecc71; font-size: 12px; font-weight: bold;")
        elif record.status == "failed":
            status_label.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
        else:
            status_label.setStyleSheet("color: #3498db; font-size: 12px; font-weight: bold;")
        layout.addWidget(status_label)

        bg_color = "rgba(46, 204, 113, 10)" if record.status == "completed" else "rgba(231, 76, 60, 10)" if record.status == "failed" else "rgba(52, 152, 219, 10)"
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                border-radius: 6px;
            }}
        """)

        return widget

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{int(seconds * 1000)}ms"
        elif seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分 {secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}时 {minutes}分"
