from ok import Logger, og

from ok.gui.tasks.TaskCard import TaskCard
from ok.gui.tasks.TaskTab import TaskTab

logger = Logger.get_logger(__name__)


class OneTimeTaskTab(TaskTab):
    def __init__(self, is_standalone=True, group_name=None):
        super().__init__()
        self.is_standalone = is_standalone
        self.group_name = group_name
        self.card_widgets = []

        # Check if this is an imported script to show delete button
        self.imported_file_name = None
        for fn, imp in og.task_manager.imported_scripts.items():
            if imp['script_name'] == self.group_name:
                self.imported_file_name = fn
                break

        # 顶部：批量启动/停止按钮
        from PySide6.QtWidgets import QHBoxLayout, QSpacerItem, QSizePolicy
        from qfluentwidgets import PushButton, FluentIcon

        self.top_btn_layout = QHBoxLayout()
        self.top_btn_layout.setContentsMargins(0, 0, 0, 10)
        self.top_btn_layout.setSpacing(10)

        self.start_all_btn = PushButton(self.tr('批量启动'), self, FluentIcon.PLAY)
        self.start_all_btn.setToolTip(self.tr('依次执行本分类下所有可见的周常日常任务'))
        self.start_all_btn.clicked.connect(self._start_all)
        self.top_btn_layout.addWidget(self.start_all_btn)

        self.stop_all_btn = PushButton(self.tr('批量停止'), self, FluentIcon.CANCEL)
        self.stop_all_btn.setToolTip(self.tr('停止执行器并清空任务队列'))
        self.stop_all_btn.clicked.connect(self._stop_all)
        self.top_btn_layout.addWidget(self.stop_all_btn)

        self.top_btn_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        if self.imported_file_name:
            self.delete_btn = PushButton(self.tr('Delete Script'), self, FluentIcon.DELETE)
            self.delete_btn.clicked.connect(self.delete_script)
            self.top_btn_layout.addWidget(self.delete_btn)

        # 批量启动按钮插到 vBoxLayout 开头（任务卡片之前）
        self.vBoxLayout.insertLayout(0, self.top_btn_layout)

        from ok.gui.Communicate import communicate
        communicate.task_list_updated.connect(self.refresh_ui)
        self.refresh_ui()

    def _start_all(self):
        if not getattr(self, 'tasks', None):
            return
        og.app.start_controller.start_onetime_all(list(self.tasks))

    def _stop_all(self):
        og.app.start_controller.stop_all()

    def delete_script(self):
        from qfluentwidgets import MessageBox
        w = MessageBox(self.tr('Confirm Delete'), 
                       self.tr('Are you sure you want to delete the script "{}"?').format(self.group_name), 
                       self.window())
        if w.exec():
            og.task_manager.delete_imported_script(self.imported_file_name)

    def refresh_ui(self):
        # Remove old cards
        for w in self.card_widgets:
            self.removeWidget(w)
            w.deleteLater()
        self.card_widgets.clear()

        self.tasks = []
        for task in og.executor.onetime_tasks:
            if not getattr(task, 'visible', True):
                continue
            task_group = getattr(task, 'group_name', None)
            if self.is_standalone and not task_group:
                self.tasks.append(task)
            elif self.group_name and task_group == self.group_name:
                self.tasks.append(task)

        for task in self.tasks:
            task_card = TaskCard(task, True)
            self.card_widgets.append(task_card)
            # Use vBoxLayout directly. Cards come after the top_btn_layout.
            self.vBoxLayout.addWidget(task_card)

    def in_current_list(self, task):
        return getattr(self, 'tasks', None) and task in self.tasks
