from ok.gui.widget.Tab import Tab


class TaskTab(Tab):
    """任务列表页签基类（触发器页 / 日常页继承）。

    历史上此处会渲染一个“Info/Value”大表格（task_info_table，固定高 300）展示
    当前任务的运行信息；该面板在触发器页与日常页都会出现，但实际价值有限且占用
    大量纵向空间，已移除。任务运行信息改为通过日志与任务卡片自身展示。
    """

    def __init__(self):
        super().__init__()

    def create_task(self):
        pass

    def in_current_list(self, task):
        return True
