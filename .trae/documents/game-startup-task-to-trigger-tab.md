# 将 GameStartupTask 移到触发器分类并修复日常任务自动启动

## Context（背景）

**问题现象**：启动 GUI 时会同时自动执行"周常日常"分类下的任务，而期望是启动后只跑触发器、日常任务需手动触发。

**根因**（两条独立的问题链）：

1. **日常任务默认自动启用**：`HomeRedDotTask` / `AlchemyDispatchTask` / `RecruitTask` 的 `on_create` 里 `self._enabled = self.config.get("_enabled", True)`，且 `DEFAULT_CONFIG["_enabled"] = True`，持久化文件 `configs/*.json` 里也存了 `_enabled: true`。`next_task()` 遍历 `onetime_tasks` 时只要 `enabled` 就会选中执行（[TaskExecutor.py:505-507](file:///d:/学习/python/ok-script/ok/task/TaskExecutor.py#L505-L507)），所以启动 GUI 后日常任务立刻跑。

2. **GameStartupTask 归类不当**：`GameStartupTask` 当前在 `onetime_tasks`（[start\_gui.py:28](file:///d:/学习/python/ok-script/start_gui.py#L28)），显示在"周常日常"tab。用户希望它归到"触发器"分类显示，且每次启动 GUI 触发一次。

**关键技术约束**：`GameStartupTask` 是 one-shot 启动流程（处理弹窗→进主界面→读体力），不是周期触发器。直接塞进 `trigger_tasks` 不可行，因为：

* `enqueue_onetime_task`([TaskExecutor.py:463-466](file:///d:/学习/python/ok-script/ok/task/TaskExecutor.py#L463-L466)) 对不在 `onetime_tasks` 的任务直接 `return False` → `enable_after_start` 机制虽调 `_mark_task_enabled`，但入队静默失败

* managed scheduler 默认启用（[config.py:41](file:///d:/学习/python/ok-script/ok/trigger/config.py#L41)），`select` 只挑 `ManagedTriggerTask`（[scheduler.py:74](file:///d:/学习/python/ok-script/ok/trigger/scheduler.py#L74)），普通 `TriggerTask` 不会被周期选中

* legacy 路径调 `should_trigger()`，`TriggerTask` 基类无此方法

**方案**：新增 `trigger_fire_once_queue` 通路，让"非 ManagedTriggerTask 的 TriggerTask"（即 GameStartupTask 这类 one-shot 触发器）能通过 `enable_after_start` 入队并以 onetime 路径执行一次（run 后自动 disable）。GameStartupTask 改继承 `TriggerTask`，显示在"触发器"tab。日常任务 `on_create` 强制 `_enabled=False`。

**预期结果**：启动 GUI → GameStartupTask 自动跑一次 → NetworkErrorHandler 周期跑 → 日常任务全部禁用等待手动触发。

***

## 改动详情

### 1. `ok/task/TaskExecutor.py` — 新增 fire\_once 一次性触发队列

**改动 A —** **`__init__`（行 90 附近）**：在 `self.onetime_task_queue = []` 后新增：

```python
self.trigger_fire_once_queue = []  # 一次性触发的 trigger_task 队列（如 GameStartupTask）
```

**改动 B —** **`enqueue_onetime_task`（行 463-472）**：在 onetime\_tasks 分支后追加 trigger\_tasks 分支。**关键：排除** **`ManagedTriggerTask`**，避免改变 NetworkErrorHandler 手动开关的行为（ManagedTriggerTask 仍靠 managed scheduler 周期选中，不入 fire\_once 队列）：

```python
def enqueue_onetime_task(self, task):
    if task in self.onetime_tasks:
        with self.lock:
            if task not in self.onetime_task_queue:
                self.onetime_task_queue.append(task)
                logger.info(f'queued onetime_task {task.name}')
        self._wake_executor()
        return True
    # 普通 TriggerTask（非 ManagedTriggerTask）走一次性触发队列
    # （如 GameStartupTask 靠 enable_after_start 启动时执行一次）
    from ok.trigger.base import ManagedTriggerTask
    if task in self.trigger_tasks and not isinstance(task, ManagedTriggerTask):
        with self.lock:
            if task not in self.trigger_fire_once_queue:
                self.trigger_fire_once_queue.append(task)
                logger.info(f'queued fire_once trigger_task {task.name}')
        self._wake_executor()
        return True
    self._wake_executor()
    return False
```

（局部 import `ManagedTriggerTask` 与文件内已有的局部 import `TriggerScheduler`（行 117）风格一致；`ok.trigger.base` 仅依赖 `ok.task.task`，无循环导入风险。）

**改动 C —** **`remove_onetime_task`（行 474-484）**：去掉 `if task not in self.onetime_tasks: return False` 早返回，同步清理 fire\_once 队列：

```python
def remove_onetime_task(self, task):
    removed = False
    with self.lock:
        while task in self.onetime_task_queue:
            self.onetime_task_queue.remove(task)
            removed = True
        while task in self.trigger_fire_once_queue:
            self.trigger_fire_once_queue.remove(task)
            removed = True
    if removed:
        self._wake_executor()
    return removed
```

**改动 D —** **`next_task`（行 501-518）**：在 `onetime_task_queue` 消费之后、`onetime_tasks` 遍历之前，插入 fire\_once 队列消费。返回 `is_trigger_task=False`，使其走 onetime 执行路径（run 后调 `disable()`，自动跑一次后禁用）：

```python
with self.lock:
    while self.onetime_task_queue:
        onetime_task = self.onetime_task_queue.pop(0)
        if onetime_task.enabled:
            logger.info(f'get queued onetime_task {onetime_task.name}')
            return onetime_task, True, False
    while self.trigger_fire_once_queue:
        fire_once = self.trigger_fire_once_queue.pop(0)
        if fire_once.enabled:
            logger.info(f'get fire_once trigger_task {fire_once.name}')
            return fire_once, True, False  # is_trigger_task=False → onetime 路径，run 后 disable()
```

**改动 E —** **`destroy`（行 667-681 附近）**：在清空 `trigger_tasks` 前补 `self.trigger_fire_once_queue = []`。

### 2. `ok/automation/game_startup_task.py` — 改继承 TriggerTask

* **行 27 import**：`from ok.task.task import BaseTask` → `from ok.task.task import TriggerTask`

* **行 37 类声明**：`class GameStartupTask(TriggerTask):`

* **行 46-51** **`__init__`**：保留 `enable_after_start=True`、`visible=True`；无需再设 `_enabled`（`TriggerTask.__init__` 已设 `default_config['_enabled']=False`、`trigger_interval=0`）

* **行 53-54** **`on_create`**：整段删除，让 `TriggerTask.on_create` 接管（`self._enabled = self.config.get('_enabled', False)` → 默认 False，靠 `enable_after_start` 入队触发）

* **新增** **`should_trigger`**（紧邻 `run()`，防御 legacy 路径 + 明确非周期触发器）：

```python
def should_trigger(self):
    # one-shot 任务，靠 enable_after_start + fire_once 队列触发；
    # 永不应被 managed scheduler 或 legacy round-robin 自动拾取。
    return False
```

* **`run()`** **方法完全不动**（行 228-267），返回 True/False 的语义与 onetime 路径兼容

### 3. `start_gui.py` — 把 GameStartupTask 移到 trigger\_tasks（行 27-35）

```python
"onetime_tasks": [
    ["ok_tasks.HomeRedDotTask", "HomeRedDotTask"],
    ["ok_tasks.AlchemyDispatchTask", "AlchemyDispatchTask"],
    ["ok_tasks.RecruitTask", "RecruitTask"],
],
"trigger_tasks": [
    ["ok.automation.game_startup_task", "GameStartupTask"],
    ["ok_tasks.NetworkErrorHandler", "NetworkErrorHandler"],
],
```

### 4. 三个日常任务 — `on_create` 强制 `_enabled=False`

**为什么必须强制而非改默认值**：`Config.verify_config` 对 `default` 中每个 key，若持久化值类型与 default 相同则保留持久化值。`True` 与 `False` 同为 bool，所以改 `DEFAULT_CONFIG` 无法清除已持久化的 `_enabled: true`（`configs/HomeRedDotTask.json` 等已确认存了 true）。必须 `on_create` 直接赋 False。

**同样模式套用三处**：

* `ok_tasks/HomeRedDotTask.py` 行 95-98

* `ok_tasks/AlchemyDispatchTask.py` 行 174-176

* `ok_tasks/RecruitTask.py` 行 87-89

```python
def on_create(self):
    # 永不自动启动；用户必须点"批量启动"或单个"Start"按钮才会执行。
    # 忽略 config 中可能残留的 _enabled=True（旧版本持久化的值）。
    self._enabled = False
    self.follow_batch_start = self.config.get("follow_batch_start", True)
    # ...保留原有的模板加载等后续语句...
```

（`DEFAULT_CONFIG["_enabled"]` 可保留不改，对行为无影响，减少 diff。）

***

## 执行流程验证（启动 GUI 后）

1. `TaskManager.init_tasks` 实例化：GameStartupTask 在 trigger\_tasks、三个日常任务在 onetime\_tasks
2. `after_init → on_create`：GameStartupTask（TriggerTask.on\_create）`_enabled=False`；三个日常任务强制 `_enabled=False`；NetworkErrorHandler `_enabled=True`（持久化值）
3. `MainWindow.showEvent` → `auto_start_on_gui=True` → `start_controller.start()` → `do_start(task=None)`
4. `do_start`：`standalone=False` → 遍历 `get_all_tasks()`，仅 GameStartupTask `enable_after_start=True` → `_mark_task_enabled` → `_enabled=True` + `enqueue_onetime_task` → 进入 `trigger_fire_once_queue`
5. `executor.start()` → `next_task`：onetime\_task\_queue 空 → **trigger\_fire\_once\_queue 有 GameStartupTask** → pop → `enabled=True` → 返回 `(GameStartupTask, True, False)`
6. `execute()` onetime 分支：`task.run()`（启动游戏流程）→ `task.disable()` → `_enabled=False`、`config['_enabled']=False` 持久化、`communicate.task_done`
7. `next_task` 再次：三队列空、onetime\_tasks 全 disabled → trigger\_tasks managed 路径：`scheduler.select` 跳过 GameStartupTask（非 ManagedTriggerTask + 已 disabled），选中 NetworkErrorHandler → 周期跑
8. **最终**：GameStartupTask 跑一次禁用，NetworkErrorHandler 周期运行，日常任务全禁用 ✓

**手动批量启动日常**：`start_onetime_all` → `do_start(first_task)`（first\_task.standalone\_start=True → 不触发 enable\_after\_start，不重跑 GameStartupTask）→ 其余任务 `_mark_task_enabled` 依次入队执行 ✓

**触发器 tab 手动开 GameStartupTask 开关**：`TriggerTask.enable` → enqueue 进 fire\_once 队列 → run → disable → 开关自动复位 OFF；下次 GUI 启动 do\_start 再次 enqueue，稳定触发一次 ✓

***

## 风险与缓解

| 风险                                                      | 缓解                                                                                                        |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `enqueue_onetime_task` 行为变更影响 NetworkErrorHandler 手动开关  | fire\_once 分支已 `排除 ManagedTriggerTask`，NetworkErrorHandler.enable 仍只设状态、不入队，靠 managed scheduler 周期选中，行为不变 |
| 旧用户持久化 `_enabled: true` 残留                              | 日常任务 on\_create 强制 False 后被忽略，残留无害（不影响内存状态）                                                               |
| legacy 模式（OK\_TRIGGER\_MANAGED=0）下 GameStartupTask 不被触发 | `should_trigger` 返回 False 防御；fire\_once 队列优先于 legacy 路径，由 do\_start 入队驱动                                  |
| GameStartupTask 跑到一半用户点开关 OFF                           | `disable()` → `remove_onetime_task` 清 fire\_once 队列；run() 在执行线程跑完，`disable()` 幂等，可接受                      |
| 改动 TaskExecutor 核心调度有回归风险                               | 改动是增量式（新增并行队列，不动现有 onetime\_task\_queue 逻辑）；建议跑现有队列相关测试                                                   |

***

## 验证方法

1. **语法检查**：`& $py -m py_compile ok\task\TaskExecutor.py ok\automation\game_startup_task.py ok_tasks\HomeRedDotTask.py ok_tasks\AlchemyDispatchTask.py ok_tasks\RecruitTask.py start_gui.py`（`$py` 为 `.venv\Scripts\python.exe`）
2. **现有测试**：`& $py -m pytest tests/test_task_executor_queue.py -q`（若存在）；以及 `& $py -m pytest -q` 全量回归
3. **启动 GUI 端到端验证**：

   * `& $py start_gui.py`

   * 确认"触发器"tab 出现 GameStartupTask 卡片

   * 确认启动后 GameStartupTask 自动执行一次（看日志 `queued fire_once trigger_task GameStartup` + 启动流程日志），跑完开关复位

   * 确认 NetworkErrorHandler 周期运行（触发器状态显示）

   * 确认"周常日常"tab 三个日常任务开关默认 OFF，启动时未自动执行

   * 手动点"批量启动"，确认三个日常任务依次执行，且不会重跑 GameStartupTask

   * 手动在触发器 tab 开 GameStartupTask 开关，确认能再触发一次并自动复位
4. **legacy 路径验证**（可选）：设环境变量 `OK_TRIGGER_MANAGED=0` 启动，确认 GameStartupTask 仍能由 fire\_once 队列触发一次、should\_trigger 不报错

