# 模拟器游戏任务自动化使用指南

## 概述

`ok-script` 自动化模块提供了基于状态机的游戏任务编排能力，支持：

- **状态机驱动**：`INIT → NAVIGATE → EXECUTE → VERIFY → DONE/RETRY/FAIL` 完整生命周期
- **配置驱动**：通过 YAML/JSON 定义任务流程，无需硬编码坐标
- **重试与超时**：每步骤独立的重试次数和超时控制
- **检查点恢复**：支持中断后从最近检查点恢复
- **结构化日志**：完整的执行日志和报告生成
- **节流与防检测**：内置随机延时，降低误操作风险

## 快速开始

### 1. 基本用法

```python
from ok.automation import GameTask, ConfigLoader, AutomationConfig, StepConfig

# 方式一：从配置文件加载
loader = ConfigLoader()
config = loader.load("examples/configs/battle_task.yaml")

# 方式二：代码构建配置
config = AutomationConfig(
    task_name="my_task",
    max_retries=3,
    global_timeout=300,
    steps=[
        StepConfig(
            name="点击开始按钮",
            type="click_feature",
            feature_name="start_button",
            threshold=0.85,
            timeout=10,
            retry_count=3,
        ),
        StepConfig(
            name="等待加载完成",
            type="wait_feature",
            feature_name="game_loaded",
            timeout=30,
        ),
    ],
)

# 创建并运行任务
class MyTask(GameTask):
    """自定义任务类，可扩展自定义步骤"""
    pass

# recorder 是 ok-script 的 BaseTask 实例
task = MyTask(config, recorder=base_task)
success = task.run()

if success:
    print("任务执行成功!")
    # 保存执行报告
    task.save_execution_report()
else:
    print(f"任务失败: {task.state_machine.get_execution_summary()}")
```

### 2. 配置文件格式

#### YAML 配置

```yaml
task_name: "example_task"
description: "示例任务"
global_timeout: 300
max_retries: 3
throttle_ms: 100
random_delay: 0.1

steps:
  - name: "步骤一"
    type: "click_feature"
    feature_name: "button_a"
    threshold: 0.85
    timeout: 10
    retry_count: 2
    post_delay: 0.5
```

#### JSON 配置

```json
{
  "task_name": "example_task",
  "max_retries": 3,
  "steps": [
    {
      "name": "步骤一",
      "type": "click_feature",
      "feature_name": "button_a",
      "threshold": 0.85,
      "timeout": 10
    }
  ]
}
```

### 3. 步骤类型

| 类型 | 说明 | 必要参数 |
|------|------|----------|
| `find_feature` | 查找特征 | `feature_name` |
| `click_feature` | 点击特征 | `feature_name` |
| `wait_feature` | 等待特征出现 | `feature_name` |
| `ocr` | OCR 文字识别 | `x`, `y`, `text` |
| `click_ocr` | 点击 OCR 文字 | `x`, `y`, `text` |
| `click_coordinate` | 点击坐标 | `x`, `y` |
| `swipe` | 滑动 | `x`, `y`, `to_x`, `to_y` |
| `send_key` | 发送按键 | `key` |
| `wait` | 等待 | `params.duration` |
| `input_text` | 输入文字 | `text` |
| `custom` | 自定义步骤 | `custom_func_name` |

### 4. 坐标系统

- **相对坐标**：`0.0 ~ 1.0`（比例），自动转换为像素
- **绝对坐标**：像素值（整数）

```python
# 相对坐标 - 屏幕中心
StepConfig(x=0.5, y=0.5)

# 绝对坐标 - 像素
StepConfig(x=500, y=300)
```

### 5. 识别模式

```python
StepConfig(
    type="click_feature",
    feature_name="button",
    recognition_mode="template",  # 模板匹配
    threshold=0.85,               # 置信度阈值
)
```

| 模式 | 说明 |
|------|------|
| `template` | 模板匹配（默认） |
| `ocr` | OCR 文字识别 |
| `color` | 颜色匹配 |
| `any` | 任意模式 |

## 高级功能

### 1. 自定义步骤

```python
class MyTask(GameTask):
    def _execute_custom(self, step: StepConfig) -> bool:
        """执行自定义步骤"""
        action = step.params.get("action")
        
        if action == "collect_item":
            return self.collect_item(step.params.get("item_id"))
        
        return False
    
    def collect_item(self, item_id: str) -> bool:
        """收集物品"""
        # 自定义逻辑
        item = self.recorder.find_one(f"item_{item_id}")
        if item:
            self.recorder.click_box(item)
            return True
        return False

# 在配置中使用
StepConfig(
    name="收集物品",
    type="custom",
    custom_func_name="collect_item",
    params={"item_id": "rare_gem"},
)
```

### 2. 状态机控制

```python
task = GameTask(config)

# 监听状态转换
def on_state_change(old_state, new_state):
    print(f"状态变化: {old_state} -> {new_state}")

task.state_machine.add_transition_callback(on_state_change)

# 检查当前状态
print(task.state_machine.state)      # TaskState.EXECUTE
print(task.state_machine.status)     # TaskStatus.RUNNING
print(task.state_machine.is_terminal) # False

# 恢复检查点
if not task.state_machine.is_terminal:
    task.state_machine.recover_from_checkpoint()
```

### 3. 进度查询

```python
# 查询执行进度
progress = task.get_progress()
# {
#     "task_name": "my_task",
#     "current_state": "execute",
#     "current_status": "running",
#     "current_step_index": 3,
#     "total_steps": 10,
#     "completed_steps": 2,
#     "progress_percent": 20.0,
#     "elapsed_time": 45.2,
#     "retries": 1,
#     "is_running": True
# }
```

### 4. 报告生成

```python
# 自动保存执行报告
report_path = task.save_execution_report()

# 或使用 TaskReporter
from ok.automation import TaskReporter

reporter = TaskReporter("my_task")
reporter.log_step_start("step1", "click_feature")
reporter.log_step_end("step1", True, duration_ms=150.5)

report = reporter.generate_report(success=True)
reporter.save_report(report)

# 获取统计
stats = reporter.get_statistics()
# {
#     "total_steps": 10,
#     "success_rate": 80.0,
#     "failed_steps": 2,
#     "total_retries": 5,
#     "average_step_duration_ms": 125.5
# }
```

### 5. 节流与防检测

```python
config = AutomationConfig(
    throttle_ms=100,      # 最小操作间隔 100ms
    random_delay=0.15,     # 随机延时变化 ±15%
)
```

系统会在每次操作后自动添加随机延时，模拟人类操作节奏，降低被检测的风险。

## 完整示例

### 战斗自动化示例

```python
from ok.automation import (
    GameTask, ConfigLoader, AutomationConfig, StepConfig
)

class BattleAutomation(GameTask):
    """战斗自动化任务"""
    
    def __init__(self, config, recorder):
        super().__init__(config, recorder)
        self.battle_count = 0
    
    def run_battle_sequence(self):
        """执行一场战斗"""
        self.battle_count += 1
        
        steps = [
            StepConfig(
                name="开始战斗",
                type="click_feature",
                feature_name="start_battle",
                threshold=0.85,
                timeout=10,
            ),
            StepConfig(
                name="等待战斗加载",
                type="wait_feature",
                feature_name="battle_scene",
                timeout=20,
            ),
            # ... 更多步骤
            StepConfig(
                name="领取奖励",
                type="click_feature",
                feature_name="reward_claim",
                threshold=0.80,
                timeout=5,
            ),
        ]
        
        self.config.steps = steps
        return self.run()


# 使用
config = AutomationConfig(
    task_name="daily_battles",
    max_retries=3,
    global_timeout=600,
    steps=[],  # 动态设置
)

task = BattleAutomation(config, recorder=base_task)

for day in range(3):
    success = task.run_battle_sequence()
    if not success:
        print(f"第 {day + 1} 天战斗失败")
        break

task.save_execution_report()
```

## API 参考

### GameTask 类

```python
class GameTask:
    def __init__(self, config, recorder=None, output_dir=None): ...
    def run(self) -> bool: ...
    def stop(self) -> None: ...
    def get_progress(self) -> Dict[str, Any]: ...
    def save_execution_report(self) -> str: ...
    
    # 步骤执行（可覆盖）
    def _execute_find_feature(self, step) -> bool: ...
    def _execute_click_feature(self, step) -> bool: ...
    def _execute_wait_feature(self, step) -> bool: ...
    def _execute_ocr(self, step) -> bool: ...
    def _execute_click_ocr(self, step) -> bool: ...
    def _execute_click_coordinate(self, step) -> bool: ...
    def _execute_swipe(self, step) -> bool: ...
    def _execute_send_key(self, step) -> bool: ...
    def _execute_wait(self, step) -> bool: ...
    def _execute_input_text(self, step) -> bool: ...
    def _execute_custom(self, step) -> bool: ...
```

### GameStateMachine 类

```python
class GameStateMachine:
    state: TaskState          # 当前状态
    status: TaskStatus        # 执行状态
    retries: int              # 重试次数
    history: List[StateTransition]  # 状态历史
    elapsed_time: float       # 执行耗时
    
    def navigate(reason) -> bool: ...
    def execute(reason) -> bool: ...
    def verify(reason) -> bool: ...
    def complete(reason) -> bool: ...
    def retry(reason) -> bool: ...
    def fail(reason) -> bool: ...
    def reset() -> None: ...
    def add_transition_callback(callback) -> None: ...
    def get_execution_summary() -> Dict: ...
    def recover_from_checkpoint() -> bool: ...
```

### TaskReporter 类

```python
class TaskReporter:
    def log(step_name, action, result, ...) -> LogEntry: ...
    def log_step_start(step_name, step_type) -> None: ...
    def log_step_end(step_name, success, ...) -> StepReport: ...
    def log_step_retry(step_name) -> None: ...
    def log_screenshot(filepath) -> None: ...
    def generate_report(success) -> TaskReport: ...
    def save_report(report, filename) -> str: ...
    def get_statistics() -> Dict: ...
```

## 注意事项

1. **依赖**: 需要 `ok-script` 核心模块，可选 `PyYAML` 用于 YAML 配置
2. **平台**: 主要支持 Windows，Android 模拟器通过 ADB 连接
3. **分辨率**: 推荐使用相对坐标 (0.0-1.0) 实现跨分辨率适配
4. **素材**: 模板匹配需要 COCO 格式的标注数据
5. **安全**: 仅在用户授权环境下使用，不提供任何绕过安全策略的功能

## 常见问题

### Q: 如何处理步骤执行过慢？

```python
# 增加步骤超时时间
StepConfig(
    name="slow_step",
    type="wait_feature",
    timeout=60,  # 延长超时
)
```

### Q: 如何实现并行执行？

当前版本不支持并行步骤。如需并行，可创建多个 GameTask 实例在不同线程运行。

### Q: 如何与 ok-script GUI 集成？

```python
# 在 BaseTask 的 run() 方法中使用
class MyOscriptTask(BaseTask):
    def run(self):
        from ok.automation import GameTask, AutomationConfig, StepConfig
        
        config = AutomationConfig(
            task_name=self.name,
            steps=[...],
        )
        task = GameTask(config, recorder=self)
        task.run()
```
