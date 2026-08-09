# 触发器统一调度架构（Unified Trigger Framework）

本文档说明 `ok/trigger/` 包引入的触发器体系架构、如何新增触发器、参数调优指南以及回滚方案。

## 1. 背景与目标

旧实现中，每个触发器各自在 `TaskExecutor.next_task` 的轮询循环里通过 `should_trigger()` 判断是否运行，存在以下共性问题：

- `TriggerTask` 基类默认 `trigger_interval = 0`，语义为“总是触发”，任何忘记覆盖该值的触发器都会 **busy-loop**。
- 轮询索引回绕时会调用 `trigger_sleep()`（默认 3000ms），**阻塞主执行线程**，拖慢一次性任务响应。
- 检测与动作耦合，无统一节流 / 冷却 / 去重 / 预算 / 超时。
- 无可观测指标，无法定位卡顿来源。
- 触发器 `run()` 抛异常会被执行器 **永久禁用**该任务。

新架构的目标：保留触发可靠性，把整个触发器系统改造为 **低开销、可扩展、可观测** 的“统一调度中心 + 分级触发策略”。

## 2. 架构总览

```
 TaskExecutor.next_task()
        │
        ├─ if enable_managed_trigger (feature flag)
        │      └─ TriggerScheduler.select(tasks)   ← 单点调度
        │             │
        │             ├─ 过滤 enabled 的 ManagedTriggerTask
        │             ├─ TriggerThrottle.cool_to_run（min_interval / cooldown / 分类 / 全局）
        │             └─ 按 priority 降序选取
        │
        └─ executor 调用 task.run()  （ManagedTriggerTask.run，基类提供）
                │
                ├─ check(context)  → TriggerDecision（轻量检测 + fingerprint + 成本估算）
                ├─ TriggerThrottle.admit（dedup + 全局预算）
                ├─ handle(context) → TriggerResult（动作 + 资源计数）
                └─ TriggerThrottle.record_run / record_cost + TriggerMetrics.record_run
```

### 模块

| 模块 | 职责 |
|------|------|
| `ok/trigger/categories.py` | `TriggerCategory` 枚举：network / battle_scene / ui_popup / task_flow / recovery |
| `ok/trigger/decision.py` | `TriggerDecision`（check 产出）、`TriggerResult`（handle 产出） |
| `ok/trigger/config.py` | `TriggerFrameworkConfig`：加载 `configs/trigger_config.json`，合并 DEFAULTS < 分类默认 < 触发器覆盖 < 类声明；含回滚开关 |
| `ok/trigger/throttle.py` | `TriggerThrottle`：单触发器冷却 / 分类冷却 / 全局最小间隔 / 滚动窗口预算 / 去重 |
| `ok/trigger/metrics.py` | `TriggerMetrics`：每触发器 / 每分类 / 全局计数 + 周期性摘要日志 |
| `ok/trigger/base.py` | `ManagedTriggerTask`：基类，提供 check/handle 契约、安全 `run()`、watchdog 超时 |
| `ok/trigger/scheduler.py` | `TriggerScheduler`：拥有共享 throttle/metrics/config，按优先级选取下一个触发器 |

## 3. 分级触发策略

1. **统一调度**：所有受管触发器由 `TriggerScheduler.select` 单点选取，禁止各自死循环。
2. **优先级**：`priority` 越高越先被选取（network 80 > recovery 70 > battle_scene 60 > ui_popup 50 > task_flow 40）。
3. **冷却兜底**：`cool_to_run` 在选取阶段即按 `min_interval` / `cooldown_seconds` / 分类冷却 / 全局最小间隔过滤——**冷却期间整段跳过 check**，避免无效的截图/匹配开销。
4. **去重**：`check()` 产出 `fingerprint`，相同 fingerprint 在 `dedup_window_seconds` 内只处理一次。
5. **预算控制**：滚动窗口内限制总运行次数 / 模板匹配次数（`budget_*_per_window`）。
6. **超时**：`check_timeout_seconds` / `handle_timeout_seconds` 通过 watchdog 线程限制单次耗时，防止主循环阻塞。
7. **重试上限**：`max_retry` 由触发器在 `handle()` 内部受限重试，并由预算 / 超时兜底。

## 4. 如何新增一个触发器

1. 在 `ok_tasks/` 新建文件，继承 `ManagedTriggerTask`：

   ```python
   from ok.trigger.base import ManagedTriggerTask, TriggerContext
   from ok.trigger.categories import TriggerCategory
   from ok.trigger.decision import TriggerDecision, TriggerResult

   class MyPopupHandler(ManagedTriggerTask):
       category = TriggerCategory.UI_POPUP
       priority = 55
       trigger_mode = "polling"

       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.name = "通用弹窗处理"
           self.description = "关闭公告/签到等通用弹窗"

       def check(self, context: TriggerContext) -> TriggerDecision:
           frame = context.frame
           # 轻量检测……
           if detected:
               return TriggerDecision(should_handle=True, fingerprint="popup_a",
                                      cost_estimate=0.4)
           return TriggerDecision(should_handle=False)

       def handle(self, context: TriggerContext) -> TriggerResult:
           # 动作……
           return TriggerResult.ok(match_calls=context.match_calls)
   ```

2. 在 `start_gui.py` 的 `trigger_tasks` 列表注册：`["ok_tasks.MyPopupHandler", "MyPopupHandler"]`。
3. （可选）在 `configs/trigger_config.json` 的 `trigger_overrides` 中按 `name` 覆盖参数。
4. 不需要实现 `run()`、`should_trigger()`——基类已提供。

**契约要点**：
- `check()` 必须尽量轻；重操作放 `handle()`。
- `check()` 通过 `context.match_calls` / `context.ocr_calls` 累加资源计数（用于预算/指标）。
- `handle()` 可读取 `context.deadline` / `context.timed_out()` 做协作式超时。
- 不要在 `check`/`handle` 里自己 `while True` 死循环；调度器负责节奏。

## 5. 参数调优指南

集中配置分两层：

- **包内默认** `ok/trigger/default_trigger_config.json`：随仓库跟踪，提供开箱即用的安全默认值。
- **部署覆盖** `configs/trigger_config.json`：被 `.gitignore` 忽略，机器本地覆盖（按子键合并，不会清空整张默认表）。

合并顺序：包内默认 < `configs/trigger_config.json` 覆盖 < 类声明属性。环境变量 `OK_TRIGGER_MANAGED` 优先级最高（热回滚）。

| 参数 | 含义 | 调优建议 |
|------|------|----------|
| `global_min_interval` | 任意两次触发器运行的最小间隔 | 卡顿严重可调大到 1.5~2.0；过大会降低异常响应 |
| `<trigger>.min_interval` | 单触发器最小检测间隔 | 网络类 2~3s；UI 弹窗类 8~15s |
| `<trigger>.cooldown_seconds` | 单触发器处理后冷却 | 处理完一个弹窗后给游戏恢复时间，5~15s |
| `dedup_window_seconds` | 相同 fingerprint 去重窗口 | 略大于单次处理耗时，避免抖动重处理 |
| `budget_ocr_per_window` | 60s 内 OCR 总次数上限 | OCR 最重，建议 30~60 |
| `budget_match_per_window` | 60s 内模板匹配总次数上限 | 视触发器数量，建议 200~400 |
| `budget_runs_per_window` | 60s 内总运行次数上限 | 安全阀，100~200 |
| `check_timeout_seconds` / `handle_timeout_seconds` | 单次超时 | check 2~3s，handle 6~10s |
| `metrics_log_interval_seconds` | 指标摘要日志间隔 | 生产 60s；调试 10~30s |

调优流程：先看 `TriggerMetrics` 摘要日志里的 `cd/dup/bud` 命中数与 `ocr/match` 计数，再针对性放宽或收紧。

## 6. 可观测性

`TriggerMetrics` 每隔 `metrics_log_interval_seconds` 输出一次结构化摘要：

```
trigger metrics summary:
  [网络错误处理] sel=20 chk=20 hnd=2 ok=2 fail=0 tmo=0 cd=18 dup=0 bud=0 ocr=2 match=32 shot=0 retries=0 sec=1.3
```

字段含义：`sel`=被调度选中 / `chk`=check 执行 / `hnd`=handle 执行 / `ok`=成功 / `fail`=失败 / `tmo`=超时 / `cd`=冷却命中 / `dup`=去重命中 / `bud`=预算命中 / `ocr`/`match`/`shot`=资源调用 / `retries`=重试 / `sec`=handle 累计耗时。

执行器关闭时（`destroy`）会强制输出一次最终摘要。

## 7. 回滚方案

新调度器受 feature flag 控制，**无需改代码即可热切换**：

- 环境变量：`OK_TRIGGER_MANAGED=0`（或 `false`/`no`/`off`）→ 立即回退到旧轮询路径。
- 配置文件：`configs/trigger_config.json` 中 `"enable_managed_trigger": false`。

回滚后：
- `next_task` 恢复旧的 round-robin + `trigger_sleep()` 行为。
- 受管触发器仍以 `ManagedTriggerTask.run()` 运行（保留单触发器的冷却/去重/超时/指标守卫），但**不再有中央优先级 / 分类冷却 / 全局预算**。
- 受管触发器的 `trigger_interval` 已与 `min_interval` 同步，旧路径节奏一致。

> 注意：`trigger_sleep()`（默认 3000ms）在回滚后会重新阻塞主线程，仅作降级用途，不建议长期使用。

## 8. 改造前后对比

| 维度 | 旧实现 | 新实现（managed） |
|------|--------|-------------------|
| 调度 | 各触发器 round-robin + `trigger_sleep(3s)` 阻塞 | 单点 `TriggerScheduler.select`，按优先级 + 冷却 |
| 冷却期检测 | 仍每 2s 跑全量模板匹配（冷却判断在检测之后） | 冷却期整段跳过 check，**0 次匹配** |
| 模板匹配尺度 | 9 尺度 × N 模板 | 4 尺度 × N 模板（-55% 调用） |
| 模板匹配频率 | 每个周期执行模板匹配 | 由 `min_interval`/`cooldown`/预算统一控制 |
| 超时 | 无 | check/handle watchdog 超时 |
| 异常 | `run()` 抛错 → 任务被永久禁用 | `run()` 永不抛出，软失败计入指标 |
| 可观测 | 无 | 每触发器/分类/全局计数 + 周期摘要 |
| 节流 | 仅 `trigger_interval` | min_interval + cooldown + 分类冷却 + 全局预算 + dedup |

### 预期收益（单触发器 NetworkErrorHandler，空闲态）

- 每分钟扫描次数：`trigger_sleep(3s)` 周期 ≈ 20 次/min → managed `min_interval=3s` 精确等待 ≈ 20 次/min（频率相近，但**冷却期不再做匹配**）。
- 每分钟模板匹配调用：旧 9 尺度 × 2 模板 × 20 ≈ **360 次/min** → 新 4 尺度 × 2 模板 × 20 ≈ **160 次/min**（-55%）；处理弹窗后的 5s 冷却期内进一步降至 **0 次**。
- 每分钟模板匹配调用：旧 9 尺度 × 2 模板 × 20 ≈ **360 次/min** → 新 4 尺度 × 2 模板 × 20 ≈ **160 次/min**（-55%）。
- 主线程阻塞：旧 `trigger_sleep` 每 full-cycle 阻塞 3s → managed 用 `_wait_for_activity(min_interval)` 精确等待，**可被一次性任务唤醒**。

> 实际数据请在目标机器上运行后查看 `TriggerMetrics` 摘要日志，对比 `match/sec` 字段。
