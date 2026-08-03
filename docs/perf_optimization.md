# GUI 卡顿 + 模拟器卡顿 性能优化

本次优化目标：默认启动模式下显著降低 CPU 占用与卡顿，优先保证稳定性与可回退。

## 优化概览

| 维度 | 旧行为 | 新行为 |
|------|--------|--------|
| debug 日志 | 默认开启 | 默认关闭，`OK_DEBUG=1` 开启 |
| 抓帧链路 | ADB screencap (~300ms/帧) 优先 | NEMU IPC 优先，失败自动降级 ADB |
| 主循环 | 无最小间隔，可能 busy-loop | 最小帧间隔 50ms（可配置） |
| GUI 日志 | 每条日志立即 emit + 整窗重绘 | 批处理 flush（150ms / 最多 200 条） |
| ADB 重连 | 失败立即重试，可能抖动 | 指数退避 1s→2s→4s→8s（上限） |
| 设备刷新 | 每次调用都执行 | 2s 冷却（可配置） |
| kill adb | 失败即遍历进程 kill | 10s 限频（可配置） |

---

## A. 快速止卡

### A1. debug 开关（start_gui.py）

`config["debug"]` 默认改为 `False`，通过环境变量 `OK_DEBUG` 控制：

```python
config = {
    "debug": os.environ.get("OK_DEBUG", "0") in ("1", "true", "yes", "on"),
    ...
}
```

- **默认**：`debug=False`，日志级别 INFO，不写 debug 日志，降低日志与渲染负担。
- **开发调试**：`set OK_DEBUG=1` 后启动，恢复旧行为。
- **回退**：设置 `OK_DEBUG=1` 即可。

### A2. 采集方式优先级（DeviceManager.do_start）

抓帧链路优先级（高 → 低）：

1. **NEMU IPC**（MuMuPlayer12 国内版，`path` 不含 `MuMuPlayerGlobal`）
2. **ADB screencap**（最终 fallback）

#### 自动升级逻辑

- 仅当 `OK_CAPTURE_PREFER_FAST=1`（默认）且设备为 MuMuPlayer12 时，自动探测 NEMU IPC。
- 探测使用 `init_nemu()`（上限 ≤0.5-1s），不使用 `get_frame()`（重试链 ~10s）。
- 探测成功 → 切换 NEMU IPC，复用连接。
- 探测失败 → 关闭临时方法，降级 ADB，**不崩溃**。
- 非 mumu 设备（LDPlayer/BlueStacks 等）跳过探测，直接 ADB，零启动延迟。

#### 可观测日志

每次启动打印最终采用的采集方法 + 原因：

```
capture method selected: NemuIpc (auto-upgrade from 'adb')
capture method selected: ADB (nemu probe failed, fallback)
capture method selected: NemuIpc (reused)
use adb capture {...}
```

#### 回退

```powershell
$env:OK_CAPTURE_PREFER_FAST = "0"  # 关闭自动升级，回退到旧逻辑（显式 ipc 用 nemu，其余 adb）
```

显式选择 ipc（GUI 下拉框）仍保留原行为，不探测、不降级。

### A3. 循环节流（TaskExecutor.next_frame）

主循环加入最小帧间隔控制，避免 busy-loop：

```python
# OK_MIN_FRAME_INTERVAL_MS 默认 50ms（~20 FPS），0 禁用
self._min_frame_interval = max(0.0, int(os.getenv('OK_MIN_FRAME_INTERVAL_MS', '50')) / 1000.0)
```

- 距上一帧不足最小间隔时，等待差额（受 `time_out` 上限约束）。
- 等待可被 `wake` 事件打断，不阻塞任务响应。
- ADB（~300ms/帧）不受影响（帧本身已慢于最小间隔）。

#### 回退

```powershell
$env:OK_MIN_FRAME_INTERVAL_MS = "0"  # 禁用节流
```

---

## B. 结构化优化

### B4. 日志/UI 刷新解耦

#### BatchedCommunicateHandler（ok/util/logger.py）

worker 线程只往线程安全 deque 追加日志（不触 UI）；守护线程 `LogBatcher` 按固定间隔批量 flush，限制 `communicate.log` 信号注入 GUI 事件队列的速率。

- **默认**：flush 间隔 150ms，单次最多 200 条。
- **防突发**：deque 超过 4 倍上限时丢弃最旧；单次 flush 超量保留最新。
- **回退**：`OK_LOG_BATCH_DISABLE=1` 改用直发 `CommunicateHandler`（旧行为）。

可调参数：

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `OK_LOG_BATCH_DISABLE` | `0` | `=1` 禁用批处理，回退直发 |
| `OK_LOG_FLUSH_MS` | `150` | flush 间隔（毫秒） |
| `OK_LOG_BATCH_MAX` | `200` | 单次 flush 最大条数 |

#### OverlayWidget 日志批重绘（ok/gui/debug/OverlayWidget.py）

`add_log` 只置脏标志 `_log_dirty`，由 `QTimer` 每 150ms 合并一次 `update()`，避免高频日志每条触发整窗 `paintEvent`。

### B5. ADB 设备刷新/重连降频（DeviceManager）

#### adb_connect 指数退避

连接失败按 1s → 2s → 4s → 8s 退避（上限），退避窗口内跳过重连，避免短时重复 `adb connect` 抖动。成功后清空退避状态。

```python
self._adb_backoff_initial = ...  # OK_ADB_CONNECT_BACKOFF_MS, 默认 1000
self._adb_backoff_max = ...      # OK_ADB_CONNECT_BACKOFF_MAX_MS, 默认 8000
```

#### do_refresh 冷却

`OK_REFRESH_COOLDOWN_MS`（默认 2000ms）内重复调用直接跳过，避免短时重复 `adb list`。

#### try_kill_adb 限频

`OK_ADB_KILL_COOLDOWN_MS`（默认 10000ms）内不重复 kill adb server，避免短时多次遍历进程。

#### 失败路径日志

所有失败路径记录明确日志：

```
adb_connect {addr} failure backoff {backoff:.1f}s
adb_connect {addr} skipped (backoff {remaining:.1f}s)
try kill adb skipped (cooldown {cooldown:.1f}s)
do_refresh skipped (cooldown {cooldown:.1f}s)
try kill adb server {e}
adb connect error {addr} {e}
```

#### 回退

| 环境变量 | 默认 | 回退值（禁用节流） |
|----------|------|---------------------|
| `OK_REFRESH_COOLDOWN_MS` | `2000` | `0` |
| `OK_ADB_CONNECT_BACKOFF_MS` | `1000` | `0` |
| `OK_ADB_CONNECT_BACKOFF_MAX_MS` | `8000` | `0` |
| `OK_ADB_KILL_COOLDOWN_MS` | `10000` | `0` |

---

## 环境变量汇总

| 环境变量 | 默认 | 说明 | 回退值 |
|----------|------|------|--------|
| `OK_DEBUG` | `0` | `=1` 开启 debug 日志 | — |
| `OK_CAPTURE_PREFER_FAST` | `1` | `=0` 关闭 NEMU 自动升级 | `0` |
| `OK_MIN_FRAME_INTERVAL_MS` | `50` | 主循环最小帧间隔（ms） | `0` 禁用 |
| `OK_LOG_BATCH_DISABLE` | `0` | `=1` 禁用日志批处理 | `1` |
| `OK_LOG_FLUSH_MS` | `150` | 日志 flush 间隔（ms） | — |
| `OK_LOG_BATCH_MAX` | `200` | 单次 flush 最大条数 | — |
| `OK_REFRESH_COOLDOWN_MS` | `2000` | 设备刷新冷却（ms） | `0` 禁用 |
| `OK_ADB_CONNECT_BACKOFF_MS` | `1000` | adb 重连初始退避（ms） | `0` 禁用 |
| `OK_ADB_CONNECT_BACKOFF_MAX_MS` | `8000` | adb 重连退避上限（ms） | `0` 禁用 |
| `OK_ADB_KILL_COOLDOWN_MS` | `10000` | kill adb 限频（ms） | `0` 禁用 |

---

## 一键回退到旧行为

如出现兼容性问题，设置以下环境变量可完全回退到优化前行为：

```powershell
$env:OK_DEBUG = "1"
$env:OK_CAPTURE_PREFER_FAST = "0"
$env:OK_MIN_FRAME_INTERVAL_MS = "0"
$env:OK_LOG_BATCH_DISABLE = "1"
$env:OK_REFRESH_COOLDOWN_MS = "0"
$env:OK_ADB_CONNECT_BACKOFF_MS = "0"
$env:OK_ADB_KILL_COOLDOWN_MS = "0"
```

---

## 测试

新增测试覆盖：

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_capture_fallback.py` | NEMU 探测成功/失败/复用、自动升级、非 mumu 跳过、`PREFER_FAST=0` 回退 |
| `tests/test_perf_throttle.py` | adb 退避序列 1→2→4→8 封顶、退避窗口跳过、do_refresh 冷却、kill adb 限频 |
| `tests/test_frame_throttle.py` | 帧间隔节流等待、禁用时不等待、首帧不等待、timeout 上限约束 |
| `tests/test_logger.py` | 批处理 flush、max_batch 截断、默认用批处理 handler、`DISABLE=1` 回退直发 |

运行：

```powershell
$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }
& $py -m pytest tests/test_capture_fallback.py tests/test_perf_throttle.py tests/test_frame_throttle.py tests/test_logger.py -v
```

---

## 改动文件清单

| 文件 | 改动 |
|------|------|
| `start_gui.py` | A1: debug 默认 False + `OK_DEBUG` 开关 |
| `ok/device/DeviceManager.py` | A2: NEMU 优先 + 降级 ADB + 可观测日志；B5: adb 退避 + refresh 冷却 + kill 限频 |
| `ok/task/TaskExecutor.py` | A3: 主循环最小帧间隔节流 |
| `ok/util/logger.py` | B4: `BatchedCommunicateHandler` 批处理日志 |
| `ok/gui/debug/OverlayWidget.py` | B4: 日志批重绘（QTimer 合并 update） |
| `tests/test_capture_fallback.py` | 新增：采集降级测试 |
| `tests/test_perf_throttle.py` | 新增：adb/refresh/kill 节流测试 |
| `tests/test_frame_throttle.py` | 新增：帧节流测试 |
| `tests/test_logger.py` | 新增：批处理日志测试 |
| `docs/perf_optimization.md` | 新增：本文档 |
