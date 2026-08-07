# GUI 卡顿 + 模拟器卡顿 联合性能优化

## Context（背景与动机）

启动后 GUI 与模拟器同时卡顿。根因有三：
1. `start_gui.py` 默认 `debug=True`，导致大量 DEBUG 日志 → 每条记录同步写 stdout + 触发 `communicate.log` 信号。
2. 抓帧链路默认走 ADB screencap（~300ms/帧 ≈ 3 FPS），是模拟器侧渲染卡顿的主因；项目已有高性能的 NEMU IPC（mumu12 专用，~10ms/帧）但默认未启用，且无降级保护。
3. 当抓帧变快后，识别主循环 `next_frame` 无最小间隔，存在空转风险；overlay 日志每条记录触发一次整窗重绘；`adb_connect` 失败路径会反复 `try_kill_adb`（遍历进程 + kill adb server）造成抖动。

目标：默认启动模式下显著降低 CPU 与卡顿，优先保证稳定性与可回退。所有新行为均可通过环境变量一键回退到旧逻辑。

---

## 改动清单

### A. 快速止卡

#### A1. `start_gui.py` — debug 默认关闭
- `config["debug"]` 由 `True` 改为读取环境变量：`os.environ.get("OK_DEBUG", "0") in ("1","true","yes","on")`。
- 默认 `False`；开发调试时 `set OK_DEBUG=1` 即可恢复。
- 不影响其它任务配置。

#### A2. 采集方式优先级 + 降级（核心）— `ok/device/DeviceManager.py`
在 `do_start` 的 adb 设备分支（现 608-644 行）引入优先级：**NEMU IPC（mumu12）→ ADB screencap（fallback）**，并用 `init_nemu()` 探测而非 `get_frame()`（探测上限 ≤0.5-1s，避免 ~10s 的 screenshot 重试链）。

新增两个 helper（放在 `use_windows_capture` 之后）：
- `_is_mumu12(emulator)`：镜像 GUI 下拉框的判定（`ok/gui/start/SelectCaptureListView.py:52-54`）—— `emulator.type == Emulator.MuMuPlayer12 and "MuMuPlayerGlobal" not in emulator.path`。非 mumu（LDPlayer/BlueStacks）直接返回 False，**跳过探测、零额外启动延迟**。
- `_try_nemu_capture(emulator, width, height)`：创建 `NemuIpcCaptureMethod` → `update_emulator` → `init_nemu()` 探测；成功返回已连接的 method（连接被后续 `do_get_frame` 复用），失败 `method.close()` 后返回 None，**永不抛异常**。

重构 `do_start` 内层 `else`（仅 ipc/adb 选择段，`capture=='windows'` 分支与 interaction 段不动）：
- `saved_capture = self.config.get('capture')`；`prefer_fast = os.getenv('OK_CAPTURE_PREFER_FAST','1')=='1'`。
- `explicit_ipc = saved_capture == 'ipc'`：显式 ipc → 直接用 nemu，**不探测、不降级**（保留原行为）。
- `auto_upgrade = prefer_fast and _is_mumu12(emulator) and saved_capture in ('adb','','auto')`：自动升级 → 走 `_try_nemu_capture`，失败降级 ADB。
- 已是 `NemuIpcCaptureMethod` 实例时：**仅 `update_emulator`，不重复探测**（避免每次 refresh 多 0.5-1s）。
- 一次性可观测日志：`capture method selected: NemuIpc (auto-upgrade) | ADB (fallback) | NemuIpc (explicit) | NemuIpc (reused)`。
- 降级路径落到原有 `ADBCaptureMethod` 创建逻辑（原代码原样保留作为 fallback 分支）。

**回退**：`OK_CAPTURE_PREFER_FAST=0` → `auto_upgrade=False`；显式 `ipc` 仍走原 nemu 路径，`adb` 走原 ADB 路径，行为与改造前完全一致。

**已知限制（v1 接受，PR 注明）**：运行中 nemu 断开不自动回退 ADB（`next_frame` 维持 1s 重试，与现有显式 ipc 一致）；用户可通过 GUI 下拉框手动切回 ADB。附带收益：ADBInteraction 检测到 `isinstance(capture, NemuIpcCaptureMethod)` 会自动走 nemu IPC 输入（更快），与现有显式 ipc 行为一致。

#### A3. 主循环节流 — `ok/task/TaskExecutor.py`
`next_frame`（262 行）增加最小帧间隔：
- `__init__` 中读 `self._min_frame_interval = int(os.getenv('OK_MIN_FRAME_INTERVAL_MS','50'))/1000`（默认 50ms，0 禁用）。
- 在 `next_frame` 进入捕获循环前：若 `_last_frame_time>0` 且距上一帧不足 `_min_frame_interval`，用 `_wait_for_activity(remaining)` 等待差额（受 `time_out` 上限约束，可被 wake 事件打断）。
- 效果：ADB（300ms/帧）不受影响；NEMU（~10ms/帧）被限速到 ~20 FPS，消除空转与 CPU 峰值。

---

### B. 结构化优化

#### B4. 日志/UI 解耦
**`ok/util/logger.py`**：新增 `BatchedCommunicateHandler`，替换 `config_logger` 中直接 `CommunicateHandler`（可经 `OK_LOG_BATCH_DISABLE=1` 回退到旧 handler）。
- worker 线程只往线程安全 `deque` 追加 `(levelno, msg)`（不触 UI）。
- 守护线程每 `OK_LOG_FLUSH_MS`（默认 150ms）flush 一次，单次最多 `OK_LOG_BATCH_MAX`（默认 200）条；超量丢最旧，防突发日志阻塞主线程。
- flush 时批量 `communicate.log.emit`，限制 GUI 事件注入速率。

**`ok/gui/debug/OverlayWidget.py`**：`add_log` 不再每条 `self.update()`；改为只追加到 `self.logs`，新增 150ms `QTimer` 周期触发 `self.update()` 重绘（批处理重绘）。`self.logs` 上限 50 不变。

#### B5. ADB 设备刷新/重连降频 — `ok/device/DeviceManager.py`
- `do_refresh` 增加冷却：`_last_refresh_time`，`OK_REFRESH_COOLDOWN_MS`（默认 2000）内重复调用直接 skip（debug 日志）。`handler.post` 的 `remove_existing+skip_if_running` 保留。
- `adb_connect(addr)` 增加按 addr 指数退避：`OK_ADB_CONNECT_BACKOFF_MS`（默认 1000）/ `OK_ADB_CONNECT_BACKOFF_MAX_MS`（默认 8000），序列 1s/2s/4s/8s 封顶。退避窗口内跳过重连，返回 None + 明确日志。
- `try_kill_adb` 限频：`OK_ADB_KILL_COOLDOWN_MS`（默认 10000）内不重复 kill，避免短时多次遍历进程 + kill adb server 抖动。

---

### 环境变量汇总（均有默认值，可回退）

| 变量 | 默认 | 作用 | 回退 |
|---|---|---|---|
| `OK_DEBUG` | 0 | 开启 debug 日志 | — |
| `OK_CAPTURE_PREFER_FAST` | 1 | 自动升级 mumu12→NEMU IPC | 0=旧 adb 行为 |
| `OK_MIN_FRAME_INTERVAL_MS` | 50 | 识别主循环最小帧间隔 | 0=不限速 |
| `OK_LOG_BATCH_DISABLE` | 0 | 关闭日志批处理 | 1=直发 handler |
| `OK_LOG_FLUSH_MS` | 150 | 日志 flush 间隔 | — |
| `OK_LOG_BATCH_MAX` | 200 | 单次 flush 最大条数 | — |
| `OK_REFRESH_COOLDOWN_MS` | 2000 | 设备刷新冷却 | 0=不冷却 |
| `OK_ADB_CONNECT_BACKOFF_MS` | 1000 | adb 重连初始退避 | 0=不退避 |
| `OK_ADB_CONNECT_BACKOFF_MAX_MS` | 8000 | adb 重连退避上限 | — |
| `OK_ADB_KILL_COOLDOWN_MS` | 10000 | kill adb server 限频 | 0=不限频 |

---

## 测试（`tests/`，沿用 unittest + patch 模式）

- `test_capture_fallback.py`：`_is_mumu12` 各分支；`_try_nemu_capture` 成功/`init_nemu` 抛异常/None 返回；`do_start` adb 分支在 prefer_fast on/off、explicit ipc、非 mumu、reuse 已有 nemu 时的捕获方法选择（mock `NemuIpcCaptureMethod`/`ADBCaptureMethod`/`init_nemu`）。
- `test_perf_throttle.py`：`adb_connect` 退避序列与窗口跳过；`do_refresh` 冷却 skip；`try_kill_adb` 限频。
- 扩展 `tests/test_logger.py`：`BatchedCommunicateHandler` 批量 flush、max_batch 截断、`OK_LOG_BATCH_DISABLE` 回退。
- `test_frame_throttle.py`：`next_frame` 在 `_min_frame_interval` 生效时等待、为 0 时不等待、受 `time_out` 约束。
- 运行：`& $py -m pytest tests/test_capture_fallback.py tests/test_perf_throttle.py tests/test_logger.py tests/test_frame_throttle.py`（`.venv\Scripts\python.exe` 优先）。

## 文档
新增 `docs/perf_optimization.md`（简短）：debug 开关、采集优先级与降级、节流参数、日志批处理、adb 退避/冷却、全部环境变量与回退说明。

## 验证（端到端）
1. `& $py -m py_compile start_gui.py ok/device/DeviceManager.py ok/task/TaskExecutor.py ok/util/logger.py ok/gui/debug/OverlayWidget.py` 通过。
2. `& $py -m pytest`（新增测试 + 既有测试）全绿。
3. 手动：mumu12 启动 → 日志出现 `capture method selected: NemuIpc (auto-upgrade)`；任务运行流畅、CPU 下降。
4. 回退演练：`set OK_CAPTURE_PREFER_FAST=0 && set OK_DEBUG=1` → 行为同改造前。
5. 故障演练：开启 mumu 后台保活 → 探测失败 → 自动降级 ADB，无崩溃。

## PR
- 分支：`feature/emulator-game-automation`（项目硬约束）。
- 标题：`perf: reduce GUI/emulator lag via capture fallback + throttling + batched UI logging`
- 描述包含：改动文件清单、每项动机、风险点与回退方式、本地验证结果（启动耗时/CPU/主观流畅度）。
