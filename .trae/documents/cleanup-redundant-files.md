# 清理项目冗余文件计划

## Context（背景）

项目 d:\学习\python\ok-script 在开发过程中积累了大量调试脚本、实验性模板图片、运行时日志和调试截图。这些文件：
- 不被正式代码（`ok/` 目录）引用，不影响运行
- 大多已在 `.gitignore` 中忽略（logs/、artifacts/、__pycache__/）
- 占用空间、干扰项目结构理解

用户要求：清理多余模板、测试图片、过时日志等冗余数据，**一定不能影响当前项目的正常运行**。

本计划基于逐文件 Grep 引用审计，每一条清理结论都有引用证据支撑。正式代码（`ok/` 下 `.py`）引用的 6 张核心模板全部保留。

## 调研依据（关键事实）

1. **正式代码引用的模板（必须保留）**——经 Grep 在 `ok/` 下确认：
   - `templates/notice_title_v2.png` → `ok/automation/close_notice_task.py:30`
   - `templates/close_button_v3.png` → `ok/automation/close_notice_task.py:31`
   - `templates/tap_to_start.png` → `ok/automation/close_notice_task.py:32`
   - `templates/checkin_title.png` → `ok/automation/checkin_close_task.py:29`
   - `templates/checkin_message.png` → `ok/automation/checkin_close_task.py:30`
   - `templates/back_button.png` → `ok/automation/back_navigation_task.py:28`

2. **digit_recognizer.py 是孤立模块**：Grep 确认 `digit_recognizer` / `DigitRecognizer` 在 `ok/` 和 `tests/` 下**无任何 import**。它仅自身 `__main__` 运行；`stamina_reader.py` 实际用 Tesseract OCR，不调用它。它用 `iterdir()` 动态加载 `templates/digits/` 下所有 `.png`，但模板 stem 必须是单数字字符才能正确识别——`calib_*`/`exp_digit_*`/`train_digit_*` 命名不符合此逻辑，是混入的校准/实验产物。

3. **templates/fonts/ 目录完全无引用**：Grep `templates/fonts` 和 `fonts/` 在全项目 `.py` 中无命中。

4. **根目录 14 个 `_*.py` 脚本**：Grep 确认均未被 `ok/` 或 `tests/` import，是独立调试/校准脚本。

5. **logs/、artifacts/、__pycache__/** 均在 `.gitignore` 中，是运行时产物。

## 环境限制

PowerShell 执行策略禁用了 `.ps1` 脚本，导致 Shell 工具无法执行任何命令（`ls`/`dir`/`find` 均失败）。因此：
- 删除操作使用 **DeleteFile 工具**（支持一次传多个路径）逐批删除
- 删除前用 **Explore 代理（Glob）** 列出每类文件的精确路径
- 无法运行 `pytest` 做运行时验证；改用**静态引用完整性校验**：删除后用 Grep 重新确认上述 6 张核心模板的引用路径仍然存在且文件未被误删

## 清理清单

### 类别 A：根目录调试脚本（14 个，可清理）
均为 `_` 前缀的独立调试/校准脚本，无正式代码依赖：
- `_debug_recognition.py`
- `_check_libs.py`
- `_extract_templates.py`
- `_extract_network_templates.py`
- `_calibrate_templates.py`
- `_analyze_digits.py`
- `_test_current.py`
- `_test_hires.py`
- `_test_template_match.py`
- `_test_tesseract.py`
- `_test_tesseract_direct.py`
- `_test_easyocr.py`
- `_test_easyocr_direct.py`
- `_test_network_error_template.py`

### 类别 B：未引用的模板图片（templates/ 根目录，10 张，可清理）
Grep 确认在 `ok/` 下无引用，仅被根目录调试脚本引用或不被任何代码引用：
- `templates/button_1.png`
- `templates/button_2.png`
- `templates/button_from_img1_precise.png`
- `templates/button_from_img2_precise.png`
- `templates/network_popup.png`
- `templates/network_popup_v2.png`
- `templates/network_confirm_button.png`
- `templates/network_confirm_button_v2.png`
- `templates/popup_1.png`
- `templates/popup_2.png`

### 类别 C：templates/digits/ 下的校准/实验模板（可清理）
`digit_recognizer.py` 虽动态加载此目录，但该模块未被正式代码调用，且这些文件命名不符合识别逻辑。保留 `digit_0.png`/`digit_1.png`/`digit_2.png`（命名规范的模板）。清理：
- `templates/digits/calib_expedition_digit_*.png`（5 个：0,1,4,5,6）
- `templates/digits/calib_std_expedition_digit_*.png`（5 个：0,1,4,5,6）
- `templates/digits/calib_std_training_digit_*.png`（4 个：0,2,4,5）
- `templates/digits/calib_training_digit_*.png`（4 个：0,2,4,5）
- `templates/digits/exp_digit_*.png`（8 个：0-7）
- `templates/digits/train_digit_*.png`（8 个：0-7）

### 类别 D：templates/fonts/ 整个目录（可清理）
完全无代码引用，是废弃的字体训练实验产物。清理整个 `templates/fonts/` 目录及其所有子目录（`2/`~`9/`）下全部文件（约 50 个 .png）。

### 类别 E：logs/ 日志文件（可清理）
运行时日志，会自动重新生成；已在 `.gitignore` 中：
- `logs/ok-script.log`
- `logs/ok-script.2026-07-26.log`
- `logs/ok-script.2026-07-27.log`
- `logs/ok-script.2026-07-28.log`
- `logs/ok-script.2026-07-29.log`
- `logs/ok-script.2026-07-30.log`
- `logs/thread_dumps.txt`

### 类别 F：artifacts/ 调试产物（可清理）
运行时生成的调试截图；已在 `.gitignore` 中。清理 `artifacts/` 下所有内容（子目录 `diagnose/`、`network_error/`、`debug_output/` 等下的全部文件）。执行前用 Glob 列出精确路径。

### 类别 G：__pycache__ 缓存（可清理）
Python 编译缓存，会自动重新生成；已在 `.gitignore` 中。清理所有 `__pycache__/` 目录下的 `.pyc`/`.pyo` 文件。执行前用 Glob 列出精确路径（约 100+ 个）。

## 保留项（明确不动）

- 正式代码引用的 6 张模板（见调研依据 1）
- `templates/digits/digit_0.png`、`digit_1.png`、`digit_2.png`
- `ok/automation/digit_recognizer.py`（正式代码区的模块，虽孤立但保留）
- `ok/automation/stamina_reader.py` 及所有 `ok/` 下源码
- `tests/` 目录下所有正式测试
- `README.md`、`AGENTS.md`、`requirements.txt`、`LICENSE.txt` 等项目文件
- `.gitignore`、`.venv/`、`.git/`、`.idea/`、`.vscode/`（只读）

## 执行步骤

1. **列出精确路径**：用 Explore 代理（Glob）分别列出类别 C/D/F/G 的完整文件路径清单（类别 A/B/E 路径已明确）。
2. **分批删除**：用 DeleteFile 工具按类别批量删除（每批传多个路径）。
3. **顺序**：先删调试脚本（A）→ 未引用模板（B/C/D）→ 日志（E）→ 调试产物（F）→ 缓存（G）。
4. **空目录处理**：DeleteFile 仅删文件；`templates/fonts/`、`artifacts/`、`logs/`、`__pycache__/` 删空文件后空目录残留无害（已被 .gitignore 忽略）。如需移除空目录，尝试 DeleteFile 传入目录路径，失败则保留。

## 验证（静态引用完整性）

删除完成后，用 Explore 代理执行：
1. **核心模板存在性**：确认以下 6 张模板文件仍存在（Glob 命中）：
   `templates/notice_title_v2.png`、`close_button_v3.png`、`tap_to_start.png`、`checkin_title.png`、`checkin_message.png`、`back_button.png`
2. **引用完整性**：Grep `templates/` 在 `ok/` 下，确认每条引用对应的文件仍存在。
3. **无残留引用断链**：Grep 确认 `ok/` 下不存在指向已删文件的引用（如已删的 `button_1` 等）。
4. **digit_recognizer 仍可导入**：确认 `ok/automation/digit_recognizer.py` 未被误删。

由于 Shell 不可用，无法运行 `pytest`；上述静态校验足以保证运行时引用不断链。
