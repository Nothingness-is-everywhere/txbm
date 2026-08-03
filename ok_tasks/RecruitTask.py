"""
招募任务（日常分类）。

流程：
  1. 检测招募入口选区是否有红色（红点）；有红点则点击该选区，无红点则结束。
  a. 点击"继续"按钮固定位置。
  b. 识别顶部目标选区：模板匹配（templates/recruit_stepb.png），
     匹配到 → 继续；未匹配到 → 回到 a 再点继续，最多循环 N 次。
  c. 点击固定区域。
  c-1. 等待5s后点击固定区域。
  c-2. 等待3s后点击固定区域，校验是否回到招募界面（步骤b模板匹配），未匹配则补点。
  d. 依次执行 4 个子任务（点入口→区域1→区域2×2→区域3→OCR识别5按钮→稀有拦截→SR最优2按钮→区域2）。
  e. 点击返回主界面按钮。
  f. 校验是否回到主界面；未回主页则按返回按钮补点几次。

模板匹配 / 主页检测复用 ok_tasks._home 的共享工具。
"""

import logging
from typing import Optional, Tuple

import numpy as np

from ok.task.task import BaseTask

from ok_tasks._home import (
    DEFAULT_HOME_ROI,
    DEFAULT_HOME_TEMPLATE_PATH,
    DEFAULT_HOME_THRESHOLD,
    is_on_home,
    load_template_image,
    match_template_in_roi,
)
from ok_tasks._red_dot import detect_red_dot, roi_center

logger = logging.getLogger("RecruitTask")


# 步骤1：招募入口模板匹配选区 (x=680, y=1452, w=64, h=80 @ 1080×1920) 中心 (712, 1492)
_RECRUIT_ROI = [0.6296, 0.7562, 0.6889, 0.7979]

# 步骤a："继续"按钮点击选区 (x=84, y=1696, w=120, h=88 @ 1080×1920) 中心 (144, 1740)
_STEP_A_CLICK_ROI = [0.0778, 0.8833, 0.1889, 0.9292]

# 步骤b：顶部目标识别选区 (x=108, y=176, w=308, h=48 @ 1080×1920) 中心 (262, 200)
# 模板文件 templates/recruit_stepb.png
_STEP_B_TEMPLATE_ROI = [0.1000, 0.0917, 0.3852, 0.1167]

# 步骤c：点击区域 (x=388, y=1532, w=296, h=68 @ 1080×1920) 中心 (536, 1566)
_STEP_C_CLICK_ROI = [0.3593, 0.7979, 0.6333, 0.8333]

# 步骤c-1：等待5s后点击 (x=804, y=72, w=140, h=72 @ 1080×1920) 中心 (874, 108)
_STEP_C1_CLICK_ROI = [0.7444, 0.0375, 0.8741, 0.0750]

# 步骤c-2：等待3s后点击 (x=388, y=1772, w=308, h=52 @ 1080×1920) 中心 (542, 1798)
# 点击后检测是否回到招募界面（步骤b模板匹配），未匹配则补点
_STEP_C2_CLICK_ROI = [0.3593, 0.9229, 0.6444, 0.9500]

# 步骤d：4 个子任务点击区域
# 子任务1 (x=456, y=460, w=172, h=32) 中心 (542, 476)
_SUB_TASK_1_ROI = [0.4222, 0.2396, 0.5815, 0.2563]
# 子任务2 (x=476, y=780, w=168, h=44) 中心 (560, 802)
_SUB_TASK_2_ROI = [0.4407, 0.4063, 0.5963, 0.4292]
# 子任务3 (x=452, y=1096, w=180, h=48) 中心 (542, 1120)
_SUB_TASK_3_ROI = [0.4185, 0.5708, 0.5852, 0.5958]
# 子任务4 (x=456, y=1408, w=184, h=40) 中心 (548, 1428)
_SUB_TASK_4_ROI = [0.4222, 0.7333, 0.5926, 0.7542]

# 步骤d子任务通用点击区域（每个子任务进入后的固定操作）
# 子任务点击区域1 (x=172, y=496, w=216, h=60) 中心 (280, 526)
_SUB_CLICK1_ROI = [0.1593, 0.2583, 0.3593, 0.2895]
# 子任务点击区域2：点两下 + 每个子任务最后点击 (x=864, y=920, w=76, h=68) 中心 (902, 954)
_SUB_CLICK2_ROI = [0.8000, 0.4792, 0.8704, 0.5146]
# 子任务点击区域3 (x=392, y=1048, w=300, h=56) 中心 (542, 1076)
_SUB_CLICK3_ROI = [0.3630, 0.5458, 0.6408, 0.5750]
# 子任务最后确认按钮模板匹配+点击区域 (x=592, y=1524, w=284, h=76) 中心 (734, 1562)
_SUB_FINAL_ROI = [0.5481, 0.7937, 0.8111, 0.8333]

# 步骤d子任务：5个候选按钮ROI（OCR识别中文，用于计算最佳2个按钮）
# 按钮1 (x=176, y=704, w=204, h=60) 中心 (278, 734)
_SUB_BTN_1_ROI = [0.1630, 0.3667, 0.3519, 0.3979]
# 按钮2 (x=436, y=704, w=216, h=56) 中心 (544, 732)
_SUB_BTN_2_ROI = [0.4037, 0.3667, 0.6037, 0.3959]
# 按钮3 (x=696, y=700, w=204, h=68) 中心 (798, 734)
_SUB_BTN_3_ROI = [0.6444, 0.3646, 0.8333, 0.4000]
# 按钮4 (x=176, y=808, w=216, h=68) 中心 (284, 842)
_SUB_BTN_4_ROI = [0.1630, 0.4208, 0.3630, 0.4562]
# 按钮5 (x=424, y=808, w=232, h=64) 中心 (540, 840)
_SUB_BTN_5_ROI = [0.3926, 0.4208, 0.6074, 0.4541]

# 步骤e：返回主界面按钮 (x=24, y=84, w=124, h=44 @ 1080×1920) 中心 (86, 106)
_FINAL_CLICK_ROI = [0.0222, 0.0438, 0.1370, 0.0667]

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大开始按钮一起执行（配置面板仅显示此项）
    "follow_batch_start": True,
    # --- 步骤1：招募入口模板匹配 ---
    "entry_template_path": "templates/recruit_entry.png",
    "entry_template_roi": list(_RECRUIT_ROI),
    "entry_threshold": 0.70,
    "click_roi": list(_RECRUIT_ROI),
    # --- 步骤a：继续按钮 ---
    "stepa_click_roi": list(_STEP_A_CLICK_ROI),
    # --- 步骤b：顶部目标识别（模板匹配） ---
    "stepb_template_path": "templates/recruit_stepb.png",
    "stepb_template_roi": list(_STEP_B_TEMPLATE_ROI),
    "stepb_threshold": 0.70,
    # --- a→b 循环控制 ---
    "ab_loop_interval": 2.0,        # 每次"点继续→等画面"的间隔
    "ab_loop_max_iters": 20,        # a→b 循环最大次数（防无限循环，0 = 不限制）
    # --- 步骤c：点击区域 ---
    "stepc_click_roi": list(_STEP_C_CLICK_ROI),
    # --- 步骤c-1：等待后点击 ---
    "stepc1_click_roi": list(_STEP_C1_CLICK_ROI),
    "stepc1_wait": 5.0,
    # --- 步骤c-2：等待后点击 + 招募界面校验 ---
    "stepc2_click_roi": list(_STEP_C2_CLICK_ROI),
    "stepc2_wait": 3.0,
    "recruit_check_max_attempts": 3,
    # --- 步骤d：4 个子任务点击区域 ---
    "subtask_1_roi": list(_SUB_TASK_1_ROI),
    "subtask_2_roi": list(_SUB_TASK_2_ROI),
    "subtask_3_roi": list(_SUB_TASK_3_ROI),
    "subtask_4_roi": list(_SUB_TASK_4_ROI),
    # --- 步骤d子任务通用点击 ---
    "sub_click1_roi": list(_SUB_CLICK1_ROI),
    "sub_click2_roi": list(_SUB_CLICK2_ROI),
    "sub_click2_times": 2,
    "sub_click3_roi": list(_SUB_CLICK3_ROI),
    "sub_final_template_path": "templates/recruit_confirm.png",
    "sub_final_template_roi": list(_SUB_FINAL_ROI),
    "sub_final_click_roi": list(_SUB_FINAL_ROI),
    "sub_final_threshold": 0.70,
    "sub_final_wait": 2.0,
    "sub_final_max_attempts": 3,
    # --- 步骤d子任务：5个候选按钮ROI（OCR识别中文，用于计算最佳2个按钮） ---
    "sub_btn_rois": [
        list(_SUB_BTN_1_ROI),
        list(_SUB_BTN_2_ROI),
        list(_SUB_BTN_3_ROI),
        list(_SUB_BTN_4_ROI),
        list(_SUB_BTN_5_ROI),
    ],
    # 识别到这些关键词时暂停并弹窗（稀有角色）
    "rare_keywords": ["领袖", "菁英"],
    # --- 步骤e：返回主界面按钮 ---
    "final_click_roi": list(_FINAL_CLICK_ROI),
    # --- 步骤f：返回主界面后校验 ---
    "home_template_path": DEFAULT_HOME_TEMPLATE_PATH,
    "home_template_roi": list(DEFAULT_HOME_ROI),
    "home_threshold": DEFAULT_HOME_THRESHOLD,
    "home_check_max_attempts": 3,
    # 点击后等待秒数
    "post_click_sleep": 1.0,
    # --- 红点检测参数（共享默认） ---
    "hsv_lower1": [0, 120, 120],
    "hsv_upper1": [10, 255, 255],
    "hsv_lower2": [170, 120, 120],
    "hsv_upper2": [180, 255, 255],
    "min_red_pixels": 20,
    "morph_open_size": 3,
    "min_component_area": 16,
    "max_component_area": 1200,
    "max_component_aspect": 1.8,
    "min_component_circularity": 0.45,
    "min_component_fill_ratio": 0.45,
}


class RecruitTask(BaseTask):
    """日常任务：招募（入口红点 → 点继续 → 识别目标 → 点击 → 子任务 → 返回主页）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "招募"
        self.description = "招募入口红点 + 点继续 + 模板匹配目标 + 子任务点击"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        # 配置面板仅显示 follow_batch_start，其他技术项隐藏
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'follow_batch_start'}
        self.config_description = {"follow_batch_start": "是否跟随大开始启动"}
        self.enable_after_start = False
        self.standalone_start = True
        self._entry_tpl: Optional[np.ndarray] = None
        self._stepb_tpl: Optional[np.ndarray] = None
        self._home_tpl: Optional[np.ndarray] = None
        self._sub_final_tpl: Optional[np.ndarray] = None

    def on_create(self):
        # 永不自动启动；用户必须点"批量启动"或单个"Start"按钮才会执行。
        # 忽略 config 中可能残留的 _enabled=True（旧版本持久化的值）。
        self._enabled = False
        self.follow_batch_start = self.config.get("follow_batch_start", True)
        self._load_entry_template()
        self._load_stepb_template()
        self._load_home_template()
        self._load_sub_final_template()

    # ---------- 辅助：模板加载 ----------
    def _load_entry_template(self):
        rel = self.config.get("entry_template_path", DEFAULT_CONFIG["entry_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"招募：入口模板未找到，步骤1将无法匹配: {rel}")
        self._entry_tpl = tpl

    def _load_stepb_template(self):
        rel = self.config.get("stepb_template_path", DEFAULT_CONFIG["stepb_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(
                f"招募：步骤b模板未找到，识别将跳过: {rel}。"
                "请把顶部目标区域截图放到 templates/recruit_stepb.png。"
            )
        self._stepb_tpl = tpl

    def _load_home_template(self):
        rel = self.config.get("home_template_path", DEFAULT_CONFIG["home_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"招募：主页模板未找到，返回主界面校验将跳过: {rel}")
        self._home_tpl = tpl

    def _load_sub_final_template(self):
        rel = self.config.get("sub_final_template_path", DEFAULT_CONFIG["sub_final_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"招募：确认按钮模板未找到，d-8将跳过: {rel}")
        self._sub_final_tpl = tpl

    # ---------- 辅助：模板匹配 ----------
    def _match_entry(self, frame: np.ndarray) -> Tuple[bool, float]:
        """在 frame 的 entry_template_roi 区域匹配招募入口模板。"""
        if self._entry_tpl is None:
            return False, 0.0
        roi = self.config.get("entry_template_roi", DEFAULT_CONFIG["entry_template_roi"])
        threshold = float(self.config.get("entry_threshold", DEFAULT_CONFIG["entry_threshold"]))
        matched, conf = match_template_in_roi(frame, self._entry_tpl, roi, threshold)
        return matched, float(conf)

    def _match_stepb(self, frame: np.ndarray) -> Tuple[bool, float]:
        """在 frame 的 stepb_template_roi 区域匹配 stepb 模板。"""
        if self._stepb_tpl is None:
            return False, 0.0
        roi = self.config.get("stepb_template_roi", DEFAULT_CONFIG["stepb_template_roi"])
        threshold = float(self.config.get("stepb_threshold", DEFAULT_CONFIG["stepb_threshold"]))
        matched, conf = match_template_in_roi(frame, self._stepb_tpl, roi, threshold)
        return matched, float(conf)

    # ---------- 辅助：主页校验 ----------
    def _is_home(self, frame: np.ndarray) -> bool:
        """通过主页模板匹配判断当前是否在主界面。"""
        if self._home_tpl is None:
            return False
        roi = self.config.get("home_template_roi", DEFAULT_CONFIG["home_template_roi"])
        threshold = float(self.config.get("home_threshold", DEFAULT_CONFIG["home_threshold"]))
        return is_on_home(frame, self._home_tpl, roi, threshold)

    def _ensure_home(self, click_roi) -> bool:
        """点击返回按钮后校验是否回到主界面；未回主页则按 click_roi 补点几次。"""
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        max_attempts = int(self.config.get("home_check_max_attempts", DEFAULT_CONFIG["home_check_max_attempts"]))
        for attempt in range(1, max_attempts + 1):
            frame = self.next_frame()
            if frame is None:
                logger.warning("招募：校验主页时无画面可用")
                return False
            if self._is_home(frame):
                logger.info(f"招募：已回到主界面（第 {attempt} 次确认）")
                return True
            h, w = frame.shape[:2]
            cx, cy = roi_center(click_roi, w, h)
            logger.info(f"招募：未回主界面，第 {attempt} 次补点返回按钮 ({cx}, {cy})")
            self.click(cx, cy)
            self.sleep(step_sleep)
        logger.warning(f"招募：{max_attempts} 次补点后仍未回主界面")
        return False

    # ---------- 辅助：子任务OCR与稀有角色处理 ----------
    def _ocr_button_texts(self, frame: np.ndarray, btn_rois) -> list:
        """对每个按钮ROI做OCR，返回中文文字列表（按按钮顺序）。"""
        texts = []
        for bidx, broi in enumerate(btn_rois, start=1):
            try:
                boxes = self.ocr(
                    x=broi[0], y=broi[1], to_x=broi[2], to_y=broi[3],
                    frame=frame,
                )
            except Exception as e:
                logger.warning(f"招募：按钮{bidx} OCR异常: {e}")
                boxes = []
            text = "".join(getattr(b, 'name', '') or '' for b in boxes)
            texts.append(text)
            logger.info(f"招募：按钮{bidx} OCR识别: '{text}'")
        return texts

    def _check_subtask_ocr_and_rare_role(self, subtask_idx: int, btn_rois: list, rare_keywords: list) -> Optional[list]:
        """OCR 识别按钮文本，并在识别到稀有角色时暂停处理当前子任务。"""
        frame_ocr = self.next_frame()
        if frame_ocr is None:
            logger.warning(f"[步骤d-{subtask_idx}] OCR前无画面可用，跳过该子任务剩余步骤")
            return None

        texts = self._ocr_button_texts(frame_ocr, btn_rois)
        full_text = "".join(texts)
        hit_rare = [kw for kw in rare_keywords if kw in full_text]
        if hit_rare:
            msg = (f"招募子任务{subtask_idx}识别到稀有角色: {hit_rare}\n"
                   f"5个按钮识别结果: {texts}")
            logger.warning(msg)
            self._alert_and_pause(msg)
            return None

        return texts

    def _select_best_buttons(self, texts: list) -> list:
        """
        根据OCR识别的5个按钮文字，选择SR概率最高的2个按钮索引（0-based）。

        枚举 C(n,2) 种2按钮组合，对每组标签调用 ok.util.recruit.calculate，
        取 SR 模式下最高 SR 概率，选概率最大的一对返回 [i, j]。
        """
        from itertools import combinations
        from ok.util.recruit import calculate as recruit_calculate

        n = len(texts)
        if n < 2:
            return []

        best_pair: list = []
        best_percent = -1.0
        for i, j in combinations(range(n), 2):
            try:
                result = recruit_calculate([texts[i], texts[j]])
            except Exception as e:
                logger.warning(f"招募：计算按钮({i + 1},{j + 1}) SR概率异常: {e}")
                continue
            if result.leader_mode or not result.sr_chances:
                # leader 模式无 SR 概率（d-5 已拦截领袖，此处兜底）；无命中组合则概率 0
                percent = 0.0
            else:
                # sr_chances 已按 percent 降序，取最高
                percent = result.sr_chances[0].percent
            logger.info(
                f"招募：按钮({i + 1},{j + 1}) 标签=['{texts[i]}','{texts[j]}'] "
                f"SR最高概率={percent * 100:.2f}%"
            )
            if percent > best_percent:
                best_percent = percent
                best_pair = [i, j]

        if best_pair:
            logger.info(
                f"招募：最佳2按钮 = 按钮{best_pair[0] + 1}+按钮{best_pair[1] + 1} "
                f"(SR概率={best_percent * 100:.2f}%)"
            )
        else:
            logger.warning("招募：未能计算出有效SR概率，不点击按钮")
        return best_pair

    def _alert_and_pause(self, message: str):
        """弹窗通知用户并暂停任务，等待用户恢复后返回（用于稀有角色手动处理）。"""
        try:
            from ok.gui.util.Alert import alert_error
            alert_error(message, tray=True, show_tab="onetime")
        except Exception as e:
            logger.error(f"招募：弹窗通知失败: {e}")
        try:
            self.pause()
        except Exception as e:
            logger.error(f"招募：暂停任务失败: {e}")
        # 等待用户在 GUI 点"继续"恢复（unpause 置 _paused=False），或执行器停止
        while self.paused:
            if self.executor.exit_event.is_set():
                return
            self.sleep(0.5)

    # ---------- 主流程 ----------
    def run(self) -> bool:
        """
        招募任务主流程：
          1. 检测招募入口红点 → 无红点直接结束
          2. a→b 循环：点继续(a) → 等画面 → 匹配目标(b)
          3. 步骤c：点击固定区域
          4. 步骤c-1：等待5s后点击
          5. 步骤c-2：等待3s后点击 + 校验是否回到招募界面（步骤b模板匹配），未匹配则补点
          6. 步骤d：依次点击 4 个子任务区域
          7. 步骤e：点击返回主界面按钮
          8. 步骤f：校验是否回到主界面，未回则补点
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        interval = float(self.config.get("ab_loop_interval", DEFAULT_CONFIG["ab_loop_interval"]))
        max_iters = int(self.config.get("ab_loop_max_iters", DEFAULT_CONFIG["ab_loop_max_iters"]))

        frame = self.executor.frame
        if frame is None:
            frame = self.next_frame()
        if frame is None:
            logger.warning("招募：无画面可用，终止")
            return False

        # 步骤1：模板匹配招募入口
        matched, conf = self._match_entry(frame)
        if not matched:
            logger.info(f"招募：入口未匹配到模板 (最高 conf={conf:.3f})，结束")
            return False
        logger.info(f"招募：入口匹配成功 (conf={conf:.3f})，点击进入")

        h, w = frame.shape[:2]
        cx_1, cy_1 = roi_center(self.config.get("click_roi", DEFAULT_CONFIG["click_roi"]), w, h)
        logger.info(f"[步骤1] 点击招募入口 ({cx_1}, {cy_1})")
        self.click(cx_1, cy_1)
        self.sleep(step_sleep)

        # ========== a→b 循环 ==========
        stepa_roi = self.config.get("stepa_click_roi", DEFAULT_CONFIG["stepa_click_roi"])
        ax, ay = roi_center(stepa_roi, w, h)

        logger.info(f"招募：进入 a→b 循环（max_iters={max_iters if max_iters > 0 else '∞'}）")
        ok_b = False
        iters = 0
        while max_iters <= 0 or iters < max_iters:
            iters += 1
            # a：点继续
            logger.info(f"[步骤a #{iters}] 点击继续按钮 ({ax}, {ay})")
            self.click(ax, ay)
            self.sleep(interval)

            cur = self.next_frame()
            if cur is None:
                logger.info(f"招募：步骤a后无画面，继续循环 (#{iters})")
                continue

            # b：模板匹配顶部目标选区
            matched, conf = self._match_stepb(cur)
            if matched:
                logger.info(f"[步骤b #{iters}] 匹配成功 (conf={conf:.3f})")
                ok_b = True
                break
            logger.info(f"[步骤b #{iters}] 未匹配到目标 (最高 conf={conf:.3f})，回到步骤a")

        if not ok_b:
            logger.warning(f"招募：a→b 循环 {iters} 次仍未匹配到步骤b目标，结束")
            return False

        # ========== 步骤c：点击固定区域 ==========
        stepc_roi = self.config.get("stepc_click_roi", DEFAULT_CONFIG["stepc_click_roi"])
        cx_c, cy_c = roi_center(stepc_roi, w, h)
        logger.info(f"[步骤c] 点击区域 ({cx_c}, {cy_c})")
        self.click(cx_c, cy_c)
        self.sleep(step_sleep)

        # ========== 步骤c-1：等待5s后点击固定区域 ==========
        stepc1_wait = float(self.config.get("stepc1_wait", DEFAULT_CONFIG["stepc1_wait"]))
        stepc1_roi = self.config.get("stepc1_click_roi", DEFAULT_CONFIG["stepc1_click_roi"])
        cx_c1, cy_c1 = roi_center(stepc1_roi, w, h)
        logger.info(f"[步骤c-1] 等待 {stepc1_wait}s 后点击 ({cx_c1}, {cy_c1})")
        self.sleep(stepc1_wait)
        self.click(cx_c1, cy_c1)

        # ========== 步骤c-2：等待3s后点击 + 校验招募界面 ==========
        # 点击后用步骤b模板匹配校验是否回到招募界面，未匹配则补点，最多 recruit_check_max_attempts 次
        stepc2_wait = float(self.config.get("stepc2_wait", DEFAULT_CONFIG["stepc2_wait"]))
        stepc2_roi = self.config.get("stepc2_click_roi", DEFAULT_CONFIG["stepc2_click_roi"])
        recruit_max = int(self.config.get("recruit_check_max_attempts", DEFAULT_CONFIG["recruit_check_max_attempts"]))
        logger.info(f"[步骤c-2] 等待 {stepc2_wait}s 后点击并校验招募界面")
        self.sleep(stepc2_wait)
        recruit_ok = False
        for attempt in range(1, recruit_max + 1):
            cx_c2, cy_c2 = roi_center(stepc2_roi, w, h)
            logger.info(f"[步骤c-2 #{attempt}] 点击 ({cx_c2}, {cy_c2})")
            self.click(cx_c2, cy_c2)
            self.sleep(step_sleep)
            frame_check = self.next_frame()
            if frame_check is not None:
                matched, conf = self._match_stepb(frame_check)
                if matched:
                    logger.info(f"[步骤c-2 #{attempt}] 已回到招募界面 (conf={conf:.3f})")
                    recruit_ok = True
                    break
                logger.info(f"[步骤c-2 #{attempt}] 未回到招募界面 (conf={conf:.3f})，补点")
            else:
                logger.info(f"[步骤c-2 #{attempt}] 无画面可用，继续补点")
        if not recruit_ok:
            logger.warning(f"招募：{recruit_max} 次补点后仍未回到招募界面，继续后续步骤")

        # ========== 步骤d：依次点击 4 个子任务区域 ==========
        subtask_rois = [
            self.config.get("subtask_1_roi", DEFAULT_CONFIG["subtask_1_roi"]),
            self.config.get("subtask_2_roi", DEFAULT_CONFIG["subtask_2_roi"]),
            self.config.get("subtask_3_roi", DEFAULT_CONFIG["subtask_3_roi"]),
            self.config.get("subtask_4_roi", DEFAULT_CONFIG["subtask_4_roi"]),
        ]
        # 子任务通用点击区域与参数
        sub_click1_roi = self.config.get("sub_click1_roi", DEFAULT_CONFIG["sub_click1_roi"])
        sub_click2_roi = self.config.get("sub_click2_roi", DEFAULT_CONFIG["sub_click2_roi"])
        sub_click2_times = int(self.config.get("sub_click2_times", DEFAULT_CONFIG["sub_click2_times"]))
        sub_click3_roi = self.config.get("sub_click3_roi", DEFAULT_CONFIG["sub_click3_roi"])
        btn_rois = self.config.get("sub_btn_rois", DEFAULT_CONFIG["sub_btn_rois"])
        rare_keywords = self.config.get("rare_keywords", DEFAULT_CONFIG["rare_keywords"])

        for idx, sub_roi in enumerate(subtask_rois, start=1):
            # d-1：点击子任务入口区域
            sx, sy = roi_center(sub_roi, w, h)
            logger.info(f"[步骤d-{idx}] 点击子任务{idx}入口 ({sx}, {sy})")
            self.click(sx, sy)
            self.sleep(step_sleep)

            # d-2：点击区域1
            c1x, c1y = roi_center(sub_click1_roi, w, h)
            logger.info(f"[步骤d-{idx}] 点击区域1 ({c1x}, {c1y})")
            self.click(c1x, c1y)
            self.sleep(step_sleep)

            # d-3：点击区域2（点两下）前先做一次OCR/稀有角色检查
            texts = self._check_subtask_ocr_and_rare_role(idx, btn_rois, rare_keywords)
            if texts is None:
                continue

            # d-3：点击区域2（点两下）
            c2x, c2y = roi_center(sub_click2_roi, w, h)
            logger.info(f"[步骤d-{idx}] 点击区域2({c2x}, {c2y})")
            self.click(c2x, c2y)
            self.sleep(step_sleep)


            # d-4：点击区域3
            c3x, c3y = roi_center(sub_click3_roi, w, h)
            logger.info(f"[步骤d-{idx}] 点击区域3 ({c3x}, {c3y})")
            self.click(c3x, c3y)
            self.sleep(step_sleep)

            # d-5：OCR识别5个候选按钮中文，并在稀有角色时暂停
            texts = self._check_subtask_ocr_and_rare_role(idx, btn_rois, rare_keywords)
            if texts is None:
                continue

            # d-7：计算SR概率最高的2个按钮并点击
            best_indices = self._select_best_buttons(texts)
            for bi in best_indices:
                if 0 <= bi < len(btn_rois):
                    bx, by = roi_center(btn_rois[bi], w, h)
                    logger.info(f"[步骤d-{idx}] 点击最佳按钮{bi + 1} ({bx}, {by})")
                    self.click(bx, by)
                    self.sleep(step_sleep)

            # d-8：模板匹配确认按钮并点击，校验未消失则补点
            if self._sub_final_tpl is None:
                logger.warning(f"[步骤d-{idx}] 确认按钮模板未加载，跳过d-8")
            else:
                sf_roi = self.config.get("sub_final_template_roi", DEFAULT_CONFIG["sub_final_template_roi"])
                sf_threshold = float(self.config.get("sub_final_threshold", DEFAULT_CONFIG["sub_final_threshold"]))
                sf_wait = float(self.config.get("sub_final_wait", DEFAULT_CONFIG["sub_final_wait"]))
                sf_max = int(self.config.get("sub_final_max_attempts", DEFAULT_CONFIG["sub_final_max_attempts"]))
                sf_click_roi = self.config.get("sub_final_click_roi", DEFAULT_CONFIG["sub_final_click_roi"])
                for attempt in range(1, sf_max + 1):
                    frame_sf = self.next_frame()
                    if frame_sf is None:
                        logger.warning(f"[步骤d-{idx}] 确认按钮校验无画面，跳过")
                        break
                    matched, conf = match_template_in_roi(frame_sf, self._sub_final_tpl, sf_roi, sf_threshold)
                    if not matched:
                        if attempt == 1:
                            logger.info(f"[步骤d-{idx}] 未匹配到确认按钮 (conf={conf:.3f})，跳过点击")
                        else:
                            logger.info(f"[步骤d-{idx}] 确认按钮已消失 (conf={conf:.3f})，点击成功")
                        break
                    cx_sf, cy_sf = roi_center(sf_click_roi, w, h)
                    logger.info(f"[步骤d-{idx}] 匹配到确认按钮 (conf={conf:.3f})，第{attempt}次点击 ({cx_sf}, {cy_sf})")
                    self.click(cx_sf, cy_sf)
                    self.sleep(sf_wait)
                else:
                    logger.warning(f"[步骤d-{idx}] 确认按钮 {sf_max} 次点击后仍存在")

        # ========== 步骤e：点击返回主界面按钮 ==========
        final_roi = self.config.get("final_click_roi", DEFAULT_CONFIG["final_click_roi"])
        fx, fy = roi_center(final_roi, w, h)
        logger.info(f"[步骤e] 点击返回主界面按钮 ({fx}, {fy})")
        self.click(fx, fy)
        self.sleep(step_sleep)

        # ========== 步骤f：校验是否回到主界面 ==========
        self._ensure_home(final_roi)

        logger.info("招募：流程完成")
        return True
