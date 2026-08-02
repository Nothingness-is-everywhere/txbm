"""
招募任务（日常分类）。

流程：
  1. 检测招募入口选区是否有红色（红点）；有红点则点击该选区，无红点则结束。
  a. 点击"继续"按钮固定位置。
  b. 识别顶部目标选区：模板匹配（templates/recruit_stepb.png），
     匹配到 → 结束；未匹配到 → 回到 a 再点继续，最多循环 N 次。

红点检测复用 ok_tasks._red_dot 的共享实现。
模板匹配复用 ok_tasks._home 的共享工具（load_template_image / match_template_in_roi）。
"""

import logging
from typing import Optional, Tuple

import numpy as np

from ok.task.task import BaseTask

from ok_tasks._home import load_template_image, match_template_in_roi
from ok_tasks._red_dot import detect_red_dot, roi_center

logger = logging.getLogger("RecruitTask")


# 步骤1：招募入口红点选区 (x=676, y=1452, w=68, h=72 @ 1080×1920)
_RECRUIT_ROI = [0.6259, 0.7562, 0.6889, 0.7937]

# 步骤a："继续"按钮点击选区 (x=84, y=1696, w=120, h=88 @ 1080×1920) 中心 (144, 1740)
_STEP_A_CLICK_ROI = [0.0778, 0.8833, 0.1889, 0.9292]

# 步骤b：顶部目标识别选区 (x=144, y=176, w=200, h=64 @ 1080×1920) 中心 (244, 208)
# 用户提供该区域截图作为模板；模板默认路径 templates/recruit_stepb.png
_STEP_B_TEMPLATE_ROI = [0.1333, 0.0917, 0.3185, 0.1250]

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大开始按钮一起执行（配置面板仅显示此项）
    "follow_batch_start": True,
    # --- 步骤1：招募入口红点检测 ---
    "red_dot_roi": list(_RECRUIT_ROI),
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
    """日常任务：招募（入口红点 → 点继续 → 识别目标）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "招募"
        self.description = "招募入口红点 + 点继续 + 模板匹配目标"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        # 配置面板仅显示 follow_batch_start，其他技术项隐藏
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'follow_batch_start'}
        self.config_description = {"follow_batch_start": "是否跟随大开始启动"}
        self.enable_after_start = False
        self.standalone_start = True
        self._stepb_tpl: Optional[np.ndarray] = None

    def on_create(self):
        # 永不自动启动；用户必须点"批量启动"或单个"Start"按钮才会执行。
        # 忽略 config 中可能残留的 _enabled=True（旧版本持久化的值）。
        self._enabled = False
        self.follow_batch_start = self.config.get("follow_batch_start", True)
        self._load_stepb_template()

    # ---------- 辅助：加载/匹配 ----------
    def _load_stepb_template(self):
        rel = self.config.get("stepb_template_path", DEFAULT_CONFIG["stepb_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(
                f"招募：步骤b模板未找到，识别将跳过: {rel}。"
                "请把顶部目标区域截图放到 templates/recruit_stepb.png。"
            )
        self._stepb_tpl = tpl

    def _match_stepb(self, frame: np.ndarray) -> Tuple[bool, float]:
        """在 frame 的 stepb_template_roi 区域匹配 stepb 模板。"""
        if self._stepb_tpl is None:
            return False, 0.0
        roi = self.config.get("stepb_template_roi", DEFAULT_CONFIG["stepb_template_roi"])
        threshold = float(self.config.get("stepb_threshold", DEFAULT_CONFIG["stepb_threshold"]))
        matched, conf = match_template_in_roi(frame, self._stepb_tpl, roi, threshold)
        return matched, float(conf)

    # ---------- 主流程 ----------
    def run(self) -> bool:
        """
        招募任务主流程：
          1. 检测招募入口红点 → 无红点直接结束
          2. a→b 循环：点继续(a) → 等画面 → 匹配目标(b)
             - 匹配到 → 成功结束
             - 未匹配 → 继续点a，循环 ab_loop_max_iters 次
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

        # 步骤1：检测招募入口选区红点
        roi = self.config.get("red_dot_roi", DEFAULT_CONFIG["red_dot_roi"])
        dot = detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
        if dot is None:
            logger.info("招募：入口无红点，结束")
            return False
        logger.info(f"招募：入口检测到红点 {dot}，点击选区")

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

        if ok_b:
            logger.info("招募：流程完成（目标已匹配到）")
            return True
        logger.warning(f"招募：a→b 循环 {iters} 次仍未匹配到步骤b目标，结束")
        return False
